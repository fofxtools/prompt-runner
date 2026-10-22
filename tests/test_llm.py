"""Unit tests for src/prompt_runner/llm.py, over the real LlamaServer, the keep-alive code and a fake `llama serve` (see conftest.py).

These start real detached processes; a fixture stops them and kills any a test leaves behind.
"""

import builtins
import io
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest
import yaml

from prompt_runner import keepalive
from prompt_runner.llama_server import LlamaServer
from prompt_runner.llm import main
from prompt_runner.server_process import ServerError, free_port

REPO = Path(__file__).resolve().parent.parent


class Tty(io.StringIO):
    """A stdin that claims to be a terminal."""

    def isatty(self) -> bool:
        return True


def fakes_running(binary: str) -> list[int]:
    pids = []
    for entry in Path("/proc").iterdir():
        if entry.name.isdigit():
            try:
                raw = (entry / "cmdline").read_bytes().split(b"\0")
            except OSError:
                continue
            if binary.encode() in raw[:2]:  # a script runs behind its interpreter
                pids.append(int(entry.name))
    return pids


class Env:
    def __init__(
        self, tmp_path: Path, fake_llama: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self.tmp = tmp_path
        self.binary = fake_llama
        self.port = free_port()
        self.state = tmp_path / "state"
        self.repo = tmp_path / "repo"
        self.monkeypatch = monkeypatch
        monkeypatch.setenv("LLM_PORT", str(self.port))
        (self.repo / "config").mkdir(parents=True)
        self.write_config()
        self.write_models()

    def entry(self, name: str, mode: str, record: bool = True) -> dict[str, Any]:
        weights = self.tmp / f"{name}.gguf"
        weights.write_bytes(b"x" * 2_000_000)
        args: dict[str, Any] = {"-m": str(weights), "--mode": mode}
        if record:
            args["--record"] = str(self.body(name))
        return {"name": name, "server_args": args}

    def body(self, name: str) -> Path:
        return self.tmp / f"body-{name}.json"

    def log_file(self) -> Path:
        """Where the log of a call without -o goes: under the repo's results folder."""
        return self.repo / "storage/results/llm-cli/runs.jsonl"

    def rows(self, path: Path | None = None) -> list[dict[str, Any]]:
        """The rows of runs.jsonl (none if it was never written)."""
        path = path or self.log_file()
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text().splitlines()]

    def messages(self, name: str) -> list[dict[str, Any]]:
        """The messages of the last request the fake server of this model got."""
        body = json.loads(self.body(name).read_text())
        assert isinstance(body["messages"], list)
        return body["messages"]

    def write_config(self, **more: Any) -> None:
        config = {
            "results_dir": "storage/results/",
            "llama_server": self.binary,
            "llm_generation_defaults": {"max_tokens": 50},
            **more,
        }
        (self.repo / "config/config.yaml").write_text(yaml.safe_dump(config))

    def write_models(self) -> None:
        models = [
            self.entry("think", "think"),
            self.entry("plain", "nothink"),
            self.entry("thinker", "thinkonly"),
            self.entry("rejects", "badctx"),
            self.entry("dies", "die"),
            self.entry("wont-start", "exit", record=False),
        ]
        models[1]["server_args"]["--mmproj"] = str(
            self.tmp / "plain-mmproj.gguf"
        )  # not on disk
        (self.repo / "config/llm_models.yaml").write_text(yaml.safe_dump(models))

    def kept(self) -> keepalive.KeptServer[LlamaServer]:
        return keepalive.kept_server("llm", LlamaServer, self.port, self.state)

    def run(
        self,
        *args: str,
        stdin: str | io.StringIO | None = None,
        lines: list[Any] | None = None,
    ) -> int:
        """Run llm; `stdin` is piped text, `lines` are what a person types in a chat (a callable runs, then types nothing)."""
        if lines is not None:
            self.monkeypatch.setattr(sys, "stdin", Tty())
            typed = iter(lines)

            def fake_input(prompt: str = "") -> str:
                item = next(typed, EOFError)
                if item is EOFError:
                    raise EOFError
                if isinstance(item, type) and issubclass(item, BaseException):
                    raise item
                if callable(item):
                    item()
                    return fake_input(prompt)
                return str(item)

            self.monkeypatch.setattr(builtins, "input", fake_input)
        else:
            self.monkeypatch.setattr(
                sys,
                "stdin",
                stdin if isinstance(stdin, io.StringIO) else io.StringIO(stdin or ""),
            )
        return main(list(args), repo_root=self.repo, state_dir=self.state)


@pytest.fixture
def env(tmp_path, fake_llama, monkeypatch):
    environment = Env(tmp_path, fake_llama, monkeypatch)
    yield environment
    try:
        keepalive.stop_server(environment.kept())
    except keepalive.KeepAliveError:
        pass
    for pid in fakes_running(fake_llama):
        os.kill(pid, signal.SIGKILL)


class TestList:
    def test_shows_size_state_and_the_loaded_model(self, env, capsys):
        env.run("--list")
        before = capsys.readouterr().out.splitlines()
        assert before[0].split() == ["NAME", "SIZE", "STATE", "VISION"]
        rows = {line.split()[0]: line.split() for line in before[1:]}
        assert rows["think"][1:3] == ["0.0", "GB"]  # 2 MB
        assert (
            rows["think"][-1] == "GB"
        )  # no projector: not vision, nothing else to say
        assert "missing" in rows["plain"]  # its --mmproj is not on disk
        assert rows["plain"][-1] == "vision"  # ... but it has one
        assert "running" not in "".join(before)

        env.run("think", "hi")
        capsys.readouterr()
        env.run("--list")
        after = capsys.readouterr().out.splitlines()
        states = {line.split()[0]: line.split()[-1] for line in after[1:]}
        assert (
            states["think"] == "running" and states["thinker"] == "GB"
        )  # only one is loaded


class TestRunOnce:
    def test_the_answer_is_on_stdout_and_the_thinking_on_stderr(self, env, capsys):
        code = env.run("think", "hi")
        out, err = capsys.readouterr()
        assert code == 0
        assert out == "Hello world\n"
        assert "Let me think." in err and "Let me think." not in out
        assert "loading think" in err and "started the llama-server" in err
        assert env.messages("think") == [{"role": "user", "content": "hi"}]

    def test_the_options_come_from_the_config_and_the_models_entry(self, env):
        models = yaml.safe_load((env.repo / "config/llm_models.yaml").read_text())
        models[0]["generation_options"] = {"temperature": 0.3}
        (env.repo / "config/llm_models.yaml").write_text(yaml.safe_dump(models))
        env.run("think", "hi")
        body = json.loads(env.body("think").read_text())
        assert (body["temperature"], body["max_tokens"], body["stream"]) == (
            0.3,
            50,
            True,
        )

    def test_the_server_stays_loaded_and_the_next_call_reuses_it(self, env, capsys):
        env.run("think", "one")
        capsys.readouterr()
        code = env.run("think", "two")
        out, err = capsys.readouterr()
        assert code == 0 and out == "Hello world\n"
        assert "reusing the llama-server already running" in err
        assert "loading" not in err and "started" not in err
        assert keepalive.running_model(env.kept()) == "think"

    def test_piped_text_is_the_prompt(self, env, capsys):
        code = env.run("think", stdin="text from a pipe\n")
        assert code == 0 and capsys.readouterr().out == "Hello world\n"
        assert env.messages("think") == [
            {"role": "user", "content": "text from a pipe"}
        ]

    def test_stdin_is_not_read_when_a_prompt_is_given(self, env):
        class NeverEnds(io.StringIO):
            def read(self, size: int | None = -1) -> str:
                raise AssertionError("an open pipe that never ends would hang here")

        assert env.run("think", "Summarize:", stdin=NeverEnds()) == 0
        assert env.messages("think")[0]["content"] == "Summarize:"

    def test_an_empty_prompt_is_refused(self, env, capsys):
        assert env.run("think", stdin="") == 1
        assert "the prompt is empty" in capsys.readouterr().err
        assert keepalive.running_model(env.kept()) is None

    def test_a_system_prompt_goes_first(self, env):
        env.run("think", "hi", "--system", "Be brief.")
        assert env.messages("think") == [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "hi"},
        ]

    def test_another_model_replaces_the_server(self, env, capsys):
        env.run("think", "hi")
        capsys.readouterr()
        env.run("plain", "hi")
        err = capsys.readouterr().err
        assert "stopping the running llama-server" in err and "loading plain" in err
        assert keepalive.running_model(env.kept()) == "plain"

    def test_a_reply_without_an_answer_is_flagged_not_hidden(self, env, capsys):
        assert env.run("thinker", "hi") == 0
        out, err = capsys.readouterr()
        assert out == ""
        assert "stopped at max_tokens" in err and "Let me think." in err

    def test_an_unknown_model_suggests_names(self, env, capsys):
        assert env.run("thnk", "hi") == 1
        err = capsys.readouterr().err
        assert "unknown model 'thnk'" in err and "did you mean: think" in err

    def test_a_config_without_the_llama_executable_is_refused(self, env, capsys):
        (env.repo / "config/config.yaml").write_text("results_dir: x\n")
        assert env.run("think", "hi") == 1
        assert "no llama_server" in capsys.readouterr().err

    def test_something_else_on_the_port_is_refused_and_left_alone(self, env, capsys):
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", env.port))
            listener.listen()
            assert env.run("think", "hi") == 1
            assert "LLM_PORT" in capsys.readouterr().err
            assert not (env.state / f"llm-{env.port}.json").exists()

    def test_a_server_that_fails_to_start_is_reported_with_its_log(self, env, capsys):
        assert env.run("wont-start", "hi") == 1
        err = capsys.readouterr().err
        assert "exited with code 3" in err and "server log:" in err
        assert keepalive.running_model(env.kept()) is None

    def test_a_request_the_server_rejects_is_an_error_and_the_server_stays(
        self, env, capsys
    ):
        assert env.run("rejects", "hi") == 1
        assert "exceeds the available context" in capsys.readouterr().err
        assert keepalive.running_model(env.kept()) == "rejects"

    def test_a_server_that_dies_is_forgotten_so_the_next_call_starts_a_new_one(
        self, env, capsys
    ):
        assert env.run("dies", "hi") == 1
        assert keepalive.running_model(env.kept()) is None
        capsys.readouterr()
        assert env.run("think", "hi") == 0
        err = capsys.readouterr().err
        assert "started the llama-server" in err and "stopping" not in err

    def test_a_server_that_hangs_is_stopped_not_left_for_the_next_call(
        self, env, capsys, monkeypatch
    ):
        env.run(
            "think", "hi"
        )  # a server this call did not start: chat() cannot stop it
        assert fakes_running(env.binary) != []

        def hangs(self: LlamaServer, *a: Any, **k: Any) -> Any:
            raise ServerError("server_died", "no output for 1800s (server stopped)")

        monkeypatch.setattr(LlamaServer, "chat", hangs)
        assert env.run("think", "again") == 1
        assert "no output" in capsys.readouterr().err
        assert keepalive.running_model(env.kept()) is None
        assert (
            fakes_running(env.binary) == []
        )  # the process itself is gone, not just forgotten


class TestImages:
    def pic(self, env: Env, name: str = "pic.png") -> str:
        path = env.tmp / name
        path.write_bytes(b"PNGDATA")
        return str(path)

    def first_content(self, env: Env, model: str) -> list[dict[str, Any]]:
        parts: list[dict[str, Any]] = json.loads(env.body(model).read_text())[
            "messages"
        ][0]["content"]
        return parts

    def test_images_go_with_the_prompt_before_its_text(self, env, capsys):
        # "plain" has an --mmproj in its server_args (the file need not exist for the fake)
        code = env.run(
            "plain",
            "What is this?",
            "--image",
            self.pic(env),
            "--image",
            self.pic(env, "b.png"),
        )
        assert code == 0
        content = self.first_content(env, "plain")
        assert [p["type"] for p in content] == ["image_url", "image_url", "text"]
        assert content[2]["text"] == "What is this?"

    def test_a_model_without_a_projector_is_refused_early(self, env, capsys):
        assert env.run("think", "hi", "--image", self.pic(env)) == 1
        assert "no --mmproj" in capsys.readouterr().err
        assert keepalive.running_model(env.kept()) is None  # nothing was loaded

    def test_a_missing_image_is_refused_early(self, env, capsys):
        assert env.run("plain", "hi", "--image", str(env.tmp / "gone.png")) == 1
        assert "image file not found" in capsys.readouterr().err
        assert keepalive.running_model(env.kept()) is None

    def test_in_a_chat_the_images_go_with_the_first_message_and_stay_in_the_history(
        self, env
    ):
        env.run(
            "plain",
            "--image",
            self.pic(env),
            lines=["describe it", "and its colors?"],
        )
        messages = json.loads(env.body("plain").read_text())["messages"]
        assert [m["role"] for m in messages] == ["user", "assistant", "user"]
        assert [p["type"] for p in messages[0]["content"]] == [
            "image_url",
            "text",
        ]  # still there
        assert messages[2]["content"] == "and its colors?"  # not sent again

    def test_a_cancelled_first_turn_keeps_the_images_for_the_next_one(
        self, env, monkeypatch
    ):
        real = LlamaServer.chat
        calls: list[int] = []

        def interrupted_once(self: LlamaServer, *a: Any, **k: Any) -> Any:
            calls.append(1)
            if len(calls) == 1:
                raise KeyboardInterrupt
            return real(self, *a, **k)

        monkeypatch.setattr(LlamaServer, "chat", interrupted_once)
        env.run("plain", "--image", self.pic(env), lines=["first", "second"])
        content = self.first_content(env, "plain")
        assert [p["type"] for p in content] == ["image_url", "text"] and content[1][
            "text"
        ] == "second"

    def test_an_image_that_cannot_be_read_mid_chat_drops_the_turn(self, env, capsys):
        path = self.pic(env)

        def deleted() -> None:
            os.remove(path)

        env.run("plain", "--image", path, lines=[deleted, "describe it", "hello"])
        assert "not found" in capsys.readouterr().err
        # the second turn went on without images: the images were pending, but the file is gone, so it fails again
        # (the chat survives; nothing is sent)
        assert not env.body("plain").exists()


class TestStop:
    def test_it_unloads_the_model(self, env, capsys):
        env.run("think", "hi")
        capsys.readouterr()
        assert env.run("--stop") == 0
        assert "stopped the llama-server" in capsys.readouterr().out
        assert keepalive.running_model(env.kept()) is None
        assert env.run("--stop") == 0
        assert "no llm server is running" in capsys.readouterr().out


class TestStatus:
    def test_not_running_then_running_then_stopped(self, env, capsys):
        assert env.run("--status") == 0
        assert f"not running (127.0.0.1:{env.port} is free)" in capsys.readouterr().out
        env.run("think", "hi")
        capsys.readouterr()
        assert env.run("--status") == 0
        out = capsys.readouterr().out
        assert (
            out.startswith("running think (pid ") and f"on 127.0.0.1:{env.port}" in out
        )
        env.run("--stop")
        capsys.readouterr()
        env.run("--status")
        assert "not running" in capsys.readouterr().out


class TestStart:
    def test_loads_the_model_without_asking_it_anything(self, env, capsys):
        assert env.run("--start", "think") == 0
        captured = capsys.readouterr()
        assert captured.out == "llama-server ready with think (started)\n"
        assert "started the llama-server" in captured.err
        assert keepalive.running_model(env.kept()) == "think"
        assert not env.body("think").exists()  # no request reached the model
        assert env.rows() == []  # and nothing is logged

    def test_a_second_start_finds_it_running(self, env, capsys):
        env.run("--start", "think")
        capsys.readouterr()
        assert env.run("--start") == 0  # no MODEL: the one that is loaded
        assert (
            capsys.readouterr().out
            == "llama-server ready with think (already running)\n"
        )

    def test_a_chat_after_it_starts_warm(self, env):
        env.run("--start", "think")
        env.run("think", "hi")
        assert env.rows()[0]["tags"]["server"] == "warm"

    def test_with_nothing_loaded_the_default_model_is_used(self, env, capsys):
        env.write_config(default_llm_model="plain")
        assert env.run("--start") == 0
        assert capsys.readouterr().out == "llama-server ready with plain (started)\n"

    def test_no_model_and_no_default_is_an_error(self, env, capsys):
        assert env.run("--start") == 1
        assert "no default_llm_model" in capsys.readouterr().err

    def test_another_model_replaces_the_loaded_one(self, env, capsys):
        env.run("--start", "think")
        capsys.readouterr()
        assert env.run("--start", "plain") == 0
        captured = capsys.readouterr()
        assert "stopping the running llama-server" in captured.err
        assert keepalive.running_model(env.kept()) == "plain"

    def test_an_unknown_model_is_an_error(self, env, capsys):
        assert env.run("--start", "thnk") == 1
        assert "did you mean: think" in capsys.readouterr().err

    def test_a_model_that_cannot_load_is_reported_with_the_log(self, env, capsys):
        assert env.run("--start", "wont-start") == 1
        err = capsys.readouterr().err
        assert "exited with code 3" in err and "server log:" in err
        assert keepalive.running_model(env.kept()) is None

    @pytest.mark.parametrize(
        "args",
        [
            ["--start", "think", "a prompt"],
            ["--start", "--system", "S"],
            ["--start", "--image", "x.png"],
            ["--start", "--reasoning", "off"],
            ["--start", "-o", "x.md"],
            ["--start", "--stop"],
            ["--start", "--list"],
        ],
    )
    def test_takes_only_a_model(self, args):
        with pytest.raises(SystemExit) as e:
            main(args)
        assert e.value.code == 2


class TestNoThinkingSampling:
    def give_think_both_sets(self, env: Env) -> None:
        models = yaml.safe_load((env.repo / "config/llm_models.yaml").read_text())
        models[0]["generation_options"] = {"temperature": 1.0, "top_p": 0.95}
        models[0]["no_thinking_options"] = {"temperature": 0.7, "top_p": 0.8}
        (env.repo / "config/llm_models.yaml").write_text(yaml.safe_dump(models))

    def sent(self, env: Env) -> dict[str, Any]:
        body: dict[str, Any] = json.loads(env.body("think").read_text())
        return body

    def test_reasoning_off_uses_the_non_thinking_sampling(self, env: Env) -> None:
        self.give_think_both_sets(env)
        assert env.run("think", "hi", "--reasoning", "off") == 0
        sent = self.sent(env)
        assert (sent["temperature"], sent["top_p"]) == (0.7, 0.8)
        assert sent["chat_template_kwargs"] == {"enable_thinking": False}
        row = env.rows()[0]
        assert row["sampling"]["temperature"] == 0.7

    def test_without_it_the_thinking_sampling_stays(self, env: Env) -> None:
        self.give_think_both_sets(env)
        assert env.run("think", "hi") == 0
        assert (self.sent(env)["temperature"], self.sent(env)["top_p"]) == (1.0, 0.95)


class TestHost:
    def test_the_server_listens_on_llm_host_and_status_finds_it(
        self, env, monkeypatch, capsys
    ):
        monkeypatch.setenv("LLM_HOST", "127.0.0.2")
        assert env.run("think", "hi") == 0
        capsys.readouterr()
        assert not keepalive._port_in_use("127.0.0.1", env.port)
        assert keepalive._port_in_use("127.0.0.2", env.port)
        monkeypatch.delenv("LLM_HOST")  # a later call without it still finds the server
        assert env.run("--status") == 0
        assert f"on 127.0.0.2:{env.port}" in capsys.readouterr().out

    def test_a_second_call_on_that_address_reuses_the_server(
        self, env, monkeypatch, capsys
    ):
        monkeypatch.setenv("LLM_HOST", "127.0.0.2")
        env.run("think", "one")
        capsys.readouterr()
        env.run("think", "two")
        assert "reusing" in capsys.readouterr().err
        assert [r["tags"]["server"] for r in env.rows()] == ["started", "warm"]

    def test_anything_but_an_ipv4_address_is_an_error(self, env, monkeypatch, capsys):
        monkeypatch.setenv("LLM_HOST", "localhost")
        assert env.run("think", "hi") == 1
        assert "LLM_HOST must be an IPv4 address" in capsys.readouterr().err
        assert env.rows() == []

    def test_listen_host_in_the_config_is_used_without_any_variable(
        self, env, monkeypatch, capsys
    ):
        monkeypatch.delenv("LLM_HOST", raising=False)
        env.write_config(listen_host="127.0.0.2")
        assert env.run("--start", "think") == 0
        assert keepalive._port_in_use("127.0.0.2", env.port)
        assert not keepalive._port_in_use("127.0.0.1", env.port)

    def test_the_variable_overrides_the_config_and_replaces_the_server(
        self, env, monkeypatch, capsys
    ):
        env.write_config(listen_host="127.0.0.2")
        env.run("--start", "think")
        capsys.readouterr()
        monkeypatch.setenv("LLM_HOST", "127.0.0.1")
        assert env.run("--start", "think") == 0
        assert "stopping the running llama-server" in capsys.readouterr().err
        assert keepalive._port_in_use("127.0.0.1", env.port)
        assert not keepalive._port_in_use("127.0.0.2", env.port)

    def test_a_bad_address_in_the_config_is_an_error(self, env, monkeypatch, capsys):
        monkeypatch.delenv("LLM_HOST", raising=False)
        env.write_config(listen_host="everywhere")
        assert env.run("think", "hi") == 1
        assert "listen_host in config/config.yaml must be an IPv4 address" in (
            capsys.readouterr().err
        )


class TestChat:
    def test_the_history_is_kept_without_the_thinking(self, env, capsys):
        code = env.run("think", lines=["hi", "and then?"])
        out, err = capsys.readouterr()
        assert code == 0
        assert out.count("Hello world") == 2
        assert env.messages("think") == [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "Hello world"},
            {"role": "user", "content": "and then?"},
        ]
        assert err.count("started the llama-server") == 1
        assert "reusing" not in err  # later turns are quiet

    def test_bye_ends_it_and_blank_lines_are_ignored(self, env, capsys):
        assert env.run("think", lines=["", "  ", "hi", "/bye", "never typed"]) == 0
        assert capsys.readouterr().out.count("Hello world") == 1
        assert len(env.messages("think")) == 1

    def test_a_system_prompt_starts_the_history(self, env):
        env.run("think", "--system", "Be brief.", lines=["hi"])
        assert [m["role"] for m in env.messages("think")] == ["system", "user"]

    def test_ctrl_c_during_a_reply_drops_that_turn(self, env, capsys, monkeypatch):
        real = LlamaServer.chat
        calls: list[int] = []

        def interrupted_once(self: LlamaServer, *a: Any, **k: Any) -> Any:
            calls.append(1)
            if len(calls) == 1:
                raise KeyboardInterrupt
            return real(self, *a, **k)

        monkeypatch.setattr(LlamaServer, "chat", interrupted_once)
        assert env.run("think", lines=["first", "second"]) == 0
        assert "(cancelled)" in capsys.readouterr().err
        assert env.messages("think") == [{"role": "user", "content": "second"}]

    def test_ctrl_c_at_the_prompt_does_not_end_the_chat(self, env, capsys):
        assert env.run("think", lines=[KeyboardInterrupt, "hi"]) == 0
        captured = capsys.readouterr()
        assert "Ctrl-D or /bye" in captured.err and "Hello world" in captured.out

    def test_a_failed_turn_is_dropped_and_the_chat_goes_on(self, env, capsys):
        assert env.run("rejects", lines=["a", "b"]) == 0
        err = capsys.readouterr().err
        assert err.count("exceeds the available context") == 2
        assert env.messages("rejects") == [
            {"role": "user", "content": "b"}
        ]  # "a" was dropped

    def test_a_reply_without_an_answer_is_not_kept_in_the_history(self, env, capsys):
        assert env.run("thinker", lines=["a", "b"]) == 0
        assert capsys.readouterr().err.count("still thinking") == 2
        assert env.messages("thinker") == [{"role": "user", "content": "b"}]

    def test_images_wait_for_a_turn_that_gets_an_answer(self, env, capsys):
        pic = env.tmp / "p.png"
        pic.write_bytes(b"x")
        models = yaml.safe_load((env.repo / "config/llm_models.yaml").read_text())
        models[2]["server_args"]["--mmproj"] = "not-on-disk.gguf"
        (env.repo / "config/llm_models.yaml").write_text(yaml.safe_dump(models))
        env.run("thinker", "--image", str(pic), lines=["a", "b"])
        content = json.loads(env.body("thinker").read_text())["messages"][0]["content"]
        assert [p["type"] for p in content] == ["image_url", "text"]  # still on "b"

    def test_a_server_replaced_between_turns_is_started_again(self, env, capsys):
        def another_call_unloads_it() -> None:
            keepalive.stop_server(env.kept())  # what another call's replacement does

        code = env.run("think", lines=["hi", another_call_unloads_it, "again"])
        err = capsys.readouterr().err
        assert code == 0
        assert (
            err.count("loading think") == 2
        )  # told again; the later "started" line is quiet
        assert err.count("started the llama-server") == 1
        assert len(env.messages("think")) == 3  # the history survived

    def test_a_server_that_dies_mid_chat_is_replaced_on_the_next_turn(
        self, env, capsys
    ):
        assert env.run("dies", lines=["a", "b"]) == 0
        err = capsys.readouterr().err
        assert (
            err.count("llm failed") == 2 and err.count("started the llama-server") == 2
        )


class TestUsage:
    def test_a_model_or_a_mode_is_required(self):
        with pytest.raises(SystemExit) as e:
            main([])
        assert e.value.code == 2

    @pytest.mark.parametrize(
        "args",
        [
            ["--list", "extra"],
            ["--status", "plain", "hi"],
            ["--stop", "--system", "S"],
            ["--list", "--stop"],
            ["--status", "--image", "x.png"],
            ["--stop", "--reasoning", "off"],
        ],
    )
    def test_the_modes_take_nothing_else(self, args):
        with pytest.raises(SystemExit) as e:
            main(args)
        assert e.value.code == 2

    def test_an_unknown_model_suggests_close_ones(self, env, capsys):
        assert env.run("plian", "hi") == 1
        assert "did you mean: plain" in capsys.readouterr().err

    def test_the_script_runs_and_documents_itself(self):
        result = subprocess.run(
            [sys.executable, str(REPO / "scripts" / "llm"), "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0
        assert "llm MODEL [PROMPT]" in result.stdout and "LLM_PORT" in result.stdout


class TestOptionOrder:
    """Options of `run` may come before or after the prompt (argparse subparsers cannot do that)."""

    def test_options_after_the_prompt(self, env, capsys):
        pic = env.tmp / "p.png"
        pic.write_bytes(b"x")
        assert env.run("plain", "What?", "--image", str(pic), "--system", "S") == 0
        assert [m["role"] for m in env.messages("plain")] == ["system", "user"]

    def test_options_between_the_model_and_the_prompt(self, env, capsys):
        pic = env.tmp / "p.png"
        pic.write_bytes(b"x")
        assert env.run("plain", "--image", str(pic), "What?") == 0
        content = json.loads(env.body("plain").read_text())["messages"][0]["content"]
        assert [p["type"] for p in content] == ["image_url", "text"] and content[1][
            "text"
        ] == "What?"

    def test_options_before_the_model(self, env):
        assert env.run("--system", "S", "think", "hi") == 0
        assert env.messages("think")[0] == {"role": "system", "content": "S"}

    def test_help_lists_the_options(self, capsys):
        with pytest.raises(SystemExit) as e:
            main(["--help"])
        assert e.value.code == 0
        out = capsys.readouterr().out
        assert "--image FILE" in out and "--system" in out


class TestReasoning:
    def request(self, env: Env, model: str) -> dict[str, Any]:
        body: dict[str, Any] = json.loads(env.body(model).read_text())
        return body

    def test_off_sends_enable_thinking_false(self, env: Env) -> None:
        assert env.run("plain", "hi", "--reasoning", "off") == 0
        assert self.request(env, "plain")["chat_template_kwargs"] == {
            "enable_thinking": False
        }

    def test_without_the_flag_nothing_is_sent(self, env: Env) -> None:
        assert env.run("plain", "hi") == 0
        assert "chat_template_kwargs" not in self.request(env, "plain")

    def test_it_merges_with_the_models_own_template_kwargs(self, env: Env) -> None:
        models = yaml.safe_load((env.repo / "config/llm_models.yaml").read_text())
        models[1]["generation_options"] = {
            "chat_template_kwargs": {"reasoning_effort": "low"}
        }
        (env.repo / "config/llm_models.yaml").write_text(yaml.safe_dump(models))
        env.run("plain", "hi", "--reasoning", "off")
        assert self.request(env, "plain")["chat_template_kwargs"] == {
            "reasoning_effort": "low",
            "enable_thinking": False,
        }

    def test_in_a_chat_every_turn_has_it(self, env: Env) -> None:
        env.run("plain", "--reasoning", "off", lines=["a", "b"])
        assert self.request(env, "plain")["chat_template_kwargs"] == {
            "enable_thinking": False
        }

    def test_other_values_are_a_usage_error(self) -> None:
        with pytest.raises(SystemExit) as e:
            main(["plain", "hi", "--reasoning", "high"])
        assert e.value.code == 2


class TestRunLog:
    def test_a_one_shot_writes_one_row_like_the_runner(self, env, capsys):
        env.run("think", "hi", "--system", "Be brief.")
        (row,) = env.rows()
        assert (row["status"], row["model"]) == ("ok", "think")
        assert row["request"]["messages"] == [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "hi"},
        ]
        assert isinstance(row["seed"], int) and row["seed"] >= 0  # picked and recorded
        assert (
            row["request"]["seed"] == row["seed"] and row["request"]["max_tokens"] == 50
        )
        assert row["output"] == {"text": "Hello world", "reasoning": "Let me think."}
        assert row["finish_reason"] == "stop"
        assert row["metrics"]["input_tokens"] == 10
        assert row["sampling"]["temperature"] == 0.8  # the fake server's own default
        assert row["server_args"]["--mode"] == "think"
        assert row["tags"] == {"turn": 1, "server": "started"}
        assert row["output_file"] is None
        assert row["server_log"] == str(env.kept().log)

    def test_a_second_call_finds_the_server_warm_and_is_its_own_run(self, env):
        env.run("think", "one")
        env.run("think", "two")
        first, second = env.rows()
        assert (first["tags"]["server"], second["tags"]["server"]) == (
            "started",
            "warm",
        )
        assert first["run_id"] != second["run_id"]

    def test_a_chat_logs_each_turn_in_one_run(self, env):
        env.run("think", lines=["hi", "and then?"])
        first, second = env.rows()
        assert first["run_id"] == second["run_id"]
        assert (first["tags"], second["tags"]) == (
            {"turn": 1, "server": "started"},
            {"turn": 2, "server": "warm"},
        )
        assert len(first["request"]["messages"]) == 1
        assert [m["role"] for m in second["request"]["messages"]] == [
            "user",
            "assistant",
            "user",
        ]
        assert second["request"]["messages"][1]["content"] == "Hello world"

    def test_the_reasoning_flag_is_in_the_logged_request(self, env):
        env.run("think", "hi", "--reasoning", "off")
        assert env.rows()[0]["request"]["chat_template_kwargs"] == {
            "enable_thinking": False
        }

    def test_images_are_logged_as_paths_never_as_data(self, env):
        pic = env.tmp / "p.png"
        pic.write_bytes(b"PNGDATA")
        env.run("plain", "what?", "--image", str(pic))
        text = env.log_file().read_text()
        assert str(pic) in text and "base64" not in text
        assert env.rows()[0]["request"]["messages"][0]["images"] == [str(pic)]

    def test_the_log_is_under_the_repo_not_the_current_folder(self, env, monkeypatch):
        elsewhere = env.tmp / "elsewhere"
        elsewhere.mkdir()
        monkeypatch.chdir(elsewhere)
        env.run("think", "hi")
        assert len(env.rows()) == 1
        assert list(elsewhere.iterdir()) == []

    def test_a_rejected_request_is_a_failed_row_and_the_server_stays(self, env, capsys):
        assert env.run("rejects", "hi") == 1
        (row,) = env.rows()
        assert (row["status"], row["kind"]) == ("failed", "job_failed")
        assert "exceeds the available context" in row["error"]
        assert row["output"] is None and row["request"]["messages"]
        err = capsys.readouterr().err
        assert f"(server log: {env.kept().log})" in err
        assert keepalive.running_model(env.kept()) == "rejects"

    def test_a_server_that_dies_is_logged_and_stopped(self, env, capsys):
        assert env.run("dies", "hi") == 1
        (row,) = env.rows()
        assert (row["status"], row["kind"]) == ("failed", "server_died")
        assert keepalive.running_model(env.kept()) is None

    def test_a_server_that_will_not_start_is_logged(self, env, capsys):
        assert env.run("wont-start", "hi") == 1
        (row,) = env.rows()
        assert (row["status"], row["kind"], row["exit_code"]) == (
            "failed",
            "server_died",
            3,
        )
        assert row["tags"] == {"turn": 1}  # it never got as far as "started"

    def test_a_cancelled_turn_is_not_logged(self, env, monkeypatch):
        real = LlamaServer.chat
        calls: list[int] = []

        def interrupted_once(self: LlamaServer, *a: Any, **k: Any) -> Any:
            calls.append(1)
            if len(calls) == 1:
                raise KeyboardInterrupt
            return real(self, *a, **k)

        monkeypatch.setattr(LlamaServer, "chat", interrupted_once)
        env.run("think", lines=["first", "second"])
        (row,) = env.rows()
        assert row["request"]["messages"] == [{"role": "user", "content": "second"}]
        assert row["tags"]["turn"] == 2  # turns count attempts

    def test_an_option_the_server_would_ignore_is_an_error_not_a_traceback(
        self, env, capsys
    ):
        models = yaml.safe_load((env.repo / "config/llm_models.yaml").read_text())
        models[0]["generation_options"] = {"num_predict": 5}
        (env.repo / "config/llm_models.yaml").write_text(yaml.safe_dump(models))
        assert env.run("think", "hi") == 1
        assert "num_predict" in capsys.readouterr().err
        assert env.rows() == []


class TestOutputFile:
    def test_the_answer_is_written_and_printed_and_the_log_goes_beside_it(
        self, env, capsys
    ):
        out = env.tmp / "answers" / "a.md"
        assert env.run("think", "hi", "-o", str(out)) == 0
        assert out.read_text() == "Hello world"
        assert "Hello world" in capsys.readouterr().out
        (row,) = env.rows(out.parent / ".llm" / "runs.jsonl")
        assert row["output_file"] == str(out)
        assert env.rows() == []  # nothing in the fixed place

    def test_a_piped_prompt_works_with_it(self, env):
        out = env.tmp / "a.md"
        assert env.run("think", "-o", str(out), stdin="from a pipe") == 0
        assert out.read_text() == "Hello world"

    def test_an_existing_file_is_never_overwritten_and_nothing_starts(
        self, env, capsys
    ):
        out = env.tmp / "a.md"
        out.write_text("mine")
        assert env.run("think", "hi", "-o", str(out)) == 1
        assert "already exists" in capsys.readouterr().err
        assert out.read_text() == "mine"
        assert keepalive.running_model(env.kept()) is None
        assert env.rows(env.tmp / ".llm" / "runs.jsonl") == []

    def test_a_folder_says_so(self, env, capsys):
        assert env.run("think", "hi", "-o", str(env.tmp)) == 1
        err = capsys.readouterr().err
        assert "is a folder; give a file name" in err and "answer.md" in err

    def test_a_chat_has_no_single_answer(self, env, capsys):
        assert env.run("think", "-o", str(env.tmp / "a.md"), lines=["hi"]) == 1
        assert "-o needs a prompt" in capsys.readouterr().err
        assert keepalive.running_model(env.kept()) is None

    def test_a_file_made_while_waiting_for_the_lock_is_not_overwritten(
        self, env, capsys
    ):
        out = env.tmp / "a.md"
        result: dict[str, int] = {}

        def second_call() -> None:
            result["code"] = env.run("think", "hi", "-o", str(out))

        with keepalive.call_lock(env.kept()):  # the first call holds the lock
            thread = threading.Thread(target=second_call)
            thread.start()
            time.sleep(0.3)  # the second call has passed its first check
            out.write_text("first call")
        thread.join(timeout=20)

        assert result["code"] == 1
        assert "already exists" in capsys.readouterr().err
        assert out.read_text() == "first call"

    @pytest.mark.parametrize(
        "args",
        [
            ["--list", "-o", "x.md"],
            ["--status", "-o", "x.md"],
            ["--stop", "-o", "x.md"],
        ],
    )
    def test_the_modes_take_no_output_file(self, args):
        with pytest.raises(SystemExit) as e:
            main(args)
        assert e.value.code == 2


class TestClosedPipe:
    """`llm MODEL "..." | head`: the reader leaves, the reply stops quietly and the server stays usable."""

    class Gone(io.StringIO):
        def write(self, text: str) -> int:
            raise BrokenPipeError(32, "Broken pipe")

    def test_it_stops_quietly_and_is_not_logged(self, env, capsys, monkeypatch):
        monkeypatch.setattr(sys, "stdout", self.Gone())
        assert env.run("think", "hi") == 141
        err = capsys.readouterr().err
        assert "llm failed" not in err and "Broken pipe" not in err
        assert env.rows() == []
        assert keepalive.running_model(env.kept()) == "think"  # not stopped

    def test_the_next_call_works(self, env, capsys, monkeypatch):
        with monkeypatch.context() as m:
            m.setattr(sys, "stdout", self.Gone())
            env.run("think", "hi")
        capsys.readouterr()
        assert env.run("think", "again") == 0
        assert "Hello world" in capsys.readouterr().out
        assert len(env.rows()) == 1

    def test_a_pipe_that_closes_as_the_reply_ends_is_the_same(self, env, monkeypatch):
        class GoneAtTheEnd(io.StringIO):
            def write(self, text: str) -> int:
                if text == "\n":
                    raise BrokenPipeError(32, "Broken pipe")
                return super().write(text)

        monkeypatch.setattr(sys, "stdout", GoneAtTheEnd())
        assert env.run("think", "hi") == 141
