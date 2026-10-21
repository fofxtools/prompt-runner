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
        env.run("list")
        before = capsys.readouterr().out.splitlines()
        assert before[0].split() == ["NAME", "SIZE", "STATE"]
        rows = {line.split()[0]: line.split() for line in before[1:]}
        assert rows["think"][1:3] == ["0.0", "GB"]  # 2 MB
        assert rows["plain"][-1] == "missing"  # its --mmproj is not on disk
        assert "running" not in "".join(before)

        env.run("run", "think", "hi")
        capsys.readouterr()
        env.run("list")
        after = capsys.readouterr().out.splitlines()
        states = {line.split()[0]: line.split()[-1] for line in after[1:]}
        assert (
            states["think"] == "running" and states["thinker"] == "GB"
        )  # only one is loaded


class TestRunOnce:
    def test_the_answer_is_on_stdout_and_the_thinking_on_stderr(self, env, capsys):
        code = env.run("run", "think", "hi")
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
        env.run("run", "think", "hi")
        body = json.loads(env.body("think").read_text())
        assert (body["temperature"], body["max_tokens"], body["stream"]) == (
            0.3,
            50,
            True,
        )

    def test_the_server_stays_loaded_and_the_next_call_reuses_it(self, env, capsys):
        env.run("run", "think", "one")
        capsys.readouterr()
        code = env.run("run", "think", "two")
        out, err = capsys.readouterr()
        assert code == 0 and out == "Hello world\n"
        assert "reusing the llama-server already running" in err
        assert "loading" not in err and "started" not in err
        assert keepalive.running_model(env.kept()) == "think"

    def test_piped_text_is_the_prompt(self, env, capsys):
        code = env.run("run", "think", stdin="text from a pipe\n")
        assert code == 0 and capsys.readouterr().out == "Hello world\n"
        assert env.messages("think") == [
            {"role": "user", "content": "text from a pipe"}
        ]

    def test_stdin_is_not_read_when_a_prompt_is_given(self, env):
        class NeverEnds(io.StringIO):
            def read(self, size: int | None = -1) -> str:
                raise AssertionError("an open pipe that never ends would hang here")

        assert env.run("run", "think", "Summarize:", stdin=NeverEnds()) == 0
        assert env.messages("think")[0]["content"] == "Summarize:"

    def test_an_empty_prompt_is_refused(self, env, capsys):
        assert env.run("run", "think", stdin="") == 1
        assert "the prompt is empty" in capsys.readouterr().err
        assert keepalive.running_model(env.kept()) is None

    def test_a_system_prompt_goes_first(self, env):
        env.run("run", "think", "hi", "--system", "Be brief.")
        assert env.messages("think") == [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "hi"},
        ]

    def test_another_model_replaces_the_server(self, env, capsys):
        env.run("run", "think", "hi")
        capsys.readouterr()
        env.run("run", "plain", "hi")
        err = capsys.readouterr().err
        assert "stopping the running llama-server" in err and "loading plain" in err
        assert keepalive.running_model(env.kept()) == "plain"

    def test_a_reply_without_an_answer_is_flagged_not_hidden(self, env, capsys):
        assert env.run("run", "thinker", "hi") == 0
        out, err = capsys.readouterr()
        assert out == ""
        assert "stopped at max_tokens" in err and "Let me think." in err

    def test_an_unknown_model_suggests_names(self, env, capsys):
        assert env.run("run", "thnk", "hi") == 1
        err = capsys.readouterr().err
        assert "unknown model 'thnk'" in err and "did you mean: think" in err

    def test_a_config_without_the_llama_executable_is_refused(self, env, capsys):
        (env.repo / "config/config.yaml").write_text("results_dir: x\n")
        assert env.run("run", "think", "hi") == 1
        assert "no llama_server" in capsys.readouterr().err

    def test_something_else_on_the_port_is_refused_and_left_alone(self, env, capsys):
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", env.port))
            listener.listen()
            assert env.run("run", "think", "hi") == 1
            assert "LLM_PORT" in capsys.readouterr().err
            assert not (env.state / f"llm-{env.port}.json").exists()

    def test_a_server_that_fails_to_start_is_reported_with_its_log(self, env, capsys):
        assert env.run("run", "wont-start", "hi") == 1
        err = capsys.readouterr().err
        assert "exited with code 3" in err and "server log:" in err
        assert keepalive.running_model(env.kept()) is None

    def test_a_request_the_server_rejects_is_an_error_and_the_server_stays(
        self, env, capsys
    ):
        assert env.run("run", "rejects", "hi") == 1
        assert "exceeds the available context" in capsys.readouterr().err
        assert keepalive.running_model(env.kept()) == "rejects"

    def test_a_server_that_dies_is_forgotten_so_the_next_call_starts_a_new_one(
        self, env, capsys
    ):
        assert env.run("run", "dies", "hi") == 1
        assert keepalive.running_model(env.kept()) is None
        capsys.readouterr()
        assert env.run("run", "think", "hi") == 0
        err = capsys.readouterr().err
        assert "started the llama-server" in err and "stopping" not in err

    def test_a_server_that_hangs_is_stopped_not_left_for_the_next_call(
        self, env, capsys, monkeypatch
    ):
        env.run(
            "run", "think", "hi"
        )  # a server this call did not start: chat() cannot stop it
        assert fakes_running(env.binary) != []

        def hangs(self: LlamaServer, *a: Any, **k: Any) -> Any:
            raise ServerError("server_died", "no output for 1800s (server stopped)")

        monkeypatch.setattr(LlamaServer, "chat", hangs)
        assert env.run("run", "think", "again") == 1
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
            "run",
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
        assert env.run("run", "think", "hi", "--image", self.pic(env)) == 1
        assert "no --mmproj" in capsys.readouterr().err
        assert keepalive.running_model(env.kept()) is None  # nothing was loaded

    def test_a_missing_image_is_refused_early(self, env, capsys):
        assert env.run("run", "plain", "hi", "--image", str(env.tmp / "gone.png")) == 1
        assert "image file not found" in capsys.readouterr().err
        assert keepalive.running_model(env.kept()) is None

    def test_in_a_chat_the_images_go_with_the_first_message_and_stay_in_the_history(
        self, env
    ):
        env.run(
            "run",
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
        env.run("run", "plain", "--image", self.pic(env), lines=["first", "second"])
        content = self.first_content(env, "plain")
        assert [p["type"] for p in content] == ["image_url", "text"] and content[1][
            "text"
        ] == "second"

    def test_an_image_that_cannot_be_read_mid_chat_drops_the_turn(self, env, capsys):
        path = self.pic(env)

        def deleted() -> None:
            os.remove(path)

        env.run(
            "run", "plain", "--image", path, lines=[deleted, "describe it", "hello"]
        )
        assert "No such file" in capsys.readouterr().err
        # the second turn went on without images: the images were pending, but the file is gone, so it fails again
        # (the chat survives; nothing is sent)
        assert not env.body("plain").exists()


class TestStop:
    def test_it_unloads_the_model(self, env, capsys):
        env.run("run", "think", "hi")
        capsys.readouterr()
        assert env.run("stop") == 0
        assert "stopped the llama-server" in capsys.readouterr().out
        assert keepalive.running_model(env.kept()) is None
        assert env.run("stop") == 0
        assert "no llm server is running" in capsys.readouterr().out


class TestChat:
    def test_the_history_is_kept_without_the_thinking(self, env, capsys):
        code = env.run("run", "think", lines=["hi", "and then?"])
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
        assert (
            env.run("run", "think", lines=["", "  ", "hi", "/bye", "never typed"]) == 0
        )
        assert capsys.readouterr().out.count("Hello world") == 1
        assert len(env.messages("think")) == 1

    def test_a_system_prompt_starts_the_history(self, env):
        env.run("run", "think", "--system", "Be brief.", lines=["hi"])
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
        assert env.run("run", "think", lines=["first", "second"]) == 0
        assert "(cancelled)" in capsys.readouterr().err
        assert env.messages("think") == [{"role": "user", "content": "second"}]

    def test_ctrl_c_at_the_prompt_does_not_end_the_chat(self, env, capsys):
        assert env.run("run", "think", lines=[KeyboardInterrupt, "hi"]) == 0
        captured = capsys.readouterr()
        assert "Ctrl-D or /bye" in captured.err and "Hello world" in captured.out

    def test_a_failed_turn_is_dropped_and_the_chat_goes_on(self, env, capsys):
        assert env.run("run", "rejects", lines=["a", "b"]) == 0
        err = capsys.readouterr().err
        assert err.count("exceeds the available context") == 2
        assert env.messages("rejects") == [
            {"role": "user", "content": "b"}
        ]  # "a" was dropped

    def test_a_reply_without_an_answer_is_not_kept_in_the_history(self, env, capsys):
        assert env.run("run", "thinker", lines=["a", "b"]) == 0
        assert capsys.readouterr().err.count("still thinking") == 2
        assert env.messages("thinker") == [{"role": "user", "content": "b"}]

    def test_images_wait_for_a_turn_that_gets_an_answer(self, env, capsys):
        pic = env.tmp / "p.png"
        pic.write_bytes(b"x")
        models = yaml.safe_load((env.repo / "config/llm_models.yaml").read_text())
        models[2]["server_args"]["--mmproj"] = "not-on-disk.gguf"
        (env.repo / "config/llm_models.yaml").write_text(yaml.safe_dump(models))
        env.run("run", "thinker", "--image", str(pic), lines=["a", "b"])
        content = json.loads(env.body("thinker").read_text())["messages"][0]["content"]
        assert [p["type"] for p in content] == ["image_url", "text"]  # still on "b"

    def test_a_server_replaced_between_turns_is_started_again(self, env, capsys):
        def another_call_unloads_it() -> None:
            keepalive.stop_server(env.kept())  # what another call's replacement does

        code = env.run("run", "think", lines=["hi", another_call_unloads_it, "again"])
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
        assert env.run("run", "dies", lines=["a", "b"]) == 0
        err = capsys.readouterr().err
        assert (
            err.count("llm failed") == 2 and err.count("started the llama-server") == 2
        )


class TestUsage:
    def test_a_command_is_required(self):
        with pytest.raises(SystemExit) as e:
            main([])
        assert e.value.code == 2

    def test_run_needs_a_model(self):
        with pytest.raises(SystemExit) as e:
            main(["run"])
        assert e.value.code == 2

    def test_the_script_runs_and_documents_itself(self):
        result = subprocess.run(
            [sys.executable, str(REPO / "scripts" / "llm"), "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0
        assert "llm run MODEL [PROMPT]" in result.stdout and "LLM_PORT" in result.stdout


class TestOptionOrder:
    """Options of `run` may come before or after the prompt (argparse subparsers cannot do that)."""

    def test_options_after_the_prompt(self, env, capsys):
        pic = env.tmp / "p.png"
        pic.write_bytes(b"x")
        assert (
            env.run("run", "plain", "What?", "--image", str(pic), "--system", "S") == 0
        )
        assert [m["role"] for m in env.messages("plain")] == ["system", "user"]

    def test_options_between_the_model_and_the_prompt(self, env, capsys):
        pic = env.tmp / "p.png"
        pic.write_bytes(b"x")
        assert env.run("run", "plain", "--image", str(pic), "What?") == 0
        content = json.loads(env.body("plain").read_text())["messages"][0]["content"]
        assert [p["type"] for p in content] == ["image_url", "text"] and content[1][
            "text"
        ] == "What?"

    def test_options_before_the_model(self, env):
        assert env.run("run", "--system", "S", "think", "hi") == 0
        assert env.messages("think")[0] == {"role": "system", "content": "S"}

    def test_run_help_lists_its_options(self, capsys):
        with pytest.raises(SystemExit) as e:
            main(["run", "--help"])
        assert e.value.code == 0
        out = capsys.readouterr().out
        assert "--image FILE" in out and "--system" in out

    def test_list_takes_no_arguments(self):
        with pytest.raises(SystemExit) as e:
            main(["list", "extra"])
        assert e.value.code == 2

    def test_an_unknown_command_is_a_usage_error(self):
        with pytest.raises(SystemExit) as e:
            main(["walk"])
        assert e.value.code == 2
