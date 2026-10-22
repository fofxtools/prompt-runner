"""llm against a server on "another machine": a fake `llama serve` on 127.0.0.1, `LLM_URL` pointing at it.

Every keepalive function is patched to fail the test: remote mode must not start, stop, lock or remember anything.
"""

import builtins
import io
import json
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
import yaml

from prompt_runner import keepalive
from prompt_runner.llama_server import LlamaServer
from prompt_runner.llm import main
from prompt_runner.server_process import free_port
from tests.test_llm import Tty

MODELS = [
    {"name": "think", "server_args": {"-m": "/models/think.gguf"}},
    {"name": "other", "server_args": {"-m": "/models/other.gguf"}},
    {
        "name": "vision",
        "server_args": {"-m": "/models/think.gguf", "--mmproj": "/models/mm.gguf"},
    },
]


@pytest.fixture
def no_keepalive(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("remote mode used keepalive")

    for name in (
        "ensure_server",
        "kept_server",
        "call_lock",
        "running_model",
        "stop_server",
        "check_supported",
        "server_status",
    ):
        monkeypatch.setattr(keepalive, name, boom)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A prompt-runner folder as the VM has it: no `llama_server` in the config."""
    root = tmp_path / "repo"
    (root / "config").mkdir(parents=True)
    config = {
        "results_dir": "storage/results/",
        "llm_generation_defaults": {"max_tokens": 50},
    }
    (root / "config/config.yaml").write_text(yaml.safe_dump(config))
    (root / "config/llm_models.yaml").write_text(yaml.safe_dump(MODELS))
    return root


@pytest.fixture
def serving(
    fake_llama: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_keepalive: None
) -> Callable[..., Any]:
    @contextmanager
    def serve(args: dict[str, str] | None = None) -> Iterator[int]:
        port = free_port()
        with LlamaServer(fake_llama, args or {}, tmp_path / "host.log", port=port):
            monkeypatch.setenv("LLM_URL", f"http://127.0.0.1:{port}")
            yield port

    return serve


def llm(
    repo: Path,
    *args: str,
    stdin: str = "",
    lines: list[str] | None = None,
    mp: pytest.MonkeyPatch | None = None,
) -> int:
    assert mp is not None
    if lines is not None:
        mp.setattr(sys, "stdin", Tty())
        typed = iter(lines)

        def fake_input(prompt: str = "") -> str:
            item = next(typed, None)
            if item is None:
                raise EOFError
            return item

        mp.setattr(builtins, "input", fake_input)
    else:
        mp.setattr(sys, "stdin", io.StringIO(stdin))
    return main(list(args), repo_root=repo)


@pytest.fixture
def run(repo: Path, monkeypatch: pytest.MonkeyPatch) -> Callable[..., int]:
    return lambda *args, **kw: llm(repo, *args, mp=monkeypatch, **kw)


def log_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()]


@pytest.fixture
def rows(repo: Path) -> Callable[[], list[dict[str, Any]]]:
    return lambda: log_rows(repo / "storage/results/llm-cli/runs.jsonl")


class TestReply:
    def test_a_one_shot_is_answered_and_logged_like_a_local_one(
        self, serving, run, rows, capsys
    ):
        with serving({"-m": "/models/think.gguf", "--mode": "think"}) as port:
            assert run("think", "hi") == 0
        captured = capsys.readouterr()
        assert captured.out == "Hello world\n"
        (row,) = rows()
        assert (row["status"], row["model"]) == ("ok", "think")
        assert row["tags"] == {
            "turn": 1,
            "server": "remote",
            "url": f"http://127.0.0.1:{port}",
        }
        assert row["engine"] == "b1-fake (remote)" and row["server_log"] is None
        assert row["server_args"] == {"-m": "/models/think.gguf"}  # as configured
        assert row["output"] == {"text": "Hello world", "reasoning": "Let me think."}
        assert isinstance(row["seed"], int) and row["metrics"]["input_tokens"] == 10

    def test_a_chat_logs_a_row_per_turn_in_one_run(self, serving, run, rows):
        with serving({"-m": "/models/think.gguf", "--mode": "nothink"}):
            assert run("think", lines=["hi", "and then?"]) == 0
        first, second = rows()
        assert first["run_id"] == second["run_id"]
        assert [r["tags"]["turn"] for r in (first, second)] == [1, 2]
        assert [m["role"] for m in second["request"]["messages"]] == [
            "user",
            "assistant",
            "user",
        ]

    def test_the_options_still_work(self, serving, run, tmp_path, repo, capsys):
        pic = tmp_path / "p.png"
        pic.write_bytes(b"PNGDATA")
        record, out = tmp_path / "body.json", tmp_path / "answers" / "a.md"
        with serving(
            {"-m": "/models/think.gguf", "--mode": "nothink", "--record": str(record)}
        ):
            code = run(
                "vision",
                "what?",
                "--image",
                str(pic),
                "--reasoning",
                "off",
                "-o",
                str(out),
            )
        assert code == 0
        body = json.loads(record.read_text())
        assert body["chat_template_kwargs"] == {"enable_thinking": False}
        assert body["messages"][0]["content"][0]["image_url"]["url"].startswith(
            "data:image/png;base64,"
        )
        assert out.read_text() == "Hello world"
        (row,) = log_rows(out.parent / ".llm" / "runs.jsonl")
        assert row["tags"]["server"] == "remote" and row["output_file"] == str(out)

    def test_a_server_that_reports_no_file_skips_the_check(self, serving, run, rows):
        with serving({"--mode": "nothink"}):  # no -m: /props has no model_path
            assert run("other", "hi") == 0
        assert rows()[0]["model"] == "other"

    def test_the_log_is_under_the_repo_even_without_a_binary_in_the_config(
        self, serving, run, repo, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        with serving({"-m": "/models/think.gguf", "--mode": "nothink"}):
            run("think", "hi")
        assert (repo / "storage/results/llm-cli/runs.jsonl").exists()
        assert not (tmp_path / "storage").exists()


class TestRefusals:
    def test_another_model_than_the_loaded_one_names_both_and_the_command_to_load_it(
        self, serving, run, rows, capsys
    ):
        with serving({"-m": "/models/think.gguf", "--mode": "nothink"}) as port:
            assert run("other", "hi") == 1
        err = capsys.readouterr().err
        assert (
            f"llm failed: the server at http://127.0.0.1:{port} has think.gguf loaded"
            in err
        )
        assert "'other' needs other.gguf" in err and err.rstrip().endswith(
            "llm --start other"
        )
        assert rows() == []  # nothing was asked of the model

    def test_in_a_chat_that_does_not_end_the_chat(self, serving, run, capsys):
        with serving({"-m": "/models/think.gguf", "--mode": "nothink"}):
            assert run("other", lines=["hi"]) == 0
        assert "llm failed: the server at" in capsys.readouterr().err

    @pytest.mark.parametrize("flag", ["--start", "--stop"])
    def test_start_and_stop_belong_to_the_host(self, serving, run, capsys, flag):
        with serving({"-m": "/models/think.gguf"}):
            assert run(flag) == 1
        err = capsys.readouterr().err
        assert f"{flag} is not available for a remote server (LLM_URL is set)" in err

    def test_a_server_that_is_not_there_is_one_line(
        self, repo, run, monkeypatch, no_keepalive, capsys
    ):
        port = free_port()
        monkeypatch.setenv("LLM_URL", f"http://127.0.0.1:{port}")
        assert run("think", "hi") == 1
        err = capsys.readouterr().err
        assert err.startswith(
            f"llm failed: cannot reach the server at http://127.0.0.1:{port}: "
        )
        assert err.count("\n") == 1

    def test_a_bad_url_is_an_error_not_a_traceback(
        self, run, monkeypatch, no_keepalive, capsys
    ):
        monkeypatch.setenv("LLM_URL", "192.168.56.1:8191")
        assert run("think", "hi") == 1
        assert "llm failed: LLM_URL must look like http://" in capsys.readouterr().err

    def test_a_server_that_dies_mid_reply_is_a_failed_row_that_points_at_the_server(
        self, serving, run, rows, capsys
    ):
        with serving({"-m": "/models/think.gguf", "--mode": "die"}) as port:
            assert run("think", "hi") == 1
        err = capsys.readouterr().err
        assert (
            f"(server: http://127.0.0.1:{port})" in err and "server stopped" not in err
        )
        (row,) = rows()
        assert (row["status"], row["kind"]) == ("failed", "server_died")


class TestCommands:
    def test_status_names_the_file_and_the_config_entries(self, serving, run, capsys):
        with serving({"-m": "/models/think.gguf"}) as port:
            assert run("--status") == 0
        assert capsys.readouterr().out == (
            f"running think.gguf at http://127.0.0.1:{port} (remote; config entry: think, vision)\n"
        )

    def test_status_of_a_file_the_config_does_not_know(self, serving, run, capsys):
        with serving({"-m": "/models/zzz.gguf"}):
            run("--status")
        assert "(remote; not in config/llm_models.yaml)" in capsys.readouterr().out

    def test_list_marks_the_models_of_the_loaded_file(self, serving, run, capsys):
        with serving({"-m": "/models/think.gguf"}):
            assert run("--list") == 0
        lines = {
            line.split()[0]: line.split()
            for line in capsys.readouterr().out.splitlines()[1:]
        }
        assert lines["think"] == ["think", "-", "running"]
        assert lines["vision"] == ["vision", "-", "running", "vision"]
        assert lines["other"] == ["other", "-"]

    def test_status_of_a_server_that_is_not_there_fails(
        self, run, monkeypatch, no_keepalive, capsys
    ):
        monkeypatch.setenv("LLM_URL", f"http://127.0.0.1:{free_port()}")
        assert run("--status") == 1
        assert "cannot reach the server" in capsys.readouterr().err
