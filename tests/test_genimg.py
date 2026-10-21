"""Unit tests for src/prompt_runner/genimg.py, against a fake sd-server."""

import base64
import json
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, ClassVar
from unittest.mock import patch

import pytest
import yaml

from prompt_runner import keepalive
from prompt_runner.genimg import main
from prompt_runner.image_runner import Job, run_jobs
from prompt_runner.keepalive import KeepAliveError
from prompt_runner.sdcpp_server import SdCppServer
from prompt_runner.server_process import ServerError

REPO = Path(__file__).resolve().parent.parent
BASE_LORA = {"path": "base.safetensors", "multiplier": 0.5}


class FakeServer:
    """Stands in for SdCppServer; `outcomes` is consumed one per generate() call."""

    instances: ClassVar[list["FakeServer"]] = []
    outcomes: ClassVar[list[Any]] = []

    def __init__(self, binary: str, args: dict[str, Any], log_path: Path) -> None:
        self.binary = binary
        self.args = args
        self.log_path = log_path
        self.requests: list[dict[str, Any]] = []
        FakeServer.instances.append(self)

    def __enter__(self) -> "FakeServer":  # noqa: PYI034
        return self

    def __exit__(self, *exc: object) -> None:
        pass

    def generate(self, request: dict[str, Any]) -> list[bytes]:
        self.requests.append(request)
        outcome = FakeServer.outcomes.pop(0) if FakeServer.outcomes else [b"png"]
        if isinstance(outcome, ServerError):
            raise outcome
        return list(outcome)


class Kept:
    """Stands in for prompt_runner.keepalive's server handling."""

    def __init__(self) -> None:
        self.started = True  # what ensure_server reports
        self.running: str | None = None  # the model of a running server
        self.ensured: list[dict[str, Any]] = []
        self.stopped = 0


@pytest.fixture(autouse=True)
def kept():
    state = Kept()

    def ensure_server(kept, binary, args, *, model_name):
        state.ensured.append({"port": kept.port, "model": model_name, "paths": kept})
        return FakeServer(binary, args, kept.log), state.started

    FakeServer.instances = []
    FakeServer.outcomes = []
    with (
        patch("prompt_runner.image_runner.SdCppServer", FakeServer),
        patch("prompt_runner.image_runner.engine_version", return_value="sd vTEST"),
        patch("prompt_runner.keepalive.ensure_server", ensure_server),
        patch("prompt_runner.keepalive.running_model", lambda kept: state.running),
        patch(
            "prompt_runner.keepalive.stop_server",
            lambda kept: setattr(state, "stopped", state.stopped + 1),
        ),
    ):
        yield state


def write_repo(root: Path, **config: Any) -> Path:
    """A minimal prompt-runner folder: config/config.yaml and config/image_models.yaml."""
    (root / "config").mkdir(parents=True)
    base = {
        "results_dir": "storage/results/",
        "sd_server": "/fake/sd-server",
        "image_server_args": {"--lora-model-dir": "/lora"},
        "image_generation_defaults": {"width": 1024, "height": 1024, "seed": -1},
        "default_image_model": "model-a",
    }
    base.update(config)
    models = [
        {
            "name": "model-a",
            "server_args": {"--diffusion-model": "/a.gguf"},
            "generation_options": {
                "sample_params": {"sample_steps": 4, "guidance": {"txt_cfg": 1.0}},
                "lora": [BASE_LORA],
            },
        },
        {"name": "model-b", "server_args": {"--diffusion-model": "/b.gguf"}},
    ]
    (root / "config/config.yaml").write_text(yaml.safe_dump(base))
    (root / "config/image_models.yaml").write_text(yaml.safe_dump(models))
    return root


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    return write_repo(tmp_path / "repo")


def run(
    repo: Path, tmp_path: Path, *args: str, name: str = "out.png"
) -> tuple[int, Path]:
    out = tmp_path / "images" / name
    code = main(
        ["a red apple", "-o", str(out), *args],
        repo_root=repo,
        state_dir=tmp_path / "state",
    )
    return code, out


def request_of(index: int = 0) -> dict[str, Any]:
    return FakeServer.instances[index].requests[0]


class TestSuccess:
    def test_default_model_size_and_success_line(self, repo, tmp_path, capsys):
        code, out = run(repo, tmp_path)
        text = capsys.readouterr().out
        assert code == 0
        assert out.read_bytes() == b"png"
        assert re.search(
            rf"saved {re.escape(str(out))} \(1024x1024, seed \d+, model model-a\)", text
        )
        assert [s.args["--diffusion-model"] for s in FakeServer.instances] == [
            "/a.gguf"
        ]
        assert FakeServer.instances[0].args["--lora-model-dir"] == "/lora"
        # No hard-coded settings: steps and CFG come from the model entry.
        assert request_of()["sample_params"]["sample_steps"] == 4

    def test_random_seed_is_resolved_and_printed(self, repo, tmp_path, capsys):
        run(repo, tmp_path)
        seed = request_of()["seed"]
        assert seed >= 0
        assert f"seed {seed}," in capsys.readouterr().out

    @pytest.mark.parametrize("seed", ["7", "0"])
    def test_explicit_seed_is_kept(self, repo, tmp_path, capsys, seed):
        run(repo, tmp_path, "--seed", seed)
        assert request_of()["seed"] == int(seed)
        assert f"seed {seed}," in capsys.readouterr().out

    def test_seed_minus_one_means_random(self, repo, tmp_path):
        run(repo, tmp_path, "--seed", "-1")
        assert request_of()["seed"] >= 0

    def test_model_flag_overrides_the_default(self, repo, tmp_path, capsys):
        code, _ = run(repo, tmp_path, "--model", "model-b")
        assert code == 0
        assert FakeServer.instances[0].args["--diffusion-model"] == "/b.gguf"
        assert "model model-b)" in capsys.readouterr().out

    def test_size_overrides_the_config(self, repo, tmp_path, capsys):
        run(repo, tmp_path, "--size", "640x384")
        assert (request_of()["width"], request_of()["height"]) == (640, 384)
        assert "(640x384, " in capsys.readouterr().out

    def test_reference_images_are_sent_as_base64(self, repo, tmp_path):
        ref = tmp_path / "in.png"
        ref.write_bytes(b"refbytes")
        run(repo, tmp_path, "--ref", str(ref))
        assert request_of()["ref_images"] == [base64.b64encode(b"refbytes").decode()]

    def test_reference_paths_are_logged_as_absolute(self, repo, tmp_path, monkeypatch):
        (tmp_path / "in.png").write_bytes(b"refbytes")
        monkeypatch.chdir(tmp_path)
        _, out = run(repo, tmp_path, "--ref", "in.png")
        row = json.loads((out.parent / ".genimg" / "runs.jsonl").read_text())
        assert row["request"]["ref_images"] == [str(tmp_path / "in.png")]

    def test_logs_go_next_to_the_output_in_a_hidden_folder(self, repo, tmp_path):
        _, out = run(repo, tmp_path)
        log = out.parent / ".genimg" / "runs.jsonl"
        rows = [json.loads(line) for line in log.read_text().splitlines()]
        assert [r["model"] for r in rows] == ["model-a"]
        assert rows[0]["status"] == "ok"
        assert rows[0]["seed"] == request_of()["seed"]
        assert rows[0]["engine"] == "sd vTEST"


class TestKeptServer:
    def test_the_port_model_and_files_go_to_ensure_server(
        self, repo, tmp_path, kept, monkeypatch
    ):
        monkeypatch.delenv("GENIMG_PORT", raising=False)
        run(repo, tmp_path)
        call = kept.ensured[0]
        assert call["port"] == 8190
        assert call["model"] == "model-a"
        assert call["paths"].state == tmp_path / "state" / "genimg-8190.json"

    def test_genimg_port_overrides_the_port(self, repo, tmp_path, kept, monkeypatch):
        monkeypatch.setenv("GENIMG_PORT", "9123")
        run(repo, tmp_path)
        assert kept.ensured[0]["port"] == 9123

    @pytest.mark.parametrize("port", ["abc", "0", "70000", ""])
    def test_a_bad_genimg_port_is_an_error(
        self, repo, tmp_path, capsys, monkeypatch, port
    ):
        monkeypatch.setenv("GENIMG_PORT", port)
        code, _ = run(repo, tmp_path)
        assert code == 1
        assert "GENIMG_PORT" in capsys.readouterr().err
        assert FakeServer.instances == []

    def test_without_model_the_running_servers_model_is_used(
        self, repo, tmp_path, kept
    ):
        kept.running = "model-b"
        run(repo, tmp_path)
        assert kept.ensured[0]["model"] == "model-b"
        assert FakeServer.instances[0].args["--diffusion-model"] == "/b.gguf"

    def test_model_flag_beats_the_running_servers_model(self, repo, tmp_path, kept):
        kept.running = "model-b"
        run(repo, tmp_path, "--model", "model-a")
        assert kept.ensured[0]["model"] == "model-a"

    @pytest.mark.parametrize(("started", "tag"), [(True, "started"), (False, "warm")])
    def test_rows_say_whether_the_server_was_warm(
        self, repo, tmp_path, kept, started, tag
    ):
        kept.started = started
        _, out = run(repo, tmp_path)
        row = json.loads((out.parent / ".genimg" / "runs.jsonl").read_text())
        assert row["tags"]["server"] == tag

    def test_a_refused_port_is_reported(self, repo, tmp_path, capsys):
        def refuse(*args, **kwargs):
            raise KeepAliveError("something is already listening on port 8190")

        with patch("prompt_runner.keepalive.ensure_server", refuse):
            code, out = run(repo, tmp_path)
        assert code == 1
        assert "already listening" in capsys.readouterr().err
        assert not out.exists()

    def test_a_success_keeps_the_server(self, repo, tmp_path, kept):
        code, _ = run(repo, tmp_path)
        assert code == 0
        assert kept.stopped == 0

    @pytest.mark.parametrize(
        ("started", "outcome", "stops"),
        [
            (False, ServerError("server_died", "gone"), 1),
            (True, ServerError("server_died", "gone"), 1),
            (True, ServerError("job_failed", "out of memory"), 1),
            (False, ServerError("job_failed", "out of memory"), 0),
        ],
    )
    def test_failure_policy(self, repo, tmp_path, kept, started, outcome, stops):
        kept.started = started
        FakeServer.outcomes = [outcome]
        code, _ = run(repo, tmp_path)
        assert code == 1
        assert kept.stopped == stops


class TestReviewFixes:
    def test_a_file_created_while_waiting_for_the_lock_is_not_overwritten(
        self, repo, tmp_path, capsys
    ):
        out = tmp_path / "images" / "out.png"
        kept = keepalive.kept_server("genimg", SdCppServer, 8190, tmp_path / "state")
        result: dict[str, int] = {}

        def second_call() -> None:
            result["code"] = main(
                ["x", "-o", str(out)], repo_root=repo, state_dir=tmp_path / "state"
            )

        with keepalive.call_lock(kept):  # the first call holds the lock
            thread = threading.Thread(target=second_call)
            thread.start()
            time.sleep(0.3)  # the second call has passed its first exists() check
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(b"first call")
        thread.join(timeout=10)

        assert result["code"] == 1
        assert "already exists" in capsys.readouterr().err
        assert out.read_bytes() == b"first call"
        assert FakeServer.instances == []

    def test_an_unsupported_system_is_refused_before_anything_starts(
        self, repo, tmp_path, capsys
    ):
        with patch("prompt_runner.keepalive.check_supported") as check:
            check.side_effect = KeepAliveError("genimg ... needs Linux")
            code, out = run(repo, tmp_path)
        assert code == 1
        assert "needs Linux" in capsys.readouterr().err
        assert not out.exists()

    def test_unknown_model_from_the_running_server_says_so(
        self, repo, tmp_path, capsys, kept
    ):
        kept.running = "renamed-model"
        code, _ = run(repo, tmp_path)
        err = capsys.readouterr().err
        assert code == 1
        assert "unknown model 'renamed-model' (the model of the running server)" in err

    def test_a_server_that_will_not_stop_is_reported_with_the_job_error(
        self, repo, tmp_path, capsys, kept
    ):
        def stuck(kept):
            raise KeepAliveError("the sd-server (pid 7) did not stop; kill it by hand")

        FakeServer.outcomes = [ServerError("server_died", "gone")]
        with patch("prompt_runner.keepalive.stop_server", stuck):
            code, _ = run(repo, tmp_path)
        err = capsys.readouterr().err
        assert code == 1
        assert "gone" in err and "did not stop" in err


class TestStart:
    def test_starts_the_default_model_with_the_runner_server_args(
        self, repo, tmp_path, capsys, kept
    ):
        code = main(["--start"], repo_root=repo, state_dir=tmp_path / "state")
        assert code == 0
        assert "sd-server ready with model-a (started)" in capsys.readouterr().out
        [call] = kept.ensured
        assert call["model"] == "model-a"
        assert call["port"] == 8190
        assert FakeServer.instances[0].args == {
            "--lora-model-dir": "/lora",
            "--diffusion-model": "/a.gguf",
        }
        # The model loads on the first request, so a tiny one is made to hold the GPU memory.
        [request] = FakeServer.instances[0].requests
        assert (request["width"], request["height"]) == (64, 64)
        assert request["sample_params"]["sample_steps"] == 4

    def test_model_flag_picks_the_model(self, repo, tmp_path, kept):
        main(["--start", "--model", "model-b"], repo_root=repo, state_dir=tmp_path)
        assert kept.ensured[0]["model"] == "model-b"

    def test_without_model_the_running_server_wins(self, repo, tmp_path, kept):
        kept.running = "model-b"
        main(["--start"], repo_root=repo, state_dir=tmp_path)
        assert kept.ensured[0]["model"] == "model-b"

    def test_already_running_is_reported(self, repo, tmp_path, capsys, kept):
        kept.started = False
        code = main(["--start"], repo_root=repo, state_dir=tmp_path)
        assert code == 0
        assert "(already running)" in capsys.readouterr().out

    def test_same_args_as_an_image_call_so_the_server_is_reused(
        self, repo, tmp_path, kept
    ):
        main(["--start"], repo_root=repo, state_dir=tmp_path / "state")
        run(repo, tmp_path)
        assert kept.ensured[0]["model"] == kept.ensured[1]["model"]
        assert FakeServer.instances[0].args == FakeServer.instances[1].args

    def test_a_failed_start_is_an_error_with_the_log_path(
        self, repo, tmp_path, capsys, monkeypatch
    ):
        def broken(*a, **k):
            raise ServerError("server_died", "sd-server exited with code 3")

        monkeypatch.setattr("prompt_runner.keepalive.ensure_server", broken)
        code = main(["--start"], repo_root=repo, state_dir=tmp_path)
        err = capsys.readouterr().err
        assert code == 1
        assert "exited with code 3" in err and "server log:" in err

    def test_a_failed_warm_up_on_a_new_server_stops_it(
        self, repo, tmp_path, capsys, kept
    ):
        FakeServer.outcomes = [ServerError("job_failed", "out of memory")]
        code = main(["--start"], repo_root=repo, state_dir=tmp_path)
        assert code == 1
        assert "out of memory" in capsys.readouterr().err
        assert kept.stopped == 1

    def test_a_failed_warm_up_on_a_warm_server_keeps_it(
        self, repo, tmp_path, capsys, kept
    ):
        kept.started = False
        FakeServer.outcomes = [ServerError("job_failed", "out of memory")]
        code = main(["--start"], repo_root=repo, state_dir=tmp_path)
        assert code == 1
        assert kept.stopped == 0

    def test_a_dead_server_is_stopped(self, repo, tmp_path, kept):
        kept.started = False
        FakeServer.outcomes = [ServerError("server_died", "gone")]
        main(["--start"], repo_root=repo, state_dir=tmp_path)
        assert kept.stopped == 1

    def test_unknown_model_is_an_error(self, repo, tmp_path, capsys):
        code = main(
            ["--start", "--model", "model-x"], repo_root=repo, state_dir=tmp_path
        )
        assert code == 1
        assert "unknown model 'model-x'" in capsys.readouterr().err


class TestStop:
    def test_stops_a_running_server(self, repo, tmp_path, capsys, kept):
        kept.running = "model-a"
        code = main(["--stop"], repo_root=repo, state_dir=tmp_path / "state")
        assert code == 0
        assert kept.stopped == 1
        assert "stopped the sd-server" in capsys.readouterr().out

    def test_nothing_running_is_not_an_error(self, repo, tmp_path, capsys, kept):
        code = main(["--stop"], repo_root=repo, state_dir=tmp_path / "state")
        assert code == 0
        assert "no sd-server of genimg is running" in capsys.readouterr().out

    def test_a_server_that_will_not_stop_fails(self, repo, tmp_path, capsys):
        def stuck(kept):
            raise KeepAliveError("the sd-server (pid 7) did not stop")

        with patch("prompt_runner.keepalive.stop_server", stuck):
            code = main(["--stop"], repo_root=repo, state_dir=tmp_path / "state")
        assert code == 1
        assert "did not stop" in capsys.readouterr().err

    @pytest.mark.parametrize(
        "args",
        [
            ["--stop", "a prompt"],
            ["--stop", "-o", "x.png"],
            ["--stop", "--model", "model-a"],
            ["--start", "a prompt"],
            ["--start", "--seed", "3"],
            ["--start", "--size", "64x64"],
            ["--start", "--lora", "x"],
            ["--start", "--stop"],
            ["a prompt"],
        ],
    )
    def test_bad_combinations_are_usage_errors(self, repo, tmp_path, args):
        with pytest.raises(SystemExit) as e:
            main(args, repo_root=repo, state_dir=tmp_path / "state")
        assert e.value.code == 2


class TestLora:
    def test_extra_loras_are_added_to_the_models_own(self, repo, tmp_path):
        run(repo, tmp_path, "--lora", "a.safetensors:0.8", "--lora", "b.safetensors")
        assert request_of()["lora"] == [
            BASE_LORA,
            {"path": "a.safetensors", "multiplier": 0.8},
            {"path": "b.safetensors", "multiplier": 1.0},
        ]

    def test_models_own_loras_apply_without_the_flag(self, repo, tmp_path):
        run(repo, tmp_path)
        assert request_of()["lora"] == [BASE_LORA]

    def test_model_without_loras_gets_only_the_extras(self, repo, tmp_path):
        run(repo, tmp_path, "--model", "model-b", "--lora", "x.safetensors:2")
        assert request_of()["lora"] == [{"path": "x.safetensors", "multiplier": 2.0}]

    def test_bad_weight_is_a_usage_error(self, repo, tmp_path):
        with pytest.raises(SystemExit) as e:
            run(repo, tmp_path, "--lora", "x.safetensors:heavy")
        assert e.value.code == 2


class TestErrors:
    def test_existing_output_is_never_overwritten(self, repo, tmp_path, capsys):
        out = tmp_path / "images" / "out.png"
        out.parent.mkdir()
        out.write_bytes(b"mine")
        code, _ = run(repo, tmp_path)
        assert code == 1
        assert "already exists" in capsys.readouterr().err
        assert out.read_bytes() == b"mine"
        assert FakeServer.instances == []

    def test_missing_reference_fails_before_a_server_starts(
        self, repo, tmp_path, capsys
    ):
        code, _ = run(repo, tmp_path, "--ref", str(tmp_path / "nope.png"))
        assert code == 1
        assert "reference image not found" in capsys.readouterr().err
        assert FakeServer.instances == []

    def test_unknown_model_suggests_close_names(self, repo, tmp_path, capsys):
        code, _ = run(repo, tmp_path, "--model", "modle-a")
        err = capsys.readouterr().err
        assert code == 1
        assert "unknown model 'modle-a'" in err
        assert "did you mean: model-a" in err
        assert FakeServer.instances == []

    def test_no_model_and_no_default_is_an_error(self, tmp_path, capsys):
        repo = write_repo(tmp_path / "repo2", default_image_model=None)
        code, _ = run(repo, tmp_path)
        assert code == 1
        assert "default_image_model" in capsys.readouterr().err

    def test_missing_config_is_an_error(self, tmp_path, capsys):
        code = main(["x", "-o", str(tmp_path / "o.png")], repo_root=tmp_path / "none")
        assert code == 1
        assert "cannot load the config" in capsys.readouterr().err

    def test_config_without_sd_server_is_an_error(self, tmp_path, capsys):
        repo = write_repo(tmp_path / "repo3", sd_server=None)
        code, _ = run(repo, tmp_path)
        assert code == 1
        assert "no sd_server" in capsys.readouterr().err

    def test_failed_generation_reports_the_error_and_server_log(
        self, repo, tmp_path, capsys
    ):
        FakeServer.outcomes = [ServerError("job_failed", "out of memory")]
        code, out = run(repo, tmp_path)
        err = capsys.readouterr().err
        assert code == 1
        assert "genimg failed: out of memory" in err
        assert f"server log: {tmp_path / 'state' / 'genimg-8190.log'}" in err
        assert not out.exists()

    def test_missing_binary_is_reported_not_a_traceback(self, repo, tmp_path, capsys):
        with patch(
            "prompt_runner.image_runner.engine_version",
            side_effect=FileNotFoundError("/fake/sd-server"),
        ):
            code, _ = run(repo, tmp_path)
        assert code == 1
        assert "genimg failed:" in capsys.readouterr().err

    def test_interrupt_exits_130(self, repo, tmp_path, capsys):
        with patch("prompt_runner.genimg.run_jobs", side_effect=KeyboardInterrupt):
            code, _ = run(repo, tmp_path)
        assert code == 130
        assert "interrupted" in capsys.readouterr().err

    @pytest.mark.parametrize("size", ["1024", "0x512", "axb", "1024x"])
    def test_bad_size_is_a_usage_error(self, repo, tmp_path, size):
        with pytest.raises(SystemExit) as e:
            run(repo, tmp_path, "--size", size)
        assert e.value.code == 2


class TestParityWithRunner:
    def test_request_equals_what_run_jobs_sends_for_the_same_options(
        self, repo, tmp_path
    ):
        ref = tmp_path / "in.png"
        ref.write_bytes(b"refbytes")
        run(repo, tmp_path, "--seed", "7", "--size", "512x768", "--ref", str(ref))

        config = yaml.safe_load((repo / "config/config.yaml").read_text())
        model = yaml.safe_load((repo / "config/image_models.yaml").read_text())[0]
        run_jobs(
            [
                Job(
                    model,
                    {
                        "prompt": "a red apple",
                        "seed": 7,
                        "width": 512,
                        "height": 768,
                        "ref_images": [str(ref)],
                    },
                    tmp_path / "direct.png",
                )
            ],
            binary=config["sd_server"],
            out_dir=tmp_path / "direct",
            server_defaults=config["image_server_args"],
            generation_defaults=config["image_generation_defaults"],
        )

        assert request_of(0) == request_of(1)
        assert FakeServer.instances[0].args == FakeServer.instances[1].args


class TestShim:
    def test_script_runs_and_documents_itself(self):
        result = subprocess.run(
            [sys.executable, str(REPO / "scripts" / "genimg"), "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0
        assert "default_image_model" in result.stdout
        assert "never overwritten" in result.stdout
