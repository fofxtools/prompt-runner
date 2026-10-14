"""Unit tests for src/prompt_runner/image_runner.py"""

import base64
import json
from pathlib import Path
from typing import Any, ClassVar
from unittest.mock import patch

import pytest

from prompt_runner.image_runner import (
    prepare_request,
    run_image_eval,
    save_image_summary,
)
from prompt_runner.sdcpp_server import SdCppError


class FakeServer:
    """Stands in for SdCppServer; `outcomes` is consumed one per generate() call."""

    instances: ClassVar[list["FakeServer"]] = []
    outcomes: ClassVar[list[Any]] = []

    def __init__(self, binary: str, args: dict[str, Any], log_path: Path) -> None:
        self.binary = binary
        self.args = args
        self.argv = [binary, "--flag"]
        self.requests: list[dict[str, Any]] = []
        FakeServer.instances.append(self)

    def __enter__(self) -> "FakeServer":  # noqa: PYI034
        return self

    def __exit__(self, *exc: object) -> None:
        pass

    def generate(self, request: dict[str, Any]) -> list[bytes]:
        self.requests.append(request)
        outcome = FakeServer.outcomes.pop(0) if FakeServer.outcomes else [b"png"]
        if isinstance(outcome, SdCppError):
            raise outcome
        return list(outcome)


@pytest.fixture(autouse=True)
def fake_server():
    FakeServer.instances = []
    FakeServer.outcomes = []
    with (
        patch("prompt_runner.image_runner.SdCppServer", FakeServer),
        patch("prompt_runner.image_runner.engine_version", return_value="sd vTEST"),
    ):
        yield FakeServer


def make_config(tmp_path: Path, **extra: Any) -> dict[str, Any]:
    return {"results_dir": str(tmp_path), "sd_server": "/bin/sd-server", **extra}


def load_summary(tmp_path: Path) -> dict[str, Any]:
    (run_dir,) = tmp_path.iterdir()
    summary: dict[str, Any] = json.loads((run_dir / "summary.json").read_text())
    return summary


class TestSaveImageSummary:
    def test_creates_summary_file(self, tmp_path):
        run_dir_name = "2026-01-10_12-34-56Z-abc123"
        (tmp_path / run_dir_name).mkdir()
        failures = [{"model": "m", "prompt": "p", "kind": "job_failed"}]

        save_image_summary(
            "2026-01-10T12:34:56Z-abc123",
            run_dir_name,
            "2026-01-10T12:34:56Z",
            str(tmp_path),
            [{"id": "a", "mode": "txt2img"}, {"id": "b", "mode": "img2img"}],
            [{"name": "flux1-schnell"}, {"name": "sd15"}],
            {"flux1-schnell": 45.67, "sd15": 23.45},
            "sd vTEST",
            failures,
        )

        summary = json.loads((tmp_path / run_dir_name / "summary.json").read_text())
        assert summary["run_id"] == "2026-01-10T12:34:56Z-abc123"
        assert summary["image"] == {
            "engine": "sd vTEST",
            "prompt_count": 2,
            "model_count": 2,
            "prompts": ["a", "b"],
            "models": ["flux1-schnell", "sd15"],
            "model_timings": {"flux1-schnell": 45.67, "sd15": 23.45},
            "failures": failures,
        }

    def test_raises_if_directory_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            save_image_summary(
                "id", "nonexistent", "t", str(tmp_path), [], [], {}, "sd", []
            )


class TestPrepareRequest:
    def test_random_seed_is_resolved_and_recorded(self, tmp_path):
        request, recorded = prepare_request({"seed": -1, "prompt": "x"}, tmp_path)
        assert request["seed"] >= 0
        assert recorded["seed"] == request["seed"]

    def test_missing_seed_is_resolved(self, tmp_path):
        request, _ = prepare_request({"prompt": "x"}, tmp_path)
        assert isinstance(request["seed"], int) and request["seed"] >= 0

    def test_explicit_seed_is_kept(self, tmp_path):
        request, _ = prepare_request({"seed": 42}, tmp_path)
        assert request["seed"] == 42

    def test_images_are_copied_inlined_and_recorded_by_name(self, tmp_path):
        src = tmp_path / "cat.jpg"
        src.write_bytes(b"cat")
        other = tmp_path / "dog.jpg"
        other.write_bytes(b"dog")
        options = {
            "seed": 1,
            "init_image": str(src),
            "ref_images": [str(src), str(other)],
        }

        request, recorded = prepare_request(options, tmp_path / "inputs" / "p1")

        assert request["init_image"] == base64.b64encode(b"cat").decode()
        assert request["ref_images"] == [
            base64.b64encode(b"cat").decode(),
            base64.b64encode(b"dog").decode(),
        ]
        assert recorded["init_image"] == "inputs/p1/init_image_cat.jpg"
        assert recorded["ref_images"] == [
            "inputs/p1/ref_images_0_cat.jpg",
            "inputs/p1/ref_images_1_dog.jpg",
        ]
        assert (
            tmp_path / "inputs" / "p1" / "init_image_cat.jpg"
        ).read_bytes() == b"cat"
        assert options["init_image"] == str(src)  # input dict untouched

    def test_missing_input_file_raises_oserror(self, tmp_path):
        with pytest.raises(OSError):
            prepare_request(
                {"seed": 1, "init_image": str(tmp_path / "nope.jpg")}, tmp_path
            )


class TestRunImageEval:
    def test_writes_images_metadata_and_summary(self, tmp_path):
        FakeServer.outcomes = [[b"png0", b"png1"]]
        config = make_config(
            tmp_path,
            image_server_args={"--lora-model-dir": "/loras", "--max-vram": 8},
            image_generation_defaults={
                "width": 512,
                "sample_params": {"sample_steps": 4},
            },
        )
        prompts = [
            {"id": "p", "mode": "txt2img", "options": {"prompt": "x", "seed": 5}}
        ]
        models = [
            {
                "name": "org/m",
                "server_args": {"--max-vram": 10, "--sage-attn": True},
                "generation_options": {"sample_params": {"guidance": {"txt_cfg": 1.0}}},
            }
        ]

        run_id = run_image_eval(config, prompts, models)

        (run_dir,) = tmp_path.iterdir()
        assert (run_dir / "image" / "p" / "org_m_0.png").read_bytes() == b"png0"
        assert (run_dir / "image" / "p" / "org_m_1.png").read_bytes() == b"png1"
        meta = json.loads(
            (run_dir / "image" / "p" / "json" / "org_m_1.json").read_text()
        )
        assert meta["run_id"] == run_id
        assert meta["mode"] == "txt2img"
        assert meta["model"] == {"name": "org/m"}
        assert meta["prompt"] == {"id": "p"}
        assert meta["engine"] == "sd vTEST"
        assert meta["server_args"] == ["/bin/sd-server", "--flag"]
        assert meta["seed"] == 6  # image 1 of a batch uses seed + 1
        assert meta["request"]["seed"] == 5
        assert meta["request"] == {
            "width": 512,
            "sample_params": {"sample_steps": 4, "guidance": {"txt_cfg": 1.0}},
            "prompt": "x",
            "seed": 5,
        }
        assert FakeServer.instances[0].args == {
            "--lora-model-dir": "/loras",
            "--max-vram": 10,
            "--sage-attn": True,
        }
        summary = load_summary(tmp_path)
        assert summary["image"]["engine"] == "sd vTEST"
        assert summary["image"]["failures"] == []
        assert "org/m" in summary["image"]["model_timings"]

    def test_init_image_is_inlined_in_request_and_named_in_metadata(self, tmp_path):
        src = tmp_path / "cat.jpg"
        src.write_bytes(b"cat")
        results = tmp_path / "results"
        config = make_config(results)
        prompts = [
            {
                "id": "p",
                "mode": "img2img",
                "options": {"prompt": "x", "init_image": str(src)},
            }
        ]

        run_image_eval(config, prompts, [{"name": "m", "server_args": {}}])

        (run_dir,) = results.iterdir()
        request = FakeServer.instances[0].requests[0]
        assert request["init_image"] == base64.b64encode(b"cat").decode()
        meta = json.loads((run_dir / "image" / "p" / "json" / "m_0.json").read_text())
        assert meta["request"]["init_image"] == "inputs/p/init_image_cat.jpg"
        assert (run_dir / "inputs" / "p" / "init_image_cat.jpg").exists()

    def test_job_failure_is_recorded_and_the_run_continues(self, tmp_path):
        FakeServer.outcomes = [SdCppError("job_failed", "out of memory"), [b"png"]]
        prompts = [
            {"id": "a", "mode": "txt2img", "options": {"prompt": "x"}},
            {"id": "b", "mode": "txt2img", "options": {"prompt": "y"}},
        ]

        run_image_eval(
            make_config(tmp_path), prompts, [{"name": "m", "server_args": {}}]
        )

        (run_dir,) = tmp_path.iterdir()
        assert not (run_dir / "image" / "a").exists()
        assert (run_dir / "image" / "b" / "m_0.png").exists()
        (failure,) = load_summary(tmp_path)["image"]["failures"]
        assert failure == {
            "model": "m",
            "prompt": "a",
            "kind": "job_failed",
            "error": "out of memory",
            "exit_code": None,
            "log": "server_logs/m.log",
        }

    def test_server_death_stops_that_model_but_not_the_next(self, tmp_path):
        FakeServer.outcomes = [SdCppError("server_died", "gone", 9), [b"png"]]
        prompts = [
            {"id": "a", "mode": "txt2img", "options": {"prompt": "x"}},
            {"id": "b", "mode": "txt2img", "options": {"prompt": "y"}},
        ]
        models = [{"name": "m1", "server_args": {}}, {"name": "m2", "server_args": {}}]

        run_image_eval(make_config(tmp_path), prompts, models)

        (run_dir,) = tmp_path.iterdir()
        assert len(FakeServer.instances[0].requests) == 1  # m1 stopped after the death
        assert (run_dir / "image" / "a" / "m2_0.png").exists()
        assert (run_dir / "image" / "b" / "m2_0.png").exists()
        (failure,) = load_summary(tmp_path)["image"]["failures"]
        assert (failure["model"], failure["kind"], failure["exit_code"]) == (
            "m1",
            "server_died",
            9,
        )

    def test_server_start_failure_is_recorded_and_next_model_runs(self, tmp_path):
        class StartFails(FakeServer):
            def __enter__(self) -> "FakeServer":
                if self.args.get("--bad"):
                    raise SdCppError("server_died", "exited with code 1", 1)
                return self

        prompts = [{"id": "a", "mode": "txt2img", "options": {"prompt": "x"}}]
        models = [
            {"name": "bad", "server_args": {"--bad": True}},
            {"name": "good", "server_args": {}},
        ]
        with patch("prompt_runner.image_runner.SdCppServer", StartFails):
            run_image_eval(make_config(tmp_path), prompts, models)

        (run_dir,) = tmp_path.iterdir()
        assert (run_dir / "image" / "a" / "good_0.png").exists()
        (failure,) = load_summary(tmp_path)["image"]["failures"]
        assert (failure["model"], failure["prompt"]) == ("bad", None)

    def test_missing_input_file_is_recorded(self, tmp_path):
        prompts = [
            {
                "id": "a",
                "mode": "img2img",
                "options": {"prompt": "x", "init_image": str(tmp_path / "nope.jpg")},
            }
        ]

        run_image_eval(
            make_config(tmp_path / "r"), prompts, [{"name": "m", "server_args": {}}]
        )

        (failure,) = load_summary(tmp_path / "r")["image"]["failures"]
        assert failure["kind"] == "job_failed"
        assert "input error" in failure["error"]
        assert FakeServer.instances[0].requests == []

    def test_mode_filter(self, tmp_path):
        prompts = [
            {"id": "t", "mode": "txt2img", "options": {"prompt": "x"}},
            {"id": "i", "mode": "img2img", "options": {"prompt": "y"}},
        ]

        run_image_eval(
            make_config(tmp_path),
            prompts,
            [{"name": "m", "server_args": {}}],
            "txt2img",
        )

        (run_dir,) = tmp_path.iterdir()
        assert (run_dir / "image" / "t").exists()
        assert not (run_dir / "image" / "i").exists()

    def test_no_server_started_when_no_prompt_matches(self, tmp_path):
        prompts = [{"id": "t", "mode": "txt2img", "options": {"prompt": "x"}}]

        run_image_eval(
            make_config(tmp_path),
            prompts,
            [{"name": "m", "server_args": {}}],
            "img2img",
        )

        assert FakeServer.instances == []

    def test_invalid_mode_filter_raises_error(self, tmp_path):
        with pytest.raises(ValueError, match="Invalid mode_filter"):
            run_image_eval(make_config(tmp_path), [], [], mode_filter="bogus")

    def test_missing_sd_server_raises_error(self, tmp_path):
        with pytest.raises(ValueError, match="sd_server"):
            run_image_eval({"results_dir": str(tmp_path)}, [], [])
