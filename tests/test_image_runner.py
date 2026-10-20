"""Unit tests for src/prompt_runner/image_runner.py"""

import base64
import json
from pathlib import Path
from typing import Any, ClassVar
from unittest.mock import patch

import pytest

from prompt_runner.image_runner import Job, prepare_request, run_jobs
from prompt_runner.sdcpp_server import SdCppError


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


def model(name: str, **server_args: Any) -> dict[str, Any]:
    return {"name": name, "server_args": {"--model": f"/{name}.gguf", **server_args}}


def job(tmp_path: Path, name: str, m: dict[str, Any], **options: Any) -> Job:
    return Job(m, {"prompt": name, "seed": 1, **options}, tmp_path / f"{name}.png")


def run(tmp_path: Path, jobs: list[Job], **kwargs: Any) -> list[dict[str, Any]]:
    return run_jobs(jobs, binary="/bin/sd-server", out_dir=tmp_path, **kwargs)


def read_log(tmp_path: Path) -> list[dict[str, Any]]:
    lines = (tmp_path / "runs.jsonl").read_text().splitlines()
    return [json.loads(line) for line in lines]


class TestPrepareRequest:
    def test_random_seed_is_resolved_and_recorded(self):
        request, recorded = prepare_request({"seed": -1, "prompt": "x"})
        assert request["seed"] >= 0
        assert recorded["seed"] == request["seed"]

    def test_missing_seed_is_resolved(self):
        request, _ = prepare_request({"prompt": "x"})
        assert isinstance(request["seed"], int) and request["seed"] >= 0

    def test_explicit_seed_is_kept(self):
        request, _ = prepare_request({"seed": 0})
        assert request["seed"] == 0

    def test_images_are_inlined_but_recorded_as_paths(self, tmp_path):
        cat, dog = tmp_path / "cat.jpg", tmp_path / "dog.jpg"
        cat.write_bytes(b"cat")
        dog.write_bytes(b"dog")
        options = {
            "seed": 1,
            "init_image": str(cat),
            "ref_images": [str(cat), str(dog)],
        }

        request, recorded = prepare_request(options)

        assert request["init_image"] == base64.b64encode(b"cat").decode()
        assert request["ref_images"] == [
            base64.b64encode(b"cat").decode(),
            base64.b64encode(b"dog").decode(),
        ]
        assert recorded["init_image"] == str(cat)
        assert recorded["ref_images"] == [str(cat), str(dog)]
        assert options["init_image"] == str(cat)  # input untouched

    def test_missing_input_file_raises_oserror(self, tmp_path):
        with pytest.raises(OSError):
            prepare_request({"seed": 1, "init_image": str(tmp_path / "nope.jpg")})


class TestRunJobsOutput:
    def test_writes_image_and_log_row(self, tmp_path):
        m = {**model("m"), "generation_options": {"sample_params": {"sample_steps": 4}}}
        j = Job(
            m,
            {"prompt": "a cat", "seed": 5},
            tmp_path / "sub" / "cat.png",
            {"prompt_id": "cat"},
        )

        rows = run(
            tmp_path,
            [j],
            server_defaults={"--sage-attn": True},
            generation_defaults={
                "width": 512,
                "sample_params": {"guidance": {"txt_cfg": 1.0}},
            },
        )

        assert (tmp_path / "sub" / "cat.png").read_bytes() == b"png"
        (row,) = rows
        assert read_log(tmp_path) == rows
        assert row["status"] == "ok"
        assert row["model"] == "m"
        assert row["image"] == "sub/cat.png"
        assert (row["batch_index"], row["batch_count"], row["seed"]) == (0, 1, 5)
        assert row["engine"] == "sd vTEST"
        assert row["tags"] == {"prompt_id": "cat"}
        assert row["index_in_server"] == 0
        assert row["server_args"] == {"--model": "/m.gguf", "--sage-attn": True}
        assert row["request"] == {
            "width": 512,
            "sample_params": {"guidance": {"txt_cfg": 1.0}, "sample_steps": 4},
            "prompt": "a cat",
            "seed": 5,
        }
        assert (row["kind"], row["error"], row["server_log"]) == (None, None, None)
        assert isinstance(row["elapsed_s"], float)

    def test_batch_writes_numbered_files_and_seeds(self, tmp_path):
        FakeServer.outcomes = [[b"a", b"b", b"c"]]

        rows = run(tmp_path, [job(tmp_path, "fox", model("m"), seed=100)])

        assert [
            (tmp_path / n).read_bytes() for n in ("fox.png", "fox_1.png", "fox_2.png")
        ] == [b"a", b"b", b"c"]
        assert [
            (r["image"], r["seed"], r["batch_index"], r["batch_count"]) for r in rows
        ] == [
            ("fox.png", 100, 0, 3),
            ("fox_1.png", 101, 1, 3),
            ("fox_2.png", 102, 2, 3),
        ]
        assert len({r["elapsed_s"] for r in rows}) == 1  # the time of the whole request

    def test_image_outside_out_dir_is_logged_as_given(self, tmp_path):
        elsewhere = tmp_path / "elsewhere" / "x.png"
        j = Job(model("m"), {"prompt": "x", "seed": 1}, elsewhere)

        (row,) = run_jobs([j], binary="/b", out_dir=tmp_path / "out")

        assert row["image"] == str(elsewhere)

    def test_input_image_is_inlined_and_logged_as_a_path(self, tmp_path):
        src = tmp_path / "cat.jpg"
        src.write_bytes(b"cat")

        (row,) = run(tmp_path, [job(tmp_path, "i", model("m"), init_image=str(src))])

        assert (
            FakeServer.instances[0].requests[0]["init_image"]
            == base64.b64encode(b"cat").decode()
        )
        assert row["request"]["init_image"] == str(src)

    def test_unresolved_seed_is_resolved_and_logged(self, tmp_path):
        (row,) = run(
            tmp_path, [Job(model("m"), {"prompt": "x", "seed": -1}, tmp_path / "x.png")]
        )

        assert row["seed"] >= 0
        assert row["request"]["seed"] == row["seed"]

    def test_write_log_false_writes_no_runs_jsonl(self, tmp_path):
        rows = run(tmp_path, [job(tmp_path, "a", model("m"))], write_log=False)

        assert len(rows) == 1
        assert not (tmp_path / "runs.jsonl").exists()

    def test_log_is_appended_across_runs(self, tmp_path):
        run(tmp_path, [job(tmp_path, "a", model("m"))])
        run(tmp_path, [job(tmp_path, "b", model("m"))])

        assert [r["image"] for r in read_log(tmp_path)] == ["a.png", "b.png"]

    def test_non_json_tag_values_are_stringified(self, tmp_path):
        j = Job(
            model("m"),
            {"prompt": "x", "seed": 1},
            tmp_path / "x.png",
            {"where": tmp_path},
        )

        run(tmp_path, [j])

        assert read_log(tmp_path)[0]["tags"] == {"where": str(tmp_path)}


class TestRunJobsGrouping:
    def test_one_server_per_distinct_model_even_if_interleaved(self, tmp_path):
        a, b = model("a"), model("b")
        jobs = [job(tmp_path, "1", a), job(tmp_path, "2", b), job(tmp_path, "3", a)]

        rows = run(tmp_path, jobs)

        assert [s.args["--model"] for s in FakeServer.instances] == [
            "/a.gguf",
            "/b.gguf",
        ]
        assert [len(s.requests) for s in FakeServer.instances] == [2, 1]
        assert [r["image"] for r in rows] == [
            "1.png",
            "3.png",
            "2.png",
        ]  # in group order
        assert [r["index_in_server"] for r in rows] == [0, 1, 0]

    def test_different_server_args_make_a_second_server(self, tmp_path):
        plain = model("m")
        variant = {
            **plain,
            "server_args": {**plain["server_args"], "--llm": "/other.gguf"},
        }

        run(tmp_path, [job(tmp_path, "1", plain), job(tmp_path, "2", variant)])

        assert len(FakeServer.instances) == 2

    def test_lora_variants_share_a_server(self, tmp_path):
        m = model("m")
        lora = [{"path": "x.safetensors", "multiplier": 1.0}]

        run(tmp_path, [job(tmp_path, "1", m), job(tmp_path, "2", m, lora=lora)])

        assert len(FakeServer.instances) == 1
        assert FakeServer.instances[0].requests[1]["lora"] == lora

    def test_server_args_merge_defaults_under_model(self, tmp_path):
        run(
            tmp_path,
            [job(tmp_path, "1", model("m", **{"--max-vram": 10}))],
            server_defaults={"--max-vram": 8, "--sage-attn": True},
        )

        assert FakeServer.instances[0].args == {
            "--model": "/m.gguf",
            "--max-vram": 10,
            "--sage-attn": True,
        }

    def test_generation_options_priority_defaults_model_job(self, tmp_path):
        m = {**model("m"), "generation_options": {"width": 768, "height": 768}}

        run(
            tmp_path,
            [job(tmp_path, "1", m, height=1024)],
            generation_defaults={"width": 512, "height": 512, "seed": 9},
        )

        request = FakeServer.instances[0].requests[0]
        assert (request["width"], request["height"], request["seed"]) == (768, 1024, 1)

    def test_server_log_goes_under_out_dir(self, tmp_path):
        run(tmp_path, [job(tmp_path, "1", model("org/m"))])

        log_path = FakeServer.instances[0].log_path
        assert log_path.parent == tmp_path / "server_logs"
        assert log_path.name.endswith("__0__org_m.log")

    def test_models_are_not_mutated(self, tmp_path):
        m = {**model("m"), "generation_options": {"width": 512}}
        before = json.dumps(m, sort_keys=True)

        run(tmp_path, [job(tmp_path, "1", m)], server_defaults={"--x": 1})

        assert json.dumps(m, sort_keys=True) == before


class TestOpenServer:
    def test_open_server_supplies_the_server_instead_of_a_new_one(self, tmp_path):
        server = FakeServer("sd", {}, tmp_path / "kept.log")
        FakeServer.instances.clear()
        calls: list[tuple[str, dict[str, Any], Path]] = []

        def open_server(binary: str, args: dict[str, Any], log_path: Path) -> Any:
            calls.append((binary, args, log_path))
            return server

        m = model("a", **{"--vae": "v"})
        run(tmp_path, [job(tmp_path, "one", m)], open_server=open_server)

        assert FakeServer.instances == []  # nothing else was started
        assert len(server.requests) == 1
        binary, args, log_path = calls[0]
        assert binary == "/bin/sd-server"
        assert args == m["server_args"]
        assert log_path.parent == tmp_path / "server_logs"

    def test_rows_record_the_supplied_servers_own_log(self, tmp_path):
        server = FakeServer("sd", {}, Path("/elsewhere/kept.log"))
        FakeServer.outcomes = [SdCppError("job_failed", "out of memory")]

        run(
            tmp_path,
            [job(tmp_path, "one", model("a"))],
            open_server=lambda binary, args, log_path: server,
        )

        assert read_log(tmp_path)[0]["server_log"] == "/elsewhere/kept.log"

    def test_the_default_still_starts_a_server_per_group(self, tmp_path):
        run(tmp_path, [job(tmp_path, "one", model("a"))])
        assert len(FakeServer.instances) == 1
        log = FakeServer.instances[0].log_path
        assert log.parent == tmp_path / "server_logs"


class TestRunJobsValidation:
    def test_duplicate_output_paths_raise(self, tmp_path):
        m = model("m")
        jobs = [
            Job(m, {"prompt": "a"}, tmp_path / "x.png"),
            Job(m, {"prompt": "b"}, tmp_path / "x.png"),
        ]

        with pytest.raises(ValueError, match="Duplicate output paths"):
            run(tmp_path, jobs)
        assert FakeServer.instances == []

    def test_missing_binary_raises(self, tmp_path):
        with pytest.raises(ValueError, match="binary"):
            run_jobs([], binary="", out_dir=tmp_path)

    def test_no_jobs_starts_nothing(self, tmp_path):
        assert run(tmp_path, []) == []
        assert FakeServer.instances == []
        assert list(tmp_path.iterdir()) == []


class TestRunJobsSkipExisting:
    def test_existing_outputs_are_skipped(self, tmp_path):
        (tmp_path / "a.png").write_bytes(b"old")
        m = model("m")

        rows = run(
            tmp_path, [job(tmp_path, "a", m), job(tmp_path, "b", m)], skip_existing=True
        )

        assert (tmp_path / "a.png").read_bytes() == b"old"
        assert [r["image"] for r in rows] == ["b.png"]  # no row for the skipped one

    def test_no_server_starts_when_everything_exists(self, tmp_path):
        (tmp_path / "a.png").write_bytes(b"old")

        rows = run(tmp_path, [job(tmp_path, "a", model("m"))], skip_existing=True)

        assert rows == []
        assert FakeServer.instances == []

    def test_existing_outputs_are_overwritten_by_default(self, tmp_path):
        (tmp_path / "a.png").write_bytes(b"old")

        run(tmp_path, [job(tmp_path, "a", model("m"))])

        assert (tmp_path / "a.png").read_bytes() == b"png"


class TestRunJobsFailures:
    def test_job_failure_is_recorded_and_the_run_continues(self, tmp_path):
        FakeServer.outcomes = [SdCppError("job_failed", "out of memory"), [b"png"]]
        m = model("m")

        rows = run(tmp_path, [job(tmp_path, "a", m), job(tmp_path, "b", m)])

        assert not (tmp_path / "a.png").exists()
        assert (tmp_path / "b.png").exists()
        failed, ok = rows
        assert (failed["status"], failed["kind"], failed["error"]) == (
            "failed",
            "job_failed",
            "out of memory",
        )
        assert failed["image"] is None and failed["seed"] == 1
        assert failed["server_log"].startswith("server_logs/")
        assert ok["status"] == "ok"
        assert read_log(tmp_path) == rows

    def test_server_death_skips_the_rest_of_the_group_but_not_the_next(self, tmp_path):
        FakeServer.outcomes = [SdCppError("server_died", "gone", 9), [b"png"]]
        a, b = model("a"), model("b")
        jobs = [job(tmp_path, "1", a), job(tmp_path, "2", a), job(tmp_path, "3", b)]

        rows = run(tmp_path, jobs)

        assert [r["status"] for r in rows] == ["failed", "skipped", "ok"]
        assert (rows[0]["kind"], rows[0]["exit_code"]) == ("server_died", 9)
        assert rows[1]["error"].startswith("not run")
        assert rows[1]["image"] is None
        assert len(FakeServer.instances[0].requests) == 1
        assert (tmp_path / "3.png").exists()

    def test_server_start_failure_is_recorded_and_the_next_group_runs(self, tmp_path):
        class StartFails(FakeServer):
            def __enter__(self) -> "FakeServer":
                if self.args.get("--bad"):
                    raise SdCppError("server_died", "exited with code 1", 1)
                return self

        jobs = [
            job(tmp_path, "1", model("bad", **{"--bad": True})),
            job(tmp_path, "2", model("bad", **{"--bad": True})),
            job(tmp_path, "3", model("good")),
        ]
        with patch("prompt_runner.image_runner.SdCppServer", StartFails):
            rows = run(tmp_path, jobs)

        assert [(r["model"], r["status"]) for r in rows] == [
            ("bad", "failed"),
            ("bad", "skipped"),
            ("good", "ok"),
        ]
        assert rows[0]["exit_code"] == 1

    def test_missing_input_file_is_recorded_without_calling_the_server(self, tmp_path):
        missing = str(tmp_path / "nope.jpg")

        (row,) = run(tmp_path, [job(tmp_path, "a", model("m"), init_image=missing)])

        assert (row["status"], row["kind"]) == ("failed", "input_error")
        assert "input error" in row["error"]
        assert FakeServer.instances[0].requests == []
