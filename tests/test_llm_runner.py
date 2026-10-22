"""Unit tests for src/prompt_runner/llm_runner.py"""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar
from unittest.mock import patch

import pytest

from prompt_runner.llama_server import ChatReply, DeltaKind
from prompt_runner.llm_runner import (
    LlmJob,
    prompt_messages,
    reply_metrics,
    request_options,
    run_llm_eval,
    run_llm_jobs,
)
from prompt_runner.server_process import ServerError

TIMINGS = {
    "cache_n": 4,
    "prompt_n": 6,
    "predicted_n": 5,
    "prompt_ms": 12.0,
    "predicted_ms": 50.0,
}


def reply(
    text: str = "Hello",
    reasoning: str = "",
    finish_reason: str | None = "stop",
    timings: dict[str, Any] | None = None,
) -> ChatReply:
    return ChatReply(
        text, reasoning, finish_reason, TIMINGS if timings is None else timings
    )


class FakeServer:
    """Stands in for LlamaServer; `outcomes` is consumed one per chat() call."""

    instances: ClassVar[list["FakeServer"]] = []
    outcomes: ClassVar[list[Any]] = []

    def __init__(self, binary: str, args: dict[str, Any], log_path: Path) -> None:
        self.binary = binary
        self.args = args
        self.log_path = log_path
        self.chats: list[tuple[list[dict[str, Any]], dict[str, Any]]] = []
        FakeServer.instances.append(self)

    def sampling_defaults(self) -> dict[str, Any]:
        return {"temperature": 0.8, "top_k": 40, "top_p": 0.95}

    def __enter__(self) -> "FakeServer":  # noqa: PYI034
        return self

    def __exit__(self, *exc: object) -> None:
        pass

    def chat(
        self,
        messages: list[dict[str, Any]],
        options: dict[str, Any] | None = None,
        on_delta: Callable[[DeltaKind, str], None] | None = None,
    ) -> ChatReply:
        self.chats.append((messages, options or {}))
        outcome = FakeServer.outcomes.pop(0) if FakeServer.outcomes else reply()
        if isinstance(outcome, (ServerError, OSError)):
            raise outcome
        if on_delta is not None:
            if outcome.reasoning:
                on_delta("reasoning", outcome.reasoning)
            if outcome.text:
                on_delta("text", outcome.text)
        return outcome


@pytest.fixture
def fake_server():
    FakeServer.instances = []
    FakeServer.outcomes = []
    with (
        patch("prompt_runner.llm_runner.LlamaServer", FakeServer),
        patch("prompt_runner.llm_runner.engine_version", return_value="llama vTEST"),
    ):
        yield FakeServer


def model(name: str, **server_args: Any) -> dict[str, Any]:
    return {"name": name, "server_args": {"-m": f"/{name}.gguf", **server_args}}


def job(tmp_path: Path, prompt_id: str, m: dict[str, Any], **options: Any) -> LlmJob:
    return LlmJob(
        m,
        [{"role": "user", "content": prompt_id}],
        tmp_path / prompt_id / f"{m['name']}.md",
        {"seed": 1, **options},
        {"prompt_id": prompt_id},
    )


def run(tmp_path: Path, jobs: list[LlmJob], **kwargs: Any) -> list[dict[str, Any]]:
    return run_llm_jobs(jobs, binary="/bin/llama", out_dir=tmp_path, **kwargs)


def read_log(tmp_path: Path) -> list[dict[str, Any]]:
    lines = (tmp_path / "runs.jsonl").read_text().splitlines()
    return [json.loads(line) for line in lines]


class TestReplyMetrics:
    def test_input_includes_the_cached_tokens(self):
        assert reply_metrics(TIMINGS) == {
            "input_tokens": 10,
            "cached_tokens": 4,
            "output_tokens": 5,
            "input_seconds": 0.012,
            "output_seconds": 0.05,
            "output_tokens_per_second": 100.0,
        }

    def test_no_timings_gives_nones(self):
        assert set(reply_metrics({}).values()) == {None}

    def test_no_cache_field_means_nothing_was_cached(self):
        metrics = reply_metrics({"prompt_n": 7, "predicted_n": 3, "predicted_ms": 0})
        assert metrics["input_tokens"] == 7
        assert metrics["output_tokens_per_second"] is None


class TestRequestOptions:
    """The sampling for non-thinking mode (`no_thinking_options`) applies whenever thinking is off."""

    MODEL: ClassVar[dict[str, Any]] = {
        "name": "m",
        "server_args": {},
        "generation_options": {"temperature": 1.0, "top_p": 0.95, "top_k": 20},
        "no_thinking_options": {"temperature": 0.7, "top_p": 0.8},
    }
    OFF: ClassVar[dict[str, Any]] = {"chat_template_kwargs": {"enable_thinking": False}}

    def test_with_thinking_on_they_are_not_used(self):
        assert request_options({"seed": 1}, self.MODEL, {}) == {
            "seed": 1,
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 20,
        }
        on = {"chat_template_kwargs": {"enable_thinking": True}}
        assert request_options(None, self.MODEL, on)["temperature"] == 1.0

    def test_a_request_that_switches_thinking_off_gets_them_over_the_models_own(self):
        options = request_options({"seed": 1}, self.MODEL, self.OFF)
        assert options == {
            "seed": 1,
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,  # the rest of the model's set stays
            **self.OFF,
        }

    def test_the_requests_own_fields_still_win(self):
        options = request_options(None, self.MODEL, {**self.OFF, "temperature": 0.2})
        assert (options["temperature"], options["top_p"]) == (0.2, 0.8)

    def test_a_model_or_the_defaults_that_switch_thinking_off_count_too(self):
        model = {
            **self.MODEL,
            "generation_options": {**self.MODEL["generation_options"], **self.OFF},
        }
        assert request_options(None, model, {})["temperature"] == 0.7
        assert request_options(self.OFF, self.MODEL, {})["temperature"] == 0.7

    def test_a_model_without_them_is_unchanged(self):
        model = {k: v for k, v in self.MODEL.items() if k != "no_thinking_options"}
        assert request_options(None, model, self.OFF)["temperature"] == 1.0

    def test_other_ways_of_saying_off_are_not_guessed(self):
        for kwargs in ({}, {"enable_thinking": None}, {"enable_thinking": 0}):
            assert (
                request_options(None, self.MODEL, {"chat_template_kwargs": kwargs})[
                    "temperature"
                ]
                == 1.0
            )

    def test_the_inputs_are_not_modified(self):
        before = json.dumps(self.MODEL, sort_keys=True)
        request_options({"seed": 1}, self.MODEL, dict(self.OFF))
        assert json.dumps(self.MODEL, sort_keys=True) == before


@pytest.mark.usefixtures("fake_server")
class TestNoThinkingInRuns:
    def model(self) -> dict[str, Any]:
        m = model("m")
        m["generation_options"] = {"temperature": 1.0, "top_p": 0.95}
        m["no_thinking_options"] = {"temperature": 0.7, "top_p": 0.8}
        return m

    def test_the_server_gets_and_the_row_records_the_non_thinking_sampling(
        self, tmp_path
    ):
        off = {"chat_template_kwargs": {"enable_thinking": False}}
        on_job = job(tmp_path, "on", self.model())
        off_job = job(tmp_path, "off", self.model(), **off)

        on_row, off_row = run(tmp_path, [on_job, off_job])

        sent = FakeServer.instances[0].chats
        assert (sent[0][1]["temperature"], sent[1][1]["temperature"]) == (1.0, 0.7)
        assert (
            off_row["sampling"]["temperature"] == 0.7
            and off_row["sampling"]["top_p"] == 0.8
        )
        assert on_row["sampling"]["temperature"] == 1.0
        assert off_row["request"]["temperature"] == 0.7

    def test_the_ollama_check_sees_the_final_options(self, tmp_path):
        m = self.model()
        m["no_thinking_options"] = {"num_predict": 5}
        off = {"chat_template_kwargs": {"enable_thinking": False}}
        with pytest.raises(ValueError, match="num_predict"):
            run(tmp_path, [job(tmp_path, "off", m, **off)])


class TestPromptMessages:
    def test_a_prompt_with_images_is_one_user_message_with_them(self):
        assert prompt_messages({"id": "x", "prompt": "Hi", "images": ["/a.png"]}) == [
            {"role": "user", "content": "Hi", "images": ["/a.png"]}
        ]

    def test_a_prompt_is_one_user_message(self):
        assert prompt_messages({"id": "x", "prompt": "Hi"}) == [
            {"role": "user", "content": "Hi"}
        ]

    def test_messages_are_kept(self):
        messages = [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "Hi"},
        ]
        assert prompt_messages({"id": "x", "messages": messages}) == messages


@pytest.mark.usefixtures("fake_server")
class TestRunLlmJobs:
    def test_a_row_records_what_ran_and_the_answer_goes_to_a_file(self, tmp_path):
        FakeServer.outcomes = [reply("Answer", "Thinking"), reply("Second")]
        m = model("m")

        rows = run(
            tmp_path,
            [job(tmp_path, "p1", m, temperature=0.5), job(tmp_path, "p2", m)],
        )

        first, second = rows
        assert (first["status"], first["model"], first["tags"]) == (
            "ok",
            "m",
            {"prompt_id": "p1"},
        )
        assert first["request"] == {
            "messages": [{"role": "user", "content": "p1"}],
            "temperature": 0.5,
            "seed": 1,
        }
        assert first["seed"] == 1
        assert first["output"] == {"text": "Answer", "reasoning": "Thinking"}
        assert first["finish_reason"] == "stop"
        assert first["metrics"] == reply_metrics(TIMINGS)
        assert first["server_args"] == m["server_args"]
        assert first["engine"] == "llama vTEST"
        assert first["output_file"] == "p1/m.md"
        assert first["server_log"].startswith("server_logs/")
        assert (first["index_in_server"], second["index_in_server"]) == (0, 1)
        assert first["server_start_s"] >= 0 and second["server_start_s"] is None
        assert (first["kind"], first["error"], first["exit_code"]) == (None, None, None)
        assert (tmp_path / "p1" / "m.md").read_text() == "Answer"  # no reasoning
        assert (tmp_path / "p2" / "m.md").read_text() == "Second"
        assert read_log(tmp_path) == rows

    def test_the_sampling_in_effect_is_recorded_whether_sent_or_the_servers_own(
        self, tmp_path
    ):
        (row,) = run(tmp_path, [job(tmp_path, "p", model("m"), top_k=20, min_p=0.0)])
        assert row["sampling"] == {
            "temperature": 0.8,  # not sent: the server's default
            "top_k": 20,  # sent
            "top_p": 0.95,
            "min_p": 0.0,  # sent, and the server did not report one
            "repeat_penalty": None,
            "presence_penalty": None,
        }

    def test_an_empty_answer_still_gets_a_file(self, tmp_path):
        FakeServer.outcomes = [reply("", "Thinking...", "length")]
        run(tmp_path, [job(tmp_path, "p", model("m"))])
        assert (tmp_path / "p" / "m.md").read_text() == ""

    @pytest.mark.parametrize("seed", [None, -1])
    def test_a_missing_or_random_seed_is_chosen_sent_and_recorded(self, tmp_path, seed):
        j = LlmJob(
            model("m"),
            [{"role": "user", "content": "x"}],
            tmp_path / "x.md",
            {} if seed is None else {"seed": seed},
        )

        (row,) = run(tmp_path, [j])

        sent = FakeServer.instances[0].chats[0][1]
        assert sent["seed"] >= 0
        assert row["seed"] == sent["seed"] == row["request"]["seed"]

    def test_request_fields_merge_defaults_then_model_then_job(self, tmp_path):
        m = model("m")
        m["generation_options"] = {
            "temperature": 0.5,
            "chat_template_kwargs": {"b": 2},
        }
        defaults = {
            "temperature": 0.2,
            "max_tokens": 100,
            "chat_template_kwargs": {"a": 1},
        }

        run(
            tmp_path,
            [job(tmp_path, "p", m, max_tokens=5)],
            generation_defaults=defaults,
        )

        assert FakeServer.instances[0].chats[0][1] == {
            "temperature": 0.5,
            "max_tokens": 5,
            "chat_template_kwargs": {"a": 1, "b": 2},
            "seed": 1,
        }

    def test_server_defaults_sit_under_the_models_own_flags(self, tmp_path):
        run(
            tmp_path,
            [job(tmp_path, "p", model("m", **{"-c": 8192}))],
            server_defaults={"-c": 4096, "-np": 1},
        )
        assert FakeServer.instances[0].args == {"-c": 8192, "-np": 1, "-m": "/m.gguf"}
        assert FakeServer.instances[0].binary == "/bin/llama"

    def test_a_server_per_distinct_setup_in_order_of_first_appearance(self, tmp_path):
        a, b = model("a"), model("b")
        jobs = [job(tmp_path, "1", a), job(tmp_path, "2", b), job(tmp_path, "3", a)]

        run(tmp_path, jobs)

        assert [s.args["-m"] for s in FakeServer.instances] == ["/a.gguf", "/b.gguf"]
        assert [len(s.chats) for s in FakeServer.instances] == [2, 1]

    def test_two_names_with_the_same_flags_share_one_server(self, tmp_path):
        x, y = model("x"), model("y")
        y["server_args"] = dict(x["server_args"])

        run(tmp_path, [job(tmp_path, "1", x), job(tmp_path, "2", y)])

        assert len(FakeServer.instances) == 1

    def test_a_reply_cut_at_max_tokens_is_flagged(self, tmp_path, capsys):
        FakeServer.outcomes = [reply("Half", finish_reason="length")]
        (row,) = run(tmp_path, [job(tmp_path, "p", model("m"))])
        assert row["status"] == "ok" and row["finish_reason"] == "length"
        assert "stopped at max_tokens" in capsys.readouterr().out

    def test_the_run_id_is_given_or_new(self, tmp_path):
        (row,) = run(tmp_path, [job(tmp_path, "p", model("m"))], run_id="my-run")
        assert row["run_id"] == "my-run"
        (row,) = run(tmp_path / "other", [job(tmp_path / "other", "p", model("m"))])
        assert row["run_id"]

    def test_the_log_can_be_switched_off(self, tmp_path):
        run(tmp_path, [job(tmp_path, "p", model("m"))], write_log=False)
        assert not (tmp_path / "runs.jsonl").exists()
        assert (tmp_path / "p" / "m.md").exists()


@pytest.mark.usefixtures("fake_server")
class TestImages:
    def image_job(self, tmp_path: Path, m: dict[str, Any], *paths: Path) -> LlmJob:
        messages = [
            {
                "role": "user",
                "content": "What is this?",
                "images": [str(p) for p in paths],
            }
        ]
        return LlmJob(
            m,
            messages,
            tmp_path / "out" / f"{m['name']}.md",
            {"seed": 1},
            {"prompt_id": "pic"},
        )

    def test_the_log_records_the_paths_and_the_server_gets_the_message_as_written(
        self, tmp_path
    ):
        pic = tmp_path / "pic.png"
        pic.write_bytes(b"png")
        m = model("vlm", **{"--mmproj": "/vlm-mmproj.gguf"})

        (row,) = run(tmp_path, [self.image_job(tmp_path, m, pic)])

        assert row["status"] == "ok"
        assert row["request"]["messages"][0]["images"] == [
            str(pic)
        ]  # a path, not base64
        assert FakeServer.instances[0].chats[0][0][0]["images"] == [str(pic)]

    def test_a_missing_image_is_refused_before_any_server_starts(self, tmp_path):
        m = model("vlm", **{"--mmproj": "/p.gguf"})
        with pytest.raises(ValueError, match="Image file.*not found.*gone.png"):
            run(tmp_path, [self.image_job(tmp_path, m, tmp_path / "gone.png")])
        assert FakeServer.instances == []

    def test_images_for_a_model_without_a_projector_are_refused(self, tmp_path):
        pic = tmp_path / "pic.png"
        pic.write_bytes(b"png")
        with pytest.raises(ValueError, match="no --mmproj"):
            run(tmp_path, [self.image_job(tmp_path, model("text-only"), pic)])
        assert FakeServer.instances == []

    def test_the_projector_may_come_from_the_shared_server_args(self, tmp_path):
        pic = tmp_path / "pic.png"
        pic.write_bytes(b"png")
        (row,) = run(
            tmp_path,
            [self.image_job(tmp_path, model("m"), pic)],
            server_defaults={"--mmproj": "/shared.gguf"},
        )
        assert row["status"] == "ok"

    def test_an_image_that_cannot_be_read_later_is_an_input_error_and_the_run_goes_on(
        self, tmp_path
    ):
        pic = tmp_path / "pic.png"
        pic.write_bytes(b"png")
        m = model("vlm", **{"--mmproj": "/p.gguf"})
        FakeServer.outcomes = [
            FileNotFoundError("[Errno 2] No such file: 'pic.png'"),
            reply("fine"),
        ]
        jobs = [self.image_job(tmp_path, m, pic), job(tmp_path, "plain", m)]
        jobs[1].out = tmp_path / "plain.md"

        failed, ok = run(tmp_path, jobs)

        assert (failed["status"], failed["kind"]) == ("failed", "input_error")
        assert "input error" in failed["error"]
        assert ok["status"] == "ok"


@pytest.mark.usefixtures("fake_server")
class TestFailures:
    def test_a_failed_job_is_recorded_and_the_run_continues(self, tmp_path):
        FakeServer.outcomes = [ServerError("job_failed", "context too small"), reply()]
        m = model("m")

        rows = run(tmp_path, [job(tmp_path, "a", m), job(tmp_path, "b", m)])

        failed, ok = rows
        assert (failed["status"], failed["kind"], failed["error"]) == (
            "failed",
            "job_failed",
            "context too small",
        )
        assert failed["output"] is None and failed["output_file"] is None
        assert failed["seed"] == 1 and failed["request"]["messages"]
        assert not (tmp_path / "a" / "m.md").exists()
        assert ok["status"] == "ok"
        assert read_log(tmp_path) == rows

    def test_server_death_skips_the_rest_of_the_group_but_not_the_next(self, tmp_path):
        FakeServer.outcomes = [ServerError("server_died", "gone", 9), reply()]
        a, b = model("a"), model("b")
        jobs = [job(tmp_path, "1", a), job(tmp_path, "2", a), job(tmp_path, "3", b)]

        rows = run(tmp_path, jobs)

        assert [r["status"] for r in rows] == ["failed", "skipped", "ok"]
        assert (rows[0]["kind"], rows[0]["exit_code"]) == ("server_died", 9)
        assert rows[1]["error"].startswith("not run")
        assert len(FakeServer.instances[0].chats) == 1

    def test_a_server_that_fails_to_start_is_recorded_and_the_next_group_runs(
        self, tmp_path
    ):
        class StartFails(FakeServer):
            def __enter__(self) -> "FakeServer":
                if self.args.get("--bad"):
                    raise ServerError("server_died", "exited with code 1", 1)
                return self

        jobs = [
            job(tmp_path, "1", model("bad", **{"--bad": True})),
            job(tmp_path, "2", model("bad", **{"--bad": True})),
            job(tmp_path, "3", model("good")),
        ]
        with patch("prompt_runner.llm_runner.LlamaServer", StartFails):
            rows = run(tmp_path, jobs)

        assert [(r["model"], r["status"]) for r in rows] == [
            ("bad", "failed"),
            ("bad", "skipped"),
            ("good", "ok"),
        ]
        assert rows[0]["exit_code"] == 1


@pytest.mark.usefixtures("fake_server")
class TestOnDelta:
    def test_pieces_arrive_in_order_and_the_row_is_still_complete(self, tmp_path):
        FakeServer.outcomes = [reply("Answer", "Thinking")]
        pieces: list[tuple[str, str]] = []

        (row,) = run(
            tmp_path,
            [job(tmp_path, "p", model("m"))],
            on_delta=lambda kind, piece: pieces.append((kind, piece)),
        )

        assert pieces == [("reasoning", "Thinking"), ("text", "Answer")]
        assert row["output"] == {"text": "Answer", "reasoning": "Thinking"}
        assert row["status"] == "ok"

    def test_it_sees_every_job_and_none_is_fine(self, tmp_path):
        FakeServer.outcomes = [reply("One"), reply("Two")]
        texts: list[str] = []
        jobs = [job(tmp_path, "a", model("m")), job(tmp_path, "b", model("m"))]

        run(tmp_path, jobs, on_delta=lambda kind, piece: texts.append(piece))
        assert texts == ["One", "Two"]

        FakeServer.outcomes = [reply("Three")]
        rows = run(tmp_path, [job(tmp_path, "c", model("m"))])  # no callback
        assert rows[0]["output"]["text"] == "Three"


@pytest.mark.usefixtures("fake_server")
class TestSay:
    def lines_of_a_run(self, tmp_path: Path, **kwargs: Any) -> list[str]:
        """Ok, failed, skipped (a server death) and a failed start, with `say` given."""

        class StartFails(FakeServer):
            def __enter__(self) -> "FakeServer":
                if self.args.get("--bad"):
                    raise ServerError("server_died", "exited with code 1", 1)
                return self

        FakeServer.outcomes = [
            reply(),
            ServerError("job_failed", "too long"),
            ServerError("server_died", "gone"),
        ]
        a, bad = model("a"), model("bad", **{"--bad": True})
        jobs = [
            job(tmp_path, "1", a),
            job(tmp_path, "2", a),
            job(tmp_path, "3", a),
            job(tmp_path, "4", a),
            job(tmp_path, "5", bad),
        ]
        said: list[str] = []
        with patch("prompt_runner.llm_runner.LlamaServer", StartFails):
            run(tmp_path, jobs, say=said.append, **kwargs)
        return said

    def test_every_progress_line_goes_through_say_and_none_to_stdout(
        self, tmp_path, capsys
    ):
        said = self.lines_of_a_run(tmp_path)

        assert capsys.readouterr().out == ""
        text = "\n".join(said)
        assert "Starting server: a (4 job(s))" in text
        assert "[1/5] 1/a.md: ok" in text
        assert "[2/5] 2/a.md: FAILED (job_failed): too long" in text
        assert "[3/5] 3/a.md: FAILED (server_died): gone" in text
        assert "[4/5] 4/a.md: skipped (server failed)" in text
        assert "Starting server: bad (1 job(s))" in text
        assert "[5/5] 5/bad.md: FAILED (server_died): exited with code 1" in text

    def test_without_say_the_lines_are_printed(self, tmp_path, capsys):
        run(tmp_path, [job(tmp_path, "p", model("m"))])
        out = capsys.readouterr().out
        assert "Starting server: m" in out and "[1/1] p/m.md: ok" in out


@pytest.mark.usefixtures("fake_server")
class TestJobsWithoutAFile:
    def bare(self, m: dict[str, Any], text: str = "hi", **tags: Any) -> LlmJob:
        return LlmJob(
            m, [{"role": "user", "content": text}], options={"seed": 1}, tags=tags
        )

    def test_no_file_is_written_and_the_answer_is_in_the_row(self, tmp_path):
        FakeServer.outcomes = [reply("Answer", "Thinking")]

        (row,) = run(tmp_path, [self.bare(model("m"))])

        assert row["status"] == "ok"
        assert row["output_file"] is None
        assert row["output"] == {"text": "Answer", "reasoning": "Thinking"}
        assert [p.name for p in tmp_path.iterdir()] == ["runs.jsonl"]  # no answer file
        assert read_log(tmp_path) == [row]

    def test_out_stays_none(self):
        assert self.bare(model("m")).out is None

    def test_any_number_of_them_can_share_a_run(self, tmp_path):
        jobs = [self.bare(model("m"), turn=1), self.bare(model("m"), turn=2)]
        rows = run(tmp_path, jobs)
        assert [r["status"] for r in rows] == ["ok", "ok"]

    def test_duplicates_are_still_found_among_the_jobs_that_have_a_file(self, tmp_path):
        m = model("m")
        jobs = [job(tmp_path, "x", m), self.bare(m), job(tmp_path, "x", m)]
        with pytest.raises(ValueError, match="Duplicate output paths"):
            run(tmp_path, jobs)

    def test_progress_names_the_model_instead_of_a_file(self, tmp_path):
        FakeServer.outcomes = [reply(), ServerError("job_failed", "no")]
        said: list[str] = []
        run(tmp_path, [self.bare(model("m")), self.bare(model("m"))], say=said.append)
        assert (
            "  [1/2] m: ok" in said[1]
            and "  [2/2] m: FAILED (job_failed): no" in said[2]
        )

    def test_a_failed_one_has_no_output_file_either(self, tmp_path):
        FakeServer.outcomes = [ServerError("job_failed", "no")]
        (row,) = run(tmp_path, [self.bare(model("m"))])
        assert (row["status"], row["output_file"]) == ("failed", None)

    def test_the_images_check_names_the_model_not_none(self, tmp_path):
        pic = tmp_path / "p.png"
        pic.write_bytes(b"x")
        bare = LlmJob(
            model("m"), [{"role": "user", "content": "x", "images": [str(pic)]}]
        )
        with pytest.raises(
            ValueError, match="Job m has images but model 'm' has no --mmproj"
        ):
            run(tmp_path, [bare])


@pytest.mark.usefixtures("fake_server")
class TestValidation:
    @pytest.mark.parametrize("where", ["defaults", "model", "job"])
    @pytest.mark.parametrize("name", ["num_predict", "num_ctx"])
    def test_ollama_options_are_refused_before_any_server_starts(
        self, tmp_path, where, name
    ):
        m = model("m")
        defaults = None
        extra: dict[str, Any] = {}
        if where == "defaults":
            defaults = {name: 5}
        elif where == "model":
            m["generation_options"] = {name: 5}
        else:
            extra = {name: 5}

        with pytest.raises(ValueError, match=f"'{name}'.*Ollama"):
            run(
                tmp_path,
                [job(tmp_path, "p", m, **extra)],
                generation_defaults=defaults,
            )
        assert FakeServer.instances == []

    def test_the_refusal_says_what_to_use_instead(self, tmp_path):
        with pytest.raises(ValueError, match="use max_tokens"):
            run(tmp_path, [job(tmp_path, "p", model("m"), num_predict=5)])

    def test_duplicate_output_paths_raise(self, tmp_path):
        m = model("m")
        jobs = [
            LlmJob(m, [{"role": "user", "content": "a"}], tmp_path / "x.md"),
            LlmJob(m, [{"role": "user", "content": "b"}], tmp_path / "x.md"),
        ]
        with pytest.raises(ValueError, match="Duplicate output paths"):
            run(tmp_path, jobs)
        assert FakeServer.instances == []

    def test_missing_binary_raises(self, tmp_path):
        with pytest.raises(ValueError, match="binary"):
            run_llm_jobs([], binary="", out_dir=tmp_path)

    def test_no_jobs_starts_nothing(self, tmp_path):
        assert run(tmp_path, []) == []
        assert FakeServer.instances == []
        assert list(tmp_path.iterdir()) == []


@pytest.mark.usefixtures("fake_server")
class TestOpenServer:
    def test_open_server_supplies_the_server_instead_of_a_new_one(self, tmp_path):
        server = FakeServer("llama", {}, tmp_path / "kept.log")
        FakeServer.instances.clear()
        calls: list[tuple[str, dict[str, Any], Path]] = []

        def open_server(binary: str, args: dict[str, Any], log_path: Path) -> Any:
            calls.append((binary, args, log_path))
            return server

        m = model("m", **{"--mmproj": "/p.gguf"})
        run(tmp_path, [job(tmp_path, "one", m)], open_server=open_server)

        assert FakeServer.instances == []
        assert len(server.chats) == 1
        binary, args, log_path = calls[0]
        assert (binary, args) == ("/bin/llama", m["server_args"])
        assert log_path.parent == tmp_path / "server_logs"

    def test_rows_record_the_supplied_servers_own_log(self, tmp_path):
        server = FakeServer("llama", {}, Path("/elsewhere/kept.log"))
        FakeServer.outcomes = [ServerError("job_failed", "no")]

        run(
            tmp_path,
            [job(tmp_path, "one", model("m"))],
            open_server=lambda binary, args, log_path: server,
        )

        assert read_log(tmp_path)[0]["server_log"] == "/elsewhere/kept.log"

    def test_a_remote_server_needs_no_binary_and_no_log_and_gives_its_own_engine(
        self, tmp_path
    ):
        server: Any = FakeServer("", {}, None)  # type: ignore[arg-type]  # as attach() without a log
        FakeServer.instances.clear()

        rows = run_llm_jobs(
            [job(tmp_path, "one", model("m"))],
            out_dir=tmp_path,
            open_server=lambda binary, args, log_path: server,
            engine="b1 (remote)",
        )

        assert rows[0]["status"] == "ok"
        assert (rows[0]["engine"], rows[0]["server_log"]) == ("b1 (remote)", None)
        assert FakeServer.instances == []

    def test_the_starting_server_line_is_only_for_a_server_the_runner_starts(
        self, tmp_path
    ):
        said: list[str] = []
        run(tmp_path, [job(tmp_path, "one", model("m"))], say=said.append)
        assert any(line.startswith("Starting server: m") for line in said)

        said.clear()
        server: Any = FakeServer("llama", {}, tmp_path / "kept.log")
        run(
            tmp_path,
            [job(tmp_path, "two", model("m"))],
            open_server=lambda binary, args, log_path: server,
            say=said.append,
        )
        assert len(said) == 1 and "[1/1] two" in said[0]  # only the progress line

    def test_no_binary_and_no_open_server_is_an_error(self, tmp_path):
        with pytest.raises(ValueError, match="binary"):
            run_llm_jobs([job(tmp_path, "one", model("m"))], out_dir=tmp_path)


@pytest.mark.usefixtures("fake_server")
class TestRunLlmEval:
    CONFIG: ClassVar[dict[str, Any]] = {
        "llama_server": "/bin/llama",
        "llm_server_args": {"-c": 4096},
        "llm_generation_defaults": {"temperature": 0.2, "seed": 3},
    }

    def config(self, tmp_path: Path, **more: Any) -> dict[str, Any]:
        return {**self.CONFIG, "results_dir": str(tmp_path), **more}

    def test_every_prompt_runs_on_every_model_and_the_results_have_a_folder(
        self, tmp_path
    ):
        prompts: list[dict[str, Any]] = [
            {"id": "story", "prompt": "Tell a story."},
            {
                "id": "tutor",
                "messages": [
                    {"role": "system", "content": "Be brief."},
                    {"role": "user", "content": "Why is the sky blue?"},
                ],
                "options": {"temperature": 0.9},
            },
        ]
        models = [model("a/b:c"), model("d")]

        run_id = run_llm_eval(self.config(tmp_path), prompts, models)

        (run_dir,) = tmp_path.iterdir()
        llm_dir = run_dir / "llm"
        # (the fake server writes no log file; a real one fills server_logs/)
        assert sorted(p.name for p in llm_dir.iterdir()) == [
            "runs.jsonl",
            "story",
            "tutor",
        ]
        assert (llm_dir / "story" / "a_b_c.md").read_text() == "Hello"
        assert (llm_dir / "tutor" / "d.md").exists()
        rows = read_log(llm_dir)
        assert {r["run_id"] for r in rows} == {run_id}
        assert [(r["model"], r["tags"]["prompt_id"]) for r in rows] == [
            ("a/b:c", "story"),
            ("a/b:c", "tutor"),
            ("d", "story"),
            ("d", "tutor"),
        ]
        story, tutor = rows[0]["request"], rows[1]["request"]
        assert story["messages"] == [{"role": "user", "content": "Tell a story."}]
        assert tutor["messages"][0]["role"] == "system"
        assert (story["temperature"], tutor["temperature"]) == (0.2, 0.9)  # prompt wins
        assert story["seed"] == 3
        assert [s.args["-c"] for s in FakeServer.instances] == [4096, 4096]

    def test_run_level_options_win_over_the_prompts_and_the_models(self, tmp_path):
        m = model("a")
        m["generation_options"] = {"temperature": 0.3, "top_k": 20}
        prompts = [
            {"id": "p", "prompt": "x", "options": {"temperature": 0.9, "top_p": 0.5}}
        ]

        run_llm_eval(self.config(tmp_path), prompts, [m], options={"temperature": 0.7})

        sent = FakeServer.instances[0].chats[0][1]
        assert (
            sent["temperature"] == 0.7
        )  # run level beats prompt (0.9), model (0.3), config
        assert (sent["top_p"], sent["top_k"]) == (0.5, 20)  # the rest still merges

    def test_a_config_without_the_llama_executable_is_refused_and_leaves_nothing(
        self, tmp_path
    ):
        config = self.config(tmp_path)
        del config["llama_server"]
        with pytest.raises(ValueError, match="llama_server"):
            run_llm_eval(config, [{"id": "p", "prompt": "x"}], [model("m")])
        assert list(tmp_path.iterdir()) == []


class TestWithAFakeLlamaServer:
    """The runner over the real LlamaServer class and a fake `llama` script (see conftest.py)."""

    def test_an_image_prompt_reaches_the_server_as_a_data_uri_and_the_log_keeps_the_path(
        self, fake_llama, tmp_path
    ):
        pic = tmp_path / "pic.jpg"
        pic.write_bytes(b"JPEGDATA")
        record = tmp_path / "body.json"
        m = {
            "name": "fake",
            "server_args": {
                "--mode": "nothink",
                "--mmproj": "x",
                "--record": str(record),
            },
        }
        prompts = [{"id": "pic", "prompt": "Describe it.", "images": [str(pic)]}]
        config = {"llama_server": fake_llama, "results_dir": str(tmp_path / "results")}

        run_llm_eval(config, prompts, [m])

        sent = json.loads(record.read_text())["messages"][0]["content"]
        assert sent[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
        assert sent[1] == {"type": "text", "text": "Describe it."}
        (runs,) = (tmp_path / "results").glob("*/llm/runs.jsonl")
        row = json.loads(runs.read_text())
        assert row["request"]["messages"][0]["images"] == [str(pic)]
        assert "base64" not in runs.read_text()

    def test_chats_stream_through_and_the_rows_are_complete(self, fake_llama, tmp_path):
        m = {"name": "fake", "server_args": {"--mode": "think"}}
        jobs = [job(tmp_path, "one", m), job(tmp_path, "two", m)]

        rows = run_llm_jobs(
            jobs,
            binary=fake_llama,
            out_dir=tmp_path,
            server_defaults={"-np": 1},
            generation_defaults={"max_tokens": 50},
        )

        assert [r["status"] for r in rows] == ["ok", "ok"]
        assert rows[0]["output"] == {
            "text": "Hello world",
            "reasoning": "Let me think.",
        }
        assert rows[0]["finish_reason"] == "stop"
        assert rows[0]["metrics"]["input_tokens"] == 10
        assert rows[0]["engine"] == "version: fake-1 (build 7)"  # the first line only
        assert (
            rows[0]["sampling"]["temperature"] == 0.8
        )  # the fake server's own default
        assert rows[0]["sampling"]["top_k"] == 40
        assert (tmp_path / "one" / "fake.md").read_text() == "Hello world"
        assert (tmp_path / rows[0]["server_log"]).is_file()  # next to runs.jsonl
        assert rows[0]["request"]["max_tokens"] == 50

    def test_on_delta_gets_the_real_servers_pieces_in_order(self, fake_llama, tmp_path):
        m = {"name": "fake", "server_args": {"--mode": "think"}}
        pieces: list[tuple[str, str]] = []

        (row,) = run_llm_jobs(
            [LlmJob(m, [{"role": "user", "content": "hi"}])],
            binary=fake_llama,
            out_dir=tmp_path,
            server_defaults={"-np": 1},
            generation_defaults={"max_tokens": 50},
            on_delta=lambda kind, piece: pieces.append((kind, piece)),
            say=lambda line: None,
        )

        assert row["status"] == "ok" and row["output_file"] is None
        assert (
            "".join(p for k, p in pieces if k == "reasoning")
            == row["output"]["reasoning"]
        )
        assert (
            "".join(p for k, p in pieces if k == "text")
            == row["output"]["text"]
            == "Hello world"
        )
        kinds = [k for k, _ in pieces]
        assert kinds.index("text") > max(
            i for i, k in enumerate(kinds) if k == "reasoning"
        )

    def test_a_server_that_dies_while_loading_is_a_failed_row(
        self, fake_llama, tmp_path
    ):
        m = {"name": "broken", "server_args": {"--mode": "exit"}}

        rows = run_llm_jobs(
            [job(tmp_path, "one", m), job(tmp_path, "two", m)],
            binary=fake_llama,
            out_dir=tmp_path,
        )

        assert [r["status"] for r in rows] == ["failed", "skipped"]
        assert (rows[0]["kind"], rows[0]["exit_code"]) == ("server_died", 3)
