"""Unit tests for src/prompt_runner/llama_server.py, against a fake `llama serve` script (see conftest.py)."""

import base64
import json
import os
import socket
from pathlib import Path
from typing import Any

import pytest

from prompt_runner import llama_server
from prompt_runner.llama_server import LlamaServer, image_part, with_image_parts
from prompt_runner.sdcpp_server import SdCppServer
from prompt_runner.server_process import ServerError, free_port

MESSAGES = [{"role": "user", "content": "Say hello."}]


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def fake(binary: str, tmp_path: Path, **flags: Any) -> LlamaServer:
    """A server over the fake script; `flags` become `--name value` (mode="die" -> --mode die)."""
    args = {f"--{name}": value for name, value in flags.items()}
    return LlamaServer(binary, args, tmp_path / "llama.log")


class TestListenAddress:
    def test_the_host_flag_takes_the_address(self):
        argv = LlamaServer.listen_argv(["llama"], 8191, "192.168.56.1")
        assert argv[-4:] == ["--host", "192.168.56.1", "--port", "8191"]

    def test_a_server_listens_only_on_its_address(self, fake_llama, tmp_path):
        port = free_port()
        with LlamaServer(
            fake_llama, {}, tmp_path / "llama.log", port=port, host="127.0.0.2"
        ) as server:
            assert server.ready()
            with pytest.raises(OSError):
                socket.create_connection(("127.0.0.1", port), timeout=1)
            assert LlamaServer.attach(port, tmp_path / "llama.log", "127.0.0.2").ready()


class TestStarting:
    def test_serve_comes_right_after_the_executable_and_the_flags_last(self):
        assert LlamaServer.listen_argv(
            ["llama", "-m", "a.gguf", "-c", "4096"], 8191
        ) == [
            "llama",
            "serve",
            "-m",
            "a.gguf",
            "-c",
            "4096",
            "--host",
            "127.0.0.1",
            "--port",
            "8191",
        ]

    def test_it_is_ready_when_health_stops_saying_503(self, fake_llama, tmp_path):
        server = fake(fake_llama, tmp_path, delay=0.6)
        try:
            server.spawn()
            assert not server.ready()  # 503 "Loading model"
            server.wait_ready()
            assert server.ready()
            assert server.argv[1] == "serve"
        finally:
            server.stop()

    def test_the_start_timeout_is_longer_than_sd_servers_and_can_be_changed(
        self, fake_llama, tmp_path, monkeypatch
    ):
        assert LlamaServer.STARTUP_TIMEOUT > SdCppServer.STARTUP_TIMEOUT
        monkeypatch.setattr(LlamaServer, "STARTUP_TIMEOUT", 0.3)
        with (
            pytest.raises(ServerError, match="llama-server not ready") as info,
            fake(fake_llama, tmp_path, delay=30),
        ):
            pass
        assert info.value.kind == "server_died"

    def test_a_server_that_exits_while_loading_is_reported(self, fake_llama, tmp_path):
        with (
            pytest.raises(ServerError) as info,
            fake(fake_llama, tmp_path, mode="exit"),
        ):
            pass
        assert info.value.kind == "server_died"
        assert info.value.exit_code == 3

    def test_leaving_the_block_stops_the_process(self, fake_llama, tmp_path):
        with fake(fake_llama, tmp_path) as server:
            pid = server.pid
            assert pid is not None and alive(pid)
        assert not alive(pid)


class TestImages:
    def test_an_image_becomes_a_data_uri_labelled_from_its_suffix(self, tmp_path):
        jpg = tmp_path / "a.jpg"
        jpg.write_bytes(b"jpeg bytes")
        part = image_part(jpg)
        assert part["type"] == "image_url"
        url = part["image_url"]["url"]
        assert (
            url == "data:image/jpeg;base64," + base64.b64encode(b"jpeg bytes").decode()
        )

    def test_an_unknown_suffix_is_labelled_png_because_the_server_needs_image_slash(
        self, tmp_path
    ):
        odd = tmp_path / "photo.data"
        odd.write_bytes(b"x")
        assert image_part(odd)["image_url"]["url"].startswith("data:image/png;base64,")

    def test_a_missing_file_is_an_oserror(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            image_part(tmp_path / "nope.png")

    def test_images_go_before_the_text_and_the_images_key_is_dropped(self, tmp_path):
        a, b = tmp_path / "a.png", tmp_path / "b.png"
        a.write_bytes(b"A")
        b.write_bytes(b"B")
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "Compare them.", "images": [str(a), str(b)]},
        ]

        sent = with_image_parts(messages)

        assert sent[0] is messages[0]  # untouched
        assert "images" in messages[1]  # the caller's message is not modified
        assert [p["type"] for p in sent[1]["content"]] == [
            "image_url",
            "image_url",
            "text",
        ]
        assert sent[1]["content"][2] == {"type": "text", "text": "Compare them."}
        assert "images" not in sent[1] and sent[1]["role"] == "user"
        assert sent[1]["content"][0]["image_url"]["url"].endswith(
            base64.b64encode(b"A").decode()
        )

    def test_content_that_is_already_parts_keeps_them_after_the_images(self, tmp_path):
        a = tmp_path / "a.png"
        a.write_bytes(b"A")
        text = {"type": "text", "text": "Hi"}
        (sent,) = with_image_parts(
            [{"role": "user", "content": [text], "images": [str(a)]}]
        )
        assert [p["type"] for p in sent["content"]] == ["image_url", "text"]

    def test_empty_images_leave_the_message_alone(self):
        message = {"role": "user", "content": "Hi", "images": []}
        assert with_image_parts([message]) == [message]

    def test_chat_sends_the_images_and_reads_them_before_any_request(
        self, fake_llama, tmp_path
    ):
        record = tmp_path / "body.json"
        pic = tmp_path / "pic.png"
        pic.write_bytes(b"PNGDATA")
        with fake(fake_llama, tmp_path, record=record) as server:
            server.chat(
                [{"role": "user", "content": "What is this?", "images": [str(pic)]}]
            )
            content = json.loads(record.read_text())["messages"][0]["content"]
            assert content[0]["image_url"]["url"].endswith(
                base64.b64encode(b"PNGDATA").decode()
            )
            assert content[1] == {"type": "text", "text": "What is this?"}

            record.unlink()
            with pytest.raises(FileNotFoundError):
                server.chat(
                    [
                        {
                            "role": "user",
                            "content": "x",
                            "images": [str(tmp_path / "gone.png")],
                        }
                    ]
                )
            assert not record.exists()  # nothing was sent


class TestSamplingDefaults:
    def test_it_reports_the_sampling_fields_the_server_would_use(
        self, fake_llama, tmp_path
    ):
        with fake(fake_llama, tmp_path) as server:
            assert server.sampling_defaults() == {
                "temperature": 0.8,
                "top_k": 40,
                "top_p": 0.95,
                "min_p": 0.05,
                "repeat_penalty": 1.0,
                "presence_penalty": 0.0,  # n_predict is not sampling
            }

    def test_a_server_that_does_not_answer_gives_nothing(self, tmp_path):
        assert (
            LlamaServer.attach(free_port(), tmp_path / "llama.log").sampling_defaults()
            == {}
        )


class TestChat:
    def test_reply_has_text_reasoning_finish_reason_and_timings(
        self, fake_llama, tmp_path
    ):
        with fake(fake_llama, tmp_path) as server:
            reply = server.chat(MESSAGES, {"temperature": 0.2})
        assert reply.text == "Hello world"
        assert reply.reasoning == "Let me think."
        assert reply.finish_reason == "stop"
        assert reply.timings["predicted_n"] == 5
        assert reply.timings["cache_n"] == 4

    def test_a_model_that_does_not_think_has_empty_reasoning(
        self, fake_llama, tmp_path
    ):
        with fake(fake_llama, tmp_path, mode="nothink") as server:
            reply = server.chat(MESSAGES)
        assert (reply.text, reply.reasoning) == ("Hello world", "")

    def test_on_delta_gets_each_piece_in_order(self, fake_llama, tmp_path):
        seen: list[tuple[str, str]] = []
        with fake(fake_llama, tmp_path) as server:
            server.chat(
                MESSAGES, on_delta=lambda kind, piece: seen.append((kind, piece))
            )
        assert seen == [
            ("reasoning", "Let me "),
            ("reasoning", "think."),
            ("text", "Hello "),
            ("text", "world"),
        ]

    def test_the_request_is_the_messages_and_options_always_streamed(
        self, fake_llama, tmp_path
    ):
        record = tmp_path / "body.json"
        options = {
            "max_tokens": -1,
            "seed": 7,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        with fake(fake_llama, tmp_path, record=record) as server:
            server.chat(MESSAGES, options)
        assert json.loads(record.read_text()) == {
            "max_tokens": -1,
            "seed": 7,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": MESSAGES,
            "stream": True,
        }
        assert options["stream"] is False  # the caller's dict is left alone

    def test_a_rejected_request_is_a_job_failure_and_the_server_stays(
        self, fake_llama, tmp_path
    ):
        with fake(fake_llama, tmp_path, mode="badctx") as server:
            with pytest.raises(
                ServerError, match="HTTP 400.*exceeds the available context"
            ) as info:
                server.chat(MESSAGES)
            assert info.value.kind == "job_failed"
            assert server.ready()

    def test_an_error_event_in_the_stream_is_a_job_failure(self, fake_llama, tmp_path):
        with fake(fake_llama, tmp_path, mode="error") as server:
            with pytest.raises(ServerError, match="decode failed") as info:
                server.chat(MESSAGES)
            assert info.value.kind == "job_failed"
            assert server.ready()

    def test_a_connection_cut_while_the_server_lives_is_a_job_failure(
        self, fake_llama, tmp_path
    ):
        with fake(fake_llama, tmp_path, mode="cut") as server:
            with pytest.raises(ServerError, match="lost the connection") as info:
                server.chat(MESSAGES)
            assert info.value.kind == "job_failed"
            assert server.ready()

    def test_a_server_that_dies_mid_reply_is_server_died_with_its_exit_code(
        self, fake_llama, tmp_path
    ):
        with (
            fake(fake_llama, tmp_path, mode="die") as server,
            pytest.raises(ServerError) as info,
        ):
            server.chat(MESSAGES)
        assert info.value.kind == "server_died"
        assert info.value.exit_code == 9

    def test_an_attached_server_that_dies_is_still_server_died(
        self, fake_llama, tmp_path
    ):
        port = free_port()
        started = LlamaServer(
            fake_llama, {"--mode": "die"}, tmp_path / "llama.log", port=port, keep=True
        )
        started.start()
        try:
            attached = LlamaServer.attach(port, tmp_path / "llama.log")
            with pytest.raises(ServerError, match="does not answer") as info:
                attached.chat(MESSAGES)
            assert info.value.kind == "server_died"
        finally:
            started.stop()

    def test_silence_for_too_long_stops_the_server(
        self, fake_llama, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(llama_server, "IDLE_TIMEOUT", 0.3)
        with fake(fake_llama, tmp_path, mode="stall") as server:
            pid = server.pid
            with pytest.raises(ServerError, match="no output") as info:
                server.chat(MESSAGES)
            assert pid is not None and not alive(
                pid
            )  # chat() stopped it, not the block
        assert info.value.kind == "server_died"

    def test_silence_on_an_attached_server_kills_nothing(
        self, fake_llama, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(llama_server, "IDLE_TIMEOUT", 0.3)
        port = free_port()
        started = LlamaServer(
            fake_llama,
            {"--mode": "stall"},
            tmp_path / "llama.log",
            port=port,
            keep=True,
        )
        started.start()
        try:
            with pytest.raises(ServerError) as info:
                LlamaServer.attach(port, tmp_path / "llama.log").chat(MESSAGES)
            assert info.value.kind == "server_died"
            assert started.pid is not None and alive(started.pid)
        finally:
            started.stop()
