"""Unit tests for src/prompt_runner/sdcpp_server.py, against a fake sd-server script."""

import json
import os
import socket

import pytest

from prompt_runner import sdcpp_server
from prompt_runner.sdcpp_server import SdCppServer
from prompt_runner.server_process import ServerError, build_argv, free_port


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


class TestBuildArgv:
    def test_true_false_none_and_scalars(self):
        args = {
            "--sage-attn": True,
            "--offload-to-cpu": False,
            "--vae": None,
            "--max-vram": 10,
            "--diffusion-model": "/m.gguf",
        }
        assert build_argv("sd", args) == [
            "sd",
            "--sage-attn",
            "--max-vram",
            "10",
            "--diffusion-model",
            "/m.gguf",
        ]


class TestSdCppServer:
    def test_generate_returns_decoded_images_and_stops_the_process(
        self, fake_binary, tmp_path
    ):
        record = tmp_path / "request.json"
        with SdCppServer(
            fake_binary, {"--record": str(record)}, tmp_path / "logs" / "m.log"
        ) as server:
            assert server._proc is not None
            pid = server._proc.pid
            images = server.generate({"prompt": "a cat", "seed": 7})
            assert "--listen-port" in server.argv
            assert server.argv[server.argv.index("-l") + 1] == "127.0.0.1"

        assert images == [b"fakepng"]
        assert json.loads(record.read_text()) == {"prompt": "a cat", "seed": 7}
        assert not alive(pid)
        assert (tmp_path / "logs" / "m.log").exists()

    def test_failed_job_raises_job_failed_and_server_survives(
        self, fake_binary, tmp_path
    ):
        with SdCppServer(fake_binary, {"--mode": "fail"}, tmp_path / "m.log") as server:
            for _ in range(2):
                with pytest.raises(ServerError) as info:
                    server.generate({"prompt": "x"})
                assert info.value.kind == "job_failed"
                assert info.value.message == "out of memory"

    def test_rejected_submit_raises_job_failed(self, fake_binary, tmp_path):
        with SdCppServer(
            fake_binary, {"--mode": "reject"}, tmp_path / "m.log"
        ) as server:
            with pytest.raises(ServerError, match="HTTP 400") as info:
                server.generate({"prompt": "x"})
            assert info.value.kind == "job_failed"

    def test_server_dying_mid_job_raises_server_died_with_exit_code(
        self, fake_binary, tmp_path
    ):
        with (
            SdCppServer(fake_binary, {"--mode": "die"}, tmp_path / "m.log") as server,
            pytest.raises(ServerError) as info,
        ):
            server.generate({"prompt": "x"})
        assert info.value.kind == "server_died"
        assert info.value.exit_code == 9

    def test_process_exiting_at_startup_raises_server_died(self, fake_binary, tmp_path):
        with (
            pytest.raises(ServerError) as info,
            SdCppServer(fake_binary, {"--mode": "exit"}, tmp_path / "m.log"),
        ):
            pass
        assert info.value.kind == "server_died"
        assert info.value.exit_code == 3

    def test_missing_binary_raises_oserror(self, tmp_path):
        with (
            pytest.raises(FileNotFoundError),
            SdCppServer(str(tmp_path / "nope"), {}, tmp_path / "m.log"),
        ):
            pass


class TestListenArgv:
    def test_appends_the_listen_flags(self):
        assert SdCppServer.listen_argv(["sd", "--vae", "v"], 8190) == [
            "sd",
            "--vae",
            "v",
            "-l",
            "127.0.0.1",
            "--listen-port",
            "8190",
        ]


class TestListenAddress:
    def test_the_listen_flag_takes_the_address(self):
        assert SdCppServer.listen_argv(["sd"], 8190, "192.168.56.1")[-4:] == [
            "-l",
            "192.168.56.1",
            "--listen-port",
            "8190",
        ]

    def test_a_server_listens_only_on_its_address(self, fake_binary, tmp_path):
        port = free_port()
        with SdCppServer(
            fake_binary, {}, tmp_path / "m.log", port=port, host="127.0.0.2"
        ) as server:
            assert server.ready()
            with pytest.raises(OSError):
                socket.create_connection(("127.0.0.1", port), timeout=1)
            assert SdCppServer.attach(port, tmp_path / "m.log", "127.0.0.2").ready()
            assert not SdCppServer.attach(port, tmp_path / "m.log").ready()


class TestKeptAndAttachedServers:
    def test_fixed_port_and_stdin_from_devnull(self, fake_binary, tmp_path):
        port = free_port()
        with SdCppServer(fake_binary, {}, tmp_path / "m.log", port=port) as server:
            assert server.argv[-2:] == ["--listen-port", str(port)]
            assert server.pid is not None
            assert os.readlink(f"/proc/{server.pid}/fd/0") == "/dev/null"
            assert server.ready()

    def test_keep_leaves_the_process_running_on_exit(self, fake_binary, tmp_path):
        with SdCppServer(fake_binary, {}, tmp_path / "m.log", keep=True) as server:
            pid = server.pid
        assert pid is not None and alive(pid)
        server.stop()
        assert not alive(pid)

    def test_attach_generates_and_never_stops_the_server(self, fake_binary, tmp_path):
        port = free_port()
        started = SdCppServer(fake_binary, {}, tmp_path / "m.log", port=port, keep=True)
        started.start()
        try:
            with SdCppServer.attach(port, tmp_path / "m.log") as attached:
                assert attached.pid is None
                assert attached.generate({"prompt": "x"}) == [b"fakepng"]
            assert started.pid is not None and alive(started.pid)
            attached.stop()  # a no-op: there is no process to stop
            assert started.ready()
        finally:
            started.stop()

    def test_ready_is_false_when_nothing_listens(self, tmp_path):
        assert not SdCppServer.attach(free_port(), tmp_path / "m.log").ready()

    def test_request_timeout_on_an_attached_server_kills_nothing(
        self, fake_binary, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(sdcpp_server, "REQUEST_TIMEOUT", 0.3)
        monkeypatch.setattr(sdcpp_server, "POLL_INTERVAL", 0.05)
        port = free_port()
        started = SdCppServer(
            fake_binary, {"--mode": "slow"}, tmp_path / "m.log", port=port, keep=True
        )
        started.start()
        try:
            with pytest.raises(ServerError) as info:
                SdCppServer.attach(port, tmp_path / "m.log").generate({"prompt": "x"})
            assert info.value.kind == "server_died"
            assert started.pid is not None and alive(started.pid)
        finally:
            started.stop()
