"""Unit tests for src/prompt_runner/sdcpp_server.py, against a fake sd-server script."""

import json
import os
import stat
import sys
from pathlib import Path

import pytest

from prompt_runner.sdcpp_server import SdCppError, SdCppServer, build_argv

FAKE_SERVER = """\
#!{python}
import base64, http.server, json, os, sys

args = sys.argv
port = int(args[args.index("--listen-port") + 1])
mode = args[args.index("--mode") + 1] if "--mode" in args else "ok"
record = args[args.index("--record") + 1] if "--record" in args else None
if mode == "exit":
    sys.exit(3)
PNG = base64.b64encode(b"fakepng").decode()


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.endswith("/capabilities"):
            self.send(200, {{}})
        elif mode == "die":
            os._exit(9)
        elif mode == "fail":
            self.send(200, {{"status": "failed", "error": {{"message": "out of memory"}}}})
        else:
            self.send(200, {{"status": "completed", "result": {{"images": [{{"b64_json": PNG}}]}}}})

    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        if record:
            open(record, "wb").write(body)
        if mode == "reject":
            self.send(400, {{"error": "bad request"}})
        else:
            self.send(202, {{"id": "job_1"}})


http.server.HTTPServer(("127.0.0.1", port), Handler).serve_forever()
"""


@pytest.fixture
def fake_binary(tmp_path: Path) -> str:
    path = tmp_path / "fake-sd-server"
    path.write_text(FAKE_SERVER.format(python=sys.executable))
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return str(path)


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
                with pytest.raises(SdCppError) as info:
                    server.generate({"prompt": "x"})
                assert info.value.kind == "job_failed"
                assert info.value.message == "out of memory"

    def test_rejected_submit_raises_job_failed(self, fake_binary, tmp_path):
        with SdCppServer(
            fake_binary, {"--mode": "reject"}, tmp_path / "m.log"
        ) as server:
            with pytest.raises(SdCppError, match="HTTP 400") as info:
                server.generate({"prompt": "x"})
            assert info.value.kind == "job_failed"

    def test_server_dying_mid_job_raises_server_died_with_exit_code(
        self, fake_binary, tmp_path
    ):
        with (
            SdCppServer(fake_binary, {"--mode": "die"}, tmp_path / "m.log") as server,
            pytest.raises(SdCppError) as info,
        ):
            server.generate({"prompt": "x"})
        assert info.value.kind == "server_died"
        assert info.value.exit_code == 9

    def test_process_exiting_at_startup_raises_server_died(self, fake_binary, tmp_path):
        with (
            pytest.raises(SdCppError) as info,
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
