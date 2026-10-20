"""Shared fixtures: a fake sd-server script.

Options (passed like sd-server flags): --mode ok|fail|reject|die|exit|slow, --record FILE (the last POST body),
--delay SECONDS (before it starts listening). It answers /sdcpp/v1/capabilities and one job per request.
"""

import stat
import sys
from pathlib import Path

import pytest

FAKE_SERVER = """\
#!{python}
import base64, http.server, json, os, sys, time

args = sys.argv
port = int(args[args.index("--listen-port") + 1])
mode = args[args.index("--mode") + 1] if "--mode" in args else "ok"
record = args[args.index("--record") + 1] if "--record" in args else None
delay = float(args[args.index("--delay") + 1]) if "--delay" in args else 0
if mode == "exit":
    print("fatal: out of memory")
    sys.exit(3)
time.sleep(delay)
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
        elif mode == "slow":
            self.send(200, {{"status": "running"}})
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


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Mark every test `all`, so `pytest -m all` runs them all (a `-m` on the command line replaces the default `-m "not gpu"`)."""
    for item in items:
        item.add_marker("all")
