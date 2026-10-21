"""Shared fixtures: a fake sd-server script and a fake llama server.

Options (passed like sd-server flags): --mode ok|fail|reject|die|exit|slow, --record FILE (the last POST body),
--delay SECONDS (before it starts listening), --delay-file FILE (the delay is read from FILE and
FILE is set to 0, so only the first start is slow). It answers /sdcpp/v1/capabilities and one job per request.

A fake `llama serve` (FAKE_LLAMA) works the same way: `serve --mode think|thinkonly|nothink|badctx|error|die|stall|cut|exit
--record FILE --delay SECONDS --port N`. /health says 503 for the first --delay seconds, then 200; a chat request
streams its answer as server-sent events.
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
if "--delay-file" in args:
    path = args[args.index("--delay-file") + 1]
    delay = float(open(path).read())
    open(path, "w").write("0")
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


FAKE_LLAMA = r"""#!__PYTHON__
import http.server, json, os, sys, time

args = sys.argv
if args[1] == "--version":
    print("version: fake-1 (build 7)")
    print("built with a fake compiler")
    sys.exit(0)
assert args[1] == "serve", args  # the real executable is `llama serve ...`
port = int(args[args.index("--port") + 1])
mode = args[args.index("--mode") + 1] if "--mode" in args else "think"
record = args[args.index("--record") + 1] if "--record" in args else None
delay = float(args[args.index("--delay") + 1]) if "--delay" in args else 0
if mode == "exit":
    print("fatal: out of memory")
    sys.exit(3)
started = time.time()
TIMINGS = {"cache_n": 4, "prompt_n": 6, "predicted_n": 5, "prompt_ms": 12.0, "predicted_ms": 50.0}


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def send_json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/props":
            params = {"temperature": 0.8, "top_k": 40, "top_p": 0.95, "min_p": 0.05, "repeat_penalty": 1.0,
                      "presence_penalty": 0.0, "n_predict": -1}
            self.send_json(200, {"default_generation_settings": {"params": params}})
        elif self.path != "/health":
            self.send_json(404, {})
        elif time.time() - started < delay:
            self.send_json(503, {"error": {"code": 503, "message": "Loading model", "type": "unavailable_error"}})
        else:
            self.send_json(200, {"status": "ok"})

    def piece(self, data):
        self.wfile.write(b"%x\r\n" % len(data) + data + b"\r\n")
        self.wfile.flush()

    def event(self, obj):
        self.piece(b"data: " + json.dumps(obj).encode() + b"\n\n")

    def delta(self, **fields):
        return {"choices": [{"index": 0, "finish_reason": None, "delta": fields}]}

    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        if record:
            open(record, "wb").write(body)
        if mode == "badctx":
            return self.send_json(400, {"error": {"code": 400, "type": "exceed_context_size_error",
                                                  "message": "request (9000 tokens) exceeds the available context size (8192 tokens), try increasing it"}})
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        self.event(self.delta(role="assistant", content=None))
        self.piece(b":\n\n")  # the server's keep-alive ping is a comment line
        if mode in ("think", "thinkonly"):
            self.event(self.delta(reasoning_content="Let me "))
            self.event(self.delta(reasoning_content="think."))
        if mode != "thinkonly":
            self.event(self.delta(content="Hello "))
        if mode == "die":
            os._exit(9)
        if mode == "stall":
            time.sleep(30)
        if mode == "cut":  # the connection ends without the closing chunk, the server stays up
            self.close_connection = True
            return
        if mode == "error":
            self.event({"error": {"code": 500, "message": "decode failed", "type": "server_error"}})
        else:
            if mode != "thinkonly":
                self.event(self.delta(content="world"))
            reason = "length" if mode == "thinkonly" else "stop"
            self.event({"choices": [{"index": 0, "finish_reason": reason, "delta": {}}], "timings": TIMINGS})
        self.piece(b"data: [DONE]\n\n")
        self.wfile.write(b"0\r\n\r\n")


server = http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler)
server.daemon_threads = True
server.serve_forever()
"""


@pytest.fixture
def fake_llama(tmp_path: Path) -> str:
    path = tmp_path / "fake-llama"
    path.write_text(FAKE_LLAMA.replace("__PYTHON__", sys.executable))
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return str(path)


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
