"""Managed stable-diffusion.cpp `sd-server` process and its native async API."""

import base64
import json
import os
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from types import TracebackType
from typing import Any, Literal

STARTUP_TIMEOUT = 120.0  # seconds until /capabilities answers (the model loads lazily)
REQUEST_TIMEOUT = 1800.0  # seconds per generation, including the lazy model load
POLL_INTERVAL = 0.5
STOP_TIMEOUT = 5.0  # seconds between SIGTERM and SIGKILL

ErrorKind = Literal["job_failed", "server_died"]


class SdCppError(Exception):
    """A failed generation. `job_failed`: the server is still alive. `server_died`: it is gone."""

    def __init__(
        self, kind: ErrorKind, message: str, exit_code: int | None = None
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.message = message
        self.exit_code = exit_code


def free_port() -> int:
    """Return a free TCP port on 127.0.0.1."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def build_argv(binary: str, args: dict[str, Any]) -> list[str]:
    """
    Turn a `server_args` mapping into a command line.

    Keys are literal sd-server flags. `True` gives the bare flag, `False` and
    `None` omit it, anything else gives `flag value`. Nothing is validated:
    sd-server rejects bad flags itself.
    """
    argv = [binary]
    for flag, value in args.items():
        if value is None or value is False:
            continue
        argv.append(flag)
        if value is not True:
            argv.append(str(value))
    return argv


def listen_argv(argv: list[str], port: int) -> list[str]:
    """Append the flags that make sd-server listen on 127.0.0.1:`port`."""
    return [*argv, "-l", "127.0.0.1", "--listen-port", str(port)]


def engine_version(binary: str) -> str:
    """Return the `sd-server --version` line, e.g. 'stable-diffusion.cpp version master-964-1d0a58a, commit 1d0a58a'."""
    result = subprocess.run(
        [binary, "--version"], capture_output=True, text=True, timeout=30, check=True
    )
    return (result.stdout or result.stderr).strip()


class SdCppServer:
    """
    One sd-server process serving one model, usable as a context manager.

    The server cannot switch models, so start one per model. By default it listens on a
    free port on 127.0.0.1 and runs in its own process group so that the whole
    tree is stopped on exit, whatever happens. With `keep=True` it is left running
    on exit (see keepalive.py), and `attach()` wraps a server that is already running.
    """

    def __init__(
        self,
        binary: str,
        args: dict[str, Any],
        log_path: Path,
        port: int | None = None,
        keep: bool = False,
    ) -> None:
        self.argv = build_argv(binary, args)
        self.log_path = log_path
        self._port = port
        self._keep = keep
        self._proc: subprocess.Popen[bytes] | None = None
        self._base_url = ""

    @classmethod
    def attach(cls, port: int, log_path: Path) -> "SdCppServer":
        """Wrap a server that is already running on `port`; this object never stops it."""
        server = cls("", {}, log_path, port=port, keep=True)
        server.argv = []
        server._base_url = f"http://127.0.0.1:{port}"
        return server

    @property
    def pid(self) -> int | None:
        """The process id, or None for an attached server."""
        return self._proc.pid if self._proc is not None else None

    def __enter__(self) -> "SdCppServer":  # noqa: PYI034 (typing.Self needs 3.11)
        # A server that is already started or attached is used as is.
        if not self._base_url:
            self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if not self._keep:
            self.stop()

    def start(self) -> None:
        """Start the process and wait until it answers; if that fails, stop it and re-raise."""
        try:
            self.spawn()
            self.wait_ready()
        except BaseException:
            self.stop()
            raise

    def spawn(self) -> None:
        """Start the process (own session, no stdin, output to the log) without waiting for it."""
        port = self._port or free_port()
        self._base_url = f"http://127.0.0.1:{port}"
        self.argv = listen_argv(self.argv, port)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "wb") as log:
            self._proc = subprocess.Popen(
                self.argv,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )

    def wait_ready(self) -> None:
        """Wait until /capabilities answers; raise `server_died` if the process exits or takes too long."""
        deadline = time.monotonic() + STARTUP_TIMEOUT
        while True:
            self._raise_if_dead()
            try:
                self._http("GET", "/sdcpp/v1/capabilities", timeout=5)
                return
            except OSError:
                pass
            if time.monotonic() > deadline:
                raise SdCppError(
                    "server_died", f"sd-server not ready after {STARTUP_TIMEOUT:.0f}s"
                )
            time.sleep(POLL_INTERVAL)

    def ready(self) -> bool:
        """Whether /capabilities answers right now."""
        try:
            self._http("GET", "/sdcpp/v1/capabilities", timeout=2)
        except (OSError, ValueError):
            return False
        return True

    def stop(self) -> None:
        """Stop the process we started (SIGTERM to its group, then SIGKILL); no-op if attached."""
        proc = self._proc
        if proc is not None and proc.poll() is None:
            self._signal_group(signal.SIGTERM)
            try:
                proc.wait(timeout=STOP_TIMEOUT)
            except subprocess.TimeoutExpired:
                self._signal_group(signal.SIGKILL)
                proc.wait()

    def _signal_group(self, sig: signal.Signals) -> None:
        if self._proc is None:
            return
        try:
            os.killpg(self._proc.pid, sig)
        except ProcessLookupError:
            pass

    def _raise_if_dead(self) -> None:
        if self._proc is not None and self._proc.poll() is not None:
            code = self._proc.returncode
            raise SdCppError("server_died", f"sd-server exited with code {code}", code)

    def _http(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
        timeout: float = 30,
    ) -> dict[str, Any]:
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(
            self._base_url + path,
            data=data,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result: dict[str, Any] = json.loads(resp.read())
            return result

    def generate(self, request: dict[str, Any]) -> list[bytes]:
        """
        Run one `img_gen` job and return the decoded images.

        Raises:
            SdCppError: `job_failed` if the server rejected or failed the job
                (it stays usable), `server_died` if the process is gone or the
                request timed out (the server is killed).
        """
        try:
            job = self._http("POST", "/sdcpp/v1/img_gen", request)
        except urllib.error.HTTPError as e:
            raise SdCppError("job_failed", _http_error_text(e)) from e
        except OSError as e:
            self._raise_if_dead()
            raise SdCppError("job_failed", f"submit failed: {e}") from e

        deadline = time.monotonic() + REQUEST_TIMEOUT
        while True:
            self._raise_if_dead()
            try:
                status = self._http("GET", f"/sdcpp/v1/jobs/{job['id']}")
            except OSError:
                status = {}
            state = status.get("status")
            if state == "completed":
                return [
                    base64.b64decode(img["b64_json"])
                    for img in status["result"]["images"]
                ]
            if state in ("failed", "cancelled"):
                error = status.get("error") or {}
                raise SdCppError("job_failed", error.get("message") or f"job {state}")
            if time.monotonic() > deadline:
                self.stop()
                raise SdCppError(
                    "server_died",
                    f"request timed out after {REQUEST_TIMEOUT:.0f}s (server stopped)",
                )
            time.sleep(POLL_INTERVAL)


def _http_error_text(e: urllib.error.HTTPError) -> str:
    body = e.read().decode(errors="replace")[:500]
    return f"HTTP {e.code}: {body}"
