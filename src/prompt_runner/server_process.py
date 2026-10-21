"""A managed server process (sd-server, llama.cpp's server): start, wait, stop, attach.

The API of each server lives in its own module (sdcpp_server.py); this one is what they share.
"""

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
from typing import Any, Literal, TypeVar

POLL_INTERVAL = 0.5
STOP_TIMEOUT = 5.0  # seconds between SIGTERM and SIGKILL

ErrorKind = Literal["job_failed", "server_died"]
S = TypeVar("S", bound="ServerProcess")


class ServerError(Exception):
    """A failed request or server. `job_failed`: the server is still alive. `server_died`: it is gone."""

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

    Keys are literal command-line flags. `True` gives the bare flag, `False` and
    `None` omit it, anything else gives `flag value`. Nothing is validated:
    the server rejects bad flags itself.
    """
    argv = [binary]
    for flag, value in args.items():
        if value is None or value is False:
            continue
        argv.append(flag)
        if value is not True:
            argv.append(str(value))
    return argv


def engine_version(binary: str) -> str:
    """Return the first line of `<binary> --version`, e.g. 'stable-diffusion.cpp version master-964-1d0a58a, commit 1d0a58a'."""
    result = subprocess.run(
        [binary, "--version"], capture_output=True, text=True, timeout=30, check=True
    )
    lines = (result.stdout or result.stderr).strip().splitlines()
    return lines[0].strip() if lines else ""


def http_error_text(e: urllib.error.HTTPError) -> str:
    """'HTTP <code>: <start of the response body>' for a rejected request."""
    body = e.read().decode(errors="replace")[:500]
    return f"HTTP {e.code}: {body}"


class ServerProcess:
    """
    One server process serving one model, usable as a context manager.

    A subclass sets `NAME` (for messages) and `READY_PATH` (a GET that answers 200 once the
    server is ready), and implements `listen_argv`. By default the server listens on a
    free port on 127.0.0.1 and runs in its own process group so that the whole tree is
    stopped on exit, whatever happens. With `keep=True` it is left running on exit (see
    keepalive.py), and `attach()` wraps a server that is already running.
    """

    NAME: str
    READY_PATH: str
    STARTUP_TIMEOUT = (
        120.0  # seconds until READY_PATH answers; a slower server raises it
    )

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
    def listen_argv(cls, argv: list[str], port: int) -> list[str]:
        """Append the flags that make the server listen on 127.0.0.1:`port`."""
        raise NotImplementedError

    @classmethod
    def attach(  # noqa: PYI019 (typing.Self needs 3.11)
        cls: type[S], port: int, log_path: Path
    ) -> S:
        """Wrap a server that is already running on `port`; this object never stops it."""
        server = cls("", {}, log_path, port=port, keep=True)
        server.argv = []
        server._base_url = f"http://127.0.0.1:{port}"
        return server

    @property
    def pid(self) -> int | None:
        """The process id, or None for an attached server."""
        return self._proc.pid if self._proc is not None else None

    def __enter__(self: S) -> S:  # noqa: PYI019 (typing.Self needs 3.11)
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
        self.argv = self.listen_argv(self.argv, port)
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
        """Wait until READY_PATH answers; raise `server_died` if the process exits or takes too long."""
        deadline = time.monotonic() + self.STARTUP_TIMEOUT
        while True:
            self._raise_if_dead()
            try:
                self._http("GET", self.READY_PATH, timeout=5)
                return
            except OSError:
                pass
            if time.monotonic() > deadline:
                raise ServerError(
                    "server_died",
                    f"{self.NAME} not ready after {self.STARTUP_TIMEOUT:.0f}s",
                )
            time.sleep(POLL_INTERVAL)

    def ready(self) -> bool:
        """Whether READY_PATH answers right now."""
        try:
            self._http("GET", self.READY_PATH, timeout=2)
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
            raise ServerError(
                "server_died", f"{self.NAME} exited with code {code}", code
            )

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
