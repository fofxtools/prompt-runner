"""Keep one sd-server running between genimg calls.

A call starts the model's server if none is ours, leaves it running, and the next call reuses it. State is a JSON file
(`pid`, the exact command line, `model_name`) next to a lock and a log, all keyed by port. A pid counts as ours only when
`/proc/<pid>/cmdline` ends with the recorded command line, so a reused pid (or a zombie) is never signalled. The same command
line decides reuse (equal to the one wanted) or replacement (different). Something else listening on the port is refused.

Linux only: it relies on /proc and fcntl.
"""

import fcntl
import json
import os
import signal
import socket
import sys
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .sdcpp_server import (
    POLL_INTERVAL,
    STARTUP_TIMEOUT,
    STOP_TIMEOUT,
    SdCppError,
    SdCppServer,
    build_argv,
    listen_argv,
)

LOCK_WAIT = 900.0  # seconds a call waits for another genimg call before giving up
LOG_TAIL_LINES = 15


class KeepAliveError(Exception):
    """The kept server cannot be used (something else owns the port, the lock never came free, or it will not stop)."""


def check_supported(proc: Path = Path("/proc")) -> None:
    """Raise KeepAliveError unless this system has /proc (Linux), which the process check needs."""
    if not (proc / "self" / "cmdline").exists():
        raise KeepAliveError(
            "genimg keeps its server between calls using /proc and fcntl, so it needs Linux"
        )


@dataclass(frozen=True)
class Paths:
    """The files of one kept server."""

    lock: Path
    state: Path
    log: Path


def paths_for(port: int, directory: Path | None = None) -> Paths:
    """The lock, state and log files for `port`, in `directory` (default: the temp folder)."""
    base = directory if directory is not None else Path(tempfile.gettempdir())
    base.mkdir(parents=True, exist_ok=True)
    stem = f"genimg-sd-{port}"
    return Paths(base / f"{stem}.lock", base / f"{stem}.json", base / f"{stem}.log")


@contextmanager
def call_lock(path: Path, wait: float = LOCK_WAIT) -> Iterator[None]:
    """
    Hold an exclusive lock for a whole genimg call, so parallel calls queue.

    The lock goes away with the process, even if it is killed. The file is not
    inherited by the server this call may start.
    """
    with open(path, "a") as lock:
        start = time.monotonic()
        told = False
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if not told:
                    print("waiting for another genimg call...", file=sys.stderr)
                    told = True
                if time.monotonic() - start > wait:
                    raise KeepAliveError(
                        f"another genimg call held {path} for over {wait:.0f}s"
                    ) from None
                time.sleep(POLL_INTERVAL)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _read_state(path: Path) -> dict[str, Any] | None:
    try:
        state = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return state if isinstance(state, dict) else None


def _write_state(path: Path, state: dict[str, Any]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(state))
    os.replace(tmp, path)  # never a half-written file, even if killed


def _cmdline(pid: int) -> list[str]:
    """The command line of `pid`; empty if it is gone (or a zombie)."""
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return []
    return raw.decode(errors="replace").split("\0")[:-1]


def _ours(state: dict[str, Any]) -> bool:
    """
    Whether the state's pid is a live process started with the recorded command line.

    The command line must end with the recorded one (a script run through an
    interpreter has the interpreter in front); a zombie's is empty.
    """
    pid, argv = state.get("pid"), state.get("argv")
    if not isinstance(pid, int) or not isinstance(argv, list) or not argv:
        return False
    return _cmdline(pid)[-len(argv) :] == argv


def _port_in_use(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            return True
    except OSError:
        return False


def _log_tail(path: Path) -> str:
    try:
        lines = path.read_text(errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(lines[-LOG_TAIL_LINES:])


def running_model(state_path: Path) -> str | None:
    """The yaml model name the kept server was last used for, if one of ours is running."""
    state = _read_state(state_path)
    if state is not None and _ours(state):
        name = state.get("model_name")
        return name if isinstance(name, str) else None
    return None


def stop_server(state_path: Path) -> None:
    """
    Stop the kept server if the state file's pid is verifiably ours, then forget the state.

    Raises:
        KeepAliveError: If the process cannot be signalled or is still alive after
            SIGKILL; the state is kept so the next call knows about it
    """
    state = _read_state(state_path)
    if state is not None and _ours(state):
        pid = state["pid"]
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(pid, sig)  # it is its own process group leader
            except ProcessLookupError:
                break
            except PermissionError as e:
                raise KeepAliveError(
                    f"cannot stop the sd-server (pid {pid}): {e}"
                ) from e
            deadline = time.monotonic() + STOP_TIMEOUT
            while _ours(state) and time.monotonic() < deadline:
                time.sleep(0.05)
            if not _ours(state):
                break
        if _ours(state):
            raise KeepAliveError(
                f"the sd-server (pid {pid}) did not stop; kill it by hand (kill -9 {pid})"
            )
        # A process drops its command line before it closes its sockets: wait for the port too.
        port = state.get("port")
        deadline = time.monotonic() + STOP_TIMEOUT
        while (
            isinstance(port, int) and _port_in_use(port) and time.monotonic() < deadline
        ):
            time.sleep(0.05)
    state_path.unlink(missing_ok=True)


def _wait_until_answering(server: SdCppServer, state: dict[str, Any]) -> bool:
    """Wait for a server of ours that may still be starting; False if it died or hangs."""
    deadline = time.monotonic() + STARTUP_TIMEOUT
    while not server.ready():
        if not _ours(state) or time.monotonic() > deadline:
            return False
        time.sleep(POLL_INTERVAL)
    return True


def ensure_server(
    binary: str,
    args: dict[str, Any],
    *,
    port: int,
    model_name: str,
    paths: Paths,
) -> tuple[SdCppServer, bool]:
    """
    Return a running server for `args` on `port`, and whether this call started it.

    Reuses ours when its command line is the one wanted, replaces ours when it
    differs, starts one when none is ours, and refuses when something else is
    listening on the port. Call it with the lock held.

    Raises:
        KeepAliveError: If something else is listening on the port, or the old
            server will not stop
        SdCppError: `server_died` if the new server fails to start (with the log tail)
        OSError: If the binary cannot be run
    """
    want = listen_argv(build_argv(binary, args), port)
    state = _read_state(paths.state)
    if state is not None and _ours(state):
        if state.get("argv") == want:
            server = SdCppServer.attach(port, paths.log)
            if _wait_until_answering(server, state):
                if state.get("model_name") != model_name:
                    _write_state(paths.state, {**state, "model_name": model_name})
                print(f"reusing the sd-server already running (pid {state['pid']})")
                return server, False
        print(f"stopping the running sd-server (pid {state['pid']})")
        stop_server(paths.state)
    else:
        # The state is stale: the process is gone or the pid was reused.
        paths.state.unlink(missing_ok=True)
    if _port_in_use(port):
        raise KeepAliveError(
            f"something is already listening on port {port} that this genimg did not "
            "start (a manual sd-server, or another session); stop it, or set GENIMG_PORT "
            "to use another port"
        )

    server = SdCppServer(binary, args, paths.log, port=port, keep=True)
    server.spawn()
    # Written before waiting: a call killed while the server starts must not orphan it.
    _write_state(
        paths.state,
        {
            "pid": server.pid,
            "port": port,
            "argv": server.argv,
            "model_name": model_name,
        },
    )
    try:
        server.wait_ready()
    except BaseException as e:
        server.stop()
        paths.state.unlink(missing_ok=True)
        if isinstance(e, SdCppError):
            tail = _log_tail(paths.log)
            raise SdCppError(
                e.kind, f"{e.message}\n{tail}" if tail else e.message, e.exit_code
            ) from e
        raise
    print(
        f"started an sd-server (pid {server.pid}, port {port}); it stays loaded for the "
        f"next genimg call (stop it with: kill {server.pid})"
    )
    return server, True
