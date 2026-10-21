"""Keep one server running between calls of a command-line tool (genimg, llm).

A call starts the model's server if none is ours, leaves it running, and the next call reuses it. State is a JSON file
(`pid`, the exact command line, `model_name`) next to a lock and a log, all named after the tool and keyed by port. A pid counts as ours only when
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
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, TypeVar

from .server_process import (
    POLL_INTERVAL,
    STOP_TIMEOUT,
    ServerError,
    ServerProcess,
    build_argv,
)

LOCK_WAIT = 900.0  # seconds a call waits for another call of the tool before giving up
LOG_TAIL_LINES = 15
S = TypeVar("S", bound=ServerProcess)


class KeepAliveError(Exception):
    """The kept server cannot be used (something else owns the port, the lock never came free, or it will not stop)."""


def check_supported(tool: str, proc: Path = Path("/proc")) -> None:
    """Raise KeepAliveError unless this system has /proc (Linux), which the process check needs."""
    if not (proc / "self" / "cmdline").exists():
        raise KeepAliveError(
            f"{tool} keeps its server between calls using /proc and fcntl, so it needs Linux"
        )


@dataclass(frozen=True)
class KeptServer(Generic[S]):
    """One kept server: the tool that keeps it, the kind of server, its port and its files."""

    tool: str
    server_class: type[S]
    port: int
    lock: Path
    state: Path
    log: Path


def kept_server(
    tool: str, server_class: type[S], port: int, directory: Path | None = None
) -> KeptServer[S]:
    """The kept server of `tool` on `port`; its files go in `directory` (default: the temp folder)."""
    base = directory if directory is not None else Path(tempfile.gettempdir())
    base.mkdir(parents=True, exist_ok=True)
    stem = f"{tool}-{port}"
    return KeptServer(
        tool,
        server_class,
        port,
        base / f"{stem}.lock",
        base / f"{stem}.json",
        base / f"{stem}.log",
    )


@contextmanager
def call_lock(kept: KeptServer[Any], wait: float = LOCK_WAIT) -> Iterator[None]:
    """
    Hold an exclusive lock for a whole call of the tool, so parallel calls queue.

    The lock goes away with the process, even if it is killed. The file is not
    inherited by the server this call may start.
    """
    with open(kept.lock, "a") as lock:
        start = time.monotonic()
        told = False
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if not told:
                    print(f"waiting for another {kept.tool} call...", file=sys.stderr)
                    told = True
                if time.monotonic() - start > wait:
                    raise KeepAliveError(
                        f"another {kept.tool} call held {kept.lock} for over {wait:.0f}s"
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


def running_model(kept: KeptServer[Any]) -> str | None:
    """The yaml model name the kept server was last used for, if one of ours is running."""
    state = _read_state(kept.state)
    if state is not None and _ours(state):
        name = state.get("model_name")
        return name if isinstance(name, str) else None
    return None


def stop_server(kept: KeptServer[Any]) -> None:
    """
    Stop the kept server if the state file's pid is verifiably ours, then forget the state.

    Raises:
        KeepAliveError: If the process cannot be signalled or is still alive after
            SIGKILL; the state is kept so the next call knows about it
    """
    name = kept.server_class.NAME
    state = _read_state(kept.state)
    if state is not None and _ours(state):
        pid = state["pid"]
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(pid, sig)  # it is its own process group leader
            except ProcessLookupError:
                break
            except PermissionError as e:
                raise KeepAliveError(f"cannot stop the {name} (pid {pid}): {e}") from e
            deadline = time.monotonic() + STOP_TIMEOUT
            while _ours(state) and time.monotonic() < deadline:
                time.sleep(0.05)
            if not _ours(state):
                break
        if _ours(state):
            raise KeepAliveError(
                f"the {name} (pid {pid}) did not stop; kill it by hand (kill -9 {pid})"
            )
        # A process drops its command line before it closes its sockets: wait for the port too.
        port = state.get("port")
        deadline = time.monotonic() + STOP_TIMEOUT
        while (
            isinstance(port, int) and _port_in_use(port) and time.monotonic() < deadline
        ):
            time.sleep(0.05)
    kept.state.unlink(missing_ok=True)


def _wait_until_answering(server: ServerProcess, state: dict[str, Any]) -> bool:
    """Wait for a server of ours that may still be starting; False if it died or hangs."""
    deadline = time.monotonic() + server.STARTUP_TIMEOUT
    while not server.ready():
        if not _ours(state) or time.monotonic() > deadline:
            return False
        time.sleep(POLL_INTERVAL)
    return True


def ensure_server(
    kept: KeptServer[S],
    binary: str,
    args: dict[str, Any],
    *,
    model_name: str,
    say: Callable[[str], None] = print,
) -> tuple[S, bool]:
    """
    Return a running server for `args` on the kept server's port, and whether this call started it.

    Reuses ours when its command line is the one wanted, replaces ours when it
    differs, starts one when none is ours, and refuses when something else is
    listening on the port. Call it with the lock held. `say` gets the progress messages
    (reusing, stopping, started); a tool whose stdout is its result passes a function
    that writes them to stderr.

    Raises:
        KeepAliveError: If something else is listening on the port, or the old
            server will not stop
        ServerError: `server_died` if the new server fails to start (with the log tail)
        OSError: If the binary cannot be run
    """
    server_class, port, tool = kept.server_class, kept.port, kept.tool
    name = server_class.NAME
    want = server_class.listen_argv(build_argv(binary, args), port)
    state = _read_state(kept.state)
    if state is not None and _ours(state):
        if state.get("argv") == want:
            server = server_class.attach(port, kept.log)
            if _wait_until_answering(server, state):
                if state.get("model_name") != model_name:
                    _write_state(kept.state, {**state, "model_name": model_name})
                say(f"reusing the {name} already running (pid {state['pid']})")
                return server, False
        say(f"stopping the running {name} (pid {state['pid']})")
        stop_server(kept)
    else:
        # The state is stale: the process is gone or the pid was reused.
        kept.state.unlink(missing_ok=True)
    if _port_in_use(port):
        raise KeepAliveError(
            f"something is already listening on port {port} that this {tool} did not "
            f"start (a manual {name}, or another session); stop it, or set "
            f"{tool.upper()}_PORT to use another port"
        )

    server = server_class(binary, args, kept.log, port=port, keep=True)
    server.spawn()
    # Written before waiting: a call killed while the server starts must not orphan it.
    _write_state(
        kept.state,
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
        kept.state.unlink(missing_ok=True)
        if isinstance(e, ServerError):
            tail = _log_tail(kept.log)
            raise ServerError(
                e.kind, f"{e.message}\n{tail}" if tail else e.message, e.exit_code
            ) from e
        raise
    say(
        f"started the {name} (pid {server.pid}, port {port}); it stays loaded for the "
        f"next {tool} call (stop it with: kill {server.pid})"
    )
    return server, True
