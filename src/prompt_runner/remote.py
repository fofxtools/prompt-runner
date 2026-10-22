"""Remote mode: genimg and llm against a server that runs on another machine.

`GENIMG_URL` / `LLM_URL` (for example `http://192.168.56.1:8190`) switch the two commands from "start and keep my own
server" to "use the one that is already running there": no lock, no state file, no start, no stop. This is for a machine
without a GPU (a VM); the server is started and switched on the host with `genimg --start` and `llm --start`.

Standard library only, and nothing from keepalive.py, so it works on Windows.
"""

import os
from collections.abc import Callable
from pathlib import PurePosixPath
from typing import Any, TypeVar
from urllib.parse import urlsplit

T = TypeVar("T")

# A server's address: host and port.
Address = tuple[str, int]


class RemoteError(Exception):
    """A remote server that cannot be used: a bad URL, not reachable, or not the model that was asked for."""


def url_of(address: Address) -> str:
    """`http://host:port`."""
    return f"http://{address[0]}:{address[1]}"


def remote_address(tool: str) -> Address | None:
    """
    The server of `tool` (from $<TOOL>_URL), or None when the variable is not set.

    Raises:
        RemoteError: If it is not `http://host:port`
    """
    var = f"{tool.upper()}_URL"
    text = os.environ.get(var, "")
    if not text:
        return None
    try:
        parts = urlsplit(text)
        port = parts.port
    except ValueError:
        port = None
        parts = urlsplit("")
    if (
        parts.scheme != "http"
        or not parts.hostname
        or port is None
        or parts.path not in ("", "/")
        or parts.query
        or parts.fragment
        or parts.username
    ):
        raise RemoteError(
            f"{var} must look like http://192.168.56.1:8190, got '{text}'"
        )
    return parts.hostname, port


def ask(address: Address, question: Callable[[], T]) -> T:
    """Ask the server something; a server that does not answer is a RemoteError, not a traceback."""
    try:
        return question()
    except (OSError, ValueError) as e:  # URLError and HTTPError are OSErrors
        raise RemoteError(f"cannot reach the server at {url_of(address)}: {e}") from e


def _files(model: dict[str, Any], flags: tuple[str, ...]) -> list[str]:
    return [str(model["server_args"][f]) for f in flags if model["server_args"].get(f)]


def models_for_file(
    models: list[dict[str, Any]], loaded: str, flags: tuple[str, ...]
) -> list[dict[str, Any]]:
    """The entries whose model file (under `flags`) is `loaded`; several entries can share one file."""
    return [m for m in models if loaded in _files(m, flags)]


def check_loaded(
    model: dict[str, Any],
    loaded: str | None,
    flags: tuple[str, ...],
    address: Address,
    load_hint: str,
) -> None:
    """
    Raise unless the server has `model`'s file loaded (a server that reports no file passes).

    Raises:
        RemoteError: Naming both files and the command that loads the model on the host
    """
    wanted = _files(model, flags)
    if loaded is None or not wanted or loaded in wanted:
        return
    raise RemoteError(
        f"the server at {url_of(address)} has {PurePosixPath(loaded).name} loaded, but "
        f"'{model['name']}' needs {PurePosixPath(wanted[0]).name}; load it on the host with: {load_hint}"
    )


def not_here(tool: str, what: str, hint: str) -> RemoteError:
    """The error for a command that only works on the host (--start, --stop)."""
    return RemoteError(
        f"{what} is not available for a remote server ({tool.upper()}_URL is set): do it on the host, with {hint}"
    )
