"""llm: talk to a llama.cpp model from the command line (list, run, stop)."""

import argparse
import contextlib
import difflib
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml

from . import keepalive
from .config import load_config, load_llm_models
from .llama_server import ChatReply, DeltaKind, LlamaServer
from .llm_runner import MMPROJ_FLAGS
from .server_process import ServerError
from .utils import merge_options, merged_server_args

with contextlib.suppress(ImportError):
    import readline  # noqa: F401  (gives input() line editing and history)

DESCRIPTION = """\
Chat with a model from config/llm_models.yaml, like `ollama run`.

  llm list                      the models, their size on disk, and which one is loaded
  llm run MODEL [PROMPT]        one reply (the prompt can be piped in), or a chat at a terminal when there is none;
                                --image FILE (repeatable) sends images with the first message
  llm stop                      unload the model
"""

EPILOG = """\
Linux only. The model's llama server stays loaded between calls, so the next call skips the load; a call for another model
replaces it, and calls queue behind one another. The server listens on port 8191 (or $LLM_PORT); if something else is
listening there, llm refuses to touch it. Files: /tmp/llm-<port>.lock, .json (what runs) and .log (the server's output).

The answer goes to stdout and the model's thinking, dimmed, to stderr, so `llm run MODEL "..." > answer.md` keeps only the
answer. In a chat, /bye or Ctrl-D ends it and Ctrl-C stops the reply being written (that turn is dropped). Server flags and
sampling come from config/config.yaml and the model's entry, as in prompt-runner.
"""
TOOL = "llm"  # names the kept server's files and $LLM_PORT
DEFAULT_PORT = 8191
FILE_FLAGS = ("-m", "--model", "--mmproj")  # the server_args that name files on disk


class LlmError(Exception):
    """An expected failure: reported as `llm failed: <message>`."""


def _say(text: str) -> None:
    """A status line, on stderr so that stdout is only the answer."""
    print(text, file=sys.stderr, flush=True)


def _port() -> int:
    """The port of the kept server: $LLM_PORT, else 8191."""
    text = os.environ.get("LLM_PORT", str(DEFAULT_PORT))
    if not text.isdigit() or not 0 < int(text) < 65536:
        raise LlmError(f"LLM_PORT must be a port number, got '{text}'")
    return int(text)


def _load(repo_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load config.yaml and llm_models.yaml."""
    try:
        config = load_config(str(repo_root / "config/config.yaml"))
        models = load_llm_models(str(repo_root / "config/llm_models.yaml"))
    except (OSError, ValueError, TypeError, yaml.YAMLError) as e:
        raise LlmError(f"cannot load the config: {e}") from e
    return config, models


def _pick_model(models: list[dict[str, Any]], name: str) -> dict[str, Any]:
    """The model entry called `name`."""
    names = [m["name"] for m in models]
    if name not in names:
        close = difflib.get_close_matches(name, names, n=3, cutoff=0.5)
        hint = f" (did you mean: {', '.join(close)}?)" if close else ""
        raise LlmError(f"unknown model '{name}'{hint}; see `llm list`")
    return models[names.index(name)]


def _list(models: list[dict[str, Any]], kept: keepalive.KeptServer[LlamaServer]) -> str:
    """The models as a table: name, size of their files, and `running` or `missing`."""
    running = keepalive.running_model(kept)
    rows = [("NAME", "SIZE", "STATE")]
    for model in models:
        files = [
            Path(str(model["server_args"][flag]))
            for flag in FILE_FLAGS
            if model["server_args"].get(flag)
        ]
        present = [f for f in files if f.is_file()]
        size = (
            f"{sum(f.stat().st_size for f in present) / 1e9:.1f} GB" if files else "-"
        )
        if len(present) < len(files):
            state = "missing"
        else:
            state = "running" if model["name"] == running else ""
        rows.append((model["name"], size, state))
    width = [max(len(row[i]) for row in rows) for i in range(2)]
    return "\n".join(
        f"{name:<{width[0]}}  {size:>{width[1]}}  {state}".rstrip()
        for name, size, state in rows
    )


def _stop(kept: keepalive.KeptServer[LlamaServer]) -> str:
    """Stop the kept server once any running call is done; return what happened."""
    with keepalive.call_lock(kept):
        was_running = keepalive.running_model(kept) is not None
        keepalive.stop_server(kept)
    return "stopped the llama-server" if was_running else "no llm server is running"


class _Printer:
    """Shows a reply as it streams: the answer on stdout, the thinking (dimmed on a terminal) on stderr."""

    def __init__(self) -> None:
        self.kind: DeltaKind | None = None
        self.dim = sys.stderr.isatty()

    def __call__(self, kind: DeltaKind, piece: str) -> None:
        if kind == "reasoning":
            sys.stderr.write(f"\x1b[2m{piece}\x1b[0m" if self.dim else piece)
            sys.stderr.flush()
        else:
            if self.kind == "reasoning":
                sys.stderr.write(
                    "\n"
                )  # the thinking ends; the answer starts on a new line
                sys.stderr.flush()
            sys.stdout.write(piece)
            sys.stdout.flush()
        self.kind = kind

    def finish(self, reply: ChatReply) -> None:
        """End the line the reply stopped on."""
        if reply.text and not reply.text.endswith("\n"):
            sys.stdout.write("\n")
            sys.stdout.flush()
        elif not reply.text and self.kind == "reasoning":
            sys.stderr.write("\n")
            sys.stderr.flush()


def _turn(
    kept: keepalive.KeptServer[LlamaServer],
    config: dict[str, Any],
    model: dict[str, Any],
    messages: list[dict[str, Any]],
    say: Callable[[str], None],
) -> ChatReply:
    """
    Send `messages` to the model, loading it first if needed, and print the reply as it comes.

    The lock is held while the server is started or replaced and while the reply
    streams, so that another call cannot replace the model under it.

    Raises:
        ServerError: If the server failed to start or the chat failed (a dead server is stopped)
        keepalive.KeepAliveError: If the port is not ours or the lock never came free
        KeyboardInterrupt: If interrupted (closing the connection stops the reply)
    """
    args = merged_server_args(config.get("llm_server_args"), model)
    options = merge_options(
        config.get("llm_generation_defaults"), model.get("generation_options")
    )
    printer = _Printer()
    with keepalive.call_lock(kept):
        if keepalive.running_model(kept) != model["name"]:
            _say(f"loading {model['name']} ...")
        server, _ = keepalive.ensure_server(
            kept, config["llama_server"], args, model_name=model["name"], say=say
        )
        try:
            reply = server.chat(messages, options, on_delta=printer)
        except ServerError as e:
            if e.kind == "server_died":
                # A dead server, or one that hung, must not be reused by the next call.
                with contextlib.suppress(keepalive.KeepAliveError):
                    keepalive.stop_server(kept)
            raise
    printer.finish(reply)
    if reply.finish_reason == "length":
        _say("(stopped at max_tokens)")
    return reply


def _user_message(text: str, images: list[str]) -> dict[str, Any]:
    """A user message, with its images if there are any."""
    return {"role": "user", "content": text, **({"images": images} if images else {})}


def _chat_loop(
    kept: keepalive.KeptServer[LlamaServer],
    config: dict[str, Any],
    model: dict[str, Any],
    messages: list[dict[str, Any]],
    images: list[str],
) -> int:
    """
    A chat: the history lives here, so each turn works even if the server was replaced meanwhile.

    `images` go with the first message that gets an answer (a dropped turn keeps them for the next).
    """
    first = True
    while True:
        try:
            line = input(">>> ").strip()
        except EOFError:
            print()
            return 0
        except KeyboardInterrupt:
            print()
            _say("(Ctrl-D or /bye ends the chat)")
            continue
        if line == "/bye":
            return 0
        if not line:
            continue
        messages.append(_user_message(line, images))
        try:
            reply = _turn(kept, config, model, messages, _say if first else _quiet)
        except KeyboardInterrupt:
            _say("\n(cancelled)")
            messages.pop()
            continue
        except ServerError as e:
            _say(f"llm failed: {e} (server log: {kept.log})")
            messages.pop()
            continue
        except OSError as e:  # an image that cannot be read; nothing was sent
            _say(f"llm failed: {e}")
            messages.pop()
            continue
        if reply.text:
            first, images = False, []
            messages.append({"role": "assistant", "content": reply.text})
        else:  # the turn is dropped, so its images stay for the next one
            _say("(no answer: it was still thinking when it stopped)")
            messages.pop()


def _images_for(
    args: argparse.Namespace, model: dict[str, Any], config: dict[str, Any]
) -> list[str]:
    """The --image files, checked: they exist, and the model has a vision projector."""
    images: list[str] = args.image or []
    missing = [path for path in images if not Path(path).is_file()]
    if missing:
        raise LlmError(f"image file not found: {', '.join(missing)}")
    server_args = merged_server_args(config.get("llm_server_args"), model)
    if images and not any(server_args.get(flag) for flag in MMPROJ_FLAGS):
        raise LlmError(
            f"model '{model['name']}' has no --mmproj, so it cannot see images"
        )
    return images


def _quiet(_text: str) -> None:
    """Ignore a progress message."""


def _run(
    args: argparse.Namespace,
    config: dict[str, Any],
    models: list[dict[str, Any]],
    kept: keepalive.KeptServer[LlamaServer],
) -> int:
    """`llm run`: one reply, or a chat."""
    model = _pick_model(models, args.model)
    if not config.get("llama_server"):
        raise LlmError("config/config.yaml has no llama_server")
    messages: list[dict[str, Any]] = (
        [{"role": "system", "content": args.system}] if args.system else []
    )
    images = _images_for(args, model, config)
    if args.prompt is not None:
        prompt = (
            args.prompt
        )  # stdin is not read: it may be an open pipe that never ends
    elif sys.stdin.isatty():
        return _chat_loop(kept, config, model, messages, images)
    else:
        prompt = sys.stdin.read().strip()  # echo "..." | llm run MODEL
    if not prompt:
        raise LlmError("the prompt is empty")
    messages.append(_user_message(prompt, images))
    try:
        _turn(kept, config, model, messages, _say)
    except ServerError as e:
        raise LlmError(f"{e} (server log: {kept.log})") from e
    return 0


COMMANDS = ("list", "run", "stop")


def build_parser() -> argparse.ArgumentParser:
    """The top-level parser: the command (its own options are parsed by `parse_args`)."""
    ap = argparse.ArgumentParser(
        prog="llm",
        description=DESCRIPTION,
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("command", choices=COMMANDS, metavar="COMMAND")
    ap.add_argument("rest", nargs=argparse.REMAINDER, help=argparse.SUPPRESS)
    return ap


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    """
    Parse the command line: `llm COMMAND ...`, with the options of `run` allowed before or after its prompt.

    (argparse subparsers cannot do that: `llm run MODEL --image FILE "prompt"` fails there, because
    the optional prompt is taken as empty before the option is seen.)
    """
    top = build_parser().parse_args(argv)
    sub = argparse.ArgumentParser(prog=f"llm {top.command}")
    if top.command == "run":
        sub.add_argument("model", help="model name (see `llm list`)")
        sub.add_argument(
            "prompt",
            nargs="?",
            help="the prompt; without one, text piped in is the prompt, and at a terminal a chat starts",
        )
        sub.add_argument("--system", help="a system prompt")
        sub.add_argument(
            "--image",
            action="append",
            metavar="FILE",
            help="an image to send with the first message (repeatable; the model needs a vision projector)",
        )
    args = sub.parse_intermixed_args(top.rest)
    args.command = top.command
    return args


def main(
    argv: list[str] | None = None,
    repo_root: Path | None = None,
    state_dir: Path | None = None,
) -> int:
    """
    Run llm.

    Args:
        argv: Command-line arguments (default: sys.argv[1:])
        repo_root: The prompt-runner folder that holds config/ (default: the folder
            this package was installed from)
        state_dir: Where the kept server's lock, state and log files go (default: the temp folder)

    Returns:
        The exit code: 0 on success, 1 on failure (a message on stderr), 130 if interrupted
    """
    args = parse_args(argv)
    root = repo_root if repo_root is not None else Path(__file__).resolve().parents[2]
    try:
        config, models = _load(root)
        keepalive.check_supported(TOOL)
        kept = keepalive.kept_server(TOOL, LlamaServer, _port(), state_dir)
        if args.command == "list":
            print(_list(models, kept))
        elif args.command == "stop":
            print(_stop(kept))
        else:
            return _run(args, config, models, kept)
    except (LlmError, keepalive.KeepAliveError, ServerError, OSError) as e:
        print(f"llm failed: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nllm: interrupted", file=sys.stderr)
        return 130
    return 0
