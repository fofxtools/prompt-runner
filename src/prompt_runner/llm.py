"""llm: chat with a llama.cpp model from the command line (one reply or a chat; --list, --start, --status, --stop)."""

import argparse
import contextlib
import difflib
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from . import keepalive, remote
from .config import load_config, load_llm_models
from .llama_server import DeltaKind, LlamaServer
from .llm_runner import MMPROJ_FLAGS, LlmJob, run_llm_jobs
from .server_process import ServerError
from .utils import generate_run_identifiers, merged_server_args, models_table

with contextlib.suppress(ImportError):
    import readline  # noqa: F401  (gives input() line editing and history)

DESCRIPTION = """\
Chat with a model from config/llm_models.yaml.

  llm MODEL [PROMPT]    one reply (the prompt can be piped in), or a chat at a terminal when there is none
  llm --list            the models, their size on disk, which one is loaded, and which can see images
  llm --start [MODEL]   load a model and stop (MODEL, else the loaded one, else default_llm_model in config.yaml)
  llm --status          whether the server is running (model, pid, port, log)
  llm --stop            unload the model

Options for a reply: --system TEXT; --image FILE (repeatable) sends images with the first message;
--reasoning off turns thinking off (for models that can); -o FILE also writes the answer to FILE.
"""

EPILOG = """\
Linux only, unless $LLM_URL is set (below). The model's llama server stays loaded between calls, so the next call skips the load; a call for another model
replaces it, and calls queue behind one another. The server listens on port 8191 (or $LLM_PORT) of the listen_host address of config.yaml (127.0.0.1 unless set; $LLM_HOST overrides it); if something else is
listening there, llm refuses to touch it. Files: /tmp/llm-<port>.lock, .json (what runs) and .log (the server's output).

The answer goes to stdout and the model's thinking, dimmed, to stderr, so `llm MODEL "..." > answer.md` keeps only the
answer. In a chat, /bye or Ctrl-D ends it and Ctrl-C stops the reply being written (that turn is dropped). Server flags and
sampling come from config/config.yaml and the model's entry, as in prompt-runner.

Every reply is logged to runs.jsonl, with the same rows as prompt-runner's LLM runner (the messages, the seed used, sampling, tokens,
speeds, server flags): in <results_dir>/llm-cli/ of the repo, or with -o in .llm/ beside the file. -o FILE also writes the answer to
FILE (a one-shot only; an existing file is never overwritten).

Remote: with LLM_URL=http://HOST:PORT the command only asks a server on another machine (for a VM without a GPU): it never
starts, stops or locks anything, and MODEL must be the model the server has loaded. See docs/windows-vm.md.
"""
TOOL = "llm"  # names the kept server's files and $LLM_PORT
FILE_FLAGS = ("-m", "--model", "--mmproj")  # the server_args that name files on disk
# the server_args that name the model file (what a remote server reports)
MODEL_FLAGS = ("-m", "--model")
DEFAULT_PORT = 8191
# What --reasoning sends. Models whose chat template has no such switch ignore it.
REASONING_OPTIONS: dict[str, dict[str, Any]] = {
    "off": {"chat_template_kwargs": {"enable_thinking": False}}
}


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
        raise LlmError(f"unknown model '{name}'{hint}; see `llm --list`")
    return models[names.index(name)]


def _list(models: list[dict[str, Any]], kept: keepalive.KeptServer[LlamaServer]) -> str:
    """The models as a table: name, size of their files, and `running` or `missing`."""
    running = keepalive.running_model(kept)
    return models_table(
        models, {running} if running else set(), FILE_FLAGS, MMPROJ_FLAGS
    )


def _start(
    args: argparse.Namespace,
    config: dict[str, Any],
    models: list[dict[str, Any]],
    kept: keepalive.KeptServer[LlamaServer],
) -> str:
    """
    Make sure the kept server is running with its model loaded; return what happened.

    The model is MODEL, else the one already loaded, else `default_llm_model` in config.yaml.
    llama.cpp loads the model before its server answers, so nothing needs to be asked of it.
    """
    if not config.get("llama_server"):
        raise LlmError("config/config.yaml has no llama_server")
    with keepalive.call_lock(kept):
        running = keepalive.running_model(kept)
        name = args.model or running or config.get("default_llm_model")
        if not name:
            raise LlmError(
                "no MODEL given, none is loaded, and config/config.yaml has no default_llm_model"
            )
        model = _pick_model(models, name)
        if running != model["name"]:
            _say(f"loading {model['name']} ...")
        try:
            _, started = keepalive.ensure_server(
                kept,
                config["llama_server"],
                merged_server_args(config.get("llm_server_args"), model),
                model_name=model["name"],
                say=_say,
            )
        except ServerError as e:
            raise LlmError(f"{e} (server log: {kept.log})") from e
    return f"llama-server ready with {model['name']} ({'started' if started else 'already running'})"


def _stop(kept: keepalive.KeptServer[LlamaServer]) -> str:
    """Stop the kept server once any running call is done; return what happened."""
    with keepalive.call_lock(kept):
        was_running = keepalive.running_model(kept) is not None
        keepalive.stop_server(kept)
    return "stopped the llama-server" if was_running else "no llm server is running"


class _OutputClosed(Exception):
    """Whoever reads stdout went away (`llm MODEL "..." | head`): stop the reply, quietly."""


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
            self._out(piece)
        self.kind = kind

    @staticmethod
    def _out(text: str) -> None:
        """Write to stdout. Not an OSError, so that it is not taken for a lost connection to the server."""
        try:
            sys.stdout.write(text)
            sys.stdout.flush()
        except BrokenPipeError as e:
            raise _OutputClosed from e

    def finish(self, text: str) -> None:
        """End the line the reply stopped on."""
        if text and not text.endswith("\n"):
            self._out("\n")
        elif not text and self.kind == "reasoning":
            sys.stderr.write("\n")
            sys.stderr.flush()


def _check_out(out: Path) -> None:
    """Refuse an -o that is a folder or exists: an answer file is never overwritten."""
    if out.is_dir():
        raise LlmError(
            f"{out} is a folder; give a file name, e.g. -o {out / 'answer.md'}"
        )
    if out.exists():
        raise LlmError(f"{out} already exists; choose another name")


def _log_dir(root: Path, config: dict[str, Any], out: Path | None) -> Path:
    """Where runs.jsonl goes: in .llm/ beside the -o file, else under the results folder of the repo."""
    if out is not None:
        return out.resolve().parent / ".llm"
    return root / Path(config["results_dir"]) / "llm-cli"


def _quiet(_text: str) -> None:
    """Ignore a progress message."""


@dataclass
class _Chat:
    """
    What the turns of one `llm MODEL` call share: the server, the model, the request fields and the log.

    `server` is the kept server of this machine, or the address of one on another machine
    ($LLM_URL). Each turn is one job through prompt-runner's run_llm_jobs, so it is logged like
    a runner row.
    """

    server: keepalive.KeptServer[LlamaServer] | remote.Address
    config: dict[str, Any]
    model: dict[str, Any]
    options: dict[str, Any]  # request fields from the command line (--reasoning)
    log_dir: Path
    out: Path | None = None  # the -o file (one-shots only)
    session: str = field(default_factory=lambda: generate_run_identifiers()[0])
    turns: int = 0

    def turn(self, messages: list[dict[str, Any]], say: Callable[[str], None]) -> str:
        """
        Send `messages` to the model, loading it first if needed, print the reply as it comes, and return its text.

        On this machine the lock is held while the server is started or replaced and while the
        reply streams, so that another call cannot replace the model under it. A remote server is
        only asked: it must have the model loaded already. The row goes to runs.jsonl whether the
        turn works or not.

        Raises:
            LlmError: If the turn failed (a dead local server is stopped), with where to look
            keepalive.KeepAliveError: If the port is not ours or the lock never came free
            KeyboardInterrupt: If interrupted (closing the connection stops the reply)
        """
        self.turns += 1
        job = LlmJob(
            self.model, messages, self.out, dict(self.options), {"turn": self.turns}
        )
        printer = _Printer()
        server = self.server
        if isinstance(server, tuple):
            row = self._remote_turn(server, job, printer)
            where = f"server: {remote.url_of(server)}"
        else:
            row = self._local_turn(server, job, printer, say)
            where = f"server log: {server.log}"
        if row["status"] != "ok":
            detail = row["error"]
            if row["kind"] in ("job_failed", "server_died"):
                detail += f" ({where})"
            raise LlmError(detail)
        text: str = row["output"]["text"]
        printer.finish(text)
        if row["finish_reason"] == "length":
            _say("(stopped at max_tokens)")
        return text

    def _local_turn(
        self,
        kept: keepalive.KeptServer[LlamaServer],
        job: LlmJob,
        printer: _Printer,
        say: Callable[[str], None],
    ) -> dict[str, Any]:
        """Run the job on the kept server of this machine, starting or replacing it if needed."""
        model = self.model

        def open_server(
            binary: str, server_args: dict[str, Any], _log_path: Path
        ) -> LlamaServer:
            if keepalive.running_model(kept) != model["name"]:
                _say(f"loading {model['name']} ...")
            server, started = keepalive.ensure_server(
                kept, binary, server_args, model_name=model["name"], say=say
            )
            job.tags["server"] = "started" if started else "warm"
            return server

        with keepalive.call_lock(kept):
            if self.out is not None:
                _check_out(self.out)  # another call may have written it meanwhile
            row = self._run_job(
                job, printer, open_server, binary=self.config["llama_server"]
            )
            if row["kind"] == "server_died":
                # A dead server, or one that hung, must not be reused by the next call.
                with contextlib.suppress(keepalive.KeepAliveError):
                    keepalive.stop_server(kept)
        return row

    def _remote_turn(
        self, address: remote.Address, job: LlmJob, printer: _Printer
    ) -> dict[str, Any]:
        """Run the job on a server somewhere else, after checking that it has this model loaded."""
        server = LlamaServer.attach(address[1], host=address[0])
        try:
            loaded = remote.ask(address, server.loaded_file)
            remote.check_loaded(
                self.model,
                loaded,
                MODEL_FLAGS,
                address,
                f"llm --start {self.model['name']}",
            )
            build = remote.ask(address, server.build_info)
        except remote.RemoteError as e:  # a chat goes on after a hiccup
            raise LlmError(str(e)) from e
        job.tags.update(server="remote", url=remote.url_of(address))
        if self.out is not None:
            _check_out(self.out)
        return self._run_job(
            job, printer, lambda *_: server, engine=f"{build or 'llama'} (remote)"
        )

    def _run_job(
        self,
        job: LlmJob,
        printer: _Printer,
        open_server: Callable[[str, dict[str, Any], Path], LlamaServer],
        binary: str = "",
        engine: str | None = None,
    ) -> dict[str, Any]:
        """Run the one job through the runner (which logs it) and return its row."""
        try:
            (row,) = run_llm_jobs(
                [job],
                binary=binary,
                out_dir=self.log_dir,
                server_defaults=self.config.get("llm_server_args"),
                generation_defaults=self.config.get("llm_generation_defaults"),
                open_server=open_server,
                engine=engine,
                on_delta=printer,
                say=_quiet,
                run_id=self.session,
            )
        except ValueError as e:  # an option the server would ignore, a missing image
            raise LlmError(str(e)) from e
        return row


def _user_message(text: str, images: list[str]) -> dict[str, Any]:
    """A user message, with its images if there are any."""
    return {"role": "user", "content": text, **({"images": images} if images else {})}


def _chat_loop(chat: _Chat, messages: list[dict[str, Any]], images: list[str]) -> int:
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
            text = chat.turn(messages, _say if first else _quiet)
        except KeyboardInterrupt:
            _say("\n(cancelled)")
            messages.pop()
            continue
        except LlmError as e:
            _say(f"llm failed: {e}")
            messages.pop()
            continue
        if text:
            first, images = False, []
            messages.append({"role": "assistant", "content": text})
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


def _run(
    args: argparse.Namespace,
    config: dict[str, Any],
    models: list[dict[str, Any]],
    server: keepalive.KeptServer[LlamaServer] | remote.Address,
    root: Path,
) -> int:
    """`llm MODEL`: one reply, or a chat."""
    model = _pick_model(models, args.model)
    if not isinstance(server, tuple) and not config.get("llama_server"):
        raise LlmError("config/config.yaml has no llama_server")
    out = Path(args.output) if args.output else None
    if out is not None:
        _check_out(out)
    messages: list[dict[str, Any]] = (
        [{"role": "system", "content": args.system}] if args.system else []
    )
    images = _images_for(args, model, config)
    chat = _Chat(
        server,
        config,
        model,
        REASONING_OPTIONS[args.reasoning] if args.reasoning else {},
        _log_dir(root, config, out),
        out,
    )
    if args.prompt is not None:
        prompt = (
            args.prompt
        )  # stdin is not read: it may be an open pipe that never ends
    elif sys.stdin.isatty():
        if out is not None:
            raise LlmError("-o needs a prompt: a chat has many answers, not one")
        return _chat_loop(chat, messages, images)
    else:
        prompt = sys.stdin.read().strip()  # echo "..." | llm MODEL
    if not prompt:
        raise LlmError("the prompt is empty")
    messages.append(_user_message(prompt, images))
    chat.turn(messages, _say)
    return 0


def _remote(
    args: argparse.Namespace,
    config: dict[str, Any],
    models: list[dict[str, Any]],
    address: remote.Address,
    root: Path,
) -> int:
    """
    The commands against a server on another machine ($LLM_URL): it is asked, never started or stopped.

    The model file the server reports as loaded decides which model entries are running.
    """
    if args.start:
        raise remote.not_here(TOOL, "--start", "llm --start [MODEL]")
    if args.stop:
        raise remote.not_here(TOOL, "--stop", "llm --stop")
    if not (args.list or args.status):
        return _run(args, config, models, address, root)
    server = LlamaServer.attach(address[1], host=address[0])
    loaded = remote.ask(address, server.loaded_file)
    matches = remote.models_for_file(models, loaded, MODEL_FLAGS) if loaded else []
    if args.list:
        print(models_table(models, {m["name"] for m in matches}, (), MMPROJ_FLAGS))
    else:
        what = PurePosixPath(loaded).name if loaded else "a model"
        entries = (
            f"config entry: {', '.join(m['name'] for m in matches)}"
            if matches
            else (
                "not in config/llm_models.yaml"
                if loaded
                else "it reports no model file"
            )
        )
        print(f"running {what} at {remote.url_of(address)} (remote; {entries})")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    ap = argparse.ArgumentParser(
        prog="llm",
        description=DESCRIPTION,
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("model", nargs="?", help="model name (see --list)")
    ap.add_argument(
        "prompt",
        nargs="?",
        help="the prompt; without one, text piped in is the prompt, and at a terminal a chat starts",
    )
    ap.add_argument("--system", help="a system prompt")
    ap.add_argument(
        "--image",
        action="append",
        metavar="FILE",
        help="an image to send with the first message (repeatable; the model needs a vision projector)",
    )
    ap.add_argument(
        "--reasoning",
        choices=list(REASONING_OPTIONS),
        help="off: turn thinking off (models whose chat template has no switch ignore it)",
    )
    ap.add_argument(
        "-o",
        "--output",
        metavar="FILE",
        help="also write the answer to FILE (a one-shot only; never overwrites), and keep runs.jsonl in .llm/ beside it",
    )
    ap.add_argument("--list", action="store_true", help="list the models and exit")
    ap.add_argument(
        "--start",
        action="store_true",
        help="start the server and load a model (MODEL, else the loaded one, else default_llm_model), then exit",
    )
    ap.add_argument(
        "--status",
        action="store_true",
        help="show whether the server is running (model, pid, port, log) and exit",
    )
    ap.add_argument(
        "--stop",
        action="store_true",
        help="unload the model (after any running call) and exit",
    )
    return ap


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    """
    Parse the command line. Options may come before or after the prompt, which argparse does only
    with parse_intermixed_args.
    """
    ap = build_parser()
    args = ap.parse_intermixed_args(argv)
    if args.list + args.start + args.status + args.stop > 1:
        ap.error("choose one of --list, --start, --status and --stop")
    options = (args.prompt, args.system, args.image, args.reasoning, args.output)
    if args.start:
        if any(options):
            ap.error("--start takes only a MODEL")
    elif args.list or args.status or args.stop:
        if args.model or any(options):
            ap.error("--list, --status and --stop take no model or other options")
    elif not args.model:
        ap.error("a MODEL is required (or use --list / --start / --status / --stop)")
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
        The exit code: 0 on success, 1 on failure (a message on stderr), 130 if interrupted, 141 if
        stdout was closed (a pipe whose reader left; the reply is stopped)
    """
    args = parse_args(argv)
    root = repo_root if repo_root is not None else Path(__file__).resolve().parents[2]
    try:
        config, models = _load(root)
        address = remote.remote_address(TOOL)
        if address is not None:
            return _remote(args, config, models, address, root)
        keepalive.check_supported(TOOL)
        kept = keepalive.kept_server(
            TOOL, LlamaServer, _port(), state_dir, keepalive.listen_host(TOOL, config)
        )
        if args.list:
            print(_list(models, kept))
        elif args.start:
            print(_start(args, config, models, kept))
        elif args.stop:
            print(_stop(kept))
        elif args.status:
            print(keepalive.server_status(kept))
        else:
            return _run(args, config, models, kept, root)
    except (
        LlmError,
        keepalive.KeepAliveError,
        remote.RemoteError,
        ServerError,
        OSError,
    ) as e:
        print(f"llm failed: {e}", file=sys.stderr)
        return 1
    except _OutputClosed:
        # Python flushes stdout again when it exits and would complain: point it at /dev/null.
        with contextlib.suppress(OSError, ValueError):
            os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        return 141  # as a process killed by SIGPIPE
    except KeyboardInterrupt:
        print("\nllm: interrupted", file=sys.stderr)
        return 130
    return 0


if (
    __name__ == "__main__"
):  # python -m prompt_runner.llm (what the Windows .cmd files run)
    sys.exit(main())
