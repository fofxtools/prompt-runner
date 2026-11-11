"""genimg: generate or edit one image with a prompt-runner image model."""

import argparse
import difflib
import os
import re
import subprocess
import sys
import unicodedata
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from . import keepalive, remote
from .config import load_config, load_image_models
from .image_runner import Job, prepare_request, run_jobs
from .sdcpp_server import SdCppServer
from .server_process import ServerError
from .utils import (
    merge_options,
    merged_server_args,
    models_table,
    random_seed,
    sanitize_fs_name,
)

DESCRIPTION = """\
Generate an image, or edit one with --ref (the prompt then describes the edit), using a model from
config/image_models.yaml. It goes through prompt-runner's run_jobs, so settings and logs match the runner.
"""

EPILOG = """\
Linux only, unless $GENIMG_URL is set (below). The sd-server stays loaded between calls, so the next call skips the model load (about 3 s). The first call starts
it and prints its pid; a call that needs a different model (different server flags) replaces it; calls queue behind
one another. `genimg --start` loads it ahead of time with a throwaway 64x64 image (e.g. before something else takes the GPU).
Stop it with `genimg --stop` (or `kill PID`); `genimg --status` shows what is running and `genimg --list` the models; it also ends with an OpenCode session. If something else is listening on the
port (8190, or $GENIMG_PORT) of the listen_host address of config.yaml (127.0.0.1 unless set; $GENIMG_HOST overrides it), genimg refuses to touch it. Files: /tmp/genimg-<port>.lock, .json (what runs) and
.log (the server's output).

Model: --model, else the model the running server was last used for, else `default_image_model` in
config/config.yaml. Size, steps, CFG and the other request fields come from config/config.yaml and the model's
entry; --size overrides the size. -1 (the default) means a random seed; the seed used is printed. --lora adds to
the LoRAs the model entry already lists (file names in the server's --lora-model-dir). The server's web UI
(http://HOST:PORT/) starts with the same size, seed, sampler, steps and CFG as its defaults; pick LoRAs there yourself.

The PNG is written to the -o file. Without -o, or with -o FOLDER/, genimg names it <date>-<time>_<first prompt words>_<model>_s<seed>.png
in the current folder (or FOLDER), adding -2, -3, ... if that name is taken. An existing file is never overwritten.
runs.jsonl goes to <output folder>/.genimg/.

Remote: with GENIMG_URL=http://HOST:PORT the command only asks a server on another machine (for a VM without a GPU): it never
starts, stops or locks anything, and the model must be the one the server has loaded. See docs/windows-vm.md.
"""
TOOL = "genimg"  # names the kept server's files and $GENIMG_PORT
# the server_args that name the main model file (what a remote server reports as loaded)
MODEL_FLAGS = ("--diffusion-model", "--model", "-m")
# the server_args that name model files on disk
FILE_FLAGS = (
    "-m",
    "--model",
    "--diffusion-model",
    "--vae",
    "--clip_l",
    "--clip_g",
    "--t5xxl",
    "--llm",
    "--llm_vision",
)
DEFAULT_PORT = 8190
# Request fields (paths into the merged generation options) and the sd-server flags that make the same value
# the default of the server's web UI. genimg's own requests send these fields anyway. LoRAs have no flag.
UI_DEFAULT_FLAGS = (
    (("width",), "--width"),
    (("height",), "--height"),
    (("seed",), "--seed"),
    (("sample_params", "sample_method"), "--sampling-method"),
    (("sample_params", "scheduler"), "--scheduler"),
    (("sample_params", "sample_steps"), "--steps"),
    (("sample_params", "custom_sigmas"), "--sigmas"),
    (("sample_params", "flow_shift"), "--flow-shift"),
    (("sample_params", "guidance", "txt_cfg"), "--cfg-scale"),
    (("sample_params", "guidance", "distilled_guidance"), "--guidance"),
)


class GenimgError(Exception):
    """An expected failure: reported as `genimg failed: <message>`."""


def _size(text: str) -> tuple[int, int]:
    """Parse WxH into (width, height)."""
    match = re.fullmatch(r"(\d+)x(\d+)", text.strip().lower())
    if not match or 0 in (int(match[1]), int(match[2])):
        raise argparse.ArgumentTypeError(
            f"expected WxH such as 1024x1024, got '{text}'"
        )
    return int(match[1]), int(match[2])


def _lora(text: str) -> dict[str, Any]:
    """Parse NAME[:WEIGHT] into a request `lora` entry."""
    name, sep, weight = text.rpartition(":")
    if not sep:
        return {"path": text, "multiplier": 1.0}
    try:
        return {"path": name, "multiplier": float(weight)}
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"LoRA weight must be a number: '{text}'"
        ) from None


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    ap = argparse.ArgumentParser(
        prog="genimg",
        description=DESCRIPTION,
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("prompt", nargs="?")
    ap.add_argument(
        "-o",
        "--output",
        help="PNG file to write, or a folder to name it in (default: named in the current folder)",
    )
    ap.add_argument(
        "--start",
        action="store_true",
        help="start the sd-server and load its model (--model, else as for an image), then exit",
    )
    ap.add_argument(
        "--stop",
        action="store_true",
        help="stop the kept sd-server (after any running call) and exit",
    )
    ap.add_argument(
        "--status",
        action="store_true",
        help="show whether the sd-server is running (model, pid, port, log) and exit",
    )
    ap.add_argument(
        "--list",
        action="store_true",
        help="list the models, their size on disk and which one is loaded, and exit",
    )
    ap.add_argument("--model", help="model name (default: default_image_model)")
    ap.add_argument("--seed", type=int, help="-1 or unset: random")
    ap.add_argument("--size", type=_size, metavar="WxH", help="default: from config")
    ap.add_argument(
        "--ref",
        action="append",
        default=[],
        metavar="IMAGE",
        help="reference image to edit (repeatable)",
    )
    ap.add_argument(
        "--lora",
        action="append",
        default=[],
        type=_lora,
        metavar="NAME[:WEIGHT]",
        help="extra LoRA file name, weight 1.0 by default (repeatable)",
    )
    return ap


def _pick_model(
    models: list[dict[str, Any]],
    name: str | None,
    config: dict[str, Any],
    from_server: bool = False,
) -> dict[str, Any]:
    """Return the model entry for `name` (--model, or the running server's), else for `default_image_model`."""
    name = name or config.get("default_image_model")
    if not name:
        raise GenimgError(
            "no model: pass --model NAME or set default_image_model in config/config.yaml"
        )
    names = [m["name"] for m in models]
    if name not in names:
        close = difflib.get_close_matches(name, names, n=3, cutoff=0.5)
        hint = f" (did you mean: {', '.join(close)}?)" if close else ""
        origin = " (the model of the running server)" if from_server else ""
        raise GenimgError(
            f"unknown model '{name}'{origin}{hint}; see config/image_models.yaml"
        )
    return models[names.index(name)]


def _job_options(
    args: argparse.Namespace, model: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    """Request fields for this image; everything else comes from the runner's merge."""
    base = merge_options(
        config.get("image_generation_defaults"), model.get("generation_options")
    )
    # Resolved here as the runner would (-1 or unset: random), so an automatic file name can carry it.
    seed = args.seed if args.seed is not None else base.get("seed")
    options: dict[str, Any] = {
        "prompt": args.prompt,
        "seed": random_seed() if seed in (None, -1) else seed,
    }
    if args.size:
        options["width"], options["height"] = args.size
    if args.ref:
        # Absolute, because runs.jsonl records the paths and lives in another folder.
        options["ref_images"] = [str(Path(ref).resolve()) for ref in args.ref]
    if args.lora:
        # Lists are replaced when options are merged, so start from the model's own LoRAs.
        options["lora"] = [*base.get("lora", []), *args.lora]
    return options


def _slug(prompt: str, words: int = 15, limit: int = 100) -> str:
    """
    A file-name-safe stand-in for the prompt: its first words in lowercase ASCII, joined by hyphens.

    Examples:
        >>> _slug("Make her dress GREEN, please!")
        'make-her-dress-green-please'
        >>> _slug("Café crème brûlée")
        'cafe-creme-brulee'
        >>> _slug("日本の猫")
        'image'
    """
    text = unicodedata.normalize("NFKD", prompt).encode("ascii", "ignore").decode()
    slug = "-".join(re.findall(r"[a-z0-9]+", text.lower())[:words])
    return slug[:limit].strip("-") or "image"


def _auto_output(folder: Path, prompt: str, model_name: str, seed: int) -> Path:
    """A new file in `folder`: <date>-<time>_<slug>_<model>_s<seed>.png, with -2, -3, ... if taken."""
    stem = (
        f"{datetime.now():%Y%m%d-%H%M%S}_{_slug(prompt)}_"
        f"{sanitize_fs_name(model_name)}_s{seed}"
    )
    out, n = folder / f"{stem}.png", 2
    while out.exists():
        out, n = folder / f"{stem}-{n}.png", n + 1
    return out


def _kept_server_args(
    server_args: dict[str, Any], model: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    """
    The kept server's flags: `server_args` plus the model's generation options as the web UI's defaults.

    A flag already in `server_args` wins. `--start` and image calls must build the same flags, or an image
    call would replace the server that --start loaded.
    """
    options = merge_options(
        config.get("image_generation_defaults"), model.get("generation_options")
    )
    args = dict(server_args)
    for path, flag in UI_DEFAULT_FLAGS:
        value: Any = options
        for key in path:
            value = value.get(key) if isinstance(value, dict) else None
        if value is None or flag in args:
            continue
        args[flag] = (
            ",".join(str(v) for v in value) if isinstance(value, list) else value
        )
    # Sampling takes its step count from the sigmas, but the web UI shows --steps (20 if unset).
    sigmas = options.get("sample_params", {}).get("custom_sigmas")
    if (
        sigmas
        and "--steps" not in args
        and args.get("--sigmas") == ",".join(str(v) for v in sigmas)
    ):
        args["--steps"] = len(sigmas) - 1
    return args


def _port() -> int:
    """The port of the kept server: $GENIMG_PORT, else 8190."""
    text = os.environ.get("GENIMG_PORT", str(DEFAULT_PORT))
    if not text.isdigit() or not 0 < int(text) < 65536:
        raise GenimgError(f"GENIMG_PORT must be a port number, got '{text}'")
    return int(text)


def _stop(port: int, state_dir: Path | None) -> str:
    """Stop the kept server once any running call is done; return what happened."""
    keepalive.check_supported(TOOL)
    kept = keepalive.kept_server(
        TOOL, SdCppServer, port, state_dir, keepalive.listen_host(TOOL)
    )
    with keepalive.call_lock(kept):
        was_running = keepalive.running_model(kept) is not None
        keepalive.stop_server(kept)
    return (
        "stopped the sd-server" if was_running else "no sd-server of genimg is running"
    )


def _status(port: int, state_dir: Path | None) -> str:
    """What the kept server is doing (without waiting for a running call)."""
    keepalive.check_supported(TOOL)
    return keepalive.server_status(
        keepalive.kept_server(
            TOOL, SdCppServer, port, state_dir, keepalive.listen_host(TOOL)
        )
    )


def _list(repo_root: Path, state_dir: Path | None) -> str:
    """The models as a table: name, size of their files, and `running` or `missing`."""
    config, models = _load(repo_root)
    keepalive.check_supported(TOOL)
    kept = keepalive.kept_server(
        TOOL, SdCppServer, _port(), state_dir, keepalive.listen_host(TOOL, config)
    )
    running = keepalive.running_model(kept)
    return models_table(models, {running} if running else set(), FILE_FLAGS)


def _load(
    repo_root: Path, need_binary: bool = True
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load config.yaml and image_models.yaml (and insist on `sd_server`, unless the server is remote)."""
    try:
        config = load_config(str(repo_root / "config/config.yaml"))
        models = load_image_models(str(repo_root / "config/image_models.yaml"))
    except (OSError, ValueError, TypeError, yaml.YAMLError) as e:
        raise GenimgError(f"cannot load the config: {e}") from e
    if need_binary and not config.get("sd_server"):
        raise GenimgError("config/config.yaml has no sd_server")
    return config, models


def _start(args: argparse.Namespace, repo_root: Path, state_dir: Path | None) -> str:
    """
    Make sure the kept server is running with its model loaded; return what happened.

    The server loads its model on the first request, so a tiny image is made to load it
    now: only then does the model hold its GPU memory.
    """
    config, models = _load(repo_root)
    keepalive.check_supported(TOOL)
    kept = keepalive.kept_server(
        TOOL, SdCppServer, _port(), state_dir, keepalive.listen_host(TOOL, config)
    )
    with keepalive.call_lock(kept):
        running = None if args.model else keepalive.running_model(kept)
        model = _pick_model(models, args.model or running, config, running is not None)
        warm_up = merge_options(
            config.get("image_generation_defaults"),
            model.get("generation_options"),
            {"prompt": "warm-up", "width": 64, "height": 64},
        )
        started = False
        try:
            server, started = keepalive.ensure_server(
                kept,
                config["sd_server"],
                _kept_server_args(
                    merged_server_args(config.get("image_server_args"), model),
                    model,
                    config,
                ),
                model_name=model["name"],
            )
            server.generate(prepare_request(warm_up)[0])
        except (ServerError, OSError) as e:
            # As for an image call: a dead server, or a new one that cannot load its model, is not kept.
            problem = ""
            if started or getattr(e, "kind", "") == "server_died":
                try:
                    keepalive.stop_server(kept)
                except keepalive.KeepAliveError as stop_error:
                    problem = f"; {stop_error}"
            raise GenimgError(f"{e} (server log: {kept.log}){problem}") from e
    return f"sd-server ready with {model['name']} ({'started' if started else 'already running'})"


def _generate(args: argparse.Namespace, repo_root: Path, state_dir: Path | None) -> str:
    """Run one job on the kept server; return the success line."""
    config, models = _load(repo_root)
    folder, out = _check_output(args)

    keepalive.check_supported(TOOL)
    kept = keepalive.kept_server(
        TOOL, SdCppServer, _port(), state_dir, keepalive.listen_host(TOOL, config)
    )
    out_dir = folder.resolve() / ".genimg"
    with keepalive.call_lock(kept):
        # Checked again: another call may have written this file while we waited for the lock.
        if out is not None and out.exists():
            raise GenimgError(f"{out} already exists; choose another name")
        running = None if args.model else keepalive.running_model(kept)
        model = _pick_model(models, args.model or running, config, running is not None)
        options = _job_options(args, model, config)
        # Named under the lock, so two calls cannot pick the same free name.
        if out is None:
            out = _auto_output(folder, args.prompt, model["name"], options["seed"])
        job = Job(model, options, out)
        started = False

        def open_server(
            binary: str, server_args: dict[str, Any], _log_path: Path
        ) -> SdCppServer:
            nonlocal started
            server, started = keepalive.ensure_server(
                kept,
                binary,
                _kept_server_args(server_args, model, config),
                model_name=model["name"],
            )
            job.tags["server"] = "started" if started else "warm"
            return server

        try:
            rows = run_jobs(
                [job],
                binary=config["sd_server"],
                out_dir=out_dir,
                server_defaults=config.get("image_server_args"),
                generation_defaults=config.get("image_generation_defaults"),
                open_server=open_server,
            )
        except (OSError, ValueError, subprocess.SubprocessError) as e:
            raise GenimgError(str(e)) from e

        row = rows[0]
        stop_problem = ""
        if row["status"] != "ok" and (
            row["kind"] == "server_died" or (started and row["kind"] == "job_failed")
        ):
            # A dead server, or a new one whose first job failed (usually a failed model
            # load), must not be reused by the next call.
            try:
                keepalive.stop_server(kept)
            except keepalive.KeepAliveError as e:
                stop_problem = f"; {e}"

    return _result(row, out, model, f"server log: {kept.log}", stop_problem)


def _check_output(args: argparse.Namespace) -> tuple[Path, Path | None]:
    """
    Where the image goes, after checking that the --ref images exist: (folder, file).

    The file is the -o file (which must not exist yet), or None when genimg names it (no -o, or -o an existing
    folder or a path ending in a slash): the name needs the model and seed, so it is chosen later.
    """
    for ref in args.ref:
        if not Path(ref).is_file():
            raise GenimgError(f"reference image not found: {ref}")
    if not args.output:
        return Path(), None
    out = Path(args.output)
    if out.is_dir() or args.output.endswith(("/", os.sep)):
        return out, None
    if out.exists():
        raise GenimgError(f"{out} already exists; choose another name")
    return out.parent, out


def _result(
    row: dict[str, Any],
    out: Path,
    model: dict[str, Any],
    where: str,
    problem: str = "",
) -> str:
    """The success line of a finished job; a failed one is an error that says `where` to look."""
    if row["status"] != "ok":
        detail = row["error"] or row["status"]
        if row["kind"] in ("job_failed", "server_died"):
            detail += f" ({where})"
        raise GenimgError(detail + problem)
    request = row["request"]
    parts = [f"seed {row['seed']}", f"model {model['name']}"]
    if request.get("width") and request.get("height"):
        parts.insert(0, f"{request['width']}x{request['height']}")
    return f"saved {out} ({', '.join(parts)})"


def _ask_loaded(
    models: list[dict[str, Any]], address: remote.Address
) -> tuple[SdCppServer, str | None, list[dict[str, Any]]]:
    """Attach to the remote server and ask which file it has loaded and which config entries that is."""
    server = SdCppServer.attach(address[1], host=address[0])
    loaded = remote.ask(address, server.loaded_file)
    matches = remote.models_for_file(models, loaded, MODEL_FLAGS) if loaded else []
    return server, loaded, matches


def _remote(args: argparse.Namespace, repo_root: Path, address: remote.Address) -> str:
    """
    The commands against a server on another machine ($GENIMG_URL): it is asked, never started or stopped.

    The model file the server reports as loaded decides which model entry is meant.
    """
    if args.start:
        raise remote.not_here(TOOL, "--start", "genimg --start [--model NAME]")
    if args.stop:
        raise remote.not_here(TOOL, "--stop", "genimg --stop")
    config, models = _load(repo_root, need_binary=False)
    if args.status or args.list:
        _, loaded, matches = _ask_loaded(models, address)
        if args.list:
            return models_table(models, {m["name"] for m in matches}, ())
        what = PurePosixPath(loaded).name if loaded else "a model"
        entries = (
            f"config entry: {', '.join(m['name'] for m in matches)}"
            if matches
            else (
                "not in config/image_models.yaml"
                if loaded
                else "it reports no model file"
            )
        )
        return f"running {what} at {remote.url_of(address)} (remote; {entries})"
    return _generate_remote(args, config, models, address)


def _generate_remote(
    args: argparse.Namespace,
    config: dict[str, Any],
    models: list[dict[str, Any]],
    address: remote.Address,
) -> str:
    """Run one job on the remote server; return the success line."""
    folder, out = _check_output(args)
    server, loaded, matches = _ask_loaded(models, address)
    url = remote.url_of(address)
    if args.model or loaded is None:
        model = _pick_model(models, args.model, config)
    elif matches:
        default = config.get("default_image_model")
        model = next((m for m in matches if m["name"] == default), matches[0])
    else:
        raise remote.RemoteError(
            f"the server at {url} has {PurePosixPath(loaded).name} loaded, which is not in "
            "config/image_models.yaml; pass --model, or load a model of the config on the host"
        )
    remote.check_loaded(
        model, loaded, MODEL_FLAGS, address, f"genimg --start --model {model['name']}"
    )
    options = _job_options(args, model, config)
    if out is None:
        out = _auto_output(folder, args.prompt, model["name"], options["seed"])
    job = Job(model, options, out)
    job.tags.update(server="remote", url=url)
    try:
        rows = run_jobs(
            [job],
            out_dir=folder.resolve() / ".genimg",
            server_defaults=config.get("image_server_args"),
            generation_defaults=config.get("image_generation_defaults"),
            open_server=lambda *_: server,
            engine="sd-server (remote)",
        )
    except (OSError, ValueError) as e:
        raise GenimgError(str(e)) from e
    return _result(rows[0], out, model, f"server: {url}")


def main(
    argv: list[str] | None = None,
    repo_root: Path | None = None,
    state_dir: Path | None = None,
) -> int:
    """
    Run genimg.

    Args:
        argv: Command-line arguments (default: sys.argv[1:])
        repo_root: The prompt-runner folder that holds config/ (default: the folder
            this package was installed from)
        state_dir: Where the kept server's lock, state and log files go (default: the temp folder)

    Returns:
        The exit code: 0 on success, 1 on failure (a message on stderr), 130 if interrupted
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    modes = "--start, --stop, --status and --list"
    if args.start + args.stop + args.status + args.list > 1:
        parser.error(f"choose one of {modes}")
    if args.start or args.stop or args.status or args.list:
        if args.prompt or args.output or args.seed is not None or args.size:
            parser.error(f"{modes} take no prompt, -o, --seed or --size")
        if args.ref or args.lora:
            parser.error(f"{modes} take no --ref or --lora")
        if (args.stop or args.status or args.list) and args.model:
            parser.error("--stop, --status and --list take no --model")
    elif not args.prompt:
        parser.error(
            "a prompt is required (or use --start / --stop / --status / --list)"
        )
    root = repo_root if repo_root is not None else Path(__file__).resolve().parents[2]
    try:
        address = remote.remote_address(TOOL)
        if address is not None:
            print(_remote(args, root, address))
        elif args.start:
            print(_start(args, root, state_dir))
        elif args.stop:
            print(_stop(_port(), state_dir))
        elif args.status:
            print(_status(_port(), state_dir))
        elif args.list:
            print(_list(root, state_dir))
        else:
            print(_generate(args, root, state_dir))
    except (GenimgError, keepalive.KeepAliveError, remote.RemoteError) as e:
        print(f"genimg failed: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("genimg: interrupted", file=sys.stderr)
        return 130
    return 0


if (
    __name__ == "__main__"
):  # python -m prompt_runner.genimg (what the Windows .cmd files run)
    sys.exit(main())
