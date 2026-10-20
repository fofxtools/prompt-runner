"""genimg: generate or edit one image with a prompt-runner image model."""

import argparse
import difflib
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from . import keepalive
from .config import load_config, load_image_models
from .image_runner import Job, run_jobs
from .sdcpp_server import SdCppServer
from .utils import merge_image_options

DESCRIPTION = """\
Generate an image, or edit one with --ref (the prompt then describes the edit), using a model from
config/image_models.yaml. It goes through prompt-runner's run_jobs, so settings and logs match the runner.
"""

EPILOG = """\
Linux only. The sd-server stays loaded between calls, so the next call skips the model load (about 3 s). The first call starts
it and prints its pid; a call that needs a different model (different server flags) replaces it; calls queue behind
one another. Stop it with `kill PID`; it also ends with an OpenCode session. If something else is listening on the
port (8190, or $GENIMG_PORT), genimg refuses to touch it. Files: /tmp/genimg-sd-<port>.lock, .json (what runs) and
.log (the server's output).

Model: --model, else the model the running server was last used for, else `default_image_model` in
config/config.yaml. Size, steps, CFG and the other request fields come from config/config.yaml and the model's
entry; --size overrides the size. -1 (the default) means a random seed; the seed used is printed. --lora adds to
the LoRAs the model entry already lists (file names in the server's --lora-model-dir).

The PNG is written to OUT; an existing file is never overwritten. runs.jsonl goes to <output folder>/.genimg/.
"""
DEFAULT_PORT = 8190


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
    ap.add_argument("prompt")
    ap.add_argument("-o", "--output", required=True, help="PNG file to write")
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
    options: dict[str, Any] = {"prompt": args.prompt}
    if args.seed is not None:
        options["seed"] = args.seed
    if args.size:
        options["width"], options["height"] = args.size
    if args.ref:
        # Absolute, because runs.jsonl records the paths and lives in another folder.
        options["ref_images"] = [str(Path(ref).resolve()) for ref in args.ref]
    if args.lora:
        # Lists are replaced when options are merged, so start from the model's own LoRAs.
        base = merge_image_options(
            config.get("image_generation_defaults"), model.get("generation_options")
        )
        options["lora"] = [*base.get("lora", []), *args.lora]
    return options


def _port() -> int:
    """The port of the kept server: $GENIMG_PORT, else 8190."""
    text = os.environ.get("GENIMG_PORT", str(DEFAULT_PORT))
    if not text.isdigit() or not 0 < int(text) < 65536:
        raise GenimgError(f"GENIMG_PORT must be a port number, got '{text}'")
    return int(text)


def _generate(args: argparse.Namespace, repo_root: Path, state_dir: Path | None) -> str:
    """Run one job on the kept server; return the success line."""
    try:
        config = load_config(str(repo_root / "config/config.yaml"))
        models = load_image_models(str(repo_root / "config/image_models.yaml"))
    except (OSError, ValueError, TypeError, yaml.YAMLError) as e:
        raise GenimgError(f"cannot load the config: {e}") from e
    if not config.get("sd_server"):
        raise GenimgError("config/config.yaml has no sd_server")

    out = Path(args.output)
    if out.exists():
        raise GenimgError(f"{out} already exists; choose another name")
    for ref in args.ref:
        if not Path(ref).is_file():
            raise GenimgError(f"reference image not found: {ref}")

    keepalive.check_supported()
    port = _port()
    paths = keepalive.paths_for(port, state_dir)
    out_dir = out.resolve().parent / ".genimg"
    with keepalive.call_lock(paths.lock):
        # Checked again: another call may have written this file while we waited for the lock.
        if out.exists():
            raise GenimgError(f"{out} already exists; choose another name")
        running = None if args.model else keepalive.running_model(paths.state)
        model = _pick_model(models, args.model or running, config, running is not None)
        job = Job(model, _job_options(args, model, config), out)
        started = False

        def open_server(
            binary: str, server_args: dict[str, Any], _log_path: Path
        ) -> SdCppServer:
            nonlocal started
            server, started = keepalive.ensure_server(
                binary,
                server_args,
                port=port,
                model_name=model["name"],
                paths=paths,
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
                keepalive.stop_server(paths.state)
            except keepalive.KeepAliveError as e:
                stop_problem = f"; {e}"

    if row["status"] != "ok":
        detail = row["error"] or row["status"]
        if row["kind"] in ("job_failed", "server_died"):
            detail += f" (server log: {paths.log})"
        raise GenimgError(detail + stop_problem)
    request = row["request"]
    parts = [f"seed {row['seed']}", f"model {model['name']}"]
    if request.get("width") and request.get("height"):
        parts.insert(0, f"{request['width']}x{request['height']}")
    return f"saved {args.output} ({', '.join(parts)})"


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
    args = build_parser().parse_args(argv)
    root = repo_root if repo_root is not None else Path(__file__).resolve().parents[2]
    try:
        print(_generate(args, root, state_dir))
    except (GenimgError, keepalive.KeepAliveError) as e:
        print(f"genimg failed: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("genimg: interrupted", file=sys.stderr)
        return 130
    return 0
