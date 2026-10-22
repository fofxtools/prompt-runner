"""Utility functions for the evals project."""

import copy
import json
import re
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sanitize_fs_name(name: str) -> str:
    r"""
    Sanitize a string to be safe for use as a filesystem name (file or directory).

    Replaces problematic characters (<>:"/\|?*) with underscores to ensure
    cross-platform compatibility (Windows, macOS, Linux).

    Args:
        name: The string to sanitize

    Returns:
        A sanitized string safe for use in filesystem names

    Examples:
        >>> sanitize_fs_name("2026-01-07T12:34:56.789012Z")
        '2026-01-07T12_34_56.789012Z'
        >>> sanitize_fs_name("model/name:tag")
        'model_name_tag'
    """
    # Replace problematic characters with underscores
    # Characters: < > : " / \ | ? *
    return re.sub(r'[<>:"/\\|?*]', "_", name)


def generate_run_identifiers() -> tuple[str, str, str]:
    """
    Generate run identifiers from the current UTC timestamp with random suffix.

    Returns:
        Tuple of (run_id, run_dir_name, created_at):
        - run_id: Unique identifier with timestamp + random suffix (e.g., "2026-01-08T12:34:56Z-a3f2c1")
        - run_dir_name: Filesystem-safe format for directory names (e.g., "2026-01-08_12-34-56Z-a3f2c1")
        - created_at: ISO-8601 timestamp for sorting/filtering (e.g., "2026-01-08T12:34:56Z")

    Examples:
        >>> run_id, run_dir_name, created_at = generate_run_identifiers()
        >>> # run_id: "2026-01-08T12:34:56Z-a3f2c1"
        >>> # run_dir_name: "2026-01-08_12-34-56Z-a3f2c1"
        >>> # created_at: "2026-01-08T12:34:56Z"
    """
    now = datetime.now(timezone.utc)

    # ISO-8601 timestamp
    created_at = now.strftime("%Y-%m-%dT%H:%M:%SZ")

    # Random suffix for uniqueness (6 hex chars = 16.7M combinations)
    suffix = secrets.token_hex(3)

    # Unique run_id with timestamp + suffix
    run_id = f"{created_at}-{suffix}"

    # Filesystem-safe directory name with suffix
    run_dir_name = now.strftime("%Y-%m-%d_%H-%M-%SZ") + f"-{suffix}"

    return run_id, run_dir_name, created_at


def merge_options(*layers: dict[str, Any] | None) -> dict[str, Any]:
    """
    Recursively merge option layers, later layers winning.

    Mappings merge key by key; everything else (scalars, lists) is replaced
    wholesale. Used for request options (global defaults < model < prompt)
    and for server flags (config defaults < model). The result shares no
    state with the inputs.

    Args:
        *layers: Option dictionaries, lowest priority first (None is skipped)

    Returns:
        A new merged dictionary

    Examples:
        >>> merge_options({'width': 512}, {'width': 1024}, None)
        {'width': 1024}
        >>> merge_options(
        ...     {'sample_params': {'sample_steps': 4, 'scheduler': 'discrete'}},
        ...     {'sample_params': {'sample_steps': 20}},
        ... )
        {'sample_params': {'sample_steps': 20, 'scheduler': 'discrete'}}
    """
    merged: dict[str, Any] = {}
    for layer in layers:
        for key, value in (layer or {}).items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = merge_options(merged[key], value)
            else:
                merged[key] = copy.deepcopy(value)
    return merged


def merged_server_args(
    server_defaults: dict[str, Any] | None, model: dict[str, Any]
) -> dict[str, Any]:
    """The server flags of a model: `server_defaults` under the model's own `server_args`."""
    return merge_options(server_defaults, model["server_args"])


def random_seed() -> int:
    """A random seed that fits the 32-bit seeds of stable-diffusion.cpp and llama.cpp."""
    return secrets.randbelow(2**31 - 1)


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    """Append one JSON line to a JSONL file (created with its parent folder if needed)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def models_table(
    models: list[dict[str, Any]],
    running: set[str],
    file_flags: tuple[str, ...],
    vision_flags: tuple[str, ...] = (),
) -> str:
    """
    The models as a table: name, size of their files on disk, and `running` or `missing`.

    A model's files are the values of its `server_args` under `file_flags` (the flags that
    name a model file). One that does not exist makes the model `missing`; `running` holds
    the names of the models whose server is up. With `vision_flags` (the flags that give a
    model a vision projector) there is a fourth column, `vision`, for the models that have one.
    """
    header: tuple[str, ...] = ("NAME", "SIZE", "STATE") + (
        ("VISION",) if vision_flags else ()
    )
    rows = [header]
    for model in models:
        args = model["server_args"]
        files = [Path(str(args[flag])) for flag in file_flags if args.get(flag)]
        present = [f for f in files if f.is_file()]
        size = (
            f"{sum(f.stat().st_size for f in present) / 1e9:.1f} GB" if files else "-"
        )
        if len(present) < len(files):
            state = "missing"
        else:
            state = "running" if model["name"] in running else ""
        row: tuple[str, ...] = (model["name"], size, state)
        if vision_flags:
            row += ("vision" if any(args.get(f) for f in vision_flags) else "",)
        rows.append(row)
    width = [max(len(row[i]) for row in rows) for i in range(len(header) - 1)]
    lines = []
    for name, size, *rest in rows:
        cells = [f"{name:<{width[0]}}", f"{size:>{width[1]}}"]
        cells += [f"{cell:<{width[2 + i]}}" for i, cell in enumerate(rest[:-1])]
        cells.append(rest[-1])
        lines.append("  ".join(cells).rstrip())
    return "\n".join(lines)
