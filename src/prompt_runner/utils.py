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
