"""Utility functions for the evals project."""

import os
import re
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


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


def create_result_structure(run_dir_name: str, results_dir: str) -> Path:
    """
    Create the result directory structure for a run.

    Args:
        run_dir_name: The filesystem-safe run directory name
        results_dir: The base results directory path

    Returns:
        Path object for the created run directory

    Raises:
        FileExistsError: If the run directory already exists
    """
    results_path = Path(results_dir)
    run_path = results_path / run_dir_name

    if run_path.exists():
        raise FileExistsError(f"Run directory already exists: {run_path}")

    # Create the directory structure
    run_path.mkdir(parents=True)

    return run_path


def merge_options(
    global_defaults: Dict[str, Any] | None,
    model_options: Dict[str, Any] | None,
    prompt_options: Dict[str, Any] | None,
) -> Dict[str, Any]:
    """
    Merge global defaults, model-level and prompt-level options.

    Priority (lowest to highest):
    1. Global defaults from config
    2. Model-level options
    3. Prompt-level options (highest priority)

    Args:
        global_defaults: Global generation defaults from config (can be None)
        model_options: Options from the model configuration (can be None)
        prompt_options: Options from the prompt configuration (can be None)

    Returns:
        Merged options dictionary (passed verbatim to ollama)
    """
    merged = {}

    if global_defaults:
        merged.update(global_defaults)

    if model_options:
        merged.update(model_options)

    if prompt_options:
        merged.update(prompt_options)

    return merged


def expand_env_vars(value: Any) -> Any:
    """
    Recursively expand environment variables in strings, dicts, and lists.

    Expands ${VAR_NAME} patterns in strings using os.path.expandvars().
    Recursively processes dictionaries and lists.

    Args:
        value: The value to process (can be str, dict, list, or other types)

    Returns:
        The value with environment variables expanded

    Examples:
        >>> os.environ['HOME'] = '/home/user'
        >>> expand_env_vars('${HOME}/models')
        '/home/user/models'
        >>> expand_env_vars({'path': '${HOME}/models', 'name': 'test'})
        {'path': '/home/user/models', 'name': 'test'}
        >>> expand_env_vars(['${HOME}/a', '${HOME}/b'])
        ['/home/user/a', '/home/user/b']
    """
    if isinstance(value, str):
        return os.path.expandvars(value)
    elif isinstance(value, dict):
        return {k: expand_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [expand_env_vars(item) for item in value]
    else:
        return value


def merge_image_options(
    global_defaults: Dict[str, Any] | None,
    model_options: Dict[str, Any] | None,
    prompt_options: Dict[str, Any] | None,
) -> Dict[str, Any]:
    """
    Merge global defaults, model-level and prompt-level options for image generation.

    Priority (lowest to highest):
    1. Global defaults from config.yaml (image_generation_defaults)
    2. Model-level generation options (passed verbatim)
    3. Prompt-level options (overwrite on conflict)

    Note: This function blindly merges generation options.
    Init-only options (e.g. vae_decode_only, keep_clip_on_cpu)
    are applied at model initialization time and cannot be
    changed per generation call.

    Args:
        global_defaults: Global image generation defaults from config (can be None)
        model_options: Generation options from the model configuration (can be None)
        prompt_options: Options from the prompt configuration (can be None)

    Returns:
        Merged options dictionary for image generation

    Examples:
        >>> merge_image_options({'width': 512}, {'cfg_scale': 1.0}, {'seed': 42})
        {'width': 512, 'cfg_scale': 1.0, 'seed': 42}
        >>> merge_image_options({'width': 512}, {'width': 1024}, None)
        {'width': 1024}
    """
    merged = {}

    if global_defaults:
        merged.update(global_defaults)

    if model_options:
        merged.update(model_options)

    if prompt_options:
        merged.update(prompt_options)

    return merged
