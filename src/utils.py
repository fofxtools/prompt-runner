"""Utility functions for the evals project."""

import re


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
