"""Windows has no `fcntl`: the modules the remote client uses must import without it."""

import subprocess
import sys
import tempfile

import pytest

MODULES = [
    "prompt_runner.keepalive",
    "prompt_runner.genimg",
    "prompt_runner.llm",
    "prompt_runner.remote",
]


@pytest.mark.parametrize("module", MODULES)
def test_imports_without_fcntl(module: str) -> None:
    code = f"import sys; sys.modules['fcntl'] = None; import {module}"  # None makes an import raise
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


def test_the_lock_itself_needs_fcntl() -> None:
    """Nothing pretends the lock works there: the remote client must not take it."""
    code = (
        "import sys; sys.modules['fcntl'] = None\n"
        "from pathlib import Path\n"
        "from prompt_runner import keepalive\n"
        "from prompt_runner.sdcpp_server import SdCppServer\n"
        "kept = keepalive.kept_server('x', SdCppServer, 1, Path(sys.argv[1]))\n"
        "with keepalive.call_lock(kept): pass\n"
    )
    with tempfile.TemporaryDirectory() as folder:
        result = subprocess.run(
            [sys.executable, "-c", code, folder],
            capture_output=True,
            text=True,
            check=False,
        )
    assert result.returncode != 0 and "fcntl" in result.stderr
