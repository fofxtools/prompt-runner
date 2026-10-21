"""Unit tests for the helpers in src/prompt_runner/server_process.py (the server class is tested through SdCppServer and LlamaServer)."""

import stat
import sys
from pathlib import Path

import pytest

from prompt_runner.server_process import engine_version


def script(tmp_path: Path, body: str) -> str:
    path = tmp_path / "tool"
    path.write_text(f"#!{sys.executable}\n{body}\n")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return str(path)


class TestEngineVersion:
    def test_only_the_first_line_is_kept(self, tmp_path):
        tool = script(
            tmp_path, "print('version: 0.6.0 (build 11630)')\nprint('built with Clang')"
        )
        assert engine_version(tool) == "version: 0.6.0 (build 11630)"

    def test_stderr_is_used_when_stdout_is_empty(self, tmp_path):
        tool = script(tmp_path, "import sys\nprint('sd version 1', file=sys.stderr)")
        assert engine_version(tool) == "sd version 1"

    def test_no_output_gives_an_empty_string(self, tmp_path):
        assert engine_version(script(tmp_path, "pass")) == ""

    def test_a_tool_that_fails_raises(self, tmp_path):
        with pytest.raises(Exception, match="exit status"):
            engine_version(script(tmp_path, "import sys\nsys.exit(2)"))
