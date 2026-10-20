"""Unit tests for src/prompt_runner/keepalive.py, against a fake sd-server (see conftest.py).

These start real detached processes; a fixture kills any that a test leaves behind.
"""

import json
import os
import signal
import subprocess
import threading
from pathlib import Path
from typing import Any

import pytest

from prompt_runner import keepalive, sdcpp_server
from prompt_runner.keepalive import (
    KeepAliveError,
    call_lock,
    ensure_server,
    paths_for,
    running_model,
    stop_server,
)
from prompt_runner.sdcpp_server import SdCppError, SdCppServer, free_port


def gone(pid: int) -> bool:
    """Whether the process is gone (a zombie counts as gone, as in keepalive)."""
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes() == b""
    except OSError:
        return True


def fakes_running(binary: str) -> list[int]:
    pids = []
    for entry in Path("/proc").iterdir():
        if entry.name.isdigit():
            try:
                raw = (entry / "cmdline").read_bytes().split(b"\0")
            except OSError:
                continue
            if binary.encode() in raw[:2]:  # a script runs behind its interpreter
                pids.append(int(entry.name))
    return pids


class Env:
    def __init__(self, binary: str, tmp_path: Path) -> None:
        self.binary = binary
        self.port = free_port()
        self.paths = paths_for(self.port, tmp_path / "state")
        self.args: dict[str, Any] = {"--model": "/a.gguf"}

    def ensure(
        self, name: str = "model-a", args: dict[str, Any] | None = None
    ) -> tuple[SdCppServer, bool]:
        return ensure_server(
            self.binary,
            self.args if args is None else args,
            port=self.port,
            model_name=name,
            paths=self.paths,
        )

    def state(self) -> dict[str, Any]:
        state = json.loads(self.paths.state.read_text())
        assert isinstance(state, dict)
        return state


@pytest.fixture
def env(fake_binary, tmp_path, monkeypatch):
    monkeypatch.setattr(sdcpp_server, "POLL_INTERVAL", 0.05)
    monkeypatch.setattr(keepalive, "POLL_INTERVAL", 0.05)
    yield Env(fake_binary, tmp_path)
    for pid in fakes_running(fake_binary):
        os.kill(pid, signal.SIGKILL)


class TestPaths:
    def test_files_are_keyed_by_port_in_a_created_folder(self, tmp_path):
        paths = paths_for(8190, tmp_path / "new")
        assert paths.lock.name == "genimg-sd-8190.lock"
        assert paths.state.name == "genimg-sd-8190.json"
        assert paths.log.name == "genimg-sd-8190.log"
        assert paths.lock.parent.is_dir()


class TestEnsureServer:
    def test_starts_a_detached_server_and_records_it(self, env, capsys):
        with call_lock(env.paths.lock):
            server, started = env.ensure()
            assert started
            assert server.ready()
            pid = server.pid
            assert pid is not None
            state = env.state()
            assert state["pid"] == pid
            assert state["model_name"] == "model-a"
            assert state["argv"][0] == env.binary
            assert state["argv"][-2:] == ["--listen-port", str(env.port)]
            assert os.getsid(pid) == pid  # its own session
            assert os.readlink(f"/proc/{pid}/fd/0") == "/dev/null"
            # The lock file must not be inherited, or the lock would outlive the call.
            fds = [
                os.readlink(f"/proc/{pid}/fd/{fd}")
                for fd in os.listdir(f"/proc/{pid}/fd")
            ]
            assert str(env.paths.lock) not in fds
        assert f"started an sd-server (pid {pid}" in capsys.readouterr().out
        assert not gone(pid)  # it outlives the call

    def test_the_next_call_reuses_it_and_updates_the_model_name(self, env, capsys):
        _, started = env.ensure("model-a")
        pid = env.state()["pid"]
        server, started_again = env.ensure("model-b")
        assert started and not started_again
        assert server.pid is None  # attached
        assert server.generate({"prompt": "x"}) == [b"fakepng"]
        assert env.state()["pid"] == pid
        assert running_model(env.paths.state) == "model-b"
        assert "reusing the sd-server already running" in capsys.readouterr().out

    def test_different_server_args_replace_it(self, env, capsys):
        env.ensure()
        old = env.state()["pid"]
        server, started = env.ensure("model-b", {"--model": "/b.gguf"})
        assert started
        assert gone(old)
        assert env.state()["pid"] == server.pid != old
        assert running_model(env.paths.state) == "model-b"
        assert "stopping the running sd-server" in capsys.readouterr().out

    def test_something_else_on_the_port_is_refused_and_left_alone(self, env, tmp_path):
        other = SdCppServer(
            env.binary, {}, tmp_path / "other.log", port=env.port, keep=True
        )
        other.start()
        try:
            with pytest.raises(KeepAliveError, match="already listening"):
                env.ensure()
            assert other.pid is not None and not gone(other.pid)
            assert not env.paths.state.exists()
        finally:
            other.stop()

    def test_a_stale_state_file_is_ignored(self, env):
        dead = subprocess.Popen(["true"])
        dead.wait()
        env.paths.state.write_text(
            json.dumps({"pid": dead.pid, "argv": ["x"], "model_name": "old"})
        )
        assert running_model(env.paths.state) is None
        server, started = env.ensure()
        assert started
        assert env.state()["pid"] == server.pid

    def test_a_reused_pid_is_never_signalled(self, env):
        sleeper = subprocess.Popen(["sleep", "60"])
        try:
            env.paths.state.write_text(
                json.dumps(
                    {"pid": sleeper.pid, "argv": [env.binary, "x"], "model_name": "m"}
                )
            )
            server, started = env.ensure()
            assert started and server.pid != sleeper.pid
            assert sleeper.poll() is None

            env.paths.state.write_text(
                json.dumps(
                    {"pid": sleeper.pid, "argv": ["sleep", "999"], "model_name": "m"}
                )
            )
            stop_server(env.paths.state)
            assert sleeper.poll() is None
            assert not env.paths.state.exists()
        finally:
            sleeper.kill()
            sleeper.wait()

    def test_a_call_killed_mid_start_does_not_orphan_the_server(self, env):
        args = {"--model": "/a.gguf", "--delay": "1.0"}
        starting = SdCppServer(
            env.binary, args, env.paths.log, port=env.port, keep=True
        )
        starting.spawn()  # what a killed call leaves behind: a process and its state
        keepalive._write_state(
            env.paths.state,
            {"pid": starting.pid, "argv": starting.argv, "model_name": "model-a"},
        )

        server, started = env.ensure(args=args)

        assert not started
        assert server.ready()
        assert fakes_running(env.binary) == [starting.pid]

    def test_the_state_is_written_before_waiting_for_the_server(self, env, monkeypatch):
        seen: list[bool] = []
        real = SdCppServer.wait_ready

        def spy(server: SdCppServer) -> None:
            seen.append(env.paths.state.exists())
            real(server)

        monkeypatch.setattr(SdCppServer, "wait_ready", spy)
        env.ensure()
        assert seen == [True]

    def test_a_server_that_never_answers_is_replaced(self, env, monkeypatch):
        monkeypatch.setattr(keepalive, "STARTUP_TIMEOUT", 0.3)
        args = {"--model": "/a.gguf", "--delay": "1.5"}
        hung = SdCppServer(env.binary, args, env.paths.log, port=env.port, keep=True)
        hung.spawn()
        keepalive._write_state(
            env.paths.state,
            {"pid": hung.pid, "argv": hung.argv, "model_name": "model-a"},
        )

        server, started = env.ensure(args=args)

        assert started
        assert hung.pid is not None and gone(hung.pid)
        assert server.pid != hung.pid

    def test_a_failed_start_reports_the_log_tail_and_leaves_no_state(self, env):
        with pytest.raises(SdCppError) as info:
            env.ensure(args={"--model": "/a.gguf", "--mode": "exit"})
        assert info.value.kind == "server_died"
        assert info.value.exit_code == 3
        assert "fatal: out of memory" in info.value.message
        assert not env.paths.state.exists()
        assert fakes_running(env.binary) == []

    def test_a_missing_binary_raises_oserror_and_leaves_no_state(self, env, tmp_path):
        with pytest.raises(FileNotFoundError):
            ensure_server(
                str(tmp_path / "nope"),
                {},
                port=env.port,
                model_name="m",
                paths=env.paths,
            )
        assert not env.paths.state.exists()


class TestStopAndRunningModel:
    def test_stop_server_stops_ours_and_forgets_it(self, env):
        env.ensure("model-a")
        pid = env.state()["pid"]
        assert running_model(env.paths.state) == "model-a"

        stop_server(env.paths.state)

        assert gone(pid)
        assert not env.paths.state.exists()
        assert running_model(env.paths.state) is None
        stop_server(env.paths.state)  # nothing left to stop: no error

    def test_running_model_is_none_without_a_state_file(self, env):
        assert running_model(env.paths.state) is None

    def test_a_corrupt_state_file_counts_as_no_state(self, env):
        env.paths.state.write_text("{not json")
        assert running_model(env.paths.state) is None
        server, started = env.ensure()
        assert started and env.state()["pid"] == server.pid


class TestStopFailures:
    def test_check_supported_needs_proc(self, tmp_path):
        keepalive.check_supported()  # this machine is Linux
        with pytest.raises(KeepAliveError, match="needs Linux"):
            keepalive.check_supported(tmp_path / "no-proc")

    def test_a_server_that_survives_is_reported_and_its_state_kept(
        self, env, monkeypatch
    ):
        env.ensure()
        monkeypatch.setattr(keepalive, "STOP_TIMEOUT", 0.3)
        monkeypatch.setattr(os, "killpg", lambda pid, sig: None)  # signals are lost

        with pytest.raises(KeepAliveError, match="did not stop"):
            stop_server(env.paths.state)

        assert env.paths.state.exists()  # still ours, so the next call knows about it
        assert running_model(env.paths.state) == "model-a"

    def test_a_permission_error_is_reported_not_raised_raw(self, env, monkeypatch):
        env.ensure()

        def deny(pid: int, sig: int) -> None:
            raise PermissionError("not permitted")

        monkeypatch.setattr(os, "killpg", deny)
        with pytest.raises(KeepAliveError, match="cannot stop"):
            stop_server(env.paths.state)
        assert env.paths.state.exists()

    def test_replacing_a_server_that_will_not_stop_is_refused(self, env, monkeypatch):
        env.ensure()
        monkeypatch.setattr(keepalive, "STOP_TIMEOUT", 0.3)
        monkeypatch.setattr(os, "killpg", lambda pid, sig: None)
        with pytest.raises(KeepAliveError, match="did not stop"):
            env.ensure("model-b", {"--model": "/b.gguf"})


class TestCallLock:
    def test_a_second_call_waits_then_gives_up(self, tmp_path, capsys):
        path = tmp_path / "x.lock"
        with (
            call_lock(path),
            pytest.raises(KeepAliveError, match="held"),
            call_lock(path, wait=0.3),
        ):
            pass
        assert "waiting for another genimg call" in capsys.readouterr().err

    def test_the_lock_is_free_again_after_the_call(self, tmp_path):
        path = tmp_path / "x.lock"
        with call_lock(path):
            pass
        with call_lock(path, wait=0.3):
            pass

    def test_a_waiting_call_proceeds_when_the_holder_finishes(self, tmp_path):
        path = tmp_path / "x.lock"
        order: list[str] = []
        holding = threading.Event()

        def holder() -> None:
            with call_lock(path):
                holding.set()
                order.append("first")
                threading.Event().wait(0.3)

        thread = threading.Thread(target=holder)
        thread.start()
        holding.wait()
        with call_lock(path, wait=5):
            order.append("second")
        thread.join()
        assert order == ["first", "second"]
