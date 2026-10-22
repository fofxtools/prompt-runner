"""Unit tests for src/prompt_runner/keepalive.py, against a fake sd-server (see conftest.py).

These start real detached processes; a fixture kills any that a test leaves behind.
"""

import json
import os
import signal
import socket
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from prompt_runner import keepalive, server_process
from prompt_runner.keepalive import (
    KeepAliveError,
    KeptServer,
    call_lock,
    ensure_server,
    kept_server,
    running_model,
    stop_server,
)
from prompt_runner.sdcpp_server import SdCppServer
from prompt_runner.server_process import ServerError, free_port


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
        self.kept = kept_server("genimg", SdCppServer, self.port, tmp_path / "state")
        self.args: dict[str, Any] = {"--model": "/a.gguf"}

    def ensure(
        self, name: str = "model-a", args: dict[str, Any] | None = None
    ) -> tuple[SdCppServer, bool]:
        return ensure_server(
            self.kept,
            self.binary,
            self.args if args is None else args,
            model_name=name,
        )

    def state(self) -> dict[str, Any]:
        state = json.loads(self.kept.state.read_text())
        assert isinstance(state, dict)
        return state

    def left_behind(self, args: dict[str, Any]) -> SdCppServer:
        """What a call killed while its server starts leaves: the process and its state file."""
        server = SdCppServer(
            self.binary, args, self.kept.log, port=self.port, keep=True
        )
        server.spawn()
        # Popen returns before exec has filled in /proc/<pid>/cmdline, which is how keepalive
        # recognizes its own process; a real call is long gone by the time the next one looks.
        while not keepalive._cmdline(server.pid or 0):
            time.sleep(0.001)
        keepalive._write_state(
            self.kept.state,
            {"pid": server.pid, "argv": server.argv, "model_name": "model-a"},
        )
        return server


@pytest.fixture
def env(fake_binary, tmp_path, monkeypatch):
    monkeypatch.setattr(server_process, "POLL_INTERVAL", 0.05)
    monkeypatch.setattr(keepalive, "POLL_INTERVAL", 0.05)
    yield Env(fake_binary, tmp_path)
    for pid in fakes_running(fake_binary):
        os.kill(pid, signal.SIGKILL)


class TestKeptServer:
    def test_files_are_named_after_the_tool_and_port_in_a_created_folder(
        self, tmp_path
    ):
        kept = kept_server("genimg", SdCppServer, 8190, tmp_path / "new")
        assert kept.lock.name == "genimg-8190.lock"
        assert kept.state.name == "genimg-8190.json"
        assert kept.log.name == "genimg-8190.log"
        assert kept.lock.parent.is_dir()
        other = kept_server("llm", SdCppServer, 8191, tmp_path / "new")
        assert other.state.name == "llm-8191.json"

    def test_messages_name_the_tool_and_the_server(self, env, tmp_path, capsys):
        class Renamed(SdCppServer):
            NAME = "fake-llama"

        kept = kept_server("llm", Renamed, env.port, tmp_path / "state")
        ensure_server(kept, env.binary, env.args, model_name="m")
        ensure_server(kept, env.binary, env.args, model_name="m")
        out = capsys.readouterr().out
        assert "started the fake-llama (pid" in out and "next llm call" in out
        assert "reusing the fake-llama already running" in out

    def test_a_foreign_listener_message_names_the_tool_and_its_port_variable(
        self, env, tmp_path
    ):
        other = SdCppServer(
            env.binary, {}, tmp_path / "other.log", port=env.port, keep=True
        )
        other.start()
        try:
            kept = kept_server("llm", SdCppServer, env.port, tmp_path / "state")
            with pytest.raises(
                KeepAliveError,
                match="this llm did not start.*manual sd-server.*LLM_PORT",
            ):
                ensure_server(kept, env.binary, env.args, model_name="m")
        finally:
            other.stop()


class TestEnsureServer:
    def test_starts_a_detached_server_and_records_it(self, env, capsys):
        with call_lock(env.kept):
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
            assert str(env.kept.lock) not in fds
        assert f"started the sd-server (pid {pid}" in capsys.readouterr().out
        assert not gone(pid)  # it outlives the call

    def test_say_receives_the_progress_messages_instead_of_print(self, env, capsys):
        said: list[str] = []
        ensure_server(env.kept, env.binary, env.args, model_name="a", say=said.append)
        ensure_server(env.kept, env.binary, env.args, model_name="a", say=said.append)
        assert capsys.readouterr().out == ""
        assert said[0].startswith("started the sd-server")
        assert said[1].startswith("reusing the sd-server already running")

    def test_the_next_call_reuses_it_and_updates_the_model_name(self, env, capsys):
        _, started = env.ensure("model-a")
        pid = env.state()["pid"]
        server, started_again = env.ensure("model-b")
        assert started and not started_again
        assert server.pid is None  # attached
        assert server.generate({"prompt": "x"}) == [b"fakepng"]
        assert env.state()["pid"] == pid
        assert running_model(env.kept) == "model-b"
        assert "reusing the sd-server already running" in capsys.readouterr().out

    def test_different_server_args_replace_it(self, env, capsys):
        env.ensure()
        old = env.state()["pid"]
        server, started = env.ensure("model-b", {"--model": "/b.gguf"})
        assert started
        assert gone(old)
        assert env.state()["pid"] == server.pid != old
        assert running_model(env.kept) == "model-b"
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
            assert not env.kept.state.exists()
        finally:
            other.stop()

    def test_a_stale_state_file_is_ignored(self, env):
        dead = subprocess.Popen(["true"])
        dead.wait()
        env.kept.state.write_text(
            json.dumps({"pid": dead.pid, "argv": ["x"], "model_name": "old"})
        )
        assert running_model(env.kept) is None
        server, started = env.ensure()
        assert started
        assert env.state()["pid"] == server.pid

    def test_a_reused_pid_is_never_signalled(self, env):
        sleeper = subprocess.Popen(["sleep", "60"])
        try:
            env.kept.state.write_text(
                json.dumps(
                    {"pid": sleeper.pid, "argv": [env.binary, "x"], "model_name": "m"}
                )
            )
            server, started = env.ensure()
            assert started and server.pid != sleeper.pid
            assert sleeper.poll() is None

            env.kept.state.write_text(
                json.dumps(
                    {"pid": sleeper.pid, "argv": ["sleep", "999"], "model_name": "m"}
                )
            )
            stop_server(env.kept)
            assert sleeper.poll() is None
            assert not env.kept.state.exists()
        finally:
            sleeper.kill()
            sleeper.wait()

    def test_a_call_killed_mid_start_does_not_orphan_the_server(self, env):
        args = {"--model": "/a.gguf", "--delay": "1.0"}
        starting = env.left_behind(args)

        server, started = env.ensure(args=args)

        assert not started
        assert server.ready()
        assert fakes_running(env.binary) == [starting.pid]

    def test_the_state_is_written_before_waiting_for_the_server(self, env, monkeypatch):
        seen: list[bool] = []
        real = SdCppServer.wait_ready

        def spy(server: SdCppServer) -> None:
            seen.append(env.kept.state.exists())
            real(server)

        monkeypatch.setattr(SdCppServer, "wait_ready", spy)
        env.ensure()
        assert seen == [True]

    def test_a_server_that_never_answers_is_replaced(self, env, monkeypatch, tmp_path):
        monkeypatch.setattr(SdCppServer, "STARTUP_TIMEOUT", 0.3)
        delay = tmp_path / "delay"
        delay.write_text("1.5")  # the first start is slow, the replacement is not
        args = {"--model": "/a.gguf", "--delay-file": str(delay)}
        hung = env.left_behind(args)

        server, started = env.ensure(args=args)

        assert started
        assert hung.pid is not None and gone(hung.pid)
        assert server.pid != hung.pid

    def test_a_failed_start_reports_the_log_tail_and_leaves_no_state(self, env):
        with pytest.raises(ServerError) as info:
            env.ensure(args={"--model": "/a.gguf", "--mode": "exit"})
        assert info.value.kind == "server_died"
        assert info.value.exit_code == 3
        assert "fatal: out of memory" in info.value.message
        assert not env.kept.state.exists()
        assert fakes_running(env.binary) == []

    def test_a_missing_binary_raises_oserror_and_leaves_no_state(self, env, tmp_path):
        with pytest.raises(FileNotFoundError):
            ensure_server(env.kept, str(tmp_path / "nope"), {}, model_name="m")
        assert not env.kept.state.exists()


class TestStopAndRunningModel:
    def test_stop_server_stops_ours_and_forgets_it(self, env):
        env.ensure("model-a")
        pid = env.state()["pid"]
        assert running_model(env.kept) == "model-a"

        stop_server(env.kept)

        assert gone(pid)
        assert not env.kept.state.exists()
        assert running_model(env.kept) is None
        stop_server(env.kept)  # nothing left to stop: no error

    def test_running_model_is_none_without_a_state_file(self, env):
        assert running_model(env.kept) is None

    def test_a_corrupt_state_file_counts_as_no_state(self, env):
        env.kept.state.write_text("{not json")
        assert running_model(env.kept) is None
        server, started = env.ensure()
        assert started and env.state()["pid"] == server.pid


class TestListenHost:
    def test_the_default_is_loopback(self, monkeypatch):
        monkeypatch.delenv("GENIMG_HOST", raising=False)
        assert keepalive.listen_host("genimg") == "127.0.0.1"
        assert keepalive.listen_host("genimg", {"results_dir": "x"}) == "127.0.0.1"

    def test_the_config_is_used_for_every_tool(self, monkeypatch):
        monkeypatch.delenv("GENIMG_HOST", raising=False)
        monkeypatch.delenv("LLM_HOST", raising=False)
        config = {"listen_host": "0.0.0.0"}
        assert keepalive.listen_host("genimg", config) == "0.0.0.0"
        assert keepalive.listen_host("llm", config) == "0.0.0.0"

    def test_the_tools_own_variable_wins_over_the_config(self, monkeypatch):
        monkeypatch.setenv("LLM_HOST", "192.168.56.1")
        monkeypatch.setenv("GENIMG_HOST", "127.0.0.2")
        config = {"listen_host": "0.0.0.0"}
        assert keepalive.listen_host("llm", config) == "192.168.56.1"
        assert keepalive.listen_host("genimg", config) == "127.0.0.2"
        assert keepalive.listen_host("llm") == "192.168.56.1"  # without a config too

    def test_another_tools_variable_does_not_count(self, monkeypatch):
        monkeypatch.setenv("LLM_HOST", "192.168.56.1")
        monkeypatch.delenv("GENIMG_HOST", raising=False)
        assert keepalive.listen_host("genimg", {"listen_host": "0.0.0.0"}) == "0.0.0.0"

    @pytest.mark.parametrize("text", ["localhost", "::1", "300.1.1.1", "", "1.2.3"])
    def test_anything_but_an_ipv4_address_in_the_variable_is_refused(
        self, monkeypatch, text
    ):
        monkeypatch.setenv("GENIMG_HOST", text)
        with pytest.raises(KeepAliveError, match="GENIMG_HOST must be an IPv4 address"):
            keepalive.listen_host("genimg", {"listen_host": "0.0.0.0"})

    @pytest.mark.parametrize(
        "value", ["localhost", "::1", 8080, ["0.0.0.0"], None, True]
    )
    def test_anything_but_an_ipv4_address_in_the_config_is_refused(
        self, monkeypatch, value
    ):
        monkeypatch.delenv("GENIMG_HOST", raising=False)
        with pytest.raises(
            KeepAliveError,
            match="listen_host in config/config.yaml must be an IPv4 address",
        ):
            keepalive.listen_host("genimg", {"listen_host": value})


class TestHost:
    """A kept server on another address (127.0.0.2 is loopback too, so no network is needed)."""

    def other(self, env: Env, host: str = "127.0.0.2") -> KeptServer[SdCppServer]:
        return kept_server("genimg", SdCppServer, env.port, env.kept.state.parent, host)

    def test_it_listens_there_and_the_state_remembers_the_address(self, env):
        kept = self.other(env)
        server, started = ensure_server(kept, env.binary, env.args, model_name="m")
        assert started and server.ready()
        assert json.loads(kept.state.read_text())["host"] == "127.0.0.2"
        assert keepalive._port_in_use("127.0.0.2", env.port)
        assert not keepalive._port_in_use("127.0.0.1", env.port)

    def test_the_next_call_reuses_it(self, env):
        kept = self.other(env)
        ensure_server(kept, env.binary, env.args, model_name="m")
        _, started = ensure_server(kept, env.binary, env.args, model_name="m")
        assert not started

    def test_another_address_replaces_it_and_the_old_port_is_waited_for(self, env):
        first = self.other(env)
        ensure_server(first, env.binary, env.args, model_name="m")
        old_pid = json.loads(first.state.read_text())["pid"]

        _, started = ensure_server(env.kept, env.binary, env.args, model_name="m")

        assert started and gone(old_pid)
        assert json.loads(env.kept.state.read_text())["host"] == "127.0.0.1"
        assert not keepalive._port_in_use("127.0.0.2", env.port)

    def test_stopping_uses_the_address_the_server_was_started_with(self, env):
        ensure_server(self.other(env), env.binary, env.args, model_name="m")
        pid = env.state()["pid"]
        stop_server(
            env.kept
        )  # this kept server says 127.0.0.1; the state says 127.0.0.2
        assert gone(pid) and not keepalive._port_in_use("127.0.0.2", env.port)

    def test_status_finds_it_through_the_state_whatever_the_callers_address(self, env):
        ensure_server(self.other(env), env.binary, env.args, model_name="m")
        line = keepalive.server_status(env.kept)  # no GENIMG_HOST here
        assert line.startswith("running m (pid ") and f"on 127.0.0.2:{env.port}" in line

    def test_a_foreign_listener_is_refused_on_that_address_only(self, env):
        kept = self.other(env)
        with socket.socket() as other:
            other.bind(("127.0.0.2", env.port))
            other.listen()
            with pytest.raises(KeepAliveError, match=f"127.0.0.2:{env.port}"):
                ensure_server(kept, env.binary, env.args, model_name="m")
            assert keepalive.server_status(kept).startswith("127.0.0.2:")


class TestServerStatus:
    def test_a_running_server_says_model_pid_port_and_log(self, env):
        env.ensure("model-a")
        line = keepalive.server_status(env.kept)
        assert line == (
            f"running model-a (pid {env.state()['pid']}) on 127.0.0.1:{env.port}; "
            f"log {env.kept.log}"
        )

    def test_nothing_running_and_the_port_free(self, env):
        assert keepalive.server_status(env.kept) == (
            f"not running (127.0.0.1:{env.port} is free)"
        )

    def test_a_stopped_server_is_not_running(self, env):
        env.ensure()
        stop_server(env.kept)
        assert keepalive.server_status(env.kept).startswith("not running")

    def test_something_else_on_the_port_is_named_as_such(self, env):
        with socket.socket() as other:
            other.bind(("127.0.0.1", env.port))
            other.listen()
            assert keepalive.server_status(env.kept) == (
                f"127.0.0.1:{env.port} is used by something else (not started by genimg)"
            )

    def test_our_process_that_does_not_answer_is_reported(self, env, monkeypatch):
        env.ensure()
        monkeypatch.setattr(SdCppServer, "ready", lambda self: False)
        assert keepalive.server_status(env.kept).startswith("not answering")

    def test_it_does_not_wait_for_the_lock(self, env):
        env.ensure()
        with call_lock(env.kept):
            assert keepalive.server_status(env.kept).startswith("running")


class TestStopFailures:
    def test_check_supported_needs_proc(self, tmp_path):
        keepalive.check_supported("genimg")  # this machine is Linux
        with pytest.raises(KeepAliveError, match="llm keeps .* needs Linux"):
            keepalive.check_supported("llm", tmp_path / "no-proc")

    def test_a_server_that_survives_is_reported_and_its_state_kept(
        self, env, monkeypatch
    ):
        env.ensure()
        monkeypatch.setattr(keepalive, "STOP_TIMEOUT", 0.3)
        monkeypatch.setattr(os, "killpg", lambda pid, sig: None)  # signals are lost

        with pytest.raises(KeepAliveError, match="did not stop"):
            stop_server(env.kept)

        assert env.kept.state.exists()  # still ours, so the next call knows about it
        assert running_model(env.kept) == "model-a"

    def test_a_permission_error_is_reported_not_raised_raw(self, env, monkeypatch):
        env.ensure()

        def deny(pid: int, sig: int) -> None:
            raise PermissionError("not permitted")

        monkeypatch.setattr(os, "killpg", deny)
        with pytest.raises(KeepAliveError, match="cannot stop"):
            stop_server(env.kept)
        assert env.kept.state.exists()

    def test_replacing_a_server_that_will_not_stop_is_refused(self, env, monkeypatch):
        env.ensure()
        monkeypatch.setattr(keepalive, "STOP_TIMEOUT", 0.3)
        monkeypatch.setattr(os, "killpg", lambda pid, sig: None)
        with pytest.raises(KeepAliveError, match="did not stop"):
            env.ensure("model-b", {"--model": "/b.gguf"})


class TestCallLock:
    def test_a_second_call_waits_then_gives_up(self, tmp_path, capsys):
        kept = kept_server("llm", SdCppServer, 8191, tmp_path)
        with (
            call_lock(kept),
            pytest.raises(KeepAliveError, match="another llm call held"),
            call_lock(kept, wait=0.3),
        ):
            pass
        assert "waiting for another llm call" in capsys.readouterr().err

    def test_the_lock_is_free_again_after_the_call(self, tmp_path):
        kept = kept_server("genimg", SdCppServer, 8190, tmp_path)
        with call_lock(kept):
            pass
        with call_lock(kept, wait=0.3):
            pass

    def test_a_waiting_call_proceeds_when_the_holder_finishes(self, tmp_path):
        kept = kept_server("genimg", SdCppServer, 8190, tmp_path)
        order: list[str] = []
        holding = threading.Event()

        def holder() -> None:
            with call_lock(kept):
                holding.set()
                order.append("first")
                threading.Event().wait(0.3)

        thread = threading.Thread(target=holder)
        thread.start()
        holding.wait()
        with call_lock(kept, wait=5):
            order.append("second")
        thread.join()
        assert order == ["first", "second"]
