"""Unit tests for src/prompt_runner/remote.py, and genimg against a server on "another machine".

The other machine is a fake sd-server on 127.0.0.1 (see conftest.py); `GENIMG_URL` points at it, and every keepalive
function is patched to fail the test: remote mode must not start, stop, lock or remember anything.
"""

import base64
import json
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
import yaml

from prompt_runner import keepalive, remote
from prompt_runner.genimg import main
from prompt_runner.remote import RemoteError
from prompt_runner.sdcpp_server import SdCppServer
from prompt_runner.server_process import free_port

ADDRESS = ("192.168.56.1", 8190)
MODELS = [
    {"name": "a", "server_args": {"--diffusion-model": "/models/a.gguf"}},
    {"name": "b", "server_args": {"--diffusion-model": "/models/b.gguf"}},
    {"name": "shared-1", "server_args": {"--diffusion-model": "/models/shared.gguf"}},
    {"name": "shared-2", "server_args": {"--diffusion-model": "/models/shared.gguf"}},
    {"name": "allinone", "server_args": {"--model": "/models/c.safetensors"}},
    {"name": "bare", "server_args": {"--vae": "/models/vae.safetensors"}},
]
FLAGS = ("--diffusion-model", "--model", "-m")


class TestRemoteAddress:
    def test_not_set_or_empty_means_local(self, monkeypatch):
        monkeypatch.delenv("GENIMG_URL", raising=False)
        assert remote.remote_address("genimg") is None
        monkeypatch.setenv("GENIMG_URL", "")
        assert remote.remote_address("genimg") is None

    @pytest.mark.parametrize(
        ("text", "address"),
        [
            ("http://192.168.56.1:8190", ("192.168.56.1", 8190)),
            ("http://192.168.56.1:8190/", ("192.168.56.1", 8190)),
            ("http://vmhost.local:8191", ("vmhost.local", 8191)),
            ("http://127.0.0.1:1", ("127.0.0.1", 1)),
        ],
    )
    def test_http_host_and_port(self, monkeypatch, text, address):
        monkeypatch.setenv("LLM_URL", text)
        assert remote.remote_address("llm") == address
        assert remote.url_of(address).startswith("http://")

    @pytest.mark.parametrize(
        "text",
        [
            "192.168.56.1:8190",
            "http://192.168.56.1",
            "https://192.168.56.1:8190",
            "ftp://192.168.56.1:21",
            "http://192.168.56.1:8190/v1",
            "http://192.168.56.1:8190?x=1",
            "http://user@192.168.56.1:8190",
            "http://192.168.56.1:notaport",
            "http://192.168.56.1:99999",
            "http://:8190",
            "localhost",
        ],
    )
    def test_anything_else_is_refused_with_the_variable_named(self, monkeypatch, text):
        monkeypatch.setenv("GENIMG_URL", text)
        with pytest.raises(RemoteError, match="GENIMG_URL must look like http://"):
            remote.remote_address("genimg")


class TestAsk:
    def test_the_answer_is_returned(self):
        assert remote.ask(ADDRESS, lambda: 7) == 7

    @pytest.mark.parametrize(
        "error",
        [ConnectionRefusedError("refused"), OSError("x"), ValueError("bad json")],
    )
    def test_a_server_that_does_not_answer_is_one_line(self, error):
        def question() -> None:
            raise error

        with pytest.raises(
            RemoteError, match=r"cannot reach the server at http://192.168.56.1:8190: "
        ):
            remote.ask(ADDRESS, question)


class TestModelFiles:
    def test_models_for_a_file_include_every_entry_that_shares_it(self):
        names = [
            m["name"]
            for m in remote.models_for_file(MODELS, "/models/shared.gguf", FLAGS)
        ]
        assert names == ["shared-1", "shared-2"]

    def test_all_in_one_checkpoints_match_through_model(self):
        (entry,) = remote.models_for_file(MODELS, "/models/c.safetensors", FLAGS)
        assert entry["name"] == "allinone"

    def test_nothing_matches_a_file_that_is_in_no_entry(self):
        assert remote.models_for_file(MODELS, "/models/zzz.gguf", FLAGS) == []

    def test_a_loaded_file_that_is_the_models_passes(self):
        remote.check_loaded(MODELS[0], "/models/a.gguf", FLAGS, ADDRESS, "hint")

    def test_a_server_that_reports_no_file_passes(self):
        remote.check_loaded(MODELS[0], None, FLAGS, ADDRESS, "hint")

    def test_a_model_without_a_model_file_passes(self):
        remote.check_loaded(MODELS[5], "/models/a.gguf", FLAGS, ADDRESS, "hint")

    def test_another_file_names_both_and_the_command_to_load_it(self):
        with pytest.raises(RemoteError) as e:
            remote.check_loaded(
                MODELS[1], "/models/a.gguf", FLAGS, ADDRESS, "genimg --start --model b"
            )
        text = str(e.value)
        assert "http://192.168.56.1:8190 has a.gguf loaded" in text
        assert "'b' needs b.gguf" in text and text.endswith("genimg --start --model b")

    def test_not_here_says_what_to_do_on_the_host(self):
        error = remote.not_here("llm", "--stop", "llm --stop")
        assert (
            str(error)
            == "--stop is not available for a remote server (LLM_URL is set): do it on the host, with llm --stop"
        )


# ---------------------------------------------------------------- genimg over HTTP


@pytest.fixture
def no_keepalive(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("remote mode used keepalive")

    for name in (
        "ensure_server",
        "kept_server",
        "call_lock",
        "running_model",
        "stop_server",
        "check_supported",
        "server_status",
    ):
        monkeypatch.setattr(keepalive, name, boom)


def write_repo(root: Path, **config: Any) -> Path:
    """A prompt-runner folder as the VM has it: no sd_server binary in the config."""
    (root / "config").mkdir(parents=True)
    base = {
        "results_dir": "storage/results/",
        "image_generation_defaults": {"width": 64, "height": 64, "seed": -1},
        "default_image_model": "a",
        **config,
    }
    (root / "config/config.yaml").write_text(yaml.safe_dump(base))
    (root / "config/image_models.yaml").write_text(yaml.safe_dump(MODELS))
    return root


@contextmanager
def host_serving(
    fake_binary: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **flags: str
) -> Iterator[int]:
    """A fake sd-server on 127.0.0.1 (`--diffusion-model` and the like in `flags`), with GENIMG_URL pointing at it."""
    port = free_port()
    args = {f"--{name.replace('_', '-')}": value for name, value in flags.items()}
    with SdCppServer(fake_binary, args, tmp_path / "host.log", port=port, keep=False):
        monkeypatch.setenv("GENIMG_URL", f"http://127.0.0.1:{port}")
        yield port


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    return write_repo(tmp_path / "repo")


@pytest.fixture
def serving(
    fake_binary: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    no_keepalive: None,
) -> Callable[..., Any]:
    return lambda **flags: host_serving(fake_binary, tmp_path, monkeypatch, **flags)


def genimg(repo: Path, *args: str) -> int:
    return main(list(args), repo_root=repo)


def rows(out: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in (out.parent / ".genimg" / "runs.jsonl").read_text().splitlines()
    ]


class TestGenerate:
    def test_the_image_is_written_here_and_logged_beside_it(
        self, repo, tmp_path, serving, capsys
    ):
        out = tmp_path / "images" / "a.png"
        with serving(diffusion_model="/models/a.gguf") as port:
            assert genimg(repo, "a red apple", "-o", str(out)) == 0

            assert out.read_bytes() == b"fakepng"  # decoded from the server's base64
            text = capsys.readouterr().out
            assert f"saved {out}" in text and "Starting server" not in text
            (row,) = rows(out)
            assert (row["status"], row["model"]) == ("ok", "a")
            assert row["tags"] == {
                "server": "remote",
                "url": f"http://127.0.0.1:{port}",
            }
            assert row["engine"] == "sd-server (remote)" and row["server_log"] is None
            assert row["server_args"] == {
                "--diffusion-model": "/models/a.gguf"
            }  # as configured
            assert isinstance(row["seed"], int)

    def test_a_config_without_the_sd_server_binary_is_fine(
        self, repo, tmp_path, serving
    ):
        assert "sd_server" not in yaml.safe_load(
            (repo / "config/config.yaml").read_text()
        )
        with serving(diffusion_model="/models/a.gguf"):
            assert genimg(repo, "x", "-o", str(tmp_path / "a.png")) == 0

    def test_another_model_than_the_loaded_one_is_refused_with_the_command_to_load_it(
        self, repo, tmp_path, serving, capsys
    ):
        out = tmp_path / "b.png"
        with serving(diffusion_model="/models/a.gguf") as port:
            assert genimg(repo, "x", "-o", str(out), "--model", "b") == 1
        err = capsys.readouterr().err
        assert (
            f"genimg failed: the server at http://127.0.0.1:{port} has a.gguf loaded"
            in err
        )
        assert "'b' needs b.gguf" in err and "genimg --start --model b" in err
        assert not out.exists()

    def test_without_model_the_loaded_file_decides(
        self, repo, tmp_path, serving, capsys
    ):
        out = tmp_path / "b.png"
        with serving(diffusion_model="/models/b.gguf"):  # the config's default is "a"
            assert genimg(repo, "x", "-o", str(out)) == 0
        assert "model b)" in capsys.readouterr().out
        assert rows(out)[0]["model"] == "b"

    def test_entries_sharing_the_loaded_file_prefer_the_default_then_the_first(
        self, tmp_path, serving, capsys
    ):
        with serving(diffusion_model="/models/shared.gguf"):
            first = write_repo(tmp_path / "r1")
            assert genimg(first, "x", "-o", str(tmp_path / "1.png")) == 0
            second = write_repo(tmp_path / "r2", default_image_model="shared-2")
            assert genimg(second, "x", "-o", str(tmp_path / "2.png")) == 0
        models = [
            row["model"] for row in rows(tmp_path / "1.png")
        ]  # both runs log beside the images
        assert models == ["shared-1", "shared-2"]

    def test_a_loaded_file_that_is_in_no_entry_says_so(
        self, repo, tmp_path, serving, capsys
    ):
        with serving(diffusion_model="/models/zzz.gguf"):
            assert genimg(repo, "x", "-o", str(tmp_path / "x.png")) == 1
        err = capsys.readouterr().err
        assert "has zzz.gguf loaded, which is not in config/image_models.yaml" in err

    def test_a_server_that_reports_no_file_skips_the_check(
        self, repo, tmp_path, serving, capsys
    ):
        with serving():  # no --diffusion-model: capabilities has no model
            assert genimg(repo, "x", "-o", str(tmp_path / "x.png"), "--model", "b") == 0
        assert "model b)" in capsys.readouterr().out

    def test_an_all_in_one_checkpoint_matches_through_model(
        self, repo, tmp_path, serving
    ):
        with serving(model="/models/c.safetensors"):
            assert genimg(repo, "x", "-o", str(tmp_path / "x.png")) == 0
        assert rows(tmp_path / "x.png")[0]["model"] == "allinone"

    def test_reference_images_travel_as_base64(self, repo, tmp_path, serving):
        ref = tmp_path / "ref.png"
        ref.write_bytes(b"REFDATA")
        record = tmp_path / "body.json"
        with serving(diffusion_model="/models/a.gguf", record=str(record)):
            assert (
                genimg(repo, "edit", "-o", str(tmp_path / "x.png"), "--ref", str(ref))
                == 0
            )
        sent = json.loads(record.read_text())
        assert sent["ref_images"] == [base64.b64encode(b"REFDATA").decode()]

    def test_a_job_the_server_fails_is_a_failed_row_and_the_server_is_left_alone(
        self, repo, tmp_path, serving, capsys
    ):
        out = tmp_path / "x.png"
        with serving(diffusion_model="/models/a.gguf", mode="fail") as port:
            assert genimg(repo, "x", "-o", str(out)) == 1
            err = capsys.readouterr().err
            assert f"out of memory (server: http://127.0.0.1:{port})" in err
            assert rows(out)[0]["status"] == "failed" and not out.exists()
            assert SdCppServer.attach(port).ready()  # not stopped

    def test_an_existing_file_or_a_folder_is_refused_before_the_server_is_asked(
        self, repo, tmp_path, monkeypatch, no_keepalive, capsys
    ):
        monkeypatch.setenv(
            "GENIMG_URL", f"http://127.0.0.1:{free_port()}"
        )  # nobody there
        taken = tmp_path / "taken.png"
        taken.write_bytes(b"mine")
        assert genimg(repo, "x", "-o", str(taken)) == 1
        assert genimg(repo, "x", "-o", str(tmp_path)) == 1
        err = capsys.readouterr().err
        assert (
            "already exists" in err
            and "is a folder" in err
            and "cannot reach" not in err
        )
        assert taken.read_bytes() == b"mine"

    def test_a_server_that_is_not_there_is_one_line(
        self, repo, tmp_path, monkeypatch, no_keepalive, capsys
    ):
        port = free_port()
        monkeypatch.setenv("GENIMG_URL", f"http://127.0.0.1:{port}")
        assert genimg(repo, "x", "-o", str(tmp_path / "x.png")) == 1
        err = capsys.readouterr().err
        assert err.startswith(
            f"genimg failed: cannot reach the server at http://127.0.0.1:{port}: "
        )
        assert err.count("\n") == 1

    def test_a_bad_url_is_an_error_not_a_traceback(
        self, repo, tmp_path, monkeypatch, no_keepalive, capsys
    ):
        monkeypatch.setenv("GENIMG_URL", "192.168.56.1:8190")
        assert genimg(repo, "x", "-o", str(tmp_path / "x.png")) == 1
        assert (
            "genimg failed: GENIMG_URL must look like http://"
            in capsys.readouterr().err
        )


class TestCommands:
    @pytest.mark.parametrize("flag", ["--start", "--stop"])
    def test_start_and_stop_belong_to_the_host(self, repo, serving, capsys, flag):
        with serving(diffusion_model="/models/a.gguf"):
            assert genimg(repo, flag) == 1
        err = capsys.readouterr().err
        assert f"{flag} is not available for a remote server (GENIMG_URL is set)" in err
        assert "do it on the host" in err

    def test_status_names_the_file_and_the_config_entries(self, repo, serving, capsys):
        with serving(diffusion_model="/models/shared.gguf") as port:
            assert genimg(repo, "--status") == 0
        assert capsys.readouterr().out == (
            f"running shared.gguf at http://127.0.0.1:{port} (remote; config entry: shared-1, shared-2)\n"
        )

    def test_status_of_a_file_the_config_does_not_know(self, repo, serving, capsys):
        with serving(diffusion_model="/models/zzz.gguf"):
            genimg(repo, "--status")
        assert "(remote; not in config/image_models.yaml)" in capsys.readouterr().out

    def test_status_of_a_server_that_reports_no_file(self, repo, serving, capsys):
        with serving():
            genimg(repo, "--status")
        assert "running a model at http://127.0.0.1:" in capsys.readouterr().out

    def test_list_marks_the_loaded_models_and_has_no_sizes(self, repo, serving, capsys):
        with serving(diffusion_model="/models/shared.gguf"):
            assert genimg(repo, "--list") == 0
        lines = {
            line.split()[0]: line.split()
            for line in capsys.readouterr().out.splitlines()[1:]
        }
        assert lines["shared-1"] == ["shared-1", "-", "running"]
        assert lines["shared-2"] == ["shared-2", "-", "running"]
        assert lines["a"] == ["a", "-"]

    def test_status_of_a_server_that_is_not_there_fails(
        self, repo, monkeypatch, no_keepalive, capsys
    ):
        monkeypatch.setenv("GENIMG_URL", f"http://127.0.0.1:{free_port()}")
        assert genimg(repo, "--status") == 1
        assert "cannot reach the server" in capsys.readouterr().err
