"""Real sd-server and llama server smoke tests. Need a GPU, the binaries and the model files.

Skipped by default; run with: pytest -m gpu
"""

import json
import shutil
from pathlib import Path

import pytest

from prompt_runner.config import load_config, load_image_models
from prompt_runner.image_runner import Job, run_jobs
from prompt_runner.llama_server import LlamaServer

pytestmark = pytest.mark.gpu

REPO = Path(__file__).resolve().parent.parent
QWEN_4B = Path(
    "/home/shared/ai/vlm/Qwen3.5-4B-Uncensored-HauhauCS-Aggressive-Q6_K.gguf"
)


def test_klein_4b_generates_one_image(tmp_path):
    config = load_config(str(REPO / "config/config.yaml"))
    (model,) = [
        m
        for m in load_image_models(str(REPO / "config/image_models.yaml"))
        if m["name"] == "flux-2-klein-4b-Q8_0"
    ]
    job = Job(
        model, {"prompt": "a pizza on a sunny beach", "seed": 1}, tmp_path / "pizza.png"
    )

    rows = run_jobs(
        [job],
        binary=config["sd_server"],
        out_dir=tmp_path,
        server_defaults=config.get("image_server_args"),
        generation_defaults=config.get("image_generation_defaults"),
    )

    assert (tmp_path / "pizza.png").read_bytes().startswith(b"\x89PNG")
    assert [r["status"] for r in rows] == ["ok"]
    assert json.loads((tmp_path / "runs.jsonl").read_text())["status"] == "ok"


def test_qwen_3_5_4b_chats_with_and_without_thinking(tmp_path):
    binary = shutil.which("llama")
    if binary is None or not QWEN_4B.exists():
        pytest.skip("needs the llama executable and the Qwen3.5-4B file")
    args = {"-m": QWEN_4B, "-c": 4096, "-np": 1, "-fitt": 2500}

    with LlamaServer(binary, args, tmp_path / "llama.log") as server:
        quick = server.chat(
            [{"role": "user", "content": "Say hello in three words."}],
            {
                "max_tokens": 40,
                "temperature": 0,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        thought = server.chat(
            [{"role": "user", "content": "What is 17*23?"}],
            {"max_tokens": 1500, "temperature": 0},
        )

    assert quick.text and quick.reasoning == ""
    assert quick.timings["predicted_n"] > 0
    assert thought.reasoning and "391" in thought.text
    assert thought.finish_reason == "stop"
    assert not server.ready()  # stopped with the block
