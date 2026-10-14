"""Real sd-server smoke test. Needs a GPU, the sd-server binary and the Klein 4B model files.

Skipped by default; run with: pytest -m gpu
"""

import json
from pathlib import Path

import pytest

from prompt_runner.config import load_config, load_image_models
from prompt_runner.image_runner import Job, run_jobs

pytestmark = pytest.mark.gpu

REPO = Path(__file__).resolve().parent.parent


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
