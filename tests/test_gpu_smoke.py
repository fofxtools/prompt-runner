"""Real sd-server smoke test. Needs a GPU, the sd-server binary and the Klein 4B model files.

Skipped by default; run with: pytest -m gpu
"""

import json
from pathlib import Path

import pytest

from prompt_runner.config import load_config, load_image_models
from prompt_runner.image_runner import run_image_eval

pytestmark = pytest.mark.gpu

REPO = Path(__file__).resolve().parent.parent


def test_klein_4b_generates_one_image(tmp_path):
    config = load_config(str(REPO / "config/config.yaml"))
    config["results_dir"] = str(tmp_path)
    models = [
        m
        for m in load_image_models(str(REPO / "config/image_models.yaml"))
        if m["name"] == "flux-2-klein-4b-Q8_0"
    ]
    prompts = [
        {
            "id": "pizza",
            "mode": "txt2img",
            "options": {"prompt": "a pizza on a sunny beach", "seed": 1},
        }
    ]

    run_image_eval(config, prompts, models)

    (run_dir,) = tmp_path.iterdir()
    png = run_dir / "image" / "pizza" / "flux-2-klein-4b-Q8_0_0.png"
    assert png.read_bytes().startswith(b"\x89PNG")
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["image"]["failures"] == []
