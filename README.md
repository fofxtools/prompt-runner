# prompt-runner

**Under construction**

Toolkit for running prompts across multiple LLM and image models and comparing the results.

## Install

```bash
pip install -e .
```

Image generation also needs a [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) `sd-server` build
(see `docs/cuda-setup.md`).

## Usage

### LLM Evaluation

```python
from prompt_runner.llm_runner import run_llm_eval
from prompt_runner.config import load_config, load_llm_prompts, load_llm_models

config = load_config("config/config.yaml")
prompts = load_llm_prompts("config/llm_prompts.yaml")
models = load_llm_models("config/llm_models.yaml")

run_llm_eval(config, prompts, models)
```

### Image Evaluation

```python
from prompt_runner.image_runner import run_image_eval
from prompt_runner.config import load_config, load_image_prompts, load_image_models

config = load_config("config/config.yaml")
prompts = load_image_prompts("config/image_prompts.yaml")
models = load_image_models("config/image_models.yaml")

run_image_eval(config, prompts, models, mode_filter="all")  # or "txt2img" / "img2img"
```

Each model runs in its own `sd-server`. Set `sd_server` (path to the binary) in `config/config.yaml`. The comments at the top of
`config/image_models.yaml` and `config/image_prompts.yaml` describe the options, which are native sd-server request fields.
A failed prompt is recorded in `summary.json` and the run continues.

Results are saved under `results_dir` (`storage/results/` by default), one folder per run, with JSON metadata alongside the outputs.
