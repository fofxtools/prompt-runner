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

### Image generation

Each distinct model setup runs in its own `sd-server`, started and stopped for you. Set `sd_server` (path to the binary) in
`config/config.yaml`; model entries are in `config/image_models.yaml` (the comments at its top describe the options).

From the command line, prompts x seeds x models with a comparison sheet (sheets need `pip install -e ".[sheets]"`):

```bash
python scripts/run_images.py config/image_prompts.yaml -o out/ --models flux-2-klein-4b-Q8_0,flux-2-klein-9b-Q4_K_M --seeds 1,2
```

From Python, give `run_jobs` an explicit list of jobs, so you decide the matrix and the file names:

```python
from pathlib import Path
from prompt_runner.config import load_config, load_image_models
from prompt_runner.image_runner import Job, run_jobs

config = load_config("config/config.yaml")
model = {m["name"]: m for m in load_image_models("config/image_models.yaml")}["flux-2-klein-4b-Q8_0"]
lora = [{"path": "klein_snofs_v1_4.safetensors", "multiplier": 1.0}]   # a request field: same server, no reload
jobs = [
    Job(model, {"prompt": "a red fox", "seed": 7, **extra}, Path("out") / f"fox__{name}.png", {"variant": name})
    for name, extra in [("plain", {}), ("lora", {"lora": lora})]
]
rows = run_jobs(jobs, binary=config["sd_server"], out_dir=Path("out"),
                server_defaults=config.get("image_server_args"),
                generation_defaults=config.get("image_generation_defaults"), skip_existing=True)
```

- `options` are native sd-server request fields (`prompt`, `seed`, `sample_params`, `lora`, `init_image`, ...), merged over the
  model's `generation_options` and the config defaults. Image fields are file paths.
- Output names must differ for every job (a duplicate is an error); put whatever varies in the name. `skip_existing` skips jobs
  whose file exists, so a rerun resumes.
- `out_dir/runs.jsonl` gets one line per image, flushed as it happens, failures and skipped jobs included (model, seed, the exact
  request, server flags, sd.cpp version, time, your `tags`). Server output is in `out_dir/server_logs/`.
- A failed job is recorded and the run continues; if a server dies, the rest of that model's jobs are recorded as skipped.
- `prompt_runner.sheets.make_sheet` builds a labelled grid of thumbnails for comparing runs.
