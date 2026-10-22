# Batch runs

Run many requests in one go: every prompt on every model, or prompts x seeds x models for images. Each run starts one server
per model, loads each model once, logs every request, and keeps going if a job fails. This is for comparing models. For a single
reply or image, use [`llm` and `genimg`](../README.md#llm-chat-with-a-local-model).

- [LLMs](#llms): `run_llm_eval`, `run_llm_jobs`.
- [Images](#images): `scripts/run_images.py`, `run_jobs`.
- Logs: see [Logs](../README.md#logs) in the README for the fields of a row.

## LLMs

Every prompt runs on every model, one `llama serve` per model:

```python
from prompt_runner.llm_runner import run_llm_eval
from prompt_runner.config import load_config, load_llm_prompts, load_llm_models

config = load_config("config/config.yaml")
prompts = load_llm_prompts("config/llm_prompts.yaml")
models = load_llm_models("config/llm_models.yaml")

run_llm_eval(config, prompts, models)
# run_llm_eval(config, prompts, models, options={"temperature": 0.7})  # one value for every model
```

Results go to `<results_dir>/<run folder>/llm/`:

- `runs.jsonl`: one row per prompt and model. See [Logs](#logs).
- `<prompt id>/<model>.md`: the answer.
- `server_logs/`: the servers' own output.

A random seed is chosen for every chat that has none, and recorded. A reply cut at `max_tokens` has the finish reason `length`.

For a vision model, a prompt can have `images:` (a list of file paths), and a user message in `messages` can have its own. They are sent before the text, and the log records the paths, not the images.

To decide the matrix and the file names yourself, give `run_llm_jobs` a list of `LlmJob`s (the same idea as `run_jobs` for images):

- A job can leave out its output file (`out`): its answer is then only in the log row.
- `on_delta(kind, piece)` is called as the reply streams in.
- `say(line)` receives the progress lines (default `print`).

## Images

From the command line, prompts x seeds x models with a comparison sheet:

```bash
python scripts/run_images.py config/image_prompts.yaml -o out/ --models flux-2-klein-4b-Q8_0,flux-2-klein-9b-Q4_K_M --seeds 1,2
```

From Python, give `run_jobs` a list of jobs, so you decide the matrix and the file names:

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

- A job's options are native sd-server request fields (`prompt`, `seed`, `sample_params`, `lora`, `init_image`, ...), merged over the model's `generation_options` and the config defaults. Image fields are file paths.
- Output names must differ for every job (a duplicate is an error). Put whatever varies in the name.
- `skip_existing` skips jobs whose file exists, so a rerun resumes.
- A failed job is recorded and the run continues. If a server dies, the rest of that model's jobs are recorded as skipped.
- `prompt_runner.sheets.make_sheet` builds a labelled grid of thumbnails for comparing runs.
