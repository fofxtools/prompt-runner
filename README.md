# prompt-runner

**Under construction**

Toolkit for running prompts across multiple LLM and image models and comparing the results.

## Install

```bash
pip install -e .
```

LLM evaluation needs [llama.cpp](https://github.com/ggml-org/llama.cpp)'s `llama` executable, and image generation a
[stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) `sd-server` build (see `docs/cuda-setup.md`).

## Usage

### LLM evaluation

Each distinct model setup runs in its own `llama serve` ([llama.cpp](https://github.com/ggml-org/llama.cpp)), started and stopped for you. Set `llama_server` in
`config/config.yaml` to the `llama` executable (a bare name is looked up on `PATH`):

```yaml
llama_server: llama
```

Model entries are in `config/llm_models.yaml` (the comments at its top describe the options): the `llama serve` flags of a model are its `server_args` (`-m`, `--mmproj`, ...), added to
`llm_server_args` in `config.yaml`; request fields (`temperature`, `max_tokens`, `seed`, ...) are its `generation_options`, added to `llm_generation_defaults`. A prompt in `config/llm_prompts.yaml` has
a `prompt` (sent as one user message) or `messages`, and optional `options` that win over the model's. For a vision model (one with an `--mmproj`), a `prompt` can have `images:` (a list of image file paths)
and a user message in `messages` can have its own `images:`: they are sent before the text, and the log records the paths, not the images. PNG, JPEG, GIF, BMP and WebP work, TIFF does not;
an image costs tokens by its size (about 1000 for a 1024-pixel square, about 4000 for 4096), which `--image-max-tokens` in the model's `server_args` can cap. Ollama's option names (`num_predict`, `num_ctx`, ...) are refused, because llama-server would ignore them.

```python
from prompt_runner.llm_runner import run_llm_eval
from prompt_runner.config import load_config, load_llm_prompts, load_llm_models

config = load_config("config/config.yaml")
prompts = load_llm_prompts("config/llm_prompts.yaml")
models = load_llm_models("config/llm_models.yaml")

run_llm_eval(config, prompts, models)
# run_llm_eval(config, prompts, models, options={"temperature": 0.7})  # one value for every model
```

Every prompt runs on every model, one server per model. The results go to `<results_dir>/<run folder>/llm/`:
`runs.jsonl` (one row per prompt and model: the request, the answer and the model's thinking in separate fields, the finish reason, token counts and speeds, the seed, the sampling in effect, the server flags and any error),
`<prompt id>/<model>.md` (the answer) and `server_logs/`. A random seed is chosen for every chat that has none, and recorded. A reply that is cut at `max_tokens` has the finish reason `length`.

From Python, give `run_llm_jobs` an explicit list of `LlmJob`s, so you decide the matrix and the file names (the same idea as `run_jobs` for images).

To share the GPU with a loaded sd-server, start the sd-server first (`genimg --start`) and raise `-fitt` for the LLM (about 6000, the MiB `--fit` leaves free; `llm_server_args` in `config.yaml`),
so llama.cpp leaves room for the image server's peak.

### One model at a time: `scripts/llm`

A command like `ollama list` and `ollama run`, over the models in `config/llm_models.yaml`, with the same server flags and sampling as the runner:

```bash
scripts/llm list                                # names, size on disk, and which model is loaded
scripts/llm run qwen3.5-4b "Say hello."         # one reply
echo "Why is the sky blue?" | scripts/llm run qwen3.5-4b > answer.md
scripts/llm run mistral-nemo-12b                # a chat (/bye or Ctrl-D ends it; Ctrl-C stops the reply being written)
scripts/llm run joycaption-beta-one --image fox.png "Describe this image."   # images go with the first message
scripts/llm stop                                # unload the model
```

The answer goes to stdout and the model's thinking (dimmed) to stderr, so a redirected answer is only the answer. `--system TEXT` sets a system prompt. The model's `llama serve` stays loaded between calls, so the next call skips the load;
a call for another model replaces it, and calls queue behind one another. It listens on port 8191 (or `$LLM_PORT`) and llm refuses to touch anything else that is listening there; `genimg`'s sd-server is separate (port 8190), so
start it first when they share the GPU (see above). A chat keeps its history in the command, so it carries on even if another call unloaded the model meanwhile. Without a prompt, `llm run` reads the prompt from a pipe; with a prompt it
never reads stdin (an open pipe that never ends would hang it). Linux only (it uses `/proc` and `fcntl`); `llama` must be on the `PATH`.

### Image generation

Each distinct model setup runs in its own `sd-server`, started and stopped for you. Set `sd_server` in `config/config.yaml` to the absolute path of the binary (no `~`).

```yaml
sd_server: /home/shared/ai/stable-diffusion.cpp/build/bin/sd-server
```

Model entries are in `config/image_models.yaml` (the comments at its top describe the options).

From the command line, prompts x seeds x models with a comparison sheet (sheets need `pip install -e ".[sheets]"`):

```bash
python scripts/run_images.py config/image_prompts.yaml -o out/ --models flux-2-klein-4b-Q8_0,flux-2-klein-9b-Q4_K_M --seeds 1,2
```

One image at a time, with `scripts/genimg` (the model is `--model`, or `default_image_model` in `config/config.yaml`; add `--ref` to edit):

```bash
scripts/genimg "a red fox in the snow" -o out/fox.png [--model NAME] [--seed N] [--size WxH]
scripts/genimg "make the fox white" --ref out/fox.png -o out/fox_white.png
```

It goes through `run_jobs`, so the request, the settings and the log match the runner's. The `sd-server` stays loaded between calls, so
the next call skips the model load (about 3 s): the first call starts it and prints its pid, a call that needs a different model replaces it,
parallel calls queue, and if something else is listening on the port (8190, or `$GENIMG_PORT`) genimg refuses to touch it. `scripts/genimg --start [--model NAME]` loads it ahead of time with a throwaway 64x64 image (the model takes its GPU memory then, so use it
before starting anything else that shares the GPU). Stop it with `scripts/genimg --stop` (or `kill PID`). This needs Linux (/proc, fcntl). Without `--model`, the model of the running server is used. The seed used is printed (`-1` or no `--seed` is random), an
existing output file is never overwritten, and `runs.jsonl` goes to `<output folder>/.genimg/`. Run `scripts/genimg --help` for the rest.

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
