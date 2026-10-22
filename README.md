# prompt-runner

Run prompts across local LLMs and image models, log every request, and compare the results. It starts and stops the
[llama.cpp](https://github.com/ggml-org/llama.cpp) and [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp)
servers for you.

> **Status:** under construction. Running models needs Linux. A Windows VM can use the servers on the Linux host:
> see [docs/windows-vm.md](docs/windows-vm.md).

## Quickstart

On a Windows VM? Skip this. The VM does not run the models, so it does not need `llama_server` or `sd_server`: go to [docs/windows-vm.md](docs/windows-vm.md).

**You need:** Linux, Python 3.10+, an NVIDIA GPU, model files, and the server programs:

- `llama` from llama.cpp, for LLMs.
- `sd-server` from stable-diffusion.cpp, for images (build notes: [docs/cuda-setup.md](docs/cuda-setup.md)).

1. Install:

   ```bash
   pip install -e .
   ```

2. In `config/config.yaml`, set `llama_server` and `sd_server` to the two programs (absolute paths).
   Check that `config/llm_models.yaml` and `config/image_models.yaml` point at your model files.

3. Check the setup. A model whose file is not found is marked `missing`:

   ```bash
   scripts/llm --list
   scripts/genimg --list
   ```

4. Try it:

   ```bash
   scripts/llm qwen3.5-4b "Say hello."
   scripts/genimg "a red fox in the snow" -o fox.png
   ```

The first call starts the model's server and leaves it loaded, so later calls are faster. Every call is logged, see [Logs](#logs).

To unload the models:

```bash
scripts/llm --stop
scripts/genimg --stop
```

### Optional: run `llm` and `genimg` from anywhere

Link the scripts into a folder on your `PATH`:

```bash
sudo ln -s "$PWD/scripts/llm" /usr/local/bin/llm
sudo ln -s "$PWD/scripts/genimg" /usr/local/bin/genimg
```

Without `sudo`, link them into `~/.local/bin` if that is on your `PATH`.

The first line of each script, `#!/home/shared/ai/prompt-runner/.venv/bin/python`, is the Python of this project's virtual environment. If your clone is somewhere else, edit that line in both scripts.

## What is in the box

| You want to | Use | Details |
|---|---|---|
| Chat with a local LLM, or get one reply | `scripts/llm` | [llm](#llm-chat-with-a-local-model) |
| Make one image, or edit one | `scripts/genimg` | [genimg](#genimg-make-one-image) |
| Run every prompt on every LLM | `run_llm_eval`, `run_llm_jobs` | [Batch runs](docs/batch-runs.md#llms) |
| Run prompts x seeds x models for images, with a comparison sheet | `scripts/run_images.py`, `run_jobs` | [Batch runs](docs/batch-runs.md#images) |
| Use the servers from a Windows VM | `genimg`, `llm` with `GENIMG_URL` / `LLM_URL` | [docs/windows-vm.md](docs/windows-vm.md) |

## Configuration

Everything is in `config/`. The comments at the top of the model files describe every option.

| File | What it holds |
|---|---|
| `config.yaml` | Paths of the two server programs, flags for every server, default models, `listen_host`, `results_dir` |
| `llm_models.yaml` | One entry per LLM: `server_args` (`llama serve` flags: `-m`, `--mmproj`, ...) and `generation_options` (request fields: sampling, ...) |
| `llm_prompts.yaml` | Prompts for batch runs: a `prompt` (one user message) or `messages`, optional `images` and `options` |
| `image_models.yaml` | One entry per image model: `server_args` (sd-server flags) and `generation_options` |
| `image_prompts.yaml` | Prompts for `scripts/run_images.py` |

How settings combine, lowest priority first:

1. `config.yaml` defaults (`llm_server_args`, `llm_generation_defaults`, `image_server_args`, `image_generation_defaults`).
2. The model's own entry.
3. The prompt's `options`, or the command line.

Ollama's option names (`num_predict`, `num_ctx`, ...) are refused, because llama-server would ignore them without a word.

## llm: chat with a local model

```bash
scripts/llm qwen3.5-4b "Say hello."                       # one reply
echo "Why is the sky blue?" | scripts/llm qwen3.5-4b      # the prompt from a pipe
scripts/llm mistral-nemo-12b                              # a chat
scripts/llm joycaption-beta-one --image fox.png "Describe this image."
scripts/llm qwen3.5-9b --reasoning off "What is 17 * 23?"
```

| Command or option | What it does |
|---|---|
| `llm MODEL "prompt"` | One reply. |
| `llm MODEL` | A chat. `/bye` or Ctrl-D ends it; Ctrl-C stops the reply being written and drops that turn. |
| `-o FILE` | Also write the answer to FILE (one reply only; an existing file is never overwritten). |
| `--system TEXT` | A system prompt. |
| `--image FILE` | Send an image with the first message; repeat for more. The model needs a vision projector: the `VISION` column of `--list` shows which models have one. PNG, JPEG, GIF, BMP and WebP work, TIFF does not. An image costs tokens by its size (about 1000 for a 1024-pixel square, about 4000 for 4096); `--image-max-tokens` in the model's `server_args` caps it. |
| `--reasoning off` | Turn thinking off, and use the sampling the model's card gives for that mode. Models whose chat template has no switch ignore it. |
| `--list` | The models: size on disk, which one is loaded, which have vision. |
| `--start [MODEL]` | Load a model without chatting: MODEL, else the one already loaded, else `default_llm_model`. |
| `--status` | Is the server running: model, pid, address, log. |
| `--stop` | Unload the model. |

Good to know:

- The answer goes to stdout and the model's thinking (dimmed) to stderr, so `llm MODEL "..." > answer.md` keeps only the answer.
- The model's `llama serve` stays loaded between calls, so the next call skips the load. A call for another model replaces it, and calls queue behind one another.
- The server listens on port 8191 (or `$LLM_PORT`). If something else is listening there, `llm` refuses to touch it.
- A chat keeps its history in the command, so it carries on even if the server was replaced between turns.
- With a prompt on the command line, `llm` never reads stdin (a pipe that never ends would hang it).
- If the reader of stdout goes away (`| head`), the reply stops quietly.
- Needs Linux (it uses `/proc` and `fcntl`), unless it runs against a remote server: [docs/windows-vm.md](docs/windows-vm.md).

## genimg: make one image

```bash
scripts/genimg "a red fox in the snow" -o out/fox.png [--model NAME] [--seed N] [--size WxH]
scripts/genimg "make the fox white" --ref out/fox.png -o out/fox_white.png      # edit an image
```

| Command or option | What it does |
|---|---|
| `genimg "prompt" -o FILE` | Make an image. An existing file is never overwritten. |
| `--model NAME` | The model. Without it: the model of the running server, else `default_image_model`. |
| `--seed N` | `-1` or no `--seed` is random. The seed used is printed. |
| `--size WxH` | Overrides the size from the config. |
| `--ref IMAGE` | A reference image to edit; repeat for more. |
| `--lora NAME[:WEIGHT]` | Add a LoRA (a file name in the server's `--lora-model-dir`; weight 1.0 by default). |
| `--list` | The models: size on disk, which one is loaded. |
| `--start [--model NAME]` | Load the model now, using a throwaway 64x64 image. |
| `--status` | Is the server running: model, pid, address, log; or whether something else holds the port. |
| `--stop` | Unload the model (or `kill PID`). |

Run `scripts/genimg --help` for the rest.

Good to know:

- It goes through `run_jobs`, so the request, the settings and the log match the batch runner's.
- The `sd-server` stays loaded between calls, so the next call skips the model load (about 3 s). The first call starts it and prints its pid. A call that needs a different model replaces it, and calls queue.
- The server listens on port 8190 (or `$GENIMG_PORT`). If something else is listening there, `genimg` refuses to touch it.
- `--start` makes the model take its GPU memory now: use it before starting anything else that shares the GPU.
- Needs Linux (it uses `/proc` and `fcntl`), unless it runs against a remote server: [docs/windows-vm.md](docs/windows-vm.md).

## Batch runs

To compare models, run many requests in one go. Each model's server is started once, every request is logged, and a failed job does not stop the run.

- **LLMs:** `run_llm_eval(config, prompts, models)` runs every prompt on every model. `run_llm_jobs` lets you decide the matrix and the file names.
- **Images:** `scripts/run_images.py` runs prompts x seeds x models and builds a comparison sheet. `run_jobs` lets you decide the matrix and the file names.

See [docs/batch-runs.md](docs/batch-runs.md) for examples and options.

## Logs

Every request is logged as one line of `runs.jsonl`, flushed as it happens, failures and skipped jobs included.

| Source | Where the log goes |
|---|---|
| `llm` | `storage/results/llm-cli/` of this repo; with `-o`, `.llm/` beside the file |
| `genimg` | `.genimg/` beside the image |
| `run_llm_eval` | `<results_dir>/<run folder>/llm/` |
| `run_jobs` | `out_dir/` |

A row has:

- The exact request (messages or prompt, options) and the seed used.
- The answer, with the model's thinking in a separate field.
- The sampling in effect, finish reason, token counts and speeds, or the time taken for an image.
- The server flags, the engine version, and any error.
- Your own `tags`.

A chat is one run with a row per turn. A reply cancelled with Ctrl-C is not logged. The servers' own output is in `/tmp/llm-<port>.log` and `/tmp/genimg-<port>.log`, and in `server_logs/` for batch runs.

## Sharing one GPU

To keep an LLM and the image server loaded together on one card:

1. Start the image server first: `genimg --start`.
2. Raise `-fitt` for the LLM (about 6000: the MiB that `--fit` leaves free) in `llm_server_args` in `config.yaml`, so llama.cpp leaves room for the image server's peak.

## Servers and networks

- The servers listen on the `listen_host` address in `config.yaml`: `127.0.0.1` (this computer only) if the key is missing, `0.0.0.0` (every address of this computer) in the shipped config.
- The servers have no login. Use `0.0.0.0` only on a computer where no one else is on the network.
- `GENIMG_HOST` and `LLM_HOST` override the key for one run.
- A server keeps the address it was started with until a call that wants another one replaces it. After changing the key: `genimg --stop; genimg --start` and `llm --stop; llm --start`.
- `genimg --status` and `llm --status` show the address.

To use the servers from a virtual machine, which has no GPU, see [docs/windows-vm.md](docs/windows-vm.md).

## More documentation

- [docs/batch-runs.md](docs/batch-runs.md): batch runs from Python and `scripts/run_images.py`.
- [docs/windows-vm.md](docs/windows-vm.md): using the servers from a Windows VM.
- [docs/cuda-setup.md](docs/cuda-setup.md): building stable-diffusion.cpp.
- [docs/development.md](docs/development.md): formatting, linting and tests.
