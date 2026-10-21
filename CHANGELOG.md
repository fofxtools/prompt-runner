# Changelog

## [0.3.0] - 2026-10-21

The LLM runner now uses llama.cpp instead of Ollama (breaking), and there is a new `llm` command. Image generation works as before.

### LLM runner
- Breaking: `run_llm_eval` starts a `llama serve` per model instead of using Ollama; the `ollama` dependency is gone.
- `config.yaml`: new `llama_server` (the `llama` executable) and `llm_server_args` (flags for every model). `llm_generation_defaults` uses `max_tokens` (was `num_predict`; now 16384, not unlimited) and `seed`, with no `temperature`.
  Ollama's `num_predict`, `num_ctx`, `num_batch`, `num_gpu` and `num_thread` are refused with a hint, because llama-server would ignore them silently.
- `llm_models.yaml`: entries have `server_args` (`-m`, `--mmproj`, ...) and `generation_options`, like the image models. It lists the 14 local GGUF models, each with a source link and the sampling its model card recommends.
- New `run_llm_jobs(jobs, ...)` with `LlmJob`, shaped like `run_jobs`. `run_llm_eval(config, prompts, models, options=None)` builds the jobs; `options` sets request fields for every chat. There is no completion mode and no `prompt_filter`.
- Images: a prompt can have `images:` (file paths), and a user message in `messages` can have its own. A missing file, or images for a model without `--mmproj`, are refused before any server starts.
- Results: `<run folder>/llm/runs.jsonl` has one row per prompt and model, with the thinking apart from the answer, finish reason, tokens, speeds, seed, server flags, model load time and the sampling in effect. The image paths are logged, not the images.
  `summary.json` and the per-mode JSON files are gone.

### llm
- New `llm` command: `llm list` (models, size on disk, which is loaded), `llm run MODEL [PROMPT]` (one reply, a piped prompt, or a chat with history; `--system`, `--image`, repeatable) and `llm stop`.
- The answer goes to stdout and the thinking to stderr. The server stays loaded between calls on port 8191 (`LLM_PORT`), and a call for another model replaces it. With a prompt, stdin is never read.

### Servers
- New `server_process.py` (`ServerProcess`, `ServerError`) and `llama_server.py` (`LlamaServer.chat()` streams the reply). `SdCppServer` is now a small subclass.
- Breaking: `SdCppError` is now `ServerError`; `build_argv`, `free_port` and `engine_version` moved to `server_process`; `STARTUP_TIMEOUT` is a class attribute.
- `keepalive` is generic: it takes a `KeptServer` made by `kept_server(tool, server_class, port)`.
- `merge_image_options` is now `merge_options`; the old flat `merge_options` and `create_result_structure` are gone. `merged_server_args` moved to `utils`, next to `random_seed`.

### genimg
- The kept server's files are now `/tmp/genimg-<port>.{lock,json,log}` (were `genimg-sd-<port>`). Run `genimg --stop` before upgrading, or genimg will not recognize the running server.

### Tests
- Fixed a test that failed about 1 time in 100 (it read a just-started process's command line too early).

## [0.2.3] - 2026-10-20

`genimg` can now start and stop its sd-server.

### genimg
- `genimg --start [--model NAME]` starts the server and loads the model, using a throwaway 64x64 image (the server otherwise loads it on the first request).
  Use it to hold the GPU memory before starting something else that shares the GPU.
- `genimg --stop` stops the server once any running call is done.

### Image runner
- New helper `merged_server_args(server_defaults, model)`; `run_jobs` uses it to build each model's server flags.

## [0.2.2] - 2026-10-20

New `genimg` command for single images, with an sd-server that stays loaded between calls.

### genimg
- `scripts/genimg "prompt" -o out.png` makes one image with any model in `image_models.yaml` (`--model`, `--seed`, `--size`, `--lora`, and
  `--ref` to edit an image). It goes through `run_jobs`, so settings and logs match the runner; `runs.jsonl` is written to `<output folder>/.genimg/`.
- The model is `--model`, else the model of the running server, else the new `default_image_model` in `config.yaml`.
- The sd-server stays loaded, so the next call skips the model load (about 3 s faster for Klein 9B at 1024x1024). The first call starts it and
  prints its pid (stop it with `kill PID`), a different model replaces it, and parallel calls queue. If something else is listening on the
  port (8190, or `GENIMG_PORT`), genimg refuses it and never kills it. Linux only (it uses `/proc` and `fcntl`).
- The seed used is printed, and an existing output file is never overwritten.
- Breaking, compared with the old standalone script: it no longer uses a hand-started sd-server (`GENIMG_URL` is gone), steps and CFG come from the
  model entry instead of being hard-coded, the size comes from `image_generation_defaults` (also with `--ref`), and it needs the project venv.

### Image runner
- `run_jobs(..., open_server=...)` lets the caller supply the server for a group. The default is unchanged: a new server, stopped afterwards.
- `SdCppServer` can run on a fixed `port`, be left running (`keep=True`) or `attach()` to a running server. `spawn()`, `wait_ready()`, `ready()`,
  `stop()` and `pid` are now public. The server is started without stdin.

### Config
- `image_generation_defaults` is now 1024x1024 (was 512x512), so runner batches without an explicit size render at 1024x1024.
- Added `default_image_model` (used by `genimg`).
- Removed `--max-vram: 10` from the Klein and Qwen-Image 2.1 entries. On the current stable-diffusion.cpp (master-964) uncapped
  runs are as fast or faster for every model tested (Juggernaut, Klein 4B and 9B, Qwen 2.1) up to 2048x2048, never ran out of memory, and
  do not depend on free RAM; the cap now only saves 2-3.5 GiB of VRAM. kv-merge Q8_0 and Juggernaut were 20% slower with it.

### Image models
- Added `qwen-image-2.1-Q6_K_XL` (replaces the Q4_K_M entry; 40 steps), `qwen-image-2.1-turbo-Q6_K_XL` (8 steps, about 8x faster, similar
  text quality) and `qwen-image-2.1-Q6_K_XL-Lightning` (4-step LoRA, about 5 s per image).
- Added `Qwen-Image-Edit-2511-Q4_K_M` and a `-Lightning` variant (4 steps).
- Added `snofsSexNudesAndOther_v14Klein9bDistilled_full_Q6_K_snofs_lora` (the SNOFS Q6_K checkpoint with its LoRA applied per request).
- Added `flux2-dev-Q4_K_M-Turbo` (FLUX.2 dev with the fal Turbo LoRA, 8 steps, about half the time).
- `flux2-dev-Q4_K_M` now points to the Unsloth GGUF (same file).

### Tests and docs
- `pytest -m all` runs every test, the GPU smoke test included. A shared fake sd-server lives in `tests/conftest.py`.
- New docs: `rendered-text-image-models.md` (in-image paragraph text: OpenAI, DeepInfra and local Qwen models), `llama-cpp-vlm-bench.md` and
  `venice-vs-local-bench.md`. The README shows how to set `sd_server`.

## [0.2.1] - 2026-10-14

Image generation now takes an explicit list of jobs instead of the config-driven `run_image_eval()`. This is a breaking change.

### Image runner
- New `run_jobs(jobs, binary=..., out_dir=...)`. A `Job` is a model entry, the request fields for one image, the output path and
  free-form `tags`. You build the list (your own matrix, your own file names); jobs are grouped so each model loads once.
- Every attempt is appended to `out_dir/runs.jsonl`, one line per image, as it happens (so a crash loses nothing): model,
  seed, exact request, server flags, sd.cpp version, time, tags. Failed and skipped jobs get a line too. `write_log=False` turns it off.
- `skip_existing=True` skips jobs whose output file exists, so a rerun resumes. Duplicate output paths are rejected up front.
- Input images stay where they are and are logged as the given paths (no copy into the output folder).
- Removed `run_image_eval()`, `save_image_summary`, the per-run folder layout (`image/<prompt>/`, `summary.json`, `inputs/`),
  the per-image JSON files and the `mode` filter.

### Prompts
- Prompt files are now a flat list of `{id, prompt, <any request field>}` (YAML or JSON), no `mode` and no nested `options`.
  An id may be any filename-safe string (so `hotdog-1` works). `config/image_prompts.yaml` is rewritten as an example.

### Added
- `prompt_runner.sheets.make_sheet()`: labelled contact sheets for comparing models, LoRAs, seeds. Needs Pillow
  (`pip install "prompt-runner[sheets]"`).
- `scripts/run_images.py`: prompts x seeds x models from the command line, with a comparison sheet.

### Image models
- Added `Juggernaut_RunDiffusionPhoto2_Lightning_4Steps` and `Juggernaut-XI-byRunDiffusion-Lightning` (6 steps, `dpm++2m_sde` with
  the `karras` scheduler, cfg 1.8; the two look the same in blind tests).
- Added `snofsSexNudesAndOther_v14Klein9bDistilled_full_fp8` (the original fp8 file: fits in 8.7 GB, same quality, about twice as
  slow as the Q5_K) and `..._full_Q6_K` (converted with `sd-cli -M convert`).
- Noted in `sd3.5_large_turbo-Q5_1` that it needs about 1024x1024; at 512-768 it has artifacts and garbled detail.
- `sd_xl_base_1.0` no longer forces 1024x1024, so the config default and `--size` apply to it like every other model.

### Tests
- Replaced the `run_image_eval` tests with tests for grouping, the failure policy, resume, the log rows and the new prompt loader;
  added tests for the sheets. The real-GPU smoke test now uses `run_jobs`.

## [0.2.0] - 2026-10-14

Image generation now runs through stable-diffusion.cpp's `sd-server` instead of the `stable-diffusion-cpp-python` binding.
This is a breaking change to the image config files; there is no backwards compatibility.

### Image runner
- Starts one `sd-server` per model (on a free local port), sends each prompt to it, and stops it afterwards. A crash in one
  model no longer takes down the whole run.
- A failed prompt is recorded and the run continues; if the server dies, the rest of that model is skipped. Failures (with the
  error, exit code and server log path) are listed in `summary.json`.
- Seeds are chosen by the runner (`-1` or no seed picks a random one) and recorded per image. A batch uses seed, seed+1, ...
- Input images (`init_image`, `mask_image`, `ref_images`) are copied into the run folder (`inputs/`) and sent as base64.
- Per-image JSON now records the sd.cpp version, the full `sd-server` command line, the exact request and the elapsed time.
  Server output goes to `server_logs/<model>.log`.
- Images are saved as the PNG bytes the server returns, so Pillow is no longer needed.

### Config (breaking)
- `config.yaml`: new `sd_server` (absolute path to the binary) and `image_server_args` (flags for every model, now
  `--lora-model-dir` and `--sage-attn`). Removed `image_init_defaults` and `image_generation_custom` (and with it `prompt_prefix`).
- `image_models.yaml`: `init_options` is replaced by `server_args`, a mapping of literal sd-server flags (`true` gives a bare
  flag, `false`/`null` drops it). `generation_options` and prompt `options` are now native sd-server request fields, for example
  `sample_params: {sample_steps: 4, guidance: {txt_cfg: 1.0}}` instead of `cfg_scale`/`sample_steps`.
- Generation options merge recursively (config defaults, then model, then prompt): mappings merge, lists and scalars replace.
- LoRAs use the `lora: [{path, multiplier}]` list; `<lora:...>` prompt tags are no longer supported.
- All 33 model entries were converted and kept. The memory flags for the old binding (`keep_*_on_cpu`, `offload_params_to_cpu`)
  are gone. `--max-vram 10` is set on the Klein and Qwen-Image 2.1 entries, where it avoids out-of-memory errors on large images.

### Image models
- `qwen-image-2.1-Q4_K_M` now works (cfg 4.0, 25 steps).

### Removed
- The `stable-diffusion-cpp-python` and `pillow` dependencies, and `scripts/patch_gguf_head_dim.py` (the unpatched PornMaster
  GGUF loads on current stable-diffusion.cpp).

### Docs
- Rewrote `cuda-setup.md` (CUDA install checked against NVIDIA's guide, stable-diffusion.cpp build, SageAttention GPU
  requirement) and trimmed the README.
- `dev-notes.md`: removed the notes that only applied to the Python binding; the fp8 note now uses `sd-cli -M convert`.

### Tests
- Added tests for the server wrapper (against a fake `sd-server` script), the recursive merge, and the failure handling in the
  runner. A real-GPU smoke test is skipped by default (`pytest -m gpu`).

## [0.1.6] - 2026-10-14

### Image models
- Replaced `PornMaster_Flux2Klein_v4_Turbo_Q5_K` with `PornMaster_Flux2Klein_v4_Turbo_Q6_K`, the original
  unpatched GGUF (loads on current stable-diffusion.cpp, so the head_dim patch is no longer needed).
- Added `flux2-dev-Q4_K_M` (FLUX.2 dev, 32B, Mistral-Small-3.2 encoder).
- Klein entries use the small-decoder VAE (`full_encoder_small_decoder.safetensors`) instead of `flux2-vae.safetensors`.

### Docs
- Added `kv-merge-gguf-frictions.md`.

## [0.1.5] - 2026-10-13

### Image models
- Added FLUX.2 Klein 4B entries: `flux-2-klein-4b` (stock bf16 safetensors) and the Unsloth GGUF
  quants `flux-2-klein-4b-Q4_K_M`, `-Q6_K` and `-Q8_0`, all with the Qwen3-4B text encoder.
- Added `flux-2-klein-9b-Q4_K_M`, `flux-2-klein-9b-kv-merge-Q8_0` and `-Q5_K`, and
  `pGGUFSNOFSSexNudesAndOther_v14Distilled_pruned_q5KM`.
- Klein entries no longer set `keep_vae_on_cpu` or `guidance` (the OOM workaround costs ~8s per image,
  and `guidance` has no effect on Klein). `sd3.5_medium` now sets `keep_clip_on_cpu: true`, which it
  needs even at 512x512.
- Moved the shared `lora_model_dir` out of the individual entries into `image_init_defaults`.
- Tidied comments on the Klein 9B and Qwen-Image-2.1 entries.

### Config
- New `image_init_defaults` in `config.yaml`: defaults merged underneath every model's
  `init_options` (a model's own value wins), passed to `initialize_stable_diffusion(model, init_defaults)`.
- An empty `image_generation_custom:` key (loaded as `None`) no longer crashes `run_image_eval()`.
- Type-check failures while loading config files (a file, entry or option that is not a dict/list)
  now raise `TypeError` instead of `ValueError`, as ruff TRY004 requires. Missing or invalid values
  still raise `ValueError`.

### Tooling
- `make lint` / `make fmt` now cover the whole repo (`.`); added `make ruff-fix`.
- Ruff and black clean: `typing.Dict`/`List` replaced with built-in generics, imports sorted, and
  `scripts/patch_gguf_head_dim.py` made executable.

### Docs
- `dev-notes.md`: added FLUX.2 Klein notes for a 16GB card (memory flags, LoRAs, speed, the Qwen-Image-Edit
  offload requirement) and the `image_init_defaults` section.

### Tests
- Added tests for `init_defaults` precedence and non-mutation, `image_init_defaults` passthrough in
  `run_image_eval()`, and the empty `image_generation_custom` case.

## [0.1.4] - 2026-10-09

### Image models
- Added SDXL-family entries: `dreamshaperXL_lightningDPMSDE`, `dreamshaperXL_alpha2Xl10`,
  `dreamshaperXL_lightningInpaint`, `kohakuXLEpsilon_rev2`,
  `sdxlFixedvaeFp16Remove_baseFixedvaeV1Fp16`, `LahCuteCartoonSDXL_alpha`.
- Fixed `flux1-schnell-Q5_K_S`'s `t5xxl_path` (was pointing at the fp16 safetensors
  encoder instead of the GGUF-quantized one).
- Removed `keep_clip_on_cpu: true` from several entries where it wasn't actually needed.
- Added `PornMaster_Flux2Klein_v4_Turbo_Q5_K`: FLUX.2 Klein 9B via a patched GGUF UNet
  (`scripts/patch_gguf_head_dim.py` - see below) plus LoRA support via `lora_model_dir`.
- Added `snofsSexNudesAndOther_v14Klein9bDistilled_full_Q5_K`: FLUX.2 Klein 9B with
  SNOFS merged directly into the weights, converted from the original fp8 safetensors
  (which stable-diffusion.cpp upconverts to F16 on load, too large for a 16GB card) to
  Q5_K GGUF via `stable_diffusion_cpp.sd_cpp.convert()`.
- Added `qwen-image-2.1-Q4_K_M`: not usable yet, `stable-diffusion-cpp-python` 0.4.7
  predates upstream's Qwen-Image-2.1 support.
- Added `Qwen-Image-Edit-2509-Q3_K_M`: working, including image editing via
  `ref_images` and LoRA support via `lora_model_dir`. Needs `offload_params_to_cpu`
  (text encoder + diffusion model alone exceed 16GB VRAM) and the native-format VAE,
  not the diffusers-format one also present in `vae/`.

### Scripts
- Added `scripts/patch_gguf_head_dim.py`: fixes community-quantized FLUX.2 GGUFs whose
  attention norm-scale tensors use `.weight` instead of the `.scale` suffix
  stable-diffusion.cpp expects, which otherwise causes a SIGFPE on load (wrong `num_heads`
  inferred) and tensor-loading failures.

### Packaging
- Moved `pillow` from the `dev` extra to core `dependencies` - `image_runner.py` imports
  it unconditionally, so a plain `pip install prompt-runner` was broken without `[dev]`.
- Added `gguf` to the `dev` extra - required by `scripts/patch_gguf_head_dim.py`.

### Docs
- `dev-notes.md`: added notes on the GGUF norm-scale tensor naming issue above,
  `stable-diffusion-cpp-python` being unpinned (a stale venv silently misdetects newer
  architectures like FLUX.2 Klein as a generic/older variant instead of failing loudly),
  the fp8-to-GGUF conversion technique, diffusers-format vs native-format VAE
  incompatibility, and VAE load failures surfacing as a segfault in `generate_image()`
  rather than an error at load time.
