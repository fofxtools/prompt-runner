# Changelog

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
