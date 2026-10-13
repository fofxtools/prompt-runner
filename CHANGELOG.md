# Changelog

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
