# Informal Developer Notes

## vae_decode_only and img2img

When an image model is initialized with vae_decode_only=true, it can only be used for txt2img. Attempting to use it for img2img will raise an error.

## mypy errors

This was added to `pyproject.toml`to fix `mypy` errors, `Skipping analyzing "stable_diffusion_cpp": module is installed, but missing library stubs or py.typed marker [import-untyped]`:

```
[[tool.mypy.overrides]]
module = "stable_diffusion_cpp.*"
ignore_missing_imports = true
```

## Flux Dev memory errors

When Flux Dev was the first model in image_models.yaml, there were memory errors (segmentation faults). Moving it to the end of the file resolved the issue.

## SDXL and resolution

SDXL is very resolution sensitive. It is designed for 1024x1024. The results are very poor at 512x512.

## Reasoning models and token use

With reasoning capable Qwen3 models (qwen3-vl:8b, qwen3:8b), you may receive empty or truncated outputs with `num_predict=512`. This is because the thinking process may use up all the tokens. Try increasing `num_predict`. `num_predict` can also be set to `-1`, for unlimited.

## FLUX.2 GGUF norm-scale tensor naming

Community-quantized FLUX.2 GGUFs (checked: PornMaster Klein 9B finetune) often use
".weight" as the suffix for every tensor uniformly, including attention Q/K-norm
scale tensors. stable-diffusion.cpp expects those specifically named
"...norm.key_norm.scale" / "...norm.query_norm.scale" - without that, num_heads
detection silently falls back to a wrong default and crashes with SIGFPE deeper in
model init, and the real tensor loader fails outright for the same reason across
every block. `scripts/patch_gguf_head_dim.py` renames the affected tensors in a
copy of the GGUF (pure rename, no re-quantization) to fix this.

## stable-diffusion-cpp-python is unpinned, and the .venv won't auto-update

`pyproject.toml` doesn't pin a version, so whatever was installed when the venv was
last created (or last `pip install --upgrade`d) is what you get - `pip install -e .`
does not upgrade already-satisfied dependencies. This matters because the library
moves fast: a venv built before FLUX.2 Klein support landed upstream will silently
misdetect a Klein GGUF as a generic (non-Klein) FLUX.2 model - wrong tensor shapes,
wrong text encoder architecture assumed, load fails - with no indication that the
real problem is just an outdated package. If a model type "isn't supported," check
the installed version first:

```
.venv/bin/pip show stable-diffusion-cpp-python
```

Rebuild with CUDA to upgrade:

```
CMAKE_ARGS="-DSD_CUDA=ON" .venv/bin/pip install --upgrade stable-diffusion-cpp-python --no-binary stable-diffusion-cpp-python --no-cache-dir
```

## fp8 safetensors checkpoints get upconverted to F16 on load - convert to GGUF instead

stable-diffusion.cpp's GGML backend doesn't keep fp8 weights at their stored size - it
upconverts them to F16 eagerly at load time, roughly doubling VRAM use for the whole
model (an fp8 checkpoint `N` bytes on disk needs roughly `2N` once loaded). This is a
GGML-backend-specific behavior, not a property of fp8 files in general: PyTorch-native
tools like ComfyUI keep fp8 weights at their actual size in memory and only cast
transiently per-op during compute, so the same file can work fine there even when it
doesn't fit here.

Fix: convert the checkpoint to a quantized GGUF once, ahead of time, rather than
loading the fp8 file directly every run. `stable_diffusion_cpp.sd_cpp.convert()` is
already available in the installed package (no separate build needed):

```python
from stable_diffusion_cpp import sd_cpp, GGML_TYPE_MAP

sd_cpp.convert(
    b"/path/to/checkpoint.safetensors",
    b"",  # vae_path, empty if not merging one in
    b"/path/to/output-Q5_K.gguf",
    GGML_TYPE_MAP["q5_k"],
    b"",  # tensor_type_rules
    False,  # convert_name
)
```

Paths must be `bytes`, not `str`. Converted output already uses correct native tensor
naming (unlike many community-quantized GGUFs - see the norm-scale tensor naming note
above), since it's written by the same tool that reads it back.

## VAE files: diffusers format vs native format aren't interchangeable

stable-diffusion.cpp's VAE loader expects the original/native checkpoint tensor naming
(e.g. `first_stage_model.decoder.mid.block_1.conv2.bias`), not the HuggingFace Diffusers
`AutoencoderKL` save format (e.g. `decoder.mid_block.attentions.0.proj.bias`). These are
different module hierarchies, not just a prefix difference, so there's no quick rename
fix the way there is for the GGUF norm-scale tensor naming issue above.

A reliable tell that you have the wrong format: a file named
`diffusion_pytorch_model.safetensors` living under a `vae/` subfolder in a HF repo is
the standard Diffusers save path - that's the one that won't work. The ComfyUI
"split_files" repackaging of
the same weights (e.g. `Comfy-Org/<Model>_ComfyUI/split_files/vae/`) uses native naming
and does work; use that source for VAEs (and similarly for diffusion models/text
encoders where a "split_files" or GGUF conversion exists) instead of the original
model author's own repo.

## VAE load failure doesn't abort - it logs and then segfaults in generate_image

If `vae_path` points at a file with the wrong tensor naming (see above), loading logs
`tensor ... not in model file` / `load tensors from model loader failed` but does not
raise an exception or stop execution. `StableDiffusion.__init__` returns successfully;
the failure only surfaces when `generate_image()` is called, as a segfault (`NULL
pointer access` from the Python binding, `dumped core` from the process). If you hit a
segfault specifically during decode, with no other error, check what was logged during
VAE loading earlier in the output - the real error is there, not at the crash site.

## FLUX.2 Klein on a 16GB card: init flags, LoRAs, and speed

Measured on an RTX 5070 Ti at 512x512, 4 steps, cfg 1.0 (stable-diffusion-cpp-python 0.4.7):

- Use **no memory flags** by default: Klein 4B ~1.2s, 9B ~1.8s per image. `keep_vae_on_cpu` adds ~8s (CPU VAE decode)
  and `keep_clip_on_cpu` adds ~5-6s (CPU text encode). They are OOM workarounds, not speedups.
- A runtime LoRA (`<lora:name:weight>` in the prompt) OOMs at 512 on the 9B models unless the entry sets
  `offload_params_to_cpu: true` (~10-30s per image depending on the model). Exception:
  `flux-2-klein-9b-kv-merge-Q8_0` (9.7GB) needs `keep_clip_on_cpu: true` just to load, and then runs a LoRA at ~9s; offload
  is much slower for it (~40s).
- `guidance` has no effect on Klein (no guidance embedding in the weights); only `cfg_scale: 1.0` and `sample_steps: 4` matter.
- `klein_snofs_v1_4` is a Klein 9B LoRA; on Klein 4B it aborts (`GGML_ASSERT(ggml_can_mul_mat(a, b))`). Applying it to the
  SNOFS-merged entries applies SNOFS twice.
- Qwen-Image-Edit-2509-Q3_K_M segfaults without `offload_params_to_cpu: true`.

A failing allocation shows up as `cudaMalloc failed: out of memory` in the log, followed by `ValueError: NULL pointer
access` or a segfault from the Python binding, not a clean exception.

## lora_model_dir lives in config.yaml (image_init_defaults)

`image_init_defaults` in `config/config.yaml` is merged underneath every model's `init_options` (the model's own value
wins), so `lora_model_dir` is set once. It only matters when a prompt contains a `<lora:name:weight>` tag; the binding
resolves relative names against it and silently skips (logs "Can not find lora") if the file isn't there.
