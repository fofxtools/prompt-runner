# Informal Developer Notes

## SDXL and resolution

SDXL is very resolution sensitive. It is designed for 1024x1024. The results are very poor at 512x512.

## Reasoning models and token use

With reasoning capable Qwen3 models (qwen3-vl:8b, qwen3:8b), you may receive empty or truncated outputs with `num_predict=512`. This is because the thinking process may use up all the tokens. Try increasing `num_predict`. `num_predict` can also be set to `-1`, for unlimited.

## fp8 safetensors checkpoints get upconverted to F16 on load - convert to GGUF instead

stable-diffusion.cpp's GGML backend doesn't keep fp8 weights at their stored size - it
upconverts them to F16 eagerly at load time, roughly doubling VRAM use for the whole
model (an fp8 checkpoint `N` bytes on disk needs roughly `2N` once loaded). This is a
GGML-backend-specific behavior, not a property of fp8 files in general: PyTorch-native
tools like ComfyUI keep fp8 weights at their actual size in memory and only cast
transiently per-op during compute, so the same file can work fine there even when it
doesn't fit here.

Fix: convert the checkpoint to a quantized GGUF once, ahead of time, rather than
loading the fp8 file directly every run, with the stable-diffusion.cpp CLI:

```bash
sd-cli -M convert -m /path/to/checkpoint.safetensors -o /path/to/output-Q5_K.gguf --type q5_K
```

## VAE files: diffusers format vs native format aren't interchangeable

stable-diffusion.cpp's VAE loader expects the original/native checkpoint tensor naming
(e.g. `first_stage_model.decoder.mid.block_1.conv2.bias`), not the HuggingFace Diffusers
`AutoencoderKL` save format (e.g. `decoder.mid_block.attentions.0.proj.bias`). These are
different module hierarchies, not just a prefix difference, so there's no quick rename
fix.

A reliable tell that you have the wrong format: a file named
`diffusion_pytorch_model.safetensors` living under a `vae/` subfolder in a HF repo is
the standard Diffusers save path - that's the one that won't work. The ComfyUI
"split_files" repackaging of
the same weights (e.g. `Comfy-Org/<Model>_ComfyUI/split_files/vae/`) uses native naming
and does work; use that source for VAEs (and similarly for diffusion models/text
encoders where a "split_files" or GGUF conversion exists) instead of the original
model author's own repo.

## FLUX.2 Klein notes

- `guidance` has no effect on Klein (no guidance embedding in the weights); only `txt_cfg: 1.0` and `sample_steps: 4` matter.
- `klein_snofs_v1_4` is a Klein 9B LoRA; on Klein 4B it aborts (`GGML_ASSERT(ggml_can_mul_mat(a, b))`). Applying it to the
  SNOFS-merged entries applies SNOFS twice.
