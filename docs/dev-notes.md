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