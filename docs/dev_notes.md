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