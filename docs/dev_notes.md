# Informal Developer Notes

## vae_decode_only and img2img

When an image model is initialized with vae_decode_only=true, it can only be used for txt2img. Attempting to use it for img2img will raise an error.