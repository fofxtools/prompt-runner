# Frictions: KV-Merge GGUF conversion

Format: what happened -> what I did -> outcome.

1. **sdnq not in the SD.Next venv** -> `pip install` gave 0.2.7 but the checkpoint was made/used with SD.Next's vendored
   0.2.8 (`extensions-builtin/sdnq/src`). Used the vendored copy via PYTHONPATH; removed the pip copy. Outcome: no mismatch.

2. **diffusers refused to load: "requires the sdnq library"** -> diffusers detects sdnq via pip metadata
   (`importlib.metadata`), and SD.Next's vendored copy has no dist-info. Rather than pip-install a (different) version,
   created `~/ai/tmp/kvconv/shim/sdnq-0.2.8.dist-info/METADATA` (Name+Version only) and put `shim/` + the vendored
   `src/` on PYTHONPATH. Outcome: the vendored 0.2.8 code is what runs, metadata check passes.

3. **`.venv` python `-I` flag ignores cwd/PYTHONPATH-for-local-modules** -> `from common import *` failed. Dropped `-I` for the
   SD.Next-venv scripts and set `PYTHONPATH=~/ai/tmp/kvconv:~/ai/tmp/kvconv/shim` instead. (prompt-runner scripts still use `-I`.)

4. **Where does `sdnq`'s dequantize handle layout?** `SDNQLayer.dequantize()` passes `skip_quantized_matmul=use_quantized_matmul`
   (which transposes for the matmul kernel). Re-implementing it risks a transposed/unrotated result. Called the dequantizer
   exactly like the library does (`d(**get_sdnq_params(L), skip_quantized_matmul=d.use_quantized_matmul)`). Verified by smoke
   test A: shape == stock Klein shape, corr 0.99 with Hadamard inverse vs 0.06 without.

5. **`sd_cpp.convert()` crashes with `tensor_type_rules`** (`GGML_ASSERT(obj_new) failed` in ggml.c; the bundled
   stable-diffusion.cpp's pre-sized context can't take a rule that makes tensors bigger). Wanted to keep the 3 modulation
   linears BF16 (stock unsloth GGUF does; plain Q5_K quantized them). Workaround: wrote `kvconv/patch_modulation_bf16.py`
   which swaps those 3 tensors back to exact BF16 (copied from the F16 safetensors) via gguf-py, same pattern as
   `prompt-runner/scripts/patch_gguf_head_dim.py` (removed in 0.2.0; it is in git at tag v0.1.6). Verified: the 3 are bit-exact, other 198 tensors byte-identical, then
   generation OK. sd.cpp-written GGUFs have no KV metadata, so the writer is created with `arch="flux2"` only.

6. **Not a friction but worth knowing:** `sd_cpp.convert()` on an 18GB F16 safetensors takes ~35s and fits in RAM
   (31GB box, ~12GB free) because the loader mmaps; my own writer streams tensor-by-tensor so RAM never held 18GB.
