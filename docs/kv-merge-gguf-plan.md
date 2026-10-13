# Plan: Flux.2-Klein-9B-KV-Merge (SDNQ uint4) -> GGUF for stable-diffusion.cpp

> **Note:** the conversion scripts mentioned below are not part of this repo. They live in `~/ai/tmp/kvconv/` and
> need the SD.Next venv (`~/ai/sdnext/venv`) plus its `sdnq` source, so they only run in that environment.

Goal: a GGUF of vladmandic/Flux.2-Klein-9B-KV-Merge-sdnq-hadamard-uint4 that loads in
`prompt-runner` (stable-diffusion-cpp-python 0.4.7), because it did anatomy noticeably better in SD.Next.
Friction log: `kv-merge-gguf-frictions.md`.

## What the research established
- Source = only `transformer/diffusion_pytorch_model.safetensors` (6.16GB) + `config.json`. Text encoder,
  VAE, tokenizer are NOT part of the conversion: reuse stock `Qwen3-8B-Q4_K_M.gguf` + `flux2-vae.safetensors`.
- Format: SDNQ uint4, Hadamard group 256, no SVD, static+dynamic quant. 144 linear layers stored as
  packed-uint8 `weight` + F32 `scale` + F32 `zero_point` (asymmetric, group size 64 along input dim).
  Norms / embedders / modulation / proj_out stay BF16. 8 double + 24 single blocks = ordinary Klein 9B shape.
- Dequantization must use SDNQ's own code (`SDNQLayer.dequantize()`), not a re-implementation - the Hadamard
  inverse is the part that silently produces noise if wrong. SD.Next vendors sdnq 0.2.8 at
  `~/ai/sdnext/extensions-builtin/sdnq/src` = the exact code that generated the good images. Use that.
- "KV" = BFL's Klein 9B-KV variant (reference-token KV cache for fast multi-ref editing; reference tokens attend
  causally). Plain text-to-image has no reference tokens -> identical math to normal Klein. Same weights layout.
  "Merge" is Vlad's own change; undocumented (model card empty).
- stable-diffusion.cpp wants BFL-native names (`double_blocks.N.img_attn.qkv`, ...), not diffusers names.
  diffusers' own `scripts/convert_flux2_to_diffusers.py` defines the mapping; we invert it:
  q/k/v -> cat -> qkv; `norm_q/k.weight` -> `norm.query/key_norm.scale`; `norm_out.linear` -> swap halves back
  -> `final_layer.adaLN_modulation.1`; renames for ff/proj/embedders/modulation (`.linear`->`.lin`).
- Known-good template for names + shapes + dtypes: `/home/shared/ai/diffusion/gguf/flux-2-klein-9b-Q4_K_M.gguf`
  (201 tensors, native naming already - no patch script needed).
- Re-quantizing a dequantized uint4 model is "lossy twice" only in theory: the weights already sit on a uint4
  grid, so Q8_0 is ~transparent and Q5_K adds modest error. Result ~= what SD.Next ran.
- gguf-py cannot produce K-quants; stable-diffusion.cpp's own `sd_cpp.convert()` can (used before for SNOFS).

## Steps (KISS)
1. **Env**: no new venv. Run with `~/ai/sdnext/venv/bin/python` (torch+cu132, diffusers 0.41 dev, gguf, safetensors)
   with `PYTHONPATH=~/ai/sdnext/extensions-builtin/sdnq/src`. Scripts live in `~/ai/tmp/kvconv/`. SD.Next untouched.
2. **Smoke test A - dequant is correct**: load transformer via diffusers (registers SDNQ), dequantize a few
   layers, check finite/sane stats, and correlate against the same tensors from the stock unsloth Klein 9B GGUF
   (gguf-py dequantize). Merge is a derivative of Klein 9B so correlation should be high; with the Hadamard
   inverse deliberately skipped it should collapse. This proves the dequant path before any big write.
3. **Convert -> F16 safetensors (native names)**: stream tensor-by-tensor (no 18GB in RAM): dequantize on GPU,
   map diffusers->native names, write a safetensors file by hand (header + sequential bytes). Linear weights F16,
   BF16 tensors copied, norms F32 (mirrors the unsloth template).
4. **Smoke test B - structure**: tensor name set + shapes must equal the unsloth GGUF exactly.
5. **Quantize**: `sd_cpp.convert()` -> Q8_0 first (diagnostic: near-lossless), then Q5_K (size parity with other
   configs, ~6.4GB).
6. **Smoke test C - generation**: load in prompt-runner config with stock Qwen3-8B + flux2-vae, anatomy prompt
   seeds 111/222/333, compare with the SD.Next output and the stock Klein 9B column.
7. Add a config entry; clean up the 18GB intermediate if the GGUFs are good.

## Risks / fallbacks
- Hadamard/layout wrong -> noise: caught by smoke test A. Fallback: let SDNQ dequantize the whole model.
- diffusers can't load the model on CPU: load tensors manually through sdnq's loader instead.
- sd.cpp bundled in the 0.4.7 binding rejects it: check Q8 first; fallback to leaving as-is and log it.
- RAM (31GB, ~12GB free): hence streaming writer.

## Outcome (2026-10-10) - done, all smoke tests passed
- A (dequant): 4 layers checked, finite, std matches stock Klein 9B, corr 0.99 (0.06 if Hadamard inverse skipped).
- B (structure): 201 tensors, names + shapes == unsloth Klein 9B GGUF.
- C (generation): Q8_0 and Q5_K load in prompt-runner, 4 steps ~29s @768, coherent anatomy plates; Q5_K ~= Q8_0.
- Files: `/home/shared/ai/diffusion/gguf/flux-2-klein-9b-kv-merge-Q8_0.gguf` (9.7GB), `...-Q5_K.gguf` (6.7GB, modulation BF16).
  Configs: `flux-2-klein-9b-kv-merge-Q8_0` / `-Q5_K` in prompt-runner `config/image_models.yaml`.
- Scripts (rerunnable, ~1 min total): `~/ai/tmp/kvconv/{common,smoke_a,convert_to_safetensors,patch_modulation_bf16}.py`.
  Run with `PYTHONPATH=~/ai/tmp/kvconv:~/ai/tmp/kvconv/shim ~/ai/sdnext/venv/bin/python ...`.
- Intermediate 18GB F16 safetensors deleted (regenerate in ~26s).
- Not verified: pixel-level parity with SD.Next's output (different sampler/seed handling); the merge's "Merge" content is unknown;
  reference-image editing with this model (KV causal attention for ref tokens) is not replicated by sd.cpp - t2i is unaffected.
- Q8_0 rebuild recipe: `convert_to_safetensors.py` (~30s, 18GB intermediate) -> `sd_cpp.convert(..., GGML_TYPE_MAP['q8_0'])` (no modulation patch needed), then delete the intermediate. Both Q8_0 and Q5_K are kept.
