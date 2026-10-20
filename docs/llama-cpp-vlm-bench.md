# llama.cpp VLM trials: Gemma4, Qwen3.6, Qwen3.8 (2026-10-16)

Speed of the three local vision models under `llama serve` on the RTX 5070 Ti (16 GB), 31 GB RAM.

## Key takeaways

- **All three run fully usable on the GPU.** The two MoE models do 80-100 tok/s; the dense one does about 12.
- **Both MoE models handle 128K context.** Generation falls 35-45% from short-prompt speed (Gemma 82 -> 49, Qwen3.6 99 -> 58 tok/s). A 123K-token prompt is read in 2-3 minutes.
- **The dense Qwen3.8 27B does not scale.** 11.6 tok/s at 8K, 5.5 at 32K, 2.8 at 64K, 1.2 at 128K. Keep it to short contexts, and use it when quality matters more than speed.
- **Gemma reads faster** (prompt processing 1.5x Qwen3.6 at 8K, 3x on images in the short test). **Qwen3.6 generates faster** (about 20% at 8K-64K).
- **The first request after a model loads is slow** (weights page in from disk): up to 40-80% lower prompt speed on the MoE models. Send a throwaway warm-up request before timing or batch work.
- **VRAM is full by design** (14-15 GB with `--fit`). Run llama-server and sd-server separately.
- Quality was not judged here. These are speed numbers only, one run per cell unless noted.

## Setup

| Item | Value |
|---|---|
| Build | `llama` 0.6.0-dev, build 11630 (commit 6868a52cb), server is `llama serve` |
| Flags | `-m <model> --mmproj <mmproj> -c <ctx> --port 8089`; `--fit` is on by default |
| Request | `/v1/chat/completions`, thinking off (`chat_template_kwargs: {"enable_thinking": false}`) |
| KV cache | f16 (default) |
| Models | Q4_K_M GGUFs with their mmproj files |

| Model | Type | Weights |
|---|---|---|
| Gemma4-26B-A4B QAT Uncensored (HauhauCS Balanced) | MoE, 4B active | Q4_K_M |
| Qwen3.6-35B-A3B Uncensored (HauhauCS Aggressive) | MoE, 3B active | Q4_K_M |
| Qwen3.8-27B Uncensored | dense | Q4_K_M |

Notes on the flags:
- `-c` is the context window in tokens (prompt + image tokens + output). Default `0` takes the model's native size, and `--fit` then shrinks it to fit VRAM.
- `--fit` is the default, so it does not need to be passed.

## 1. Short prompts (context 8192)

Load time, a 400-token story from a 26-token prompt, and a 300-token description of one image.

| Model | Load | Text gen (tok/s) | Image gen (tok/s) | Image prompt (tok/s) | VRAM (GB) |
|---|---|---|---|---|---|
| Gemma4 26B-A4B | 24 s | 89 | 90 | 299 | 15.0 |
| Qwen3.6 35B-A3B | 26 s | 79 | 93 | 100 | 14.6 |
| Qwen3.8 27B (dense) | 22 s | 12.7 | 12.4 | 150 | 14.1 |

The image prompt was 142 tokens for Gemma and 276 for Qwen, so the image-prompt column is not a like-for-like comparison.

## 2. Context sweep

One prompt filled to about 90% of `-c`, then 128 tokens generated. 8K is the warm rerun (average of runs 2-3, see section 3); the other rows are from the first sweep.

**Generation speed (tok/s)**

| Context | Gemma4 | Qwen3.6 | Qwen3.8 (dense) |
|---|---|---|---|
| 8K | 81 | 99 | 11.6 |
| 32K | 62 | 88 | 5.5 |
| 64K | 54 | 66 | 2.8 |
| 128K | 49 | 58 | 1.2 |

**Prompt processing (tok/s)**

| Context | Gemma4 | Qwen3.6 | Qwen3.8 (dense) |
|---|---|---|---|
| 8K | 1,575 | 1,029 | 858 |
| 32K | 1,325 | 798 | 728 |
| 64K | 1,149 | 759 | 551 |
| 128K | 969 | 685 | 394 |

**Time to read a 123K-token prompt (128K row):** Gemma 129 s, Qwen3.6 181 s, Qwen3.8 374 s.

VRAM stayed at 14.0-15.0 GB in every run, so context size did not change it with `--fit`.

## 3. Cold start

The first sweep's 8K row was the first request after each model loaded. Rerun with a warm-up request, then three timed 8K requests (7,050 prompt tokens):

| Model | Cold: prompt / gen | Warm (runs 2-3): prompt / gen |
|---|---|---|
| Gemma4 | 1,111 / 67 | 1,575 / 81-82 |
| Qwen3.6 | 566 / 78 | 1,030 / 98-100 |
| Qwen3.8 (dense) | 862 / 11.8 | 850-870 / 11.5-11.8 |

The weights are memory-mapped, so the first request pays to page them in. The dense model reads all weights on every token, so it was already warm. Run 1 (right after a "hi" warm-up) was still partly cold for the MoE models.

The 32K-128K rows ran with the page cache already warm from the 8K run, but they were not rerun.

## Caveats

- Single runs, short outputs (50-130 tokens): treat gaps under about 20% as noise.
- Synthetic filler text for the long prompts, not real code or documents.
- Thinking was off. Thinking-on speeds, quality, and VLM accuracy were not tested.
- Not tested: q8_0 KV cache, `--n-cpu-moe`, MTP/speculative decoding, running next to sd-server.
