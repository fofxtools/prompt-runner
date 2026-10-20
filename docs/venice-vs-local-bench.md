# Local llama.cpp vs Venice API: speed comparison (2026-10-16)

Same prompts and the same streaming client against the local `llama serve` (RTX 5070 Ti 16 GB) and the Venice OpenAI-compatible API. Local-only results and setup are in [llama-cpp-vlm-bench.md](llama-cpp-vlm-bench.md).

## Key takeaways

- **Venice is faster almost everywhere.** The one exception is Gemma text generation, where the closest uncensored slug matches your GPU (65 vs 70 tok/s).
- **Reading long prompts is the biggest gap.** A 126K-token prompt gets its first token in 6-29 s on Venice vs 123-169 s locally (4-27x).
- **Qwen3.6:** Venice generates about 2.5x faster (197 vs 79 tok/s) and holds 190-280 tok/s out to 126K.
- **Qwen3.8 27B dense:** Venice is 12-15x faster (124 vs 10 tok/s). Locally it is only usable at 8K or less; on Venice it behaves like any other model.
- **E2EE slugs are slower and text-only** (and capped at 64K / 32K). They work with a plain request (no client-side encryption needed), but there is no speed reason to use them.
- **Venice speed varies a lot run to run** (e.g. 58-92 tok/s on the same Gemma request), and one 126K request returned a 504 before succeeding on retry. Use medians.
- Speed only. The Venice models are not your exact fine-tunes or quantizations, so quality is not compared.

## Equivalents tested

| Local model | Venice slug | Notes | Output $/M tokens |
|---|---|---|---|
| Gemma4 26B-A4B uncensored | `gemma-4-uncensored` | Closest uncensored Gemma with vision, 256K. Size not stated. | 0.50 |
| | `google-gemma-4-26b-a4b-it` | Stock model, same architecture (speed reference) | 0.40 |
| | `e2ee-gemma-4-26b-a4b-uncensored-p` | Explicit 26B-A4B uncensored, text-only, 64K | 0.88 |
| Qwen3.6 35B-A3B | `qwen3-6-35b-a3b` | Stock, not uncensored (no uncensored 35B-A3B exists), 256K | 1.00 |
| | `e2ee-qwen3-6-35b-a3b` | FP8, text-only, 32K | 1.18 |
| Qwen3.8 27B | `qwen-3-8-27b` | Tagged uncensored, vision, 262K | 3.20 |

`qwen-3-6-plus` ("Qwen 3.6 Plus Uncensored") is a different, much larger model and was not used. Prices are from the API model list.

## Generation speed (tok/s, median)

| Model | Local text | Venice text | Local image | Venice image |
|---|---|---|---|---|
| Gemma (`gemma-4-uncensored`) | 70 | 65 | 71 | 139 |
| Gemma (stock `google-gemma-4-26b-a4b-it`) | n/a | 123 | n/a | 105 |
| Gemma E2EE | n/a | 54 | n/a | no vision |
| Qwen3.6 (`qwen3-6-35b-a3b`) | 79 | 197 | 85 | 220 |
| Qwen3.6 E2EE | n/a | 112 | n/a | no vision |
| Qwen3.8 (`qwen-3-8-27b`) | 10 | 124 | 10 | 153 |

Time to first token on a short prompt was 0.5-0.7 s on Venice for every slug.

## Time to first token on long prompts (seconds)

| Prompt | Local Gemma | Venice `gemma-4-uncensored` | Local Qwen3.6 | Venice Qwen3.6 | Local Qwen3.8 | Venice Qwen3.8 |
|---|---|---|---|---|---|---|
| 8K | 8.2 | 1.2 | 11.4 | 0.8 | 12.5 | 1.1 |
| 30K | 26.5 | 3.4 | 40.9 | 1.4 | 42.1 | 3.1 |
| 63K | 56.2 | 9.2 | 82.1 | 2.7 | not run | 6.6 |
| 126K | 122.6 | 28.8 | 169.2 | 6.2 | not run | 18.0 |

## Generation speed at long context (tok/s)

| Prompt | Local Gemma | Venice `gemma-4-uncensored` | Local Qwen3.6 | Venice Qwen3.6 | Local Qwen3.8 | Venice Qwen3.8 |
|---|---|---|---|---|---|---|
| 8K | 58 | 78 | 86 | 262 | 8.2 | 171 |
| 30K | 66 | 45 | 80 | 216 | 5.7 | 173 |
| 63K | 67 | 53 | 80 | 281 | not run | 151 |
| 126K | 61 | 49 | 71 | 191 | not run | 122 |

Output was about 50 tokens, so these are noisy; read them as ballpark.

## Method

- One streaming client (`/chat/completions`, `stream_options.include_usage`) for both sides. TTFT is the time to the first content token; decode speed is `(completion_tokens - 1) / (last - first token time)`.
- Venice: `venice_parameters.disable_thinking=true` and `include_venice_system_prompt=false`. Local: `chat_template_kwargs.enable_thinking=false`.
- Long prompts are filler text starting with a unique random string, so prefix caching cannot hide the prompt-reading time. Where Venice reports cached tokens (`gemma-4-uncensored`) it was 0; the other slugs did not report the field.
- Per model: 3 text runs (400 tokens), 2 image runs (300 tokens), long prompts of about 8K / 30K / 61K / 120K tokens (2 runs up to 30K, 1 beyond), 128 tokens generated. Slugs that exceed a model's context skip the larger sizes.
- Local: one fresh server per model for long prompts, with `-np 1 -cram 0`, and `-c 131072` (`-c 40960` for Qwen3.8).

## Caveats

- **Local server config matters.** At `-c 131072` local generation is lower than in the first benchmark (Gemma 70 vs 89, Qwen3.6 79 vs 99 tok/s at `-c 8192`). A fair local number needs `-c` sized for the use case.
- **Keep prompt caching off when benchmarking locally.** A long-lived server stored every prompt in host RAM (default 8 GiB), which squeezed the page cache and produced a 1,084 s time to first token for a 126K prompt (116 tok/s prefill vs about 745 on a fresh server). The long rows above come from the fresh-server rerun.
- Short outputs, one or two runs per cell, and hosted-service variance: treat gaps under about 20% as noise.
- Thinking was off everywhere. Thinking-on speed and any quality comparison were not tested.
