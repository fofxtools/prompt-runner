# Rendered text in children's-book pages: hosted APIs vs local Qwen Image 2.1

Date: 2026-10-20. Question: which image models can render a paragraph (or three) of storybook text correctly, inside a finished page,
and how good does the page look overall?

## Key takeaways

1. **The newest OpenAI models render paragraphs essentially perfectly, even at `low` quality.** gpt-image-2.5 (Sunburst, Flare) and
   gpt-image-2 scored 8/8 on strict OCR at low, medium and high, and had no text flaws in the blind review. `low` costs about **$0.006 per
   1024² image** (gpt-image-2: $0.0065).
2. **Paying for a higher tier buys nothing visible for text or page quality.** gpt-image-2 `high` costs $0.21 (35x low) and takes about 2
   minutes per image. Do not use `high`, `xhigh` or `max` for text.
3. **Older OpenAI models need a higher tier, and some never get there.** gpt-image-1.5 needs `high`; chatgpt-image-latest needs `medium`
   and still clipped the last line twice; gpt-image-1 fails at every tier; gpt-image-1-mini only works at `high`.
4. **Local Qwen Image 2.1 (all four variants tested) is not good enough for finished text pages.** Strict OCR: only Turbo gets 3 paragraphs right (4/4); no variant gets more than 1/4 on 1 paragraph. It repeatedly misspells ("spreed",
   "hys", "pickic"), sometimes writes microscopic text in a mostly empty panel, and two images had no text at all. Turbo handles 3
   paragraphs best on strict OCR (4/4 on the original test), but looks flat next to the OpenAI pages.
5. **DeepInfra Seedream is a fallback only.** 4.5 and 5.0 Pro reproduce the text but the pages are less polished, with smaller subjects
   and occasional layout oddities. 4.5 added quotation marks around the text in 2 of 6 pages. P-Image (cheap, 3 s) fails (0/8).
6. **OCR alone overstates quality.** A VLM reader autocorrects typos and read half-clipped lines, so it passed images an editor would
   reject. Use OCR as a screen, then a blind visual check.

## Suggested tier list (for storybook pages with in-image text)

| Tier | Model and setting | Cost per image | Why |
|---|---|---|---|
| **S** | gpt-image-2.5 Flare `low` | $0.006 | 8/8 OCR, no flaws, fastest of the top group (about 11 s) |
| **S** | gpt-image-2 `low` | $0.0065 | same quality, about 14 s |
| **S** | gpt-image-2.5 Sunburst `low` or `medium` | $0.006 / $0.014 | no flaws, best average page score in the review |
| A | gpt-image-2.5 Flare / Sunburst `medium`, gpt-image-2 `medium` | $0.014 / $0.054 | same quality; medium on gpt-image-2 costs 9x low for no gain |
| B | gpt-image-1.5 `high`, chatgpt-image-latest `high` | about $0.15 | correct text, strong art, but 25x the price of the S tier |
| C | Seedream 5.0 Pro (DeepInfra), Seedream 4.5 | $0.05-0.10 / $0.04 | text right, pages weaker, 4.5 adds stray quotes or rules |
| D | Qwen Image 2.1 Turbo / Lightning (local) | free, 5-9 s | usable for art-only pages; text has typos on most pages |
| F | gpt-image-1 (any), gpt-image-1-mini below `high`, gpt-image-1.5 `low`, chatgpt-image-latest `low`, P-Image, Qwen 2.1 base Q4_K_M/Q6_K_XL | n/a | garbled, missing or tiny text |

Not worth running: `high`, `xhigh`, `max` on any gpt-image-2.x model, and chatgpt-image-latest `medium` (clipped last line).

## Results

### Strict OCR (llama-server Qwen3.8 27B, transcription prompt; 4 images per cell)

| Variant | 1 paragraph exact | 3 paragraphs exact |
|---|---|---|
| gpt-image-2.5 Sunburst low / medium / high | 4/4, 4/4, 4/4 | 4/4, 4/4, 4/4 |
| gpt-image-2.5 Flare low / medium / high | 4/4, 4/4, 4/4 | 4/4, 4/4, 4/4 |
| gpt-image-2 low / medium / high | 4/4, 4/4, 4/4 | 4/4, 4/4, 4/4 |
| gpt-image-1.5 low / medium / high | 0/4, 1/4, 4/4 | 0/4, 4/4, 4/4 |
| chatgpt-image-latest low / medium / high | 0/4, 4/4, 4/4 | 0/4, 4/4, 4/4 |
| gpt-image-1 low / medium / high | 0/4, 0/4, 0/4 | 0/4, 0/4, 0/4 |
| gpt-image-1-mini low / medium / high | 0/4, 0/4, 4/4 | 0/4, 0/4, 1/4 |
| Seedream 4.5 / 5.0 Pro | 4/4, 3/4 | 4/4, 4/4 |
| P-Image | 0/4 | 0/4 |
| Qwen 2.1 Turbo Q6_K_XL (local) | 1/4 | 4/4 |
| Qwen 2.1 Q4_K_M / Q6_K_XL / Lightning (local) | 1/4, 1/4, 0/4 | 0/4, 2/4, 0/4 |

Exact means the same words after lowercasing and dropping punctuation. Word recall per variant was also recorded
(0.04-1.00).

### Blind visual review (an editor's view, 15-18 variants per case)

Scored by eye on shuffled, unlabeled sheets: text accuracy (typos, clipping, stray marks) and overall page quality (1-5: composition,
character appeal, text panel design, legibility). 6 pages per hosted variant (fox and bear, 1 and 3 paragraphs, two samples; 4 for the
other 3 Qwen variants on the earlier cases), reviewed by one reader (the assistant), so treat the scores as relative.

| Variant | Pages | Text flaws | Avg page (1-5) |
|---|---|---|---|
| gpt-image-2.5 Sunburst medium | 6 | 0 | 4.5 |
| gpt-image-2.5 Sunburst high | 6 | 0 | 4.4 |
| gpt-image-2 medium | 6 | 0 | 4.4 |
| gpt-image-2.5 Flare low | 6 | 0 | 4.4 |
| gpt-image-2 low | 6 | 0 | 4.4 |
| gpt-image-2.5 Sunburst low | 6 | 0 | 4.3 |
| gpt-image-2.5 Flare medium | 6 | 0 | 4.3 |
| gpt-image-2 high | 6 | 0 | 4.3 |
| gpt-image-2.5 Flare high | 6 | 1 (comma for a period) | 4.3 |
| chatgpt-image-latest high | 6 | 0 | 4.3 |
| gpt-image-1.5 high | 6 | 0 | 4.3 |
| chatgpt-image-latest medium | 6 | 1 (last line clipped by the bottom edge) | 3.8 |
| Seedream 5.0 Pro | 6 | 0 | 3.6 |
| Seedream 4.5 | 6 | 3 (2 added quotation marks, 1 stray lines) | 3.4 |
| Qwen 2.1 Lightning (Q4_K_M) | 8 | 8 (typos) | 2.6 |
| Qwen 2.1 Turbo (Q6_K_XL) | 6 | 5 (typos) | 2.5 |
| Qwen 2.1 Q6_K_XL | 8 | 7 (typos, one page with no text, one with tiny text, one garbled) | 1.8 |
| Qwen 2.1 Q4_K_M | 8 | 8 (typos, one no text, one garbled) | 1.7 |

Observations:
- The top 11 are within 0.2 points of each other; page quality tracks the model family, not the quality tier.
- Text panel style differs by model: parchment with ruled dividers (gpt-image-2.x), plain cream, framed panel with sprigs
  (chatgpt-image-latest/1.5), 3-column text (one gpt-image-2 low page), a torn-edge overlay. The parchment-and-divider look reads most like a
  children's book.
- Qwen common failure: a bold condensed font, small text, big empty panel, photoreal rather than watercolor, and "spreed" for "spread" on
  nearly every page.

### Cost and speed (1024x1024, measured over 8 images per variant, USD from the API usage field)

| Variant | Cost per image | Median time |
|---|---|---|
| gpt-image-2.5 Flare low / medium / high | $0.006 / $0.014 / $0.054 | 11 / 12 / 22 s |
| gpt-image-2.5 Sunburst low / medium / high | $0.006 / $0.014 / $0.054 | 14 / 18 / 37 s |
| gpt-image-2 low / medium / high | $0.0065 / $0.054 / $0.21 | 14 / 37 / 113 s |
| gpt-image-1.5 low / medium / high | $0.017 / $0.048 / $0.149 | 6 / 12 / 28 s |
| chatgpt-image-latest low / medium / high | $0.024 / $0.049 / $0.148 | 7 / 12 / 28 s |
| gpt-image-1 low / medium / high | $0.011 / $0.043 / $0.168 | 10 / 17 / 47 s |
| gpt-image-1-mini low / medium / high | $0.003 / $0.009 / $0.034 | 8 / 11 / 26 s |
| Seedream 4.5 (DeepInfra, 2048² then downscaled) | about $0.04 listed | 13 s |
| Seedream 5.0 Pro (DeepInfra, 1024²) | $0.05-0.10 listed | 54 s |
| P-Image (DeepInfra) | about $0.005 listed | 3 s |
| Qwen 2.1 Turbo, local RTX 5070 Ti | free | 8 s (12 s first request) |

DeepInfra costs are the listed prices; its API does not report usage in the response.

## Follow-up: harder text (busy multi-character action scenes, about 100 words with dialogue)

Three scenes, each with 4-6 characters and an action beat (dragon rescue on a collapsing bridge, kitchen cake chaos, runaway pig in a
market), one ~100-word paragraph with quotation marks, parentheses, em-dashes, an ellipsis, an all-caps sound word and invented names.
2 samples per scene, 6 images per model, 1024x1024, blind page review plus the same strict OCR.

| Variant | Cost per image | Median time | Strict OCR exact (of 6) | Real text flaws (blind) | Avg page (1-5) |
|---|---|---|---|---|---|
| gpt-image-2 low | $0.0071 | 18 s | 4 | 0 | 4.8 |
| gpt-image-2.5 Flare medium | $0.0144 | 15 s | 4 | 0 | 4.5 |
| gpt-image-2.5 Sunburst low | $0.0071 | 17 s | 3 | 2 (one "Blew" for "Below", one stray mark) | 4.4 |
| gpt-image-2.5 Flare low | $0.0071 | 13 s | 5 | 1 (smudge near the text) | 4.3 |
| Qwen Image 2.1 Turbo (local) | free | 8 s | 0 | 6 of 6 pages | 2.7 |

Findings:
- **The OpenAI `low` models still render ~100 words with dialogue correctly.** The 20 OpenAI pages had 3 real defects in total (one
  wrong word, two stray marks). The remaining OCR misses were the invented name "Grizzlefen" read as "Grizzlefem", which looks like an OCR
  misread of small type, not a rendering error; I could not confirm it at full size on every image.
- **Medium did not help.** Flare medium costs twice as much as Flare low and was not measurably better.
- **The page quality order is within noise at the top** (4.3-4.8). gpt-image-2 low had the best art and layouts (framed panels, dynamic
  action poses, mid-leap dog).
- **Qwen Turbo failed on every page.** Typos in every paragraph ("fishmanger", "noldody", "woulpled", "frofting", "hinsf"), dropped
  parentheses ("more of a squeak, really" with no brackets), lost sentence breaks, a bold condensed font, and calmer compositions (the
  knight smiling on a static bridge, the cake standing upright in the kitchen scene). It also showed the weakest handling of action.
- **Duplicate characters:** one OpenAI kitchen page drew two beagles (a single dog in the text). Check for this in busy scenes.
- Total spend for 24 OpenAI images: $0.21.

### Editorial pass (whole-page quality, intent and reader view)

A second blind pass on the same 30 images, scored as an editor would: does the picture match the prompt (props, action, character count),
are anatomy and objects coherent, does a reader get the story. Issues a reviewer pointed out by hand were all real; 5 of 6 were also caught
blind, the one missed was Flare medium's knight with hands and torso outside the bridge rail (dragon_bridge sample 2).

| Variant | Avg page (1-5) | Recurring issues |
|---|---|---|
| gpt-image-2.5 Sunburst low | 4.5 | small text smear on one page, towel on a stick |
| gpt-image-2.5 Flare medium | 4.4 | extra dog (kitchen s2), anatomy slip on the bridge, oversized towel |
| gpt-image-2 low | 4.4 | rope "bridge" is a single diagonal rope (bridge s2), smudged text panel (market s2) |
| gpt-image-2.5 Flare low | 4.3 | extra dog (kitchen s2), hands outside the rail (bridge s2), stray mark before a quote |
| Qwen Image 2.1 Turbo | 2.5 | 6 of 6 pages fall short of the prompt: missing golden egg, no storm, calm poses, pig without the loaf, cake upright instead of tumbling, a literal rope lasso instead of a dish towel |

- **Typical OpenAI defects are small and random:** a duplicate character, an awkward hand position, a smudge. They come from the draw, not
  the model (Flare low and medium both drew the extra dog in the same scene). Budget for re-rolling about 1 page in 6 and checking the
  character count and the text panel by eye.
- **Qwen's problem is intent, not only text:** it softens the action (no peril, no spill, no chase) and drops key props, so even with
  correct text its pages would not tell the story.
- **Scores are close at the top (4.3-4.5):** pick on price and speed. Flare low ($0.007, 13 s) is the cheapest and fastest.

## Method (short)

- Prompt: a watercolor or 3D storybook scene with a cream text panel and the exact paragraph(s) quoted.
  Scenes: fox_picnic and bear_bedtime. 1 and 3 paragraphs, 2 samples each. OpenAI has no seed, so the samples are independent draws.
- Sizes: 1024x1024 for all hosted models except Seedream 4.5 (requires 2048x2048, downscaled for review). The OpenAI 2.5 models accept
  custom sizes with at least 655,360 pixels (816x816 is the smallest square, about 12% cheaper).
- OCR: llama-server qwen3.8-27b, temperature 0, bottom text crop upscaled 1.5x, prompt asks for letter-by-letter transcription.
- Review: shuffled letters, key hidden until the notes were saved; text crops and whole-page sheets.
- The generation, OCR and review scripts and the images live outside this repo; the reproduction recipe is the prompt, scenes, models and settings above.


## Caveats

- One prompt template, two scenes, 6-8 images per variant. The OCR differences between the top group and the rest are large; the order
  inside the top group is within noise.
- The page scores come from one reader and one pass; they are a relative ranking, not a measurement.
- The prompt was tuned during the earlier Qwen work, so Qwen may be at a small disadvantage.
- Only paragraph-length plain prose was tested. Dialogue with quotes, very long paragraphs, small type, or text over art may separate the
  top group; the harder-text follow-up above found the same ranking.
- OpenAI prices and model lineups change often; recheck the pricing page before budgeting. Quality `max` and `xhigh` of gpt-image-2.5 were
  deliberately not tested.
- Local Qwen image sizes and settings follow `config/image_models.yaml` defaults. A 512x512 Turbo smoke test (2 images) gave one near-correct
  1-paragraph page ("spreed", "checkerred") and an exact 3-paragraph page.
