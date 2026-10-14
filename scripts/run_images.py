#!/usr/bin/env python3
"""Generate prompts x seeds x models on managed sd-servers and write a comparison sheet.

    python scripts/run_images.py prompts.yaml -o out/ --models a,b [--seeds 1,2] [--size 1024]
                                 [--skip-existing] [--no-sheets] [--no-log]

prompts.yaml (or .json): a list of {id, prompt, <any native sd-server request field>}.
Models are names from config/image_models.yaml; server paths and defaults come from config/config.yaml.
Output: out/<id>__s<seed>__<model>.png, out/runs.jsonl (one line per image, failures included),
out/server_logs/, out/sheets/compare.png (rows = prompt x seed, columns = models).
--seeds replaces any seed set in a prompt. Experiments with more dimensions (LoRA, encoders, ...)
are a few lines of itertools.product around run_jobs(); see the README.
"""

import argparse
import itertools
import sys
from pathlib import Path

from prompt_runner.config import load_config, load_image_models, load_image_prompts
from prompt_runner.image_runner import Job, run_jobs
from prompt_runner.sheets import make_sheet
from prompt_runner.utils import sanitize_fs_name


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("prompts", help="prompts file (YAML or JSON)")
    ap.add_argument("-o", "--out", required=True, help="output folder")
    ap.add_argument("--models", required=True, help="comma-separated model names")
    ap.add_argument("--seeds", default="1", help="comma-separated seeds (default: 1)")
    ap.add_argument("--size", type=int, help="width and height, overriding the config")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--no-sheets", action="store_true")
    ap.add_argument("--no-log", action="store_true", help="do not write runs.jsonl")
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--models-file", default="config/image_models.yaml")
    args = ap.parse_args(argv)

    config = load_config(args.config)
    by_name = {m["name"]: m for m in load_image_models(args.models_file)}
    names = args.models.split(",")
    unknown = [n for n in names if n not in by_name]
    if unknown:
        sys.exit(f"Unknown model(s): {unknown}. See {args.models_file}")
    prompts = load_image_prompts(args.prompts)
    seeds = [int(s) for s in args.seeds.split(",")]
    out = Path(args.out)

    defaults = dict(config.get("image_generation_defaults") or {})
    if args.size:
        defaults.update(width=args.size, height=args.size)

    rows = [(p, s) for p in prompts for s in seeds]  # sheet rows
    jobs = []
    for (prompt, seed), name in itertools.product(rows, names):
        options = {k: v for k, v in prompt.items() if k != "id"} | {"seed": seed}
        file_name = (
            f"{sanitize_fs_name(prompt['id'])}__s{seed}__{sanitize_fs_name(name)}.png"
        )
        jobs.append(
            Job(by_name[name], options, out / file_name, {"prompt_id": prompt["id"]})
        )

    results = run_jobs(
        jobs,
        binary=config["sd_server"],
        out_dir=out,
        server_defaults=config.get("image_server_args"),
        generation_defaults=defaults,
        skip_existing=args.skip_existing,
        write_log=not args.no_log,
    )
    bad = [r for r in results if r["status"] != "ok"]
    print(
        f"\n{len(results) - len(bad)} image(s) ok, {len(bad)} failed or skipped. Output: {out}"
    )

    if not args.no_sheets:
        by_out = {
            (j.tags["prompt_id"], j.options["seed"], j.model["name"]): j.out
            for j in jobs
        }
        make_sheet(
            out / "sheets" / "compare.png",
            f"models: {', '.join(names)}",
            names,
            [f"{p['id']}\nseed {s}" for p, s in rows],
            lambda r, c: by_out[(rows[r][0]["id"], rows[r][1], names[c])],
        )
        print(f"Sheet: {out / 'sheets' / 'compare.png'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
