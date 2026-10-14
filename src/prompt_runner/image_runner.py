"""Image generation runner logic."""

import base64
import json
import secrets
import shutil
import time
from pathlib import Path
from typing import Any

from .sdcpp_server import SdCppError, SdCppServer, engine_version
from .utils import (
    create_result_structure,
    generate_run_identifiers,
    merge_image_options,
    sanitize_fs_name,
)

MAX_SEED = 2**31 - 1


def save_image_summary(
    run_id: str,
    run_dir_name: str,
    created_at: str,
    results_dir: str,
    prompts: list[dict[str, Any]],
    models: list[dict[str, Any]],
    model_timings: dict[str, float],
    engine: str,
    failures: list[dict[str, Any]],
) -> None:
    """
    Save the run summary metadata to summary.json.

    Args:
        run_id: The unique run identifier with timestamp + suffix
        run_dir_name: The filesystem-safe run directory name
        created_at: The ISO-8601 timestamp when the run was created
        results_dir: The base results directory path
        prompts: List of prompt dictionaries
        models: List of model dictionaries
        model_timings: Dict mapping model names to elapsed time in seconds
            (server start and model load included)
        engine: The sd-server version string
        failures: One dict per failure (model, prompt, kind, error, exit_code, log)

    Raises:
        FileNotFoundError: If the run directory does not exist
    """
    results_path = Path(results_dir)
    run_path = results_path / run_dir_name
    summary_path = run_path / "summary.json"

    if not run_path.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_path}")

    # Extract prompt IDs and model names
    prompt_ids = [prompt["id"] for prompt in prompts]
    model_names = [model["name"] for model in models]

    # Create summary structure
    summary = {
        "run_id": run_id,
        "created_at": created_at,
        "image": {
            "engine": engine,
            "prompt_count": len(prompts),
            "model_count": len(models),
            "prompts": prompt_ids,
            "models": model_names,
            "model_timings": model_timings,
            "failures": failures,
        },
    }

    # Write summary to file
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def record_failure(
    failures: list[dict[str, Any]],
    model_name: str,
    prompt_id: str | None,
    error: SdCppError,
    log: str,
) -> None:
    """Print a failure and append it to the run's failure list (prompt_id None = server start)."""
    print(f"    FAILED ({error.kind}): {error.message}")
    failures.append(
        {
            "model": model_name,
            "prompt": prompt_id,
            "kind": error.kind,
            "error": error.message,
            "exit_code": error.exit_code,
            "log": log,
        }
    )


def prepare_request(
    options: dict[str, Any], inputs_dir: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Resolve the seed and inline the input images of a native request.

    `init_image`, `mask_image` and `ref_images` are file paths in the config.
    Each is copied into inputs_dir and sent to the server as base64.

    Args:
        options: The merged request fields (not modified)
        inputs_dir: Where input files are copied to (run_dir/inputs/<prompt_id>)

    Returns:
        (request, recorded): the request to send, and the same request with the
        images replaced by their file names relative to the run directory.

    Raises:
        OSError: If an input file cannot be read
    """
    recorded = dict(options)
    seed = options.get("seed")
    if seed is None or seed == -1:
        recorded["seed"] = secrets.randbelow(MAX_SEED)
    request = dict(recorded)

    def inline(key: str, path: str, suffix: str = "") -> tuple[str, str]:
        src = Path(path)
        inputs_dir.mkdir(parents=True, exist_ok=True)
        dest = inputs_dir / f"{key}{suffix}_{src.name}"
        shutil.copyfile(src, dest)
        relative = f"inputs/{inputs_dir.name}/{dest.name}"
        return base64.b64encode(dest.read_bytes()).decode(), relative

    for key in ("init_image", "mask_image"):
        if options.get(key):
            request[key], recorded[key] = inline(key, options[key])
    if options.get("ref_images"):
        pairs = [
            inline("ref_images", path, f"_{i}")
            for i, path in enumerate(options["ref_images"])
        ]
        request["ref_images"] = [b64 for b64, _ in pairs]
        recorded["ref_images"] = [rel for _, rel in pairs]
    return request, recorded


def run_image_eval(
    config: dict[str, Any],
    prompts: list[dict[str, Any]],
    models: list[dict[str, Any]],
    mode_filter: str = "all",
) -> str:
    """
    Run image generation evaluation across prompts and models.

    Starts one sd-server per model, runs the (mode-filtered) prompts through it
    and stops it. A failed job is recorded and the run continues; if the server
    dies the rest of that model is skipped. Failures end up in summary.json.

    Args:
        config: Configuration dictionary containing results_dir, sd_server (path
            to the binary), and optionally image_server_args and
            image_generation_defaults
        prompts: List of prompt configurations from image_prompts.yaml
        models: List of model configurations from image_models.yaml
        mode_filter: Filter prompts by mode ("txt2img", "img2img", or "all")

    Returns:
        run_id: The unique run identifier

    Raises:
        ValueError: If mode_filter is invalid or config has no sd_server
    """
    if mode_filter not in ("txt2img", "img2img", "all"):
        raise ValueError(
            f"Invalid mode_filter: {mode_filter}. Must be 'txt2img', 'img2img', or 'all'"
        )
    binary = config.get("sd_server")
    if not binary:
        raise ValueError(
            "Config must contain 'sd_server' (path to the sd-server binary)"
        )

    selected = [p for p in prompts if mode_filter in ("all", p["mode"])]
    engine = engine_version(binary)

    # Generate run identifiers
    run_id, run_dir_name, created_at = generate_run_identifiers()
    results_dir = config["results_dir"]

    # Create result directory structure
    run_path = create_result_structure(run_dir_name, results_dir)

    global_defaults = config.get("image_generation_defaults")
    server_defaults = config.get("image_server_args")

    model_timings: dict[str, float] = {}
    failures: list[dict[str, Any]] = []

    # Process each model
    for model in models:
        if not selected:
            break  # no prompt matches the mode filter: don't start any server
        model_name = model["name"]
        sanitized_model_name = sanitize_fs_name(model_name)
        log_path = run_path / "server_logs" / f"{sanitized_model_name}.log"
        server_args = merge_image_options(server_defaults, model["server_args"])
        log_rel = str(log_path.relative_to(run_path))
        print(f"Starting server: {model_name}")

        start_time = time.time()
        try:
            with SdCppServer(binary, server_args, log_path) as server:
                for prompt in selected:
                    prompt_id = prompt["id"]
                    print(f"  Generating: {prompt_id} ({prompt['mode']})")

                    options = merge_image_options(
                        global_defaults,
                        model.get("generation_options"),
                        prompt.get("options"),
                    )
                    try:
                        request, recorded = prepare_request(
                            options, run_path / "inputs" / prompt_id
                        )
                    except OSError as e:
                        record_failure(
                            failures,
                            model_name,
                            prompt_id,
                            SdCppError("job_failed", f"input error: {e}"),
                            log_rel,
                        )
                        continue

                    gen_start = time.time()
                    try:
                        images = server.generate(request)
                    except SdCppError as e:
                        record_failure(failures, model_name, prompt_id, e, log_rel)
                        if e.kind == "server_died":
                            break
                        continue
                    elapsed = round(time.time() - gen_start, 3)

                    prompt_dir = run_path / "image" / prompt_id
                    prompt_json_dir = prompt_dir / "json"
                    prompt_json_dir.mkdir(parents=True, exist_ok=True)
                    for i, png in enumerate(images):
                        img_filename = f"{sanitized_model_name}_{i}.png"
                        (prompt_dir / img_filename).write_bytes(png)

                        # A batch uses seed, seed+1, ...
                        metadata = {
                            "run_id": run_id,
                            "created_at": created_at,
                            "mode": prompt["mode"],
                            "model": {"name": model_name},
                            "prompt": {"id": prompt_id},
                            "engine": engine,
                            "server_args": server.argv,
                            "request": recorded,
                            "seed": recorded["seed"] + i,
                            "elapsed_seconds": elapsed,
                        }
                        json_path = prompt_json_dir / f"{sanitized_model_name}_{i}.json"
                        with open(json_path, "w", encoding="utf-8") as f:
                            json.dump(metadata, f, indent=2)

                        print(f"    Saved: {img_filename}")
        except SdCppError as e:
            # The server failed to start (or died on its way up).
            record_failure(failures, model_name, None, e, log_rel)

        model_timings[model_name] = round(time.time() - start_time, 3)

    # Save run metadata with timings
    save_image_summary(
        run_id,
        run_dir_name,
        created_at,
        results_dir,
        prompts,
        models,
        model_timings,
        engine,
        failures,
    )

    return run_id
