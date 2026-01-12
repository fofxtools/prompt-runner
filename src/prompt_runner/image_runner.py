"""Image generation runner logic."""

import json
import time
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image
from stable_diffusion_cpp import StableDiffusion

from .utils import (
    create_result_structure,
    generate_run_identifiers,
    merge_image_options,
    sanitize_fs_name,
)


def save_image_summary(
    run_id: str,
    run_dir_name: str,
    created_at: str,
    results_dir: str,
    prompts: List[Dict[str, Any]],
    models: List[Dict[str, Any]],
    model_timings: Dict[str, float],
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
            "prompt_count": len(prompts),
            "model_count": len(models),
            "prompts": prompt_ids,
            "models": model_names,
            "model_timings": model_timings,
        },
    }

    # Write summary to file
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def initialize_stable_diffusion(model_config: Dict[str, Any]) -> StableDiffusion:
    """
    Initialize a StableDiffusion instance from model configuration.

    Extracts all parameters from model_config["init_options"] and passes them
    to StableDiffusion. The "name" and "generation_options" fields are excluded.

    All init_options are passed through as-is. StableDiffusion will validate
    and fail fast if invalid.

    Args:
        model_config: Model configuration dictionary.
            - "name": model identifier (excluded, metadata only)
            - "init_options": dict containing StableDiffusion initialization parameters
              (diffusion_model_path, model_path, clip_l_path, vae_path,
              keep_clip_on_cpu, vae_decode_only, etc.)
            - "generation_options": (excluded, used during generation)

    Returns:
        Initialized StableDiffusion instance

    Raises:
        TypeError: If StableDiffusion receives invalid parameters (fail-fast)

    Examples:
        >>> config = {
        ...     "name": "flux1-schnell",
        ...     "init_options": {
        ...         "diffusion_model_path": "/path/to/model.gguf",
        ...         "clip_l_path": "/path/to/clip_l.safetensors",
        ...         "keep_clip_on_cpu": True
        ...     },
        ...     "generation_options": {
        ...         "cfg_scale": 1.0,
        ...         "sample_steps": 6
        ...     }
        ... }
        >>> sd = initialize_stable_diffusion(config)
    """
    # Extract init_options (StableDiffusion initialization parameters)
    if "init_options" not in model_config:
        raise ValueError(
            f"Model '{model_config.get('name', 'unknown')}' missing 'init_options' field"
        )

    # Pass all init_options to StableDiffusion
    # Let StableDiffusion validate parameters and fail fast if invalid
    return StableDiffusion(**model_config["init_options"])


def generate_image(
    sd: StableDiffusion,
    model_config: Dict[str, Any],
    prompt_config: Dict[str, Any],
    options: Dict[str, Any],
) -> List[Image.Image]:
    """
    Generate image(s) using a StableDiffusion instance.

    This function is a thin pass-through layer over
    stable-diffusion-cpp-python. All StableDiffusion parameters
    (including prompt text, batching, and generation options)
    must be provided via prompt_config["options"] and options.

    Orchestration-only fields such as "id" and "mode" are not
    passed to StableDiffusion.

    No validation or whitelisting is performed here.
    StableDiffusion is expected to validate parameters and
    fail fast if invalid.

    Args:
        sd: Initialized StableDiffusion instance
        model_config: Model configuration dictionary (unused; kept for symmetry)
        prompt_config: Prompt configuration dictionary containing an "options" dict
        options: Global / merged generation defaults

    Returns:
        List of generated PIL Image objects
    """
    params = dict(options)
    params.update(prompt_config["options"])

    return sd.generate_image(**params)


def run_image_eval(
    config: Dict[str, Any],
    prompts: List[Dict[str, Any]],
    models: List[Dict[str, Any]],
    mode_filter: str = "all",
) -> str:
    """
    Run image generation evaluation across prompts and models.

    Args:
        config: Configuration dictionary containing results_dir and image_generation_defaults
        prompts: List of prompt configurations from image_prompts.yaml
        models: List of model configurations from image_models.yaml
        mode_filter: Filter prompts by mode ("txt2img", "img2img", or "all")

    Returns:
        run_id: The unique run identifier

    Raises:
        ValueError: If mode_filter is not "txt2img", "img2img", or "all"
    """
    if mode_filter not in ("txt2img", "img2img", "all"):
        raise ValueError(
            f"Invalid mode_filter: {mode_filter}. Must be 'txt2img', 'img2img', or 'all'"
        )

    # Generate run identifiers
    run_id, run_dir_name, created_at = generate_run_identifiers()
    results_dir = config["results_dir"]

    # Create result directory structure
    run_path = create_result_structure(run_dir_name, results_dir)

    # Get global defaults
    global_defaults = config.get("image_generation_defaults")

    # Track timing per model
    model_timings = {}

    # Process each model
    for model in models:
        model_name = model["name"]
        print(f"Initializing model: {model_name}")

        # Start timing for this model
        start_time = time.time()

        # Initialize StableDiffusion
        sd = initialize_stable_diffusion(model)

        # Process each prompt
        for prompt in prompts:
            prompt_id = prompt["id"]
            prompt_mode = prompt["mode"]

            # Filter by mode
            if mode_filter != "all" and prompt_mode != mode_filter:
                continue

            print(f"  Generating: {prompt_id} ({prompt_mode})")

            # Create prompt subdirectories
            prompt_dir = run_path / "image" / prompt_id
            prompt_json_dir = prompt_dir / "json"
            prompt_dir.mkdir(parents=True, exist_ok=True)
            prompt_json_dir.mkdir(parents=True, exist_ok=True)

            # Merge generation options
            merged_options = merge_image_options(
                global_defaults,
                model.get("generation_options"),
                prompt.get("options"),
            )

            # Generate images
            images = generate_image(sd, model, prompt, merged_options)

            # Save each generated image
            sanitized_model_name = sanitize_fs_name(model_name)
            for i, img in enumerate(images):
                # Save PNG
                img_filename = f"{sanitized_model_name}_{i}.png"
                img_path = prompt_dir / img_filename
                img.save(img_path)

                # Save metadata JSON
                json_filename = f"{sanitized_model_name}_{i}.json"
                json_path = prompt_json_dir / json_filename

                # Build metadata with only set fields
                metadata = {
                    "run_id": run_id,
                    "created_at": created_at,
                    "mode": prompt_mode,
                    "model": {"name": model_name},
                    "prompt": {"id": prompt_id},
                    "generation_options": merged_options,
                }

                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(metadata, f, indent=2)

                print(f"    Saved: {img_filename}")

        # Record elapsed time for this model
        model_timings[model_name] = round(time.time() - start_time, 3)

    # Save run metadata with timings
    save_image_summary(
        run_id, run_dir_name, created_at, results_dir, prompts, models, model_timings
    )

    return run_id
