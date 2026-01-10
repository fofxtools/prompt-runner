"""Image generation runner logic."""

import json
from pathlib import Path
from typing import Any, Dict, List

from stable_diffusion_cpp import StableDiffusion


def save_image_summary(
    run_id: str,
    run_dir_name: str,
    created_at: str,
    results_dir: str,
    prompts: List[Dict[str, Any]],
    models: List[Dict[str, Any]],
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
        },
    }

    # Write summary to file
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def initialize_stable_diffusion(model_config: Dict[str, Any]) -> StableDiffusion:
    """
    Initialize a StableDiffusion instance from model configuration.

    Extracts all parameters from model_config["options"] and passes them
    to StableDiffusion. The "name" field is metadata only and excluded.

    All options are passed through as-is. StableDiffusion will validate
    and fail fast if invalid.

    Args:
        model_config: Model configuration dictionary.
            - "name": model identifier (excluded, metadata only)
            - "options": dict containing all StableDiffusion parameters
              (paths, init options, generation options)

    Returns:
        Initialized StableDiffusion instance

    Raises:
        TypeError: If StableDiffusion receives invalid parameters (fail-fast)

    Examples:
        >>> config = {
        ...     "name": "flux1-schnell",
        ...     "options": {
        ...         "diffusion_model_path": "/path/to/model.gguf",
        ...         "clip_l_path": "/path/to/clip_l.safetensors",
        ...         "keep_clip_on_cpu": True,
        ...         "cfg_scale": 1.0
        ...     }
        ... }
        >>> sd = initialize_stable_diffusion(config)
    """
    # Extract options (all StableDiffusion parameters)
    if "options" not in model_config:
        raise ValueError(
            f"Model '{model_config.get('name', 'unknown')}' missing 'options' field"
        )

    # Pass all options to StableDiffusion
    # Let StableDiffusion validate parameters and fail fast if invalid
    return StableDiffusion(**model_config["options"])
