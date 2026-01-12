"""Configuration loading and validation for the evals project."""

import re
from pathlib import Path
from typing import Any, Dict, List

import yaml


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """
    Load and validate the main configuration file.

    Args:
        config_path: Path to the config.yaml file (default: config/config.yaml)

    Returns:
        Dictionary containing the configuration

    Raises:
        FileNotFoundError: If the config file doesn't exist
        yaml.YAMLError: If the YAML is invalid
        ValueError: If the configuration structure is invalid
    """
    config_file = Path(config_path)

    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError("Config file must contain a YAML dictionary")

    # Validate required fields
    if "results_dir" not in config:
        raise ValueError("Config must contain 'results_dir' field")

    return config


def load_llm_prompts(
    prompts_path: str = "config/llm_prompts.yaml",
) -> List[Dict[str, Any]]:
    """
    Load and validate prompts from the prompts configuration file.

    Each prompt must have:
    - id: string matching regex ^[a-z0-9_]+$
    - prompt: the prompt text (for completion mode or simple chat)
      OR
    - messages: list of message dicts (for multi-turn chat)
    - options: (optional) dict of generation options

    Args:
        prompts_path: Path to the prompts.yaml file (default: config/llm_prompts.yaml)

    Returns:
        List of prompt dictionaries

    Raises:
        FileNotFoundError: If the prompts file doesn't exist
        yaml.YAMLError: If the YAML is invalid
        ValueError: If prompt validation fails
    """
    prompts_file = Path(prompts_path)

    if not prompts_file.exists():
        raise FileNotFoundError(f"Prompts file not found: {prompts_path}")

    with open(prompts_file, "r") as f:
        prompts = yaml.safe_load(f)

    if not isinstance(prompts, list):
        raise ValueError("Prompts file must contain a YAML list")

    # Validate prompt IDs
    prompt_id_pattern = re.compile(r"^[a-z0-9_]+$")
    seen_ids = set()

    for i, prompt in enumerate(prompts):
        if not isinstance(prompt, dict):
            raise ValueError(f"Prompt at index {i} must be a dictionary")

        if "id" not in prompt:
            raise ValueError(f"Prompt at index {i} missing required 'id' field")

        prompt_id = prompt["id"]

        # Must have either 'prompt' or 'messages', but not both
        has_prompt = "prompt" in prompt
        has_messages = "messages" in prompt

        if not has_prompt and not has_messages:
            raise ValueError(
                f"Prompt '{prompt_id}' must have either 'prompt' or 'messages' field"
            )

        if has_prompt and has_messages:
            raise ValueError(
                f"Prompt '{prompt_id}' cannot have both 'prompt' and 'messages' fields"
            )

        # Validate messages format if present
        if has_messages:
            if not isinstance(prompt["messages"], list):
                raise ValueError(
                    f"Prompt '{prompt_id}' field 'messages' must be a list"
                )
            if not prompt["messages"]:
                raise ValueError(
                    f"Prompt '{prompt_id}' field 'messages' must not be empty"
                )
            for msg_idx, msg in enumerate(prompt["messages"]):
                if not isinstance(msg, dict):
                    raise ValueError(
                        f"Prompt '{prompt_id}' message at index {msg_idx} must be a dict"
                    )
                if "role" not in msg or "content" not in msg:
                    raise ValueError(
                        f"Prompt '{prompt_id}' message at index {msg_idx} must have 'role' and 'content'"
                    )
                valid_roles = {"system", "user", "assistant"}
                if msg["role"] not in valid_roles:
                    raise ValueError(
                        f"Prompt '{prompt_id}' message at index {msg_idx} has invalid 'role'. "
                        f"Must be one of: {valid_roles}"
                    )

        if "options" in prompt and not isinstance(prompt["options"], dict):
            raise ValueError(
                f"Prompt '{prompt_id}' field 'options' must be a dictionary"
            )

        # Validate ID format
        if not prompt_id_pattern.match(prompt_id):
            raise ValueError(
                f"Prompt ID '{prompt_id}' is invalid. "
                f"Must match pattern: ^[a-z0-9_]+$"
            )

        # Check for duplicates
        if prompt_id in seen_ids:
            raise ValueError(f"Duplicate prompt ID found: '{prompt_id}'")

        seen_ids.add(prompt_id)

    return prompts


def load_llm_models(
    models_path: str = "config/llm_models.yaml",
) -> List[Dict[str, Any]]:
    """
    Load and validate models from the models configuration file.

    The file must contain a YAML list where each model has:
    - name: the model identifier
    - options: (optional) dict of default generation options

    Args:
        models_path: Path to the models.yaml file (default: config/llm_models.yaml)

    Returns:
        List of model dictionaries

    Raises:
        FileNotFoundError: If the models file doesn't exist
        yaml.YAMLError: If the YAML is invalid
        ValueError: If model validation fails
    """
    models_file = Path(models_path)

    if not models_file.exists():
        raise FileNotFoundError(f"Models file not found: {models_path}")

    with open(models_file, "r") as f:
        models = yaml.safe_load(f)

    if not isinstance(models, list):
        raise ValueError("Models file must contain a YAML list")

    # Validate each model
    for i, model in enumerate(models):
        if not isinstance(model, dict):
            raise ValueError(f"Model at index {i} must be a dictionary")

        if "name" not in model:
            raise ValueError(f"Model at index {i} missing required 'name' field")

        if "options" in model and not isinstance(model["options"], dict):
            raise ValueError(
                f"Model '{model.get('name', i)}' field 'options' must be a dictionary"
            )

    return models


def load_image_models(
    models_path: str = "config/image_models.yaml",
) -> List[Dict[str, Any]]:
    """
    Load and validate image models from the models configuration file.

    The file must contain a YAML list where each model has:
    - name: the model identifier
    - init_options: dict containing all StableDiffusion initialization parameters
      (diffusion_model_path, model_path, clip_l_path, clip_g_path, t5xxl_path,
      llm_path, vae_path, keep_clip_on_cpu, vae_decode_only, etc.)
    - generation_options: (optional) dict containing default generation parameters
      (cfg_scale, sample_steps, width, height, etc.)

    Args:
        models_path: Path to the image_models.yaml file (default: config/image_models.yaml)

    Returns:
        List of model dictionaries from the configuration file

    Raises:
        FileNotFoundError: If the models file doesn't exist
        yaml.YAMLError: If the YAML is invalid
        ValueError: If model validation fails
    """
    models_file = Path(models_path)

    if not models_file.exists():
        raise FileNotFoundError(f"Models file not found: {models_path}")

    with open(models_file, "r") as f:
        models = yaml.safe_load(f)

    if not isinstance(models, list):
        raise ValueError("Models file must contain a YAML list")

    # Validate each model
    for i, model in enumerate(models):
        if not isinstance(model, dict):
            raise ValueError(f"Model at index {i} must be a dictionary")

        if "name" not in model:
            raise ValueError(f"Model at index {i} missing required 'name' field")

        # Validate init_options field (required)
        if "init_options" not in model:
            raise ValueError(
                f"Model at index {i} missing required 'init_options' field"
            )

        if not isinstance(model["init_options"], dict):
            raise ValueError(
                f"Model '{model.get('name', i)}' field 'init_options' must be a dictionary"
            )

        # Validate generation_options field (optional)
        if "generation_options" in model and not isinstance(
            model["generation_options"], dict
        ):
            raise ValueError(
                f"Model '{model.get('name', i)}' field 'generation_options' must be a dictionary"
            )

    return models


def load_image_prompts(
    prompts_path: str = "config/image_prompts.yaml",
) -> List[Dict[str, Any]]:
    """
    Load and validate image prompts from the prompts configuration file.

    Each prompt must have:
    - id: string matching regex ^[a-z0-9_]+$
    - mode: "txt2img" or "img2img"
    - options: dict containing all StableDiffusion.generate_image() parameters
      (prompt, negative_prompt, cfg_scale, sample_steps, seed, batch_count, etc.)

    Args:
        prompts_path: Path to the image_prompts.yaml file (default: config/image_prompts.yaml)

    Returns:
        List of prompt dictionaries

    Raises:
        FileNotFoundError: If the prompts file doesn't exist
        yaml.YAMLError: If the YAML is invalid
        ValueError: If prompt validation fails
    """
    prompts_file = Path(prompts_path)

    if not prompts_file.exists():
        raise FileNotFoundError(f"Prompts file not found: {prompts_path}")

    with open(prompts_file, "r") as f:
        prompts = yaml.safe_load(f)

    if not isinstance(prompts, list):
        raise ValueError("Prompts file must contain a YAML list")

    # Validate prompt IDs and mode
    prompt_id_pattern = re.compile(r"^[a-z0-9_]+$")
    seen_ids = set()

    for i, prompt in enumerate(prompts):
        if not isinstance(prompt, dict):
            raise ValueError(f"Prompt at index {i} must be a dictionary")

        if "id" not in prompt:
            raise ValueError(f"Prompt at index {i} missing required 'id' field")

        prompt_id = prompt["id"]

        # Validate ID format
        if not prompt_id_pattern.match(prompt_id):
            raise ValueError(
                f"Prompt ID '{prompt_id}' is invalid. "
                f"Must match pattern: ^[a-z0-9_]+$"
            )

        # Check for duplicates
        if prompt_id in seen_ids:
            raise ValueError(f"Duplicate prompt ID found: '{prompt_id}'")

        seen_ids.add(prompt_id)

        # Validate mode
        if "mode" not in prompt:
            raise ValueError(f"Prompt '{prompt_id}' missing required 'mode' field")

        mode = prompt["mode"]
        if mode not in ("txt2img", "img2img"):
            raise ValueError(
                f"Prompt '{prompt_id}' has invalid mode '{mode}'. "
                f"Must be 'txt2img' or 'img2img'"
            )

        # Validate options field (required)
        if "options" not in prompt:
            raise ValueError(f"Prompt '{prompt_id}' missing required 'options' field")

        if not isinstance(prompt["options"], dict):
            raise ValueError(
                f"Prompt '{prompt_id}' field 'options' must be a dictionary"
            )

    return prompts
