"""Configuration loading and validation for the evals project."""

import re
from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str = "config/config.yaml") -> dict[str, Any]:
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
        TypeError: If a value has the wrong type (e.g. not a dict or list)
    """
    config_file = Path(config_path)

    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise TypeError("Config file must contain a YAML dictionary")

    # Validate required fields
    if "results_dir" not in config:
        raise ValueError("Config must contain 'results_dir' field")

    return config


def load_llm_prompts(
    prompts_path: str = "config/llm_prompts.yaml",
) -> list[dict[str, Any]]:
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
        TypeError: If a value has the wrong type (e.g. not a dict or list)
    """
    prompts_file = Path(prompts_path)

    if not prompts_file.exists():
        raise FileNotFoundError(f"Prompts file not found: {prompts_path}")

    with open(prompts_file, "r") as f:
        prompts = yaml.safe_load(f)

    if not isinstance(prompts, list):
        raise TypeError("Prompts file must contain a YAML list")

    # Validate prompt IDs
    prompt_id_pattern = re.compile(r"^[a-z0-9_]+$")
    seen_ids = set()

    for i, prompt in enumerate(prompts):
        if not isinstance(prompt, dict):
            raise TypeError(f"Prompt at index {i} must be a dictionary")

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
                raise TypeError(f"Prompt '{prompt_id}' field 'messages' must be a list")
            if not prompt["messages"]:
                raise ValueError(
                    f"Prompt '{prompt_id}' field 'messages' must not be empty"
                )
            for msg_idx, msg in enumerate(prompt["messages"]):
                if not isinstance(msg, dict):
                    raise TypeError(
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
            raise TypeError(
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
) -> list[dict[str, Any]]:
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
        TypeError: If a value has the wrong type (e.g. not a dict or list)
    """
    models_file = Path(models_path)

    if not models_file.exists():
        raise FileNotFoundError(f"Models file not found: {models_path}")

    with open(models_file, "r") as f:
        models = yaml.safe_load(f)

    if not isinstance(models, list):
        raise TypeError("Models file must contain a YAML list")

    # Validate each model
    for i, model in enumerate(models):
        if not isinstance(model, dict):
            raise TypeError(f"Model at index {i} must be a dictionary")

        if "name" not in model:
            raise ValueError(f"Model at index {i} missing required 'name' field")

        if "options" in model and not isinstance(model["options"], dict):
            raise TypeError(
                f"Model '{model.get('name', i)}' field 'options' must be a dictionary"
            )

    return models


def load_image_models(
    models_path: str = "config/image_models.yaml",
) -> list[dict[str, Any]]:
    """
    Load and validate image models from the models configuration file.

    The file must contain a YAML list where each model has:
    - name: the model identifier
    - server_args: dict of literal sd-server flags (--diffusion-model, --vae,
      --llm, --max-vram, ...); true gives a bare flag, false/null omits it
    - generation_options: (optional) dict of default native request fields
      (width, height, sample_params, lora, etc.)

    Args:
        models_path: Path to the image_models.yaml file (default: config/image_models.yaml)

    Returns:
        List of model dictionaries from the configuration file

    Raises:
        FileNotFoundError: If the models file doesn't exist
        yaml.YAMLError: If the YAML is invalid
        ValueError: If model validation fails
        TypeError: If a value has the wrong type (e.g. not a dict or list)
    """
    models_file = Path(models_path)

    if not models_file.exists():
        raise FileNotFoundError(f"Models file not found: {models_path}")

    with open(models_file, "r") as f:
        models = yaml.safe_load(f)

    if not isinstance(models, list):
        raise TypeError("Models file must contain a YAML list")

    # Validate each model
    for i, model in enumerate(models):
        if not isinstance(model, dict):
            raise TypeError(f"Model at index {i} must be a dictionary")

        if "name" not in model:
            raise ValueError(f"Model at index {i} missing required 'name' field")

        # Validate server_args field (required)
        if "server_args" not in model:
            raise ValueError(f"Model at index {i} missing required 'server_args' field")

        if not isinstance(model["server_args"], dict):
            raise TypeError(
                f"Model '{model.get('name', i)}' field 'server_args' must be a dictionary"
            )

        # Validate generation_options field (optional)
        if "generation_options" in model and not isinstance(
            model["generation_options"], dict
        ):
            raise TypeError(
                f"Model '{model.get('name', i)}' field 'generation_options' must be a dictionary"
            )

    return models


def load_image_prompts(
    prompts_path: str = "config/image_prompts.yaml",
) -> list[dict[str, Any]]:
    """
    Load and validate image prompts from a YAML or JSON file.

    The file must contain a list where each prompt has:
    - id: filename-safe string (letters, digits, "_", ".", "-"), unique
    - prompt: the prompt text
    - any other native sd-server request field (negative_prompt, seed,
      sample_params, init_image, strength, lora, etc.), passed through as-is

    Args:
        prompts_path: Path to the prompts file (default: config/image_prompts.yaml)

    Returns:
        List of prompt dictionaries

    Raises:
        FileNotFoundError: If the prompts file doesn't exist
        yaml.YAMLError: If the file is invalid
        ValueError: If prompt validation fails
        TypeError: If a value has the wrong type (e.g. not a dict or list)
    """
    prompts_file = Path(prompts_path)

    if not prompts_file.exists():
        raise FileNotFoundError(f"Prompts file not found: {prompts_path}")

    with open(prompts_file, "r") as f:
        prompts = yaml.safe_load(f)  # JSON is valid YAML

    if not isinstance(prompts, list):
        raise TypeError("Prompts file must contain a list")

    prompt_id_pattern = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    seen_ids = set()

    for i, prompt in enumerate(prompts):
        if not isinstance(prompt, dict):
            raise TypeError(f"Prompt at index {i} must be a dictionary")

        if "id" not in prompt:
            raise ValueError(f"Prompt at index {i} missing required 'id' field")

        prompt_id = prompt["id"]

        if not isinstance(prompt_id, str) or not prompt_id_pattern.match(prompt_id):
            raise ValueError(
                f"Prompt ID '{prompt_id}' is invalid. "
                f"Must match pattern: ^[A-Za-z0-9][A-Za-z0-9_.-]*$"
            )

        if prompt_id in seen_ids:
            raise ValueError(f"Duplicate prompt ID found: '{prompt_id}'")

        seen_ids.add(prompt_id)

        if "prompt" not in prompt:
            raise ValueError(f"Prompt '{prompt_id}' missing required 'prompt' field")

    return prompts
