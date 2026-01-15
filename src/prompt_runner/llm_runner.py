"""LLM runner logic for executing prompts across multiple models."""

import json
import time
from pathlib import Path
from typing import Any, Dict, List

import ollama

from .utils import (
    create_result_structure,
    generate_run_identifiers,
    merge_options,
    sanitize_fs_name,
)


def save_llm_summary(
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
        "llm": {
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


def save_llm_result(
    run_dir_name: str,
    results_dir: str,
    prompt_id: str,
    model_name: str,
    mode: str,
    result_data: Dict[str, Any],
) -> None:
    """
    Save an LLM result to JSON and markdown files.

    Args:
        run_dir_name: The filesystem-safe run directory name
        results_dir: The base results directory path
        prompt_id: The prompt identifier
        model_name: The model name (will be sanitized for filename)
        mode: The generation mode ("completion" or "chat")
        result_data: The complete result dictionary to save

    Raises:
        FileNotFoundError: If the run directory does not exist
    """
    results_path = Path(results_dir)
    run_path = results_path / run_dir_name
    prompt_path = run_path / "llm" / prompt_id

    if not run_path.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_path}")

    # Create prompt directory if it doesn't exist
    prompt_path.mkdir(parents=True, exist_ok=True)

    # Create filename with mode suffix
    sanitized_model = sanitize_fs_name(model_name)
    result_file = prompt_path / f"{sanitized_model}__{mode}.json"

    # Write JSON result to file
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(result_data, f, indent=2)

    # Create markdown directory and save markdown file
    markdown_path = prompt_path / "markdown"
    markdown_path.mkdir(exist_ok=True)
    markdown_file = markdown_path / f"{sanitized_model}__{mode}.md"

    # Write markdown file with output text
    with open(markdown_file, "w", encoding="utf-8") as f:
        f.write(result_data["output"]["text"])


def generate_response_completion(
    client: ollama.Client, model: str, prompt: str, options: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Generate a response using the completion API (ollama.Client.generate).

    Args:
        client: Ollama client instance
        model: Model name to use
        prompt: Prompt text to send
        options: Generation options to pass to ollama

    Returns:
        Dictionary with 'output' and 'metrics' keys:
        - output.text: Generated response text
        - metrics.done_reason: Reason generation stopped
        - metrics.input_tokens: Number of input tokens processed
        - metrics.output_tokens: Number of output tokens generated
        - metrics.total_tokens: Total tokens (input + output)
        - metrics.load_seconds: Time spent loading model (seconds)
        - metrics.input_seconds: Time spent processing input (seconds)
        - metrics.output_seconds: Time spent generating output (seconds)
        - metrics.total_seconds: Total generation time (seconds)
        - metrics.output_tokens_per_second: Output generation speed (tokens/second)
    """
    response = client.generate(model=model, prompt=prompt, options=options)

    # Extract token counts
    input_tokens = response.get("prompt_eval_count")
    output_tokens = response.get("eval_count")
    total_tokens = None
    if input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens

    # Extract timing metrics (convert from nanoseconds to seconds, round to 3 decimals)
    # Ollama field mapping:
    #   load_duration -> load_seconds
    #   prompt_eval_duration -> input_seconds
    #   eval_duration -> output_seconds
    #   total_duration -> total_seconds
    load_seconds = None
    if response.get("load_duration") is not None:
        load_seconds = round(response["load_duration"] / 1e9, 3)

    input_seconds = None
    if response.get("prompt_eval_duration") is not None:
        input_seconds = round(response["prompt_eval_duration"] / 1e9, 3)

    output_seconds = None
    if response.get("eval_duration") is not None:
        output_seconds = round(response["eval_duration"] / 1e9, 3)

    total_seconds = None
    if response.get("total_duration") is not None:
        total_seconds = round(response["total_duration"] / 1e9, 3)

    # Calculate throughput
    output_tokens_per_second = None
    if (
        output_tokens is not None
        and output_seconds is not None
        and isinstance(output_seconds, (int, float))
        and output_seconds > 0
    ):
        output_tokens_per_second = round(output_tokens / output_seconds, 3)

    return {
        "output": {
            "text": response["response"],
        },
        "metrics": {
            "done_reason": response.get("done_reason"),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "load_seconds": load_seconds,
            "input_seconds": input_seconds,
            "output_seconds": output_seconds,
            "total_seconds": total_seconds,
            "output_tokens_per_second": output_tokens_per_second,
        },
    }


def generate_response_chat(
    client: ollama.Client,
    model: str,
    messages: List[Dict[str, str]],
    options: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Generate a response using the chat API (ollama.chat).

    Args:
        client: Ollama client instance
        model: Model name to use
        messages: List of message dicts with 'role' and 'content' keys
        options: Generation options to pass to ollama

    Returns:
        Dictionary with 'output' and 'metrics' keys:
        - output.text: Generated response text
        - metrics.done_reason: Reason generation stopped
        - metrics.input_tokens: Number of input tokens processed
        - metrics.output_tokens: Number of output tokens generated
        - metrics.total_tokens: Total tokens (input + output)
        - metrics.load_seconds: Time spent loading model (seconds)
        - metrics.input_seconds: Time spent processing input (seconds)
        - metrics.output_seconds: Time spent generating output (seconds)
        - metrics.total_seconds: Total generation time (seconds)
        - metrics.output_tokens_per_second: Output generation speed (tokens/second)
    """
    response = client.chat(model=model, messages=messages, options=options)

    # Extract token counts
    input_tokens = response.get("prompt_eval_count")
    output_tokens = response.get("eval_count")
    total_tokens = None
    if input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens

    # Extract timing metrics (convert from nanoseconds to seconds, round to 3 decimals)
    # Ollama field mapping:
    #   load_duration -> load_seconds
    #   prompt_eval_duration -> input_seconds
    #   eval_duration -> output_seconds
    #   total_duration -> total_seconds
    load_seconds = None
    if response.get("load_duration") is not None:
        load_seconds = round(response["load_duration"] / 1e9, 3)

    input_seconds = None
    if response.get("prompt_eval_duration") is not None:
        input_seconds = round(response["prompt_eval_duration"] / 1e9, 3)

    output_seconds = None
    if response.get("eval_duration") is not None:
        output_seconds = round(response["eval_duration"] / 1e9, 3)

    total_seconds = None
    if response.get("total_duration") is not None:
        total_seconds = round(response["total_duration"] / 1e9, 3)

    # Calculate throughput
    output_tokens_per_second = None
    if (
        output_tokens is not None
        and output_seconds is not None
        and isinstance(output_seconds, (int, float))
        and output_seconds > 0
    ):
        output_tokens_per_second = round(output_tokens / output_seconds, 3)

    return {
        "output": {
            "text": response["message"]["content"],
        },
        "metrics": {
            "done_reason": response.get("done_reason"),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "load_seconds": load_seconds,
            "input_seconds": input_seconds,
            "output_seconds": output_seconds,
            "total_seconds": total_seconds,
            "output_tokens_per_second": output_tokens_per_second,
        },
    }


def run_llm_eval(
    config: Dict[str, Any],
    prompts: List[Dict[str, Any]],
    models: List[Dict[str, Any]],
    prompt_filter: str = "all",
) -> str:
    """
    Run LLM evaluation across selected prompts and models.

    Args:
        config: Configuration dictionary with 'results_dir' key
        prompts: List of prompt dictionaries with 'id' and either 'prompt' or 'messages'
        models: List of model dictionaries with 'name' key
        prompt_filter: Which prompts to include - "completion", "chat", or "all" (default: "all")

    Returns:
        run_id: The ISO-8601 formatted run identifier

    Raises:
        ValueError: If prompt_filter is not "completion", "chat", or "all"
    """
    if prompt_filter not in ("completion", "chat", "all"):
        raise ValueError(
            f"Invalid prompt_filter: {prompt_filter}. Must be 'completion', 'chat', or 'all'"
        )

    # Generate run identifiers
    run_id, run_dir_name, created_at = generate_run_identifiers()
    results_dir = config["results_dir"]

    # Create result directory structure
    create_result_structure(run_dir_name, results_dir)

    # Initialize ollama client
    client = ollama.Client()

    # Get global defaults
    global_defaults = config.get("llm_generation_defaults")

    # Track timing per model
    model_timings = {}

    # Process each model
    for model_dict in models:
        model_name = model_dict["name"]
        model_options = model_dict.get("options")

        print(f"\nProcessing model: {model_name}")

        # Start timing for this model
        start_time = time.time()

        # Process each prompt
        for prompt_dict in prompts:
            prompt_id = prompt_dict["id"]
            prompt_options = prompt_dict.get("options")

            # Determine prompt type based on structure
            is_completion_prompt = "prompt" in prompt_dict
            is_chat_prompt = "messages" in prompt_dict

            # Filter prompts based on prompt_filter
            if prompt_filter == "completion" and not is_completion_prompt:
                continue
            if prompt_filter == "chat" and not is_chat_prompt:
                continue

            # Validate prompt structure
            if not is_completion_prompt and not is_chat_prompt:
                raise ValueError(
                    f"Prompt '{prompt_id}' must have either 'prompt' or 'messages' field"
                )

            # Merge options (global < model < prompt priority)
            options = merge_options(global_defaults, model_options, prompt_options)

            # Run prompt in its native form
            if is_completion_prompt:
                current_mode = "completion"
                print(f"  Prompt: {prompt_id} (mode: {current_mode})")
                prompt_text = prompt_dict["prompt"]
                response = generate_response_completion(
                    client, model_name, prompt_text, options
                )
            else:  # is_chat_prompt
                current_mode = "chat"
                print(f"  Prompt: {prompt_id} (mode: {current_mode})")
                messages = prompt_dict["messages"]
                response = generate_response_chat(client, model_name, messages, options)

            # Create result dictionary
            result_data = {
                "run_id": run_id,
                "created_at": created_at,
                "prompt_id": prompt_id,
                "model": model_name,
                "mode": current_mode,
                "output": response["output"],
                "metrics": response["metrics"],
            }

            # Save result
            save_llm_result(
                run_dir_name,
                results_dir,
                prompt_id,
                model_name,
                current_mode,
                result_data,
            )

            print("    ✓ Saved result")

        # Record elapsed time for this model
        model_timings[model_name] = round(time.time() - start_time, 3)

    # Save run metadata with timings
    save_llm_summary(
        run_id, run_dir_name, created_at, results_dir, prompts, models, model_timings
    )

    print(f"\n✓ Evaluation complete. Run ID: {run_id}")
    return run_id
