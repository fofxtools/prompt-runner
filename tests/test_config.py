"""Unit tests for src/config.py"""

import pytest
from src.config import load_config, load_prompts, load_models


class TestLoadConfig:
    """Tests for load_config function."""

    def test_load_valid_config(self, tmp_path):
        """Test loading a valid config file."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
results_dir: /storage/results/
llm_runner:
  continue_on_length_cutoff: false
  max_continuations: 3
"""
        )
        config = load_config(str(config_file))
        assert config["results_dir"] == "/storage/results/"
        assert "llm_runner" in config

    def test_missing_file(self):
        """Test that missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent.yaml")

    def test_missing_results_dir(self, tmp_path):
        """Test that missing results_dir raises ValueError."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("llm_runner:\n  max_continuations: 3\n")
        with pytest.raises(ValueError):
            load_config(str(config_file))

    def test_missing_llm_runner(self, tmp_path):
        """Test that missing llm_runner raises ValueError."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("results_dir: /storage/\n")
        with pytest.raises(ValueError):
            load_config(str(config_file))


class TestLoadPrompts:
    """Tests for load_prompts function."""

    def test_load_valid_prompts(self, tmp_path):
        """Test loading valid prompts."""
        prompts_file = tmp_path / "prompts.yaml"
        prompts_file.write_text(
            """
- id: test_prompt
  prompt: "Test prompt text"
- id: another_prompt
  prompt: "Another test"
  options:
    temperature: 0.9
"""
        )
        prompts = load_prompts(str(prompts_file))
        assert len(prompts) == 2
        assert prompts[0]["id"] == "test_prompt"
        assert prompts[1]["options"]["temperature"] == 0.9

    def test_missing_file(self):
        """Test that missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_prompts("nonexistent.yaml")

    def test_invalid_prompt_id(self, tmp_path):
        """Test that invalid prompt ID raises ValueError."""
        prompts_file = tmp_path / "prompts.yaml"
        prompts_file.write_text("- id: Invalid-ID\n  prompt: test\n")
        with pytest.raises(ValueError):
            load_prompts(str(prompts_file))

    def test_duplicate_prompt_id(self, tmp_path):
        """Test that duplicate IDs raise ValueError."""
        prompts_file = tmp_path / "prompts.yaml"
        prompts_file.write_text(
            """
- id: test_id
  prompt: first
- id: test_id
  prompt: second
"""
        )
        with pytest.raises(ValueError):
            load_prompts(str(prompts_file))

    def test_missing_prompt_field(self, tmp_path):
        """Test that missing prompt field raises ValueError."""
        prompts_file = tmp_path / "prompts.yaml"
        prompts_file.write_text("- id: test_id\n")
        with pytest.raises(ValueError):
            load_prompts(str(prompts_file))


class TestLoadModels:
    """Tests for load_models function."""

    def test_load_valid_models(self, tmp_path):
        """Test loading valid models."""
        models_file = tmp_path / "models.yaml"
        models_file.write_text(
            """
- name: model1
  options:
    temperature: 0.7
- name: model2
"""
        )
        models = load_models(str(models_file))
        assert len(models) == 2
        assert models[0]["name"] == "model1"
        assert models[0]["options"]["temperature"] == 0.7
        assert models[1]["name"] == "model2"

    def test_missing_file(self):
        """Test that missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_models("nonexistent.yaml")

    def test_missing_name_field(self, tmp_path):
        """Test that missing name field raises ValueError."""
        models_file = tmp_path / "models.yaml"
        models_file.write_text("- options:\n    temperature: 0.7\n")
        with pytest.raises(ValueError):
            load_models(str(models_file))
