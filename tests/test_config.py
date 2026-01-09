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
generation_defaults:
  temperature: 0.2
  num_predict: 512
"""
        )
        config = load_config(str(config_file))
        assert config["results_dir"] == "/storage/results/"
        assert "generation_defaults" in config

    def test_missing_file(self):
        """Test that missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent.yaml")

    def test_missing_results_dir(self, tmp_path):
        """Test that missing results_dir raises ValueError."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("generation_defaults:\n  temperature: 0.2\n")
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
        """Test that missing both prompt and messages fields raises ValueError."""
        prompts_file = tmp_path / "prompts.yaml"
        prompts_file.write_text("- id: test_id\n")
        with pytest.raises(ValueError):
            load_prompts(str(prompts_file))

    def test_both_prompt_and_messages(self, tmp_path):
        """Test that having both prompt and messages raises ValueError."""
        prompts_file = tmp_path / "prompts.yaml"
        prompts_file.write_text(
            """
- id: test_id
  prompt: "Hello"
  messages:
    - role: user
      content: "Hi"
"""
        )
        with pytest.raises(ValueError):
            load_prompts(str(prompts_file))

    def test_valid_messages_format(self, tmp_path):
        """Test that valid messages format is accepted."""
        prompts_file = tmp_path / "prompts.yaml"
        prompts_file.write_text(
            """
- id: chat_prompt
  messages:
    - role: system
      content: "You are helpful"
    - role: user
      content: "Hello"
"""
        )
        prompts = load_prompts(str(prompts_file))
        assert len(prompts) == 1
        assert prompts[0]["id"] == "chat_prompt"
        assert len(prompts[0]["messages"]) == 2
        assert prompts[0]["messages"][0]["role"] == "system"

    def test_empty_messages_list(self, tmp_path):
        """Test that empty messages list raises ValueError."""
        prompts_file = tmp_path / "prompts.yaml"
        prompts_file.write_text(
            """
- id: test_id
  messages: []
"""
        )
        with pytest.raises(ValueError):
            load_prompts(str(prompts_file))

    def test_invalid_message_format(self, tmp_path):
        """Test that invalid message format raises ValueError."""
        prompts_file = tmp_path / "prompts.yaml"
        prompts_file.write_text(
            """
- id: test_id
  messages:
    - role: user
"""
        )
        with pytest.raises(ValueError):
            load_prompts(str(prompts_file))

    def test_invalid_message_role(self, tmp_path):
        """Test that invalid message role raises ValueError."""
        prompts_file = tmp_path / "prompts.yaml"
        prompts_file.write_text(
            """
- id: test_id
  messages:
    - role: invalid_role
      content: "Hello"
"""
        )
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
