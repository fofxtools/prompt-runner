"""Unit tests for src/config.py"""

import pytest

from prompt_runner.config import (
    load_config,
    load_image_models,
    load_image_prompts,
    load_llm_models,
    load_llm_prompts,
)


class TestLoadConfig:
    """Tests for load_config function."""

    def test_load_valid_config(self, tmp_path):
        """Test loading a valid config file."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
results_dir: /storage/results/
llm_generation_defaults:
  temperature: 0.2
  num_predict: 512
""")
        config = load_config(str(config_file))
        assert config["results_dir"] == "/storage/results/"
        assert "llm_generation_defaults" in config

    def test_missing_file(self):
        """Test that missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent.yaml")

    def test_missing_results_dir(self, tmp_path):
        """Test that missing results_dir raises ValueError."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("llm_generation_defaults:\n  temperature: 0.2\n")
        with pytest.raises(ValueError):
            load_config(str(config_file))


class TestLoadPrompts:
    """Tests for load_llm_prompts function."""

    def test_load_valid_prompts(self, tmp_path):
        """Test loading valid prompts."""
        prompts_file = tmp_path / "prompts.yaml"
        prompts_file.write_text("""
- id: test_prompt
  prompt: "Test prompt text"
- id: another_prompt
  prompt: "Another test"
  options:
    temperature: 0.9
""")
        prompts = load_llm_prompts(str(prompts_file))
        assert len(prompts) == 2
        assert prompts[0]["id"] == "test_prompt"
        assert prompts[1]["options"]["temperature"] == 0.9

    def test_missing_file(self):
        """Test that missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_llm_prompts("nonexistent.yaml")

    def test_invalid_prompt_id(self, tmp_path):
        """Test that invalid prompt ID raises ValueError."""
        prompts_file = tmp_path / "prompts.yaml"
        prompts_file.write_text("- id: Invalid-ID\n  prompt: test\n")
        with pytest.raises(ValueError):
            load_llm_prompts(str(prompts_file))

    def test_duplicate_prompt_id(self, tmp_path):
        """Test that duplicate IDs raise ValueError."""
        prompts_file = tmp_path / "prompts.yaml"
        prompts_file.write_text("""
- id: test_id
  prompt: first
- id: test_id
  prompt: second
""")
        with pytest.raises(ValueError):
            load_llm_prompts(str(prompts_file))

    def test_missing_prompt_field(self, tmp_path):
        """Test that missing both prompt and messages fields raises ValueError."""
        prompts_file = tmp_path / "prompts.yaml"
        prompts_file.write_text("- id: test_id\n")
        with pytest.raises(ValueError):
            load_llm_prompts(str(prompts_file))

    def test_both_prompt_and_messages(self, tmp_path):
        """Test that having both prompt and messages raises ValueError."""
        prompts_file = tmp_path / "prompts.yaml"
        prompts_file.write_text("""
- id: test_id
  prompt: "Hello"
  messages:
    - role: user
      content: "Hi"
""")
        with pytest.raises(ValueError):
            load_llm_prompts(str(prompts_file))

    def test_valid_messages_format(self, tmp_path):
        """Test that valid messages format is accepted."""
        prompts_file = tmp_path / "prompts.yaml"
        prompts_file.write_text("""
- id: chat_prompt
  messages:
    - role: system
      content: "You are helpful"
    - role: user
      content: "Hello"
""")
        prompts = load_llm_prompts(str(prompts_file))
        assert len(prompts) == 1
        assert prompts[0]["id"] == "chat_prompt"
        assert len(prompts[0]["messages"]) == 2
        assert prompts[0]["messages"][0]["role"] == "system"

    def test_empty_messages_list(self, tmp_path):
        """Test that empty messages list raises ValueError."""
        prompts_file = tmp_path / "prompts.yaml"
        prompts_file.write_text("""
- id: test_id
  messages: []
""")
        with pytest.raises(ValueError):
            load_llm_prompts(str(prompts_file))

    def test_invalid_message_format(self, tmp_path):
        """Test that invalid message format raises ValueError."""
        prompts_file = tmp_path / "prompts.yaml"
        prompts_file.write_text("""
- id: test_id
  messages:
    - role: user
""")
        with pytest.raises(ValueError):
            load_llm_prompts(str(prompts_file))

    def test_invalid_message_role(self, tmp_path):
        """Test that invalid message role raises ValueError."""
        prompts_file = tmp_path / "prompts.yaml"
        prompts_file.write_text("""
- id: test_id
  messages:
    - role: invalid_role
      content: "Hello"
""")
        with pytest.raises(ValueError):
            load_llm_prompts(str(prompts_file))


class TestLoadModels:
    """Tests for load_llm_models function."""

    def test_load_valid_models(self, tmp_path):
        """Test loading valid models."""
        models_file = tmp_path / "models.yaml"
        models_file.write_text("""
- name: model1
  options:
    temperature: 0.7
- name: model2
""")
        models = load_llm_models(str(models_file))
        assert len(models) == 2
        assert models[0]["name"] == "model1"
        assert models[0]["options"]["temperature"] == 0.7
        assert models[1]["name"] == "model2"

    def test_missing_file(self):
        """Test that missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_llm_models("nonexistent.yaml")

    def test_missing_name_field(self, tmp_path):
        """Test that missing name field raises ValueError."""
        models_file = tmp_path / "models.yaml"
        models_file.write_text("- options:\n    temperature: 0.7\n")
        with pytest.raises(ValueError):
            load_llm_models(str(models_file))


class TestLoadImageModels:
    """Tests for load_image_models function."""

    def test_load_valid_models(self, tmp_path):
        """Test loading valid image models."""
        models_file = tmp_path / "image_models.yaml"
        models_file.write_text("""
- name: flux1-schnell
  server_args:
    --diffusion-model: /home/shared/ai/models/flux.gguf
    --clip_l: /home/shared/ai/encoders/clip_l.safetensors
    --sage-attn: true
  generation_options:
    sample_params: {sample_steps: 6}
- name: sd15
  server_args:
    -m: /home/shared/ai/models/sd15.safetensors
""")
        models = load_image_models(str(models_file))
        assert len(models) == 2
        assert models[0]["name"] == "flux1-schnell"
        assert (
            models[0]["server_args"]["--diffusion-model"]
            == "/home/shared/ai/models/flux.gguf"
        )
        assert models[0]["server_args"]["--sage-attn"] is True
        assert models[0]["generation_options"]["sample_params"] == {"sample_steps": 6}
        assert models[1]["name"] == "sd15"
        assert (
            models[1]["server_args"]["-m"] == "/home/shared/ai/models/sd15.safetensors"
        )

    def test_missing_file(self):
        """Test that missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_image_models("nonexistent.yaml")

    def test_missing_name_field(self, tmp_path):
        """Test that missing name field raises ValueError."""
        models_file = tmp_path / "image_models.yaml"
        models_file.write_text("""
- server_args:
    -m: /path/to/model
""")
        with pytest.raises(ValueError, match="missing required 'name' field"):
            load_image_models(str(models_file))

    def test_missing_server_args_field(self, tmp_path):
        """Test that missing server_args field raises ValueError."""
        models_file = tmp_path / "image_models.yaml"
        models_file.write_text("- name: test-model\n")
        with pytest.raises(ValueError, match="missing required 'server_args' field"):
            load_image_models(str(models_file))

    def test_server_args_must_be_a_mapping(self, tmp_path):
        """Test that a non-dict server_args raises TypeError."""
        models_file = tmp_path / "image_models.yaml"
        models_file.write_text("- name: test-model\n  server_args: [-m, x]\n")
        with pytest.raises(TypeError, match="'server_args' must be a dictionary"):
            load_image_models(str(models_file))


class TestLoadImagePrompts:
    """Tests for load_image_prompts function."""

    def test_load_valid_prompts(self, tmp_path):
        """Prompts are flat: id, prompt and any other request fields."""
        prompts_file = tmp_path / "image_prompts.yaml"
        prompts_file.write_text("""
- id: cute_cat
  prompt: A cute fluffy cat
- id: cat_img2img
  prompt: A refined cat image
  init_image: assets/input/cat.jpg
  strength: 0.7
  sample_params: {sample_steps: 8}
""")
        prompts = load_image_prompts(str(prompts_file))
        assert len(prompts) == 2
        assert prompts[0] == {"id": "cute_cat", "prompt": "A cute fluffy cat"}
        assert prompts[1]["strength"] == 0.7
        assert prompts[1]["init_image"] == "assets/input/cat.jpg"
        assert prompts[1]["sample_params"] == {"sample_steps": 8}

    def test_json_file(self, tmp_path):
        """JSON is valid YAML, so a prompts.json works."""
        prompts_file = tmp_path / "prompts.json"
        prompts_file.write_text('[{"id": "hotdog-1", "prompt": "A hot dog"}]')
        assert load_image_prompts(str(prompts_file)) == [
            {"id": "hotdog-1", "prompt": "A hot dog"}
        ]

    def test_missing_file(self):
        """Test that missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_image_prompts("nonexistent.yaml")

    def test_not_a_list(self, tmp_path):
        prompts_file = tmp_path / "image_prompts.yaml"
        prompts_file.write_text("id: x\nprompt: y\n")
        with pytest.raises(TypeError, match="must contain a list"):
            load_image_prompts(str(prompts_file))

    def test_prompt_must_be_a_dict(self, tmp_path):
        prompts_file = tmp_path / "image_prompts.yaml"
        prompts_file.write_text("- just a string\n")
        with pytest.raises(TypeError, match="must be a dictionary"):
            load_image_prompts(str(prompts_file))

    @pytest.mark.parametrize(
        "bad_id", ["has space", "../escape", "-leading", "a/b", ""]
    )
    def test_invalid_prompt_id(self, tmp_path, bad_id):
        """IDs end up in filenames, so path characters and spaces are rejected."""
        prompts_file = tmp_path / "image_prompts.yaml"
        prompts_file.write_text(f'- id: "{bad_id}"\n  prompt: test\n')
        with pytest.raises(ValueError, match="invalid"):
            load_image_prompts(str(prompts_file))

    def test_missing_id(self, tmp_path):
        prompts_file = tmp_path / "image_prompts.yaml"
        prompts_file.write_text("- prompt: test\n")
        with pytest.raises(ValueError, match="missing required 'id'"):
            load_image_prompts(str(prompts_file))

    def test_duplicate_prompt_id(self, tmp_path):
        """Test that duplicate IDs raise ValueError."""
        prompts_file = tmp_path / "image_prompts.yaml"
        prompts_file.write_text("""
- id: test_id
  prompt: first
- id: test_id
  prompt: second
""")
        with pytest.raises(ValueError, match="Duplicate"):
            load_image_prompts(str(prompts_file))

    def test_missing_prompt(self, tmp_path):
        prompts_file = tmp_path / "image_prompts.yaml"
        prompts_file.write_text("- id: test_id\n  negative_prompt: x\n")
        with pytest.raises(ValueError, match="missing required 'prompt'"):
            load_image_prompts(str(prompts_file))
