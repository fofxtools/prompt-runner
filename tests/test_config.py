"""Unit tests for src/config.py"""

from pathlib import Path
from typing import Any

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


class TestPromptImages:
    def write(self, tmp_path: Path, text: str) -> str:
        path = tmp_path / "prompts.yaml"
        path.write_text(text)
        return str(path)

    def test_a_prompt_and_a_message_can_have_images(self, tmp_path):
        prompts = load_llm_prompts(
            self.write(
                tmp_path,
                """
- id: one
  prompt: Describe this.
  images: [/a.png, /b.jpg]
- id: two
  messages:
    - {role: system, content: Be brief.}
    - {role: user, content: Compare, images: [/a.png]}
""",
            )
        )
        assert prompts[0]["images"] == ["/a.png", "/b.jpg"]
        assert prompts[1]["messages"][1]["images"] == ["/a.png"]

    @pytest.mark.parametrize("images", ["/a.png", "[1, 2]", "['']", "[[]]"])
    def test_images_must_be_a_list_of_paths(self, tmp_path, images):
        with pytest.raises(TypeError, match="images"):
            load_llm_prompts(
                self.write(tmp_path, f"- id: p\n  prompt: x\n  images: {images}\n")
            )

    def test_images_on_a_prompt_with_messages_are_refused(self, tmp_path):
        text = (
            "- id: p\n  images: [/a.png]\n  messages:\n    - {role: user, content: x}\n"
        )
        with pytest.raises(ValueError, match="in the message"):
            load_llm_prompts(self.write(tmp_path, text))

    def test_only_a_user_message_can_have_images(self, tmp_path):
        text = "- id: p\n  messages:\n    - {role: assistant, content: x, images: [/a.png]}\n"
        with pytest.raises(ValueError, match="only a user message"):
            load_llm_prompts(self.write(tmp_path, text))


class TestNoThinkingOptions:
    def load(self, tmp_path: Path, extra: str) -> list[dict[str, Any]]:
        path = tmp_path / "m.yaml"
        path.write_text(f"- name: m\n  server_args: {{-m: /m.gguf}}\n{extra}")
        return load_llm_models(str(path))

    def test_a_dictionary_is_accepted(self, tmp_path):
        (model,) = self.load(tmp_path, "  no_thinking_options: {temperature: 0.7}\n")
        assert model["no_thinking_options"] == {"temperature": 0.7}

    @pytest.mark.parametrize("value", ["0.7", "[0.7]", "null"])
    def test_anything_else_is_refused(self, tmp_path, value):
        with pytest.raises(TypeError, match="no_thinking_options"):
            self.load(tmp_path, f"  no_thinking_options: {value}\n")


class TestLoadModels:
    """Tests for load_llm_models function."""

    def test_load_valid_models(self, tmp_path):
        """Test loading valid models."""
        models_file = tmp_path / "models.yaml"
        models_file.write_text("""
- name: model1
  server_args:
    -m: /models/one.gguf
    --mmproj: /models/one-mmproj.gguf
  generation_options:
    temperature: 0.7
- name: model2
  server_args:
    -m: /models/two.gguf
""")
        models = load_llm_models(str(models_file))
        assert len(models) == 2
        assert models[0]["name"] == "model1"
        assert models[0]["server_args"]["--mmproj"] == "/models/one-mmproj.gguf"
        assert models[0]["generation_options"]["temperature"] == 0.7
        assert models[1]["name"] == "model2"

    def test_missing_file(self):
        """Test that missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_llm_models("nonexistent.yaml")

    def test_missing_name_field(self, tmp_path):
        """Test that missing name field raises ValueError."""
        models_file = tmp_path / "models.yaml"
        models_file.write_text("- server_args:\n    -m: /m.gguf\n")
        with pytest.raises(ValueError, match="name"):
            load_llm_models(str(models_file))

    def test_an_ollama_style_entry_without_server_args_is_refused(self, tmp_path):
        models_file = tmp_path / "models.yaml"
        models_file.write_text("- name: qwen3:8b\n")
        with pytest.raises(
            ValueError, match="'qwen3:8b' missing required 'server_args'"
        ):
            load_llm_models(str(models_file))

    def test_server_args_must_be_a_mapping(self, tmp_path):
        models_file = tmp_path / "models.yaml"
        models_file.write_text("- name: m\n  server_args: [-m, x]\n")
        with pytest.raises(TypeError, match="server_args"):
            load_llm_models(str(models_file))

    def test_options_is_called_generation_options_now(self, tmp_path):
        models_file = tmp_path / "models.yaml"
        models_file.write_text(
            "- name: m\n  server_args: {-m: x}\n  options: {temperature: 0.7}\n"
        )
        with pytest.raises(ValueError, match="generation_options"):
            load_llm_models(str(models_file))

    def test_generation_options_must_be_a_mapping(self, tmp_path):
        models_file = tmp_path / "models.yaml"
        models_file.write_text(
            "- name: m\n  server_args: {-m: x}\n  generation_options: [1]\n"
        )
        with pytest.raises(TypeError, match="generation_options"):
            load_llm_models(str(models_file))

    def test_the_shipped_models_file_loads_and_uses_llama_option_names(self):
        from prompt_runner.llm_runner import OLLAMA_ONLY

        models = load_llm_models("config/llm_models.yaml")
        assert models and all("-m" in m["server_args"] for m in models)
        for m in models:
            assert not set(m.get("generation_options", {})) & set(OLLAMA_ONLY), m[
                "name"
            ]
            assert "repetition_penalty" not in m.get(
                "generation_options", {}
            )  # llama.cpp: repeat_penalty


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
