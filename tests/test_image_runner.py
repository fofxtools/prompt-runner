"""Unit tests for src/prompt_runner/image_runner.py"""

import json
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image
from prompt_runner.image_runner import (
    generate_image,
    initialize_stable_diffusion,
    save_image_summary,
)


class TestSaveImageSummary:
    """Tests for save_image_summary function."""

    def test_creates_summary_file(self, tmp_path):
        """Test that summary.json is created with correct structure."""
        run_id = "2026-01-10T12:34:56Z-abc123"
        run_dir_name = "2026-01-10_12-34-56Z-abc123"
        created_at = "2026-01-10T12:34:56Z"
        run_path = tmp_path / run_dir_name
        run_path.mkdir()

        prompts = [
            {
                "id": "cute_cat_txt2img",
                "mode": "txt2img",
                "options": {"prompt": "A cute cat"},
            },
            {
                "id": "cat_img2img",
                "mode": "img2img",
                "options": {"prompt": "Refined cat"},
            },
        ]
        models = [
            {"name": "flux1-schnell"},
            {"name": "sd15"},
        ]

        save_image_summary(
            run_id, run_dir_name, created_at, str(tmp_path), prompts, models
        )

        summary_file = run_path / "summary.json"
        assert summary_file.exists()

        with open(summary_file, "r", encoding="utf-8") as f:
            summary = json.load(f)

        assert summary["run_id"] == run_id
        assert summary["created_at"] == created_at
        assert summary["image"]["prompt_count"] == 2
        assert summary["image"]["model_count"] == 2
        assert summary["image"]["prompts"] == ["cute_cat_txt2img", "cat_img2img"]
        assert summary["image"]["models"] == ["flux1-schnell", "sd15"]

    def test_raises_if_directory_missing(self, tmp_path):
        """Test that FileNotFoundError is raised if directory doesn't exist."""
        run_id = "2026-01-10T12:34:56Z-abc123"
        run_dir_name = "nonexistent"
        created_at = "2026-01-10T12:34:56Z"
        prompts = [{"id": "test", "mode": "txt2img", "options": {"prompt": "Test"}}]
        models = [{"name": "model"}]

        with pytest.raises(FileNotFoundError):
            save_image_summary(
                run_id, run_dir_name, created_at, str(tmp_path), prompts, models
            )


class TestInitializeStableDiffusion:
    """Tests for initialize_stable_diffusion function."""

    @patch("prompt_runner.image_runner.StableDiffusion")
    def test_passes_all_options_to_stable_diffusion(self, mock_sd_class):
        """Test that all options are passed to StableDiffusion."""
        mock_instance = MagicMock()
        mock_sd_class.return_value = mock_instance

        model_config = {
            "name": "test-model",
            "options": {
                "model_path": "/path/to/model.safetensors",
                "diffusion_model_path": "/path/to/diffusion.gguf",
                "clip_l_path": "/path/to/clip_l.safetensors",
                "clip_g_path": "/path/to/clip_g.safetensors",
                "t5xxl_path": "/path/to/t5xxl.safetensors",
                "llm_path": "/path/to/llm.gguf",
                "vae_path": "/path/to/vae.safetensors",
                "keep_clip_on_cpu": True,
                "cfg_scale": 1.0,
            },
        }

        result = initialize_stable_diffusion(model_config)

        # Should pass all options
        mock_sd_class.assert_called_once_with(
            model_path="/path/to/model.safetensors",
            diffusion_model_path="/path/to/diffusion.gguf",
            clip_l_path="/path/to/clip_l.safetensors",
            clip_g_path="/path/to/clip_g.safetensors",
            t5xxl_path="/path/to/t5xxl.safetensors",
            llm_path="/path/to/llm.gguf",
            vae_path="/path/to/vae.safetensors",
            keep_clip_on_cpu=True,
            cfg_scale=1.0,
        )
        assert result == mock_instance

    @patch("prompt_runner.image_runner.StableDiffusion")
    def test_passes_all_options_verbatim(self, mock_sd_class):
        """Test that all options are passed through as-is."""
        mock_instance = MagicMock()
        mock_sd_class.return_value = mock_instance

        model_config = {
            "name": "flux1-schnell",
            "options": {
                "diffusion_model_path": "/path/to/model.gguf",
                "vae_decode_only": True,
                "keep_clip_on_cpu": True,
                "cfg_scale": 1.0,
                "sample_steps": 6,
            },
        }

        result = initialize_stable_diffusion(model_config)

        # Should pass all options (StableDiffusion will validate)
        mock_sd_class.assert_called_once_with(
            diffusion_model_path="/path/to/model.gguf",
            vae_decode_only=True,
            keep_clip_on_cpu=True,
            cfg_scale=1.0,
            sample_steps=6,
        )
        assert result == mock_instance

    def test_raises_if_options_missing(self):
        """Test that ValueError is raised if options field is missing."""
        model_config = {
            "name": "sd15",
        }

        with pytest.raises(ValueError, match="missing 'options' field"):
            initialize_stable_diffusion(model_config)


class TestGenerateImage:
    """Tests for generate_image function."""

    def test_txt2img_basic(self):
        """Test basic txt2img generation."""
        mock_sd = MagicMock()
        mock_image = Image.new("RGB", (512, 512), color="red")
        mock_sd.generate_image.return_value = [mock_image]

        model_config = {"name": "test-model", "options": {}}
        prompt_config = {
            "id": "test",
            "mode": "txt2img",
            "options": {"prompt": "a cute cat"},
        }
        options = {"width": 512, "height": 512}

        result = generate_image(mock_sd, model_config, prompt_config, options)

        assert len(result) == 1
        assert result[0] == mock_image
        mock_sd.generate_image.assert_called_once_with(
            width=512, height=512, prompt="a cute cat"
        )

    def test_txt2img_with_negative_prompt(self):
        """Test txt2img with negative prompt (pass-through)."""
        mock_sd = MagicMock()
        mock_image = Image.new("RGB", (512, 512), color="blue")
        mock_sd.generate_image.return_value = [mock_image]

        model_config = {"name": "test-model", "options": {}}
        prompt_config = {
            "id": "test",
            "mode": "txt2img",
            "options": {
                "prompt": "a cute cat",
                "negative_prompt": "ugly",
            },
        }
        options = {}

        result = generate_image(mock_sd, model_config, prompt_config, options)

        assert len(result) == 1
        mock_sd.generate_image.assert_called_once_with(
            prompt="a cute cat", negative_prompt="ugly"
        )

    def test_txt2img_multiple_images(self):
        """Test generating multiple images using batch_count."""
        mock_sd = MagicMock()
        mock_images = [
            Image.new("RGB", (512, 512), color="green"),
            Image.new("RGB", (512, 512), color="blue"),
            Image.new("RGB", (512, 512), color="red"),
        ]
        mock_sd.generate_image.return_value = mock_images

        model_config = {"name": "test-model", "options": {}}
        prompt_config = {
            "id": "test",
            "mode": "txt2img",
            "options": {"prompt": "test", "batch_count": 3},
        }
        options = {}

        result = generate_image(mock_sd, model_config, prompt_config, options)

        assert len(result) == 3
        mock_sd.generate_image.assert_called_once_with(prompt="test", batch_count=3)

    def test_img2img_with_strength(self):
        """Test img2img with strength parameter (pass-through)."""
        mock_sd = MagicMock()
        mock_image = Image.new("RGB", (512, 512), color="yellow")
        mock_sd.generate_image.return_value = [mock_image]

        model_config = {"name": "test-model", "options": {}}
        prompt_config = {
            "id": "test",
            "mode": "img2img",
            "options": {
                "prompt": "blue eyes",
                "init_image": "/path/to/image.png",
                "strength": 0.7,
            },
        }
        options = {}

        result = generate_image(mock_sd, model_config, prompt_config, options)

        assert len(result) == 1
        call_kwargs = mock_sd.generate_image.call_args[1]
        assert call_kwargs["prompt"] == "blue eyes"
        assert call_kwargs["init_image"] == "/path/to/image.png"
        assert call_kwargs["strength"] == 0.7

    def test_pass_through_all_prompt_fields(self):
        """Test that all fields in options are passed through."""
        mock_sd = MagicMock()
        mock_image = Image.new("RGB", (512, 512), color="red")
        mock_sd.generate_image.return_value = [mock_image]

        model_config = {"name": "test-model", "options": {}}
        prompt_config = {
            "id": "test",
            "mode": "txt2img",
            "options": {
                "prompt": "test",
                "negative_prompt": "bad",
                "custom_field": "custom_value",  # Should be passed through
                "another_field": 123,  # Should be passed through
            },
        }
        options = {"width": 512}

        generate_image(mock_sd, model_config, prompt_config, options)

        call_kwargs = mock_sd.generate_image.call_args[1]
        assert call_kwargs["prompt"] == "test"
        assert call_kwargs["negative_prompt"] == "bad"
        assert call_kwargs["custom_field"] == "custom_value"
        assert call_kwargs["another_field"] == 123
        assert call_kwargs["width"] == 512
        # Metadata fields should NOT be passed through
        assert "id" not in call_kwargs
        assert "mode" not in call_kwargs
        assert "options" not in call_kwargs
