"""Unit tests for src/prompt_runner/image_runner.py"""

import json
from unittest.mock import MagicMock, patch

import pytest
from prompt_runner.image_runner import initialize_stable_diffusion, save_image_summary


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
            {"id": "cute_cat_txt2img", "mode": "txt2img", "prompt": "A cute cat"},
            {"id": "cat_img2img", "mode": "img2img", "prompt": "Refined cat"},
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
        prompts = [{"id": "test", "mode": "txt2img", "prompt": "Test"}]
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
