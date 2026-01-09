"""Unit tests for src/llm_runner.py"""

import json
import pytest
from unittest.mock import Mock
from src.llm_runner import (
    generate_run_identifiers,
    create_result_structure,
    save_llm_summary,
    save_llm_result,
    merge_options,
    generate_response_completion,
    generate_response_chat,
)


class TestGenerateRunIdentifiers:
    """Tests for generate_run_identifiers function."""

    def test_returns_tuple(self):
        """Test that function returns a tuple of three strings."""
        result = generate_run_identifiers()
        assert isinstance(result, tuple)
        assert len(result) == 3
        run_id, run_dir_name, created_at = result
        assert isinstance(run_id, str)
        assert isinstance(run_dir_name, str)
        assert isinstance(created_at, str)

    def test_run_id_format(self):
        """Test run_id has ISO-8601 format with random suffix."""
        run_id, _, _ = generate_run_identifiers()
        assert "T" in run_id
        assert ":" in run_id  # ISO format has colons
        # Should have format: YYYY-MM-DDTHH:MM:SSZ-XXXXXX (6 hex chars)
        assert run_id.count("-") == 3  # 2 in date + 1 before suffix
        parts = run_id.split("-")
        suffix = parts[-1]
        assert len(suffix) == 6  # 6 hex characters
        assert all(c in "0123456789abcdef" for c in suffix)

    def test_run_dir_name_format(self):
        """Test run_dir_name is filesystem-safe with suffix."""
        _, run_dir_name, _ = generate_run_identifiers()
        assert "_" in run_dir_name
        assert ":" not in run_dir_name  # No colons in filesystem name
        # Should have format: YYYY-MM-DD_HH-MM-SSZ-XXXXXX
        assert run_dir_name.count("-") == 5  # 2 in date + 2 in time + 1 before suffix
        parts = run_dir_name.split("-")
        suffix = parts[-1]
        assert len(suffix) == 6  # 6 hex characters
        assert all(c in "0123456789abcdef" for c in suffix)

    def test_created_at_format(self):
        """Test created_at is valid ISO-8601 timestamp."""
        _, _, created_at = generate_run_identifiers()
        assert "T" in created_at
        assert created_at.endswith("Z")
        assert ":" in created_at
        # Should NOT have suffix
        assert created_at.count("-") == 2  # Only in date part

    def test_suffix_match(self):
        """Test that run_id and run_dir_name have matching suffixes and timestamps."""
        run_id, run_dir_name, _ = generate_run_identifiers()

        # Extract suffix from run_id (last part after splitting by '-')
        run_id_suffix = run_id.split("-")[-1]
        # Extract suffix from run_dir_name (last part after splitting by '-')
        run_dir_suffix = run_dir_name.split("-")[-1]

        # Both suffixes should match
        assert run_id_suffix == run_dir_suffix, "Both suffixes should match"
        assert len(run_id_suffix) == 6  # 6 hex characters
        assert all(c in "0123456789abcdef" for c in run_id_suffix)

        # Extract timestamp portions (everything before the last '-')
        # run_id format: 2026-01-08T12:34:56Z-a3f2c1
        # run_dir_name format: 2026-01-08_12-34-56Z-a3f2c1
        run_id_timestamp = run_id.rsplit("-", 1)[0]  # "2026-01-08T12:34:56Z"
        run_dir_timestamp = run_dir_name.rsplit("-", 1)[0]  # "2026-01-08_12-34-56Z"

        # Convert run_id timestamp to run_dir_name format for comparison
        # Replace 'T' with '_' and ':' with '-'
        run_id_as_dir_format = run_id_timestamp.replace("T", "_").replace(":", "-")

        # Both timestamp portions should match (after format conversion)
        assert (
            run_id_as_dir_format == run_dir_timestamp
        ), f"Timestamps should match: {run_id_as_dir_format} vs {run_dir_timestamp}"

    def test_uniqueness(self):
        """Test that consecutive calls generate unique run_ids."""
        run_id_1, _, _ = generate_run_identifiers()
        run_id_2, _, _ = generate_run_identifiers()
        assert run_id_1 != run_id_2  # Random suffix ensures uniqueness


class TestCreateResultStructure:
    """Tests for create_result_structure function."""

    def test_creates_directory(self, tmp_path):
        """Test that directory is created."""
        run_dir_name = "2026-01-08_12-34-56Z"
        result = create_result_structure(run_dir_name, str(tmp_path))

        assert result.exists()
        assert result.is_dir()
        assert result == tmp_path / run_dir_name

    def test_raises_if_exists(self, tmp_path):
        """Test that FileExistsError is raised if directory exists."""
        run_dir_name = "2026-01-08_12-34-56Z"
        create_result_structure(run_dir_name, str(tmp_path))

        with pytest.raises(FileExistsError):
            create_result_structure(run_dir_name, str(tmp_path))


class TestSaveLlmSummary:
    """Tests for save_llm_summary function."""

    def test_creates_summary_file(self, tmp_path):
        """Test that summary.json is created with correct structure."""
        run_id = "2026-01-08T12:34:56Z-abc123"
        run_dir_name = "2026-01-08_12-34-56Z-abc123"
        created_at = "2026-01-08T12:34:56Z"
        run_path = tmp_path / run_dir_name
        run_path.mkdir()

        prompts = [
            {"id": "test1", "prompt": "Test 1"},
            {"id": "test2", "prompt": "Test 2"},
        ]
        models = [{"name": "model1"}, {"name": "model2"}]

        save_llm_summary(
            run_id, run_dir_name, created_at, str(tmp_path), prompts, models
        )

        summary_file = run_path / "summary.json"
        assert summary_file.exists()

        with open(summary_file, "r", encoding="utf-8") as f:
            summary = json.load(f)

        assert summary["run_id"] == run_id
        assert summary["created_at"] == created_at
        assert summary["llm"]["prompt_count"] == 2
        assert summary["llm"]["model_count"] == 2
        assert summary["llm"]["prompts"] == ["test1", "test2"]
        assert summary["llm"]["models"] == ["model1", "model2"]

    def test_raises_if_directory_missing(self, tmp_path):
        """Test that FileNotFoundError is raised if directory doesn't exist."""
        run_id = "2026-01-08T12:34:56Z-abc123"
        run_dir_name = "nonexistent"
        created_at = "2026-01-08T12:34:56Z"
        prompts = [{"id": "test", "prompt": "Test"}]
        models = [{"name": "model"}]

        with pytest.raises(FileNotFoundError):
            save_llm_summary(
                run_id, run_dir_name, created_at, str(tmp_path), prompts, models
            )


class TestSaveLlmResult:
    """Tests for save_llm_result function."""

    def test_creates_result_file_completion(self, tmp_path):
        """Test that result file is created with correct filename for completion mode."""
        run_id = "2026-01-08T12:34:56Z-abc123"
        run_dir_name = "2026-01-08_12-34-56Z-abc123"
        created_at = "2026-01-08T12:34:56Z"
        run_path = tmp_path / run_dir_name
        run_path.mkdir()

        result_data = {
            "run_id": run_id,
            "created_at": created_at,
            "mode": "completion",
            "output": {"text": "Test response", "done_reason": "stop"},
        }

        save_llm_result(
            run_dir_name,
            str(tmp_path),
            "test_prompt",
            "gemma3:4b",
            "completion",
            result_data,
        )

        # Check JSON file
        result_file = run_path / "llm" / "test_prompt" / "gemma3_4b__completion.json"
        assert result_file.exists()

        with open(result_file, "r", encoding="utf-8") as f:
            saved_data = json.load(f)

        assert saved_data["run_id"] == run_id
        assert saved_data["created_at"] == created_at
        assert saved_data["mode"] == "completion"
        assert saved_data["output"]["text"] == "Test response"

        # Check markdown file
        markdown_file = run_path / "llm" / "test_prompt" / "markdown" / "gemma3_4b__completion.md"
        assert markdown_file.exists()

        with open(markdown_file, "r", encoding="utf-8") as f:
            markdown_content = f.read()

        assert markdown_content == "Test response"

    def test_creates_result_file_chat(self, tmp_path):
        """Test that result file is created with correct filename for chat mode."""
        run_id = "2026-01-08T12:34:56Z-abc123"
        run_dir_name = "2026-01-08_12-34-56Z-abc123"
        created_at = "2026-01-08T12:34:56Z"
        run_path = tmp_path / run_dir_name
        run_path.mkdir()

        result_data = {
            "run_id": run_id,
            "created_at": created_at,
            "mode": "chat",
            "output": {"text": "Chat response", "done_reason": "stop"},
        }

        save_llm_result(
            run_dir_name,
            str(tmp_path),
            "test_prompt",
            "llama3.1:8b",
            "chat",
            result_data,
        )

        # Check JSON file
        result_file = run_path / "llm" / "test_prompt" / "llama3.1_8b__chat.json"
        assert result_file.exists()

        with open(result_file, "r", encoding="utf-8") as f:
            saved_data = json.load(f)

        assert saved_data["run_id"] == run_id
        assert saved_data["created_at"] == created_at
        assert saved_data["mode"] == "chat"

        # Check markdown file
        markdown_file = run_path / "llm" / "test_prompt" / "markdown" / "llama3.1_8b__chat.md"
        assert markdown_file.exists()

        with open(markdown_file, "r", encoding="utf-8") as f:
            markdown_content = f.read()

        assert markdown_content == "Chat response"

    def test_creates_prompt_directory(self, tmp_path):
        """Test that prompt directory is created if it doesn't exist."""
        run_id = "2026-01-08T12:34:56Z-abc123"
        run_dir_name = "2026-01-08_12-34-56Z-abc123"
        created_at = "2026-01-08T12:34:56Z"
        run_path = tmp_path / run_dir_name
        run_path.mkdir()

        result_data = {
            "run_id": run_id,
            "created_at": created_at,
            "mode": "completion",
            "output": {"text": "Test output", "done_reason": "stop"},
        }

        save_llm_result(
            run_dir_name,
            str(tmp_path),
            "new_prompt",
            "model1",
            "completion",
            result_data,
        )

        prompt_dir = run_path / "llm" / "new_prompt"
        assert prompt_dir.exists()
        assert prompt_dir.is_dir()

    def test_raises_if_run_directory_missing(self, tmp_path):
        """Test that FileNotFoundError is raised if run directory doesn't exist."""
        result_data = {
            "run_id": "2026-01-08T12:34:56Z-abc123",
            "created_at": "2026-01-08T12:34:56Z",
            "mode": "completion",
        }

        with pytest.raises(FileNotFoundError):
            save_llm_result(
                "nonexistent",
                str(tmp_path),
                "prompt",
                "model",
                "completion",
                result_data,
            )


class TestMergeOptions:
    """Tests for merge_options function."""

    def test_all_none(self):
        """Test that empty dict is returned when all are None."""
        result = merge_options(None, None, None)
        assert result == {}

    def test_global_only(self):
        """Test that global defaults are returned when others are None."""
        global_opts = {"temperature": 0.5, "num_predict": 256}
        result = merge_options(global_opts, None, None)
        assert result == {"temperature": 0.5, "num_predict": 256}

    def test_model_only(self):
        """Test that model options are returned when others are None."""
        model_opts = {"temperature": 0.7, "num_predict": 100}
        result = merge_options(None, model_opts, None)
        assert result == {"temperature": 0.7, "num_predict": 100}

    def test_prompt_only(self):
        """Test that prompt options are returned when others are None."""
        prompt_opts = {"temperature": 0.9}
        result = merge_options(None, None, prompt_opts)
        assert result == {"temperature": 0.9}

    def test_model_overrides_global(self):
        """Test that model options override global defaults."""
        global_opts = {"temperature": 0.5, "num_predict": 256}
        model_opts = {"temperature": 0.7}
        result = merge_options(global_opts, model_opts, None)

        assert result["temperature"] == 0.7  # Overridden by model
        assert result["num_predict"] == 256  # From global

    def test_prompt_overrides_all(self):
        """Test that prompt options override both global and model options."""
        global_opts = {"temperature": 0.5, "num_predict": 256}
        model_opts = {"temperature": 0.7, "num_predict": 100}
        prompt_opts = {"temperature": 0.9, "top_p": 0.95}
        result = merge_options(global_opts, model_opts, prompt_opts)

        assert result["temperature"] == 0.9  # Overridden by prompt
        assert result["num_predict"] == 100  # From model
        assert result["top_p"] == 0.95  # From prompt


class TestGenerateResponseCompletion:
    """Tests for generate_response_completion function."""

    def test_returns_formatted_response(self):
        """Test that response is correctly formatted."""
        mock_client = Mock()
        mock_client.generate.return_value = {
            "response": "Test response text",
            "done_reason": "stop",
        }

        result = generate_response_completion(
            mock_client, "test-model", "test prompt", {}
        )

        assert result["text"] == "Test response text"
        assert result["done_reason"] == "stop"
        mock_client.generate.assert_called_once_with(
            model="test-model", prompt="test prompt", options={}
        )

    def test_handles_missing_done_reason(self):
        """Test that missing done_reason is handled."""
        mock_client = Mock()
        mock_client.generate.return_value = {"response": "Test response"}

        result = generate_response_completion(
            mock_client, "test-model", "test prompt", {}
        )

        assert result["text"] == "Test response"
        assert result["done_reason"] is None


class TestGenerateResponseChat:
    """Tests for generate_response_chat function."""

    def test_returns_formatted_response(self):
        """Test that response is correctly formatted."""
        mock_client = Mock()
        mock_client.chat.return_value = {
            "message": {"content": "Chat response text"},
            "done_reason": "stop",
        }

        messages = [{"role": "user", "content": "Hello"}]
        result = generate_response_chat(mock_client, "test-model", messages, {})

        assert result["text"] == "Chat response text"
        assert result["done_reason"] == "stop"
        mock_client.chat.assert_called_once_with(
            model="test-model", messages=messages, options={}
        )

    def test_handles_missing_done_reason(self):
        """Test that missing done_reason is handled."""
        mock_client = Mock()
        mock_client.chat.return_value = {"message": {"content": "Chat response"}}

        messages = [{"role": "user", "content": "Hello"}]
        result = generate_response_chat(mock_client, "test-model", messages, {})

        assert result["text"] == "Chat response"
        assert result["done_reason"] is None


class TestRunLlmEval:
    """Tests for run_llm_eval function."""

    def test_invalid_prompt_filter_raises_error(self):
        """Test that invalid prompt_filter raises ValueError."""
        from src.llm_runner import run_llm_eval

        config = {"results_dir": "/tmp/test"}
        prompts = [{"id": "test", "prompt": "Test"}]
        models = [{"name": "test-model"}]

        with pytest.raises(ValueError, match="Invalid prompt_filter"):
            run_llm_eval(config, prompts, models, prompt_filter="invalid")

    @pytest.mark.parametrize("prompt_filter", ["completion", "chat", "all"])
    def test_valid_prompt_filters_accepted(self, prompt_filter):
        """Test that valid prompt_filter values are accepted without ValueError."""
        from src.llm_runner import run_llm_eval
        from unittest.mock import patch

        config = {"results_dir": "/tmp/test"}
        prompts = [{"id": "test", "prompt": "Test"}]
        models = [{"name": "test-model"}]

        # Mock all external dependencies to test only prompt_filter validation
        with patch("src.llm_runner.create_result_structure"), \
             patch("src.llm_runner.save_llm_summary"), \
             patch("src.llm_runner.save_llm_result"), \
             patch("src.llm_runner.ollama.Client"):
            # Should not raise ValueError for valid prompt_filter values
            run_llm_eval(config, prompts, models, prompt_filter=prompt_filter)
