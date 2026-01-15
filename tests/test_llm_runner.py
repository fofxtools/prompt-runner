"""Unit tests for src/llm_runner.py"""

import json
import pytest
from unittest.mock import Mock
from prompt_runner.llm_runner import (
    save_llm_summary,
    save_llm_result,
    generate_response_completion,
    generate_response_chat,
)


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
        model_timings = {"model1": 12.34, "model2": 8.56}

        save_llm_summary(
            run_id,
            run_dir_name,
            created_at,
            str(tmp_path),
            prompts,
            models,
            model_timings,
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
        assert summary["llm"]["model_timings"] == {"model1": 12.34, "model2": 8.56}

    def test_raises_if_directory_missing(self, tmp_path):
        """Test that FileNotFoundError is raised if directory doesn't exist."""
        run_id = "2026-01-08T12:34:56Z-abc123"
        run_dir_name = "nonexistent"
        created_at = "2026-01-08T12:34:56Z"
        prompts = [{"id": "test", "prompt": "Test"}]
        models = [{"name": "model"}]
        model_timings = {}

        with pytest.raises(FileNotFoundError):
            save_llm_summary(
                run_id,
                run_dir_name,
                created_at,
                str(tmp_path),
                prompts,
                models,
                model_timings,
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
        markdown_file = (
            run_path / "llm" / "test_prompt" / "markdown" / "gemma3_4b__completion.md"
        )
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
        markdown_file = (
            run_path / "llm" / "test_prompt" / "markdown" / "llama3.1_8b__chat.md"
        )
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


class TestGenerateResponseCompletion:
    """Tests for generate_response_completion function."""

    def test_returns_formatted_response(self):
        """Test that response is correctly formatted."""
        mock_client = Mock()
        mock_client.generate.return_value = {
            "response": "Test response text",
            "done_reason": "stop",
            "prompt_eval_count": 10,
            "eval_count": 20,
            "load_duration": 1000000,  # 0.001 seconds in nanoseconds
            "prompt_eval_duration": 1234000000,  # 1.234 seconds in nanoseconds
            "eval_duration": 2345000000,  # 2.345 seconds in nanoseconds
            "total_duration": 3580000000,  # 3.580 seconds in nanoseconds
        }

        result = generate_response_completion(
            mock_client, "test-model", "test prompt", {}
        )

        assert result["output"]["text"] == "Test response text"
        assert result["metrics"]["done_reason"] == "stop"
        assert result["metrics"]["input_tokens"] == 10
        assert result["metrics"]["output_tokens"] == 20
        assert result["metrics"]["total_tokens"] == 30
        assert result["metrics"]["load_seconds"] == 0.001
        assert result["metrics"]["input_seconds"] == 1.234
        assert result["metrics"]["output_seconds"] == 2.345
        assert result["metrics"]["total_seconds"] == 3.580
        assert (
            result["metrics"]["output_tokens_per_second"] == 8.529
        )  # 20 / 2.345 = 8.529
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

        assert result["output"]["text"] == "Test response"
        assert result["metrics"]["done_reason"] is None
        assert result["metrics"]["input_tokens"] is None
        assert result["metrics"]["output_tokens"] is None
        assert result["metrics"]["total_tokens"] is None
        assert result["metrics"]["load_seconds"] is None
        assert result["metrics"]["input_seconds"] is None
        assert result["metrics"]["output_seconds"] is None
        assert result["metrics"]["total_seconds"] is None
        assert result["metrics"]["output_tokens_per_second"] is None


class TestGenerateResponseChat:
    """Tests for generate_response_chat function."""

    def test_returns_formatted_response(self):
        """Test that response is correctly formatted."""
        mock_client = Mock()
        mock_client.chat.return_value = {
            "message": {"content": "Chat response text"},
            "done_reason": "stop",
            "prompt_eval_count": 15,
            "eval_count": 25,
            "load_duration": 2000000,  # 0.002 seconds in nanoseconds
            "prompt_eval_duration": 1500000000,  # 1.5 seconds in nanoseconds
            "eval_duration": 2500000000,  # 2.5 seconds in nanoseconds
            "total_duration": 4002000000,  # 4.002 seconds in nanoseconds
        }

        messages = [{"role": "user", "content": "Hello"}]
        result = generate_response_chat(mock_client, "test-model", messages, {})

        assert result["output"]["text"] == "Chat response text"
        assert result["metrics"]["done_reason"] == "stop"
        assert result["metrics"]["input_tokens"] == 15
        assert result["metrics"]["output_tokens"] == 25
        assert result["metrics"]["total_tokens"] == 40
        assert result["metrics"]["load_seconds"] == 0.002
        assert result["metrics"]["input_seconds"] == 1.5
        assert result["metrics"]["output_seconds"] == 2.5
        assert result["metrics"]["total_seconds"] == 4.002
        assert result["metrics"]["output_tokens_per_second"] == 10.0  # 25 / 2.5
        mock_client.chat.assert_called_once_with(
            model="test-model", messages=messages, options={}
        )

    def test_handles_missing_done_reason(self):
        """Test that missing done_reason is handled."""
        mock_client = Mock()
        mock_client.chat.return_value = {"message": {"content": "Chat response"}}

        messages = [{"role": "user", "content": "Hello"}]
        result = generate_response_chat(mock_client, "test-model", messages, {})

        assert result["output"]["text"] == "Chat response"
        assert result["metrics"]["done_reason"] is None
        assert result["metrics"]["input_tokens"] is None
        assert result["metrics"]["output_tokens"] is None
        assert result["metrics"]["total_tokens"] is None
        assert result["metrics"]["load_seconds"] is None
        assert result["metrics"]["input_seconds"] is None
        assert result["metrics"]["output_seconds"] is None
        assert result["metrics"]["total_seconds"] is None
        assert result["metrics"]["output_tokens_per_second"] is None


class TestRunLlmEval:
    """Tests for run_llm_eval function."""

    def test_invalid_prompt_filter_raises_error(self):
        """Test that invalid prompt_filter raises ValueError."""
        from prompt_runner.llm_runner import run_llm_eval

        config = {"results_dir": "/tmp/test"}
        prompts = [{"id": "test", "prompt": "Test"}]
        models = [{"name": "test-model"}]

        with pytest.raises(ValueError, match="Invalid prompt_filter"):
            run_llm_eval(config, prompts, models, prompt_filter="invalid")

    @pytest.mark.parametrize("prompt_filter", ["completion", "chat", "all"])
    def test_valid_prompt_filters_accepted(self, prompt_filter):
        """Test that valid prompt_filter values are accepted without ValueError."""
        from prompt_runner.llm_runner import run_llm_eval
        from unittest.mock import patch

        config = {"results_dir": "/tmp/test"}
        prompts = [{"id": "test", "prompt": "Test"}]
        models = [{"name": "test-model"}]

        # Mock all external dependencies to test only prompt_filter validation
        with (
            patch("prompt_runner.utils.create_result_structure"),
            patch("prompt_runner.llm_runner.save_llm_summary"),
            patch("prompt_runner.llm_runner.save_llm_result"),
            patch("prompt_runner.llm_runner.ollama.Client"),
        ):
            # Should not raise ValueError for valid prompt_filter values
            run_llm_eval(config, prompts, models, prompt_filter=prompt_filter)
