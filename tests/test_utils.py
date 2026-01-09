"""Unit tests for src/utils.py"""

import pytest
from prompt_runner.utils import (
    create_result_structure,
    generate_run_identifiers,
    merge_options,
    sanitize_fs_name,
)


class TestSanitizeFsName:
    """Tests for sanitize_fs_name function."""

    def test_iso_timestamp(self):
        """Test sanitizing ISO-8601 timestamp."""
        assert (
            sanitize_fs_name("2026-01-07T12:34:56.789012Z")
            == "2026-01-07T12_34_56.789012Z"
        )

    def test_model_name_with_slashes_and_colons(self):
        """Test sanitizing model name with slashes and colons."""
        assert sanitize_fs_name("model/name:tag") == "model_name_tag"

    def test_already_safe_name(self):
        """Test that safe names are unchanged."""
        assert sanitize_fs_name("simple_name") == "simple_name"

    def test_all_problematic_characters(self):
        """Test all problematic characters are replaced."""
        assert sanitize_fs_name('complex<>:"/\\|?*name') == "complex_________name"

    def test_empty_string(self):
        """Test empty string."""
        assert sanitize_fs_name("") == ""


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
