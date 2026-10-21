"""Unit tests for src/utils.py"""

from prompt_runner.utils import (
    generate_run_identifiers,
    merge_options,
    merged_server_args,
    random_seed,
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


class TestMergeOptions:
    """Tests for merge_options function."""

    def test_all_none(self):
        """Test that empty dict is returned when all are None."""
        result = merge_options(None, None, None)
        assert result == {}

    def test_global_only(self):
        """Test that global defaults are returned when others are None."""
        global_opts = {"width": 512, "height": 512}
        result = merge_options(global_opts, None, None)
        assert result == {"width": 512, "height": 512}

    def test_prompt_overrides_all(self):
        """Test that prompt options override both global and model options."""
        global_opts = {"width": 512, "cfg_scale": 7.0}
        model_opts = {"cfg_scale": 1.0, "sample_steps": 6}
        prompt_opts = {"seed": 42, "cfg_scale": 3.5}
        result = merge_options(global_opts, model_opts, prompt_opts)

        assert result["width"] == 512  # From global
        assert result["sample_steps"] == 6  # From model
        assert result["cfg_scale"] == 3.5  # Overridden by prompt
        assert result["seed"] == 42  # From prompt

    def test_nested_mappings_merge_and_lists_replace(self):
        """Mappings merge key by key; lists and scalars are replaced."""
        global_opts = {"sample_params": {"sample_steps": 4, "scheduler": "discrete"}}
        model_opts = {"sample_params": {"sample_steps": 20}, "lora": [{"path": "a"}]}
        prompt_opts = {"lora": [{"path": "b", "multiplier": 0.5}]}
        result = merge_options(global_opts, model_opts, prompt_opts)

        assert result["sample_params"] == {"sample_steps": 20, "scheduler": "discrete"}
        assert result["lora"] == [{"path": "b", "multiplier": 0.5}]

    def test_result_shares_no_state_with_inputs(self):
        """Mutating the result must not change the inputs."""
        model_opts = {"sample_params": {"sample_steps": 4}, "lora": [{"path": "a"}]}
        result = merge_options(None, model_opts, None)
        result["sample_params"]["sample_steps"] = 99
        result["lora"].append({"path": "b"})

        assert model_opts == {
            "sample_params": {"sample_steps": 4},
            "lora": [{"path": "a"}],
        }


class TestMergedServerArgs:
    def test_the_models_flags_win_over_the_defaults(self):
        model = {"name": "m", "server_args": {"-c": 8192, "-m": "/m.gguf"}}
        assert merged_server_args({"-c": 4096, "-np": 1}, model) == {
            "-c": 8192,
            "-np": 1,
            "-m": "/m.gguf",
        }

    def test_no_defaults(self):
        model = {"name": "m", "server_args": {"-m": "/m.gguf"}}
        assert merged_server_args(None, model) == {"-m": "/m.gguf"}


class TestRandomSeed:
    def test_fits_a_32_bit_seed(self):
        seeds = {random_seed() for _ in range(50)}
        assert all(0 <= seed < 2**31 for seed in seeds)
        assert len(seeds) > 1
