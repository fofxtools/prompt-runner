"""Unit tests for src/utils.py"""

from src.utils import sanitize_fs_name


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
