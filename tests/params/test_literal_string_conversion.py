"""
Tests for Literal-aware string conversion in parameter processing.

This module tests the intelligent handling of string values that match Literal
type options, particularly for cases where "none" is both:
- A valid Literal value (e.g., recurrence_type="none" to disable recurrence)
- A string that would normally convert to Python None

The fix ensures that when "none" is a valid Literal option, it's preserved as
a string instead of being converted to None and filtered out.

Run with: pytest tests/params/test_literal_string_conversion.py -v
"""

import pytest
import tempfile
import yaml
from pathlib import Path
from typing import Dict, Any

from dynvision.params import ModelParams, InitParams
from dynvision.models.dyrcnn import DyRCNNx8


def create_minimal_config(overrides: Dict[str, Any] = None) -> Path:
    """Create a minimal config file for testing."""
    config = {
        "seed": 0,
        "log_level": "INFO",
        "model_name": "DyRCNNx8",
        **(overrides or {}),
    }

    temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    yaml.dump(config, temp_file)
    temp_file.close()
    return Path(temp_file.name)


class TestLiteralStringConversion:
    """Test suite for Literal-aware string conversion."""

    def test_literal_none_preserved_as_string(self):
        """Test that 'none' is preserved as string when it's a valid Literal value."""
        params = ModelParams(
            model_name="DyRCNNx8",
            seed=0,
            log_level="INFO",
            recurrence_type="none",  # Valid Literal value
        )

        assert params.recurrence_type == "none"
        assert isinstance(params.recurrence_type, str)

    def test_literal_none_case_insensitive(self):
        """Test that Literal matching is case-insensitive."""
        test_cases = ["none", "None", "NONE", "nOnE"]

        for value in test_cases:
            params = ModelParams(
                model_name="DyRCNNx8",
                seed=0,
                log_level="INFO",
                recurrence_type=value,
            )
            # Should normalize to lowercase
            assert params.recurrence_type == "none"
            assert isinstance(params.recurrence_type, str)

    def test_other_literal_values_work(self):
        """Test that other Literal values are unaffected."""
        literal_values = ["full", "self", "depthwise", "pointwise", "local"]

        for value in literal_values:
            params = ModelParams(
                model_name="DyRCNNx8",
                seed=0,
                log_level="INFO",
                recurrence_type=value,
            )
            assert params.recurrence_type == value
            assert isinstance(params.recurrence_type, str)

    def test_none_still_converts_to_python_none(self):
        """Test that None/null still convert to Python None when not provided."""
        params = ModelParams(
            model_name="DyRCNNx8",
            seed=0,
            log_level="INFO",
            # recurrence_type not provided
        )

        assert params.recurrence_type is None

    def test_cli_parsing_preserves_literal_none(self):
        """Test that CLI parsing preserves 'none' as string for Literal fields."""
        config_path = create_minimal_config()

        try:
            params = ModelParams.from_cli_and_config(
                config_path=str(config_path),
                args=["--rctype", "none"],  # Using alias
            )

            assert params.recurrence_type == "none"
            assert isinstance(params.recurrence_type, str)
        finally:
            config_path.unlink()

    def test_cli_alias_rctype_none(self):
        """Test that rctype alias works with 'none' value."""
        config_path = create_minimal_config()

        try:
            params = ModelParams.from_cli_and_config(
                config_path=str(config_path),
                args=["--rctype", "none"],
            )

            assert params.recurrence_type == "none"
        finally:
            config_path.unlink()

    def test_config_file_preserves_literal_none(self):
        """Test that config file values are preserved for Literal fields."""
        config_path = create_minimal_config({"recurrence_type": "none"})

        try:
            params = ModelParams.from_cli_and_config(
                config_path=str(config_path), args=[]
            )

            assert params.recurrence_type == "none"
        finally:
            config_path.unlink()

    def test_cli_overrides_config_for_literal_none(self):
        """Test CLI precedence over config file for Literal values."""
        config_path = create_minimal_config({"recurrence_type": "full"})

        try:
            params = ModelParams.from_cli_and_config(
                config_path=str(config_path),
                args=["--recurrence_type", "none"],
            )

            assert params.recurrence_type == "none"
        finally:
            config_path.unlink()

    def test_none_filtering_in_get_model_kwargs(self):
        """Test that None values are still filtered out in get_model_kwargs."""
        params = ModelParams(
            model_name="DyRCNNx8",
            seed=0,
            log_level="INFO",
            recurrence_type=None,  # Explicitly None
            n_classes=10,  # Set at least one value to avoid filter_kwargs error
        )

        kwargs = params.get_model_kwargs(DyRCNNx8)

        # None values should be filtered out
        assert "recurrence_type" not in kwargs
        # Non-None values should be present
        assert "n_classes" in kwargs
        assert kwargs["n_classes"] == 10

    def test_literal_none_passed_to_model_kwargs(self):
        """Test that 'none' string is passed to model kwargs, not filtered."""
        params = ModelParams(
            model_name="DyRCNNx8",
            seed=0,
            log_level="INFO",
            recurrence_type="none",  # String literal
            n_classes=10,
        )

        kwargs = params.get_model_kwargs(DyRCNNx8)

        # String 'none' should NOT be filtered
        assert "recurrence_type" in kwargs
        assert kwargs["recurrence_type"] == "none"

    def test_model_creation_with_literal_none(self):
        """Test that model can be created with recurrence_type='none'."""
        params = ModelParams(
            model_name="DyRCNNx8",
            seed=0,
            log_level="INFO",
            recurrence_type="none",
            n_classes=10,
        )

        kwargs = params.get_model_kwargs(DyRCNNx8)
        model = DyRCNNx8(**kwargs)

        assert model.recurrence_type == "none"
        assert isinstance(model.recurrence_type, str)

    def test_convert_string_value_directly(self):
        """Test the _convert_string_value method directly."""
        # Test Literal field: 'none' should be preserved
        result = ModelParams._convert_string_value("none", "recurrence_type")
        assert result == "none"
        assert isinstance(result, str)

        # Test non-existent field: 'none' should convert to None
        result = ModelParams._convert_string_value("none", "nonexistent_field")
        assert result is None

        # Test other Literal values
        result = ModelParams._convert_string_value("full", "recurrence_type")
        assert result == "full"

    def test_null_string_conversion(self):
        """Test that 'null' behaves like 'none' for non-Literal fields."""
        # For non-Literal fields, both should convert to None
        result1 = ModelParams._convert_string_value("none", "nonexistent_field")
        result2 = ModelParams._convert_string_value("null", "nonexistent_field")

        assert result1 is None
        assert result2 is None

    def test_empty_string_conversion(self):
        """Test that empty string converts to None."""
        result = ModelParams._convert_string_value("", "nonexistent_field")
        assert result is None

    def test_boolean_literal_not_affected(self):
        """Test that boolean values are not affected by Literal logic."""
        # Test 'true'/'false' strings
        result_true = ModelParams._convert_string_value("true", "some_field")
        result_false = ModelParams._convert_string_value("false", "some_field")

        assert result_true is True
        assert result_false is False

    def test_numeric_string_conversion_not_affected(self):
        """Test that numeric string conversion is not affected."""
        # This would need a numeric field to test properly
        # For now, just verify non-numeric strings stay as strings
        result = ModelParams._convert_string_value("not_a_number", "some_field")
        assert result == "not_a_number"
        assert isinstance(result, str)

    def test_snakemake_wildcard_scenario(self):
        """Test the actual Snakemake use case: model_args wildcard with rctype=none."""
        # Simulate what happens when Snakemake parses:
        # model_name: DyRCNNx8
        # model_args: :rctype=none
        # This gets converted to CLI: --rctype none

        config_path = create_minimal_config()

        try:
            params = ModelParams.from_cli_and_config(
                config_path=str(config_path),
                args=["--rctype", "none"],
            )

            # Verify the parameter is set correctly
            assert params.recurrence_type == "none"

            # Verify it's passed to model kwargs
            kwargs = params.get_model_kwargs(DyRCNNx8)
            assert kwargs.get("recurrence_type") == "none"

        finally:
            config_path.unlink()


class TestLiteralConversionEdgeCases:
    """Test edge cases and potential issues."""

    def test_literal_with_mixed_case_in_definition(self):
        """Test that case normalization works correctly."""
        # recurrence_type Literal has lowercase values
        # Ensure uppercase input normalizes correctly
        params = ModelParams(
            model_name="DyRCNNx8",
            seed=0,
            log_level="INFO",
            recurrence_type="FULL",
        )
        assert params.recurrence_type == "full"

    def test_invalid_literal_value_rejected(self):
        """Test that invalid Literal values are rejected by Pydantic."""
        with pytest.raises(Exception):  # Pydantic ValidationError
            ModelParams(
                model_name="DyRCNNx8",
                seed=0,
                log_level="INFO",
                recurrence_type="invalid_value",
            )

    def test_backward_compatibility_none_default(self):
        """Test that existing code relying on None defaults still works."""
        # When recurrence_type is not specified, it should default to None
        params = ModelParams(
            model_name="DyRCNNx8",
            seed=0,
            log_level="INFO",
            n_classes=10,  # Set at least one value to avoid filter_kwargs error
        )

        assert params.recurrence_type is None

        # This None should be filtered in get_model_kwargs
        kwargs = params.get_model_kwargs(DyRCNNx8)
        assert "recurrence_type" not in kwargs
        # But non-None values should be present
        assert "n_classes" in kwargs


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
