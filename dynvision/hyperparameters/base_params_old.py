"""
Base parameter handling system for DynVision using Pydantic.

This module provides the foundation for type-safe, validated parameter management
across all DynVision scripts with support for CLI parsing, config file loading,
and alias resolution.
"""

from pydantic import BaseModel, Field, field_validator, ConfigDict, model_validator
from typing import Dict, Any, Optional, List, Union, Type
from pathlib import Path
import argparse
import yaml
import json
import logging
import sys
from dynvision.utils import str_to_bool


class DynVisionValidationError(ValueError):
    """Custom validation error for DynVision parameter validation."""

    def __init__(self, message: str, field_name: Optional[str] = None):
        self.field_name = field_name
        super().__init__(f"{field_name}: {message}" if field_name else message)


class DynVisionConfigError(Exception):
    """Configuration-specific error with context information."""

    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        self.context = context or {}
        super().__init__(f"{message}\nContext: {self.context}")


class BaseParams(BaseModel):
    """
    Base parameter class providing common functionality for all DynVision configs.

    Handles CLI argument parsing, configuration file loading, alias resolution,
    and provides validation patterns for consistent parameter management.
    """

    # Common parameters that appear across scripts
    seed: int = Field(default=42, description="Random seed for reproducibility")
    log_level: str = Field(
        default="INFO",
        description="Logging level",
        pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$",
    )

    model_config = ConfigDict(
        extra="allow",  # Allow additional fields
        validate_assignment=True,  # Validate fields when assigned after creation
        use_enum_values=True,  # Use enum values in serialization
        validate_by_name=True,  # Allow validation using field names and aliases
    )

    @classmethod
    def get_aliases(cls) -> Dict[str, str]:
        """
        Return mapping of aliases to full parameter names.

        Override in child classes to add specific aliases.

        Returns:
            Dict mapping alias names to full parameter names
        """
        return {
            "log": "log_level",
        }

    @classmethod
    def _resolve_aliases(cls, params: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve aliases to full parameter names."""
        aliases = cls.get_aliases()
        resolved = {}

        for key, value in params.items():
            if key in aliases.keys():
                full_name = aliases[key]
                # If both alias and full name provided, alias takes precedence
                resolved[full_name] = value
            elif key not in resolved:
                resolved[key] = value

        return resolved

    def update_field(
        self, field_name: str, new_value: Any, verbose: bool = False
    ) -> None:
        """
        Update a single field in-place with automatic validation.

        Args:
            field_name: Name of field to update (supports dot notation)
            new_value: New value to set
            verbose: Whether to log the change
        """
        if "." in field_name:
            # Handle nested fields like "model.n_classes"
            parts = field_name.split(".", 1)
            obj = getattr(self, parts[0])
            obj.update_field(parts[1], new_value, verbose)
            return

        if verbose:
            old_value = getattr(self, field_name, None)
            logging.info(f"Updating {field_name}: {old_value} -> {new_value}")

        # Pydantic automatically validates on setattr
        setattr(self, field_name, new_value)

    @model_validator(mode="before")
    @classmethod
    def convert_string_values(cls, values):
        """
        Convert string representations to appropriate Python types.

        Handles:
        - 'None', 'none', 'null', '' -> None
        - 'True', 'true', 'False', 'false' -> bool
        - String numbers -> int/float (if field expects numeric type)

        This applies to ALL parameter sources (CLI, config files, direct instantiation).
        """
        if isinstance(values, dict):
            for key, value in values.items():
                if isinstance(value, str):
                    values[key] = cls._convert_string_value(value, key)
        return values

    @classmethod
    def _convert_string_value(cls, value: str, field_name: str = None) -> Any:
        """Convert a single string value to appropriate type."""

        # Handle None representations
        if value.lower() in ("none", "null", ""):
            return None

        # Handle boolean representations
        if value.lower() in ("true", "false"):
            return value.lower() == "true"

        # Handle numeric representations (only if field expects numeric type)
        if field_name and field_name in cls.model_fields:
            field_info = cls.model_fields[field_name]
            expected_type = getattr(field_info, "annotation", str)

            # Handle Optional types
            if (
                hasattr(expected_type, "__origin__")
                and expected_type.__origin__ is Union
            ):
                args = expected_type.__args__
                if len(args) == 2 and type(None) in args:
                    expected_type = args[0] if args[1] is type(None) else args[1]

            # Try to convert to expected numeric type
            if expected_type in (int, float):
                try:
                    return expected_type(value)
                except ValueError:
                    pass  # Keep as string if conversion fails

        # Return as string if no conversion applies
        return value

    @classmethod
    def from_cli_and_config(
        cls,
        config_path: Optional[Union[str, Path]] = None,
        override_kwargs: Optional[Dict[str, Any]] = None,
        args: Optional[List[str]] = None,
    ) -> "BaseParams":
        """
        Create config instance with proper precedence: override_kwargs > CLI args > config file > defaults.

        Args:
            config_path: Path to YAML/JSON configuration file
            override_kwargs: Direct parameter overrides (highest priority)
            args: CLI arguments list (None to use sys.argv)

        Returns:
            Configured parameter instance
        """
        # Load from config file first (lowest priority)
        config_data = {}
        if config_path:
            config_data = cls._load_config_file(config_path)

        # Parse CLI args (medium priority)
        cli_args = cls._parse_cli_args(args)

        # Apply override kwargs (highest priority)
        if override_kwargs:
            cli_args.update(override_kwargs)

        # Merge with precedence and resolve aliases
        final_params = {**config_data, **cli_args}
        final_params = cls._resolve_aliases(final_params)

        try:
            return cls(**final_params)
        except Exception as e:
            raise DynVisionValidationError(f"Parameter validation failed: {e}")

    @classmethod
    def _load_config_file(cls, config_path: Union[str, Path]) -> Dict[str, Any]:
        """Load configuration from YAML or JSON file (flat structure)."""
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r") as f:
            if config_path.suffix.lower() in [".yml", ".yaml"]:
                data = yaml.safe_load(f) or {}
            elif config_path.suffix.lower() == ".json":
                data = json.load(f) or {}
            else:
                raise FileNotFoundError(
                    f"Unsupported config file format: {config_path.suffix}"
                )

        return data

    @classmethod
    def _parse_cli_args(cls, args: Optional[List[str]] = None) -> Dict[str, Any]:
        """Parse CLI arguments based on model fields."""
        parser = argparse.ArgumentParser(
            description=f"Parameters for {cls.__name__}",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )

        # Add arguments for each field
        for field_name, field_info in cls.model_fields.items():
            cls._add_field_argument(parser, field_name, field_info)

        # Add alias arguments
        for alias, full_name in cls.get_aliases().items():
            if full_name in cls.model_fields:
                field_info = cls.model_fields[full_name]
                cls._add_field_argument(parser, alias, field_info, is_alias=True)

        # Parse arguments
        if args is None:
            args = sys.argv[1:]

        parsed_args, parsed_unknown = parser.parse_known_args(args)
        unknown_kwargs = cls._parse_unknown_args(parsed_unknown)

        # Only return non-None values to allow proper precedence
        param_dict = {k: v for k, v in vars(parsed_args).items() if v is not None}
        param_dict.update(unknown_kwargs)
        return param_dict

    @classmethod
    def _add_field_argument(
        cls,
        parser: argparse.ArgumentParser,
        name: str,
        field_info,
        is_alias: bool = False,
    ):
        """Add a single field as CLI argument."""
        arg_name = f"--{name}"

        # Base argument configuration
        kwargs = {
            "help": field_info.description or f"Parameter {name}",
            "dest": name,
            "default": argparse.SUPPRESS,  # Let Pydantic handle defaults
        }

        # Type-specific handling
        if hasattr(field_info, "type_"):
            field_type = field_info.type_
        elif hasattr(field_info, "annotation"):
            field_type = field_info.annotation
        else:
            field_type = str

        # Handle Optional types
        if hasattr(field_type, "__origin__") and field_type.__origin__ is Union:
            args = field_type.__args__
            if len(args) == 2 and type(None) in args:
                field_type = args[0] if args[1] is type(None) else args[1]

        # Configure argument based on type
        if field_type == bool:
            kwargs["type"] = str_to_bool
        elif field_type in [int, float, str]:
            kwargs["type"] = field_type
        elif hasattr(field_type, "__origin__") and field_type.__origin__ is list:
            kwargs["nargs"] = "+"
            if len(field_type.__args__) > 0:
                kwargs["type"] = field_type.__args__[0]

        parser.add_argument(arg_name, **kwargs)

    @classmethod
    def _parse_unknown_args(cls, unknown_args: List[str]) -> Dict[str, Any]:
        """
        Parse unknown command line arguments into parameter dictionary.

        Converts arguments of the form:
        --param1 value1 --param2 value2 --flag --list_param val1 val2
        into:
        {'param1': 'value1', 'param2': 'value2', 'flag': True, 'list_param': ['val1', 'val2']}

        Args:
            unknown_args: List of unknown command line arguments

        Returns:
            Dictionary of parameter names and values

        Raises:
            DynVisionValidationError: If parsing fails
        """
        if not unknown_args:
            return {}

        params = {}
        current_key = None
        current_values = []

        i = 0
        while i < len(unknown_args):
            arg = unknown_args[i]

            if arg.startswith("-"):
                # Save previous parameter if exists
                if current_key is not None:
                    params[current_key] = cls._process_parameter_value(current_values)

                # Start new parameter
                current_key = arg.lstrip("-")
                current_values = []

                # Check if next arg is also a flag (boolean parameter)
                if i + 1 >= len(unknown_args) or unknown_args[i + 1].startswith("-"):
                    params[current_key] = True
                    current_key = None

            elif current_key is not None:
                # Collect values for current parameter
                current_values.append(arg)
            else:
                # Orphaned value - log warning but continue
                logging.warning(f"Orphaned CLI argument ignored: {arg}")

            i += 1

        # Handle last parameter
        if current_key is not None:
            params[current_key] = cls._process_parameter_value(current_values)

        return params

    @classmethod
    def _process_parameter_value(cls, values: List[str]) -> Any:
        """
        Simplified parameter value processing - just handle list vs single value.
        Type conversion is now handled by convert_string_values validator.
        """
        if not values:
            return True  # Boolean flag with no value

        if len(values) == 1:
            return values[0]  # Single value - type conversion handled by validator
        else:
            return values  # Multiple values as list - type conversion handled by validator

    def setup_logging(self) -> None:
        """Configure logging based on log_level parameter."""
        logging.basicConfig(
            level=getattr(logging, self.log_level.upper()),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            force=True,  # Override existing configuration
        )

        logger = logging.getLogger(__name__)
        logger.info(f"Logging configured at {self.log_level} level")

    @classmethod
    def get_script_context(cls) -> Optional[str]:
        """
        Determine which script is currently running for context-aware validation.

        Returns:
            Script context ('init_model', 'train_model', 'test_model', or None)
        """
        if len(sys.argv) > 0:
            script_name = Path(sys.argv[0]).stem.lower()
            if "init" in script_name:
                return "init_model"
            elif "train" in script_name:
                return "train_model"
            elif "test" in script_name:
                return "test_model"
        return None

    @field_validator("log_level")
    def validate_log_level(cls, v):
        """Ensure log_level is valid."""
        if isinstance(v, str):
            v = v.upper()
            if v not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
                raise ValueError(f"Invalid log level: {v}")
        return v

    def validate_consistency_with(
        self, other: "BaseParams", fields_to_check: Optional[List[str]] = None
    ) -> List[str]:
        """
        Check parameter consistency with another config instance.

        Args:
            other: Another parameter instance to compare with
            fields_to_check: Specific fields to check (None for common fields)

        Returns:
            List of inconsistent field names
        """
        if fields_to_check is None:
            # Check common fields between instances
            fields_to_check = set(self.model_fields.keys()) & set(
                other.model_fields.keys()
            )

        inconsistent = []
        for field_name in fields_to_check:
            if hasattr(self, field_name) and hasattr(other, field_name):
                self_value = getattr(self, field_name)
                other_value = getattr(other, field_name)

                if self_value != other_value:
                    inconsistent.append(field_name)
                    logging.warning(
                        f"Parameter inconsistency in '{field_name}': "
                        f"{self_value} vs {other_value}"
                    )

        return inconsistent

    def to_dict(self, exclude_none: bool = True) -> Dict[str, Any]:
        """Convert to dictionary, optionally excluding None values."""
        return self.dict(exclude_none=exclude_none)

    def to_flat_dict(self) -> Dict[str, Any]:
        """Convert to flat dictionary suitable for config files."""
        # For now, same as to_dict since BaseParams is already flat
        # TODO: Can be extended for hierarchical configs if needed
        return self.to_dict()


# Example usage and testing
if __name__ == "__main__":
    # Example of how BaseParams would be used

    # Test basic instantiation
    params = BaseParams(seed=123, log_level="DEBUG")
    print(f"Basic params: {params}")

    # Test CLI parsing with known and unknown args
    try:
        # This would work with actual CLI args like: --seed 456 --log DEBUG --custom_param value --flag
        test_args = [
            "--seed",
            "456",
            "--log",
            "DEBUG",
            "--custom_param",
            "value",
            "--flag",
            "--list_param",
            "a",
            "b",
            "c",
        ]
        cli_params = BaseParams.from_cli_and_config(args=test_args)
        print(f"CLI params with unknown args: {cli_params}")
    except DynVisionValidationError as e:
        print(f"CLI parsing (expected to fail without valid CLI context): {e}")

    # Test unknown args parsing directly
    unknown_args = [
        "--model_type",
        "RCNN",
        "--learning_rate",
        "0.001",
        "--use_cuda",
        "--layers",
        "64",
        "128",
        "256",
    ]
    parsed_unknown = BaseParams._parse_unknown_args(unknown_args)
    print(f"Parsed unknown args: {parsed_unknown}")

    # Test type inference
    type_test_args = [
        "--int_param",
        "42",
        "--float_param",
        "3.14",
        "--bool_param",
        "true",
        "--string_param",
        "hello",
    ]
    parsed_types = BaseParams._parse_unknown_args(type_test_args)
    print(f"Type inference test: {parsed_types}")
    print(
        f"Types: int={type(parsed_types['int_param'])}, float={type(parsed_types['float_param'])}, bool={type(parsed_types['bool_param'])}, str={type(parsed_types['string_param'])}"
    )

    # Test config file simulation
    import tempfile
    import os

    # Create temporary config file
    config_data = {"seed": 789, "log_level": "WARNING"}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config_data, f)
        temp_config = f.name

    try:
        file_params = BaseParams.from_cli_and_config(config_path=temp_config)
        print(f"Config file params: {file_params}")
    finally:
        os.unlink(temp_config)

    # Test validation
    params.setup_logging()
    print("Logging configured successfully")

    print(f"Script context: {BaseParams.get_script_context()}")
