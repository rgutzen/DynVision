"""Configuration validation system for the rhythmic visual attention project.

This module provides validation for the configuration system, ensuring that:
- Required parameters are present
- Parameter types are correct
- Values are within valid ranges
- Dependencies between parameters are satisfied

Example:
    ```python
    from validate_config import validate_configuration
    
    # Load and validate configuration
    config = load_config()
    validate_configuration(config)
    ```
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('config.validation')

class ConfigurationError(Exception):
    """Base class for configuration validation errors."""
    pass

class MissingParameterError(ConfigurationError):
    """Error raised when a required parameter is missing."""
    pass

class InvalidValueError(ConfigurationError):
    """Error raised when a parameter value is invalid."""
    pass

class DependencyError(ConfigurationError):
    """Error raised when parameter dependencies are not satisfied."""
    pass

def validate_type(
    value: Any,
    expected_type: Union[type, tuple],
    path: str
) -> None:
    """Validate that a value is of the expected type.

    Args:
        value: Value to validate
        expected_type: Expected type or tuple of types
        path: Configuration path for error messages

    Raises:
        InvalidValueError: If value is not of expected type
    """
    if not isinstance(value, expected_type):
        raise InvalidValueError(
            f"Parameter '{path}' must be of type {expected_type}, "
            f"got {type(value)}"
        )

def validate_numeric_range(
    value: Union[int, float],
    min_value: Optional[Union[int, float]] = None,
    max_value: Optional[Union[int, float]] = None,
    path: str = ""
) -> None:
    """Validate that a numeric value is within the specified range.

    Args:
        value: Value to validate
        min_value: Minimum allowed value (inclusive)
        max_value: Maximum allowed value (inclusive)
        path: Configuration path for error messages

    Raises:
        InvalidValueError: If value is outside allowed range
    """
    if min_value is not None and value < min_value:
        raise InvalidValueError(
            f"Parameter '{path}' must be >= {min_value}, got {value}"
        )
    if max_value is not None and value > max_value:
        raise InvalidValueError(
            f"Parameter '{path}' must be <= {max_value}, got {value}"
        )

def validate_model_config(config: Dict[str, Any]) -> None:
    """Validate model configuration parameters.

    Args:
        config: Model configuration dictionary

    Raises:
        ConfigurationError: If validation fails
    """
    required_params = {
        'classifier_name': str,
        'retain_graph': bool,
        'n_timesteps': int,
        'dt': (int, float),
        'tau': (int, float),
        't_recurrence': (int, float),
        't_feedforward': (int, float),
        'recurrence_influence': str
    }

    for param, param_type in required_params.items():
        if param not in config:
            raise MissingParameterError(f"Missing required parameter: {param}")
        validate_type(config[param], param_type, param)

    # Validate numeric ranges
    validate_numeric_range(config['n_timesteps'], 1, None, 'n_timesteps')
    validate_numeric_range(config['dt'], 0, None, 'dt')
    validate_numeric_range(config['tau'], 0, None, 'tau')
    validate_numeric_range(config['t_recurrence'], 0, None, 't_recurrence')
    validate_numeric_range(config['t_feedforward'], 0, None, 't_feedforward')

    # Validate enum values
    if config['recurrence_influence'] not in ['additive', 'multiplicative']:
        raise InvalidValueError(
            "recurrence_influence must be 'additive' or 'multiplicative'"
        )

def validate_data_config(config: Dict[str, Any]) -> None:
    """Validate data configuration parameters.

    Args:
        config: Data configuration dictionary

    Raises:
        ConfigurationError: If validation fails
    """
    required_params = {
        'mounted_datasets': list,
        'data_resolution': dict,
        'data_groups': dict
    }

    for param, param_type in required_params.items():
        if param not in config:
            raise MissingParameterError(f"Missing required parameter: {param}")
        validate_type(config[param], param_type, param)

    # Validate data resolutions
    for dataset, resolution in config['data_resolution'].items():
        validate_type(resolution, int, f"data_resolution.{dataset}")
        validate_numeric_range(resolution, 1, None, f"data_resolution.{dataset}")

def validate_workflow_config(config: Dict[str, Any]) -> None:
    """Validate workflow configuration parameters.

    Args:
        config: Workflow configuration dictionary

    Raises:
        ConfigurationError: If validation fails
    """
    required_params = {
        'model_name': str,
        'seed': str,
        'model_args': dict,
        'experiment': list,
        'loss': list,
        'optimizer': str,
        'learning_rate': (int, float),
        'epochs': int,
        'batch_size': int
    }

    for param, param_type in required_params.items():
        if param not in config:
            raise MissingParameterError(f"Missing required parameter: {param}")
        validate_type(config[param], param_type, param)

    # Validate numeric ranges
    validate_numeric_range(config['learning_rate'], 0, None, 'learning_rate')
    validate_numeric_range(config['epochs'], 1, None, 'epochs')
    validate_numeric_range(config['batch_size'], 1, None, 'batch_size')

def validate_experiment_config(config: Dict[str, Any]) -> None:
    """Validate experiment configuration parameters.

    Args:
        config: Experiment configuration dictionary

    Raises:
        ConfigurationError: If validation fails
    """
    if 'experiment_config' not in config:
        raise MissingParameterError("Missing experiment_config section")

    for exp_name, exp_config in config['experiment_config'].items():
        required_params = {
            'parameter': str,
            'data_loader': str,
            'data_args': dict
        }

        for param, param_type in required_params.items():
            if param not in exp_config:
                raise MissingParameterError(
                    f"Missing parameter '{param}' in experiment '{exp_name}'"
                )
            validate_type(exp_config[param], param_type, f"{exp_name}.{param}")

def validate_configuration(config: Dict[str, Any]) -> None:
    """Validate the complete configuration.

    Args:
        config: Complete configuration dictionary

    Raises:
        ConfigurationError: If validation fails
    """
    try:
        validate_model_config(config)
        validate_data_config(config)
        validate_workflow_config(config)
        validate_experiment_config(config)
        logger.info("Configuration validation successful")
    except ConfigurationError as e:
        logger.error(f"Configuration validation failed: {str(e)}")
        raise

def load_and_validate_config(config_dir: Union[str, Path]) -> Dict[str, Any]:
    """Load and validate configuration from files.

    Args:
        config_dir: Directory containing configuration files

    Returns:
        Dict[str, Any]: Validated configuration dictionary

    Raises:
        ConfigurationError: If validation fails
    """
    config_dir = Path(config_dir)
    config_files = [
        'config_defaults.yaml',
        'config_data.yaml',
        'config_workflow.yaml',
        'config_experiments.yaml'
    ]

    config = {}
    for filename in config_files:
        filepath = config_dir / filename
        if not filepath.exists():
            logger.warning(f"Configuration file not found: {filepath}")
            continue

        with open(filepath) as f:
            file_config = yaml.safe_load(f)
            if file_config:
                config.update(file_config)

    validate_configuration(config)
    return config

if __name__ == '__main__':
    import sys
    if len(sys.argv) != 2:
        print("Usage: python validate_config.py <config_dir>")
        sys.exit(1)

    try:
        config = load_and_validate_config(sys.argv[1])
        print("Configuration validation successful")
        sys.exit(0)
    except ConfigurationError as e:
        print(f"Configuration validation failed: {str(e)}")
        sys.exit(1)