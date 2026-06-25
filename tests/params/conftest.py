"""
Pytest configuration and shared fixtures for parameter tests.

This file provides common fixtures and utilities used across
all parameter-related test modules.
"""

import pytest
import tempfile
import yaml
from pathlib import Path
from typing import Dict, Any


@pytest.fixture
def temp_config_file():
    """
    Fixture that creates and cleans up temporary config files.

    Usage:
        def test_example(temp_config_file):
            config_path = temp_config_file({"seed": 42, "log_level": "info"})
            # ... use config_path ...
            # Cleanup happens automatically
    """
    created_files = []

    def _create_config(config_data: Dict[str, Any]) -> Path:
        """Create a temporary config file with the given data."""
        temp_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        )
        yaml.dump(config_data, temp_file, default_flow_style=False)
        temp_file.close()
        config_path = Path(temp_file.name)
        created_files.append(config_path)
        return config_path

    yield _create_config

    # Cleanup all created files
    for file_path in created_files:
        if file_path.exists():
            file_path.unlink()


@pytest.fixture
def temp_model_files():
    """
    Fixture that creates temporary model-related files for testing.

    Returns a dict with paths to:
    - model_state: .pth file
    - dataset: directory
    - results: .json file
    - responses: .h5 file

    All files are automatically cleaned up after the test.
    """
    # Create temporary files
    model_file = tempfile.NamedTemporaryFile(suffix=".pth", delete=False)
    model_file.close()

    dataset_dir = tempfile.mkdtemp()

    results_file = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
    results_file.close()

    responses_file = tempfile.NamedTemporaryFile(suffix=".h5", delete=False)
    responses_file.close()

    paths = {
        "model_state": Path(model_file.name),
        "dataset": Path(dataset_dir),
        "results": Path(results_file.name),
        "responses": Path(responses_file.name),
    }

    yield paths

    # Cleanup
    import shutil

    for key, path in paths.items():
        if path.exists():
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()


@pytest.fixture
def base_config_defaults():
    """
    Fixture that loads the real config_defaults.yaml file.

    Returns the parsed YAML as a dictionary.
    """
    defaults_path = (
        Path(__file__).parents[2] / "dynvision" / "configs" / "config_defaults.yaml"
    )
    with open(defaults_path, "r") as f:
        return yaml.safe_load(f)
