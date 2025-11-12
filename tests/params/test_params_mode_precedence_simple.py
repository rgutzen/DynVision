"""
Simple tests for mode-specific parameter precedence.

These tests verify the core mode-scoped parameter system without
requiring file system validation. Focuses on basic precedence rules
without complex end-to-end flows.

Run with: pytest tests/params/test_params_mode_precedence_simple.py -v
"""

import pytest
import tempfile
import yaml
from pathlib import Path
from typing import Dict, Any

from dynvision.params import InitParams


def create_test_config(overrides: Dict[str, Any]) -> Path:
    """Create a minimal test config with overrides."""
    config = {
        # Base required params
        "seed": 0,
        "log_level": "info",
        # Data params
        "data_name": "mnist",
        "data_group": "all",
        "train": False,
        "use_ffcv": False,
        "batch_size": 64,
        "num_workers": 0,
        "persistent_workers": False,
        "drop_last": False,
        "train_ratio": 0.9,
        "pixel_range": "0-1",
        "data_timesteps": 1,
        "non_input_value": 0,
        "non_label_index": -1,
        "use_distributed": False,
        "encoding": "image",
        "writer_mode": "proportion",
        "max_resolution": 224,
        "compress_probability": 0.25,
        "jpeg_quality": 60,
        "chunksize": 1000,
        "page_size": 4194304,
        "batches_ahead": 3,
        "order": "QUASI_RANDOM",
        "pin_memory": False,
        "shuffle": False,
        "prefetch_factor": 1,
        **overrides,
    }

    temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    yaml.dump(config, temp_file)
    temp_file.close()
    return Path(temp_file.name)


def test_init_mode_base_override():
    """Test that init mode overrides base value."""
    config_file = create_test_config(
        {
            "use_ffcv": True,  # Level 5: base
            "init.use_ffcv": False,  # Level 3: mode (should win)
        }
    )

    try:
        params = InitParams.from_cli_and_config(
            config_path=str(config_file),
            override_kwargs={
                "output": "/tmp/test_output.pt",
                "init_with_pretrained": False,
            },
        )

        assert params.data.use_ffcv is False, "init.use_ffcv should override base"
    finally:
        config_file.unlink()


def test_init_mode_component_override():
    """Test that init.data.* overrides init.*."""
    config_file = create_test_config(
        {
            "num_workers": 8,  # Level 5: base
            "init.num_workers": 4,  # Level 3: mode
            "init.data.num_workers": 0,  # Level 2: mode+component (should win)
        }
    )

    try:
        params = InitParams.from_cli_and_config(
            config_path=str(config_file),
            override_kwargs={
                "output": "/tmp/test_output.pt",
                "init_with_pretrained": False,
            },
        )

        assert (
            params.data.num_workers == 0
        ), "init.data.num_workers should be most specific"
    finally:
        config_file.unlink()


def test_component_scoped_override():
    """Test that data.* overrides base."""
    config_file = create_test_config(
        {
            "batch_size": 128,  # Level 5: base
            "data.batch_size": 256,  # Level 4: component (should win)
        }
    )

    try:
        params = InitParams.from_cli_and_config(
            config_path=str(config_file),
            override_kwargs={
                "output": "/tmp/test_output.pt",
                "init_with_pretrained": False,
            },
        )

        assert params.data.batch_size == 256, "data.batch_size should override base"
    finally:
        config_file.unlink()


def test_cli_override():
    """Test that CLI overrides all config levels."""
    config_file = create_test_config(
        {
            "batch_size": 128,  # Level 5: base
            "init.data.batch_size": 256,  # Level 2: mode+component
        }
    )

    try:
        params = InitParams.from_cli_and_config(
            config_path=str(config_file),
            override_kwargs={
                "output": "/tmp/test_output.pt",
                "init_with_pretrained": False,
                "init.data.batch_size": 512,  # CLI override (Level 1)
            },
        )

        assert params.data.batch_size == 512, "CLI should override all config levels"
    finally:
        config_file.unlink()


def test_wrong_mode_ignored():
    """Test that parameters for other modes are ignored."""
    config_file = create_test_config(
        {
            "use_ffcv": False,  # Level 5: base (should win for init)
            "train.use_ffcv": True,  # Wrong mode (ignored by init)
            "test.use_ffcv": True,  # Wrong mode (ignored by init)
        }
    )

    try:
        params = InitParams.from_cli_and_config(
            config_path=str(config_file),
            override_kwargs={
                "output": "/tmp/test_output.pt",
                "init_with_pretrained": False,
            },
        )

        assert params.data.use_ffcv is False, "Wrong mode overrides should be ignored"
    finally:
        config_file.unlink()


def test_full_precedence_chain():
    """Test complete precedence hierarchy."""
    config_file = create_test_config(
        {
            "pin_memory": True,  # Level 5: base
            "data.pin_memory": True,  # Level 4: component
            "init.pin_memory": True,  # Level 3: mode
            "init.data.pin_memory": False,  # Level 2: mode+component (should win)
        }
    )

    try:
        params = InitParams.from_cli_and_config(
            config_path=str(config_file),
            override_kwargs={
                "output": "/tmp/test_output.pt",
                "init_with_pretrained": False,
            },
        )

        assert (
            params.data.pin_memory is False
        ), "init.data.pin_memory should have highest config priority"
    finally:
        config_file.unlink()


def test_backward_compatibility():
    """Test that unscoped parameters still work."""
    config_file = create_test_config(
        {
            "batch_size": 256,
            "shuffle": False,
            "num_workers": 4,
        }
    )

    try:
        params = InitParams.from_cli_and_config(
            config_path=str(config_file),
            override_kwargs={
                "output": "/tmp/test_output.pt",
                "init_with_pretrained": False,
            },
        )

        assert params.data.batch_size == 256, "Unscoped batch_size should work"
        assert params.data.shuffle is False, "Unscoped shuffle should work"
        assert params.data.num_workers == 4, "Unscoped num_workers should work"
    finally:
        config_file.unlink()


if __name__ == "__main__":
    # Run with pytest or directly
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "-v":
        pytest.main([__file__, "-v"])
    else:
        # Run tests directly
        print("Running mode-specific parameter tests...")
        test_init_mode_base_override()
        print("✓ test_init_mode_base_override")
        test_init_mode_component_override()
        print("✓ test_init_mode_component_override")
        test_component_scoped_override()
        print("✓ test_component_scoped_override")
        test_cli_override()
        print("✓ test_cli_override")
        test_wrong_mode_ignored()
        print("✓ test_wrong_mode_ignored")
        test_full_precedence_chain()
        print("✓ test_full_precedence_chain")
        test_backward_compatibility()
        print("✓ test_backward_compatibility")
        print("\nAll tests passed! ✓")
