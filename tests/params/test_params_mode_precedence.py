"""
Tests for mode-specific parameter precedence.

These tests verify that the mode-scoped parameter system works correctly
with the proper precedence hierarchy:
1. CLI arguments (highest)
2. mode.component.param
3. mode.param
4. component.param
5. param (base defaults, lowest)

Run with: pytest tests/params/test_params_mode_precedence.py -v
"""

import pytest
import tempfile
import yaml
from pathlib import Path
from typing import Dict, Any

from dynvision.params import InitParams, TrainingParams, TestingParams


def create_temp_config(config_data: Dict[str, Any]) -> Path:
    """Helper to create a temporary config file."""
    # Add required base parameters if not present
    defaults = {
        # Base params
        "seed": 0,
        "log_level": "info",
        # Data params
        "data_name": "mnist",
        "data_group": "all",
        "train": True,
        "use_ffcv": True,
        "batch_size": 256,
        "num_workers": 4,
        "persistent_workers": True,
        "drop_last": True,
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
        "shuffle": True,
        "prefetch_factor": 1,
        # Trainer params
        "epochs": 200,
        "check_val_every_n_epoch": 10,
        "log_every_n_steps": 20,
        "num_sanity_val_steps": 0,
        "accumulate_grad_batches": 4,
        "precision": "bf16-mixed",
        "deterministic": False,
        "devices": 1,
        "num_nodes": 1,
        "accelerator": "auto",
        "enable_progress_bar": False,
        "benchmark": True,
        "gradient_clip_algorithm": "norm",
        "limit_val_batches": 0.2,
        "reload_dataloaders_every_n_epochs": 0,
        "early_stopping_min_delta": 0.0,
        "early_stopping_monitor": "val_loss",
        "early_stopping_mode": "min",
        "save_top_k": 2,
        "monitor_checkpoint": "val_loss",
        "checkpoint_mode": "min",
        "save_last": True,
        "every_n_epochs": 50,
    }
    # Merge with provided config (provided values override defaults)
    full_config = {**defaults, **config_data}

    temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    yaml.dump(full_config, temp_file)
    temp_file.close()
    return Path(temp_file.name)


class TestInitModeOverrides:
    """Test init mode-specific parameter overrides."""

    def test_init_mode_data_override(self):
        """Test that init.data.* overrides work correctly."""
        config_data = {
            "use_ffcv": True,  # Level 5: base
            "data.use_ffcv": True,  # Level 4: component
            "init.use_ffcv": False,  # Level 3: mode (should win for init)
        }

        config_file = create_temp_config(config_data)
        try:
            params = InitParams.from_cli_and_config(
                config_path=str(config_file),
                override_kwargs={
                    "dataset": "/tmp/test_dataset",
                    "output": "/tmp/test_output.pt",
                    "init_with_pretrained": False,
                },
            )

            assert (
                params.data.use_ffcv is False
            ), "init.use_ffcv should override base and component"
        finally:
            config_file.unlink()

    def test_init_mode_component_specific(self):
        """Test that init.data.* is more specific than init.*."""
        config_data = {
            "num_workers": 8,  # Level 5: base
            "init.num_workers": 4,  # Level 3: mode
            "init.data.num_workers": 0,  # Level 2: mode+component (should win)
        }

        config_file = create_temp_config(config_data)
        try:
            params = InitParams.from_cli_and_config(
                config_path=str(config_file),
                override_kwargs={
                    "dataset": "/tmp/test_dataset",
                    "output": "/tmp/test_output.pt",
                    "init_with_pretrained": False,
                },
            )

            assert (
                params.data.num_workers == 0
            ), "init.data.num_workers should be most specific"
        finally:
            config_file.unlink()


class TestTrainingModeOverrides:
    """Test training mode-specific parameter overrides."""

    def test_train_mode_data_override(self):
        """Test that train.data.* overrides work correctly."""
        config_data = {
            "shuffle": False,  # Level 5: base
            "data.shuffle": False,  # Level 4: component
            "train.shuffle": True,  # Level 3: mode (should win for training)
        }

        config_file = create_temp_config(config_data)
        try:
            params = TrainingParams.from_cli_and_config(
                config_path=str(config_file),
                override_kwargs={
                    "input_model_state": "/tmp/input.pt",
                    "output_model_state": "/tmp/output.pt",
                },
            )

            assert params.data.shuffle is True, "train.shuffle should override base"
        finally:
            config_file.unlink()


class TestTestingModeOverrides:
    """Test testing mode-specific parameter overrides."""

    def test_test_mode_batch_size_precedence(self):
        """Test full precedence hierarchy with batch_size."""
        config_data = {
            "batch_size": 128,  # Level 5: base
            "data.batch_size": 256,  # Level 4: component
            "test.batch_size": 512,  # Level 3: mode
            "test.data.batch_size": 1024,  # Level 2: mode+component (should win)
        }

        config_file = create_temp_config(config_data)
        try:
            params = TestingParams.from_cli_and_config(
                config_path=str(config_file),
                override_kwargs={
                    "input_model_state": "/tmp/input.pt",
                    "dataset": "/tmp/dataset",
                    "output_results": "/tmp/results.csv",
                    "output_responses": "/tmp/responses.pt",
                    "verbose": False,
                },
            )

            assert (
                params.data.batch_size == 1024
            ), "test.data.batch_size should have highest priority"
        finally:
            config_file.unlink()

    def test_test_mode_trainer_devices(self):
        """Test mode+component override for trainer."""
        config_data = {
            "devices": 8,  # Level 5: base
            "trainer.devices": 4,  # Level 4: component
            "test.devices": 2,  # Level 3: mode
            "test.trainer.devices": 1,  # Level 2: mode+component (should win)
        }

        config_file = create_temp_config(config_data)
        try:
            params = TestingParams.from_cli_and_config(
                config_path=str(config_file),
                override_kwargs={
                    "input_model_state": "/tmp/input.pt",
                    "dataset": "/tmp/dataset",
                    "output_results": "/tmp/results.csv",
                    "output_responses": "/tmp/responses.pt",
                    "verbose": False,
                },
            )

            assert (
                params.trainer.devices == 1
            ), "test.trainer.devices should be most specific"
        finally:
            config_file.unlink()

    def test_test_mode_use_ffcv_override(self):
        """Test that test.data.use_ffcv overrides base."""
        config_data = {
            "use_ffcv": True,  # Level 5: base (should win for base)
            "test.data.use_ffcv": False,  # Level 2: mode+component (should win for test)
        }

        config_file = create_temp_config(config_data)
        try:
            params = TestingParams.from_cli_and_config(
                config_path=str(config_file),
                override_kwargs={
                    "input_model_state": "/tmp/input.pt",
                    "dataset": "/tmp/dataset",
                    "output_results": "/tmp/results.csv",
                    "output_responses": "/tmp/responses.pt",
                    "verbose": False,
                },
            )

            assert (
                params.data.use_ffcv is False
            ), "test.data.use_ffcv should override base"
        finally:
            config_file.unlink()


class TestWrongModeIgnored:
    """Test that parameters for other modes are ignored."""

    def test_train_params_ignored_by_init(self):
        """Test that train.* parameters are ignored by InitParams."""
        config_data = {
            "use_ffcv": True,  # Level 5: base (should win for init)
            "train.use_ffcv": False,  # Wrong mode (ignored by init)
        }

        config_file = create_temp_config(config_data)
        try:
            params = InitParams.from_cli_and_config(
                config_path=str(config_file),
                override_kwargs={
                    "dataset": "/tmp/test_dataset",
                    "output": "/tmp/test_output.pt",
                    "init_with_pretrained": False,
                },
            )

            # train.use_ffcv should be ignored since mode is "init"
            assert (
                params.data.use_ffcv is True
            ), "Wrong mode override should be ignored"
        finally:
            config_file.unlink()

    def test_init_params_ignored_by_test(self):
        """Test that init.* parameters are ignored by TestingParams."""
        config_data = {
            "num_workers": 8,  # Level 5: base (should win for test)
            "init.num_workers": 0,  # Wrong mode (ignored by test)
        }

        config_file = create_temp_config(config_data)
        try:
            params = TestingParams.from_cli_and_config(
                config_path=str(config_file),
                override_kwargs={
                    "input_model_state": "/tmp/input.pt",
                    "dataset": "/tmp/dataset",
                    "output_results": "/tmp/results.csv",
                    "output_responses": "/tmp/responses.pt",
                    "verbose": False,
                },
            )

            # init.num_workers should be ignored since mode is "test"
            assert (
                params.data.num_workers == 8
            ), "Wrong mode override should be ignored"
        finally:
            config_file.unlink()


class TestUnscopedParameterRouting:
    """Test that unscoped parameters are still routed correctly."""

    def test_unscoped_model_param(self):
        """Test that unscoped model parameters route correctly."""
        config_data = {
            "n_classes": 100,  # Unscoped, should route to model
        }

        config_file = create_temp_config(config_data)
        try:
            params = TestingParams.from_cli_and_config(
                config_path=str(config_file),
                override_kwargs={
                    "input_model_state": "/tmp/input.pt",
                    "dataset": "/tmp/dataset",
                    "output_results": "/tmp/results.csv",
                    "output_responses": "/tmp/responses.pt",
                    "verbose": False,
                },
            )

            assert (
                params.model.n_classes == 100
            ), "Unscoped params should still route to components"
        finally:
            config_file.unlink()


class TestCLIOverride:
    """Test that CLI overrides work at all scope levels."""

    def test_cli_overrides_mode_component(self):
        """Test that CLI args override even mode+component config."""
        config_data = {
            "batch_size": 128,  # Level 5: base
            "test.data.batch_size": 1024,  # Level 2: mode+component
        }

        config_file = create_temp_config(config_data)
        try:
            # CLI override should win
            params = TestingParams.from_cli_and_config(
                config_path=str(config_file),
                override_kwargs={
                    "input_model_state": "/tmp/input.pt",
                    "dataset": "/tmp/dataset",
                    "output_results": "/tmp/results.csv",
                    "output_responses": "/tmp/responses.pt",
                    "verbose": False,
                    "test.data.batch_size": 2048,  # CLI override (Level 1)
                },
            )

            assert (
                params.data.batch_size == 2048
            ), "CLI should override all config levels"
        finally:
            config_file.unlink()


class TestBackwardCompatibility:
    """Test that existing unscoped parameters still work."""

    def test_unscoped_params_work(self):
        """Test that traditional unscoped parameters continue to work."""
        config_data = {
            "batch_size": 256,
            "shuffle": False,
            "num_workers": 4,
        }

        config_file = create_temp_config(config_data)
        try:
            params = TestingParams.from_cli_and_config(
                config_path=str(config_file),
                override_kwargs={
                    "input_model_state": "/tmp/input.pt",
                    "dataset": "/tmp/dataset",
                    "output_results": "/tmp/results.csv",
                    "output_responses": "/tmp/responses.pt",
                    "verbose": False,
                },
            )

            assert params.data.batch_size == 256, "Unscoped batch_size should work"
            assert params.data.shuffle is False, "Unscoped shuffle should work"
            assert params.data.num_workers == 4, "Unscoped num_workers should work"
        finally:
            config_file.unlink()


if __name__ == "__main__":
    # Run with pytest
    pytest.main([__file__, "-v"])
