"""
Tests for mode-specific config merging and conflict resolution.

These tests verify the complete flow of mode-specific configuration:
1. Deep merging of mode overrides (test.data.train → data.train)
2. Conflict resolution (removing base-level keys that conflict with scoped keys)
3. Component section flattening (nested dict → dotted notation)
4. Preservation of all required fields through the entire pipeline

These tests specifically cover the bugs discovered and fixed in the mode config system.

Run with: pytest tests/params/test_params_mode_config_merging.py -v
"""

import pytest
import tempfile
import yaml
from pathlib import Path
from typing import Dict, Any
from unittest.mock import patch

from dynvision.params import TestingParams
from dynvision.params.composite_params import CompositeParams


def create_temp_config(config_data: Dict[str, Any]) -> Path:
    """Helper to create a temporary config file."""
    temp_file = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8"
    )
    yaml.dump(config_data, temp_file, default_flow_style=False)
    temp_file.close()
    return Path(temp_file.name)


class TestModeConfigMerging:
    """Test mode-specific config merging with deep_merge."""

    def test_deep_merge_simple_override(self):
        """Test that mode overrides are merged into base config."""
        base = {"data": {"train": True, "batch_size": 64}}
        override = {"data": {"train": False}}

        CompositeParams._deep_merge(base, override)

        assert base["data"]["train"] is False  # Overridden
        assert base["data"]["batch_size"] == 64  # Preserved

    def test_deep_merge_nested_sections(self):
        """Test deep merge with multiple nested sections."""
        base = {
            "data": {"train": True, "shuffle": True},
            "trainer": {"devices": 1, "max_epochs": 10},
        }
        override = {
            "data": {"train": False, "sampler": "RoundRobinSampler"},
            "trainer": {"devices": 2},
        }

        CompositeParams._deep_merge(base, override)

        # Data section
        assert base["data"]["train"] is False  # Overridden
        assert base["data"]["shuffle"] is True  # Preserved
        assert base["data"]["sampler"] == "RoundRobinSampler"  # Added

        # Trainer section
        assert base["trainer"]["devices"] == 2  # Overridden
        assert base["trainer"]["max_epochs"] == 10  # Preserved

    def test_deep_merge_creates_new_sections(self):
        """Test that deep merge creates new sections if they don't exist."""
        base = {"seed": 42}
        override = {"data": {"train": False}, "trainer": {"devices": 1}}

        CompositeParams._deep_merge(base, override)

        assert "data" in base
        assert base["data"]["train"] is False
        assert "trainer" in base
        assert base["trainer"]["devices"] == 1
        assert base["seed"] == 42  # Original preserved


class TestConflictResolution:
    """Test conflict resolution between base-level and scoped keys."""

    def test_remove_conflicting_base_keys_simple(self):
        """Test that base-level 'train' is removed when data.train exists."""
        config = {
            "train": True,  # Base-level
            "data": {"train": False},  # Scoped (from mode override)
        }
        mode_overrides = {"data": {"train": False}}

        CompositeParams._remove_conflicting_base_keys(config, mode_overrides)

        assert "train" not in config  # Base-level removed
        assert config["data"]["train"] is False  # Scoped preserved

    def test_remove_conflicting_preserves_non_conflicting(self):
        """Test that non-conflicting base-level keys are preserved."""
        config = {
            "seed": 42,  # Non-conflicting
            "train": True,  # Conflicts with data.train
            "data": {"train": False},
        }
        mode_overrides = {"data": {"train": False}}

        CompositeParams._remove_conflicting_base_keys(config, mode_overrides)

        assert config["seed"] == 42  # Preserved
        assert "train" not in config  # Removed
        assert config["data"]["train"] is False  # Scoped preserved

    def test_remove_conflicting_multiple_components(self):
        """Test conflict resolution across multiple components."""
        config = {
            "train": True,  # Conflicts with data.train
            "devices": 1,  # Conflicts with trainer.devices
            "num_nodes": 1,  # Conflicts with trainer.num_nodes
            "accelerator": "auto",  # Conflicts with trainer.accelerator
            "seed": 42,  # Non-conflicting
            "data": {"train": False, "sampler": "RoundRobinSampler"},
            "trainer": {"devices": 1, "num_nodes": 1, "accelerator": "auto"},
        }
        mode_overrides = {
            "data": {"train": False, "sampler": "RoundRobinSampler"},
            "trainer": {"devices": 1, "num_nodes": 1, "accelerator": "auto"},
        }

        CompositeParams._remove_conflicting_base_keys(config, mode_overrides)

        # Conflicting base-level keys removed
        assert "train" not in config
        assert "devices" not in config
        assert "num_nodes" not in config
        assert "accelerator" not in config

        # Non-conflicting preserved
        assert config["seed"] == 42

        # Scoped values preserved
        assert config["data"]["train"] is False
        assert config["data"]["sampler"] == "RoundRobinSampler"
        assert config["trainer"]["devices"] == 1
        assert config["trainer"]["num_nodes"] == 1
        assert config["trainer"]["accelerator"] == "auto"

    def test_conflict_resolution_copies_base_as_fallback(self):
        """Test that base-level values are copied to scoped section if missing."""
        config = {
            "devices": 1,  # Base-level, will be copied to trainer.devices
            "trainer": {"max_epochs": 10},  # Scoped section exists but missing devices
        }
        mode_overrides = {
            "trainer": {
                "max_epochs": 10,
                "devices": 1,
            }  # Must include devices in override for it to be detected
        }

        CompositeParams._remove_conflicting_base_keys(config, mode_overrides)

        # Base-level key copied to scoped section as fallback (already there from mode override)
        assert config["trainer"]["devices"] == 1
        assert "devices" not in config  # Base-level removed

    def test_conflict_resolution_creates_scoped_section(self):
        """Test that scoped section is created if it doesn't exist."""
        config = {
            "devices": 1,  # Base-level
            "num_nodes": 1,
        }
        mode_overrides = {
            "trainer": {"devices": 2}  # Trainer section doesn't exist in config yet
        }

        CompositeParams._remove_conflicting_base_keys(config, mode_overrides)

        # Scoped section created
        assert "trainer" in config
        assert isinstance(config["trainer"], dict)
        # Base-level removed after copying
        assert "devices" not in config


class TestComponentSectionFlattening:
    """Test flattening of nested component sections to dotted notation."""

    def test_flatten_simple_component(self):
        """Test flattening a single component section."""
        # Mock the component classes
        with patch.object(
            CompositeParams,
            "get_component_classes",
            return_value={"data": object, "trainer": object, "model": object},
        ):
            config = {
                "seed": 42,  # Non-component
                "data": {"train": False, "batch_size": 64},
                "trainer": {"devices": 1, "max_epochs": 10},
            }

            flattened = CompositeParams._flatten_component_sections(config)

            # Non-component keys preserved
            assert flattened["seed"] == 42

            # Component sections flattened to dotted notation
            assert "data" not in flattened
            assert flattened["data.train"] is False
            assert flattened["data.batch_size"] == 64

            assert "trainer" not in flattened
            assert flattened["trainer.devices"] == 1
            assert flattened["trainer.max_epochs"] == 10

    def test_flatten_preserves_non_component_nested_dicts(self):
        """Test that non-component nested dicts are not flattened."""
        with patch.object(
            CompositeParams,
            "get_component_classes",
            return_value={"data": object, "trainer": object},
        ):
            config = {
                "data": {"train": False},  # Component - will be flattened
                "custom_config": {  # Not a component - preserved as-is
                    "nested": {"value": 123}
                },
            }

            flattened = CompositeParams._flatten_component_sections(config)

            # Component flattened
            assert flattened["data.train"] is False

            # Non-component preserved
            assert "custom_config" in flattened
            assert isinstance(flattened["custom_config"], dict)
            assert flattened["custom_config"]["nested"]["value"] == 123

    def test_flatten_all_required_trainer_fields(self):
        """Test that all required trainer fields are preserved after flattening."""
        with patch.object(
            CompositeParams,
            "get_component_classes",
            return_value={"trainer": object},
        ):
            config = {
                "trainer": {
                    "devices": 1,
                    "num_nodes": 1,
                    "accelerator": "auto",
                    "max_epochs": 10,
                    "logger": False,
                    "enable_progress_bar": True,
                }
            }

            flattened = CompositeParams._flatten_component_sections(config)

            # All fields present in dotted notation
            assert flattened["trainer.devices"] == 1
            assert flattened["trainer.num_nodes"] == 1
            assert flattened["trainer.accelerator"] == "auto"
            assert flattened["trainer.max_epochs"] == 10
            assert flattened["trainer.logger"] is False
            assert flattened["trainer.enable_progress_bar"] is True


class TestEndToEndModeConfigFlow:
    """Test the complete mode config flow from YAML to component instantiation."""

    def get_base_config_with_defaults(self) -> Dict[str, Any]:
        """Load the real defaults config and add minimal test params."""
        defaults_path = (
            Path(__file__).parents[2]
            / "dynvision"
            / "configs"
            / "config_defaults.yaml"
        )
        with open(defaults_path, "r") as f:
            config = yaml.safe_load(f)

        # Create temporary paths for testing (validators check these exist)
        test_model_file = tempfile.NamedTemporaryFile(suffix=".pth", delete=False)
        test_model_file.close()

        test_data_dir = tempfile.mkdtemp()
        test_results_file = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        test_results_file.close()
        test_responses_file = tempfile.NamedTemporaryFile(suffix=".h5", delete=False)
        test_responses_file.close()

        # Add minimal required params for test
        config.update(
            {
                "data_name": "imagenette",
                "data_group": "one",
                "model_name": "TestModel",
                "n_classes": 10,
                # TestingParams-specific required fields
                "input_model_state": test_model_file.name,
                "dataset_path": test_data_dir,
                "output_results": test_results_file.name,
                "output_responses": test_responses_file.name,
            }
        )
        return config

    def test_test_mode_overrides_data_train(self):
        """Test that test mode correctly overrides data.train to False."""
        config_data = self.get_base_config_with_defaults()

        # Override specific values for this test
        config_data["seed"] = 42
        config_data["train"] = True  # Base-level train
        config_data["batch_size"] = 64  # Test mode overrides
        if "test" not in config_data:
            config_data["test"] = {}
        if "data" not in config_data["test"]:
            config_data["test"]["data"] = {}

        config_data["test"]["data"].update(
            {
                "train": False,  # Override to test mode
                "sampler": "RoundRobinSampler",
                "shuffle": False,
            }
        )

        config_path = create_temp_config(config_data)

        try:
            # Load config through TestingParams (mode="test")
            params = TestingParams.from_cli_and_config(config_path=str(config_path))

            # Verify that test mode override was applied
            assert params.data.train is False  # NOT True from base config
            assert params.data.sampler == "RoundRobinSampler"
            assert params.data.shuffle is False

            # Verify other params preserved
            assert params.data.batch_size == 64
            assert params.seed == 42

        finally:
            config_path.unlink()

    def test_test_mode_preserves_trainer_required_fields(self):
        """Test that trainer required fields are not lost during mode config processing."""
        config_data = self.get_base_config_with_defaults()

        # Test mode overrides (doesn't override most trainer params)
        if "test" not in config_data:
            config_data["test"] = {}
        config_data["test"]["data"] = {"train": False}
        config_data["test"]["trainer"] = {"logger": None}  # Override only logger

        config_path = create_temp_config(config_data)

        try:
            # This should NOT raise validation errors for missing trainer fields
            params = TestingParams.from_cli_and_config(config_path=str(config_path))

            # Verify trainer fields are present (exact values from config_defaults.yaml)
            assert params.trainer.devices is not None
            assert params.trainer.num_nodes is not None
            assert params.trainer.accelerator is not None
            assert params.trainer.logger is None  # Overridden by test mode

        finally:
            config_path.unlink()

    def test_mode_overrides_dont_affect_other_components(self):
        """Test that mode overrides for one component don't affect others."""
        config_data = self.get_base_config_with_defaults()

        # Set base values
        config_data["train"] = True
        config_data["shuffle"] = True  # Base-level shuffle
        config_data["epochs"] = 100  # Base value for trainer

        # Test mode overrides (only data and trainer)
        if "test" not in config_data:
            config_data["test"] = {}
        config_data["test"]["data"] = {"train": False, "shuffle": False}
        config_data["test"]["trainer"] = {"epochs": 1}  # Override for testing

        config_path = create_temp_config(config_data)

        try:
            params = TestingParams.from_cli_and_config(config_path=str(config_path))

            # Data overrides applied
            assert params.data.train is False
            assert params.data.shuffle is False

            # Trainer overrides applied
            assert params.trainer.epochs == 1

            # Model NOT affected (no test mode overrides for model)
            assert params.model.model_name == "TestModel"
            assert params.model.n_classes == 10

        finally:
            config_path.unlink()


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_mode_overrides(self):
        """Test that empty mode override section is handled correctly."""
        config = {"seed": 42, "test": {}}
        mode_overrides = {}

        # Should not raise an error
        CompositeParams._deep_merge(config, mode_overrides)
        CompositeParams._remove_conflicting_base_keys(config, mode_overrides)

        assert config["seed"] == 42

    def test_mode_override_with_no_corresponding_base_key(self):
        """Test mode override for a key that doesn't exist at base level."""
        config = {"data": {"batch_size": 64}}
        mode_overrides = {"data": {"sampler": "RoundRobinSampler"}}

        CompositeParams._deep_merge(config, mode_overrides)
        CompositeParams._remove_conflicting_base_keys(config, mode_overrides)

        # New key added without issue
        assert config["data"]["sampler"] == "RoundRobinSampler"
        assert config["data"]["batch_size"] == 64

    def test_non_dict_mode_override_replaces_value(self):
        """Test that non-dict mode overrides completely replace base values."""
        base = {"data": {"train": True}}
        override = {"data": {"train": False}}

        CompositeParams._deep_merge(base, override)

        # Value replaced, not merged
        assert base["data"]["train"] is False

    def test_flatten_empty_component_section(self):
        """Test flattening with empty component sections."""
        with patch.object(
            CompositeParams,
            "get_component_classes",
            return_value={"data": object},
        ):
            config = {"data": {}, "seed": 42}

            flattened = CompositeParams._flatten_component_sections(config)

            # Empty component section results in no dotted keys
            assert "data" not in flattened
            assert "seed" in flattened
            assert not any(k.startswith("data.") for k in flattened.keys())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
