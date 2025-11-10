"""
Streamlined configuration handling for DynVision workflow.

Handles mode application, wildcard integration, and per-rule config generation
with automatic file management and timestamping.
"""

import logging
import hashlib
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime
import yaml
from types import SimpleNamespace

logger = logging.getLogger(__name__)


class ConfigHandler:
    """Handles configuration processing, mode application, and per-rule config generation."""

    def __init__(self, base_config: Any, project_paths: Any, local: bool = True):
        """Initialize the config handler with base configuration.

        Args:
            base_config: Base configuration (dict or SimpleNamespace)
            project_paths: Project paths object for file locations
            local: Whether running locally (affects mode detection)
        """
        if isinstance(base_config, dict):
            self.base_config = base_config.copy()
        elif hasattr(base_config, "__dict__"):
            self.base_config = vars(base_config).copy()
        else:
            raise TypeError(
                "base_config must be a dictionary or an object with __dict__."
            )

        self.project_paths = project_paths
        self.local = local
        self._detect_available_modes()

        # Create config directory
        self.config_dir = self.project_paths.large_logs / "configs"
        self.config_dir.mkdir(parents=True, exist_ok=True)

        logger.info("ConfigHandler initialized")

    def _detect_available_modes(self) -> None:
        """Detect modes with namespace `use_<mode>_mode` in config."""
        use_mode_keys = [
            key
            for key in self.base_config.keys()
            if key.startswith("use_") and key.endswith("_mode")
        ]
        self.available_modes = {
            k.replace("use_", "").replace("_mode", ""): self.base_config[k]
            for k in use_mode_keys
        }

    def _apply_modes(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply active modes to configuration.

        Args:
            config: Configuration dictionary to modify

        Returns:
            Modified configuration with modes applied
        """
        active_modes = []

        for mode_name in self.available_modes.keys():
            mode_setting = config.get(f"use_{mode_name}_mode", False)

            if mode_setting == "auto":
                if hasattr(self, f"_detect_{mode_name}_mode"):
                    active = getattr(self, f"_detect_{mode_name}_mode")()
                else:
                    active = False
            else:
                active = bool(mode_setting)

            if active:
                self._apply_mode(config, mode_name)
                active_modes.append(mode_name)

        # Store active modes in config for reference
        config["_active_modes"] = active_modes
        return config

    def _apply_mode(self, config: Dict[str, Any], mode_name: str) -> None:
        """Apply a specific mode to the configuration.

        Args:
            config: Configuration dictionary to modify
            mode_name: Name of the mode to apply
        """
        if mode_name in config:
            mode_config = config[mode_name]
            if isinstance(mode_config, dict):
                config.update(mode_config)
                logger.debug(
                    f"Applied mode '{mode_name}' with {len(mode_config)} settings"
                )
            else:
                logger.warning(f"Mode '{mode_name}' configuration is not a dictionary")
        else:
            logger.debug(f"Mode '{mode_name}' is active but not defined in config")

    def _add_wildcards_to_config(
        self, config: Dict[str, Any], wildcards: Any
    ) -> Dict[str, Any]:
        """Add wildcard variables to configuration.

        Args:
            config: Configuration dictionary to modify
            wildcards: Snakemake wildcards object

        Returns:
            Configuration with wildcards added
        """
        if wildcards is None:
            return config

        # Add wildcards section
        config["wildcards"] = {}

        # Extract wildcard attributes
        for attr_name in dir(wildcards):
            if not attr_name.startswith("_"):
                try:
                    attr_value = getattr(wildcards, attr_name)
                    # Only add simple types (str, int, float)
                    if isinstance(attr_value, (str, int, float)):
                        config["wildcards"][attr_name] = attr_value
                        # Also add to top level (potentially overwriting existing)
                        config[attr_name] = attr_value
                except AttributeError:
                    continue

        logger.debug(f"Added {len(config['wildcards'])} wildcards to config")
        return config

    def _generate_unique_filename(self, wildcards: Any = None, rule_name=None) -> str:
        """Generate a unique filename for the config file.

        Args:
            wildcards: Snakemake wildcards for filename uniqueness

        Returns:
            Unique filename string
        """
        timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

        if wildcards is not None:
            # Create a hash from wildcards for uniqueness
            wildcard_str = "_" + "_".join(wildcards)
        else:
            wildcard_str = "_"

        if rule_name:
            rule_name_str = f"_{rule_name}"
        else:
            rule_name_str = "_"

        return f"config_{timestamp}{rule_name_str}{wildcard_str}.yaml"

    def process_configs(self, config: Any, wildcards: Any = None) -> Path:
        """Process configuration with modes and wildcards, write to file.

        Args:
            config: Current snakemake config (includes file + CLI settings)
            wildcards: Snakemake wildcards object

        Returns:
            Path to the generated config file
        """
        # Convert config to dict if needed
        if isinstance(config, SimpleNamespace):
            working_config = vars(config).copy()
        elif isinstance(config, dict):
            working_config = config.copy()
        else:
            working_config = dict(config)

        # Add wildcards
        working_config = self._add_wildcards_to_config(working_config, wildcards)

        # Apply modes
        working_config = self._apply_modes(working_config)

        # Add metadata
        working_config["_metadata"] = {
            "generated_at": datetime.now().isoformat(),
            "local_execution": self.local,
            "wildcard_count": len(working_config.get("wildcards", {})),
        }

        # Generate unique filename and write config
        filename = self._generate_unique_filename(wildcards)
        config_path = (
            self.config_dir / filename
        )  # place not restricted by number of files

        with open(config_path, "w") as f:
            f.write("# Automatically generated configuration file\n")
            f.write(f"# Generated at: {working_config['_metadata']['generated_at']}\n")
            if wildcards:
                f.write(f"# Wildcards: {dict(vars(wildcards))}\n")
            f.write("\n")
            yaml.dump(working_config, f, default_flow_style=False, sort_keys=False)

        logger.debug(f"Generated config file: {config_path}")
        return config_path

    # Mode detection methods
    def _detect_local_mode(self) -> bool:
        return self.local

    def _detect_debug_mode(self) -> bool:
        """Auto-detect if debug mode should be enabled."""
        log_level = self.base_config.get("log_level", "INFO").upper()
        epochs = self.base_config.get("epochs", 100)
        return log_level == "DEBUG" or epochs <= 5

    def _detect_large_dataset_mode(self) -> bool:
        """Auto-detect if large dataset optimizations should be enabled."""
        data_name = self.base_config.get("data_name", "").lower()
        large_datasets = ["imagenet", "coco", "openimages"]
        return any(large_name in data_name for large_name in large_datasets)

    def _detect_distributed_mode(self) -> bool:
        """Auto-detect if distributed training should be enabled."""
        # Add logic based on your cluster detection
        return not self.local and self.base_config.get("use_distributed", False)

    def log_available_modes(self) -> None:
        """Log available modes and their status."""
        if not self.available_modes:
            logger.info("No configuration modes available.")
            return

        max_mode_length = max(
            len(mode_name) for mode_name in self.available_modes.keys()
        )

        logger.info("=" * 40)
        logger.info(" Configuration Modes ".center(40, "="))
        logger.info("=" * 40)
        for mode_name, initial_status in self.available_modes.items():
            logger.info(f"\t {mode_name:<{max_mode_length}} | {str(initial_status)}")
