"""
Model initialization parameters for DynVision using Pydantic.

This module provides type-safe parameter management for model initialization,
including dataset dimension inference and pretrained model configuration.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import logging

from pydantic import Field, model_validator, ConfigDict

from dynvision.params.base_params import BaseParams, DynVisionValidationError
from dynvision.params.model_params import ModelParams
from dynvision.params.data_params import DataParams

logger = logging.getLogger(__name__)


class InitParams(BaseParams):
    """
    Model initialization parameters with automatic dimension inference.

    Combines ModelParams and DataParams with initialization-specific configuration
    for creating and saving initialized models.
    """

    # === CORE COMPONENT COMPOSITION ===
    model: ModelParams = Field(description="Model architecture parameters")
    data: DataParams = Field(
        description="Data loading parameters for dimension inference"
    )

    # === INITIALIZATION-SPECIFIC PARAMETERS ===
    dataset: Optional[Path] = Field(
        default=None, description="Path to dataset for dimension inference"
    )
    output: Path = Field(description="Path to save initialized model")
    seed: int = Field(default=42, description="Random seed for initialization")
    init_with_pretrained: bool = Field(
        default=False, description="Initialize with pretrained weights if available"
    )

    model_config = ConfigDict(
        extra="allow",  # Allow additional CLI arguments
        validate_assignment=True,
        use_enum_values=True,
        validate_by_name=True,
    )

    @classmethod
    def get_aliases(cls) -> Dict[str, str]:
        """Return mapping of aliases to full parameter names for all components."""
        aliases = super().get_aliases()

        # Add initialization-specific aliases
        aliases.update(
            {
                # Model aliases (routed to model component)
                "model_name": "model.model_name",
                "classes": "model.n_classes",
                "tsteps": "model.n_timesteps",
                "rctype": "model.recurrence_type",
                # Data aliases (routed to data component)
                "data_name": "data.data_name",
                "resolution": "data.resolution",
            }
        )

        return aliases

    @model_validator(mode="after")
    def validate_initialization_config(self) -> "InitParams":
        """Validate initialization-specific configuration."""
        # Ensure output directory exists
        if self.output:
            self.output.parent.mkdir(parents=True, exist_ok=True)
        return self

    @classmethod
    def validate_data_config(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        update_dict = {
            "use_ffcv": False,
            "use_distributed": False,
            "batch_size": 32,
            "num_workers": 0,
            "pin_memory": False,
        }
        config = cls.update_kwargs(config, update_dict, verbose=True)
        return config

    @classmethod
    def validate_model_config(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        update_dict = {
            "store_responses": 0,
        }
        config = cls.update_kwargs(config, update_dict, verbose=True)
        return config

    def update_model_parameters_from_dataset(
        self,
        input_dims: Tuple[int, ...],
        n_classes: int,
        verbose: bool = True,
    ) -> None:
        """
        Update model parameters based on dataset characteristics.

        Args:
            input_dims: Input dimensions from dataset (channels, height, width) or (timesteps, channels, height, width)
            n_classes: Number of classes in dataset
            verbose: Whether to log warnings for parameter changes
        """
        # Update input dimensions
        if self.model.input_dims != input_dims:
            self.model.update_field("input_dims", input_dims, verbose=verbose)

        # Update n_timesteps if provided in input_dims
        if len(input_dims) == 4:  # (timesteps, channels, height, width)
            n_timesteps = input_dims[0]
            if n_timesteps > 1 and self.model.n_timesteps != n_timesteps:
                self.model.update_field("n_timesteps", n_timesteps, verbose=verbose)

        # Update n_classes
        if self.model.n_classes != n_classes:
            self.model.update_field("n_classes", n_classes, verbose=verbose)

    def get_model_kwargs(self, model_class=None) -> Dict[str, Any]:
        return self.model.get_model_kwargs(model_class)

    def get_dataset_kwargs(self) -> Dict[str, Any]:
        return self.data.get_dataset_kwargs()

    def get_dataloader_kwargs(self) -> Dict[str, Any]:
        return self.data.get_dataloader_kwargs()

    @classmethod
    def from_cli_and_config(
        cls,
        config_path: Optional[Path] = None,
        override_kwargs: Optional[Dict[str, Any]] = None,
        args: Optional[list] = None,
    ) -> "InitParams":
        """
        Create InitParams instance from CLI and config with proper component separation.
        """
        # Get raw parameters using BaseParams method
        params = cls.get_params_from_cli_and_config(
            config_path=config_path,
            override_kwargs=override_kwargs,
            args=args,
        )

        # Separate into component configurations
        separated_params = cls._separate_component_configs(params)

        # Create the InitParams instance
        try:
            return cls(**separated_params)
        except Exception as e:
            raise DynVisionValidationError(f"InitParams creation failed: {e}")

    @classmethod
    def _separate_component_configs(cls, params: Dict[str, Any]) -> Dict[str, Any]:
        """Separate flat parameter dict into component configurations."""
        model_params = {}
        data_params = {}
        base_params = {}

        # Get field names for each component
        model_fields = set(ModelParams.model_fields.keys())
        data_fields = set(DataParams.model_fields.keys())
        base_fields = set(cls.model_fields.keys()) - {"model", "data"}

        for key, value in params.items():
            # Handle dotted notation (e.g., "model.learning_rate")
            if "." in key:
                component, field = key.split(".", 1)
                if component == "model":
                    model_params[field] = value
                elif component == "data":
                    data_params[field] = value
                else:
                    base_params[key] = value
            else:
                # Use mutually exclusive assignment logic
                if key in data_fields:
                    data_params[key] = value
                elif key in base_fields:
                    base_params[key] = value
                elif key in model_fields:
                    model_params[key] = value
                else:
                    # Unknown parameters go to model_params as fallback
                    model_params[key] = value
                    logger.debug(
                        f"Assigning unknown parameter '{key}' to model_params"
                    )

        # Validate component configurations
        cls.validate_data_config(data_params)
        cls.validate_model_config(model_params)

        # Create component instances
        try:
            components = {
                "model": ModelParams(**model_params),
                "data": DataParams(**data_params),
                **base_params,
            }
            return components
        except Exception as e:
            raise DynVisionValidationError(f"Component configuration failed: {e}")
