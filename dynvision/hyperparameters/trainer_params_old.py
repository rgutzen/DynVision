"""
Trainer parameter handling for DynVision using Pydantic.

This module provides type-safe parameter management for PyTorch Lightning Trainer
configuration, including training behavior, system performance, and callback settings.

Note: Optimizer and scheduler parameters belong in ModelParams since they are
configured in the model's configure_optimizers() method in PyTorch Lightning.
"""

from pydantic import Field, field_validator, model_validator, ConfigDict
from typing import Dict, Any, Optional, List, Union, Tuple
import logging
from pathlib import Path
import pytorch_lightning as pl
from dynvision.hyperparameters.base_params import BaseParams, DynVisionValidationError


class DistributedParams(BaseParams):
    """
    Distributed training parameters for PyTorch Lightning.

    Handles distributed training configuration including strategy selection,
    device allocation, and strategy-specific parameters.
    """

    # Core Distributed Parameters
    strategy: str = Field(
        default="ddp",
        description="Distributed strategy (ddp, ddp_spawn, fsdp, deepspeed, etc.)",
    )
    devices: Union[int, str] = Field(
        default=4,
        description="Number of devices (GPUs) per node or device specification",
    )
    num_nodes: int = Field(default=1, description="Number of compute nodes", ge=1)
    accelerator: str = Field(
        default="gpu", description="Accelerator type (gpu, cpu, tpu, auto)"
    )

    # Training Behavior for Distributed
    sync_batchnorm: bool = Field(
        default=True, description="Synchronize batch normalization across devices"
    )
    precision: Optional[Union[int, str]] = Field(
        default="bf16-mixed", description="Training precision for distributed training"
    )

    # Strategy-Specific Configuration
    strategy_kwargs: Dict[str, Any] = Field(
        default_factory=lambda: {
            "find_unused_parameters": False,
            "gradient_as_bucket_view": True,
            "process_group_backend": "nccl",
        },
        description="Strategy-specific keyword arguments",
    )

    model_config = ConfigDict(
        extra="allow",  # Allow additional fields
        validate_assignment=True,  # Validate fields when assigned after creation
        use_enum_values=True,  # Use enum values in serialization
        validate_by_name=True,  # Allow validation using field names and aliases
    )

    @field_validator("strategy")
    def validate_strategy(cls, v):
        """Validate distributed strategy."""
        valid_strategies = [
            "ddp",
            "ddp_spawn",
            "ddp_sharded",
            "ddp_sharded_spawn",
            "fsdp",
            "deepspeed",
            "ddp_find_unused_parameters_false",
            "ddp_find_unused_parameters_true",
            "horovod",
        ]
        if v not in valid_strategies:
            logging.warning(
                f"Unknown strategy: {v}. Valid options: {valid_strategies}"
            )
        return v

    @field_validator("accelerator")
    def validate_accelerator(cls, v):
        """Validate accelerator type."""
        valid_accelerators = ["cpu", "gpu", "tpu", "auto"]
        if v not in valid_accelerators:
            raise ValueError(
                f"Invalid accelerator: {v}. Valid options: {valid_accelerators}"
            )
        return v

    @field_validator("devices")
    def validate_devices(cls, v):
        """Validate device specification."""
        if isinstance(v, int) and v <= 0:
            raise ValueError("devices must be positive when specified as integer")
        elif isinstance(v, str):
            # Allow string specifications like "0,1,2,3" or "auto"
            if v not in ["auto"] and not all(c.isdigit() or c in ",[]" for c in v):
                raise ValueError("Invalid device string specification")
        return v

    @field_validator("precision")
    def validate_precision(cls, v):
        """Validate precision setting."""
        if v is not None:
            valid_precisions = [
                16,
                32,
                64,
                "16",
                "32",
                "64",
                "bf16",
                "16-mixed",
                "bf16-mixed",
            ]
            if v not in valid_precisions:
                raise ValueError(
                    f"Invalid precision: {v}. Valid options: {valid_precisions}"
                )
        return v

    @model_validator(mode="after")
    def validate_distributed_config(self):
        """Validate distributed configuration consistency."""
        # Strategy-specific validations
        if self.strategy in ["ddp", "ddp_spawn"] and self.accelerator == "cpu":
            logging.warning("DDP with CPU may have limited performance benefits")

        if (
            self.strategy == "fsdp"
            and isinstance(self.devices, int)
            and self.devices < 2
        ):
            logging.warning("FSDP typically requires multiple devices for benefits")

        # Validate strategy_kwargs based on strategy
        if self.strategy.startswith("ddp"):
            self._validate_ddp_kwargs()
        elif self.strategy == "fsdp":
            self._validate_fsdp_kwargs()
        elif self.strategy == "deepspeed":
            self._validate_deepspeed_kwargs()

        return self

    def _validate_ddp_kwargs(self):
        """Validate DDP-specific kwargs."""
        valid_ddp_kwargs = [
            "find_unused_parameters",
            "gradient_as_bucket_view",
            "process_group_backend",
            "bucket_cap_mb",
            "static_graph",
        ]

        for key in self.strategy_kwargs:
            if key not in valid_ddp_kwargs:
                logging.warning(
                    f"Unknown DDP kwarg: {key}. Valid options: {valid_ddp_kwargs}"
                )

        # Validate process_group_backend
        if "process_group_backend" in self.strategy_kwargs:
            backend = self.strategy_kwargs["process_group_backend"]
            valid_backends = ["nccl", "gloo", "mpi"]
            if backend not in valid_backends:
                raise ValueError(
                    f"Invalid process_group_backend: {backend}. Valid options: {valid_backends}"
                )

    def _validate_fsdp_kwargs(self):
        """Validate FSDP-specific kwargs."""
        valid_fsdp_kwargs = [
            "sharding_strategy",
            "cpu_offload",
            "mixed_precision",
            "auto_wrap_policy",
            "backward_prefetch",
            "forward_prefetch",
        ]

        for key in self.strategy_kwargs:
            if key not in valid_fsdp_kwargs:
                logging.warning(
                    f"Unknown FSDP kwarg: {key}. Valid options: {valid_fsdp_kwargs}"
                )

    def _validate_deepspeed_kwargs(self):
        """Validate DeepSpeed-specific kwargs."""
        valid_deepspeed_kwargs = [
            "config",
            "stage",
            "offload_optimizer",
            "offload_parameters",
            "cpu_checkpointing",
            "contiguous_gradients",
            "overlap_comm",
        ]

        for key in self.strategy_kwargs:
            if key not in valid_deepspeed_kwargs:
                logging.warning(
                    f"Unknown DeepSpeed kwarg: {key}. Valid options: {valid_deepspeed_kwargs}"
                )

    def create_strategy(self):
        """
        Create PyTorch Lightning strategy object based on configuration.

        Returns:
            PyTorch Lightning strategy instance
        """
        try:
            import pytorch_lightning as pl
        except ImportError:
            raise ImportError("PyTorch Lightning is required for strategy creation")

        strategy_name = self.strategy.lower()
        strategy_kwargs = self.strategy_kwargs.copy()

        if strategy_name in ["ddp", "ddp_spawn"]:
            if strategy_name == "ddp":
                return pl.strategies.DDPStrategy(**strategy_kwargs)
            else:
                return pl.strategies.DDPSpawnStrategy(**strategy_kwargs)

        elif strategy_name == "fsdp":
            return pl.strategies.FSDPStrategy(**strategy_kwargs)

        elif strategy_name == "deepspeed":
            return pl.strategies.DeepSpeedStrategy(**strategy_kwargs)

        elif strategy_name in ["ddp_sharded", "ddp_sharded_spawn"]:
            # For FairScale sharded strategies (if available)
            try:
                if strategy_name == "ddp_sharded":
                    return pl.strategies.DDPShardedStrategy(**strategy_kwargs)
                else:
                    return pl.strategies.DDPSpawnShardedStrategy(**strategy_kwargs)
            except AttributeError:
                logging.warning(
                    f"Strategy {strategy_name} not available, falling back to DDP"
                )
                return pl.strategies.DDPStrategy(**strategy_kwargs)

        else:
            # For string-based strategies
            logging.warning(f"Using string-based strategy: {self.strategy}")
            return self.strategy

    def get_trainer_kwargs(self) -> Dict[str, Any]:
        """
        Get trainer-specific kwargs from distributed configuration.

        Returns:
            Dictionary of trainer kwargs
        """
        trainer_kwargs = {
            "strategy": self.create_strategy(),
            "devices": self.devices,
            "num_nodes": self.num_nodes,
            "accelerator": self.accelerator,
            "sync_batchnorm": self.sync_batchnorm,
        }

        # Add precision if specified
        if self.precision is not None:
            trainer_kwargs["precision"] = self.precision

        return trainer_kwargs

    def get_override_params(self) -> Dict[str, Any]:
        """
        Get parameters that should override trainer settings.

        Returns:
            Dictionary of parameters to override in trainer config
        """
        overrides = self.get_trainer_kwargs

        # In Pydantic v2 with extra="allow", extra fields are stored in __pydantic_extra__
        if hasattr(self, "__pydantic_extra__"):
            overrides.update(self.__pydantic_extra__)

        return overrides


class TrainerParams(BaseParams):
    """
    Training parameters for PyTorch Lightning trainer configuration.

    Handles PyTorch Lightning Trainer-specific parameters including training behavior,
    system performance, callbacks, and advanced training options.

    Note: Optimizer and scheduler configuration belongs in ModelParams since they
    are configured in the model's configure_optimizers() method.
    """

    # Core Training Parameters
    epochs: int = Field(default=100, description="Number of training epochs", ge=1)
    check_val_every_n_epoch: int = Field(
        default=1, description="Validation frequency (epochs)", ge=1
    )
    log_every_n_steps: int = Field(
        default=50, description="Logging frequency (steps)", ge=1
    )

    # Training Behavior
    accumulate_grad_batches: int = Field(
        default=1, description="Number of batches to accumulate gradients", ge=1
    )
    precision: Union[int, str] = Field(
        default=32, description="Training precision (16, 32, 64, 'bf16', etc.)"
    )
    deterministic: Union[bool, str] = Field(
        default=False,
        description="Enable deterministic training (True, False, 'warn')",
    )

    # System Performance
    enable_progress_bar: bool = Field(
        default=True, description="Enable training progress bar"
    )
    profiler: Optional[str] = Field(
        default=None, description="Profiler type (simple, advanced, pytorch, None)"
    )
    benchmark: bool = Field(
        default=False, description="Enable PyTorch cudnn benchmark for performance"
    )

    # Distributed Training Configuration
    use_distributed: bool = Field(
        default=False, description="Enable distributed training with default settings"
    )
    distributed: Optional[DistributedParams] = Field(
        default=None,
        description="Distributed training configuration (auto-created if use_distributed=True)",
    )

    # Response Storage (for analysis during evaluation)
    store_responses: int = Field(
        default=0, description="Number of responses to store (0 = disabled)", ge=0
    )
    store_responses_on_cpu: bool = Field(
        default=True, description="Store responses on CPU to save GPU memory"
    )

    # Advanced Training Options
    gradient_clip_val: Optional[float] = Field(
        default=None, description="Gradient clipping value (None = disabled)"
    )
    gradient_clip_algorithm: str = Field(
        default="norm", description="Gradient clipping algorithm (norm, value)"
    )

    # Early Stopping
    early_stopping_patience: Optional[int] = Field(
        default=None, description="Early stopping patience (None = disabled)"
    )
    early_stopping_min_delta: float = Field(
        default=0.0, description="Minimum change for early stopping"
    )
    early_stopping_monitor: str = Field(
        default="val_loss", description="Metric to monitor for early stopping"
    )
    early_stopping_mode: str = Field(
        default="min", description="Early stopping mode (min, max)"
    )

    # Checkpoint Configuration
    save_top_k: int = Field(
        default=1, description="Number of best checkpoints to save"
    )
    monitor_checkpoint: str = Field(
        default="val_loss", description="Metric to monitor for checkpoint saving"
    )
    checkpoint_mode: str = Field(
        default="min", description="Checkpoint mode (min, max)"
    )
    save_last: bool = Field(default=True, description="Save the last checkpoint")

    class Config:
        extra = "allow"
        validate_assignment = True
        use_enum_values = True
        validate_by_name = True

    @classmethod
    def get_aliases(cls) -> Dict[str, str]:
        """Return mapping of aliases to full parameter names."""
        aliases = super().get_aliases()
        aliases.update(
            {
                "ep": "epochs",
                "val_freq": "check_val_every_n_epoch",
                "log_freq": "log_every_n_steps",
                "accum": "accumulate_grad_batches",
                "prec": "precision",
                "prog": "enable_progress_bar",
                "prof": "profiler",
                "clip": "gradient_clip_val",
                "patience": "early_stopping_patience",
                "topk": "save_top_k",
                "determ": "deterministic",
                "dist": "use_distributed",
            }
        )
        return aliases

    @field_validator("precision")
    def validate_precision(cls, v):
        """Validate precision setting."""
        valid_precisions = [
            16,
            32,
            64,
            "16",
            "32",
            "64",
            "bf16",
            "16-mixed",
            "bf16-mixed",
        ]
        if v not in valid_precisions:
            raise ValueError(
                f"Invalid precision: {v}. Valid options: {valid_precisions}"
            )
        return v

    @field_validator("deterministic")
    def validate_deterministic(cls, v):
        """Validate deterministic setting."""
        if isinstance(v, str) and v not in ["warn"]:
            raise ValueError(
                f"Invalid deterministic string: {v}. Valid options: 'warn'"
            )
        return v

    @field_validator("profiler")
    def validate_profiler(cls, v):
        """Validate profiler setting."""
        if v is not None:
            valid_profilers = ["simple", "advanced", "pytorch", "xla"]
            if v not in valid_profilers:
                raise ValueError(
                    f"Invalid profiler: {v}. Valid options: {valid_profilers}"
                )
        return v

    @field_validator("gradient_clip_algorithm")
    def validate_gradient_clip_algorithm(cls, v):
        """Validate gradient clipping algorithm."""
        valid_algorithms = ["norm", "value"]
        if v not in valid_algorithms:
            raise ValueError(
                f"Invalid gradient clip algorithm: {v}. Valid options: {valid_algorithms}"
            )
        return v

    @field_validator("distributed")
    def validate_distributed_config(cls, v):
        """Validate distributed configuration."""
        if v is not None and not isinstance(v, DistributedParams):
            raise ValueError(
                "distributed must be a DistributedParams instance or None"
            )
        return v

    @field_validator("early_stopping_mode", "checkpoint_mode")
    def validate_mode(cls, v):
        """Validate mode settings."""
        valid_modes = ["min", "max"]
        if v not in valid_modes:
            raise ValueError(f"Invalid mode: {v}. Valid options: {valid_modes}")
        return v

    @model_validator(mode="after")
    def validate_training_config(self):
        """Validate training configuration consistency."""
        # Handle distributed configuration
        if self.use_distributed:
            if self.distributed is None:
                # Create default distributed configuration
                self.update_field("distributed", DistributedParams())
                logging.info("Created default distributed configuration")

            # Apply override parameters from distributed config to trainer params
            override_params = self.distributed.get_override_params()
            for key, value in override_params.items():
                if hasattr(self, key):
                    old_value = getattr(self, key)
                    # Use object.__setattr__ to avoid triggering validation recursion
                    self.update_field(key, value)
                    logging.info(f"Overriding {key}: {old_value} -> {value}")
                else:
                    logging.warning(f"Cannot override unknown parameter: {key}")
        elif self.distributed is not None:
            logging.warning(
                "Distributed configuration provided but use_distributed=False"
            )

        # Validate early stopping configuration
        if self.early_stopping_patience is not None:
            if self.early_stopping_patience <= 0:
                raise ValueError("early_stopping_patience must be positive")

        # Validate checkpoint configuration
        if self.save_top_k < -1:
            raise ValueError("save_top_k must be >= -1 (-1 saves all)")

        # Validate gradient clipping
        if self.gradient_clip_val is not None and self.gradient_clip_val <= 0:
            raise ValueError("gradient_clip_val must be positive")

        return self

    def _get_world_size(self) -> int:
        """Get distributed world size."""
        if not self.use_distributed or not self.distributed:
            return 1

        devices = getattr(self.distributed, "devices", 1)
        num_nodes = getattr(self.distributed, "num_nodes", 1)

        if isinstance(devices, int):
            return devices * num_nodes
        elif isinstance(devices, list):
            return len(devices) * num_nodes
        else:
            return num_nodes  # Conservative estimate for string device specs

    @classmethod
    def _create_distributed_params(
        cls, distributed_dict: Dict[str, Any]
    ) -> DistributedParams:
        """
        Create DistributedParams from dictionary configuration.

        Args:
            distributed_dict: Dictionary containing distributed configuration

        Returns:
            DistributedParams instance
        """
        try:
            return DistributedParams(**distributed_dict)
        except Exception as e:
            raise DynVisionValidationError(
                f"Failed to create distributed configuration: {e}"
            )

    @classmethod
    def from_cli_and_config(
        cls,
        config_path: Optional[Union[str, Path]] = None,
        override_kwargs: Optional[Dict[str, Any]] = None,
        args: Optional[List[str]] = None,
    ) -> "TrainerParams":
        """
        Create config instance with proper handling of distributed parameters.

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

        # Handle distributed configuration specially
        if "distributed" in final_params and isinstance(
            final_params["distributed"], dict
        ):
            distributed_dict = final_params.pop("distributed")
            distributed_params = cls._create_distributed_params(distributed_dict)
            final_params["distributed"] = distributed_params

        try:
            return cls(**final_params)
        except Exception as e:
            raise DynVisionValidationError(f"Parameter validation failed: {e}")

    def get_pytorch_lightning_trainer_kwargs(self) -> Dict[str, Any]:
        """
        Convert trainer parameters to PyTorch Lightning Trainer kwargs.

        Returns:
            Dictionary of kwargs for pl.Trainer initialization
        """
        trainer_kwargs = {
            "max_epochs": self.epochs,
            "check_val_every_n_epoch": self.check_val_every_n_epoch,
            "log_every_n_steps": self.log_every_n_steps,
            "accumulate_grad_batches": self.accumulate_grad_batches,
            "precision": self.precision,
            "enable_progress_bar": self.enable_progress_bar,
            "profiler": self.profiler,
            "benchmark": self.benchmark,
            "deterministic": self.deterministic,
        }

        # Add gradient clipping if specified
        if self.gradient_clip_val is not None:
            trainer_kwargs["gradient_clip_val"] = self.gradient_clip_val
            trainer_kwargs["gradient_clip_algorithm"] = self.gradient_clip_algorithm

        # Add distributed training configuration if enabled
        if self.use_distributed and self.distributed is not None:
            distributed_kwargs = self.distributed.get_trainer_kwargs()
            trainer_kwargs.update(distributed_kwargs)

        return trainer_kwargs

    def get_distributed_config(self) -> Optional[DistributedParams]:
        """
        Get distributed training configuration.

        Returns:
            DistributedParams instance or None if disabled
        """
        if not self.use_distributed or self.distributed is None:
            return None

        return self.distributed

    def get_early_stopping_callback_kwargs(self) -> Optional[Dict[str, Any]]:
        """
        Get early stopping callback configuration.

        Returns:
            Dictionary of kwargs for EarlyStopping callback or None if disabled
        """
        if self.early_stopping_patience is None:
            return None

        return {
            "monitor": self.early_stopping_monitor,
            "patience": self.early_stopping_patience,
            "min_delta": self.early_stopping_min_delta,
            "mode": self.early_stopping_mode,
            "verbose": True,
        }

    def get_checkpoint_callback_kwargs(self) -> Dict[str, Any]:
        """
        Get model checkpoint callback configuration.

        Returns:
            Dictionary of kwargs for ModelCheckpoint callback
        """
        return {
            "monitor": self.monitor_checkpoint,
            "save_top_k": self.save_top_k,
            "mode": self.checkpoint_mode,
            "save_last": self.save_last,
            "auto_insert_metric_name": False,
        }

    def validate_context_requirements(self) -> List[str]:
        """
        Validate context-specific requirements for training.

        Returns:
            List of validation warnings/errors
        """
        issues = []

        # Check precision settings for training stability
        if self.precision in [16, "16", "bf16", "16-mixed", "bf16-mixed"]:
            issues.append(
                "Mixed precision training enabled - monitor for training instability"
            )

        return issues


# Example usage and testing
if __name__ == "__main__":
    # Test basic instantiation with defaults
    trainer_params = TrainerParams()
    print(f"Default trainer params: {trainer_params}")

    # Test with custom parameters
    custom_params = TrainerParams(
        epochs=200,
        precision="16-mixed",
        early_stopping_patience=10,
        deterministic="warn",
    )
    print(f"Custom trainer params: {custom_params}")

    if DistributedParams:
        distributed_params_obj = DistributedParams(
            strategy="ddp",
            devices=2,
            num_nodes=1,
            precision="bf16-mixed",
            strategy_kwargs={
                "find_unused_parameters": False,
                "process_group_backend": "nccl",
            },
        )

        trainer_with_distributed = TrainerParams(
            use_distributed=True, distributed=distributed_params_obj
        )
        print(f"Trainer with DistributedParams object: {trainer_with_distributed}")

        # Test distributed training configuration with dictionary (will be converted)
        trainer_with_dict_distributed = TrainerParams.from_cli_and_config(
            override_kwargs={
                "use_distributed": True,
                "distributed": {
                    "strategy": "fsdp",
                    "devices": 4,
                    "num_nodes": 2,
                    "precision": "bf16-mixed",
                    "strategy_kwargs": {"sharding_strategy": "FULL_SHARD"},
                    # These will override trainer params
                    "epochs": 300,
                },
            }
        )
        print(
            f"Trainer with dict distributed (converted): {trainer_with_dict_distributed}"
        )
        print(f"Note: epochs override: {trainer_with_dict_distributed.epochs}")

    # Test CLI parsing with aliases
    test_args = [
        "--ep",
        "50",
        "--patience",
        "5",
        "--prec",
        "32",
        "--dist",
        "true",
        "--auto_opt",
        "false",
    ]

    try:
        cli_params = TrainerParams.from_cli_and_config(args=test_args)
        print(f"CLI params: {cli_params}")

        # Test PyTorch Lightning integration
        trainer_kwargs = cli_params.get_pytorch_lightning_trainer_kwargs()
        print(f"PyTorch Lightning trainer kwargs: {trainer_kwargs}")

        # Test distributed configuration
        distributed_config = cli_params.get_distributed_config()
        print(f"Distributed config: {distributed_config}")

        # Test callback configurations
        early_stopping_kwargs = cli_params.get_early_stopping_callback_kwargs()
        print(f"Early stopping kwargs: {early_stopping_kwargs}")

        checkpoint_kwargs = cli_params.get_checkpoint_callback_kwargs()
        print(f"Checkpoint kwargs: {checkpoint_kwargs}")

        # Test validation
        issues = cli_params.validate_context_requirements()
        if issues:
            print(f"Validation issues: {issues}")

    except Exception as e:
        print(f"Error: {e}")

    print("\n--- Advanced Distributed Training Examples ---")

    # Simple distributed training (uses defaults)
    simple_distributed = TrainerParams(use_distributed=True)
    print(f"Simple distributed config: {simple_distributed.get_distributed_config()}")

    # Test strategy creation
    if simple_distributed.distributed:
        try:
            # This would work if PyTorch Lightning is installed
            strategy = simple_distributed.distributed.create_strategy()
            print(f"Created strategy: {type(strategy)}")
            pass
        except ImportError:
            print("PyTorch Lightning not available for strategy creation test")

    print("\n--- Config File Example ---")

    # Test configuration file loading
    import tempfile
    import yaml
    import os

    config_data = {
        "epochs": 150,
        "early_stopping_patience": 15,
        "precision": "16-mixed",
        "accumulate_grad_batches": 2,
        "check_val_every_n_epoch": 2,
        "deterministic": False,
        "use_distributed": True,
        "distributed": {
            "strategy": "ddp",
            "devices": 4,
            "precision": "bf16-mixed",
            "strategy_kwargs": {
                "find_unused_parameters": False,
                "gradient_as_bucket_view": True,
                "process_group_backend": "nccl",
            },
            # These parameters will override trainer settings
            "log_every_n_steps": 25,
        },
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config_data, f)
        temp_config = f.name

    try:
        file_params = TrainerParams.from_cli_and_config(config_path=temp_config)
        print(f"Config file params: {file_params}")

        # Show override behavior
        print(f"Distributed config: {file_params.distributed}")

        # Test trainer kwargs with overrides
        final_trainer_kwargs = file_params.get_pytorch_lightning_trainer_kwargs()
        print(f"Final trainer kwargs: {final_trainer_kwargs}")

    finally:
        os.unlink(temp_config)
