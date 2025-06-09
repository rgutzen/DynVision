"""Train a neural network model on a dataset.

This script handles the complete training pipeline for DynVision models, including:
- Data loading and preprocessing with FFCV/PyTorch support
- Model initialization and configuration
- Distributed training with PyTorch Lightning
- Checkpointing and monitoring
- Result saving and analysis

The script supports various training configurations with features like:
- Early stopping with minimum performance threshold
- Learning rate monitoring and scaling
- Weight distribution tracking
- Temporal dynamics monitoring for recurrent models

Example:
    $ python train_model.py --config_path configs/train_config.yaml --model_name DyRCNNx4
"""

import argparse
import logging
import multiprocessing
import os
import sys
import yaml
import json
from contextlib import contextmanager
from dataclasses import dataclass
from pydantic import BaseModel, Field, validator
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from dynvision import models
from dynvision.data.dataloader import (
    _adjust_data_dimensions,
    _adjust_label_dimensions,
    get_train_val_loaders,
)
from dynvision.data.ffcv_dataloader import get_ffcv_dataloader
from dynvision.project_paths import project_paths
from dynvision.utils import (
    parse_parameters,
    filter_kwargs,
    str_to_bool,
    handle_errors,
)
from dynvision.visualization import callbacks as custom_callbacks

logger = logging.getLogger(__name__)


@dataclass
class DistributedConfig:
    """Configuration for distributed training."""

    strategy: str = "ddp"
    find_unused_parameters: bool = False
    gradient_as_bucket_view: bool = True
    process_group_backend: str = "nccl"
    sync_batchnorm: bool = True
    num_nodes: int = 1
    accelerator: str = "gpu"


class TrainingConfig(BaseModel):
    """Configuration container with automatic type conversion and validation."""

    # Required fields (no defaults)
    model_name: str
    input_model_state: Path
    output_model_state: Path
    dataset_train: Path
    dataset_val: Path
    data_name: str

    # Simple default values
    epochs: int = Field(default=100, gt=0, description="Number of training epochs")
    batch_size: int = Field(default=32, gt=0, description="Training batch size")
    learning_rate: float = Field(default=1e-3, gt=0, description="Learning rate")
    resolution: int = Field(default=224, gt=0, description="Input resolution")
    n_timesteps: int = Field(default=1, gt=0, description="Number of timesteps")

    # Training behavior with defaults
    check_val_every_n_epoch: int = Field(
        default=1, gt=0, description="Validation frequency"
    )
    accumulate_grad_batches: int = Field(
        default=1, gt=0, description="Gradient accumulation"
    )
    precision: str = Field(
        default="32", regex=r"^(16|32|64|bf16)$", description="Numerical precision"
    )
    use_ffcv: bool = Field(default=False, description="Use FFCV for data loading")
    store_responses: int = Field(
        default=0, ge=0, description="Number of responses to store"
    )

    # Environment-based defaults
    use_distributed: bool = Field(
        default_factory=lambda: os.environ.get("USE_DISTRIBUTED", "false").lower()
        == "true",
        description="Enable distributed training",
    )
    enable_progress_bar: bool = Field(
        default_factory=lambda: not bool(
            os.environ.get("SLURM_JOB_ID")
        ),  # Disable in SLURM by default
        description="Show progress bar",
    )

    class Config:
        # Allow automatic type conversion
        validate_assignment = True
        use_enum_values = True
        # Allow extra fields in the configuration
        extra = "allow"

    @validator("input_model_state", "dataset_train", "dataset_val")
    def validate_file_exists(cls, v):
        if not v.exists():
            raise ValueError(f"File not found: {v}")
        return v

    @validator("output_model_state")
    def validate_output_dir(cls, v):
        v.parent.mkdir(parents=True, exist_ok=True)
        return v

    @classmethod
    def from_args(
        cls, args: argparse.Namespace, config_file: Optional[Path] = None
    ) -> "TrainingConfig":
        """Create configuration from argparse Namespace with optional config file."""

        # Start with config file if provided
        config_dict = {}
        if config_file and config_file.exists():
            config_dict = cls._load_config_file(config_file)

        # Override with command line arguments (CLI takes precedence)
        cli_dict = {k: v for k, v in vars(args).items() if v is not None}
        config_dict.update(cli_dict)

        # Handle special cases
        config_dict = cls._preprocess_args(config_dict)

        return cls(**config_dict)

    @staticmethod
    def _load_config_file(config_path: Path) -> Dict[str, Any]:
        """Load configuration from YAML or JSON file."""
        with open(config_path) as f:
            if config_path.suffix.lower() in [".yml", ".yaml"]:
                return yaml.safe_load(f) or {}
            elif config_path.suffix.lower() == ".json":
                return json.load(f)
            else:
                raise ValueError(
                    f"Unsupported config file format: {config_path.suffix}"
                )

    @staticmethod
    def _preprocess_args(args_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess arguments before Pydantic validation."""
        # Handle list arguments that come from argparse as strings
        list_fields = ["loss", "loss_config"]
        for field in list_fields:
            if field in args_dict and isinstance(args_dict[field], str):
                args_dict[field] = [args_dict[field]]

        # Handle boolean strings
        bool_fields = ["use_ffcv", "enable_progress_bar"]
        for field in bool_fields:
            if field in args_dict and isinstance(args_dict[field], str):
                args_dict[field] = args_dict[field].lower() in (
                    "true",
                    "1",
                    "yes",
                    "on",
                )

        return args_dict

    def to_model_kwargs(self) -> Dict[str, Any]:
        """Extract arguments suitable for model constructor."""
        return {
            **self.model_args,
            "n_timesteps": self.n_timesteps,
            "input_dims": getattr(self, "input_dims", None),
            "n_classes": getattr(self, "n_classes", None),
        }


class EarlyStoppingWithMin(pl.callbacks.EarlyStopping):
    """Enhanced early stopping with minimum performance threshold.

    This callback extends the standard early stopping by adding a minimum
    performance threshold that must be met before early stopping is considered.
    This prevents premature stopping when the model hasn't learned the task yet.

    Args:
        monitor: Metric to monitor for early stopping
        patience: Number of epochs to wait for improvement
        min_val_accuracy: Minimum validation accuracy required before stopping
        **kwargs: Additional arguments passed to EarlyStopping
    """

    def __init__(
        self,
        monitor: str = "val_accuracy",
        patience: int = 5,
        min_val_accuracy: float = 0.7,
        **kwargs: Any,
    ) -> None:
        super().__init__(monitor=monitor, patience=patience, **kwargs)
        self.min_val_accuracy = min_val_accuracy

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Check validation metrics and update early stopping state.

        Args:
            trainer: PyTorch Lightning trainer instance
            pl_module: The model being trained
        """
        current_value = trainer.callback_metrics.get(self.monitor)
        if current_value is not None and current_value >= self.min_val_accuracy:
            super().on_validation_epoch_end(trainer, pl_module)
        else:
            # Reset wait count if minimum performance not met
            self.wait_count = 0


class DistributedTrainingManager:
    """Manages distributed training configuration and setup."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.gpu_count = torch.cuda.device_count()
        self.use_distributed = self._should_use_distributed()

    def _should_use_distributed(self) -> bool:
        """Determine if distributed training should be used."""
        if self.config.use_distributed:
            if self.gpu_count > 1:
                return True
            else:
                logger.warning(
                    "Distributed training requested but only one GPU available. "
                    "Falling back to single-GPU training."
                )
        return False

    def get_distributed_info(
        self, trainer: Optional[pl.Trainer] = None
    ) -> Dict[str, Any]:
        """Get distributed training information.

        Args:
            trainer: PyTorch Lightning trainer for distributed info

        Returns:
            Dictionary containing distributed training configuration
        """
        if not self.use_distributed:
            return {"distributed": False, "rank": 0, "world_size": 1}

        # Try to get info from trainer first
        if (
            trainer is not None
            and hasattr(trainer, "world_size")
            and trainer.world_size > 1
        ):
            return {
                "distributed": True,
                "rank": trainer.global_rank,
                "world_size": trainer.world_size,
            }

        # Fallback to environment variables
        return {
            "distributed": True,
            "rank": int(os.environ.get("RANK", "0")),
            "world_size": int(os.environ.get("WORLD_SIZE", "1")),
        }

    def configure_trainer_strategy(self, config: TrainingConfig) -> Dict[str, Any]:
        """Configure trainer strategy for distributed training.

        Args:
            config: Training configuration

        Returns:
            Dictionary containing trainer configuration
        """
        if not self.use_distributed:
            return {
                "devices": 1,
                "strategy": "auto",
                "num_nodes": 1,
                "accelerator": "auto",
                "sync_batchnorm": False,
            }

        # Distributed training configuration
        dist_config = config.distributed
        strategy = pl.strategies.DDPStrategy(
            find_unused_parameters=dist_config.find_unused_parameters,
            gradient_as_bucket_view=dist_config.gradient_as_bucket_view,
            process_group_backend=dist_config.process_group_backend,
        )

        return {
            "strategy": strategy,
            "devices": self.gpu_count,
            "num_nodes": dist_config.num_nodes,
            "accelerator": dist_config.accelerator,
            "sync_batchnorm": dist_config.sync_batchnorm,
        }


class DataLoaderFactory:
    """Factory for creating training and validation data loaders."""

    def __init__(
        self, config: TrainingConfig, distributed_manager: DistributedTrainingManager
    ):
        self.config = config
        self.distributed_manager = distributed_manager

    def create_loaders(
        self, trainer: Optional[pl.Trainer] = None
    ) -> Tuple[DataLoader, DataLoader]:
        """Create training and validation data loaders.

        Args:
            trainer: PyTorch Lightning trainer for distributed info

        Returns:
            Tuple of (train_loader, val_loader)
        """
        # Get data normalization parameters
        data_stats = getattr(self.config, "data_statistics", {})
        data_mean = data_stats.get(self.config.data_name, {}).get("mean", 0.5)
        data_std = data_stats.get(self.config.data_name, {}).get("std", 0.5)

        # Base dataloader arguments
        dataloader_args = {
            "batch_size": self.config.batch_size,
            "encoding": "image",
            "resolution": self.config.data_resolution,
            "normalize": (data_mean, data_std),
        }

        # Add distributed training parameters
        dist_info = self.distributed_manager.get_distributed_info(trainer)
        dataloader_args.update(dist_info)

        if dist_info["distributed"]:
            logger.info(
                f"Setting up distributed data loading: rank={dist_info['rank']}, "
                f"world_size={dist_info['world_size']}"
            )

        # Create loaders based on configuration
        if self.config.use_ffcv:
            return self._create_ffcv_loaders(dataloader_args)
        else:
            return self._create_pytorch_loaders(dataloader_args)

    def _create_ffcv_loaders(
        self, dataloader_args: Dict[str, Any]
    ) -> Tuple[DataLoader, DataLoader]:
        """Create FFCV data loaders for optimized performance."""
        train_loader = get_ffcv_dataloader(
            path=self.config.dataset_train,
            data_transform="ffcv_train",
            **dataloader_args,
        )

        val_loader = get_ffcv_dataloader(
            path=self.config.dataset_val,
            data_transform="ffcv_test",
            **dataloader_args,
        )

        return train_loader, val_loader

    def _create_pytorch_loaders(
        self, dataloader_args: Dict[str, Any]
    ) -> Tuple[DataLoader, DataLoader]:
        """Create standard PyTorch data loaders."""
        return get_train_val_loaders(
            path_train=self.config.dataset_train,
            path_val=self.config.dataset_val,
            data_transform="ffcv_train",  # TODO: Update to use appropriate transform
            **dataloader_args,
        )


class CallbackManager:
    """Manages training callbacks and checkpointing."""

    def __init__(self, config: TrainingConfig):
        self.config = config

    def setup_callbacks(self) -> Tuple[List[pl.Callback], Path]:
        """Set up training callbacks and return checkpoint path.

        Returns:
            Tuple of (callbacks_list, checkpoint_path)
        """
        callbacks = []

        # Add model-specific callbacks
        if self.config.n_timesteps > 1:
            callbacks.append(custom_callbacks.MonitorClassifierResponses())

        callbacks.append(custom_callbacks.MonitorWeightDistributions())

        # Setup checkpointing
        checkpoint_path = self._setup_checkpointing(callbacks)

        # Add early stopping
        callbacks.append(
            EarlyStoppingWithMin(
                monitor="val_accuracy", patience=5, mode="max", min_val_accuracy=0.7
            )
        )

        # Add learning rate monitor
        callbacks.append(pl.callbacks.LearningRateMonitor(logging_interval="epoch"))

        return callbacks, checkpoint_path

    def _setup_checkpointing(self, callbacks: List[pl.Callback]) -> Path:
        """Setup model checkpointing callback.

        Args:
            callbacks: List to append checkpoint callback to

        Returns:
            Path to checkpoint directory
        """
        checkpoint_path = (
            project_paths.large_logs
            / "checkpoints"
            / self.config.output_model_state.stem
        )

        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            monitor="val_loss",
            save_top_k=1,
            mode="min",
            dirpath=checkpoint_path.parent,
            filename=checkpoint_path.name + "-{epoch:02d}-{val_loss:.2f}",
            save_last=True,
        )

        callbacks.append(checkpoint_callback)
        return checkpoint_path


class ModelManager:
    """Manages model loading, initialization, and saving."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def load_and_initialize_model(
        self, input_shape: Tuple[int, ...]
    ) -> pl.LightningModule:
        """Load and initialize model from checkpoint.

        Args:
            input_shape: Shape of input data (n_timesteps, channels, height, width)

        Returns:
            Initialized PyTorch Lightning model
        """
        # Load state dict to determine number of classes
        state_dict = torch.load(
            self.config.input_model_state, map_location=self.device
        )
        n_classes = self._extract_num_classes(state_dict)

        # Update configuration with inferred parameters
        self.config.additional_args["input_dims"] = input_shape
        self.config.additional_args["n_classes"] = n_classes

        # Create and initialize model
        model_class = getattr(models, self.config.model_name)
        model_args, _ = filter_kwargs(
            model_class, vars(self.config) | self.config.additional_args
        )
        model = model_class(**model_args).to(self.device)

        # Load state dict
        model.load_state_dict(state_dict)

        # Initialize model-specific settings
        if hasattr(model, "set_residual_timesteps"):
            model.set_residual_timesteps()

        return model

    def _extract_num_classes(self, state_dict: Dict[str, torch.Tensor]) -> int:
        """Extract number of classes from state dict.

        Args:
            state_dict: Model state dictionary

        Returns:
            Number of output classes
        """
        # Get the last layer (classifier) to determine number of classes
        last_key = next(reversed(state_dict))
        return len(state_dict[last_key])

    def save_model(self, model: pl.LightningModule) -> None:
        """Save trained model state.

        Args:
            model: Trained model to save
        """
        self.config.output_model_state.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), self.config.output_model_state)
        logger.info(f"Model saved to {self.config.output_model_state}")


class TrainingOrchestrator:
    """Main orchestrator for the training pipeline."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.distributed_manager = DistributedTrainingManager(config)
        self.dataloader_factory = DataLoaderFactory(config, self.distributed_manager)
        self.callback_manager = CallbackManager(config)
        self.model_manager = ModelManager(config)

    @contextmanager
    def training_context(self):
        """Context manager for training setup and cleanup."""
        try:
            # Setup
            torch.set_float32_matmul_precision("medium")
            torch.cuda.empty_cache()

            # Log system information
            self._log_system_info()

            yield

        finally:
            # Cleanup
            torch.cuda.empty_cache()

    def _log_system_info(self) -> None:
        """Log system and training configuration information."""
        import os

        num_cpu_cores = len(os.sched_getaffinity(0))
        num_gpu_cores = torch.cuda.device_count()

        logger.info(
            f"Available compute resources: CPU={num_cpu_cores}, GPU={num_gpu_cores}"
        )
        logger.info(f"Training device: {self.model_manager.device}")
        logger.info(
            f"Distributed training: {self.distributed_manager.use_distributed}"
        )
        logger.info(f"Learning rate: {self.config.learning_rate}")

    def _setup_trainer(
        self, callbacks: List[pl.Callback], pl_logger: pl.loggers.Logger
    ) -> pl.Trainer:
        """Setup PyTorch Lightning trainer.

        Args:
            callbacks: List of training callbacks
            pl_logger: PyTorch Lightning logger

        Returns:
            Configured trainer
        """
        # Base trainer configuration
        trainer_kwargs = {
            "callbacks": callbacks,
            "max_epochs": self.config.epochs,
            "logger": pl_logger,
            "precision": self.config.precision,
            "check_val_every_n_epoch": self.config.check_val_every_n_epoch,
            "accumulate_grad_batches": self.config.accumulate_grad_batches,
            "enable_progress_bar": self.config.enable_progress_bar,
            "benchmark": True,
            "log_every_n_steps": getattr(self.config, "log_every_n_steps", 50),
            "limit_train_batches": 1.0,
            "limit_val_batches": 0.25,
            "num_sanity_val_steps": 0,
            "reload_dataloaders_every_n_epochs": 0,
        }

        # Add distributed training configuration
        trainer_kwargs.update(
            self.distributed_manager.configure_trainer_strategy(self.config)
        )

        # Filter valid arguments
        trainer_kwargs, _ = filter_kwargs(pl.Trainer, trainer_kwargs)

        return pl.Trainer(**trainer_kwargs)

    def _log_training_data_example(
        self, train_loader: DataLoader, pl_logger: pl.loggers.Logger
    ) -> Tuple[int, ...]:
        """Log example training data and return input shape.

        Args:
            train_loader: Training data loader
            pl_logger: PyTorch Lightning logger

        Returns:
            Input shape tuple
        """
        inputs, label_indices, *paths = next(iter(train_loader))
        inputs = _adjust_data_dimensions(inputs)
        label_indices = _adjust_label_dimensions(label_indices)

        logger.info(f"Input shape: {inputs.size()}")
        logger.info(f"Label shape: {label_indices.size()}")
        logger.info(
            f"Pixel values in first batch: {inputs.mean():.3f} ± {inputs.std():.3f}"
        )

        batch_size, n_timesteps, *input_shape = inputs.shape

        # Log example images
        if hasattr(pl_logger, "log_image"):
            pl_logger.log_image(
                key="input_samples",
                images=[
                    inputs[0, t] for t in range(min(n_timesteps, 5))
                ],  # Limit to 5 images
                caption=[
                    str(label_indices[0, t].item()) for t in range(min(n_timesteps, 5))
                ],
            )

        return (n_timesteps, *input_shape)

    def run_training(self) -> int:
        """Run the complete training pipeline.

        Returns:
            Exit code (0 for success)
        """
        with self.training_context():
            try:
                # Initialize logger
                pl_logger = pl.loggers.WandbLogger(
                    project=project_paths.project_name,
                    save_dir=project_paths.large_logs,
                    config=vars(self.config),
                    tags=["train"],
                )

                # Setup callbacks and trainer
                callbacks, checkpoint_path = self.callback_manager.setup_callbacks()
                trainer = self._setup_trainer(callbacks, pl_logger)

                # Setup data loaders
                train_loader, val_loader = self.dataloader_factory.create_loaders(
                    trainer
                )

                # Log training data example
                input_shape = self._log_training_data_example(train_loader, pl_logger)

                # Load and initialize model
                model = self.model_manager.load_and_initialize_model(input_shape)

                # Check for existing checkpoint
                checkpoint_path = self._find_existing_checkpoint(checkpoint_path)

                # Train model
                trainer.fit(model, train_loader, val_loader, ckpt_path=checkpoint_path)

                # Save trained model
                self.model_manager.save_model(model)

                return 0

            except Exception as e:
                logger.error(f"Training failed: {e}")
                return 1

    def _find_existing_checkpoint(self, checkpoint_path: Path) -> Optional[Path]:
        """Find existing checkpoint for resuming training.

        Args:
            checkpoint_path: Base checkpoint path

        Returns:
            Path to existing checkpoint or None
        """
        files = list(checkpoint_path.parent.glob(f"{checkpoint_path.name}*"))
        if files:
            latest_checkpoint = files[-1]
            logger.info(f"Found existing checkpoint: {latest_checkpoint}")
            return latest_checkpoint
        return None


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and return the argument parser for model training.

    Returns:
        Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description="Train a neural network model with DynVision.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "--input_model_state",
        type=Path,
        required=True,
        help="Path to initial model state",
    )
    parser.add_argument(
        "--output_model_state",
        type=Path,
        required=True,
        help="Path to save trained model",
    )
    parser.add_argument(
        "--model_name", type=str, required=True, help="Name of the model class"
    )
    parser.add_argument(
        "--dataset_train", type=Path, required=True, help="Path to training dataset"
    )
    parser.add_argument(
        "--dataset_val", type=Path, required=True, help="Path to validation dataset"
    )
    parser.add_argument(
        "--data_name", type=str, required=True, help="Name of data used for transform"
    )
    parser.add_argument("--loss", nargs="+", type=str, help="Loss function names")

    # Configuration
    parser.add_argument(
        "--config_path", type=Path, help="Path to the training configuration file"
    )

    # parameters with special handling
    parser.add_argument(
        "--n_timesteps",
        "--tsteps",
        type=int,
        help="Number of timesteps to repeat image",
    )

    parser.add_argument(
        "--enable_progress_bar",
        type=str_to_bool,
        help="Show progress bar during training",
    )
    parser.add_argument(
        "--use_ffcv", type=str_to_bool, help="Use FFCV for data loading"
    )

    parser.add_argument(
        "--loss_config", nargs="+", type=str, help="Loss function configurations"
    )
    parser.add_argument(
        "--benchmark", type=str_to_bool, help="Enable benchmarking mode"
    )

    return parser


@handle_errors(verbose=True)
def main() -> int:
    """Main entry point for training."""
    parser = create_argument_parser()
    # Parse command line arguments and combine them with configuration file
    args = parse_parameters(parser)

    # Convert parsed arguments to TrainingConfig
    known, unknown = filter_kwargs(TrainingConfig, vars(args))
    config = TrainingConfig(**known, additional_args=unknown)

    # Create and run training orchestrator
    orchestrator = TrainingOrchestrator(config)
    return orchestrator.run_training()


def run_main() -> int:
    """Entry point for distributed training."""
    return main()


if __name__ == "__main__":
    # Set process start method for SLURM compatibility
    if os.environ.get("SLURM_JOB_ID"):
        multiprocessing.set_start_method("spawn", force=True)

    # Run training
    sys.exit(run_main())
