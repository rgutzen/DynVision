"""Train a neural network model on a dataset.

This script handles the complete training pipeline, including:
- Data loading and preprocessing
- Model initialization and configuration
- Training
- Checkpointing and monitoring
- Result saving and analysis

The script supports various training configurations and includes features like:
- Early stopping with minimum performance threshold
- Learning rate monitoring
- Weight distribution tracking
- Temporal dynamics monitoring for applicable models

Example:
    $ python train_model.py --config_path configs/train_config.yaml --model_name MyModel
"""

import argparse
import logging
import multiprocessing
import os
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


class EarlyStoppingWithMin(pl.callbacks.EarlyStopping):
    """Enhanced early stopping with minimum performance threshold.

    This callback extends the standard early stopping by adding a minimum
    performance threshold that must be met before early stopping is considered.

    Args:
        monitor: Metric to monitor
        patience: Number of epochs to wait for improvement
        min_val_accuracy: Minimum validation accuracy required
        **kwargs: Additional arguments passed to EarlyStopping
    """

    def __init__(
        self,
        monitor: str = "val_accuracy",
        patience: int = 5,
        min_val_accuracy: float = 0.75,
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
            self.wait_count = 0


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and return the argument parser for model training.

    Returns:
        argparse.ArgumentParser: Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description="Train a neural network model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config_path",
        type=Path,
        help="Path to the training configuration file",
    )
    parser.add_argument(
        "--input_model_state",
        type=Path,
        required=True,
        help="Path to initial model state",
    )
    parser.add_argument("--model_name", type=str, help="Name of the model class")
    parser.add_argument(
        "--dataset_train", type=Path, required=True, help="Path to training dataset"
    )
    parser.add_argument(
        "--dataset_val", type=Path, required=True, help="Path to validation dataset"
    )
    parser.add_argument(
        "--data_name",
        type=str,
        required=True,
        help="Name of data used for transform",
    )
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, help="Training batch size")
    parser.add_argument(
        "--output_model_state",
        type=Path,
        required=True,
        help="Path to save trained model",
    )
    parser.add_argument("--resolution", type=int, help="Input image resolution")
    parser.add_argument(
        "--n_timesteps",
        "--tsteps",
        type=int,
        help="Number of timesteps to repeat image",
    )
    parser.add_argument("--seed", type=str, help="Random seed for reproducibility")
    parser.add_argument(
        "--check_val_every_n_epoch",
        type=int,
        help="Validation check interval",
    )
    parser.add_argument(
        "--accumulate_grad_batches",
        type=int,
        help="Number of batches for gradient accumulation",
    )
    parser.add_argument(
        "--precision", type=str, help="Numerical precision for training"
    )
    parser.add_argument("--profiler", type=str, default="None", help="Profiler type")
    parser.add_argument(
        "--benchmark", type=str_to_bool, help="Enable benchmarking mode"
    )
    parser.add_argument(
        "--store_responses", type=int, help="Number of responses to store"
    )
    parser.add_argument(
        "--enable_progress_bar",
        type=str_to_bool,
        default=True,
        help="Show progress bar during training",
    )
    parser.add_argument(
        "--loss", nargs="+", type=str, required=True, help="Loss function names"
    )
    parser.add_argument(
        "--loss_config", nargs="+", type=str, help="Loss function configurations"
    )
    parser.add_argument(
        "--use_ffcv", type=str_to_bool, help="Use FFCV for data loading"
    )
    return parser


@handle_errors(verbose=True)
def setup_data_loaders(
    config: Any, dataloader_args: Dict[str, Any], trainer=None
) -> Tuple[DataLoader, DataLoader]:
    """Set up training and validation data loaders.

    Args:
        config: Configuration object
        dataloader_args: Arguments for data loader initialization
        trainer: PyTorch Lightning trainer (for distributed training info)

    Returns:
        Tuple containing:
            - Training data loader
            - Validation data loader
    """
    is_distributed = False

    if (
        trainer is not None
        and hasattr(trainer, "world_size")
        and trainer.world_size > 1
    ):
        is_distributed = True
        current_rank = trainer.global_rank
        current_world_size = trainer.world_size
    elif (
        os.environ.get("USE_DISTRIBUTED", "false").lower() == "true"
        or int(os.environ.get("WORLD_SIZE", 1)) > 1
    ):
        # Fallback for singularity containers where trainer might not be fully initialized
        is_distributed = True
        current_rank = int(os.environ.get("RANK", "0"))
        current_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    else:
        # Single GPU training
        current_rank = 0
        current_world_size = 1

    if is_distributed:
        # For distributed training, add rank and world size info
        dataloader_args.update(
            {
                "rank": current_rank,
                "world_size": current_world_size,
                "distributed": True,
            }
        )
        logger.info(
            f"Setting up distributed data loading: rank={current_rank}, world_size={current_world_size}"
        )

    # Check if PyTorch distributed is actually initialized
    is_distributed_initialized = (
        torch.distributed.is_available() and torch.distributed.is_initialized()
    )

    logger.info(
        f"Data loader setup: is_distributed={is_distributed}, "
        f"world_size_env={current_world_size}, distributed_initialized={is_distributed_initialized}"
    )

    # Only use distributed data loading if PyTorch distributed is properly initialized
    # Otherwise, let PyTorch Lightning handle the distributed setup later
    if is_distributed_initialized and is_distributed:
        current_rank = torch.distributed.get_rank()
        current_world_size = torch.distributed.get_world_size()

        dataloader_args.update(
            {
                "rank": current_rank,
                "world_size": current_world_size,
                "distributed": True,
            }
        )
        logger.info(
            f"Using distributed data loading: rank={current_rank}, world_size={current_world_size}"
        )
    else:
        # Don't use distributed data loading - PyTorch Lightning will handle distribution
        if is_distributed:
            logger.info(
                "Distributed training planned but process group not initialized yet - using single-process data loading"
            )
        else:
            logger.info("Single GPU training - using single-process data loading")

        dataloader_args.update(
            {
                "distributed": False,
            }
        )

    if config.use_ffcv:
        train_loader = get_ffcv_dataloader(
            path=config.dataset_train,
            data_transform=f"ffcv_train",
            **dataloader_args,
        )

        val_loader = get_ffcv_dataloader(
            path=config.dataset_val,
            data_transform=f"ffcv_test",
            **dataloader_args,
        )
    else:
        # TODO: Add support for non-ffcv data loaders
        train_loader, val_loader = get_train_val_loaders(
            path_train=config.dataset_train,
            path_val=config.dataset_val,
            data_transform="ffcv_train",
            **dataloader_args,
        )
    return train_loader, val_loader


def setup_callbacks(config: Any) -> List[pl.Callback]:
    """Set up training callbacks.

    Args:
        model: Model instance
        config: Configuration object

    Returns:
        List[pl.Callback]: List of configured callbacks
    """
    callbacks = []

    if hasattr(config, "n_timesteps") and config.n_timesteps > 1:
        callbacks.append(custom_callbacks.MonitorClassifierResponses())

    callbacks.append(custom_callbacks.MonitorWeightDistributions())

    # Setup checkpointing
    checkpoint_path = (
        project_paths.large_logs / "checkpoints" / config.output_model_state.stem
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

    # Add early stopping
    early_stop_callback = EarlyStoppingWithMin(
        monitor="val_accuracy", patience=5, mode="max", min_val_accuracy=0.7
    )
    callbacks.append(early_stop_callback)

    # Add learning rate monitor
    callbacks.append(pl.callbacks.LearningRateMonitor(logging_interval="epoch"))

    return callbacks, checkpoint_path


def setup_trainer(
    config: Any,
    train_logger: pl.loggers.WandbLogger,
    callbacks: List[pl.Callback] = [],
) -> pl.Trainer:
    # Base trainer settings
    trainer_kwargs = {
        "callbacks": callbacks,
        "max_epochs": config.epochs,
        "logger": train_logger,
        "precision": config.precision,
        "check_val_every_n_epoch": config.check_val_every_n_epoch,
        "accumulate_grad_batches": config.accumulate_grad_batches,
        "profiler": None if config.profiler == "None" else config.profiler,
        "enable_progress_bar": config.enable_progress_bar,
        "benchmark": getattr(config, "benchmark", True),
        "log_every_n_steps": int(config.log_every_n_steps),
        "limit_train_batches": 1.0,  # Use full dataset
        "limit_val_batches": 0.25,  # Use smaller validation set
        "num_sanity_val_steps": 0,  # Skip sanity check
        "reload_dataloaders_every_n_epochs": 0,  # Don't reload unnecessarily
    }
    # Check GPU availability - use PyTorch's detection
    gpu_count = torch.cuda.device_count()
    use_distributed = os.environ.get("USE_DISTRIBUTED", "false").lower() == "true"

    logger.info(f"GPU configuration: gpu_count={gpu_count}")
    logger.info(f"Distributed training enabled: {use_distributed}")

    # Configure training strategy based on available resources
    if use_distributed and gpu_count > 1:
        # PyTorch Lightning DDP - let Lightning handle process spawning
        logger.info(f"Using PyTorch Lightning DDP with {gpu_count} GPUs")

        dist_config = config.distributed

        # Set distributed training parameters
        trainer_kwargs.update(
            {
                "strategy": dist_config.get("strategy", "ddp"),
                "devices": gpu_count,  # Use all visible GPUs
                "num_nodes": dist_config.get("num_nodes", 1),
                "accelerator": dist_config.get("accelerator", "gpu"),
                "sync_batchnorm": dist_config.get("sync_batchnorm", True),
            }
        )

        # Add DDP-specific settings if using DDP
        if trainer_kwargs["strategy"] == "ddp":
            trainer_kwargs["strategy"] = pl.strategies.DDPStrategy(
                find_unused_parameters=dist_config.get(
                    "find_unused_parameters", False
                ),
                gradient_as_bucket_view=dist_config.get(
                    "gradient_as_bucket_view", True
                ),
                process_group_backend=dist_config.get("process_group_backend", "nccl"),
                # start_method="spawn",  # Force process spawning
            )

        # Log distributed training configuration
        logger.info("Distributed training configuration:")
        logger.info(f"Strategy: {trainer_kwargs.get('strategy')}")
        logger.info(f"Number of devices: {trainer_kwargs.get('devices')}")
        logger.info(f"Number of nodes: {trainer_kwargs.get('num_nodes')}")
    else:
        # Single GPU training
        logger.info("Using single GPU training")
        trainer_kwargs.update(
            {
                "devices": 1,
                "strategy": "auto",
                "num_nodes": 1,
                "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
                "sync_batchnorm": False,
            }
        )

    trainer_kwargs, _ = filter_kwargs(
        pl.Trainer,
        trainer_kwargs,
    )
    return pl.Trainer(**trainer_kwargs)


@handle_errors(verbose=True)
def run_training(config) -> int:
    # Initialize logger for pytorch lightning
    pl_logger = pl.loggers.WandbLogger(
        project=project_paths.project_name,
        save_dir=project_paths.large_logs,
        config=vars(config),
        tags=["train"],
    )

    logger.setLevel(getattr(logging, config.log_level.upper(), "INFO"))

    # Log system information
    num_cpu_cores = len(os.sched_getaffinity(0))
    num_gpu_cores = torch.cuda.device_count()
    logger.info(
        f"Available compute resources: CPU={num_cpu_cores}, GPU={num_gpu_cores}"
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Trying to use device {device}")

    # Scale learning rate if enabled
    print(f"Learning Rate: {config.learning_rate}")
    if hasattr(config, "lr_scaling") and config.lr_scaling["enabled"]:
        scale = min(
            config.batch_size / config.lr_scaling["ref_batch_size"],
            config.lr_scaling["threshold_factor"],
        )
        config.learning_rate *= scale
        logger.info(
            f"Scaled learning rate by {scale:.4f} for batch size {config.batch_size}"
        )

    # Setup training
    callbacks, checkpoint_path = setup_callbacks(config)
    trainer = setup_trainer(callbacks=callbacks, config=config, train_logger=pl_logger)

    # Setup data loaders
    data_mean = config.data_statistics[config.data_name]["mean"]
    data_std = config.data_statistics[config.data_name]["std"]
    dataloader_args = {
        # not adding time extension in the data loader causes the model to repeat input
        # over time during the model steps which is more efficient, because it requires
        # less GPU transfer
        # "n_timesteps": config.n_timesteps,
        "batch_size": config.batch_size,
        "encoding": "image",
        "resolution": config.resolution,
        "normalize": (data_mean, data_std),
    }
    train_loader, val_loader = setup_data_loaders(
        config, dataloader_args=dataloader_args, trainer=trainer
    )

    # Log example training data
    inputs, label_indices, *paths = next(iter(train_loader))
    inputs = _adjust_data_dimensions(inputs)
    label_indices = _adjust_label_dimensions(label_indices)

    logger.info(f"input shape: {inputs.size()}")
    logger.info(f"label shape: {label_indices.size()}")
    logger.info(
        f"pixel values in first batch: {inputs.mean():.3f} ± {inputs.std():.3f}"
    )

    batch_size, n_timesteps, *input_shape = inputs.shape
    pl_logger.log_image(
        key="input_samples",
        images=[inputs[0, t] for t in range(n_timesteps)],
        caption=[str(label_indices[0, t]) for t in range(n_timesteps)],
    )

    # Load model
    state_dict = torch.load(config.input_model_state, map_location=device)
    last_key = next(reversed(state_dict))
    n_classes = len(state_dict[last_key])

    input_dims = (n_timesteps, *input_shape)
    setattr(config, "input_dims", input_dims)
    setattr(config, "n_classes", n_classes)

    model_class = getattr(models, config.model_name)
    model_args, _ = filter_kwargs(model_class, vars(config))
    model = model_class(**model_args).to(device)
    model.load_state_dict(state_dict)

    # Determine residual timesteps before training to avoid issues in distributed mode
    if model.hasattr("set_residual_timesteps"):
        model.set_residual_timesteps()

    # Load checkpoint if available
    files = list(checkpoint_path.parent.glob(f"{checkpoint_path.name}*"))
    if files:
        checkpoint_path = files[-1]
        model = model_class.load_from_checkpoint(checkpoint_path)
        logger.info(f"Resumed from checkpoint: {checkpoint_path}")
    else:
        checkpoint_path = None

    # Train model
    torch.set_float32_matmul_precision("medium")
    torch.cuda.empty_cache()

    trainer.fit(
        model,
        train_loader,
        val_loader,
        ckpt_path=checkpoint_path,
    )

    # Save trained model
    config.output_model_state.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), config.output_model_state)
    logger.info(f"Model saved to {config.output_model_state}")

    return 0


def main() -> int:
    """Main entry point for training."""
    parser = create_argument_parser()
    config = parse_parameters(parser)
    return run_training(config)


def run_main():
    """Entry point for distributed training."""
    return main()


if __name__ == "__main__":
    # Set process start method for SLURM
    if os.environ.get("SLURM_JOB_ID"):
        multiprocessing.set_start_method("spawn", force=True)

    # Run training
    exit(run_main())
