"""Test a trained neural network model on a dataset.

This script handles the evaluation of trained models, including:
- Loading and preprocessing test data
- Running model inference
- Computing and saving test metrics
- Storing model responses for analysis

Example:
    $ python test_model.py --config_path configs/test_config.yaml --model_name MyModel
"""

import argparse
import logging
import multiprocessing
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from dynvision.data.dataloader import (
    get_data_loader_class,
    _adjust_data_dimensions,
    _adjust_label_dimensions,
)

from dynvision.data.datasets import get_dataset
from dynvision.project_paths import project_paths
from dynvision.utils import (
    filter_kwargs,
    parse_parameters,
    str_to_bool,
    handle_errors,
    load_model_and_weights,
)

logging.basicConfig(level=logging.INFO)
pylogger = logging.getLogger(__name__)


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Test a trained neural network model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config_path",
        type=Path,
        required=True,
        help="Path to the test configuration file",
    )
    parser.add_argument(
        "--input_model_state",
        type=Path,
        required=True,
        help="Path to saved model state",
    )
    parser.add_argument(
        "--model_name", type=str, required=True, help="Name of the model class"
    )
    parser.add_argument(
        "--dataset", type=Path, required=True, help="Path to the test dataset"
    )
    parser.add_argument(
        "--data_loader", type=str, required=True, help="Name of the data loader class"
    )
    parser.add_argument(
        "--data_transform",
        type=str,
        required=True,
        help="Name of data transform function",
    )
    parser.add_argument(
        "--target_transform", type=str, help="Name of target transform function"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for testing"
    )
    parser.add_argument(
        "--output_results", type=Path, required=True, help="Path to save test results"
    )
    parser.add_argument(
        "--output_responses",
        type=Path,
        required=True,
        help="Path to save model responses",
    )
    parser.add_argument(
        "--store_responses", type=int, help="Number of responses to store (0 for all)"
    )
    parser.add_argument(
        "--benchmark", type=str_to_bool, default=False, help="Enable benchmarking mode"
    )
    parser.add_argument(
        "--enable_progress_bar",
        type=str_to_bool,
        default=True,
        help="Show progress bar during testing",
    )
    parser.add_argument(
        "--precision", type=str, default="32", help="Numerical precision for testing"
    )
    parser.add_argument("--loss", nargs="+", type=str, help="Loss function names")
    parser.add_argument(
        "--verbose",
        type=str_to_bool,
        default=False,
        help="Print full Python traceback on errors",
    )
    return parser


def setup_data_loader(
    dataset: Any,
    data_loader_class: Any,
    batch_size: int,
    num_workers: int,
    **kwargs: Any,
) -> DataLoader:
    """Set up the data loader for testing.

    Args:
        dataset: Dataset instance
        data_loader_class: DataLoader class to use
        batch_size: Batch size for testing
        num_workers: Number of worker processes
        **kwargs: Additional arguments for the data loader

    Returns:
        DataLoader: Configured data loader instance
    """
    data_args = filter_kwargs(data_loader_class, kwargs)
    # Remove specific arguments that should not be passed
    for arg in ["dataset", "shuffle", "num_workers"]:
        data_args.pop(arg, None)

    pylogger.info(
        f"Creating DataLoader with batch_size={batch_size}, num_workers={num_workers}"
    )

    # Check if batch size might be too large
    if hasattr(dataset, "__len__"):
        total_samples = len(dataset)
        if batch_size > total_samples // 4:
            pylogger.warning(
                f"Batch size ({batch_size}) is large relative to dataset size ({total_samples}). "
                "Consider reducing batch size if memory issues occur."
            )

    return data_loader_class(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=False,  # Disable persistent workers
        prefetch_factor=None,  # Disable prefetching
        pin_memory=True,  # Keep pin_memory for GPU transfer
        **data_args,
    )


def setup_trainer(
    logger: Optional[pl.loggers.WandbLogger],
    enable_progress_bar: bool,
    precision: str,
    benchmark: bool,
) -> pl.Trainer:
    """Set up the PyTorch Lightning trainer.

    Args:
        logger: Optional WandB logger instance
        enable_progress_bar: Whether to show progress bar
        precision: Numerical precision for testing
        benchmark: Whether to enable benchmarking

    Returns:
        pl.Trainer: Configured trainer instance
    """
    return pl.Trainer(
        logger=logger,
        accelerator="auto",
        devices=1,
        strategy="auto",
        precision=precision,
        enable_progress_bar=enable_progress_bar,
        benchmark=benchmark,
    )


def save_results(
    model: torch.nn.Module, output_results: Path, output_responses: Path
) -> None:
    """Save test results and model responses.

    Args:
        model: Tested model instance
        output_results: Path to save test results
        output_responses: Path to save model responses
    """
    # Save test results
    results_df = model.get_classifier_dataframe()
    output_results.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_results, index=False)
    pylogger.info(f"Test results saved to {output_results}")

    # Save unit responses
    output_responses.parent.mkdir(parents=True, exist_ok=True)
    pylogger.info("Saving responses...")
    for layer, response in model.responses.items():
        pylogger.info(
            f"\tLayer {layer} {response.shape} "
            f"-> {response.nbytes / (1024 * 1024):.2f} MB"
        )
    torch.save(model.responses, output_responses)
    pylogger.info(f"Model responses saved to {output_responses}")


@handle_errors(verbose=False)
def run_testing(config, **kwargs) -> int:

    # Initialize logger if specified
    if hasattr(config, "logger") and config.logger:
        logger = pl.loggers.WandbLogger(
            project=project_paths.project_name,
            config=vars(config),
            tags=["test"],
        )
    else:
        logger = None

    # Log system information
    num_cpu_cores = multiprocessing.cpu_count()
    num_gpu_cores = torch.cuda.device_count()
    pylogger.info(
        f"Available compute resources: CPU={num_cpu_cores}, GPU={num_gpu_cores}"
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pylogger.info(f"Trying to use device {device}")

    # Load dataset
    pylogger.info(f"Loading dataset from {config.dataset}")
    dataset = get_dataset(
        config.dataset,
        data_name=config.data_name,
        data_transform="test",
        target_transform=config.target_transform,
        cache_size=100,  # Reduce cache size for testing
        pin_memory=True,
    )
    total_samples = len(dataset)
    pylogger.info(f"Dataset loaded with {total_samples} samples")

    # Setup data loader with minimal configuration
    config.num_workers = 0  # Use single worker for testing
    config.data_loader_class = get_data_loader_class(config.data_loader)

    # Calculate optimal batch size if too large
    if config.batch_size > total_samples // 4:
        suggested_batch_size = max(1, total_samples // 8)
        pylogger.warning(
            f"Reducing batch size from {config.batch_size} to {suggested_batch_size} "
            "to prevent memory issues"
        )
        config.batch_size = suggested_batch_size

    data_loader = setup_data_loader(**vars(config))

    inputs, label_indices, *paths = next(iter(data_loader))
    inputs = _adjust_data_dimensions(inputs)
    label_indices = _adjust_label_dimensions(label_indices)
    batch_size, n_timesteps, *input_shape = inputs.shape
    input_dims = (n_timesteps, *input_shape)
    setattr(config, "input_dims", input_dims)

    pylogger.info(f"input shape: {inputs.shape}")
    pylogger.info(
        f"pixel values in first batch: {inputs.mean():.3f} ± {inputs.std():.3f}"
    )

    # Load model and weights
    pylogger.info(f"Loading model {config.model_name} from {config.input_model_state}")
    model = load_model_and_weights(
        model_name=config.model_name,
        state_dict_path=config.input_model_state,
        config=config,
        device=device,
    )

    # Configure model for testing
    model.store_responses_on_cpu = True  # Force CPU storage for responses
    model.store_responses = min(
        config.store_responses, total_samples
    )  # Limit response storage

    torch.set_float32_matmul_precision("medium")

    # Setup trainer
    trainer = setup_trainer(
        logger=logger,
        enable_progress_bar=True,
        precision=config.precision,
        benchmark=config.benchmark,
    )

    # Log memory usage before testing
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        pylogger.info(
            f"GPU memory allocated before testing: {torch.cuda.memory_allocated() / 1e6:.2f}MB"
        )
        pylogger.info(
            f"GPU memory cached before testing: {torch.cuda.memory_reserved() / 1e6:.2f}MB"
        )

    # Test the model with additional logging and safety checks
    pylogger.info("Starting model testing...")
    try:
        pylogger.info("Initiating trainer.test() call...")

        # Clear GPU cache before testing
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            pylogger.info(
                f"Initial GPU memory: {torch.cuda.memory_allocated() / 1e6:.2f}MB"
            )

        trainer.test(model, data_loader)
        pylogger.info("trainer.test() completed successfully")

        # Log peak memory usage
        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated() / 1e6
            pylogger.info(f"Peak GPU memory usage: {peak_memory:.2f}MB")

            # Warning if memory usage is high
            if peak_memory > 1000:  # More than 1GB
                pylogger.warning(
                    "High GPU memory usage detected. Consider reducing batch_size or "
                    "store_responses if this causes issues."
                )

    except Exception as e:
        pylogger.error(f"Error during testing: {str(e)}")
        if torch.cuda.is_available():
            pylogger.error(
                f"GPU memory at error: {torch.cuda.memory_allocated() / 1e6:.2f}MB"
            )
            torch.cuda.empty_cache()  # Try to free memory
        raise
    finally:
        # Cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    pylogger.info(f"Proceeding to save results to {config.output_results}")

    # Save results
    save_results(
        model=model,
        output_results=config.output_results,
        output_responses=config.output_responses,
    )

    return 0


def main() -> int:
    """Main function to test the model."""
    parser = create_argument_parser()
    config = parse_parameters(parser)

    # Update the verbose setting for the decorator
    load_model_and_weights.__wrapped__.__defaults__ = (config.verbose,)
    run_testing.__wrapped__.__defaults__ = (config.verbose,)

    return run_testing(config)


if __name__ == "__main__":
    exit(main())
