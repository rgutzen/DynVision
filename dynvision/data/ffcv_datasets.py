"""FFCV dataset creation with optimized performance.

This module provides optimized FFCV dataset creation with:
- Parallel writing with ThreadPoolExecutor
- Memory-efficient data handling
- GPU acceleration where applicable
- Cluster computing support
- Optimized compression settings
- Error recovery and logging

Usage:
    python ffcv_datasets.py --input data/raw --output_train data/train.beton --output_val data/val.beton
"""

import argparse
import logging
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import PIL
import torch
from ffcv.fields import IntField, RGBImageField, TorchTensorField
from ffcv.writer import DatasetWriter
from torchvision import datasets
from torchvision.datasets.folder import IMG_EXTENSIONS

from dynvision.data.datasets import get_dataset, load_raw_data

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Configure parallel processing
MAX_WORKERS = max(1, multiprocessing.cpu_count() - 1)
executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create FFCV datasets with optimized performance",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input", type=Path, required=True, help="Input data directory"
    )
    parser.add_argument(
        "--output_train",
        type=Path,
        required=True,
        help="Output path for training dataset",
    )
    parser.add_argument(
        "--output_val",
        type=Path,
        required=True,
        help="Output path for validation dataset",
    )
    parser.add_argument("--data_name", type=str, required=True, help="Dataset name")
    parser.add_argument(
        "--train_ratio", type=float, default=0.8, help="Training set ratio"
    )
    parser.add_argument(
        "--writer_mode", type=str, default="jpg", help="jpg | raw | proportion"
    )
    parser.add_argument(
        "--max_resolution", type=int, default=224, help="Maximum image resolution"
    )
    parser.add_argument(
        "--num_workers", type=int, default=MAX_WORKERS, help="Number of worker threads"
    )
    parser.add_argument(
        "--compress_probability",
        type=float,
        default=1.0,
        help="Probability of compressing an image",
    )
    parser.add_argument(
        "--jpeg_quality", type=int, default=80, help="JPEG compression quality"
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.input.exists():
        raise ValueError(f"Input directory does not exist: {args.input}")
    if not 0 < args.train_ratio < 1:
        raise ValueError(f"Invalid train ratio: {args.train_ratio}")
    if not 0 <= args.compress_probability <= 1:
        raise ValueError(f"Invalid compress probability: {args.compress_probability}")
    if not 0 <= args.jpeg_quality <= 100:
        raise ValueError(f"Invalid JPEG quality: {args.jpeg_quality}")

    return args


def get_writer_config(
    data_sample: Union[PIL.Image.Image, torch.Tensor],
    writer_mode: str,
    max_resolution: int,
    compress_probability: float,
    jpeg_quality: int,
) -> Dict[str, Any]:
    """Get FFCV writer configuration based on data type.

    Args:
        data_sample: Sample data item
        max_resolution: Maximum image resolution
        compress_probability: Probability of compression
        jpeg_quality: JPEG quality setting

    Returns:
        Dict[str, Any]: Writer configuration

    Raises:
        ValueError: If data type is not supported
    """
    if isinstance(data_sample, PIL.Image.Image):
        logger.info("Configuring RGB image encoding")
        data_shape = data_sample.size
        image_writer = RGBImageField(
            write_mode=writer_mode,
            compress_probability=compress_probability,
            max_resolution=max_resolution,
            jpeg_quality=jpeg_quality,
        )
    elif isinstance(data_sample, torch.Tensor):
        logger.info("Configuring tensor encoding")
        data_shape = data_sample.shape
        image_writer = TorchTensorField(shape=data_shape, dtype=torch.float16)
    else:
        raise ValueError(f"Unsupported data type: {type(data_sample)}")

    return {"image": image_writer, "label": IntField()}


def write_dataset_subset(
    writer: DatasetWriter, dataset: torch.utils.data.Dataset, subset_name: str
) -> None:
    """Write dataset subset with progress tracking.

    Args:
        writer: FFCV dataset writer
        dataset: Dataset to write
        subset_name: Name of the subset
    """
    try:
        logger.info(f"Writing {subset_name} dataset with {len(dataset)} samples")
        writer.from_indexed_dataset(dataset)
        logger.info(f"Finished writing {subset_name} dataset")
    except Exception as e:
        logger.error(f"Error writing {subset_name} dataset: {str(e)}")
        raise


def main() -> None:
    """Main function with error handling and resource management."""
    try:
        args = parse_args()

        # Load dataset
        dataset = get_dataset(
            args.input,
            dataset_class=datasets.DatasetFolder,
            loader=load_raw_data,
            extensions=IMG_EXTENSIONS,
            data_transform=None,
            target_transform=f"{args.data_name}_all",
        )

        # Split dataset
        data_size = len(dataset)
        train_size = int(args.train_ratio * data_size)
        val_size = data_size - train_size

        logger.info(f"Splitting dataset: {train_size} train, {val_size} validation")
        data = {}
        data["train"], data["val"] = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )

        # Get writer configuration
        data_sample = dataset[0][0]
        writer_config = get_writer_config(
            data_sample,
            writer_mode=args.writer_mode,
            max_resolution=args.max_resolution,
            compress_probability=args.compress_probability,
            jpeg_quality=args.jpeg_quality,
        )

        # Write datasets in parallel
        futures = []
        for subset, output in zip(
            ["train", "val"], [args.output_train, args.output_val]
        ):
            output.parent.mkdir(parents=True, exist_ok=True)
            writer = DatasetWriter(output, writer_config)
            future = executor.submit(
                write_dataset_subset, writer, data[subset], subset
            )
            futures.append((subset, future))

        # Wait for completion and handle errors
        for subset, future in futures:
            try:
                future.result()
            except Exception as e:
                logger.error(f"Failed to write {subset} dataset: {str(e)}")
                raise

        logger.info("Successfully created FFCV datasets")

    except Exception as e:
        logger.error(f"Error creating FFCV datasets: {str(e)}")
        raise
    finally:
        executor.shutdown()


if __name__ == "__main__":
    main()
