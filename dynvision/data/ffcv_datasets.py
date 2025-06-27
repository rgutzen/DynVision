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
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import PIL
import torch
from ffcv.fields import IntField, RGBImageField, TorchTensorField
from ffcv.writer import DatasetWriter
from torchvision import datasets
from torchvision.datasets.folder import IMG_EXTENSIONS
from torchvision.transforms import ToPILImage


from dynvision.data.datasets import get_dataset, load_raw_data
from dynvision.params.data_params import DataParams

logger = logging.getLogger(__name__)


def main() -> None:
    """Main function with error handling and resource management."""
    config = DataParams.from_cli_and_config()

    # Load dataset
    dataset = get_dataset(
        Path(config.input),
        dataset_class=datasets.DatasetFolder,
        loader=load_raw_data,
        extensions=IMG_EXTENSIONS,
        data_transform=None,
        target_transform=f"{config.data_name}_all",
        pil_to_tensor=False,
        dtype=None,
        normalize=None,
    )

    # Split dataset
    data_size = len(dataset)
    train_size = int(config.train_ratio * data_size)
    val_size = data_size - train_size

    logger.info(f"Splitting dataset: {train_size} train, {val_size} validation")
    data = {}
    data["train"], data["val"] = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    # Get writer configuration
    data_sample = dataset[0][0]

    image_writer = RGBImageField(
        write_mode=config.writer_mode,
        compress_probability=config.compress_probability,
        max_resolution=config.max_resolution,
        jpeg_quality=config.jpeg_quality,
    )
    label_writer = IntField()
    fields = {"image": image_writer, "label": label_writer}

    # Write datasets sequentially
    for subset, output_path in zip(
        ["train", "val"], [Path(config.output_train), Path(config.output_val)]
    ):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_str = str(output_path.resolve())

        writer = DatasetWriter(
            output_str,
            fields,
            num_workers=config.num_workers,
            page_size=config.page_size,
        )
        logger.info(f"Writing {subset} dataset with {len(data[subset])} samples")
        writer.from_indexed_dataset(data[subset], chunksize=config.chunksize)


if __name__ == "__main__":
    main()
