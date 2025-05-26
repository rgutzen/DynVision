import logging
import multiprocessing
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from .ffcv_operations import ExtendDataTimeFFCV, ExtendLabelTimeFFCV
import torch
from torch import dtype as torch_dtype
import numpy as np
from dynvision.utils import handle_errors
from ffcv.fields.decoders import (
    IntDecoder,
    NDArrayDecoder,
    RandomResizedCropRGBImageDecoder,
    SimpleRGBImageDecoder,
)
from ffcv.loader import Loader, OrderOption
from ffcv.pipeline.allocation_query import AllocationQuery
from ffcv.pipeline.operation import Operation
from ffcv.pipeline.state import State
from ffcv.transforms import (
    Convert,
    ToDevice,
    ToTensor,
    ToTorchImage,
    NormalizeImage,
    RandomResizedCrop,
)

from dynvision.data.dataloader import _adjust_data_dimensions, _adjust_label_dimensions
from dynvision.data.transforms import get_data_transform, get_target_transform
from dynvision.utils import alias_kwargs, filter_kwargs

logger = logging.getLogger(__name__)

MAX_WORKERS = min(16, len(os.sched_getaffinity(0)) - 1)
# Constants for large dataset detection
MAX_RESOLUTION = 112
MAX_TIMESTEPS = 20
MAX_WORKERS_LARGE_DATASET = 4
MAX_BATCH_SIZE = 256


@dataclass
class DatasetParams:
    """Parameters for dataset loading with memory optimizations."""

    batch_size: int
    batches_ahead: int
    order: OrderOption
    os_cache: bool
    dtype: Optional[torch.dtype]
    num_workers: int


def _optimize_large_dataset_params(
    resolution: int,
    n_timesteps: int,
    batch_size: int,
    batches_ahead: int,
    order: OrderOption,
    os_cache: Optional[bool],
    dtype: Optional[torch.dtype],
    num_workers: int,
) -> DatasetParams:
    """Optimize loading parameters for large datasets.

    Centralizes memory optimization logic for large datasets:
    - Adjusts caching strategy (disables OS cache)
    - Uses QUASI_RANDOM ordering for better memory efficiency
    - Reduces batch prefetching
    - Sets memory-efficient data types

    Args:
        resolution: Image resolution
        n_timesteps: Number of timesteps
        batch_size: Original batch size
        batches_ahead: Number of batches to prefetch
        order: Data traversal order
        os_cache: Whether to use OS caching
        dtype: Data type for tensors

    Returns:
        DatasetParams with optimized values for large dataset handling
    """
    is_large_dataset = resolution > MAX_RESOLUTION or n_timesteps > MAX_TIMESTEPS

    if not is_large_dataset:
        return DatasetParams(
            batch_size=batch_size,
            batches_ahead=batches_ahead,
            order=order,
            os_cache=True if os_cache is None else os_cache,
            dtype=dtype,
            num_workers=num_workers,
        )

    optimized_params = DatasetParams(
        batch_size=min(batch_size, MAX_BATCH_SIZE),
        batches_ahead=min(batches_ahead, 2),
        order=OrderOption.QUASI_RANDOM if order == OrderOption.RANDOM else order,
        os_cache=(
            False if os_cache is None else os_cache
        ),  # Disable OS cache for large datasets
        dtype=torch.float16 if dtype is None else dtype,
        num_workers=min(num_workers, MAX_WORKERS_LARGE_DATASET),
    )
    if optimized_params.batch_size != batch_size:
        logger.info(
            f"Large dataset detected, reducing batch_size from {batch_size} to {optimized_params.batch_size}"
        )
    # Log optimization decisions
    if optimized_params.batches_ahead != batches_ahead:
        logger.info(
            f"Large dataset detected, reducing batches_ahead from {batches_ahead} to {optimized_params.batches_ahead}"
        )
    if optimized_params.order != order:
        logger.info(
            "Large dataset detected, switching to QUASI_RANDOM ordering for better memory efficiency"
        )
    if optimized_params.dtype != dtype and dtype is None:
        logger.info(
            "Large dataset detected, using float16 dtype for memory efficiency"
        )
    if optimized_params.num_workers != num_workers:
        logger.info(
            f"Large dataset detected, reducing worker count from {num_workers} to {optimized_params.num_workers} for better memory efficiency"
        )
    return optimized_params


def _build_image_pipeline(
    encoding: str,
    resolution: int,
    data_transform: Optional[List[Callable]],
    n_timesteps: int = 0,
    normalize: Optional[Tuple[List, List]] = None,
    dtype: Optional[torch.dtype] = torch.float16,
    device: Optional[torch.device] = None,
) -> List[Operation]:
    if encoding == "tensor":
        pipeline = [NDArrayDecoder(), ToTensor()]
    elif encoding == "image":
        # pipeline = [
        #     SimpleRGBImageDecoder(),
        #     RandomResizedCrop(scale=(1, 1), ratio=(1, 1), size=resolution),
        # ]
        pipeline = [RandomResizedCropRGBImageDecoder((resolution, resolution))]
    else:
        raise ValueError(f"Unsupported encoding type: {encoding}")

    if data_transform:
        pipeline.extend(data_transform)

    if normalize:
        pipeline.append(
            NormalizeImage(
                mean=np.array(normalize[0]) * 255,
                std=np.array(normalize[1]) * 255,
                type=float,
            )
        )

    if n_timesteps > 1:
        ## More flexible but also more expensive option to extend time dimension
        ## because more data needs to be transferred to GPU
        pipeline.append(ExtendDataTimeFFCV(n_timesteps))

    pipeline.append(ToTensor())
    pipeline.append(ToTorchImage(convert_back_int16=False))

    if dtype:
        pipeline.append(Convert(dtype))

    if device:
        pipeline.append(ToDevice(device))

    return pipeline


def _build_label_pipeline(
    target_transform: Optional[List[Callable]],
    n_timesteps: int = 0,
    device: Optional[torch.device] = None,
) -> List[Operation]:
    pipeline = [IntDecoder(), ToTensor()]

    if target_transform:
        pipeline.extend(target_transform)

    if n_timesteps > 1:
        pipeline.append(ExtendLabelTimeFFCV(n_timesteps))

    if device:
        pipeline.append(ToDevice(device))

    return pipeline


@handle_errors()
def get_ffcv_dataloader(
    path: Union[str, Path],
    batch_size: int = 1,
    n_timesteps: int = 0,
    num_workers: int = MAX_WORKERS,
    data_transform: Optional[Union[str, List[str]]] = None,
    target_transform: Optional[str] = None,
    normalize: Optional[Tuple[List, List]] = None,  # (mean, std)
    order: OrderOption = OrderOption.RANDOM,
    os_cache: Optional[bool] = None,
    encoding: str = "image",
    resolution: int = 224,
    drop_last: bool = True,
    dtype: Optional[torch.dtype] = torch.float16,
    batches_ahead: int = 3,
    train: bool = True,
    verbose: bool = False,
    device: Optional[torch.device] = None,
    distributed: bool = False,
    seed: int = 42,
    rank: Optional[int] = None,
    world_size: Optional[int] = None,
    **kwargs: Any,
) -> Loader:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file does not exist: {path}")

    # Optimize parameters for large datasets
    optimized_params = _optimize_large_dataset_params(
        resolution=resolution,
        n_timesteps=n_timesteps,
        batch_size=batch_size,
        batches_ahead=batches_ahead,
        order=order,
        os_cache=os_cache,
        dtype=dtype,
        num_workers=num_workers,
    )
    batch_size = optimized_params.batch_size
    batches_ahead = optimized_params.batches_ahead
    order = optimized_params.order
    os_cache = optimized_params.os_cache
    dtype = optimized_params.dtype

    if distributed:
        # Try to get distributed info from parameters first, then from torch.distributed
        if rank is not None and world_size is not None:
            current_rank = rank
            current_world_size = world_size
        elif torch.distributed.is_available() and torch.distributed.is_initialized():
            current_world_size = torch.distributed.get_world_size()
            current_rank = torch.distributed.get_rank()
        else:
            # Fallback to environment variables (useful in singularity containers)
            current_world_size = int(os.environ.get("WORLD_SIZE", "1"))
            current_rank = int(os.environ.get("RANK", "0"))
            logger.info(
                f"Using environment variables for distributed info: rank={current_rank}, world_size={current_world_size}"
            )

        # Adjust batch size for distributed training
        original_batch_size = batch_size
        batch_size = max(1, batch_size // current_world_size)
        logger.info(
            f"Adjusted batch size from {original_batch_size} to {batch_size} for distributed training (rank {current_rank}/{current_world_size})"
        )

        order = OrderOption.RANDOM
        os_cache = True
        drop_last = True

    data_transform = get_data_transform(data_transform) or []
    target_transform = get_target_transform(target_transform) or []

    if not train:
        encoding = "image"
        data_transform = []

    image_pipeline = _build_image_pipeline(
        encoding=encoding,
        resolution=resolution,
        data_transform=data_transform,
        device=device,
        dtype=dtype,
        n_timesteps=n_timesteps,
        normalize=normalize,
    )
    label_pipeline = _build_label_pipeline(
        target_transform=target_transform,
        n_timesteps=n_timesteps,
        device=device,
    )

    pipelines = {"image": image_pipeline, "label": label_pipeline}

    if verbose:
        logger.info(f"Image pipeline: {[type(p).__name__ for p in image_pipeline]}")
        logger.info(f"Label pipeline: {[type(p).__name__ for p in label_pipeline]}")

    logger.info(f"Creating FFCV loader from {path}")

    return Loader(
        str(path),
        batch_size=batch_size,
        num_workers=optimized_params.num_workers,
        order=order,
        pipelines=pipelines,
        os_cache=os_cache,
        drop_last=drop_last,
        batches_ahead=batches_ahead,
        distributed=distributed,
        seed=seed,
        **kwargs,
    )
