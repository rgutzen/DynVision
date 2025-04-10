"""Data transforms.

Usage:
    from dynvision.data.transforms import get_data_transform
    
    transform = get_data_transform(
        'ffcv_train',
    )
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import torchvision as tv
import ffcv
from .operations import IndexToLabel

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

transform_presets = dict(
    ffcv_train=[
        ffcv.transforms.RandomHorizontalFlip(),
        ffcv.transforms.RandomBrightness(0.2),
        ffcv.transforms.RandomContrast(0.2),
        ffcv.transforms.RandomSaturation(0.2),
    ],
    ffcv_test=[],
    train=[
        tv.transforms.RandomRotation(10),
        tv.transforms.RandomAffine(0, translate=(0.1, 0.1)),
        tv.transforms.RandomHorizontalFlip(),
        tv.transforms.ColorJitter(
            brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1
        ),
    ],
    test=[],
    ffcv_train_mnist=[
        ffcv.transforms.RandomBrightness(0.2),
        ffcv.transforms.RandomContrast(0.2),
        ffcv.transforms.RandomSaturation(0.2),
    ],
    train_mnist=[
        tv.transforms.RandomRotation(10),
        tv.transforms.RandomAffine(0, translate=(0.1, 0.1)),
        tv.transforms.RandomHorizontalFlip(),
        tv.transforms.ColorJitter(
            brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1
        ),
        tv.transforms.Grayscale(num_output_channels=1),
    ],
    test_mnist=[
        tv.transforms.Grayscale(num_output_channels=1),
    ],
)


def get_data_transform(
    transform: Union[str, List[str], Dict[str, Any], None] = None,
    **kwargs: Any,
) -> Optional[List[Any]]:
    """Get data transforms.

    Args:
        transform: Transform specification
        device: Device to use for transforms
        memory_format: Memory format for tensors
        dtype: Data type for mixed precision
        **kwargs: Additional transform arguments

    Returns:
        List of transforms or None

    Raises:
        ValueError: If transform specification is invalid
    """
    if transform is None:
        return None

    out = []

    # Handle different transform types
    if isinstance(transform, list):
        for t in transform:
            out.extend(get_data_transform(t))

    elif isinstance(transform, dict):
        for t, k in transform.items():
            k.update(kwargs)
            out.extend(get_data_transform(t, **k))

    elif isinstance(transform, str):
        transform = transform.lower()
        if transform in transform_presets.keys():
            pass
        elif "ffcv_train" in transform:
            transform = "ffcv_train"
        elif "train" in transform:
            transform = "train"
        elif "ffcv_test" in transform:
            transform = "ffcv_test"
        elif "test" in transform:
            transform = "test"
        else:
            ValueError(f"No transform with name {transform} found in presets!")

        out.extend(transform_presets[transform])

    return out


def get_target_transform(transform: str, **kwargs):
    out = []

    if transform is None:
        return None
    elif isinstance(transform, list):
        for t in transform:
            out += get_target_transform(t, **kwargs)
    elif isinstance(transform, dict):
        for t, k in transform.items():
            out += get_target_transform(t, **k)
    elif isinstance(transform, str):
        transform = transform.replace("imagenette", "imagenet")  # hack

        if len(transform.lower().split("_")) != 2:
            raise ValueError(
                f"Expect target transform name as 'dataset_datagroup', got {transform}"
            )
        data_name, data_group = transform.lower().split("_")

        if data_group != "all":
            out += [IndexToLabel(data_name, data_group)]
    else:
        pass

    return out
