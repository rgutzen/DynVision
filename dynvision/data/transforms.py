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
        # tv.transforms.RandomRotation(10),
        # tv.transforms.RandomAffine(0, translate=(0.1, 0.1)),
        tv.transforms.RandomHorizontalFlip(),
        # tv.transforms.ColorJitter(
        #     brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1
        # ),
    ],
    test=[],
    ffcv_train_mnist=[
        ffcv.transforms.RandomBrightness(0.2),
        ffcv.transforms.RandomContrast(0.2),
        ffcv.transforms.RandomSaturation(0.2),
    ],
    mnist=[
        tv.transforms.Grayscale(num_output_channels=1),
    ],
    imagenet=[
        tv.transforms.Resize(256),
        tv.transforms.CenterCrop(224),
    ],
    imagenette=[
        tv.transforms.Resize(256),
        tv.transforms.CenterCrop(224),
    ],
)


def get_data_transform(
    transform: Union[str, List[str], Dict[str, Any], None] = None,
    data_name: Optional[str] = None,
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
    transforms_to_apply = []

    # Always apply data_name specific transforms first if available
    if (
        data_name is not None
        and data_name.lower() in transform_presets.keys()
        and (not "ffcv" in str(transform))
    ):
        transforms_to_apply.extend(transform_presets[data_name.lower()])

    # Handle different transform types
    if transform is None:
        return transforms_to_apply if transforms_to_apply else None
    elif isinstance(transform, list):
        for t in transform:
            additional = get_data_transform(t)
            if additional:
                transforms_to_apply.extend(additional)
    elif isinstance(transform, dict):
        for t, k in transform.items():
            k.update(kwargs)
            additional = get_data_transform(t, **k)
            if additional:
                transforms_to_apply.extend(additional)
    elif isinstance(transform, str):
        transform = transform.lower()
        if transform in transform_presets:
            transforms_to_apply.extend(transform_presets[transform])
        elif "ffcv_train" in transform:
            transforms_to_apply.extend(transform_presets["ffcv_train"])
        elif "train" in transform:
            transforms_to_apply.extend(transform_presets["train"])
        elif "ffcv_test" in transform:
            transforms_to_apply.extend(transform_presets["ffcv_test"])
        elif "test" in transform:
            transforms_to_apply.extend(transform_presets["test"])
        else:
            raise ValueError(f"No transform with name {transform} found in presets!")

    return transforms_to_apply if transforms_to_apply else None


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
