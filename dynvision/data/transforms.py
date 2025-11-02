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

logger = logging.getLogger(__name__)

transform_presets = dict(
    ffcv_train=[
        ffcv.transforms.RandomHorizontalFlip(),
        ffcv.transforms.RandomBrightness(0.2),
        ffcv.transforms.RandomContrast(0.2),
        ffcv.transforms.RandomSaturation(0.2),
        ffcv.transforms.RandomTranslate(padding=22, fill=(0, 0, 0)),
    ],
    ffcv_test=[],
    train=[
        tv.transforms.RandomRotation(10),
        tv.transforms.RandomAffine(0, translate=(0.1, 0.1)),
        tv.transforms.RandomHorizontalFlip(),
        tv.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    ],
    test=[],
    ffcv_train_mnist=[  # deprecated!
        ffcv.transforms.RandomBrightness(0.2),
        ffcv.transforms.RandomContrast(0.2),
        ffcv.transforms.RandomSaturation(0.2),
    ],
    train_mnist=[
        tv.transforms.RandomRotation(10),
        tv.transforms.RandomAffine(0, translate=(0.1, 0.1)),
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
    """Get data transforms for image preprocessing.

    Processes transform specifications to create a list of transformations
    to apply to input data. Handles various transform specification formats
    and supports preset transform configurations.

    Args:
        transform: Transform specification that can be:
            - None: Uses data_name specific transforms if available
            - list: Processes each item recursively
            - dict: Processes each key-value pair recursively, merging kwargs
            - str: Looks up transform in presets or infers from names containing
                  'train' or 'test'
            - other: Attempts to append the transform directly
        data_name: Name of the dataset to apply specific preset transforms
            (e.g., 'imagenet', 'mnist') if available
        **kwargs: Additional parameters passed to recursive calls or used for
            transform configuration

    Returns:
        List of transforms or None if no transforms are specified

    Raises:
        ValueError: If string transform name is not found in presets
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
        else:
            for name in transform_presets.keys():
                if name in transform:
                    transforms_to_apply.extend(transform_presets[name])
                    break
        if transforms_to_apply == []:
            raise ValueError(f"No transform with name {transform} found in presets!")
    else:
        try:
            transforms_to_apply.append(transform)
        except Exception as e:
            logger.error(f"Error applying transform {transform}: {e}")

    return transforms_to_apply if transforms_to_apply else None


def get_target_transform(transform: str, **kwargs):
    """Get target (label) transforms for dataset preprocessing.

    Processes transform specifications to create a list of transformations
    to apply to target labels. Support for various transform specification formats,
    including strings, lists, and dictionaries.

    Args:
        transform: Transform specification that can be:
            - None: Returns None
            - list: Processes each item recursively
            - dict: Processes each key-value pair recursively
            - str: Expected format "dataset_datagroup", creates IndexToLabel transformer
                   if datagroup is not "all"
        **kwargs: Additional parameters passed to recursive calls or transformers

    Returns:
        List of target transforms or None if no transforms are specified

    Raises:
        ValueError: If string transform doesn't follow the expected "dataset_datagroup" format
    """
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
