import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.datasets.folder import IMG_EXTENSIONS, default_loader

from dynvision.data.transforms import (
    get_data_transform,
    get_target_transform,
)

TENSOR_EXTENSIONS = (".pt", ".pth")
EXTENSIONS = IMG_EXTENSIONS + TENSOR_EXTENSIONS


def load_raw_data(path: str) -> Any:
    if Path(path).suffix.lower() in IMG_EXTENSIONS:
        return default_loader(path)
    elif Path(path).suffix.lower() in TENSOR_EXTENSIONS:
        return torch.load(path).float()
    else:
        raise ValueError(f"Unsupported file extension: {Path(path).suffix}")


class PathFolder(datasets.DatasetFolder):
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = load_raw_data,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> None:

        super().__init__(
            root=root,
            loader=loader,
            extensions=EXTENSIONS if is_valid_file is None else None,
            transform=transform,
            target_transform=target_transform,
            is_valid_file=is_valid_file,
        )

    def target_str_to_list(self, target: str) -> List[int]:
        return [int(t) if t.isdigit() else -1 for t in target]

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target, path) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)

        if self.transform is None:
            sample = transforms.ToTensor()(sample)
        else:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        if len(sample.shape) == 3:  # image input
            pass
        elif len(sample.shape) == 4:  # video input
            # target = torch.tensor(self.target_str_to_list(target)) # undo
            if sample.shape[0] != len(target):
                raise ValueError(
                    f"Sample and target have different lengths: {sample.shape[0]} != {len(target)}"
                )
        else:
            raise ValueError(f"Invalid sample shape: {sample.shape}")

        return sample, target, path


def insert_a_before_b(transform_list, a=transforms.ToTensor(), b=transforms.Normalize):
    if isinstance(transform_list, list):
        for i, transform in enumerate(transform_list):
            if isinstance(transform, b):
                transform_list.insert(i, a)
                break
    return transform_list


def get_dataset(
    data_path: Path = None,
    data_transform=None,
    target_transform=None,
    dataset_class=PathFolder,
    **kwargs,
) -> Dataset:
    # set data_path to the parent directory if it is a file
    if data_path.is_file():
        if data_path.with_suffix("").exists():
            data_path = data_path.with_suffix("")
        else:
            data_path = data_path.parent

    contains_folders = any(f.is_dir() for f in data_path.iterdir())
    if not contains_folders:
        warnings.warn(
            f"{data_path} contains no category folders! Changing to parent directory."
        )
        data_path = data_path.parent

    # get transforms
    data_transform = get_data_transform(data_transform)
    target_transform = get_target_transform(target_transform)

    if isinstance(data_transform, list):
        data_transform = transforms.Compose(data_transform)
    if isinstance(target_transform, list):
        target_transform = transforms.Compose(target_transform)

    return dataset_class(
        root=data_path,
        transform=data_transform,
        target_transform=target_transform,
        **kwargs,
    )
