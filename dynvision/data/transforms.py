import numpy as np
import torch
import torchvision as tv
import yaml
from ffcv import transforms

from dynvision.project_paths import project_paths


transform_sets = dict(
    none=[],
    mnist_train=[
        tv.transforms.RandomRotation(10),
        tv.transforms.RandomResizedCrop(28, scale=(0.8, 1.2)),
        # transforms.Pad(padding=(28 // 2, 28 // 2), fill=0),
        tv.transforms.RandomAffine(0, translate=(0.1, 0.1)),
        tv.transforms.Grayscale(num_output_channels=1),
        tv.transforms.PILToTensor(),
        tv.transforms.ConvertImageDtype(torch.float16),
        tv.transforms.Normalize((0.1307,), (0.3081,)),
    ],
    mnist_test=[
        tv.transforms.Grayscale(num_output_channels=1),
        tv.transforms.PILToTensor(),
        tv.transforms.ConvertImageDtype(torch.float16),
        tv.transforms.Normalize((0.1307,), (0.3081,)),
    ],
    mnist_train_ffcv=[
        # transforms.RandomResizedCrop(scale=(0.9, 1.1), ratio=(1, 1), size=28),
        transforms.ToTensor(),
        transforms.ToTorchImage(convert_back_int16=False),
        transforms.Convert(torch.float16),
        # tv.transforms.RandomRotation(10),
        # tv.transforms.RandomAffine(0, translate=(0.1, 0.1)),
        tv.transforms.Grayscale(num_output_channels=1),
        tv.transforms.Normalize(
            mean=np.array([0.1307]) * 255,
            std=np.array([0.3081]) * 255,
        ),
    ],
    mnist_test_ffcv=[
        # transforms.RandomResizedCrop(scale=(1.0, 1.0), ratio=(1, 1), size=28),
        transforms.ToTensor(),
        transforms.ToTorchImage(convert_back_int16=False),
        transforms.Convert(torch.float16),
        tv.transforms.Grayscale(num_output_channels=1),
        tv.transforms.Normalize(
            mean=np.array([0.1307]) * 255,
            std=np.array([0.3081]) * 255,
        ),
    ],
    imagenet_train=[
        # Randomly crop the image to 224x224
        tv.transforms.RandomResizedCrop(224),
        # Randomly flip the image horizontally
        tv.transforms.RandomHorizontalFlip(),
        # Randomly adjust the brightness, contrast, saturation, and hue
        tv.transforms.ColorJitter(
            brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1
        ),
        # Randomly rotate the image by up to 10 degrees
        # tv.transforms.RandomRotation(10),
        tv.transforms.PILToTensor(),
        tv.transforms.ConvertImageDtype(torch.float),
        tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ],
    imagenet_train_ffcv=[
        transforms.RandomHorizontalFlip(),
        # transforms.RandomBrightness(0.2),
        # transforms.RandomContrast(0.2),
        # transforms.RandomSaturation(0.2),
        # transforms.Convert(torch.float16),
        transforms.ToTensor(),
        transforms.ToTorchImage(convert_back_int16=False),
        # tv.transforms.Normalize(mean=np.array([0.480, 0.448, 0.397])*255,
        #                         std=np.array([0.277, 0.269, 0.282])*255),
        transforms.NormalizeImage(
            mean=np.array([0.480, 0.448, 0.397]) * 255,
            std=np.array([0.277, 0.269, 0.282]) * 255,
            type=np.float16,
        ),
    ],
    imagenet_test=[
        tv.transforms.CenterCrop(224),
        tv.transforms.PILToTensor(),
        tv.transforms.ConvertImageDtype(torch.float),
        tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ],
    imagenet_test_ffcv=[
        transforms.NormalizeImage(
            mean=np.array([0.485, 0.456, 0.406]) * 255,
            std=np.array([0.229, 0.224, 0.225]) * 255,
            type=np.float16,
        ),
    ],
    tinyimagenet_train=[
        # transforms.ToTensor(),
        tv.transforms.RandomResizedCrop(64),
        tv.transforms.RandomHorizontalFlip(p=0.5),
        tv.transforms.ColorJitter(
            brightness=0.15, contrast=0.15, saturation=0.15, hue=0.1
        ),
        # tv.transforms.RandomRotation(15),
        # tv.transforms.RandomErasing(p=0.5, scale=(0.02, 0.1), ratio=(0.3, 3.3))
        # tv.transforms.RandomCrop(64, padding=4),
        tv.transforms.PILToTensor(),
        tv.transforms.ConvertImageDtype(torch.float),
        tv.transforms.Normalize(mean=[0.480, 0.448, 0.397], std=[0.277, 0.269, 0.282]),
    ],
    tinyimagenet_train_ffcv=[
        transforms.RandomResizedCrop(scale=(0.9, 1.0), ratio=(1, 1), size=64),
        transforms.RandomHorizontalFlip(),
        transforms.RandomBrightness(0.2),
        transforms.RandomContrast(0.2),
        transforms.RandomSaturation(0.2),
        transforms.ToTensor(),
        transforms.ToTorchImage(convert_back_int16=False),
        transforms.Convert(torch.float16),
        tv.transforms.Normalize(
            mean=np.array([0.480, 0.448, 0.397]) * 255,
            std=np.array([0.277, 0.269, 0.282]) * 255,
        ),
        # transforms.NormalizeImage(mean=np.array([0.480, 0.448, 0.397])*255,
        #                           std=np.array([0.277, 0.269, 0.282])*255, type=np.float16),
    ],
    tinyimagenet_test=[
        tv.transforms.CenterCrop(64),
        tv.transforms.PILToTensor(),
        tv.transforms.ConvertImageDtype(torch.float),
        tv.transforms.Normalize(mean=[0.480, 0.448, 0.397], std=[0.277, 0.269, 0.282]),
    ],
    tinyimagenet_test_ffcv=[
        transforms.RandomResizedCrop(scale=(1.0, 1.0), ratio=(1, 1), size=64),
        transforms.ToTensor(),
        transforms.ToTorchImage(convert_back_int16=False),
        transforms.Convert(torch.float16),
        tv.transforms.Normalize(
            mean=np.array([0.480, 0.448, 0.397]) * 255,
            std=np.array([0.277, 0.269, 0.282]) * 255,
        ),
    ],
    cifar10_train=[
        tv.transforms.RandomCrop(32, padding=4),
        tv.transforms.RandomHorizontalFlip(),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ],
    cifar10_test=[
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ],
    cifar10_train_ffcv=[
        transforms.RandomResizedCrop(scale=(0.9, 1.0), ratio=(1, 1), size=32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.ToTorchImage(convert_back_int16=False),
        transforms.Convert(torch.float16),
        tv.transforms.Normalize(
            mean=np.array([0.5071, 0.4867, 0.4408]) * 255,
            std=np.array([0.2675, 0.2565, 0.2761]) * 255,
        ),
    ],
    cifar10_test_ffcv=[
        transforms.RandomResizedCrop(scale=(1.0, 1.0), ratio=(1, 1), size=32),
        transforms.ToTensor(),
        transforms.ToTorchImage(convert_back_int16=False),
        transforms.Convert(torch.float16),
        tv.transforms.Normalize(
            mean=np.array([0.5071, 0.4867, 0.4408]) * 255,
            std=np.array([0.2675, 0.2565, 0.2761]) * 255,
        ),
    ],
    cifar100_train_ffcv=[
        transforms.RandomResizedCrop(scale=(0.9, 1.0), ratio=(1, 1), size=32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.ToTorchImage(convert_back_int16=False),
        transforms.Convert(torch.float16),
        tv.transforms.Normalize(
            mean=np.array([0.5071, 0.4867, 0.4408]) * 255,
            std=np.array([0.2675, 0.2565, 0.2761]) * 255,
        ),
    ],
    cifar100_test_ffcv=[
        transforms.RandomResizedCrop(scale=(1.0, 1.0), ratio=(1, 1), size=32),
        transforms.ToTensor(),
        transforms.ToTorchImage(convert_back_int16=False),
        transforms.Convert(torch.float16),
        tv.transforms.Normalize(
            mean=np.array([0.5071, 0.4867, 0.4408]) * 255,
            std=np.array([0.2675, 0.2565, 0.2761]) * 255,
        ),
    ],
    cifar100_train=[
        # transforms.ToTensor(),
        tv.transforms.RandomResizedCrop(32),
        tv.transforms.RandomHorizontalFlip(p=0.5),
        # tv.transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.1),
        # tv.transforms.RandomRotation(15),
        # tv.transforms.RandomErasing(p=0.5, scale=(0.02, 0.1), ratio=(0.3, 3.3))
        # tv.transforms.RandomCrop(32, padding=4),
        tv.transforms.PILToTensor(),
        tv.transforms.ConvertImageDtype(torch.float16),
        tv.transforms.Normalize(
            mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]
        ),
    ],
    cifar100_test=[
        tv.transforms.CenterCrop(32),
        tv.transforms.PILToTensor(),
        tv.transforms.ConvertImageDtype(torch.float16),
        tv.transforms.Normalize(
            mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]
        ),
    ],
    totensor=[
        # tv.transforms.ConvertImageDtype(torch.float16),
    ],
)


class IndexToLabel(torch.nn.Module):
    def __init__(self, data_name, data_group=None):
        super(IndexToLabel, self).__init__()

        with open(project_paths.scripts.configs / "config_data.yaml", "r") as file:
            data_groups = yaml.safe_load(file)["data_groups"]

        self.data_group_labels = data_groups[data_name][data_group]

    def _index_to_label(self, index: int) -> int:
        return int(self.data_group_labels[index])

    def forward(self, x):
        x = self._index_to_label(x)
        return x


def get_data_transform(transform: str, **kwargs):
    out = []

    if transform is None:
        return None
    elif isinstance(transform, list):
        for t in transform:
            out += get_data_transform(t)
    elif isinstance(transform, dict):
        for t, k in transform.items():
            out += get_data_transform(t, **k)
    elif isinstance(transform, str):
        if transform.lower() in transform_sets:
            out += transform_sets[transform.lower()]
        elif hasattr(transforms, transform):
            out += [getattr(transforms, transform)(**kwargs)]
    elif isinstance(transform, tv.transforms.transforms.Compose):
        return transform
    else:
        raise ValueError(f"Invalid data transform: {transform}")

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
