import argparse
import logging
from pathlib import Path

import PIL
import torch
from ffcv.fields import IntField, RGBImageField, TorchTensorField
from ffcv.writer import DatasetWriter
from torchvision import datasets
from torchvision.datasets.folder import IMG_EXTENSIONS

from dynvision.data.datasets import get_dataset, load_raw_data

parser = argparse.ArgumentParser(description="FFCV datasets")
parser.add_argument("--input", type=Path)
parser.add_argument("--output_train", type=Path)
parser.add_argument("--output_val", type=Path)
parser.add_argument("--data_name", type=str)
parser.add_argument("--train_ratio", type=float, default=0.8)
parser.add_argument("--max_resolution", type=int, default=224)


def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    args, unknown = parser.parse_known_args()

    # Load the torch dataset
    dataset = get_dataset(
        args.input,
        dataset_class=datasets.DatasetFolder,
        loader=load_raw_data,
        extensions=IMG_EXTENSIONS,
        data_transform=None,
        target_transform=f"{args.data_name}_all",
    )

    # split the dataset into train and validation
    data_size = len(dataset)
    train_size = int(args.train_ratio * data_size)
    val_size = data_size - train_size

    data = {}
    data["train"], data["val"] = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    data_sample = dataset[0][0]

    # Set ffcv encoding based on the data type
    if isinstance(data_sample, PIL.Image.Image):
        print("Encoding data as RGB images")
        data_shape = data_sample.size
        image_writer = RGBImageField(
            write_mode="proportion",  # Randomly compress
            compress_probability=0.25,  # Compress a random 1/4 of the dataset
            max_resolution=args.max_resolution,  # Resize anything above 256 to 256
            jpeg_quality=50,  # Use 50% quality when compressing an image using JPG,
        )
    elif isinstance(data_sample, torch.Tensor):
        print("Encoding data as torch tensors")
        data_shape = data_sample.shape
        image_writer = TorchTensorField(shape=data_shape, dtype=torch.float32)
    else:
        raise ValueError(f"Unsupported data type: {type(data_sample)}")

    for subset, output in zip(["train", "val"], [args.output_train, args.output_val]):
        writer = DatasetWriter(output, {"image": image_writer, "label": IntField()})
        writer.from_indexed_dataset(data[subset])
    return None


if __name__ == "__main__":
    main()
