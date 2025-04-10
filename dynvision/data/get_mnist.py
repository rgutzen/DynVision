import argparse
import logging
from pathlib import Path

from torchvision import datasets
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Download MNIST dataset")
parser.add_argument("--output", type=Path)
parser.add_argument("--raw_data_path", type=Path)
parser.add_argument("--subset", type=str)
parser.add_argument("--ext", type=str, default="png")


def download_mnist(output_folder, train):
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    logging.info(f"Downloading MNIST dataset to {output_folder}...")
    dataset = datasets.MNIST(root=output_folder, train=train, download=True)

    logging.info("Download complete.")
    return dataset


def store_mnist_images(image_folder, dataset, ext="png"):
    # Create a new folder to store the images
    image_folder.mkdir(parents=True, exist_ok=True)

    logging.info(f"Saving images to: {image_folder}")
    # Iterate through the MNIST dataset and save the images to the new folder
    for i, (image, label) in tqdm(enumerate(dataset), total=len(dataset)):
        label_folder = image_folder / str(label)
        label_folder.mkdir(parents=True, exist_ok=True)

        image_path = label_folder / f"{i}.{ext}"
        image.save(image_path)

    return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    args = parser.parse_args()

    if args.subset in ["train", "test"]:
        train = args.subset == "train"
    else:
        raise ValueError(f"Invalid subset: {args.subset}")

    logger.info("loading MNIST data from local directory")

    dataset = download_mnist(output_folder=args.raw_data_path, train=train)

    if args.output.suffix:  # If the output is a file
        args.output = args.output.parent

    store_mnist_images(args.output, dataset, ext=args.ext)
