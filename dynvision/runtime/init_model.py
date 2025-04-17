import argparse
from pathlib import Path
import re
import torch

import logging

from dynvision import models
from dynvision.data.dataloader import get_data_loader, StandardDataLoader
from dynvision.data.ffcv_dataloader import get_ffcv_dataloader

from dynvision.data.datasets import get_dataset
from dynvision.utils import (
    filter_kwargs,
    parse_parameters,
    parse_kwargs,
    set_seed,
    str2dict,
    str_to_bool,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


parser = argparse.ArgumentParser(description="Train a model on a dataset.")
parser.add_argument("--config_path", type=Path, help="Path to the config file")
parser.add_argument("--model_name", type=str, help="Model class")
parser.add_argument("--dataset", type=Path)
parser.add_argument("--data_name", type=str)
parser.add_argument("--seed", type=int, help="Random seed")
parser.add_argument("--output", type=Path)
parser.add_argument("--status", type=str)
parser.add_argument(
    "--init_with_pretrained", type=str_to_bool, help="Initialize with pretrained model"
)


def init_model(model_name, **kwargs):
    if hasattr(models, model_name):
        model_class = getattr(models, model_name)
        kwargs = filter_kwargs(model_class, kwargs)
        model = model_class(**kwargs)
        # model.setup(stage="fit")
        return model
    else:
        raise ValueError(f"Invalid model name: {model_name}")


def arg_similarity(a, b):
    a = str2dict(a, assigner="=")
    b = str2dict(b, assigner="=")
    identical_entries = sum(1 for key in a if key in b and a[key] == b[key])
    return identical_entries


def find_similar_trained_model(directory, model_name, data_name, seed, model_args=""):
    """
    Find the most similar trained model in the directory, by comparing the model arguments and seeds.
    It chooses the model with the most similar arguments, the fewest arguments, and the largest seed, in that priority order.
    """

    pattern = re.compile(
        rf"{re.escape(model_name)}(:.*?)?_(\d+)_{re.escape(data_name)}_trained\.pt$"
    )
    pretrained_candidates = []
    for candidate in directory.glob("*.pt"):
        if pattern.match(candidate.name):
            cmodel_args, cseed = pattern.match(candidate.name).groups()
            cmodel_args = cmodel_args.lstrip(":")
            pretrained_candidates.append((candidate, cmodel_args, cseed))

    if pretrained_candidates:
        best_candidate = max(
            pretrained_candidates,
            key=lambda x: (arg_similarity(x[1], model_args), len(x[1]), int(x[2])),
        )
        return best_candidate[0]
    else:
        return False


def main():
    config = parse_parameters(parser)

    set_seed(config.seed)

    if config.dataset is not None:
        # Load the dataset
        dataset = get_dataset(
            config.dataset,
            data_transform="test",
            data_name=config.data_name,
            pin_memory=torch.cuda.is_available(),
        )

        dataloader = StandardDataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )

        n_classes = len(dataset.classes)

        # Get the input dimensions
        inputs, labels, *paths = next(dataloader.__iter__())
        input_dims = inputs.shape[1:]  # exclude batch dimension

        setattr(config, "input_dims", input_dims)
        setattr(config, "n_classes", n_classes)

        logging.info(f"Input dimensions: {input_dims}")
        logging.info(f"Number of classes: {n_classes}")

    # Initialize the model
    model = init_model(
        store_responses=False,
        **vars(config),
    )

    if config.init_with_pretrained:
        if ":" in config.output.stem:
            model_args = config.output.stem.split("_")[0].split(":")[1]
        else:
            model_args = ""

        pretrained_file = find_similar_trained_model(
            directory=config.output.parent,
            model_name=config.model_name,
            data_name=config.data_name,
            seed=config.seed,
            model_args=model_args,
        )

        if pretrained_file:
            logger.info(f"Loading pretrained model from {pretrained_file}")
            model.load_state_dict(torch.load(pretrained_file, weights_only=True))

    config.output.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving model to {config.output}")
    if not len(model.state_dict()):
        raise ValueError(
            "Model state dict is empty. Please check the model initialization."
        )

    torch.save(model.state_dict(), config.output)

    return None


if __name__ == "__main__":
    main()
