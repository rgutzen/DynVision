import argparse
from pathlib import Path
import re

import torch

from dynvision import models
from dynvision.data.dataloader import (
    get_data_loader,
    get_ffcv_dataloader,
)
from dynvision.data.datasets import get_dataset
from dynvision.utils.utils import (
    filter_kwargs,
    parse_kwargs,
    set_seed,
    str2dict,
    str_to_bool,
)
from difflib import SequenceMatcher

parser = argparse.ArgumentParser(description="Train a model on a dataset.")
parser.add_argument("--model_name", type=str, help="Model class")
parser.add_argument("--dataset", type=Path, default=None)
parser.add_argument("--data_name", type=str)
parser.add_argument("--seed", type=int, default=0, help="Random seed")
parser.add_argument("--output", type=Path)
parser.add_argument("--init_with_pretrained", type=str_to_bool, default=False)


def init_model(model_name, **kwargs):
    if hasattr(models, model_name):
        model_class = getattr(models, model_name)
        # kwargs = filter_kwargs(model_class, kwargs)
        model = model_class(**kwargs)
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
    args, unkwown = parser.parse_known_args()
    kwargs = parse_kwargs(unkwown)

    set_seed(args.seed)

    if args.dataset is not None:
        # Load the dataset
        dataset = get_dataset(args.dataset, data_transform=f"{args.data_name}_test")
        # dataloader = get_data_loader(dataset, batch_size=2)  # batch_size=2 for testing
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=2, shuffle=True, num_workers=8, pin_memory=True
        )

        n_classes = len(dataset.classes)

        # Get the input dimensions
        inputs, labels, *paths = next(dataloader.__iter__())
        input_dims = inputs.shape[1:]  # exclude batch dimension
        
        kwargs.update({"input_dims": input_dims, "n_classes": n_classes})

    # Initialize the model
    model = init_model(
        args.model_name,
        store_responses=False,
        **kwargs,
    )

    if args.init_with_pretrained:
        if ":" in args.output.stem:
            model_args = args.output.stem.split("_")[0].split(":")[1]
        else:
            model_args = ""

        pretrained_file = find_similar_trained_model(
            directory=args.output.parent,
            model_name=args.model_name,
            data_name=args.data_name,
            seed=args.seed,
            model_args=model_args,
        )

        if pretrained_file:
            print(f"Loading pretrained model from {pretrained_file}")
            model.load_state_dict(torch.load(pretrained_file, weights_only=True))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.output)

    return None


if __name__ == "__main__":
    main()
