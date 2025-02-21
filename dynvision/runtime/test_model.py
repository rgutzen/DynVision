import argparse
import logging
import multiprocessing
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch

from dynvision import models
from dynvision.data.dataloader import get_data_loader
from dynvision.data.datasets import get_dataset
from dynvision.project_paths import project_paths
from dynvision.utils.utils import (
    parse_kwargs,
    parse_string2dict,
    load_config,
    str_to_bool,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

defaults = SimpleNamespace(
    **load_config(project_paths.scripts.configs / "config_defaults.yaml")
)

parser = argparse.ArgumentParser(description="Test a model on a dataset.")
parser.add_argument("--input_model_state", type=Path, help="Path to model state")
parser.add_argument("--model_name", type=str, help="Name of model class")
parser.add_argument("--dataset", type=Path, help="Path to the dataset")
parser.add_argument("--data_loader", type=str, default=None, help="class name")
parser.add_argument("--data_transform", type=str, default=None, help="func")
parser.add_argument("--target_transform", type=str, default=None, help="func name")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
parser.add_argument("--output_results", type=Path, help="Path to save test results")
parser.add_argument("--output_responses", type=Path, help="Path to save responses")
parser.add_argument("--store_responses", type=int, default=150, help="Limit the data")
parser.add_argument(
    "--benchmark", type=str_to_bool, default=True, help="Input shape is constant?"
)
parser.add_argument(
    "--enable_progress_bar", type=str_to_bool, default=True, help="Show progress bar?"
)
parser.add_argument(
    "--precision", type=str, default="bf16-mixed", help="Weight precision"
)
parser.add_argument(
    "--loss",
    nargs="+",
    type=str,
    default=["CrossEntropyLoss"],
    help="Loss function name",
)
parser.add_argument("--loss_config", nargs="+", type=str, default=None)


def main():
    args, unknown = parser.parse_known_args()
    kwargs = parse_kwargs(unknown)

    # Load the dataset
    dataset = get_dataset(
        args.dataset,
        data_transform=f"{args.data_transform}_test",
        target_transform=args.target_transform,
    )

    # num_workers =  max(1, int(multiprocessing.cpu_count() / 2))
    num_workers = 8

    # Create the data loader
    data_loader = get_data_loader(
        dataset,
        dataloader=args.data_loader,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        **kwargs,
    )

    # Load model state
    state_dict = torch.load(args.input_model_state)
    last_key = next(reversed(state_dict))
    n_classes = len(state_dict[last_key])

    # Parse loss criterion
    criterion_params = [
        (loss, parse_string2dict(config))
        for loss, config in zip(args.loss, args.loss_config)
    ]

    # Load the model
    inputs, labels, paths = next(data_loader.__iter__())
    input_dims = inputs.shape[1:]

    model_class = getattr(models, args.model_name)
    model = model_class(
        input_dims=input_dims,
        n_classes=n_classes,
        store_responses=args.store_responses,
        criterion_params=criterion_params,
        **kwargs,
    )
    
    model.load_state_dict(state_dict)
    
    # Initialize the logger
    if defaults.logger:
        logger = pl.loggers.WandbLogger(
            project=project_paths.project_name,
            config=vars(args) | kwargs,
            tags=["test"],
        )
        # wandb.init()  # trick to log tables
    else:
        logger = None

    # # # #
    # Test the model
    trainer = pl.Trainer(
        logger=logger,
        accelerator="auto",
        devices=1,
        strategy="auto",
        precision=args.precision,
        enable_progress_bar=args.enable_progress_bar,
        benchmark=args.benchmark,
    )
    trainer.test(model, data_loader)

    # Save test results
    results_df = model.get_classifier_dataframe()

    args.output_results.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(args.output_results, index=False)

    # Save unit responses
    args.output_responses.parent.mkdir(parents=True, exist_ok=True)
    print("Saving responses ...")
    for layer, response in model.responses.items():
        print(
            f"\t Layer {layer} {response.shape} "
            + f"-> {response.nbytes / (1024 * 1024):.2f} MB"
        )
    torch.save(model.responses, args.output_responses)

    return None


if __name__ == "__main__":
    main()
