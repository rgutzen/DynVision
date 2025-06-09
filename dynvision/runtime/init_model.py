"""
Initialize a neural network model with type-safe parameter management.

This script creates and saves initialized models with automatic dimension inference
from datasets and optional pretrained weight loading.

Example:
    $ python init_model.py --model_name DyRCNNx4 --data_name cifar100 --output model.pt
"""

import logging
import re
from pathlib import Path
from typing import Optional

import torch

from dynvision import models
from dynvision.data.dataloader import StandardDataLoader
from dynvision.data.datasets import get_dataset
from dynvision.hyperparameters.init_params import InitParams
from dynvision.utils import filter_kwargs, set_seed, str2dict, handle_errors

logger = logging.getLogger(__name__)


def find_similar_trained_model(
    directory: Path, model_name: str, data_name: str, seed: int, model_args: str = ""
) -> Optional[Path]:
    """
    Find the most similar trained model in the directory by comparing model arguments and seeds.

    Chooses the model with the most similar arguments, fewest arguments, and largest seed,
    in that priority order.

    Args:
        directory: Directory to search for trained models
        model_name: Name of the model architecture
        data_name: Name of the dataset
        seed: Random seed
        model_args: Model arguments string for comparison

    Returns:
        Path to the most similar trained model, or None if none found
    """
    pattern = re.compile(
        rf"{re.escape(model_name)}(:.*?)?_(\d+)_{re.escape(data_name)}_trained\.pt$"
    )

    pretrained_candidates = []
    for candidate in directory.glob("*.pt"):
        match = pattern.match(candidate.name)
        if match:
            cmodel_args, cseed = match.groups()
            cmodel_args = cmodel_args.lstrip(":") if cmodel_args else ""
            pretrained_candidates.append((candidate, cmodel_args, cseed))

    if pretrained_candidates:
        best_candidate = max(
            pretrained_candidates,
            key=lambda x: (arg_similarity(x[1], model_args), -len(x[1]), int(x[2])),
        )
        return best_candidate[0]

    return None


def arg_similarity(a: str, b: str) -> int:
    """Calculate similarity between two argument strings."""
    a_dict = str2dict(a, assigner="=") if a else {}
    b_dict = str2dict(b, assigner="=") if b else {}
    return sum(1 for key in a_dict if key in b_dict and a_dict[key] == b_dict[key])


def infer_dimensions_from_dataset(config: InitParams) -> None:
    """
    Infer model dimensions from dataset and update configuration.

    Args:
        config: InitParams instance to update
    """
    if not config.dataset or not config.dataset.exists():
        logger.warning(
            "No dataset provided or dataset not found. Using config defaults."
        )
        return

    try:
        # Load dataset for dimension inference
        dataset_kwargs = config.get_dataset_kwargs()
        dataset = get_dataset(config.dataset, **dataset_kwargs)

        # Create simple dataloader
        dataloader_kwargs = config.get_dataloader_kwargs()
        dataloader = StandardDataLoader(dataset, **dataloader_kwargs)

        # Get sample batch
        inputs, labels, *paths = next(iter(dataloader))
        input_dims = inputs.shape[1:]  # Exclude batch dimension
        n_classes = len(dataset.classes)

        logger.info(f"Inferred from dataset:")
        logger.info(f"  - Input dimensions: {input_dims}")
        logger.info(f"  - Number of classes: {n_classes}")

        # Update model configuration
        config.update_model_parameters_from_dataset(
            input_dims=input_dims, n_classes=n_classes, validate_consistency=True
        )

    except Exception as e:
        logger.warning(f"Failed to infer dimensions from dataset: {e}")
        logger.warning("Continuing with configuration defaults")


def create_and_initialize_model(config: InitParams) -> torch.nn.Module:
    """
    Create and initialize model with configuration.

    Args:
        config: InitParams instance with model configuration

    Returns:
        Initialized model
    """
    # Get model class
    if not hasattr(models, config.model.model_name):
        raise ValueError(f"Invalid model name: {config.model.model_name}")

    model_class = getattr(models, config.model.model_name)

    # Get filtered kwargs for model creation
    model_kwargs = config.get_model_creation_kwargs(model_class)

    logger.info(f"Creating {config.model.model_name} with:")
    logger.info(f"  - Input dims: {model_kwargs.get('input_dims')}")
    logger.info(f"  - N classes: {model_kwargs.get('n_classes')}")
    logger.info(f"  - N timesteps: {model_kwargs.get('n_timesteps')}")

    # Create model
    model = model_class(**model_kwargs)

    # Handle pretrained initialization
    if config.init_with_pretrained:
        load_pretrained_weights(model, config)
    else:
        # Initialize parameters
        if hasattr(model, "_init_parameters"):
            model._init_parameters()

    return model


def load_pretrained_weights(model: torch.nn.Module, config: InitParams) -> None:
    """
    Load pretrained weights if available.

    Args:
        model: Model to load weights into
        config: InitParams with pretrained configuration
    """
    # Extract model args from output filename
    model_args = ""
    if ":" in config.output.stem:
        model_args = config.output.stem.split("_")[0].split(":")[1]

    # Find similar pretrained model
    pretrained_file = find_similar_trained_model(
        directory=config.output.parent,
        model_name=config.model.model_name,
        data_name=config.data.data_name,
        seed=config.seed,
        model_args=model_args,
    )

    if pretrained_file:
        logger.info(f"Loading pretrained weights from {pretrained_file}")
        try:
            state_dict = torch.load(pretrained_file, weights_only=True)
            model.load_state_dict(state_dict)
        except Exception as e:
            logger.warning(f"Failed to load pretrained weights: {e}")
            logger.info("Proceeding with random initialization")
            if hasattr(model, "_init_parameters"):
                model._init_parameters()
    else:
        logger.info("No similar pretrained model found. Using random initialization.")
        if hasattr(model, "_init_parameters"):
            model._init_parameters()


@handle_errors(verbose=True)
def main() -> int:
    """Main entry point for model initialization."""
    try:
        # Load configuration
        config = InitParams.from_cli_and_config()
        config.setup_logging()

        # Set random seed
        set_seed(config.seed)

        # Infer dimensions from dataset if provided
        infer_dimensions_from_dataset(config)

        # Create and initialize model
        model = create_and_initialize_model(config)

        # Validate model state
        if not len(model.state_dict()):
            raise ValueError("Model state dict is empty. Check model initialization.")

        # Save initialized model
        torch.save(model.state_dict(), config.output)
        return 0

    except Exception as e:
        logger.error(f"Model initialization failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
