"""Visualization utility functions for the DynVision toolbox.

This module provides visualization-related utilities:
- Layer response analysis
- Peak detection and analysis
- Accuracy calculation
- Response data loading
- Plot saving
"""

import logging
import sys
from copy import deepcopy
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import torch

from .data_utils import load_df
from .string_utils import extract_param_from_string

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def layer_power(response: torch.Tensor) -> torch.Tensor:
    """Calculate mean power of layer responses.

    Args:
        response: Layer response tensor

    Returns:
        Mean power tensor of shape [n_features, n_timesteps]
    """
    return response.mean(dim=list(range(2, response.dim())))


def peak_time(response: torch.Tensor) -> torch.Tensor:
    """Calculate peak time for each feature.

    Args:
        response: Layer response tensor

    Returns:
        Tensor of peak times for each feature
    """
    mean_power = layer_power(response)
    return mean_power.argmax(dim=1)


def peak_height(response: torch.Tensor) -> torch.Tensor:
    """Calculate peak height for each feature.

    Args:
        response: Layer response tensor

    Returns:
        Tensor of peak heights for each feature
    """
    mean_power = layer_power(response)
    max_values, max_indices = mean_power.max(dim=1)
    return max_values


def peak_ratio(response: torch.Tensor, min_delay: int = 3) -> torch.Tensor:
    """Calculate ratio between first and second peaks.

    Args:
        response: Layer response tensor
        min_delay: Minimum separation between peaks

    Returns:
        Tensor of peak ratios for each feature
    """
    mean_power = layer_power(response)
    peak1_index = mean_power.argmax(dim=1)
    peak1_value = torch.tensor(
        [
            deepcopy(mean_power[channel, i].item())
            for channel, i in enumerate(peak1_index)
        ]
    )

    for channel, i in enumerate(peak1_index):
        mean_power[channel, i - min_delay : i + min_delay] = float("-inf")

    peak2_index = mean_power.argmax(dim=1)
    peak2_value = [mean_power[channel, i] for channel, i in enumerate(peak2_index)]

    ratio = torch.Tensor(
        [
            p1 / p2 if i1 < i2 else p2 / p1
            for i1, p1, i2, p2 in zip(
                peak1_index, peak1_value, peak2_index, peak2_value
            )
        ]
    )
    return ratio


def calculate_accuracy(df: pd.DataFrame) -> float:
    """Calculate classification accuracy.

    Args:
        df: DataFrame with label_index and guess_index columns

    Returns:
        Classification accuracy as float
    """
    dfi = df[df.label_index != -1]
    n_correct = (dfi.guess_index == dfi.label_index).sum()
    accuracy = n_correct / len(dfi)
    return accuracy


def load_responses(
    pt_files: List[Path],
    csv_files: List[Path],
    data_arg_key: str = "contrast",
    measures: List[str] = ["power", "peak_time", "peak_height"],
    category: str = "rctype",
) -> Tuple[pd.DataFrame, List[str]]:
    """Load and process model responses.

    Args:
        pt_files: List of PyTorch response files
        csv_files: List of CSV label files
        data_arg_key: Key for data arguments
        measures: List of measures to compute
        category: Category key for grouping

    Returns:
        Tuple of (DataFrame with responses, list of layer names)
    """
    dfs = []

    for pt_file, csv_file in zip(pt_files, csv_files):
        arg_value = extract_param_from_string(
            pt_file.stem, key=data_arg_key, value_type=float
        )

        cat_value = extract_param_from_string(
            pt_file.stem, key=category, value_type=None
        )

        if not arg_value == extract_param_from_string(
            csv_file.stem, key=data_arg_key, value_type=float
        ):
            raise ValueError(f"{data_arg_key} values do not match!")
        if not cat_value == extract_param_from_string(
            csv_file.stem, key=category, value_type=None
        ):
            raise ValueError(f"{category} do not match!")

        df = load_df(csv_file)
        df[data_arg_key] = arg_value
        df[category] = cat_value

        n_classes = len(df.class_index.unique())

        responses = torch.load(pt_file, map_location=torch.device("cpu"))
        layer_names = list(responses.keys())
        n_samples, n_timesteps, *_ = responses[layer_names[0]].shape

        for layer in layer_names:
            if "power" in measures:
                df[f"{layer}_power"] = (
                    layer_power(responses[layer])
                    .flatten()
                    .repeat_interleave(n_classes)
                )
            if "peak_time" in measures:
                df[f"{layer}_peak_time"] = peak_time(
                    responses[layer]
                ).repeat_interleave(n_classes * n_timesteps)
            if "peak_height" in measures:
                df[f"{layer}_peak_height"] = peak_height(
                    responses[layer]
                ).repeat_interleave(n_classes * n_timesteps)
            if "peak_ratio" in measures:
                df[f"{layer}_peak_ratio"] = peak_ratio(
                    responses[layer]
                ).repeat_interleave(n_classes * n_timesteps)

        dfs.append(df)

    df = pd.concat(dfs, axis=0)

    return df, layer_names


def save_plot(file_path: Path, dpi: int = 300, **kwargs) -> None:
    """Save matplotlib plot with error handling.

    Args:
        file_path: Path to save plot
        dpi: Resolution for saved plot
        **kwargs: Additional arguments for savefig

    Returns:
        None
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(exist_ok=True, parents=True)

    try:
        plt.savefig(fname=file_path, dpi=dpi, bbox_inches="tight", **kwargs)

    except Exception as e:
        logging.error(f"Failed to save plot: {e}")
        try:  # save empty plot
            plt.subplots()
            plt.savefig(fname=file_path)
        except Exception as e:
            logging.error(f"Failed to save empty plot: {e}")

    finally:
        if "ipykernel" in sys.modules:
            plt.show()
        else:
            plt.close()

    logging.info(f"Plot saved successfully at {file_path}")

    return None