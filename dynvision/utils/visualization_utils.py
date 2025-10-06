"""Visualization utility functions for the DynVision toolbox.

This module provides visualization-related utilities:
- Layer response analysis
- Peak detection and analysis
- Accuracy calculation
- Response data loading
- Plot saving
- Common plotting functions
"""

import json
import logging
import sys
from copy import deepcopy
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import numpy as np
import pandas as pd
import torch

from .data_utils import load_df
from .string_utils import extract_param_from_string


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert tensor to numpy array with proper dtype handling.

    Args:
        tensor: PyTorch tensor in any precision format

    Returns:
        NumPy array in float32 format
    """
    return tensor.cpu().float().numpy()


logger = logging.getLogger(__name__)


def layer_response_avg(response: torch.Tensor) -> torch.Tensor:
    """Calculate mean response of layer activations.

    Args:
        response: Layer response tensor

    Returns:
        Mean absolute response tensor of shape [batch_size, n_timesteps]
    """
    return response.abs().mean(dim=list(range(2, response.dim())))


def layer_response_std(response: torch.Tensor) -> torch.Tensor:
    """Calculate standard deviation of layer activations.

    Args:
        response: Layer response tensor

    Returns:
        Standard deviation of absolute response tensor of shape [batch_size, n_timesteps]
    """
    return response.abs().std(dim=list(range(2, response.dim())))


def peak_time(mean_response: torch.Tensor) -> torch.Tensor:
    """Calculate peak time for each feature.

    Args:
        response: Layer response tensor

    Returns:
        Tensor of peak times for each feature
    """
    return mean_response.argmax(dim=1)


def peak_height(mean_response: torch.Tensor) -> torch.Tensor:
    """Calculate peak height for each feature.

    Args:
        response: Layer response tensor

    Returns:
        Tensor of peak heights for each feature
    """
    max_values, max_indices = mean_response.max(dim=1)
    return max_values


def peak_ratio(mean_response: torch.Tensor, min_delay: int = 3) -> torch.Tensor:
    """Calculate ratio between first and second peaks.

    Args:
        response: Layer response tensor
        min_delay: Minimum separation between peaks

    Returns:
        Tensor of peak ratios for each feature
    """
    peak1_index = mean_response.argmax(dim=1)
    peak1_value = torch.tensor(
        [
            deepcopy(mean_response[channel, i].item())
            for channel, i in enumerate(peak1_index)
        ]
    )

    for channel, i in enumerate(peak1_index):
        mean_response[channel, i - min_delay : i + min_delay] = float("-inf")

    peak2_index = mean_response.argmax(dim=1)
    peak2_value = [mean_response[channel, i] for channel, i in enumerate(peak2_index)]

    ratio = torch.Tensor(
        [
            p1 / p2 if i1 < i2 else p2 / p1
            for i1, p1, i2, p2 in zip(
                peak1_index, peak1_value, peak2_index, peak2_value
            )
        ]
    )
    return ratio


def spatial_variance(response: torch.Tensor) -> torch.Tensor:
    """Calculate variance across spatial dimensions at each timepoint.

    Args:
        response: Layer response tensor [batch_size, n_timesteps, ...spatial_dims]

    Returns:
        Variance tensor of shape [batch_size, n_timesteps]
    """
    # Handle different tensor dimensionalities
    if response.dim() < 3:
        raise ValueError(
            f"Response tensor must have at least 3 dimensions, got {response.dim()}"
        )
    elif response.dim() == 3:
        # [batch, time, channels] - no spatial dimensions
        return torch.zeros_like(response[:, :, 0])  # Return zeros for spatial variance
    elif response.dim() == 4:
        # [batch, time, channels, spatial] - 1D spatial
        spatial_var = response.var(dim=3)  # [batch, time, channels]
        return spatial_var.mean(dim=2)  # [batch, time]
    elif response.dim() == 5:
        # [batch, time, channels, dim_y, dim_x] - 2D spatial
        spatial_var = response.var(dim=(3, 4))  # [batch, time, channels]
        return spatial_var.mean(dim=2)  # [batch, time]
    else:
        # Higher dimensions - treat all beyond channels as spatial
        spatial_dims = list(range(3, response.dim()))
        spatial_var = response.var(dim=spatial_dims)  # [batch, time, channels]
        return spatial_var.mean(dim=2)  # [batch, time]


def feature_variance(response: torch.Tensor) -> torch.Tensor:
    """Calculate variance across channel dimensions at each timepoint.

    Args:
        response: Layer response tensor [batch_size, n_timesteps, n_channels, ...spatial_dims]

    Returns:
        Feature variance tensor of shape [batch_size, n_timesteps]
    """
    # Handle different tensor dimensionalities
    if response.dim() < 3:
        raise ValueError(
            f"Response tensor must have at least 3 dimensions, got {response.dim()}"
        )
    elif response.dim() == 3:
        # [batch, time, channels] - no spatial dimensions
        return response.var(dim=2)  # [batch, time]
    elif response.dim() == 4:
        # [batch, time, channels, spatial] - 1D spatial
        feature_var = response.var(dim=2)  # [batch, time, spatial]
        return feature_var.mean(dim=2)  # [batch, time]
    elif response.dim() == 5:
        # [batch, time, channels, dim_y, dim_x] - 2D spatial
        feature_var = response.var(dim=2)  # [batch, time, dim_y, dim_x]
        return feature_var.mean(dim=(2, 3))  # [batch, time]
    else:
        # Higher dimensions - treat all beyond channels as spatial
        feature_var = response.var(dim=2)  # [batch, time, ...spatial_dims]
        spatial_dims = list(range(2, feature_var.dim()))
        return feature_var.mean(dim=spatial_dims)  # [batch, time]


def calculate_accuracy(df: pd.DataFrame) -> float:
    """Calculate classification accuracy.

    Args:
        df: DataFrame with label_index and guess_index columns

    Returns:
        Classification accuracy as float
    """
    dfi = df[df.label_index >= 0]
    accuracy = (dfi.guess_index == dfi.label_index).mean()
    return accuracy


def load_config_from_args(palette_str=None, naming_str=None, ordering_str=None):
    """Load configuration from JSON strings passed via command line.

    Args:
        palette_str: JSON string for palette
        naming_str: JSON string for naming
        ordering_str: JSON string for ordering

    Returns:
        Dictionary with config sections
    """
    config = {}

    if palette_str:
        try:
            config["palette"] = json.loads(palette_str)
        except json.JSONDecodeError as e:
            print(f"Warning: Could not parse palette JSON: {e}")
            config["palette"] = {}
    else:
        config["palette"] = {}

    if naming_str:
        try:
            config["naming"] = json.loads(naming_str)
        except json.JSONDecodeError as e:
            print(f"Warning: Could not parse naming JSON: {e}")
            config["naming"] = {}
    else:
        config["naming"] = {}

    if ordering_str:
        try:
            config["ordering"] = json.loads(ordering_str)
        except json.JSONDecodeError as e:
            print(f"Warning: Could not parse ordering JSON: {e}")
            config["ordering"] = {}
    else:
        config["ordering"] = {}

    return config


def get_display_name(key: str, config: Dict) -> str:
    """Get display name/symbol for a key from config.

    Args:
        key: The key to look up
        config: Configuration dictionary

    Returns:
        Display name or symbol, fallback to key if not found
    """
    naming = config.get("naming", {})
    return naming.get(key.lower(), key)


def get_color(key: str, config: Dict) -> Optional[str]:
    """Get color for a key from config.

    Args:
        key: The key to look up
        config: Configuration dictionary

    Returns:
        Color string or None if not found
    """
    palette = config.get("palette", {})
    return palette.get(key)


def get_ordering(category: str, config: Dict) -> Optional[List[str]]:
    """Get ordering for a category from config.

    Args:
        category: The category to look up
        config: Configuration dictionary

    Returns:
        List of ordered values or None if not found
    """
    ordering = config.get("ordering", {})
    return ordering.get(category)


def order_layers(layer_names: List[str], config: Dict) -> List[str]:
    """Order layers according to visual hierarchy.

    Args:
        layer_names: List of layer names
        config: Configuration dictionary

    Returns:
        Ordered list of layer names (IT, V4, V2, V1)
    """
    # Get ordering from config or use default
    hierarchy_order = get_ordering("layers", config) or ["IT", "V4", "V2", "V1"]

    # Get naming mapping from config
    naming = config.get("naming", {})

    # Filter out classifier layers
    filtered_layers = [
        layer for layer in layer_names if "classifier" not in layer.lower()
    ]

    # Sort layers according to hierarchy
    def get_sort_key(layer_name):
        # Map layer name to display name
        mapped_name = naming.get(layer_name, layer_name)
        if mapped_name in hierarchy_order:
            return hierarchy_order.index(mapped_name)
        else:
            return len(hierarchy_order)  # Put unknown layers at the end

    ordered_layers = sorted(filtered_layers, key=get_sort_key)
    return ordered_layers


def calculate_label_indicator(
    df: pd.DataFrame, category: str, y_range: tuple, step_height: float = 0.25
) -> pd.DataFrame:
    """Calculate label indicator (step function) at each time step.

    Args:
        df: DataFrame containing classifier responses
        category: Category column name
        y_range: Tuple of (y_min, y_max) for the current subplot

    Returns:
        DataFrame with times_index and label_indicator (relative to subplot height)
    """
    # Get first model's data to determine label validity at each time step
    first_model = df[category].iloc[0]
    model_data = df[df[category] == first_model]
    y_min, y_max = y_range

    # Calculate step height as 25% of the y-axis range
    if step_height >= 0 and step_height < 1:
        # step_height is a fraction of the y-axis range
        step_height = (y_max - y_min) * step_height
    elif step_height >= 1:
        # step_height is an absolute value, ensure it does not exceed 25% of y-axis range
        max_step = (y_max - y_min) * 0.25
        step_height = min(step_height, max_step)
    else:
        raise ValueError("step_height must be positive")

    indicator_data = []
    for time_step in sorted(model_data.times_index.unique()):
        time_data = model_data[model_data.times_index == time_step]

        # Check if any labels are valid (>= 0) at this time step
        valid_labels = (time_data.label_index >= 0).any()
        indicator_value = y_min + step_height if valid_labels else y_min

        indicator_data.append(
            {"times_index": time_step, "label_indicator": indicator_value}
        )

    return pd.DataFrame(indicator_data)


def get_category_plotting_settings(
    category: str, category_values: List, config: Dict
) -> Tuple[List, Dict[str, str]]:
    """Get plotting settings for different categories.

    Args:
        category: The category name
        category_values: List of unique category values
        config: Configuration dictionary

    Returns:
        Tuple of (order_list, colors_dict)
    """
    # Get ordering from config
    model_order = get_ordering(category, config)

    if model_order:
        # Filter to only include values that exist in the data
        model_order = [val for val in model_order if val in category_values]
        # Add any missing values from data
        for val in category_values:
            if val not in model_order:
                model_order.append(val)
    else:
        # Try to order by float values if no config ordering
        try:
            float_values = [(float(val), val) for val in category_values]
            float_values.sort(key=lambda x: x[0])
            model_order = [val for _, val in float_values]
        except (ValueError, TypeError):
            # Can't convert to float, use alphabetical order
            model_order = sorted(category_values)

    # Get colors from config
    colors = {}
    for val in model_order:
        color = get_color(val, config)
        if color:
            colors[val] = color

    # Fill in missing colors with defaults
    if len(colors) < len(model_order):
        if len(model_order) <= 10:
            default_colors = plt.cm.tab10(np.arange(len(model_order)))
        else:
            default_colors = plt.cm.tab20(np.arange(len(model_order)) % 20)

        for i, val in enumerate(model_order):
            if val not in colors:
                colors[val] = mcolors.rgb2hex(default_colors[i])

    return model_order, colors


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
            pt_file.name, key=data_arg_key, value_type=float
        )

        cat_value = extract_param_from_string(
            pt_file.name, key=category, value_type=None
        )

        if not arg_value == extract_param_from_string(
            csv_file.name, key=data_arg_key, value_type=float
        ):
            raise ValueError(f"{data_arg_key} values do not match!")
        if not cat_value == extract_param_from_string(
            csv_file.name, key=category, value_type=None
        ):
            raise ValueError(f"{category} do not match!")

        df = load_df(csv_file)
        df[data_arg_key] = arg_value
        df[category] = cat_value
        n_classes = len(df.class_index.unique())

        # Remove 'class_index' and 'response' columns if present
        for col in ["class_index", "response"]:
            if col in df.columns:
                df = df.drop(columns=col)

        # Collapse redundant rows by grouping on all remaining columns and aggregating with 'first'
        df = df.groupby(list(df.columns), as_index=False).first()

        responses = torch.load(pt_file, map_location=torch.device("cpu"))

        max_timesteps = max(tensor.shape[1] for tensor in responses.values())

        for layer_name, tensor in responses.items():
            pad_len = max_timesteps - tensor.shape[1]
            if pad_len > 0:
                # Pad at the start along axis 1 (time steps) with zeros
                # For shape [batch, timesteps, n_channels, dim_y, dim_x], pad for axis 1
                # torch.nn.functional.pad expects (dim_x, dim_y, n_channels, timesteps)
                # So pad = (0,0, 0,0, 0,0, pad_len,0)
                pad = (0, 0, 0, 0, 0, 0, pad_len, 0)
                tensor = torch.nn.functional.pad(tensor, pad, mode="constant", value=0)
                responses[layer_name] = tensor

        layer_names = list(responses.keys())
        n_samples, n_timesteps, *_ = responses[layer_names[0]].shape

        for layer in layer_names:
            try:
                if "response_avg" in measures:
                    response_avg_tensor = (
                        layer_response_avg(responses[layer])
                        .flatten()
                        .repeat_interleave(n_classes)
                    )
                    df[f"{layer}_response_avg"] = tensor_to_numpy(response_avg_tensor)

                if "response_std" in measures:
                    response_std_tensor = (
                        layer_response_std(responses[layer])
                        .flatten()
                        .repeat_interleave(n_classes)
                    )
                    df[f"{layer}_response_std"] = tensor_to_numpy(response_std_tensor)

                if "peak_time" in measures:
                    peak_time_tensor = peak_time(responses[layer]).repeat_interleave(
                        n_classes * n_timesteps
                    )
                    df[f"{layer}_peak_time"] = tensor_to_numpy(peak_time_tensor)

                if "peak_height" in measures:
                    peak_height_tensor = peak_height(
                        responses[layer]
                    ).repeat_interleave(n_classes * n_timesteps)
                    df[f"{layer}_peak_height"] = tensor_to_numpy(peak_height_tensor)

                if "peak_ratio" in measures:
                    peak_ratio_tensor = peak_ratio(responses[layer]).repeat_interleave(
                        n_classes * n_timesteps
                    )
                    df[f"{layer}_peak_ratio"] = tensor_to_numpy(peak_ratio_tensor)
            except Exception as e:
                logger.error(
                    f"Failed to convert tensor to numpy for layer {layer}: {e}"
                )
                raise

        dfs.append(df)

    df = pd.concat(dfs, axis=0)

    return df, layer_names


def chunk_lists(lst1, lst2, chunk_size):
    """Split two lists into chunks of specified size."""
    for i in range(0, len(lst1), chunk_size):
        yield lst1[i : i + chunk_size], lst2[i : i + chunk_size]


def load_responses_in_batches(
    responses_files,
    test_outputs_files,
    data_arg_key,
    measures,
    category,
    batch_size=1,
    filter_first_label_set=False,
):
    """Load responses in batches to manage memory efficiently."""

    if len(responses_files) != len(test_outputs_files):
        raise ValueError(
            "Number of response files must match number of test output files"
        )

    print(f"Processing {len(responses_files)} files in batches of {batch_size}")

    all_dataframes = []
    layer_names = None
    first_label_set = None

    # Process files in batches
    batch_count = 0
    for response_batch, output_batch in chunk_lists(
        responses_files, test_outputs_files, batch_size
    ):
        batch_count += 1
        print(
            f"Processing batch {batch_count}/{(len(responses_files) + batch_size - 1) // batch_size}"
        )
        print(
            f"  Files: {len(response_batch)} response files, {len(output_batch)} output files"
        )

        try:
            # Load this batch
            batch_df, batch_layer_names = load_responses(
                response_batch,
                output_batch,
                data_arg_key=data_arg_key,
                measures=measures,
                category=category,
            )

            # Store layer names from first batch
            if layer_names is None:
                layer_names = batch_layer_names
            else:
                # Verify layer names are consistent across batches
                if set(layer_names) != set(batch_layer_names):
                    print(
                        f"Warning: Layer names differ between batches. "
                        f"First batch: {layer_names}, Current batch: {batch_layer_names}"
                    )

            if filter_first_label_set:
                # Get first label_set from first batch and filter immediately for efficiency
                first_label_set = batch_df.label_set.unique()[0]
                print(f"  Using first label_set: {first_label_set}")

                # Filter for first label_set only to reduce memory usage
                if first_label_set is not None:
                    batch_df = batch_df[
                        batch_df.label_set.str.contains(str(first_label_set))
                    ]
                    print(f"  After filtering for label_set: {len(batch_df)} rows")

                # Remove label_set column to save memory since we no longer need it
                if "label_set" in batch_df.columns:
                    batch_df = batch_df.drop("label_set", axis=1)

            all_dataframes.append(batch_df)
            print(f"  Batch {batch_count} processed: {len(batch_df)} rows")

            # Clear memory
            del batch_df

        except Exception as e:
            print(f"Error processing batch {batch_count}: {e}")
            print(f"  Response files: {response_batch}")
            print(f"  Output files: {output_batch}")
            continue

    if not all_dataframes:
        raise ValueError("No data was successfully loaded from any batch")

    # Combine all dataframes
    print(f"Combining {len(all_dataframes)} batches...")
    combined_df = pd.concat(all_dataframes, ignore_index=True)

    # Clear memory
    del all_dataframes

    print(f"Total combined data: {len(combined_df)} rows")
    return combined_df, layer_names


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
