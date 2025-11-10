"""Plot training accuracy over epochs and loss curves with comprehensive model statistics."""

"""Plot training accuracy over epochs and loss curves with comprehensive model statistics."""

import argparse
import json
import logging
import re
import ast  # Add this import for ast.literal_eval
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.ndimage import uniform_filter1d

from dynvision.utils.visualization_utils import (
    save_plot,
    load_config_from_args,
    get_display_name,
    get_color,
    calculate_label_indicator,
    get_category_plotting_settings,
)


logger = logging.getLogger(__name__)


DEFAULT_PALETTE = {
    "full": "#1f77b4",  # Blue
    "self": "#ff7f0e",  # Orange
    "depthpointwise": "#2ca02c",  # Green
    "pointdepthwise": "#d62728",  # Red
    "local": "#9467bd",  # Purple
    "localdepthwise": "#8c564b",  # Brown
}

DEFAULT_COLOR = "#5a5a5a"


def _coerce_to_json_string(value: Optional[str]) -> Optional[str]:
    """Convert loose JSON-like CLI strings into valid JSON strings."""

    if value is None:
        return None

    text = value.strip()
    if not text:
        return None

    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        try:
            literal_value = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            logger.warning("Unable to parse configuration string: %s", text)
            return None

        try:
            return json.dumps(literal_value)
        except (TypeError, ValueError):
            logger.warning(
                "Unable to convert configuration literal to JSON: %s", text
            )
            return None


def parse_palette(palette_input: Optional[Union[str, Dict[str, str]]]) -> Dict[str, str]:
    """Parse palette input from CLI/config into a dictionary."""

    if palette_input is None:
        return {}

    if isinstance(palette_input, dict):
        return {str(k): str(v) for k, v in palette_input.items()}

    try:
        parsed = ast.literal_eval(palette_input)
    except (ValueError, SyntaxError):
        logger.warning("Could not parse palette input: %s", palette_input)
        return {}

    if not isinstance(parsed, dict):
        logger.warning("Palette input is not a mapping: %s", palette_input)
        return {}

    return {str(k): str(v) for k, v in parsed.items()}


def _resolve_palette(
    config: Optional[Dict[str, Dict[str, str]]],
    palette_input: Optional[Union[str, Dict[str, str]]],
) -> Dict[str, str]:
    """Merge default palette with config overrides and CLI overrides."""

    palette = DEFAULT_PALETTE.copy()

    if config:
        palette.update(config.get("palette", {}))

    parsed_palette = parse_palette(palette_input)
    if parsed_palette:
        palette.update(parsed_palette)

    return palette


def _determine_model_order(
    all_models: Sequence[str], config: Optional[Dict[str, Dict[str, str]]]
) -> List[str]:
    """Determine plotting order for model categories."""

    candidate_order: List[str] = []

    if config:
        ordering = config.get("ordering", {})
        for key in ("model_type", "model_types", "recurrence_type", "recurrence", "rctype"):
            values = ordering.get(key) or ordering.get(key.lower())
            if values:
                candidate_order = list(values)
                break

    if not candidate_order:
        candidate_order = list(MODEL_ORDER)

    ordered = [model for model in candidate_order if model in all_models]
    remaining = [model for model in sorted(set(all_models)) if model not in ordered]
    ordered.extend(remaining)

    return ordered


def _get_model_color(
    model_type: str, colors: Dict[str, str], config: Optional[Dict[str, Dict[str, str]]]
) -> str:
    """Resolve color for a model based on config overrides and palette."""

    if config:
        configured_color = get_color(model_type, config)
        if configured_color:
            return configured_color

    if model_type in colors:
        return colors[model_type]

    model_lower = model_type.lower()
    if model_lower in colors:
        return colors[model_lower]

    logger.debug("Falling back to default color for model '%s'", model_type)
    return DEFAULT_COLOR


def _format_model_label(model_type: str, config: Optional[Dict[str, Dict[str, str]]]) -> str:
    """Format model label using config naming overrides when available."""

    if config:
        display = get_display_name(model_type, config)
        if display:
            return display

    return model_type.replace("wise", "w.")


def _load_test_data(test_data_paths: Sequence[Path]) -> pd.DataFrame:
    """Load and concatenate test performance CSV files."""

    if not test_data_paths:
        raise ValueError("No test data paths provided")

    dataframes = []
    for idx, path in enumerate(test_data_paths, start=1):
        logger.info("Loading test data file %s/%s: %s", idx, len(test_data_paths), path)
        df = pd.read_csv(path)
        logger.debug("Loaded test data shape %s from %s", df.shape, path)
        dataframes.append(df)

    concatenated = pd.concat(dataframes, ignore_index=True)
    logger.info(
        "Concatenated test data: %s rows, %s columns",
        concatenated.shape[0],
        concatenated.shape[1],
    )
    return concatenated

# Global model order for consistent plotting across all subplots
MODEL_ORDER = [
    "full",
    "self",
    "depthpointwise",
    "pointdepthwise",
    "local",
    "localdepthwise",
]


def parse_model_identifier_from_column(column_name: str) -> str:
    """
    Extract model identifier from W&B column name.

    Args:
        column_name: Full column name from W&B export

    Returns:
        Model identifier (e.g., 'full', 'pointdepthwise', etc.)
    """
    # Extract rctype from the column name
    match = re.search(r"recurrence_type: ([^-]+) -", column_name)
    if match:
        result = match.group(1).strip()
        return result
    else:
        return "unknown"
def load_accuracy_data(accuracy_csv_path: Path) -> pd.DataFrame:
    """
    Load and process training and validation accuracy data from W&B export.

    Args:
        accuracy_csv_path: Path to accuracy CSV file

    Returns:
        Processed DataFrame with columns: epoch, model_type, train_accuracy, val_accuracy
    """
    df = pd.read_csv(accuracy_csv_path)

    # Find all accuracy columns (excluding MIN/MAX variants)
    train_accuracy_columns = [
        col
        for col in df.columns
        if "train_accuracy" in col and "__MIN" not in col and "__MAX" not in col
    ]
    val_accuracy_columns = [
        col
        for col in df.columns
        if "val_accuracy" in col and "__MIN" not in col and "__MAX" not in col
    ]

    processed_data = []

    # Group by model type
    model_types = set()
    for col in train_accuracy_columns + val_accuracy_columns:
        model_types.add(parse_model_identifier_from_column(col))

    for model_type in model_types:
        train_col = next(
            (
                col
                for col in train_accuracy_columns
                if parse_model_identifier_from_column(col) == model_type
            ),
            None,
        )
        val_col = next(
            (
                col
                for col in val_accuracy_columns
                if parse_model_identifier_from_column(col) == model_type
            ),
            None,
        )

        for idx, row in df.iterrows():
            train_acc = (
                row[train_col] if train_col and not pd.isna(row[train_col]) else None
            )
            val_acc = row[val_col] if val_col and not pd.isna(row[val_col]) else None

            if train_acc is not None or val_acc is not None:
                processed_data.append(
                    {
                        "epoch": row["epoch"],
                        "model_type": model_type,
                        "train_accuracy": train_acc,
                        "val_accuracy": val_acc,
                    }
                )

    return pd.DataFrame(processed_data)


def load_loss_data(loss_csv_path: Path, loss_type: str) -> pd.DataFrame:
    """
    Load and process loss data from W&B export.

    Args:
        loss_csv_path: Path to loss CSV file
        loss_type: Type of loss ('energy' or 'cross_entropy')

    Returns:
        Processed DataFrame with columns: epoch, model_type, {loss_type}_loss
    """
    df = pd.read_csv(loss_csv_path)

    # Find all loss columns (excluding MIN/MAX variants)
    loss_columns = [
        col
        for col in df.columns
        if loss_type.lower().replace("_", "").replace(" ", "")
        in col.lower().replace("_", "").replace(" ", "")
        and "__MIN" not in col
        and "__MAX" not in col
    ]

    processed_data = []

    for col in loss_columns:
        model_type = parse_model_identifier_from_column(col)

        # Create records for this model
        for idx, row in df.iterrows():
            if not pd.isna(row[col]):  # Only include non-NaN values
                processed_data.append(
                    {
                        "epoch": row["epoch"],
                        "model_type": model_type,
                        f"{loss_type}_loss": row[col],
                    }
                )

    return pd.DataFrame(processed_data)


def load_memory_data(memory_csv_path: Path) -> pd.DataFrame:
    """
    Load and process GPU memory allocation data from W&B export.

    Args:
        memory_csv_path: Path to memory CSV file

    Returns:
        Processed DataFrame with columns: model_type, gpu_mem_alloc
    """
    df = pd.read_csv(memory_csv_path)

    # Find all memory columns (excluding MIN/MAX variants)
    memory_columns = [
        col
        for col in df.columns
        if "memoryAllocatedBytes" in col and "__MIN" not in col and "__MAX" not in col
    ]

    processed_data = []

    for col in memory_columns:
        model_type = parse_model_identifier_from_column(col)

        # Create records for this model
        for idx, row in df.iterrows():
            if not pd.isna(row[col]):  # Only include non-NaN values
                processed_data.append(
                    {
                        "model_type": model_type,
                        "gpu_mem_alloc": row[col] / (1024**3),  # Convert bytes to GB
                    }
                )
    return pd.DataFrame(processed_data)


def load_epoch_data(epoch_csv_path: Path) -> pd.DataFrame:
    """
    Load and process epoch timing data from W&B export.

    Calculate epoch duration by analyzing the progression of epoch numbers over time
    for each model type.

    Args:
        epoch_csv_path: Path to epoch CSV file

    Returns:
        Processed DataFrame with columns: model_type, epoch_duration
    """
    df = pd.read_csv(epoch_csv_path)

    # Find epoch timing columns (excluding MIN/MAX variants)
    # Look for columns like "recurrence_type: {model_type} - epoch"
    epoch_columns = [
        col
        for col in df.columns
        if "recurrence_type:" in col
        and " - epoch" in col
        and "__MIN" not in col
        and "__MAX" not in col
    ]

    processed_data = []

    for col in epoch_columns:
        model_type = parse_model_identifier_from_column(col)

        # Get time and epoch data for this model, filtering out NaN values
        model_data = df[["Relative Time (Process)", col]].dropna()
        model_data = model_data[model_data[col] > 0]  # Only positive epoch values

        if len(model_data) < 2:  # Need at least 2 points to calculate slope
            continue

        # Sort by time to ensure proper order
        model_data = model_data.sort_values("Relative Time (Process)")

        times = model_data["Relative Time (Process)"].values
        epochs = model_data[col].values

        # Calculate epoch durations by finding time differences between epoch changes
        epoch_durations = []

        for i in range(1, len(epochs)):
            if epochs[i] > epochs[i - 1]:  # Epoch has increased
                time_diff = times[i] - times[i - 1]
                epoch_diff = epochs[i] - epochs[i - 1]
                if epoch_diff > 0:  # Avoid division by zero
                    duration_per_epoch = time_diff / epoch_diff
                    # Add multiple entries for this duration (one per epoch completed)
                    for _ in range(int(epoch_diff)):
                        epoch_durations.append(duration_per_epoch)

        # Add all calculated durations to processed data
        for duration in epoch_durations:
            processed_data.append(
                {
                    "model_type": model_type,
                    "epoch_duration": duration,
                }
            )

    return pd.DataFrame(processed_data)


def smooth_data(data: pd.Series, window_size: int = 5) -> pd.Series:
    """
    Smooth data using uniform filter.

    Args:
        data: Data to smooth
        window_size: Size of smoothing window

    Returns:
        Smoothed data
    """
    if len(data) < window_size:
        return data
    return pd.Series(uniform_filter1d(data.values, size=window_size), index=data.index)


def format_parameter_count(n_params: int) -> str:
    """
    Format parameter count in millions with 1 decimal place.

    Args:
        n_params: Number of parameters

    Returns:
        Formatted string like "5.4M"
    """
    if n_params is None:
        return "N/A"
    millions = n_params / 1_000_000
    return f"{millions:.2f}M"


def format_memory(mem_gb: float) -> str:
    """
    Format memory in GB with 1 decimal place.

    Args:
        mem_gb: Memory in GB

    Returns:
        Formatted string like "2.4GB"
    """
    if mem_gb is None:
        return "N/A"
    return f"{mem_gb:.1f}GB"


def format_time(time_seconds: float) -> str:
    """
    Format time in appropriate units.

    Args:
        time_seconds: Time in seconds

    Returns:
        Formatted string like "1.5s" or "2.3m"
    """
    if time_seconds is None:
        return "N/A"
    if time_seconds < 60:
        return f"{time_seconds:.1f}s"
    else:
        return f"{time_seconds/60:.1f}m"


def calculate_model_statistics(
    accuracy_df: pd.DataFrame,
    energy_loss_df: pd.DataFrame,
    cross_entropy_loss_df: pd.DataFrame,
    memory_df: pd.DataFrame,
    epoch_df: pd.DataFrame,
) -> Dict[str, Dict[str, float]]:
    """
    Calculate comprehensive statistics for each model.

    Args:
        accuracy_df: Training and validation accuracy data
        energy_loss_df: Energy loss data
        cross_entropy_loss_df: Cross entropy loss data
        memory_df: GPU memory allocation data
        epoch_df: Epoch timing data

    Returns:
        Dictionary with model statistics
    """
    stats = {}

    # Parameter counts for each model type
    # format: without skip connections + skip connections
    skip_count = 93_345
    parameter_counts = {
        "depthpointwise": 5_425_195 + skip_count,  # 5_425_195,  # 6_043_240,
        "pointdepthwise": 5_425_195 + skip_count,  # 5_425_195,  # 6_043_240,
        "full": 8_441_449 + skip_count,  # 8_441_449,  # 9_059_494,
        "self": 5_045_025 + skip_count,  # 5_045_025,  # 5_663_070,
        "local": 5_043_111 + skip_count,  # 5_043_111,  # 5_661_156,
        "localdepthwise": 5_054_065 + skip_count,  # 5_054_065,  # 5_672_110,
    }

    # Get unique model types
    all_models = set()
    for df in [
        accuracy_df,
        energy_loss_df,
        cross_entropy_loss_df,
        memory_df,
        epoch_df,
    ]:
        if not df.empty and "model_type" in df.columns:
            all_models.update(df.model_type.unique())

    for model_type in all_models:
        stats[model_type] = {}

        # Get max train and validation accuracy for this model
        if not accuracy_df.empty and model_type in accuracy_df.model_type.values:
            model_acc_data = accuracy_df[accuracy_df.model_type == model_type]
            max_train_acc = model_acc_data.train_accuracy.max()
            max_val_acc = model_acc_data.val_accuracy.max()
            stats[model_type]["max_train_acc"] = max_train_acc
            stats[model_type]["max_val_acc"] = max_val_acc
        else:
            stats[model_type]["max_train_acc"] = None
            stats[model_type]["max_val_acc"] = None

        # Get average GPU memory allocation
        if not memory_df.empty and "model_type" in memory_df.columns:
            model_mem_data = memory_df[memory_df.model_type == model_type]
            if not model_mem_data.empty:
                avg_gpu_mem = model_mem_data["gpu_mem_alloc"].mean()
                stats[model_type]["avg_gpu_mem"] = avg_gpu_mem
            else:
                stats[model_type]["avg_gpu_mem"] = None
        else:
            stats[model_type]["avg_gpu_mem"] = None

        # Get average runtime per epoch
        if not epoch_df.empty and "model_type" in epoch_df.columns:
            model_epoch_data = epoch_df[epoch_df.model_type == model_type]
            if not model_epoch_data.empty:
                avg_epoch_time = model_epoch_data["epoch_duration"].mean()
                stats[model_type]["avg_epoch_time"] = avg_epoch_time
            else:
                stats[model_type]["avg_epoch_time"] = None
        else:
            stats[model_type]["avg_epoch_time"] = None

        # Add parameter count
        stats[model_type]["n_parameters"] = parameter_counts.get(model_type, None)

    return stats


def plot_training_losses(
    accuracy_df: pd.DataFrame,
    energy_loss_df: pd.DataFrame,
    cross_entropy_loss_df: pd.DataFrame,
    memory_df: pd.DataFrame,
    epoch_df: pd.DataFrame,
    palette: Optional[Union[str, Dict[str, str]]],
    test_data=None,
    dt=None,
    category_col="model_type",
    config=None,
    figsize: Tuple[int, int] = (24, 8),
) -> plt.Figure:
    """
    Create three-panel plot:
    - Left: training/validation accuracy + loss curves
    - Center: comprehensive stats table
    - Right: test performance (accuracy/confidence) + V1 response over time

    Args:
        accuracy_df: Processed accuracy data
        energy_loss_df: Processed energy loss data
        cross_entropy_loss_df: Processed cross entropy loss data
        memory_df: Processed memory data
        epoch_df: Processed epoch timing data
    palette: Color palette override (string or dict)
        test_data: Optional DataFrame with test performance and layer responses
        dt: Optional time step duration in milliseconds for x-axis in test plots
        category_col: Column name containing model categories in test data
        config: Dictionary with visualization configuration
        figsize: Figure dimensions

    Returns:
        Matplotlib figure
    """
    # Set style
    sns.set_context("talk")
    sns.set_style("ticks")

    # Parse color palette
    colors = parse_palette(palette)

    # Create figure with custom layout for three columns
    fig = plt.figure(figsize=figsize)

    # Define grid layout:
    # - Left column: 2 subplots (training accuracy, loss)
    # - Center column: Statistics table
    # - Right column: 2 subplots (test accuracy/confidence, V1 response)

    # Left column: Training plots
    ax1 = fig.add_subplot(2, 3, 1)  # Top left - Training/Validation accuracy
    ax2 = fig.add_subplot(2, 3, 4)  # Bottom left - Loss curves

    # Center column: Stats table (spanning both rows) - adjusted to provide more space
    ax3 = fig.add_axes([0.35, 0.1, 0.31, 0.8])  # [left, bottom, width, height]

    # Right column: Test performance (only if test_data is provided)
    if test_data is not None:
        ax4 = fig.add_subplot(2, 3, 3)  # Top right - Accuracy/Confidence
        ax5 = fig.add_subplot(2, 3, 6)  # Bottom right - V1 Response
    else:
        ax4, ax5 = None, None

    # Get available model types
    all_models = set()
    for df in [accuracy_df, energy_loss_df, cross_entropy_loss_df]:
        if df is not None and "model_type" in df.columns:
            all_models.update(df.model_type.unique())

    # Filter MODEL_ORDER to only include available models
    available_model_order = [m for m in MODEL_ORDER if m in all_models]

    # Plot 1: Training and validation accuracy over epochs (top left)
    # Plot training accuracy (solid lines)
    for model_type in available_model_order:
        if model_type in accuracy_df.model_type.values:
            model_data = accuracy_df[accuracy_df.model_type == model_type]
            # Apply smoothing to training accuracy
            if len(model_data) > 0 and not model_data.train_accuracy.isnull().all():
                sns.lineplot(
                    data=model_data,
                    x="epoch",
                    y="train_accuracy",
                    color=colors.get(model_type, "gray"),
                    linewidth=2,
                    label=model_type,
                    alpha=0.8,
                    ax=ax1,
                )

    # Plot validation accuracy (dotted lines)
    for model_type in available_model_order:
        if model_type in accuracy_df.model_type.values:
            model_data = accuracy_df[accuracy_df.model_type == model_type]
            # Debugging to check validation data
            print(
                f"Model {model_type} val accuracy data available: {not model_data.val_accuracy.isnull().all()}"
            )

            # Apply smoothing to validation accuracy if data exists
            if len(model_data) > 0 and not model_data.val_accuracy.isnull().all():
                sns.lineplot(
                    data=model_data,
                    x="epoch",
                    y="val_accuracy",
                    color=colors.get(model_type, "gray"),
                    linewidth=2,
                    linestyle=":",
                    alpha=0.8,
                    ax=ax1,
                )

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.grid(True, alpha=0.3)

    # Add legend for Training/Validation in bottom right of this plot
    from matplotlib.lines import Line2D

    train_val_legend = [
        Line2D([0], [0], color="black", linewidth=2, linestyle="-", label="Training"),
        Line2D(
            [0], [0], color="black", linewidth=2, linestyle=":", label="Validation"
        ),
    ]
    ax1.legend(handles=train_val_legend, loc="lower right", frameon=False)

    # Plot 2: Loss curves over training steps (bottom left)
    # Plot cross entropy loss (solid lines) using matplotlib for consistent handling
    for model_type in available_model_order:
        if model_type in cross_entropy_loss_df.model_type.values:
            model_data = cross_entropy_loss_df[
                cross_entropy_loss_df.model_type == model_type
            ]
            if (
                len(model_data) > 0
                and not model_data.cross_entropy_loss.isnull().all()
            ):
                # Apply smoothing to cross entropy loss
                smoothed_loss = smooth_data(
                    model_data.cross_entropy_loss, window_size=5
                )
                ax2.plot(
                    model_data.epoch,
                    smoothed_loss,
                    color=colors.get(model_type, "gray"),
                    linewidth=2,
                    alpha=0.8,
                    label=model_type,
                )

    # Plot energy loss (dashed lines) using matplotlib
    for model_type in available_model_order:
        if model_type in energy_loss_df.model_type.values:
            model_data = energy_loss_df[energy_loss_df.model_type == model_type]
            if len(model_data) > 0 and not model_data.energy_loss.isnull().all():
                # Apply smoothing to energy loss
                smoothed_energy = smooth_data(model_data.energy_loss, window_size=5)
                ax2.plot(
                    model_data.epoch,
                    smoothed_energy,
                    color=colors.get(model_type, "gray"),
                    linewidth=2,
                    linestyle="--",
                    alpha=0.8,
                )

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.grid(True, alpha=0.3)

    # Add legend for loss types
    loss_legend = [
        Line2D(
            [0], [0], color="black", linewidth=2, linestyle="-", label="Cross Entropy"
        ),
        Line2D([0], [0], color="black", linewidth=2, linestyle="--", label="Energy"),
    ]
    ax2.legend(handles=loss_legend, loc="upper right", frameon=False)

    # Plot 3: Comprehensive Statistics table (center column)
    ax3.axis("off")  # Turn off axis for text display

    # Calculate statistics
    stats = calculate_model_statistics(
        accuracy_df, energy_loss_df, cross_entropy_loss_df, memory_df, epoch_df
    )

    # Create table format with adjusted column widths for better spacing
    col_widths = [
        0.16,
        0.14,
        0.17,
        0.14,
        0.16,
        0.17,
    ]  # Slightly reduced widths to prevent overlap
    col_offset = 0.0  # Adjusted offset for better alignment
    col_headers = [
        "Recurrence",
        "# Params",
        "Avg epoch\nruntime",
        "Avg GPU\nMem",
        "Max\nTrain Acc",
        "Max\nVal Acc",
    ]

    # Header row
    y_start = 0.9
    row_height = 0.12

    for i, header in enumerate(col_headers):
        x_pos = sum(col_widths[:i]) + col_offset
        ax3.text(
            x_pos + col_widths[i] / 2,
            y_start,
            header,
            fontsize=13,
            fontweight="bold",
            ha="center",
            va="center",
        )

    # Data rows
    for row_idx, model_type in enumerate(available_model_order):
        if model_type in stats:
            model_stats = stats[model_type]
            y_pos = y_start - (row_idx + 1) * row_height
            model_label = model_type.replace("wise", "w.")

            # Model name (colored)
            x_pos = col_widths[0] / 2 + col_offset
            ax3.text(
                x_pos,
                y_pos,
                model_label,
                fontsize=13,
                color=colors.get(model_type, "#000000"),
                fontweight="bold",
                ha="center",
                va="center",
            )

            fontsize = 14

            # Number of parameters
            n_params = model_stats.get("n_parameters")
            text_str = format_parameter_count(n_params)
            x_pos = col_widths[0] + col_widths[1] / 2 + col_offset
            ax3.text(
                x_pos, y_pos, text_str, fontsize=fontsize, ha="center", va="center"
            )

            # Average runtime per epoch
            avg_time = model_stats.get("avg_epoch_time")
            text_str = format_time(avg_time) if avg_time is not None else "N/A"
            x_pos = sum(col_widths[:2]) + col_widths[2] / 2 + col_offset
            ax3.text(
                x_pos, y_pos, text_str, fontsize=fontsize, ha="center", va="center"
            )

            # Average GPU memory
            avg_mem = model_stats.get("avg_gpu_mem")
            text_str = format_memory(avg_mem) if avg_mem is not None else "N/A"
            x_pos = sum(col_widths[:3]) + col_widths[3] / 2 + col_offset
            ax3.text(
                x_pos, y_pos, text_str, fontsize=fontsize, ha="center", va="center"
            )

            # Max train accuracy
            max_train = model_stats.get("max_train_acc")
            text_str = f"{max_train:.3f}" if max_train is not None else "N/A"
            x_pos = sum(col_widths[:4]) + col_widths[4] / 2 + col_offset
            ax3.text(
                x_pos, y_pos, text_str, fontsize=fontsize, ha="center", va="center"
            )

            # Max val accuracy
            max_val = model_stats.get("max_val_acc")
            text_str = f"{max_val:.3f}" if max_val is not None else "N/A"
            x_pos = sum(col_widths[:5]) + col_widths[5] / 2 + col_offset
            ax3.text(
                x_pos, y_pos, text_str, fontsize=fontsize, ha="center", va="center"
            )

    # Only add right column plots if test_data is provided
    if test_data is not None and ax4 is not None and ax5 is not None:
        # Plot 4: Test Performance (Accuracy/Confidence over time) similar to plot_test_performance.py
        plot_test_accuracy_confidence(test_data, ax4, colors, dt, category_col)

        # Plot 5: V1 Response over time (similar to plot_response.py but only for V1)
        plot_v1_response(test_data, ax5, colors, dt, category_col)

    # Final styling - remove grids and spines
    sns.despine(ax=ax1)
    sns.despine(ax=ax2)

    if test_data is not None:
        for ax in [ax4, ax5]:
            sns.despine(ax=ax)
            ax.grid(True, alpha=0.3)

    # Adjust spacing
    plt.subplots_adjust(wspace=0.3, hspace=0.3)

    return fig


def plot_test_accuracy_confidence(
    test_data, ax, colors, dt=None, category_col="model_type"
):
    """
    Create the test accuracy and confidence plot for the top-right panel

    Args:
        test_data: DataFrame with test performance data
        ax: Matplotlib axis to plot on
        colors: Dictionary mapping model types to colors
        dt: Optional time step duration in milliseconds
        category_col: Column name containing model categories
    """
    # Convert time axis if dt is provided
    time_col = "times_index"
    if dt is not None and time_col in test_data.columns:
        test_data = test_data.copy()
        test_data["time_ms"] = test_data[time_col] * dt
        time_col = "time_ms"
        xlabel = "Time (ms)"
    else:
        xlabel = "Time Step"

    # Get unique model types
    model_types = sorted(test_data[category_col].unique())

    # Plot accuracy for each model type
    for model_type in model_types:
        model_data = test_data[test_data[category_col] == model_type]
        if len(model_data) == 0 or "accuracy" not in model_data.columns:
            continue

        # Group by time to get average accuracy
        time_avg_data = model_data.groupby(time_col)["accuracy"].mean().reset_index()

        # Plot accuracy as solid line
        ax.plot(
            time_avg_data[time_col],
            time_avg_data["accuracy"],
            color=colors.get(model_type, "gray"),
            linewidth=2,
            alpha=0.8,
        )

        # Plot confidence if available (dotted line)
        if "confidence_avg" in model_data.columns:
            time_conf_data = (
                model_data.groupby(time_col)["confidence_avg"].mean().reset_index()
            )
            ax.plot(
                time_conf_data[time_col],
                time_conf_data["confidence_avg"],
                color=colors.get(model_type, "gray"),
                linewidth=2,
                linestyle=":",
                alpha=0.8,
            )

    # Add label indicator
    if (
        len(test_data) > 0
        and "label_index" in test_data.columns
        and time_col in test_data.columns
    ):
        y_min, y_max = ax.get_ylim()

        # Get unique combinations of time and label
        label_times = (
            test_data.groupby([time_col, "label_index"])
            .size()
            .reset_index()[[time_col, "label_index"]]
        )

        # Calculate label indicator (black line at bottom when stimulus is present)
        indicator_height = (y_max - y_min) * 0.05
        indicator_base = y_min

        # Create indicator series
        unique_times = sorted(test_data[time_col].unique())
        label_indicator = np.zeros(len(unique_times))

        # Create a mapping from time values to indices
        time_to_idx = {t: i for i, t in enumerate(unique_times)}

        # Mark time points where label is not -1 (indicating presence of stimulus)
        for _, row in label_times.iterrows():
            if row["label_index"] != -1:  # Assuming -1 means no stimulus
                time_val = row[time_col]
                time_idx = time_to_idx.get(time_val)
                if time_idx is not None:
                    label_indicator[time_idx] = indicator_height

        # Plot indicator
        indicator_values = indicator_base + label_indicator

        ax.plot(
            unique_times,
            indicator_values,
            color="black",
            linewidth=2,
            drawstyle="steps-pre",
            alpha=0.6,
        )

    # Add legend for metrics - moved to center center
    from matplotlib.lines import Line2D

    metric_legend = [
        Line2D([0], [0], color="black", linewidth=2, linestyle="-", label="Accuracy"),
        Line2D(
            [0], [0], color="black", linewidth=2, linestyle=":", label="Confidence"
        ),
        Line2D([0], [0], color="black", linewidth=2, alpha=0.6, label="Stimulus"),
    ]
    ax.legend(handles=metric_legend, loc="center", frameon=False)

    # Customize subplot
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Performance")


def plot_v1_response(test_data, ax, colors, dt=None, category_col="model_type"):
    """
    Create the V1 response plot for the bottom-right panel

    Args:
        test_data: DataFrame with layer response data
        ax: Matplotlib axis to plot on
        colors: Dictionary mapping model types to colors
        dt: Optional time step duration in milliseconds
        category_col: Column name containing model categories
    """
    # Look for V1 response column
    v1_columns = [
        col
        for col in test_data.columns
        if "V1_response_avg" in col or "layer1_response_avg" in col
    ]

    if not v1_columns:
        ax.text(
            0.5,
            0.5,
            "No V1 response data available",
            ha="center",
            va="center",
            fontsize=14,
        )
        return

    v1_col = v1_columns[0]  # Use the first V1 column found

    # Convert time axis if dt is provided
    time_col = "times_index"
    if dt is not None and time_col in test_data.columns:
        test_data = test_data.copy()
        test_data["time_ms"] = test_data[time_col] * dt
        time_col = "time_ms"
        xlabel = "Time (ms)"
    else:
        xlabel = "Time Step"

    # Get unique model types
    model_types = sorted(test_data[category_col].unique())

    # Plot V1 response for each model type
    for model_type in model_types:
        model_data = test_data[test_data[category_col] == model_type]
        if len(model_data) == 0 or v1_col not in model_data.columns:
            continue

        # Group by time to get average V1 response
        time_avg_data = model_data.groupby(time_col)[v1_col].mean().reset_index()

        # Plot V1 response
        ax.plot(
            time_avg_data[time_col],
            time_avg_data[v1_col],
            color=colors.get(model_type, "gray"),
            linewidth=2,
            alpha=0.8,
        )

    # Add label indicator (same as in accuracy/confidence plot)
    if (
        len(test_data) > 0
        and "label_index" in test_data.columns
        and time_col in test_data.columns
    ):
        y_min, y_max = ax.get_ylim()

        # Get unique combinations of time and label
        label_times = (
            test_data.groupby([time_col, "label_index"])
            .size()
            .reset_index()[[time_col, "label_index"]]
        )

        # Calculate label indicator
        indicator_height = (y_max - y_min) * 0.05
        indicator_base = y_min

        # Create indicator series
        unique_times = sorted(test_data[time_col].unique())
        label_indicator = np.zeros(len(unique_times))

        # Create a mapping from time values to indices
        time_to_idx = {t: i for i, t in enumerate(unique_times)}

        # Mark time points where label is not -1
        for _, row in label_times.iterrows():
            if row["label_index"] != -1:  # Assuming -1 means no stimulus
                time_val = row[time_col]
                time_idx = time_to_idx.get(time_val)
                if time_idx is not None:
                    label_indicator[time_idx] = indicator_height

        # Plot indicator
        indicator_values = indicator_base + label_indicator

        ax.plot(
            unique_times,
            indicator_values,
            color="black",
            linewidth=2,
            drawstyle="steps-pre",
            alpha=0.6,
            label="Stimulus",
        )

    # Customize subplot
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Avg V1 Response")  # Changed as requested
    # Remove title as requested
def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Plot training accuracy and loss curves with comprehensive model statistics."
    )
    parser.add_argument(
        "--accuracy_csv",
        type=Path,
        required=True,
        help="Path to training/validation accuracy CSV from W&B",
    )
    parser.add_argument(
        "--memory_csv",
        type=Path,
        required=True,
        help="Path to GPU memory allocation CSV from W&B",
    )
    parser.add_argument(
        "--epoch_csv",
        type=Path,
        required=True,
        help="Path to epoch timing CSV from W&B",
    )
    parser.add_argument(
        "--energy_csv",
        type=Path,
        required=True,
        help="Path to energy loss CSV from W&B",
    )
    parser.add_argument(
        "--cross_entropy_csv",
        type=Path,
        required=True,
        help="Path to cross entropy loss CSV from W&B",
    )
    parser.add_argument(
        "--test_data",
        type=Path,
        required=False,
        help="Path to test performance data CSV for third column plots",
    )
    parser.add_argument(
        "--palette",
        type=str,
        default="{'full': '#1f77b4', 'self': '#ff7f0e', 'depthpointwise': '#2ca02c', 'pointdepthwise': '#d62728', 'local': '#9467bd', 'localdepthwise': '#8c564b'}",
        help="Color palette dictionary mapping model types to hex colors",
    )
    parser.add_argument(
        "--ordering",
        type=str,
        default=None,
        help="JSON formatted ordering dictionary for model types",
    )
    parser.add_argument(
        "--naming",
        type=str,
        default=None,
        help="JSON formatted naming dictionary for model types",
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Path to output file"
    )
    parser.add_argument(
        "--category",
        type=str,
        default="model_type",
        help="Column name containing model categories in test data",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=None,
        help="Time step duration in milliseconds for x-axis in test plots",
    )

    args, unknown = parser.parse_known_args()

    # Validate input files
    required_files = [
        (args.accuracy_csv, "Accuracy CSV"),
        (args.memory_csv, "Memory CSV"),
        (args.epoch_csv, "Epoch CSV"),
        (args.energy_csv, "Energy loss CSV"),
        (args.cross_entropy_csv, "Cross entropy loss CSV"),
    ]

    # Only validate test_data if provided
    if args.test_data:
        required_files.append((args.test_data, "Test performance data CSV"))

    for file_path, file_name in required_files:
        if not file_path.exists():
            raise FileNotFoundError(f"{file_name} not found: {file_path}")

    # Create output directory if needed
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Load configuration from command line arguments
    config = load_config_from_args(
        palette_str=args.palette, naming_str=args.naming, ordering_str=args.ordering
    )

    # Load and process data
    print("Loading accuracy data...")
    accuracy_df = load_accuracy_data(args.accuracy_csv)
    print(
        f"Found {len(accuracy_df.model_type.unique())} model types in accuracy data: {sorted(accuracy_df.model_type.unique())}"
    )

    print("Loading memory data...")
    memory_df = load_memory_data(args.memory_csv)
    if not memory_df.empty and "model_type" in memory_df.columns:
        print(
            f"Found {len(memory_df.model_type.unique())} model types in memory data: {sorted(memory_df.model_type.unique())}"
        )
    else:
        print("No memory data processed")

    print("Loading epoch timing data...")
    epoch_df = load_epoch_data(args.epoch_csv)
    if not epoch_df.empty and "model_type" in epoch_df.columns:
        print(
            f"Found {len(epoch_df.model_type.unique())} model types in epoch data: {sorted(epoch_df.model_type.unique())}"
        )
    else:
        print("No epoch data processed")

    print("Loading energy loss data...")
    energy_loss_df = load_loss_data(args.energy_csv, "energy")
    print(
        f"Found {len(energy_loss_df.model_type.unique())} model types in energy loss data: {sorted(energy_loss_df.model_type.unique())}"
    )

    print("Loading cross entropy loss data...")
    cross_entropy_loss_df = load_loss_data(args.cross_entropy_csv, "cross_entropy")
    print(
        f"Found {len(cross_entropy_loss_df.model_type.unique())} model types in cross entropy loss data: {sorted(cross_entropy_loss_df.model_type.unique())}"
    )

    # Load test performance data if provided
    test_data_df = None
    if args.test_data:
        print(f"Loading test performance data from: {args.test_data}")
        test_data_df = pd.read_csv(args.test_data)
        print(f"Test data shape: {test_data_df.shape}")
        if args.category in test_data_df.columns:
            print(
                f"Found {len(test_data_df[args.category].unique())} model types in test data: {sorted(test_data_df[args.category].unique())}"
            )
        else:
            print(f"Warning: Category column '{args.category}' not found in test data")

    # Generate plot
    fig = plot_training_losses(
        accuracy_df,
        energy_loss_df,
        cross_entropy_loss_df,
        memory_df,
        epoch_df,
        args.palette,
        test_data=test_data_df,
        dt=args.dt,
        category_col=args.category,
        config=config,
    )

    # Save plot
    fig.savefig(args.output, bbox_inches="tight", dpi=300)
    print(f"Plot saved to: {args.output}")

    plt.close(fig)


if __name__ == "__main__":
    main()
