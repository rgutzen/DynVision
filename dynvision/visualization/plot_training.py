"""Plot training accuracy over epochs and loss curves with comprehensive model statistics."""

import argparse
import ast
from pathlib import Path
from typing import Tuple, Dict, List
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.ndimage import uniform_filter1d

from dynvision.utils import load_df

# Global model order for consistent plotting across all subplots
MODEL_ORDER = [
    "self",
    "full",
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


def parse_palette(palette_str: str) -> Dict[str, str]:
    """
    Parse palette string into dictionary.

    Args:
        palette_str: String representation of color palette dictionary

    Returns:
        Dictionary mapping model names to colors
    """
    try:
        # Try to parse as literal dictionary
        return ast.literal_eval(palette_str)
    except (ValueError, SyntaxError):
        # Fallback to default palette if parsing fails
        print(
            f"Warning: Could not parse palette '{palette_str}', using default colors"
        )
        return {
            "full": "#1f77b4",  # Blue
            "self": "#ff7f0e",  # Orange
            "depthpointwise": "#2ca02c",  # Green
            "pointdepthwise": "#d62728",  # Red
            "local": "#9467bd",  # Purple
            "localdepthwise": "#8c564b",  # Brown
        }


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
    return f"{millions:.1f}M"


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
    parameter_counts = {
        "depthpointwise": 6_043_240,
        "pointdepthwise": 6_043_240,
        "full": 9_059_494,
        "self": 5_663_070,
        "local": 5_661_156,
        "localdepthwise": 5_672_110,
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
    palette: str,
    figsize: Tuple[int, int] = (16, 8),
) -> plt.Figure:
    """
    Create two-panel plot: training/validation accuracy + loss curves (left) and comprehensive stats table (right).

    Args:
        accuracy_df: Processed accuracy data
        energy_loss_df: Processed energy loss data
        cross_entropy_loss_df: Processed cross entropy loss data
        memory_df: Processed memory data
        epoch_df: Processed epoch timing data
        palette: Color palette string (dictionary format)
        figsize: Figure dimensions

    Returns:
        Matplotlib figure
    """
    # Set style
    sns.set_context("talk")
    sns.set_style("ticks")

    # Parse color palette
    colors = parse_palette(palette)

    # Create figure with 2x2 subplot layout but use custom positioning
    fig = plt.figure(figsize=figsize)

    # Left half: 2 subplots stacked vertically
    ax1 = fig.add_subplot(2, 2, 1)  # Top left - Training/Validation accuracy
    ax2 = fig.add_subplot(2, 2, 3)  # Bottom left - Loss curves

    # Right half: Single large area for statistics table (increased width and adjusted position)
    ax3 = fig.add_axes([0.52, 0.1, 0.46, 0.8])  # [left, bottom, width, height]

    # Get available model types
    all_models = set()
    for df in [accuracy_df, energy_loss_df, cross_entropy_loss_df]:
        if not df.empty and "model_type" in df.columns:
            all_models.update(df.model_type.unique())

    # Filter MODEL_ORDER to only include available models
    available_model_order = [m for m in MODEL_ORDER if m in all_models]

    # Plot 1: Training and validation accuracy over epochs (top left)

    # Plot training accuracy (solid lines) without smoothing
    for model_type in available_model_order:
        if model_type in accuracy_df.model_type.values:
            model_data = accuracy_df[accuracy_df.model_type == model_type]
            train_data = model_data.dropna(subset=["train_accuracy"]).sort_values(
                "epoch"
            )
            if not train_data.empty:
                ax1.plot(
                    train_data.epoch,
                    train_data.train_accuracy,
                    color=colors.get(model_type, "#000000"),
                    linewidth=2,
                    alpha=0.8,
                    label=(
                        model_type if model_type == available_model_order[0] else ""
                    ),  # Only label first for legend
                )

    # Plot validation accuracy (dotted lines) without smoothing
    for model_type in available_model_order:
        if model_type in accuracy_df.model_type.values:
            model_data = accuracy_df[accuracy_df.model_type == model_type]
            val_data = model_data.dropna(subset=["val_accuracy"]).sort_values("epoch")
            if not val_data.empty:
                ax1.plot(
                    val_data.epoch,
                    val_data.val_accuracy,
                    color=colors.get(model_type, "#000000"),
                    linewidth=2,
                    linestyle=":",
                    alpha=0.8,
                )

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")

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

    # Plot cross entropy loss (solid lines) using seaborn
    ce_plot_data = []
    for model_type in available_model_order:
        if model_type in cross_entropy_loss_df.model_type.values:
            model_data = cross_entropy_loss_df[
                cross_entropy_loss_df.model_type == model_type
            ].sort_values("epoch")
            if not model_data.empty:
                # Smooth the loss data
                smoothed_loss = smooth_data(
                    model_data.cross_entropy_loss, window_size=10
                )
                for epoch, loss_val in zip(model_data.epoch, smoothed_loss):
                    ce_plot_data.append(
                        {"epoch": epoch, "loss": loss_val, "model_type": model_type}
                    )

    if ce_plot_data:
        ce_plot_df = pd.DataFrame(ce_plot_data)
        sns.lineplot(
            data=ce_plot_df,
            x="epoch",
            y="loss",
            hue="model_type",
            palette=colors,
            ax=ax2,
            linewidth=2,
            alpha=0.8,
        )

    # Plot energy loss (dashed lines) using matplotlib for line style control
    for model_type in available_model_order:
        if model_type in energy_loss_df.model_type.values:
            model_data = energy_loss_df[
                energy_loss_df.model_type == model_type
            ].sort_values("epoch")
            if not model_data.empty:
                # Smooth the loss data
                smoothed_loss = smooth_data(model_data.energy_loss, window_size=10)
                ax2.plot(
                    model_data.epoch,
                    smoothed_loss,
                    color=colors.get(model_type, "#000000"),
                    linewidth=2,
                    linestyle="--",
                    alpha=0.8,
                )

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")

    # Add legend for loss types
    loss_legend = [
        Line2D(
            [0], [0], color="black", linewidth=2, linestyle="-", label="Cross Entropy"
        ),
        Line2D([0], [0], color="black", linewidth=2, linestyle="--", label="Energy"),
    ]
    ax2.legend(handles=loss_legend, loc="upper right", frameon=False)

    # Plot 3: Comprehensive Statistics table (right half)
    ax3.axis("off")  # Turn off axis for text display

    # Calculate statistics
    stats = calculate_model_statistics(
        accuracy_df, energy_loss_df, cross_entropy_loss_df, memory_df, epoch_df
    )

    # Create table format with adjusted column widths for better spacing
    col_widths = [0.17, 0.15, 0.18, 0.15, 0.17, 0.18]  # 6 columns with more spacing
    col_offset = -0.02  # Reduced offset for better alignment
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
            y_pos = y_start - (row_idx + 1) * row_height
            model_stats = stats[model_type]
            model_label = model_type.replace("wise", "w.")

            # Model name (colored)
            x_pos = col_widths[0] / 2 + col_offset
            ax3.text(
                x_pos,
                y_pos,
                model_label,
                fontsize=13,
                color=colors.get(model_type, "#000000"),  # Use black as fallback
                fontweight="bold",
                ha="center",
                va="center",
            )

            # Number of parameters
            n_params = model_stats.get("n_parameters")
            text_str = format_parameter_count(n_params)
            x_pos = col_widths[0] + col_widths[1] / 2 + col_offset
            ax3.text(x_pos, y_pos, text_str, fontsize=13, ha="center", va="center")

            # Average runtime per epoch
            avg_time = model_stats.get("avg_epoch_time")
            text_str = format_time(avg_time) if avg_time is not None else "N/A"
            x_pos = sum(col_widths[:2]) + col_widths[2] / 2 + col_offset
            ax3.text(x_pos, y_pos, text_str, fontsize=13, ha="center", va="center")

            # Average GPU memory
            avg_mem = model_stats.get("avg_gpu_mem")
            text_str = format_memory(avg_mem) if avg_mem is not None else "N/A"
            x_pos = sum(col_widths[:3]) + col_widths[3] / 2 + col_offset
            ax3.text(x_pos, y_pos, text_str, fontsize=13, ha="center", va="center")

            # Max train accuracy
            max_train = model_stats.get("max_train_acc")
            text_str = f"{max_train:.3f}" if max_train is not None else "N/A"
            x_pos = sum(col_widths[:4]) + col_widths[4] / 2 + col_offset
            ax3.text(x_pos, y_pos, text_str, fontsize=13, ha="center", va="center")

            # Max val accuracy
            max_val = model_stats.get("max_val_acc")
            text_str = f"{max_val:.3f}" if max_val is not None else "N/A"
            x_pos = sum(col_widths[:5]) + col_widths[5] / 2 + col_offset
            ax3.text(x_pos, y_pos, text_str, fontsize=13, ha="center", va="center")

    # Final styling - remove grids and spines
    for ax in [ax1, ax2]:
        sns.despine(ax=ax)
        ax.grid(False)

    # Adjust spacing
    plt.subplots_adjust(wspace=0.3, hspace=0.3)

    return fig


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
        "--palette",
        type=str,
        default="{'full': '#1f77b4', 'self': '#ff7f0e', 'depthpointwise': '#2ca02c', 'pointdepthwise': '#d62728', 'local': '#9467bd', 'localdepthwise': '#8c564b'}",
        help="Color palette dictionary mapping model types to hex colors",
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Path to output file"
    )

    args = parser.parse_args()

    # Validate input files
    required_files = [
        (args.accuracy_csv, "Accuracy CSV"),
        (args.memory_csv, "Memory CSV"),
        (args.epoch_csv, "Epoch CSV"),
        (args.energy_csv, "Energy loss CSV"),
        (args.cross_entropy_csv, "Cross entropy loss CSV"),
    ]

    for file_path, file_name in required_files:
        if not file_path.exists():
            raise FileNotFoundError(f"{file_name} not found: {file_path}")

    # Create output directory if needed
    args.output.parent.mkdir(parents=True, exist_ok=True)

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

    # Generate plot
    fig = plot_training_losses(
        accuracy_df,
        energy_loss_df,
        cross_entropy_loss_df,
        memory_df,
        epoch_df,
        args.palette,
    )

    # Save plot
    fig.savefig(args.output, bbox_inches="tight", dpi=300)
    print(f"Plot saved to: {args.output}")

    plt.close(fig)


if __name__ == "__main__":
    main()
