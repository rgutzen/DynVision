"""Plot training accuracy over epochs and testing accuracy over time steps."""

import argparse
from pathlib import Path
from typing import Tuple, Dict, List
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.ndimage import uniform_filter1d
from scipy.special import softmax

from dynvision.utils import load_df


def parse_model_identifier_from_column(column_name: str) -> str:
    """
    Extract model identifier from W&B column name.

    Args:
        column_name: Full column name from W&B export

    Returns:
        Model identifier (e.g., 'full', 'pointdepthwise', etc.)
    """
    # Extract rctype from the column name
    match = re.search(r"rctype=([^+]+)", column_name)
    if match:
        return match.group(1)
    return "unknown"


def load_training_data(training_csv_path: Path) -> pd.DataFrame:
    """
    Load and process training accuracy data from W&B export.

    Args:
        training_csv_path: Path to training CSV file

    Returns:
        Processed DataFrame with columns: epoch, model_type, train_accuracy
    """
    df = pd.read_csv(training_csv_path)

    # Find all train_accuracy columns (excluding MIN/MAX variants)
    accuracy_columns = [
        col
        for col in df.columns
        if "train_accuracy" in col and "__MIN" not in col and "__MAX" not in col
    ]

    processed_data = []

    for col in accuracy_columns:
        model_type = parse_model_identifier_from_column(col)

        # Create records for this model
        for idx, row in df.iterrows():
            if not pd.isna(row[col]):  # Only include non-NaN values
                processed_data.append(
                    {
                        "epoch": row["epoch"],
                        "model_type": model_type,
                        "train_accuracy": row[col],
                    }
                )

    return pd.DataFrame(processed_data)


def load_validation_data(validation_csv_path: Path) -> pd.DataFrame:
    """
    Load and process validation accuracy data from W&B export.

    Args:
        validation_csv_path: Path to validation CSV file

    Returns:
        Processed DataFrame with columns: epoch, model_type, val_accuracy
    """
    df = pd.read_csv(validation_csv_path)

    # Find all val_accuracy columns (excluding MIN/MAX variants)
    accuracy_columns = [
        col
        for col in df.columns
        if "val_accuracy" in col and "__MIN" not in col and "__MAX" not in col
    ]

    processed_data = []

    for col in accuracy_columns:
        model_type = parse_model_identifier_from_column(col)

        # Create records for this model
        for idx, row in df.iterrows():
            if not pd.isna(row[col]):  # Only include non-NaN values
                processed_data.append(
                    {
                        "epoch": row["epoch"],
                        "model_type": model_type,
                        "val_accuracy": row[col],
                    }
                )

    return pd.DataFrame(processed_data)


def load_energy_loss_data(energy_csv_path: Path) -> pd.DataFrame:
    """
    Load and process energy loss data from W&B export.

    Args:
        energy_csv_path: Path to energy loss CSV file

    Returns:
        Processed DataFrame with columns: step, model_type, energy_loss
    """
    df = pd.read_csv(energy_csv_path)

    # Find all energy loss columns (excluding MIN/MAX variants)
    loss_columns = [
        col
        for col in df.columns
        if "loss/EnergyLoss" in col and "__MIN" not in col and "__MAX" not in col
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
                        "energy_loss": row[col],
                    }
                )

    return pd.DataFrame(processed_data)


def load_cross_entropy_loss_data(cross_entropy_csv_path: Path) -> pd.DataFrame:
    """
    Load and process cross entropy loss data from W&B export.

    Args:
        cross_entropy_csv_path: Path to cross entropy loss CSV file

    Returns:
        Processed DataFrame with columns: step, model_type, cross_entropy_loss
    """
    df = pd.read_csv(cross_entropy_csv_path)

    # Find all cross entropy loss columns (excluding MIN/MAX variants)
    loss_columns = [
        col
        for col in df.columns
        if "loss/CrossEntropyLoss" in col and "__MIN" not in col and "__MAX" not in col
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
                        "cross_entropy_loss": row[col],
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


def load_testing_data(testing_csv_path: Path) -> pd.DataFrame:
    """
    Load testing data from single CSV file with new format.

    Args:
        testing_csv_path: Path to testing CSV file

    Returns:
        DataFrame with model_type column added
    """
    print(f"Loading testing data from {testing_csv_path.name}...")

    # Load the CSV
    df = pd.read_csv(testing_csv_path)

    # The rctype column contains the model type information
    df["model_type"] = df["rctype"]

    print(f"Loaded testing data with {len(df)} records")
    print(f"Found model types: {sorted(df.model_type.unique())}")

    return df


def calculate_testing_accuracy_over_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate testing accuracy at each time step for each model type.

    Args:
        df: DataFrame containing testing data with model_type column

    Returns:
        DataFrame with columns: times_index, model_type, test_accuracy
    """
    # Use all data, don't filter for valid labels
    accuracy_by_time = []

    for model_type in sorted(df.model_type.unique()):
        model_data = df[df.model_type == model_type]

        for time_step in sorted(model_data.times_index.unique()):
            time_data = model_data[model_data.times_index == time_step]

            if len(time_data) > 0:
                accuracy = (time_data.guess_index == time_data.label_index).mean()
                accuracy_by_time.append(
                    {
                        "times_index": time_step,
                        "model_type": model_type,
                        "test_accuracy": accuracy,
                    }
                )

    return pd.DataFrame(accuracy_by_time)


def calculate_confidence_over_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate model confidence using classifier power as proxy for each model type.

    Args:
        df: DataFrame containing testing data with model_type column

    Returns:
        DataFrame with columns: times_index, model_type, confidence
    """
    confidence_by_time = []

    # Check if classifier_power column exists
    if "classifier_power" not in df.columns:
        print(
            "Warning: classifier_power column not found, skipping confidence calculation"
        )
        return pd.DataFrame(columns=["times_index", "model_type", "confidence"])

    for model_type in sorted(df.model_type.unique()):
        model_data = df[df.model_type == model_type]

        for time_step in sorted(model_data.times_index.unique()):
            time_data = model_data[model_data.times_index == time_step]

            if len(time_data) > 0:
                # Use classifier_power as a proxy for confidence
                # Normalize to 0-1 range within this time step
                powers = time_data["classifier_power"].values
                if len(powers) > 0 and powers.max() > powers.min():
                    normalized_powers = (powers - powers.min()) / (
                        powers.max() - powers.min()
                    )
                    avg_confidence = normalized_powers.mean()
                else:
                    avg_confidence = 0.5  # Default if no variation

                confidence_by_time.append(
                    {
                        "times_index": time_step,
                        "model_type": model_type,
                        "confidence": avg_confidence,
                    }
                )

    return pd.DataFrame(confidence_by_time)


def calculate_label_indicator(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate label indicator (step function) at each time step.

    Args:
        df: DataFrame containing testing data

    Returns:
        DataFrame with times_index and label_indicator (0.2 if labels valid, 0 if invalid)
    """
    # Get first model's data to determine label validity at each time step
    first_model = df.model_type.iloc[0]
    model_data = df[df.model_type == first_model]

    indicator_data = []
    for time_step in sorted(model_data.times_index.unique()):
        time_data = model_data[model_data.times_index == time_step]

        # Check if any labels are valid (>= 0) at this time step
        valid_labels = (time_data.label_index >= 0).any()
        indicator_value = 0.2 if valid_labels else 0.0

        indicator_data.append(
            {"times_index": time_step, "label_indicator": indicator_value}
        )

    return pd.DataFrame(indicator_data)


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


def calculate_model_statistics(
    training_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    energy_loss_df: pd.DataFrame,
    cross_entropy_loss_df: pd.DataFrame,
) -> Dict[str, Dict[str, float]]:
    """
    Calculate max train accuracy and max val accuracy for each model.

    Args:
        training_df: Training accuracy data
        validation_df: Validation accuracy data
        energy_loss_df: Energy loss data
        cross_entropy_loss_df: Cross entropy loss data

    Returns:
        Dictionary with model statistics
    """
    stats = {}

    # Parameter counts for each model type (including new ones)
    parameter_counts = {
        "depthpointwise": 5_425_195,
        "pointdepthwise": 5_425_195,
        "full": 8_441_449,
        "self": 5_043_047,
        # "local": None,  # Parameter count not provided
        # "localdepthwise": None,  # Parameter count not provided
    }

    # Get unique model types
    all_models = set()
    if not training_df.empty:
        all_models.update(training_df.model_type.unique())
    if not validation_df.empty:
        all_models.update(validation_df.model_type.unique())
    if not energy_loss_df.empty:
        all_models.update(energy_loss_df.model_type.unique())
    if not cross_entropy_loss_df.empty:
        all_models.update(cross_entropy_loss_df.model_type.unique())

    for model_type in all_models:
        stats[model_type] = {}

        # Get max train accuracy for this model
        if not training_df.empty and model_type in training_df.model_type.values:
            model_train_data = training_df[training_df.model_type == model_type]
            max_train_acc = model_train_data.train_accuracy.max()
            stats[model_type]["max_train_acc"] = max_train_acc
        else:
            stats[model_type]["max_train_acc"] = None

        # Get max validation accuracy
        if not validation_df.empty and model_type in validation_df.model_type.values:
            model_val_data = validation_df[validation_df.model_type == model_type]
            max_val_acc = model_val_data.val_accuracy.max()
            stats[model_type]["max_val_acc"] = max_val_acc
        else:
            stats[model_type]["max_val_acc"] = None

        # Add parameter count
        stats[model_type]["n_parameters"] = parameter_counts.get(model_type, None)

    return stats


def plot_training_testing_losses(
    training_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    testing_df: pd.DataFrame,
    energy_loss_df: pd.DataFrame,
    cross_entropy_loss_df: pd.DataFrame,
    figsize: Tuple[int, int] = (16, 8),
) -> plt.Figure:
    """
    Create three-panel plot with statistics: training/validation accuracy, testing accuracy, loss curves, and stats.

    Args:
        training_df: Processed training data
        validation_df: Processed validation data
        testing_df: DataFrame containing testing data with model_type
        energy_loss_df: Processed energy loss data
        cross_entropy_loss_df: Processed cross entropy loss data
        figsize: Figure dimensions

    Returns:
        Matplotlib figure
    """
    # Set style
    sns.set_context("talk")
    sns.set_style("ticks")

    # Create figure with custom subplot positions to make stats table larger
    fig = plt.figure(figsize=figsize)

    # Define subplot positions [left, bottom, width, height]
    ax1 = fig.add_subplot(2, 2, 1)  # Top left - Training/Validation
    ax2 = fig.add_subplot(2, 2, 2)  # Top right - Testing
    ax3 = fig.add_subplot(2, 2, 3)  # Bottom left - Loss curves

    # Make stats table wider by taking more space
    ax4 = fig.add_axes(
        [0.58, 0.1, 0.4, 0.4]
    )  # [left, bottom, width, height] - bigger stats table

    # Define model order and custom colors (expanded for new recurrence types)
    model_order = [
        "full",
        "self",
        "depthpointwise",
        "pointdepthwise",
        # "local",
        # "localdepthwise",
    ]
    colors = {
        "full": "#DAA520",  # Golden
        "self": "#228B22",  # Forest green
        "depthpointwise": "#4682B4",  # Steel blue
        "pointdepthwise": "#9932CC",  # Dark orchid
        # "local": "#FF6347",  # Tomato red
        # "localdepthwise": "#20B2AA",  # Light sea green
    }

    # Get available model types from all datasets
    training_models = (
        set(training_df.model_type.unique()) if not training_df.empty else set()
    )
    validation_models = (
        set(validation_df.model_type.unique()) if not validation_df.empty else set()
    )
    testing_models = (
        set(testing_df.model_type.unique()) if not testing_df.empty else set()
    )
    energy_loss_models = (
        set(energy_loss_df.model_type.unique()) if not energy_loss_df.empty else set()
    )
    cross_entropy_loss_models = (
        set(cross_entropy_loss_df.model_type.unique())
        if not cross_entropy_loss_df.empty
        else set()
    )
    all_models = (
        training_models.union(validation_models)
        .union(testing_models)
        .union(energy_loss_models)
        .union(cross_entropy_loss_models)
    )

    # Filter model_order to only include available models
    available_model_order = [m for m in model_order if m in all_models]

    # Plot 1: Training and validation accuracy over epochs (top left)
    for model_type in available_model_order:
        # Plot training accuracy (solid line)
        if model_type in training_models:
            model_data = training_df[training_df.model_type == model_type]
            ax1.plot(
                model_data.epoch,
                model_data.train_accuracy,
                color=colors[model_type],
                marker="o",
                markersize=4,
                linewidth=2,
                linestyle="-",
                label=model_type,
            )

        # Plot validation accuracy (dotted line)
        if model_type in validation_models:
            model_data = validation_df[validation_df.model_type == model_type]
            ax1.plot(
                model_data.epoch,
                model_data.val_accuracy,
                color=colors[model_type],
                marker="s",
                markersize=4,
                linewidth=2,
                linestyle=":",
                alpha=0.7,
            )

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")

    # Add legend for Training/Validation in lower left of this plot
    from matplotlib.lines import Line2D

    train_val_legend = [
        Line2D([0], [0], color="black", linewidth=2, linestyle="-", label="Training"),
        Line2D(
            [0], [0], color="black", linewidth=2, linestyle=":", label="Validation"
        ),
    ]
    ax1.legend(handles=train_val_legend, loc="lower left", frameon=False)

    # Plot 2: Testing accuracy and confidence over time steps (top right)
    testing_accuracy_df = calculate_testing_accuracy_over_time(testing_df)
    confidence_df = calculate_confidence_over_time(testing_df)
    label_indicator_df = calculate_label_indicator(testing_df)

    for model_type in available_model_order:
        # Plot accuracy (solid line)
        if model_type in testing_models:
            model_data = testing_accuracy_df[
                testing_accuracy_df.model_type == model_type
            ]
            ax2.plot(
                model_data.times_index,
                model_data.test_accuracy,
                color=colors[model_type],
                marker="o",
                markersize=4,
                linewidth=2,
                linestyle="-",
            )

        # Plot confidence (dashed line) - only if confidence data exists
        if not confidence_df.empty and model_type in testing_models:
            model_data = confidence_df[confidence_df.model_type == model_type]
            if not model_data.empty:
                ax2.plot(
                    model_data.times_index,
                    model_data.confidence,
                    color=colors[model_type],
                    linewidth=2,
                    linestyle="--",
                    alpha=0.7,
                )

    # Add step function for label indicator
    ax2.plot(
        label_indicator_df.times_index,
        label_indicator_df.label_indicator,
        color="black",
        linewidth=3,
        drawstyle="steps-mid",
    )

    # Add "input presentation" text below the upper part of the step function
    # Find the middle of the high part of the step function
    high_indices = label_indicator_df[label_indicator_df.label_indicator == 0.2][
        "times_index"
    ]
    if not high_indices.empty:
        mid_point = (high_indices.min() + high_indices.max()) / 2
        ax2.text(
            mid_point,
            0.1,
            "input presentation",
            ha="center",
            va="center",
            fontsize=11,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

    ax2.set_xlabel("Time Step")
    ax2.set_ylabel("Testing Performance")

    # Add legend for accuracy/confidence in center right
    legend_elements = [
        Line2D([0], [0], color="black", linewidth=2, linestyle="-", label="Accuracy")
    ]
    if not confidence_df.empty:
        legend_elements.append(
            Line2D(
                [0],
                [0],
                color="black",
                linewidth=2,
                linestyle="--",
                label="Confidence",
            )
        )

    ax2.legend(handles=legend_elements, loc="center right", frameon=False)

    # Plot 3: Loss curves over training steps (bottom left)
    for model_type in available_model_order:
        # Plot cross entropy loss (solid line)
        if model_type in cross_entropy_loss_models:
            model_data = cross_entropy_loss_df[
                cross_entropy_loss_df.model_type == model_type
            ].sort_values("epoch")
            # Smooth the loss data
            smoothed_loss = smooth_data(model_data.cross_entropy_loss, window_size=10)
            ax3.plot(
                model_data.epoch,
                smoothed_loss,
                color=colors[model_type],
                linewidth=2,
                linestyle="-",
                label=model_type if model_type not in energy_loss_models else "",
            )

        # Plot energy loss (dashed line)
        if model_type in energy_loss_models:
            model_data = energy_loss_df[
                energy_loss_df.model_type == model_type
            ].sort_values("epoch")
            # Smooth the loss data
            smoothed_loss = smooth_data(model_data.energy_loss, window_size=10)
            ax3.plot(
                model_data.epoch,
                smoothed_loss,
                color=colors[model_type],
                linewidth=2,
                linestyle="--",
                alpha=0.7,
            )

    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Loss")

    # Add legend for loss types
    loss_legend = [
        Line2D(
            [0], [0], color="black", linewidth=2, linestyle="-", label="Cross Entropy"
        ),
        Line2D([0], [0], color="black", linewidth=2, linestyle="--", label="Energy"),
    ]
    ax3.legend(handles=loss_legend, loc="upper right", frameon=False)

    # Plot 4: Statistics table (bottom right) - bigger and larger fonts
    ax4.axis("off")  # Turn off axis for text display

    # Calculate statistics
    stats = calculate_model_statistics(
        training_df, validation_df, energy_loss_df, cross_entropy_loss_df
    )

    # Create table format with larger space
    col_widths = [0.25, 0.22, 0.26, 0.27]  # 4 columns
    col_offset = -0.2
    col_headers = [
        "Model",
        "# Params",
        "Max Train\nAccuracy",
        "Max Val\nAccuracy",
    ]

    # Header row - larger font
    y_start = 0.7
    row_height = 0.16

    for i, header in enumerate(col_headers):
        x_pos = sum(col_widths[:i]) + col_offset
        ax4.text(
            x_pos + col_widths[i] / 2,
            y_start,
            header,
            fontsize=15,
            fontweight="bold",
            ha="center",
            va="center",
        )

    # Data rows - larger font
    for row_idx, model_type in enumerate(available_model_order):
        if model_type in stats:
            y_pos = y_start - (row_idx + 1) * row_height
            model_stats = stats[model_type]
            model_label = model_type.replace("wise", "w.")

            # Model name (colored, larger font)
            x_pos = col_widths[0] / 2 + col_offset
            ax4.text(
                x_pos,
                y_pos,
                model_label,
                fontsize=15,
                color=colors[model_type],
                fontweight="bold",
                ha="center",
                va="center",
            )

            # Number of parameters (larger font, formatted)
            n_params = model_stats.get("n_parameters")
            text_str = format_parameter_count(n_params)
            x_pos = col_widths[0] + col_widths[1] / 2 + col_offset
            ax4.text(x_pos, y_pos, text_str, fontsize=15, ha="center", va="center")

            # Max train accuracy (larger font)
            max_train = model_stats.get("max_train_acc")
            text_str = f"{max_train:.3f}" if max_train else "N/A"
            x_pos = sum(col_widths[:2]) + col_widths[2] / 2 + col_offset
            ax4.text(x_pos, y_pos, text_str, fontsize=15, ha="center", va="center")

            # Max val accuracy (larger font)
            max_val = model_stats.get("max_val_acc")
            text_str = f"{max_val:.3f}" if max_val else "N/A"
            x_pos = sum(col_widths[:3]) + col_widths[3] / 2 + col_offset
            ax4.text(x_pos, y_pos, text_str, fontsize=15, ha="center", va="center")

    # Final styling - remove grids and spines
    for ax in [ax1, ax2, ax3]:
        sns.despine(ax=ax)
        ax.grid(False)

    # Adjust spacing to reduce horizontal space between plots
    plt.subplots_adjust(wspace=0.25, hspace=0.3)

    return fig


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Plot training and testing accuracy comparison with loss curves."
    )
    parser.add_argument(
        "--training_csv",
        type=Path,
        required=True,
        help="Path to training accuracy CSV from W&B",
    )
    parser.add_argument(
        "--validation_csv",
        type=Path,
        required=True,
        help="Path to validation accuracy CSV from W&B",
    )
    parser.add_argument(
        "--testing_csv",
        type=Path,
        required=True,
        help="Path to testing CSV file with new format",
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
        "--output", type=Path, required=True, help="Path to output file"
    )

    args = parser.parse_args()

    # Validate input files
    required_files = [
        (args.training_csv, "Training CSV"),
        (args.validation_csv, "Validation CSV"),
        (args.testing_csv, "Testing CSV"),
        (args.energy_csv, "Energy loss CSV"),
        (args.cross_entropy_csv, "Cross entropy loss CSV"),
    ]

    for file_path, file_name in required_files:
        if not file_path.exists():
            raise FileNotFoundError(f"{file_name} not found: {file_path}")

    # Create output directory if needed
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Load and process data
    print("Loading training data...")
    training_df = load_training_data(args.training_csv)
    print(
        f"Found {len(training_df.model_type.unique())} model types in training data: {sorted(training_df.model_type.unique())}"
    )

    print("Loading validation data...")
    validation_df = load_validation_data(args.validation_csv)
    print(
        f"Found {len(validation_df.model_type.unique())} model types in validation data: {sorted(validation_df.model_type.unique())}"
    )

    print("Loading energy loss data...")
    energy_loss_df = load_energy_loss_data(args.energy_csv)
    print(
        f"Found {len(energy_loss_df.model_type.unique())} model types in energy loss data: {sorted(energy_loss_df.model_type.unique())}"
    )

    print("Loading cross entropy loss data...")
    cross_entropy_loss_df = load_cross_entropy_loss_data(args.cross_entropy_csv)
    print(
        f"Found {len(cross_entropy_loss_df.model_type.unique())} model types in cross entropy loss data: {sorted(cross_entropy_loss_df.model_type.unique())}"
    )

    print("Loading testing data...")
    testing_df = load_testing_data(args.testing_csv)
    print(
        f"Found {len(testing_df.model_type.unique())} model types in testing data: {sorted(testing_df.model_type.unique())}"
    )

    # Generate plot
    fig = plot_training_testing_losses(
        training_df, validation_df, testing_df, energy_loss_df, cross_entropy_loss_df
    )

    # Save plot
    fig.savefig(args.output, bbox_inches="tight", dpi=300)

    plt.close(fig)


if __name__ == "__main__":
    main()
