"""Plot training overview with training/validation accuracy, losses, and test performance.

This module creates a comprehensive training visualization with:
- Left column: Training/validation accuracy (top), energy/cross-entropy loss (bottom)
- Right column: Test performance panel (top), layer response ridge plots (bottom)
- Horizontal category legend between left panels
"""

import argparse
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from dynvision.utils.visualization_utils import (
    load_config_from_args,
    get_display_name,
    save_plot,
)
from dynvision.visualization.plot_responses import (
    _plot_accuracy_panel,
    _plot_response_ridges,
    _extract_dimension_values,
    _get_colors_for_dimension,
    _get_dimension_key,
    _normalize_dimension,
    _validate_dimensions,
    _format_legend_label,
    FORMATTING as RESPONSES_FORMATTING,
)

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel("INFO")

# Training plot layout configuration
TRAINING_LAYOUT = {
    # Figure dimensions
    "figure_width": 14,
    "figure_height": 14,
    # Column layout (relative coordinates 0-1)
    "left_column_width": 0.45,  # Configurable
    "right_column_width": 0.50,
    "column_spacing": 0.08,
    "left_margin": 0.06,
    "right_margin": 0.02,
    # Left column vertical layout
    "top_margin": 0.05,
    "accuracy_height": 0.35,
    "accuracy_bottom_margin": 0.02,
    "legend_height": 0.1,
    "legend_top_margin": 0.02,  # Space above legend
    "legend_bottom_margin": 0.02,  # Space below legend (increased to move legend down)
    "loss_height": 0.35,
    "loss_bottom_margin": 0.05,
    # Right column follows plot_responses layout
    "performance_height": 0.2,
    "performance_pad": 0.05,
    "ridge_legend_height": 0.08,
    "ridge_legend_pad": 0.01,
    "ridge_overlap": 0.15,  # Overlap fraction between ridge plots
    # Panel letters
    "panel_letter_offset_x": -0.05,
    "panel_letter_offset_y": 0.015,
}

# Formatting configuration (inherit from plot_responses)
TRAINING_FORMATTING = {
    **RESPONSES_FORMATTING,
    "fontsize_panel_label": 18,
    "max_global_ymax": 1.5,  # Increased from 1.5 to accommodate higher response values
}


def _auto_detect_category_key(df: pd.DataFrame) -> Optional[str]:
    """Auto-detect category key from grouped W&B CSV column names.

    Looks for columns with pattern: "<key>: <value> - <metric>"

    Args:
        df: DataFrame with W&B export columns

    Returns:
        Detected category key or None
    """
    # Pattern: "category_key: category_value - metric_name"
    pattern = r"^([^:]+):\s*[^-]+-\s*.+$"

    for col in df.columns:
        match = re.match(pattern, col)
        if match:
            category_key = match.group(1).strip()
            logger.info(
                f"Auto-detected category key: '{category_key}' from column '{col}'"
            )
            return category_key

    logger.warning("Could not auto-detect category key from column names")
    return None


def _extract_category_value_from_column(
    column_name: str, category_key: str
) -> Optional[str]:
    """Extract category value from grouped W&B column name.

    Args:
        column_name: Column name like "energy_loss_weight: 0.05 - train_accuracy"
        category_key: Category key like "energy_loss_weight"

    Returns:
        Category value like "0.05" or None if not found
    """
    # Pattern: "category_key: category_value - metric_name"
    pattern = rf"^{re.escape(category_key)}:\s*([^-]+)\s*-\s*.+$"
    match = re.match(pattern, column_name)

    if match:
        return match.group(1).strip()
    return None


def _standardize_category_value(value: str) -> str:
    """Standardize category value formatting consistently.

    Converts numeric strings to float for consistent comparison.
    E.g., "0", "0.0", "0.00" all become 0.0
    """
    value_str = str(value).strip()

    # Handle boolean-like values consistently
    if value_str.lower() in ["true", "false"]:
        return value_str.lower()

    # Try to convert to numeric for consistent comparison
    try:
        return float(value_str)
    except (ValueError, TypeError):
        # Not numeric, return as string
        return value_str


def _parse_accuracy_data(
    accuracy_csv: Path, category_key: str
) -> Tuple[pd.DataFrame, str]:
    """Parse training/validation accuracy CSV from W&B export.

    Args:
        accuracy_csv: Path to accuracy CSV file
        category_key: Category key to look for (may be auto-detected)

    Returns:
        Tuple of (processed DataFrame, actual category_key used)
        DataFrame has columns: epoch, category_value, train_accuracy, val_accuracy
    """
    df = pd.read_csv(accuracy_csv)

    # Auto-detect category key if needed
    actual_category_key = category_key
    if category_key not in df.columns:
        detected_key = _auto_detect_category_key(df)
        if detected_key:
            actual_category_key = detected_key
            logger.info(f"Using auto-detected category key: '{actual_category_key}'")
        else:
            logger.warning(
                f"Category key '{category_key}' not found and auto-detection failed"
            )
            return pd.DataFrame(), category_key

    # Find training and validation accuracy columns
    train_acc_cols = [
        col
        for col in df.columns
        if "train_accuracy" in col.lower()
        and "__MIN" not in col
        and "__MAX" not in col
    ]
    val_acc_cols = [
        col
        for col in df.columns
        if "val_accuracy" in col.lower() and "__MIN" not in col and "__MAX" not in col
    ]

    if not train_acc_cols and not val_acc_cols:
        logger.warning("No train_accuracy or val_accuracy columns found")
        return pd.DataFrame(), actual_category_key

    logger.info(
        f"Found {len(train_acc_cols)} train_accuracy columns, {len(val_acc_cols)} val_accuracy columns"
    )

    # Extract data for each category value
    processed_data = []

    # Get all category values from column names
    category_values = set()
    for col in train_acc_cols + val_acc_cols:
        cat_val = _extract_category_value_from_column(col, actual_category_key)
        if cat_val:
            category_values.add(cat_val)

    for cat_val in category_values:
        # Find columns for this category value
        train_col = next(
            (
                col
                for col in train_acc_cols
                if _extract_category_value_from_column(col, actual_category_key)
                == cat_val
            ),
            None,
        )
        val_col = next(
            (
                col
                for col in val_acc_cols
                if _extract_category_value_from_column(col, actual_category_key)
                == cat_val
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
                        "category_value": _standardize_category_value(cat_val),
                        "train_accuracy": train_acc,
                        "val_accuracy": val_acc,
                    }
                )

    result_df = pd.DataFrame(processed_data)
    logger.info(
        f"Parsed {len(result_df)} accuracy data points for {len(category_values)} category values"
    )

    return result_df, actual_category_key


def _parse_loss_data(loss_csv: Path, category_key: str) -> Tuple[pd.DataFrame, str]:
    """Parse loss CSV from W&B export.

    Args:
        loss_csv: Path to loss CSV file
        category_key: Category key to look for (may be auto-detected)

    Returns:
        Tuple of (processed DataFrame, actual category_key used)
        DataFrame has columns: epoch, category_value, cross_entropy_loss, energy_loss
    """
    df = pd.read_csv(loss_csv)

    # Auto-detect category key if needed
    actual_category_key = category_key
    if category_key not in df.columns:
        detected_key = _auto_detect_category_key(df)
        if detected_key:
            actual_category_key = detected_key
            logger.info(f"Using auto-detected category key: '{actual_category_key}'")
        else:
            logger.warning(
                f"Category key '{category_key}' not found and auto-detection failed"
            )
            return pd.DataFrame(), category_key

    # Find loss columns
    ce_loss_cols = [
        col
        for col in df.columns
        if "CrossEntropyLoss" in col and "__MIN" not in col and "__MAX" not in col
    ]
    energy_loss_cols = [
        col
        for col in df.columns
        if "EnergyLoss" in col and "__MIN" not in col and "__MAX" not in col
    ]

    if not ce_loss_cols and not energy_loss_cols:
        logger.warning("No CrossEntropyLoss or EnergyLoss columns found")
        return pd.DataFrame(), actual_category_key

    logger.info(
        f"Found {len(ce_loss_cols)} CrossEntropyLoss columns, {len(energy_loss_cols)} EnergyLoss columns"
    )

    # Extract data for each category value
    processed_data = []

    # Get all category values from column names
    category_values = set()
    for col in ce_loss_cols + energy_loss_cols:
        cat_val = _extract_category_value_from_column(col, actual_category_key)
        if cat_val:
            category_values.add(cat_val)

    for cat_val in category_values:
        # Find columns for this category value
        ce_col = next(
            (
                col
                for col in ce_loss_cols
                if _extract_category_value_from_column(col, actual_category_key)
                == cat_val
            ),
            None,
        )
        energy_col = next(
            (
                col
                for col in energy_loss_cols
                if _extract_category_value_from_column(col, actual_category_key)
                == cat_val
            ),
            None,
        )

        for idx, row in df.iterrows():
            ce_loss = row[ce_col] if ce_col and not pd.isna(row[ce_col]) else None
            energy_loss = (
                row[energy_col]
                if energy_col and not pd.isna(row[energy_col])
                else None
            )

            if ce_loss is not None or energy_loss is not None:
                processed_data.append(
                    {
                        "epoch": row["epoch"],
                        "category_value": _standardize_category_value(cat_val),
                        "cross_entropy_loss": ce_loss,
                        "energy_loss": energy_loss,
                    }
                )

    result_df = pd.DataFrame(processed_data)
    logger.info(
        f"Parsed {len(result_df)} loss data points for {len(category_values)} category values"
    )

    return result_df, actual_category_key


def _plot_training_accuracy_panel(
    fig: plt.Figure,
    column_left: float,
    column_width: float,
    bottom: float,
    height: float,
    accuracy_df: pd.DataFrame,
    category_key: str,
    hue_values: List[str],
    colors: Dict[str, str],
    show_ylabel: bool = True,
    show_legend: bool = True,
    xlim: Optional[Tuple[float, float]] = None,
    **kwargs,
) -> None:
    """Plot training and validation accuracy panel.

    Args:
        fig: Matplotlib figure
        column_left: Left position of panel
        column_width: Width of panel
        bottom: Bottom position of panel
        height: Height of panel
        accuracy_df: DataFrame with train/val accuracy data
        category_key: Column name for category dimension
        hue_values: Ordered list of category values
        colors: Color mapping for category values
        show_ylabel: Whether to show y-axis label
        show_legend: Whether to show legend
        **kwargs: Override FORMATTING defaults
    """
    fmt = {**TRAINING_FORMATTING, **kwargs}

    ax = fig.add_axes([column_left, bottom, column_width, height])
    ax.patch.set_alpha(0)

    if accuracy_df.empty:
        ax.text(
            0.5,
            0.5,
            "No training accuracy data",
            ha="center",
            va="center",
            fontsize=14,
            alpha=0.6,
        )
        return

    # Note: accuracy_df["category_value"] is already standardized to numeric/string
    # Standardize hue_values for comparison
    hue_values_std = [_standardize_category_value(v) for v in hue_values]

    # Plot training accuracy (solid lines)
    for cat_val_orig, cat_val_std in zip(hue_values, hue_values_std):
        cat_data = accuracy_df[accuracy_df["category_value"] == cat_val_std]
        if not cat_data.empty:
            train_data = cat_data.dropna(subset=["train_accuracy"])
            if not train_data.empty:
                ax.plot(
                    train_data["epoch"],
                    train_data["train_accuracy"],
                    color=colors.get(cat_val_orig, fmt["greyscale_color"]),
                    linewidth=fmt["linewidth_main"],
                    linestyle="-",
                    alpha=fmt["alpha_line"],
                )

    # Plot validation accuracy (dotted lines)
    for cat_val_orig, cat_val_std in zip(hue_values, hue_values_std):
        cat_data = accuracy_df[accuracy_df["category_value"] == cat_val_std]
        if not cat_data.empty:
            val_data = cat_data.dropna(subset=["val_accuracy"])
            if not val_data.empty:
                ax.plot(
                    val_data["epoch"],
                    val_data["val_accuracy"],
                    color=colors.get(cat_val_orig, fmt["greyscale_color"]),
                    linewidth=fmt["linewidth_main"],
                    linestyle=":",
                    alpha=fmt["alpha_line"],
                )

    # Styling
    ax.set_ylim(-0.01, 1.01)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    # Set x-axis limits
    if xlim is not None:
        ax.set_xlim(xlim[0], xlim[1])
    else:
        ax.margins(x=0)

    # Log axis limits
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    logger.info(
        f"Training accuracy panel: x-axis [{xmin}, {xmax}], y-axis [{ymin}, {ymax}]"
    )

    if show_ylabel:
        ax.set_ylabel(
            "Train Accuracy", fontsize=fmt["fontsize_axis"], fontweight="bold"
        )
    else:
        ax.set_ylabel("")
        ax.set_yticklabels([])

    ax.set_xlabel("Epoch", fontsize=fmt["fontsize_axis"])
    ax.tick_params(labelsize=fmt["fontsize_tick"])

    # Add legend for line styles
    if show_legend:
        from matplotlib.lines import Line2D

        legend_handles = [
            Line2D(
                [0],
                [0],
                color="black",
                linewidth=fmt["linewidth_main"],
                linestyle="-",
                label="Training",
                alpha=fmt["alpha_line"],
            ),
            Line2D(
                [0],
                [0],
                color="black",
                linewidth=fmt["linewidth_main"],
                linestyle=":",
                label="Validation",
                alpha=fmt["alpha_line"],
            ),
        ]
        ax.legend(
            handles=legend_handles,
            loc="lower right",
            frameon=False,
            fontsize=fmt["fontsize_legend"] - 2,
        )

    ax.grid(True, alpha=0.3)
    sns.despine(ax=ax, left=True, bottom=True)


def _plot_loss_panel(
    fig: plt.Figure,
    column_left: float,
    column_width: float,
    bottom: float,
    height: float,
    loss_df: pd.DataFrame,
    category_key: str,
    hue_values: List[str],
    colors: Dict[str, str],
    show_ylabel: bool = True,
    show_legend: bool = True,
    xlim: Optional[Tuple[float, float]] = None,
    energy_loss_scale: float = 10.0,
    **kwargs,
) -> None:
    """Plot cross-entropy and energy loss panel.

    Args:
        fig: Matplotlib figure
        column_left: Left position of panel
        column_width: Width of panel
        bottom: Bottom position of panel
        height: Height of panel
        loss_df: DataFrame with loss data
        category_key: Column name for category dimension
        hue_values: Ordered list of category values
        colors: Color mapping for category values
        show_ylabel: Whether to show y-axis label
        show_legend: Whether to show legend
        xlim: X-axis limits (min, max)
        energy_loss_scale: Scale factor for energy loss (default: 10.0)
        **kwargs: Override FORMATTING defaults
    """
    fmt = {**TRAINING_FORMATTING, **kwargs}

    ax = fig.add_axes([column_left, bottom, column_width, height])
    ax.patch.set_alpha(0)

    if loss_df.empty:
        ax.text(
            0.5, 0.5, "No loss data", ha="center", va="center", fontsize=14, alpha=0.6
        )
        return

    # Note: loss_df["category_value"] is already standardized to numeric/string
    # Standardize hue_values for comparison
    hue_values_std = [_standardize_category_value(v) for v in hue_values]

    # Plot cross-entropy loss (solid lines)
    for cat_val_orig, cat_val_std in zip(hue_values, hue_values_std):
        cat_data = loss_df[loss_df["category_value"] == cat_val_std]
        if not cat_data.empty:
            ce_data = cat_data.dropna(subset=["cross_entropy_loss"])
            if not ce_data.empty:
                ax.plot(
                    ce_data["epoch"],
                    ce_data["cross_entropy_loss"],
                    color=colors.get(cat_val_orig, fmt["greyscale_color"]),
                    linewidth=fmt["linewidth_main"],
                    linestyle="-",
                    alpha=fmt["alpha_line"],
                )

    # Plot energy loss (dashed lines) - scaled and smoothed
    # Energy loss is 0 during validation steps, causing periodic drops to 0
    # We smooth the data to remove these outliers while preserving truly zero series
    for cat_val_orig, cat_val_std in zip(hue_values, hue_values_std):
        cat_data = loss_df[loss_df["category_value"] == cat_val_std]
        if not cat_data.empty:
            energy_data = cat_data.dropna(subset=["energy_loss"]).copy()
            if not energy_data.empty:
                energy_values = energy_data["energy_loss"].values
                epochs = energy_data["epoch"].values

                # Check if energy loss is all zeros (disabled energy loss)
                if np.all(energy_values == 0):
                    # Plot as-is for truly disabled energy loss
                    ax.plot(
                        epochs,
                        energy_values * energy_loss_scale,
                        color=colors.get(cat_val_orig, fmt["greyscale_color"]),
                        linewidth=fmt["linewidth_main"],
                        linestyle=":",
                        alpha=fmt["alpha_line"],
                    )
                else:
                    # Apply smoothing to remove validation step outliers
                    # Use median filter to remove isolated zeros while preserving trends
                    from scipy.ndimage import median_filter

                    # Apply median filter with window size 3 to remove isolated outliers
                    # This preserves the general trend better than simple interpolation
                    smoothed_values = median_filter(
                        energy_values, size=3, mode="nearest"
                    )

                    # For any remaining zeros (edges or consecutive zeros), interpolate
                    smoothed_values_clean = smoothed_values.copy().astype(float)
                    smoothed_values_clean[smoothed_values_clean == 0] = np.nan

                    import pandas as pd

                    series = pd.Series(smoothed_values_clean, index=epochs)
                    series_interpolated = series.interpolate(
                        method="linear", limit_direction="both"
                    )

                    # Fill any remaining NaNs (e.g., all zeros at start/end)
                    series_interpolated = series_interpolated.fillna(0)

                    ax.plot(
                        epochs,
                        series_interpolated.values * energy_loss_scale,
                        color=colors.get(cat_val_orig, fmt["greyscale_color"]),
                        linewidth=fmt["linewidth_main"],
                        linestyle=":",
                        alpha=fmt["alpha_line"],
                    )

    # Set x-axis limits
    if xlim is not None:
        ax.set_xlim(xlim[0], xlim[1])
    else:
        ax.margins(x=0)

    # Log axis limits
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    logger.info(f"Loss panel: x-axis [{xmin}, {xmax}], y-axis [{ymin}, {ymax}]")

    # Styling
    if show_ylabel:
        ax.set_ylabel("Loss", fontsize=fmt["fontsize_axis"], fontweight="bold")
    else:
        ax.set_ylabel("")
        ax.set_yticklabels([])

    ax.set_xlabel("Epoch", fontsize=fmt["fontsize_axis"])
    ax.tick_params(labelsize=fmt["fontsize_tick"])

    # Add legend for line styles
    if show_legend:
        from matplotlib.lines import Line2D

        # Format energy loss scale for legend
        if energy_loss_scale == 1.0:
            energy_label = "Energy"
        else:
            # Format scale: show as integer if whole number, otherwise with decimals
            scale_str = (
                f"{int(energy_loss_scale)}"
                if energy_loss_scale == int(energy_loss_scale)
                else f"{energy_loss_scale:.1f}"
            )
            energy_label = f"Energy × {scale_str}"

        legend_handles = [
            Line2D(
                [0],
                [0],
                color="black",
                linewidth=fmt["linewidth_main"],
                linestyle="-",
                label="Cross Entropy",
                alpha=fmt["alpha_line"],
            ),
            Line2D(
                [0],
                [0],
                color="black",
                linewidth=fmt["linewidth_main"],
                linestyle=":",
                label=energy_label,
                alpha=fmt["alpha_line"],
            ),
        ]
        ax.legend(
            handles=legend_handles,
            loc="upper right",
            frameon=False,
            fontsize=fmt["fontsize_legend"] - 2,
        )

    ax.grid(True, alpha=0.3)
    sns.despine(ax=ax, left=True, bottom=True)


def _add_panel_letters(
    fig: plt.Figure,
    left_col_left: float,
    right_col_left: float,
    accuracy_top: float,
    loss_top: float,
    performance_top: float,
    ridge_top: float,
    layout: Dict,
    fmt: Dict,
) -> None:
    """Add panel letters A-D to identify plot sections.

    Args:
        fig: Matplotlib figure
        left_col_left: Left position of left column
        right_col_left: Left position of right column
        accuracy_top: Top position of accuracy panel
        loss_top: Top position of loss panel
        performance_top: Top position of performance panel
        ridge_top: Top position of ridge plots
        layout: Layout configuration dict
        fmt: Formatting configuration dict
    """
    panel_letters = [
        ("A)", left_col_left, accuracy_top),  # Training/validation accuracy
        ("B)", left_col_left, loss_top),  # Loss panel
        ("C)", right_col_left, performance_top),  # Performance panel
        ("D)", right_col_left, ridge_top),  # Ridge plots
    ]

    for letter, x_pos, y_pos in panel_letters:
        fig.text(
            x_pos + layout["panel_letter_offset_x"],
            y_pos + layout["panel_letter_offset_y"],
            letter,
            fontsize=fmt["fontsize_panel_label"],
            fontweight="bold",
            ha="center",
            va="top",
        )


def _plot_category_legend(
    fig: plt.Figure,
    column_left: float,
    column_width: float,
    bottom: float,
    height: float,
    category_key: str,
    hue_values: List[str],
    colors: Dict[str, str],
    config: Dict,
    dt: float = 2.0,
    **kwargs,
) -> None:
    """Add horizontal category legend between left panels.

    Args:
        fig: Matplotlib figure
        column_left: Left position of legend
        column_width: Width of legend
        bottom: Bottom position of legend
        height: Height of legend
        category_key: Column name for category dimension
        hue_values: Ordered list of category values
        colors: Color mapping for category values
        config: Configuration dict with naming mappings
        dt: Temporal resolution in ms per timestep
        **kwargs: Override FORMATTING defaults
    """
    fmt = {**TRAINING_FORMATTING, **kwargs}

    legend_ax = fig.add_axes([column_left, bottom, column_width, height])
    legend_ax.set_xlim(0, 1)
    legend_ax.set_ylim(0, 1)
    legend_ax.axis("off")
    legend_ax.patch.set_alpha(0)

    # Create legend elements
    legend_elements = []
    legend_labels = []

    for val in hue_values:
        if val in colors:
            legend_elements.append(
                plt.Line2D(
                    [0],
                    [0],
                    color=colors[val],
                    linewidth=fmt["linewidth_main"],
                    alpha=fmt["alpha_line"],
                )
            )
            legend_labels.append(_format_legend_label(category_key, val, config, dt))

    if legend_elements:
        # Get symbol for title
        symbol = get_display_name(category_key, config)

        n_cols = min(len(legend_elements), 7)
        logger.debug(
            f"Adding category legend with {len(legend_elements)} elements in {n_cols} columns"
        )

        legend = legend_ax.legend(
            legend_elements,
            legend_labels,
            loc="center",
            ncol=n_cols,
            frameon=False,
            fontsize=fmt["fontsize_legend"],
            handlelength=2,
            handletextpad=0.5,
            columnspacing=1.0,
            title=symbol,
            title_fontsize=fmt["fontsize_legend"],
        )
        # Make legend title bold
        legend.get_title().set_fontweight("bold")


def plot_training_overview(
    test_data: Union[Path, List[Path]],
    accuracy_csv: Path,
    loss_csv: Path,
    output: Path,
    subplot_var: str = "layers",
    hue_var: str = "category",
    column_var: Optional[str] = None,
    category_key: Optional[str] = None,
    parameter_key: Optional[str] = None,
    confidence_measure: Optional[Union[str, List[str]]] = "first_label_confidence",
    accuracy_measure: Optional[Union[str, List[str]]] = "accuracy",
    dt: float = 2.0,
    energy_loss_scale: float = 10.0,
    config: Optional[Dict] = None,
    **kwargs,
) -> plt.Figure:
    """Create comprehensive training overview plot.

    Args:
        test_data: Path(s) to test_data.csv with layer responses and performance
        accuracy_csv: Path to training/validation accuracy CSV
        loss_csv: Path to loss CSV
        output: Path to save figure
        subplot_var: Variable for ridge plot subplots (default: "layers")
        hue_var: Variable for color coding (default: "category")
        column_var: Variable for columns (optional)
        category_key: Column name for category dimension
        parameter_key: Column name for parameter dimension
        confidence_measure: Column name(s) for confidence metrics
        accuracy_measure: Column name(s) for accuracy metrics
        dt: Temporal resolution in ms per timestep
        energy_loss_scale: Scale factor for energy loss display (default: 10.0)
        config: Configuration dict with palette, naming, ordering
        **kwargs: Override LAYOUT and FORMATTING defaults

    Returns:
        Matplotlib figure
    """
    logger.info("=" * 60)
    logger.info("Starting training overview plot")
    logger.info("=" * 60)

    # Normalize dimension configuration
    subplot_var, subplot_limit = _normalize_dimension(subplot_var)
    hue_var, hue_limit = _normalize_dimension(hue_var)
    column_var, column_limit = _normalize_dimension(column_var)

    if subplot_var is None or hue_var is None:
        raise ValueError("subplot and hue dimensions cannot be empty")

    _validate_dimensions(
        subplot_var=subplot_var, hue_var=hue_var, column_var=column_var
    )

    if config is None:
        config = {"palette": {}, "naming": {}, "ordering": {}}

    # Merge layout and formatting defaults with overrides
    layout = {
        **TRAINING_LAYOUT,
        **{k: v for k, v in kwargs.items() if k in TRAINING_LAYOUT},
    }
    fmt = {
        **TRAINING_FORMATTING,
        **{k: v for k, v in kwargs.items() if k in TRAINING_FORMATTING},
    }

    # Load test data
    if isinstance(test_data, list):
        logger.info(f"Loading and concatenating test data from {len(test_data)} files")
        dfs = [pd.read_csv(path) for path in test_data]
        test_df = pd.concat(dfs, ignore_index=True)
    else:
        logger.info(f"Loading test data from: {test_data}")
        test_df = pd.read_csv(test_data)

    logger.info(f"Test data: {len(test_df)} rows, {len(test_df.columns)} columns")

    # Parse training data
    logger.info(f"Parsing accuracy data from: {accuracy_csv}")
    accuracy_df, detected_category_key = _parse_accuracy_data(
        accuracy_csv, category_key
    )

    logger.info(f"Parsing loss data from: {loss_csv}")
    loss_df, _ = _parse_loss_data(loss_csv, detected_category_key)

    # Use original category_key for test data lookups (may be alias like "energyloss")
    # Use detected_category_key for W&B CSV parsing (full name like "energy_loss_weight")
    test_data_category_key = category_key
    wandb_category_key = detected_category_key

    # Extract dimension values from test data
    logger.info("Extracting dimension values from test data...")
    logger.info(f"Using category key '{test_data_category_key}' for test data lookups")

    hue_key = _get_dimension_key(
        dimension=hue_var,
        category_key=test_data_category_key,
        parameter_key=parameter_key,
    )
    subplot_key = _get_dimension_key(
        dimension=subplot_var,
        category_key=test_data_category_key,
        parameter_key=parameter_key,
    )
    column_key = (
        _get_dimension_key(
            dimension=column_var,
            category_key=test_data_category_key,
            parameter_key=parameter_key,
        )
        if column_var
        else None
    )

    hue_values = _extract_dimension_values(
        df=test_df,
        dimension=hue_var,
        dimension_key=hue_key,
        config=config,
        dimension_limit=hue_limit,
    )
    subplot_values = _extract_dimension_values(
        df=test_df,
        dimension=subplot_var,
        dimension_key=subplot_key,
        config=config,
        dimension_limit=subplot_limit,
    )

    if not hue_values or not subplot_values:
        raise ValueError("Cannot create plot with empty dimension values")

    # Get colors for hue dimension
    colors = _get_colors_for_dimension(
        values=hue_values, dimension_key=hue_key, config=config
    )

    # Standardize and filter W&B data to only include category values present in test data
    # The W&B CSVs may have more category values than the test data
    # (e.g., test data has [0.0, 0.1, 0.5] but W&B has [0, 0.01, 0.05, 0.1, 0.5, ...])
    logger.info(f"Test data category values: {hue_values}")

    # Standardize hue_values to numeric if possible for consistent comparison
    hue_values_standardized = []
    for v in hue_values:
        try:
            hue_values_standardized.append(float(v))
        except (ValueError, TypeError):
            hue_values_standardized.append(v)

    test_values_set = set(hue_values_standardized)
    logger.info(
        f"Test data standardized values: {sorted(test_values_set, key=lambda x: (isinstance(x, str), x))}"
    )

    # Filter accuracy data to only include test data category values
    if not accuracy_df.empty:
        wandb_acc_values_orig = sorted(accuracy_df["category_value"].unique())
        logger.info(f"W&B accuracy category values (all): {wandb_acc_values_orig}")

        # Keep only rows with category values present in test data
        accuracy_df = accuracy_df[
            accuracy_df["category_value"].isin(test_values_set)
        ].copy()

        remaining_acc_values = sorted(accuracy_df["category_value"].unique())
        logger.info(f"W&B accuracy category values (filtered): {remaining_acc_values}")
        logger.info(f"Filtered accuracy data: {len(accuracy_df)} rows")

    # Filter loss data to only include test data category values
    if not loss_df.empty:
        wandb_loss_values_orig = sorted(loss_df["category_value"].unique())
        logger.info(f"W&B loss category values (all): {wandb_loss_values_orig}")

        # Keep only rows with category values present in test data
        loss_df = loss_df[loss_df["category_value"].isin(test_values_set)].copy()

        remaining_loss_values = sorted(loss_df["category_value"].unique())
        logger.info(f"W&B loss category values (filtered): {remaining_loss_values}")
        logger.info(f"Filtered loss data: {len(loss_df)} rows")

    # Create figure
    fig = plt.figure(figsize=(layout["figure_width"], layout["figure_height"]))
    sns.set_context("talk")

    # Calculate column positions
    left_col_left = layout["left_margin"]
    left_col_width = layout["left_column_width"]
    right_col_left = left_col_left + left_col_width + layout["column_spacing"]
    right_col_width = layout["right_column_width"]

    # Calculate left column x-axis limits (epoch range)
    # Start at 0, round max up to next decimal (e.g., 349 -> 350)
    import math

    if not accuracy_df.empty or not loss_df.empty:
        epoch_max_acc = accuracy_df["epoch"].max() if not accuracy_df.empty else 0
        epoch_max_loss = loss_df["epoch"].max() if not loss_df.empty else 0
        epoch_max = max(epoch_max_acc, epoch_max_loss)
        # Round up to next multiple of 10
        left_xmax = math.ceil(epoch_max / 10) * 10
        left_xmin = 0
        logger.info(
            f"Left column x-axis limits: [0, {left_xmax}] (data max: {epoch_max})"
        )
    else:
        left_xmin, left_xmax = 0, 100

    # Calculate left column panel positions (from top to bottom)
    current_top = 1 - layout["top_margin"]

    # Training accuracy panel
    accuracy_bottom = current_top - layout["accuracy_height"]
    logger.info("Creating training accuracy panel...")
    _plot_training_accuracy_panel(
        fig=fig,
        column_left=left_col_left,
        column_width=left_col_width,
        bottom=accuracy_bottom,
        height=layout["accuracy_height"],
        accuracy_df=accuracy_df,
        category_key="category_value",
        hue_values=hue_values,
        colors=colors,
        show_ylabel=True,
        show_legend=True,
        xlim=(left_xmin, left_xmax),
        **fmt,
    )

    # Category legend (moved down slightly)
    current_top = (
        accuracy_bottom
        - layout["accuracy_bottom_margin"]
        - layout["legend_top_margin"]
    )
    legend_bottom = current_top - layout["legend_height"]
    logger.info("Adding category legend...")
    _plot_category_legend(
        fig=fig,
        column_left=left_col_left,
        column_width=left_col_width,
        bottom=legend_bottom,
        height=layout["legend_height"],
        category_key=hue_key,
        hue_values=hue_values,
        colors=colors,
        config=config,
        dt=dt,
        **fmt,
    )

    # Loss panel
    current_top = legend_bottom - layout["legend_bottom_margin"]
    loss_bottom = current_top - layout["loss_height"]
    logger.info("Creating loss panel...")
    _plot_loss_panel(
        fig=fig,
        column_left=left_col_left,
        column_width=left_col_width,
        bottom=loss_bottom,
        height=layout["loss_height"],
        loss_df=loss_df,
        category_key="category_value",
        hue_values=hue_values,
        colors=colors,
        show_ylabel=True,
        show_legend=True,
        xlim=(left_xmin, left_xmax),
        energy_loss_scale=energy_loss_scale,
        **fmt,
    )

    # Right column: performance panel + ridge plots
    # Calculate right column x-axis limits (time in ms)
    # Start at 0, round max up to next multiple of 10
    if "times_index" in test_df.columns:
        time_max = test_df["times_index"].max() * dt
        # Round up to next multiple of 10
        right_xmax = math.ceil(time_max / 10) * 10
        right_xmin = 0
        logger.info(
            f"Right column x-axis limits: [0, {right_xmax}] (data max: {time_max} ms)"
        )
    else:
        right_xmin, right_xmax = 0, 120

    # Calculate positions
    right_top = 1 - layout["top_margin"]
    performance_bottom = right_top - layout["performance_height"]

    logger.info("Creating test performance panel...")
    perf_ax = fig.add_axes(
        [
            right_col_left,
            performance_bottom,
            right_col_width,
            layout["performance_height"],
        ]
    )

    _plot_accuracy_panel(
        ax=perf_ax,
        data=test_df,
        hue_var=hue_var,
        hue_key=hue_key,
        hue_values=hue_values,
        colors=colors,
        dt=dt,
        show_ylabel=True,
        show_legend=True,
        accuracy_cols=(
            [accuracy_measure]
            if isinstance(accuracy_measure, str)
            else accuracy_measure
        ),
        confidence_cols=(
            [confidence_measure]
            if isinstance(confidence_measure, str)
            else confidence_measure
        ),
        **fmt,
    )

    # Set x-axis limits
    perf_ax.set_xlim(right_xmin, right_xmax)

    # Log axis limits
    xmin, xmax = perf_ax.get_xlim()
    ymin, ymax = perf_ax.get_ylim()
    logger.info(f"Performance panel: x-axis [{xmin}, {xmax}], y-axis [{ymin}, {ymax}]")

    # Ridge plots - align bottom with left column loss panel
    # Calculate required ridge_height to align bottoms properly
    ridge_top = performance_bottom - layout["performance_pad"]

    # The ridge plot function calculates:
    # spacing = ridge_height / n_subplots * (1 - ridge_overlap)
    # plot_height = ridge_height / n_subplots * 1.4
    # bottom_position = ridge_top - (n_subplots - 1) * spacing - plot_height
    #
    # We want: bottom_position = loss_bottom
    # Solving for ridge_height:
    # loss_bottom = ridge_top - (n_subplots - 1) * (ridge_height / n_subplots * (1 - ridge_overlap)) - (ridge_height / n_subplots * 1.4)
    # loss_bottom = ridge_top - ridge_height * [(n_subplots - 1) * (1 - ridge_overlap) / n_subplots + 1.4 / n_subplots]
    # ridge_height = (ridge_top - loss_bottom) / [(n_subplots - 1) * (1 - ridge_overlap) / n_subplots + 1.4 / n_subplots]

    n_subplots = len(subplot_values)
    if n_subplots > 0:
        denominator = (n_subplots - 1) * (
            1 - layout["ridge_overlap"]
        ) / n_subplots + 1.4 / n_subplots
        ridge_height = (ridge_top - loss_bottom) / denominator
    else:
        ridge_height = ridge_top - loss_bottom

    logger.info("Creating ridge plots...")
    logger.info(f"Subplot values for ridges: {subplot_values}")
    logger.info(
        f"Ridge plot: top={ridge_top}, bottom target={loss_bottom}, height={ridge_height:.4f}"
    )
    logger.info(f"Ridge overlap: {layout['ridge_overlap']}, n_subplots: {n_subplots}")

    # Calculate actual bottom position for verification
    if n_subplots > 0:
        spacing = ridge_height / n_subplots * (1 - layout["ridge_overlap"])
        plot_height = ridge_height / n_subplots * 1.4
        calculated_bottom = ridge_top - (n_subplots - 1) * spacing - plot_height
        logger.info(
            f"Calculated ridge bottom: {calculated_bottom:.6f}, target: {loss_bottom:.6f}, diff: {calculated_bottom - loss_bottom:.6f}"
        )

    ridge_axes = _plot_response_ridges(
        fig=fig,
        column_left=right_col_left,
        column_width=right_col_width,
        data=test_df,
        subplot_var=subplot_var,
        subplot_key=subplot_key,
        subplot_values=subplot_values,
        hue_var=hue_var,
        hue_key=hue_key,
        hue_values=hue_values,
        colors=colors,
        dt=dt,
        show_ylabel=True,
        config=config,
        ridge_top=ridge_top,
        ridge_height=ridge_height,
        ridge_overlap=layout["ridge_overlap"],
        **fmt,
    )

    # Note: No separate ridge legend - we already have the category legend on the left

    # Set x-axis limits to match performance panel
    for ax in ridge_axes:
        ax.set_xlim(right_xmin, right_xmax)

    # Log ridge plot axis limits and verify bottom alignment
    if ridge_axes:
        # Check top and bottom ridge plots
        top_ax = ridge_axes[0]
        bottom_ax = ridge_axes[-1]

        xmin, xmax = top_ax.get_xlim()
        ymin, ymax = top_ax.get_ylim()
        logger.info(
            f"Ridge plots: global x-axis [{xmin}, {xmax}], y-axis [{ymin}, {ymax}]"
        )

        # Get actual position of bottommost ridge plot
        bottom_pos = bottom_ax.get_position()
        logger.info(
            f"Ridge plot actual bottom: {bottom_pos.y0:.4f}, target: {loss_bottom:.4f}, diff: {(bottom_pos.y0 - loss_bottom):.4f}"
        )

    # Align y-labels
    all_axes = ridge_axes
    fig.align_ylabels(all_axes)

    # Add panel letters
    logger.info("Adding panel letters...")
    _add_panel_letters(
        fig=fig,
        left_col_left=left_col_left,
        right_col_left=right_col_left,
        accuracy_top=1 - layout["top_margin"],
        loss_top=legend_bottom - layout["legend_bottom_margin"],
        performance_top=1 - layout["top_margin"],
        ridge_top=ridge_top,
        layout=layout,
        fmt=fmt,
    )

    logger.info(f"Saving figure to: {output}")
    save_plot(output)
    logger.info("=" * 60)
    logger.info("Training overview plot complete")
    logger.info("=" * 60)

    return fig


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Plot training overview with accuracy, loss, and test performance"
    )
    parser.add_argument(
        "--test_data",
        type=Path,
        nargs="+",
        required=True,
        help="Path(s) to test_data.csv file(s). Multiple paths will be concatenated.",
    )
    parser.add_argument(
        "--accuracy_csv",
        type=Path,
        required=True,
        help="Path to training/validation accuracy CSV",
    )
    parser.add_argument(
        "--loss_csv",
        type=Path,
        required=True,
        help="Path to loss CSV with CrossEntropyLoss and EnergyLoss",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output figure path",
    )
    parser.add_argument(
        "--subplot",
        type=str,
        default="layers",
        help="Variable for ridge plot subplots (default: layers)",
    )
    parser.add_argument(
        "--hue",
        type=str,
        default="category",
        help="Variable for color coding (default: category)",
    )
    parser.add_argument(
        "--column",
        type=str,
        default=None,
        help="Variable for columns (optional)",
    )
    parser.add_argument(
        "--category-key",
        type=str,
        required=True,
        help="Category column name (will be auto-detected if not found)",
    )
    parser.add_argument(
        "--parameter-key",
        type=str,
        help="Parameter column name",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=2.0,
        help="Temporal resolution in ms per timestep",
    )
    parser.add_argument(
        "--energy-loss-scale",
        type=float,
        default=10.0,
        help="Scale factor for energy loss display (default: 10.0)",
    )
    parser.add_argument(
        "--confidence-measure",
        type=str,
        default="first_label_confidence",
        help="Confidence measure column name",
    )
    parser.add_argument(
        "--accuracy-measure",
        type=str,
        default="accuracy",
        help="Accuracy measure column name",
    )
    parser.add_argument(
        "--palette",
        type=str,
        help="JSON color palette",
    )
    parser.add_argument(
        "--naming",
        type=str,
        help="JSON naming dict",
    )
    parser.add_argument(
        "--ordering",
        type=str,
        help="JSON ordering dict",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    args, unknown = parser.parse_known_args()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s: %(message)s",
    )

    if unknown:
        logger.info(f"Ignoring unknown arguments: {unknown}")

    # Load config
    config = load_config_from_args(
        palette_str=args.palette,
        naming_str=args.naming,
        ordering_str=args.ordering,
    )

    # Handle single vs multiple test data files
    test_data_input = args.test_data if len(args.test_data) > 1 else args.test_data[0]

    # Plot
    plot_training_overview(
        test_data=test_data_input,
        accuracy_csv=args.accuracy_csv,
        loss_csv=args.loss_csv,
        output=args.output,
        subplot_var=args.subplot,
        hue_var=args.hue,
        column_var=args.column,
        category_key=args.category_key,
        parameter_key=args.parameter_key,
        confidence_measure=args.confidence_measure,
        accuracy_measure=args.accuracy_measure,
        dt=args.dt,
        energy_loss_scale=args.energy_loss_scale,
        config=config,
    )


if __name__ == "__main__":
    main()
