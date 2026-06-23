"""plot_all_dynamics_manuscript.py
Comprehensive dynamics manuscript figure with all three experiments.

This script creates a composite figure showing:
- Panel A: Performance traces (accuracy + confidence) for a middle parameter value
- Panel B: Layer response ridge plots for focus layer across parameter values
- Panels C, D, E: Summary statistics for all three experiments (interval, duration, contrast)
  arranged in columns (i, ii, iii)

Example:
    ```bash
    python plot_all_dynamics_manuscript.py \
        --data-interval interval_responses.csv \
        --data-duration duration_responses.csv \
        --data-contrast contrast_responses.csv \
        --groen-data-interval /path/to/groen2022_interval_data.csv \
        --groen-data-duration /path/to/groen2022_duration_data.csv \
        --groen-data-contrast /path/to/groen2022_contrast_data.csv \
        --output dynamics_manuscript.png \
        --focus-layer V1 --dt 2.0
    ```
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

from dynvision.utils.visualization_utils import (
    save_plot,
    load_config_from_args,
    get_display_name,
    DT_CONVERT_PARAMS,
    calculate_label_indicator,
)

# Import functions from plot_dynamics
from dynvision.visualization.plot_dynamics import (
    determine_experiment_info,
    calculate_metrics,
    create_panel_b,
    # DYNAMICS_LAYOUT,
    DYNAMICS_STYLE,
    _extract_dimension_values,
    _get_colors_for_dimension,
    _add_horizontal_legend,
    RESPONSES_FORMATTING,
)

from dynvision.visualization.plot_responses import _format_classifier_label

# Import Groen plotting functions from plot_dynamics_with_groen
from dynvision.visualization.plot_dynamics_with_groen import (
    load_groen_data,
    plot_groen_temporal_summation,
    plot_groen_adaptation_recovery_simple,
    plot_groen_contrast_time_to_peak,
    DYNAMICS_LAYOUT_EXTENDED,
)

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# =============================================================================
# Layout Configuration for All-Experiments Manuscript Figure
# =============================================================================
DYNAMICS_LAYOUT = {
    "figure_width": 15,
    "figure_height": 13,
    "plots_start": 0.05,
    "left_plots_width": 0.6,
    "right_plots_width": 0.3,
    "gap_width": 0.1,
    # Vertical positions
    "upper_top": 0.98,
    "upper_bottom": 0.75,
    "lower_top": 0.72,
    "lower_bottom": 0.01,
    "legend_height": 0.08,
    "legend_margin": 0.045,
    # Positioning helpers
    "y_label_x_position": 0.00,
    "panel_label_x_offset": -0.04,
    "panel_label_y_offset": 0.04,
}
# Number of experiment columns (interval, duration, contrast)
_N_COLUMNS = 3

# Gap between experiment columns (in figure coordinates)
_COLUMN_GAP = 0.08

# Calculate column widths for right side
_right_total_width = DYNAMICS_LAYOUT["right_plots_width"] * _N_COLUMNS
_available_column_width = (
    _right_total_width - (_N_COLUMNS - 1) * _COLUMN_GAP
) / _N_COLUMNS

# Vertical layout from extended dynamics (3 rows: C, D, E)
_panel_gap = 0.10
_n_typical_params = 7
_ridge_overlap = 0.2
_ridge_height = DYNAMICS_LAYOUT["lower_top"] - DYNAMICS_LAYOUT["lower_bottom"]
_ridge_spacing = _ridge_height / _n_typical_params * (1 - _ridge_overlap)
_ridge_plot_height = _ridge_height / _n_typical_params * 1.4
_panel_b_actual_bottom = (
    DYNAMICS_LAYOUT["lower_top"]
    - (_n_typical_params - 1) * _ridge_spacing
    - _ridge_plot_height
)

_right_bottom = _panel_b_actual_bottom
_right_top = DYNAMICS_LAYOUT["upper_top"]
_total_height = _right_top - _right_bottom
_available_height = _total_height - 2 * _panel_gap
_panel_height = _available_height / 3

ALL_DYNAMICS_LAYOUT = {
    **DYNAMICS_LAYOUT,
    # Column configuration
    "n_columns": _N_COLUMNS,
    "column_gap": _COLUMN_GAP,
    "column_width": _available_column_width,
    # Vertical positions for right-side panels (C, D, E rows)
    "right_lower_bottom": _right_bottom,
    "right_lower_top": _right_bottom + _panel_height,
    "right_middle_bottom": _right_bottom + _panel_height + _panel_gap,
    "right_middle_top": _right_bottom + 2 * _panel_height + _panel_gap,
    "right_upper_bottom": _right_bottom + 2 * _panel_height + 2 * _panel_gap,
    "right_upper_top": _right_top,
}

# Experiment order for columns
EXPERIMENT_ORDER = ["interval", "duration", "contrast"]
EXPERIMENT_COLUMN_LABELS = ["i", "ii", "iii"]

FORMATTING = {
    **RESPONSES_FORMATTING,
    **DYNAMICS_STYLE,
    "fontsize_panel_labels": 20,  # Match plot_training.py panel label size
}
# =============================================================================
# Panel Label Functions
# =============================================================================


def add_panel_labels_all_experiments(fig):
    """Add panel labels for the all-experiments manuscript figure.

    Labels:
    - A), B) on left side
    - C) i, D) i, E) i for first column
    - ii, ii, ii for second column (no letter prefix)
    - iii, iii, iii for third column (no letter prefix)
    """
    layout = ALL_DYNAMICS_LAYOUT
    panel_x_offset = layout["panel_label_x_offset"]
    panel_y_offset = layout["panel_label_y_offset"]
    plots_start = layout["plots_start"]
    left_width = layout["left_plots_width"]
    gap_width = layout["gap_width"]
    column_width = layout["column_width"]
    column_gap = layout["column_gap"]

    # Left side panel labels (A and B)
    left_x = plots_start + panel_x_offset
    fig.text(
        x=left_x,
        y=layout["upper_top"] + panel_y_offset,
        s="A)",
        fontsize=FORMATTING["fontsize_panel_labels"],
        fontweight="bold",
        ha="left",
        va="top",
    )
    fig.text(
        x=left_x,
        y=layout["lower_top"] - 3 * panel_y_offset,
        s="B)",
        fontsize=FORMATTING["fontsize_panel_labels"],
        fontweight="bold",
        ha="left",
        va="top",
    )

    # Right side panel labels for each column
    right_start = plots_start + left_width + gap_width

    for col_idx in range(_N_COLUMNS):
        col_x = (
            right_start + col_idx * (column_width + column_gap) + 1.5 * panel_x_offset
        )
        col_label = EXPERIMENT_COLUMN_LABELS[col_idx]

        # Panel C row
        if col_idx == 0:
            label_c = f"C) {col_label}"
        else:
            label_c = col_label
        fig.text(
            x=col_x,
            y=layout["right_upper_top"] + 1.2 * panel_y_offset,
            s=label_c,
            fontsize=FORMATTING["fontsize_panel_labels"],
            fontweight="bold",
            ha="left",
            va="top",
        )

        # Panel D row
        if col_idx == 0:
            label_d = f"D) {col_label}"
        else:
            label_d = col_label
        fig.text(
            x=col_x,
            y=layout["right_middle_top"] + 1.2 * panel_y_offset,
            s=label_d,
            fontsize=FORMATTING["fontsize_panel_labels"],
            fontweight="bold",
            ha="left",
            va="top",
        )

        # Panel E row
        if col_idx == 0:
            label_e = f"E) {col_label}"
        else:
            label_e = col_label
        fig.text(
            x=col_x,
            y=layout["right_lower_top"] + 1.2 * panel_y_offset,
            s=label_e,
            fontsize=FORMATTING["fontsize_panel_labels"],
            fontweight="bold",
            ha="left",
            va="top",
        )


def _add_column_titles(fig):
    """Add descriptive titles above Panel C columns.

    - Column i (interval):  "Interval"
    - Column ii (duration): "Duration"
    - Column iii (contrast): "Contrast"
    """
    layout = ALL_DYNAMICS_LAYOUT
    fmt = FORMATTING
    plots_start = layout["plots_start"]
    left_width = layout["left_plots_width"]
    gap_width = layout["gap_width"]
    column_width = layout["column_width"]
    column_gap = layout["column_gap"]

    title_kw = dict(
        fontsize=fmt["fontsize_panel_labels"],
        fontweight="bold",
        ha="center",
        va="bottom",
        color="dimgray",
    )

    titles = ["Interval", "Duration", "Contrast"]
    right_start = plots_start + left_width + gap_width
    title_y = layout["right_upper_top"] + 0.012

    # Panel A: title centered above left-side performance panel
    panel_a_x = plots_start + left_width / 2
    panel_a_y = layout["upper_top"] + 0.015
    fig.text(panel_a_x, panel_a_y, "Interval", **title_kw)

    for col_idx, title in enumerate(titles):
        col_x = right_start + col_idx * (column_width + column_gap) + column_width / 2
        fig.text(col_x, title_y, title, **title_kw)


# =============================================================================
# Panel A: Performance Traces
# =============================================================================


def create_panel_a_performance(
    fig,
    df: pd.DataFrame,
    parameter: str,
    category: str,
    middle_param_value,
    hue_values: List,
    colors: Dict[str, str],
    dt: float,
    config: Dict,
    time_offset: float = 0.0,
) -> plt.Axes:
    """Create Panel A showing performance traces (accuracy + confidence).

    Shows accuracy (solid line with circle markers) and confidence (dotted line
    with square markers) over time for the middle parameter value, with lines
    colored by category.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to add panel to
    df : pd.DataFrame
        Model response data with accuracy and confidence columns
    parameter : str
        Parameter column name
    category : str
        Category column name for hue coloring
    middle_param_value : any
        The middle parameter value to display
    hue_values : list
        Unique category values for coloring
    colors : dict
        Color mapping for categories
    dt : float
        Time step in milliseconds
    config : dict
        Configuration dictionary

    Returns
    -------
    ax : matplotlib.axes.Axes
        Axes containing performance plot
    """
    layout = ALL_DYNAMICS_LAYOUT

    # Panel position (same as original Panel A)
    plots_start = layout["plots_start"]
    left_width = layout["left_plots_width"]
    upper_top = layout["upper_top"]
    upper_bottom = layout["upper_bottom"]

    panel_height = upper_top - upper_bottom
    ax = fig.add_axes([plots_start, upper_bottom, left_width, panel_height])

    # Filter data for middle parameter value
    # Convert both to string to ensure type consistency
    df_filtered = df[df[parameter].astype(str) == str(middle_param_value)].copy()

    if len(df_filtered) == 0:
        ax.text(
            0.5,
            0.5,
            f"No data for {parameter}={middle_param_value}",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return ax

    # Create time in ms
    if "time_ms" not in df_filtered.columns and "times_index" in df_filtered.columns:
        df_filtered["time_ms"] = df_filtered["times_index"] * dt + time_offset

    # Standardize category values
    df_filtered[category] = df_filtered[category].astype(str)
    hue_values_str = [str(val) for val in hue_values]

    # Determine accuracy and confidence columns
    accuracy_col = None
    confidence_col = None

    for col in ["accuracy", "top1_accuracy", "classifier_accuracy"]:
        if col in df_filtered.columns:
            accuracy_col = col
            break

    for col in ["first_label_confidence", "confidence", "top1_confidence"]:
        if col in df_filtered.columns:
            confidence_col = col
            break

    if accuracy_col is None and confidence_col is None:
        ax.text(
            0.5,
            0.5,
            "No accuracy or confidence data",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return ax

    # Plot accuracy traces (solid lines with circle markers)
    if accuracy_col is not None:
        sns.lineplot(
            data=df_filtered,
            x="time_ms",
            y=accuracy_col,
            hue=category,
            hue_order=hue_values_str,
            palette=colors,
            ax=ax,
            legend=False,
            linewidth=FORMATTING["linewidth_main"],
            linestyle="-",
            marker="o",
            markersize=3,
            alpha=FORMATTING["alpha_line"],
            errorbar=("se", 1),
            err_style="band",
        )

    # Plot confidence traces (dotted lines with square markers)
    if confidence_col is not None:
        sns.lineplot(
            data=df_filtered,
            x="time_ms",
            y=confidence_col,
            hue=category,
            hue_order=hue_values_str,
            palette=colors,
            ax=ax,
            legend=False,
            linewidth=FORMATTING["linewidth_main"],
            linestyle=":",
            marker="s",
            markersize=2,
            alpha=FORMATTING["alpha_line"],
            errorbar=None,
        )

    # Add label indicator step function
    try:
        label_indicator_df = calculate_label_indicator(
            df_filtered,
            category,
            (0, 1),
            0.15,
        )
        indicator_time = label_indicator_df["times_index"] * dt + time_offset
        ax.plot(
            indicator_time,
            label_indicator_df["label_indicator"],
            color="dimgray",
            linewidth=FORMATTING["linewidth_indicator"],
            drawstyle="steps-mid",
            alpha=FORMATTING["alpha_indicator"],
        )
    except Exception as e:
        logger.warning(f"Could not calculate label indicator: {e}")

    # Styling
    ax.set_xlabel("Time (ms)", fontsize=FORMATTING["fontsize_label"])
    ax.set_ylabel(
        "Performance",
        fontsize=FORMATTING["fontsize_label"],
        fontweight="bold",
    )
    ax.tick_params(axis="both", labelsize=FORMATTING["fontsize_tick"])
    ax.grid(True, alpha=0.3, linestyle=":", zorder=1)
    sns.despine(ax=ax, left=True, bottom=True)

    # Add legend for Accuracy/Confidence line styles
    legend_elements = []
    if accuracy_col is not None:
        legend_elements.append(
            Line2D(
                [0],
                [0],
                color="black",
                linewidth=FORMATTING["linewidth_main"],
                linestyle="-",
                marker="o",
                markersize=4,
                label="Accuracy",
                alpha=FORMATTING["alpha_line"],
            )
        )
    if confidence_col is not None:
        legend_elements.append(
            Line2D(
                [0],
                [0],
                color="black",
                linewidth=FORMATTING["linewidth_main"],
                linestyle=":",
                marker="s",
                markersize=3,
                label="Confidence",
                alpha=FORMATTING["alpha_line"],
            )
        )

    if legend_elements:
        ax.legend(
            handles=legend_elements,
            loc="lower center",
            frameon=False,
            fontsize=FORMATTING["fontsize_legend"],
            ncol=1,
        )

    # Add parameter value annotation
    param_display = get_display_name(parameter, config)
    param_val_str = str(middle_param_value)
    if parameter.lower() in DT_CONVERT_PARAMS and dt is not None:
        try:
            param_val_ms = float(middle_param_value) * dt
            param_val_str = f"{param_display}={param_val_ms:.0f} ms"
        except (ValueError, TypeError):
            pass

    ax.text(
        0.95,
        0.25,
        param_val_str,
        horizontalalignment="right",
        verticalalignment="center",
        transform=ax.transAxes,
        fontsize=FORMATTING["fontsize_label"],
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="white",
            edgecolor="gray",
            alpha=0.8,
        ),
    )

    return ax


# =============================================================================
# Panel C: Groen Data (per column)
# =============================================================================


def _get_column_position(
    column_idx: int, row: str
) -> Tuple[float, float, float, float]:
    """Calculate axes position for a panel in the right-side grid.

    Parameters
    ----------
    column_idx : int
        Column index (0, 1, or 2 for i, ii, iii)
    row : str
        Row identifier: 'upper' (C), 'middle' (D), or 'lower' (E)

    Returns
    -------
    tuple
        (left, bottom, width, height) in figure coordinates
    """
    layout = ALL_DYNAMICS_LAYOUT
    plots_start = layout["plots_start"]
    left_width = layout["left_plots_width"]
    gap_width = layout["gap_width"]
    column_width = layout["column_width"]
    column_gap = layout["column_gap"]

    # Calculate horizontal position
    right_start = plots_start + left_width + gap_width
    left = right_start + column_idx * (column_width + column_gap)

    # Calculate vertical position based on row
    if row == "upper":
        bottom = layout["right_upper_bottom"]
        top = layout["right_upper_top"]
    elif row == "middle":
        bottom = layout["right_middle_bottom"]
        top = layout["right_middle_top"]
    else:  # lower
        bottom = layout["right_lower_bottom"]
        top = layout["right_lower_top"]

    height = top - bottom

    return (left, bottom, column_width, height)


def create_panel_c_groen_column(
    fig,
    experiment_type: str,
    groen_data_path: Path,
    column_idx: int,
) -> plt.Axes:
    """Create Panel C (Groen data) for a specific experiment column.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to add panel to
    experiment_type : str
        Type of experiment ('interval', 'duration', or 'contrast')
    groen_data_path : Path
        Path to the Groen CSV file for this experiment
    column_idx : int
        Column index (0, 1, or 2)

    Returns
    -------
    ax : matplotlib.axes.Axes
        Axes containing Groen plot
    """
    pos = _get_column_position(column_idx, "upper")
    ax = fig.add_axes(pos)

    # ax.set_title(
    #     experiment_type.capitalize(),
    #     fontsize=FORMATTING["fontsize_title"],
    #     fontweight="bold",
    # )

    if groen_data_path is None or not Path(groen_data_path).exists():
        ax.text(
            0.5,
            0.5,
            f"Groen data not provided\nfor {experiment_type}",
            ha="center",
            va="center",
            fontsize=FORMATTING["fontsize_label"],
            alpha=0.7,
            transform=ax.transAxes,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        return ax

    try:
        df_groen = load_groen_data(experiment_type, groen_data_path.parent)

        exp_info = determine_experiment_info(experiment_type)
        exp_type = exp_info["type"]

        if exp_type == "duration":
            plot_groen_temporal_summation(ax, df_groen)
            # Show only "Linear Prediction" legend for duration (column ii)
            ax.legend().remove()
            linear_pred_line = Line2D(
                [0],
                [0],
                color="black",
                linestyle="--",
                linewidth=1.5,
                alpha=0.5,
                label="Linear Prediction",
            )
            ax.legend(
                handles=[linear_pred_line],
                loc="lower right",
                frameon=False,
                fontsize=FORMATTING["fontsize_legend"],
            )
        elif exp_type == "interval":
            # Keep full Groen legend for interval (column i)
            plot_groen_adaptation_recovery_simple(ax, df_groen)
        elif exp_type == "contrast":
            plot_groen_contrast_time_to_peak(ax, df_groen)
            ax.legend().remove()
            # Fix x-axis label for contrast to be consistent
            ax.set_xlabel("Stimulus Contrast", fontsize=FORMATTING["fontsize_label"])
        else:
            ax.text(
                0.5,
                0.5,
                f"Unknown experiment type:\n{experiment_type}",
                ha="center",
                va="center",
                fontsize=FORMATTING["fontsize_label"],
                alpha=0.7,
                transform=ax.transAxes,
            )
            ax.set_xticks([])
            ax.set_yticks([])

    except FileNotFoundError as e:
        logger.error(str(e))
        ax.text(
            0.5,
            0.5,
            f"Groen data not found\n{experiment_type}",
            ha="center",
            va="center",
            fontsize=FORMATTING["fontsize_label"],
            alpha=0.7,
            transform=ax.transAxes,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    except Exception as e:
        logger.error(f"Error creating Groen panel: {e}")
        ax.text(
            0.5,
            0.5,
            f"Error loading Groen data:\n{str(e)[:30]}...",
            ha="center",
            va="center",
            fontsize=FORMATTING["fontsize_label"],
            alpha=0.7,
            transform=ax.transAxes,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    return ax


# =============================================================================
# Panel D: Metrics for Category (per column)
# =============================================================================


def create_panel_d_metrics_column(
    fig,
    df: pd.DataFrame,
    focus_layer: str,
    parameter: str,
    category: str,
    exp_info: Dict,
    hue_values: List,
    colors: Dict[str, str],
    dt: float,
    config: Dict,
    column_idx: int,
) -> plt.Axes:
    """Create Panel D (metric trend for category) in a specific column.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to add panel to
    df : pd.DataFrame
        Model response data
    focus_layer : str
        Layer to focus on
    parameter : str
        Parameter column name
    category : str
        Category column name
    exp_info : dict
        Experiment info dictionary
    hue_values : list
        Unique category values
    colors : dict
        Color mapping for categories
    dt : float
        Time step in milliseconds
    config : dict
        Configuration dictionary
    column_idx : int
        Column index (0, 1, or 2)

    Returns
    -------
    ax : matplotlib.axes.Axes
        Axes containing metrics plot
    """
    pos = _get_column_position(column_idx, "middle")
    ax = fig.add_axes(pos)

    metric = exp_info["metric"]
    metric_col = f"{focus_layer}_{metric}"

    if metric_col not in df.columns:
        df = calculate_metrics(
            df=df,
            focus_layer=focus_layer,
            parameter=parameter,
            category=category,
            metric=metric,
            dt=dt,
        )

    df = df.copy()
    df[category] = df[category].astype(str)
    hue_values_str = [str(val) for val in hue_values]

    # Check if parameter is temporal and needs conversion to ms
    is_temporal = exp_info["type"].lower() in DT_CONVERT_PARAMS and dt is not None

    # Create x-axis column (converted to ms if temporal)
    x_col = f"{parameter}_ms" if is_temporal else parameter
    if is_temporal:
        df[x_col] = df[parameter].astype(float) * dt

    if not df[metric_col].notna().any():
        ax.text(
            0.5,
            0.5,
            f"No metric data\nfor layer {focus_layer}",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return ax

    # Plot using seaborn lineplot
    sns.lineplot(
        data=df,
        x=x_col,
        y=metric_col,
        ax=ax,
        marker="o",
        hue=category,
        hue_order=hue_values_str,
        palette=colors,
        markersize=8,
        linewidth=FORMATTING["linewidth_main"],
        err_style="bars",
        errorbar="se",
        legend=False,
    )

    # Add reference line for peak_ratio metric
    if metric == "peak_ratio":
        ax.axhline(
            1.0,
            color="gray",
            linestyle=":",
            linewidth=1.5,
            alpha=0.7,
            zorder=1,
        )
    elif metric == "summed_response":
        x_sorted = df[x_col].sort_values()
        linear_prediction = x_sorted / df[x_col].max()
        ax.plot(
            x_sorted,
            linear_prediction,
            "k--",
            linewidth=1.5,
            alpha=0.5,
            zorder=2,
        )
        ax.set_xlim(0, df[x_col].max())
        ax.set_ylim(0, 1)

    # Styling
    metric_display = exp_info.get("metric_label", metric.replace("_", " ").title())

    # No x-axis label for Panel D (shared with Panel E below)
    ax.set_xlabel("")
    ax.set_ylabel(
        metric_display,
        fontsize=FORMATTING["fontsize_label"],
        fontweight="bold",
    )

    # Hide x tick labels (shared axis with Panel E)
    ax.tick_params(axis="y", labelsize=FORMATTING["fontsize_tick"])
    ax.tick_params(axis="x", labelbottom=False)
    ax.grid(True, alpha=0.3, linestyle=":", zorder=1)
    sns.despine(ax=ax, left=True, bottom=True)

    return ax


# =============================================================================
# Panel E: Metrics for Category2 (per column)
# =============================================================================


def create_panel_e_category2_column(
    fig,
    df_category2: pd.DataFrame,
    focus_layer: str,
    parameter: str,
    category2: str,
    exp_info: Dict,
    dt: float,
    config: Dict,
    column_idx: int,
    show_legend: bool = False,
) -> plt.Axes:
    """Create Panel E (metric trend for category2) in a specific column.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to add panel to
    df_category2 : pd.DataFrame
        Category-2 response data
    focus_layer : str
        Layer to focus on
    parameter : str
        Parameter column name
    category2 : str
        Category-2 column name
    exp_info : dict
        Experiment info dictionary
    dt : float
        Time step in milliseconds
    config : dict
        Configuration dictionary
    column_idx : int
        Column index (0, 1, or 2)
    show_legend : bool, optional
        Whether to show the category2 legend on this panel (default False)

    Returns
    -------
    ax : matplotlib.axes.Axes
        Axes containing category-2 metrics plot
    """
    pos = _get_column_position(column_idx, "lower")
    ax = fig.add_axes(pos)

    if df_category2 is None or category2 is None:
        ax.text(
            0.5,
            0.5,
            "Category-2 data\nnot provided",
            ha="center",
            va="center",
            fontsize=FORMATTING["fontsize_label"],
            alpha=0.7,
            transform=ax.transAxes,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        return ax

    # Extract category-2 values
    cat2_values = _extract_dimension_values(
        df_category2, "category2", category2, config
    )

    # Generate plasma colors for category-2
    plasma_cmap = plt.cm.plasma
    n_colors = len(cat2_values)
    colors_cat2 = {
        str(val): plasma_cmap(i / (n_colors - 1) if n_colors > 1 else 0.5)
        for i, val in enumerate(cat2_values)
    }

    metric = exp_info["metric"]
    metric_col = f"{focus_layer}_{metric}"

    df = df_category2.copy()
    if metric_col not in df.columns:
        df = calculate_metrics(
            df=df,
            focus_layer=focus_layer,
            parameter=parameter,
            category=category2,
            metric=metric,
            dt=dt,
        )

    df[category2] = df[category2].astype(str)
    cat2_values_str = [str(val) for val in cat2_values]

    # Check if parameter is temporal and needs conversion to ms
    is_temporal = exp_info["type"].lower() in DT_CONVERT_PARAMS and dt is not None

    # Create x-axis column (converted to ms if temporal)
    x_col = f"{parameter}_ms" if is_temporal else parameter
    if is_temporal:
        df[x_col] = df[parameter].astype(float) * dt

    if not df[metric_col].notna().any():
        ax.text(
            0.5,
            0.5,
            f"No metric data\nfor layer {focus_layer}",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return ax

    # Plot using seaborn lineplot
    sns.lineplot(
        data=df,
        x=x_col,
        y=metric_col,
        ax=ax,
        marker="o",
        hue=category2,
        hue_order=cat2_values_str,
        palette=colors_cat2,
        markersize=8,
        linewidth=FORMATTING["linewidth_main"],
        err_style="bars",
        errorbar="se",
    )

    # Add reference line for peak_ratio metric
    if metric == "peak_ratio":
        ax.axhline(
            1.0,
            color="gray",
            linestyle=":",
            linewidth=1.5,
            alpha=0.7,
            zorder=1,
        )
    elif metric == "summed_response":
        x_sorted = df[x_col].sort_values()
        linear_prediction = x_sorted / df[x_col].max()
        ax.plot(
            x_sorted,
            linear_prediction,
            "k--",
            linewidth=1.5,
            alpha=0.5,
            zorder=2,
        )
        ax.set_xlim(0, df[x_col].max())
        ax.set_ylim(0, 1)

    # Styling
    param_display = exp_info.get("full_label", get_display_name(parameter, config))
    metric_display = exp_info.get("metric_label", metric.replace("_", " ").title())

    # Build x-axis label with units for temporal parameters
    if is_temporal:
        x_label = f"{param_display} (ms)"
    else:
        x_label = param_display

    ax.set_xlabel(x_label, fontsize=FORMATTING["fontsize_label"])
    ax.set_ylabel(
        metric_display,
        fontsize=FORMATTING["fontsize_label"],
        fontweight="bold",
    )
    ax.tick_params(axis="both", labelsize=FORMATTING["fontsize_tick"])
    ax.grid(True, alpha=0.3, linestyle=":", zorder=1)
    sns.despine(ax=ax, left=True, bottom=True)

    # Add legend with plasma-colored lines (only on specified panel)
    if show_legend:
        legend = ax.legend(
            fontsize=FORMATTING["fontsize_legend"],
            frameon=True,
            fancybox=True,
            framealpha=0.7,
            loc="best",
            title=get_display_name(category2, config),
            title_fontsize=FORMATTING["fontsize_legend"],
        )
        legend.get_frame().set_facecolor("white")
        legend.get_frame().set_edgecolor("none")
        for text in legend.get_texts():
            if text.get_text() == "1.0":
                text.set_text("1.0 *")
        if legend.get_title():
            legend.get_title().set_fontweight("bold")
    else:
        # Remove the auto-generated legend from seaborn
        ax.legend().remove()

    return ax


# =============================================================================
# Main Plotting Function
# =============================================================================


def plot_all_dynamics_manuscript(
    # Data for each experiment
    df_interval: pd.DataFrame,
    df_duration: pd.DataFrame,
    df_contrast: pd.DataFrame,
    # Category2 data for each experiment
    df_category2_interval: Optional[pd.DataFrame],
    df_category2_duration: Optional[pd.DataFrame],
    df_category2_contrast: Optional[pd.DataFrame],
    # Groen data paths
    groen_data_interval: Optional[Path],
    groen_data_duration: Optional[Path],
    groen_data_contrast: Optional[Path],
    # Focus layers for B, D, E (one per experiment)
    focus_layers: Dict[str, str],
    # Parameters for each experiment (different per experiment)
    parameter_interval: str = "interval",
    parameter_duration: str = "duration",
    parameter_contrast: str = "contrast",
    # Category and category2 (same across all experiments)
    category: str = "contrast",
    category2: Optional[str] = None,
    # Output
    save_path: Optional[Path] = None,
    dt: Optional[float] = None,
    time_offset: float = 0.0,
    config: Optional[Dict] = None,
):
    """Create comprehensive dynamics manuscript figure with all three experiments.

    Parameters
    ----------
    df_interval, df_duration, df_contrast : pd.DataFrame
        Model response data for each experiment
    df_category2_interval, df_category2_duration, df_category2_contrast : pd.DataFrame, optional
        Category-2 response data for each experiment (for Panel E)
    groen_data_interval, groen_data_duration, groen_data_contrast : Path, optional
        Paths to Groen CSV files for each experiment
    focus_layers : Dict[str, str]
        Layer to focus on for each experiment, e.g. {"interval": "V1", "duration": "V2", "contrast": "IT"}
    parameter_interval, parameter_duration, parameter_contrast : str
        Parameter column names for each experiment (different per experiment)
    category : str
        Category column name (same across all experiments, shown in horizontal legend)
    category2 : str, optional
        Category-2 column name (same across all experiments, for Panel E)
    save_path : Path, optional
        Path to save figure
    dt : float, optional
        Time step in milliseconds
    config : dict, optional
        Configuration dictionary with palette, naming, ordering

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure
    """
    logger.info("=" * 60)
    logger.info("Starting all-dynamics manuscript plotting")
    logger.info("=" * 60)

    # Initialize config if not provided
    if config is None:
        config = {"palette": {}, "naming": {}, "ordering": {}}

    # Organize data by experiment (category and category2 are same across all)
    experiments_data = {
        "interval": {
            "df": df_interval,
            "df_category2": df_category2_interval,
            "groen_path": groen_data_interval,
            "parameter": parameter_interval,
        },
        "duration": {
            "df": df_duration,
            "df_category2": df_category2_duration,
            "groen_path": groen_data_duration,
            "parameter": parameter_duration,
        },
        "contrast": {
            "df": df_contrast,
            "df_category2": df_category2_contrast,
            "groen_path": groen_data_contrast,
            "parameter": parameter_contrast,
        },
    }

    # Use interval experiment for Panel A and B (first experiment)
    primary_exp = experiments_data["interval"]
    df_primary = primary_exp["df"]
    parameter_primary = primary_exp["parameter"]

    # Extract dimension values from primary experiment
    layer_names = _extract_dimension_values(df_primary, "layers", "layers", config)
    hue_values = _extract_dimension_values(df_primary, "category", category, config)
    param_values = _extract_dimension_values(
        df_primary, "parameter", parameter_primary, config
    )

    logger.info(f"Layers: {layer_names}")
    logger.info(f"Category values: {hue_values}")
    logger.info(f"Parameter values: {param_values}")
    logger.info(f"Focus layers: {focus_layers}")
    logger.info(f"Category: {category}, Category2: {category2}")

    # Get colors for category (same across all experiments)
    colors = _get_colors_for_dimension(hue_values, category, config)

    # Select middle parameter value for Panel A
    middle_idx = len(param_values) // 2
    middle_param_value = param_values[middle_idx]
    logger.info(f"Using middle parameter value for Panel A: {middle_param_value}")

    # Convert time if needed
    if dt is not None:
        df_primary = df_primary.copy()
        df_primary["time_ms"] = df_primary["times_index"] * dt + time_offset

    # Create figure
    fig = plt.figure(
        figsize=(
            ALL_DYNAMICS_LAYOUT["figure_width"],
            ALL_DYNAMICS_LAYOUT["figure_height"],
        )
    )
    sns.set_style("ticks")
    sns.set_context("talk")

    # Add panel labels
    add_panel_labels_all_experiments(fig)

    # Create Panel A: Performance traces
    logger.info("Creating Panel A: Performance traces")
    panel_a_ax = create_panel_a_performance(
        fig,
        df_primary,
        parameter_primary,
        category,
        middle_param_value,
        hue_values,
        colors,
        dt,
        config,
        time_offset=time_offset,
    )

    # Create Panel B: Ridge plots for focus layer (uses primary/interval experiment's focus layer)
    logger.info("Creating Panel B: Parameter variation ridge plots")
    panel_b_axes = create_panel_b(
        fig,
        df_primary,
        focus_layers[parameter_primary],
        parameter_primary,
        category,
        param_values,
        hue_values,
        colors,
        dt,
        config,
        normalize="max",
        time_offset=time_offset,
    )
    for ax in panel_b_axes:
        ax.set_yticks([0, 0.5])
        ax.set_yticklabels(["0", "0.5"])

    # Create right-side panels for each experiment column
    panel_c_axes = []
    panel_d_axes = []
    panel_e_axes = []

    for col_idx, exp_name in enumerate(EXPERIMENT_ORDER):
        exp_data = experiments_data[exp_name]
        df_exp = exp_data["df"]
        df_cat2 = exp_data["df_category2"]
        groen_path = exp_data["groen_path"]
        param = exp_data["parameter"]

        # Get experiment info
        exp_info = determine_experiment_info(exp_name, config)

        # Extract values for this experiment (using shared category)
        exp_hue_values = _extract_dimension_values(
            df_exp, "category", category, config
        )
        exp_colors = _get_colors_for_dimension(exp_hue_values, category, config)

        logger.info(f"Creating column {col_idx} ({exp_name})")

        # Panel C: Groen data
        logger.info(f"  Creating Panel C: Groen {exp_name} data")
        panel_c_ax = create_panel_c_groen_column(fig, exp_name, groen_path, col_idx)
        panel_c_axes.append(panel_c_ax)

        # Panel D: Metrics for category
        logger.info(f"  Creating Panel D: {exp_name} metrics (category)")
        panel_d_ax = create_panel_d_metrics_column(
            fig,
            df_exp,
            focus_layers[exp_name],
            param,
            category,
            exp_info,
            exp_hue_values,
            exp_colors,
            dt,
            config,
            col_idx,
        )
        panel_d_axes.append(panel_d_ax)

        # Panel E: Metrics for category2
        # Show legend only on the last column (rightmost)
        is_last_column = col_idx == len(EXPERIMENT_ORDER) - 1
        logger.info(f"  Creating Panel E: {exp_name} metrics (category2)")
        panel_e_ax = create_panel_e_category2_column(
            fig,
            df_cat2,
            focus_layers[exp_name],
            param,
            category2,
            exp_info,
            dt,
            config,
            col_idx,
            show_legend=is_last_column,
        )
        panel_e_axes.append(panel_e_ax)

        # Sync x-axis limits between panels D and E for this column
        if panel_d_ax and panel_e_ax:
            try:
                xlim_d = panel_d_ax.get_xlim()
                xlim_e = panel_e_ax.get_xlim()
                shared_xlim = (min(xlim_d[0], xlim_e[0]), max(xlim_d[1], xlim_e[1]))
                panel_d_ax.set_xlim(shared_xlim)
                panel_e_ax.set_xlim(shared_xlim)
                panel_d_ax.set_xticks(panel_e_ax.get_xticks())
            except Exception as e:
                logger.warning(f"Could not sync axes for column {col_idx}: {e}")

    # Add layer circle indicator for each column (centered between D and E rows)
    layout = ALL_DYNAMICS_LAYOUT
    circle_y = (layout["right_middle_bottom"] + layout["right_lower_top"]) / 2

    for col_idx, exp_name in enumerate(EXPERIMENT_ORDER):
        focus_layer = focus_layers[exp_name]
        col_left = (
            layout["plots_start"]
            + layout["left_plots_width"]
            + layout["gap_width"]
            + col_idx * (layout["column_width"] + layout["column_gap"])
        )
        circle_x = col_left + layout["column_width"] / 2

        layer_colors = _get_colors_for_dimension([focus_layer], "layers", config)
        layer_color = layer_colors.get(focus_layer, "#808080ff")
        pad = 0.5 if focus_layer == "IT" else 0.4

        fig.text(
            circle_x,
            circle_y,
            focus_layer,
            horizontalalignment="center",
            verticalalignment="center",
            transform=fig.transFigure,
            bbox=dict(
                boxstyle=f"circle,pad={pad}",
                facecolor=layer_color,
                edgecolor="#353535ff",
                linewidth=2,
                alpha=0.8,
            ),
            fontsize=FORMATTING["fontsize_label"],
            fontweight="bold",
        )

    # Add horizontal legend (between panels A and B)
    legend_left = ALL_DYNAMICS_LAYOUT["plots_start"]
    legend_width = ALL_DYNAMICS_LAYOUT["left_plots_width"]
    legend_height = ALL_DYNAMICS_LAYOUT["legend_height"]
    # Increased top padding (6× margin instead of 3×) to move legend
    # further from Panel A; reduced effective bottom padding as a side effect
    legend_bottom = (
        ALL_DYNAMICS_LAYOUT["lower_top"] - 3 * ALL_DYNAMICS_LAYOUT["legend_margin"]
    )

    legend = _add_horizontal_legend(
        fig=fig,
        hue_var="category",
        hue_key=category,
        hue_values=hue_values,
        colors=colors,
        config=config,
        dt=dt or 1.0,
        legend_bot=legend_bottom,
        **(
            FORMATTING
            | {
                "legend_height": legend_height,
                "left_margin": legend_left,
                "column_width": legend_width,
                "max_elements": 4,
            }
        ),
    )
    default_values = ["full", "output", "1.0"]
    if legend is not None:
        for text in legend.get_texts():
            label = text.get_text()
            if str(label).lower() in default_values:
                text.set_text(f"{label} *")

    # Add column titles
    _add_column_titles(fig)

    # Align y-axis labels across columns
    fig.align_ylabels()

    # Save figure
    if save_path:
        logger.info(f"Saving figure to {save_path}")
        save_plot(file_path=Path(save_path), dpi=300)

    logger.info("=" * 60)
    logger.info("All-dynamics manuscript plotting complete")
    logger.info("=" * 60)

    return fig


# =============================================================================
# CLI Entry Point
# =============================================================================


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Create comprehensive dynamics manuscript figure with all three experiments"
    )

    # Data inputs for each experiment
    parser.add_argument(
        "--data-interval",
        type=Path,
        nargs="+",
        required=True,
        help="Path(s) to interval experiment response data CSV file(s)",
    )
    parser.add_argument(
        "--data-duration",
        type=Path,
        nargs="+",
        required=True,
        help="Path(s) to duration experiment response data CSV file(s)",
    )
    parser.add_argument(
        "--data-contrast",
        type=Path,
        nargs="+",
        required=True,
        help="Path(s) to contrast experiment response data CSV file(s)",
    )

    # Category2 data inputs
    parser.add_argument(
        "--data-category2-interval",
        type=Path,
        nargs="*",
        help="Path(s) to interval experiment category-2 CSV file(s)",
    )
    parser.add_argument(
        "--data-category2-duration",
        type=Path,
        nargs="*",
        help="Path(s) to duration experiment category-2 CSV file(s)",
    )
    parser.add_argument(
        "--data-category2-contrast",
        type=Path,
        nargs="*",
        help="Path(s) to contrast experiment category-2 CSV file(s)",
    )

    # Groen data inputs
    parser.add_argument(
        "--groen-data-interval",
        type=Path,
        help="Path to Groen interval experiment CSV file",
    )
    parser.add_argument(
        "--groen-data-duration",
        type=Path,
        help="Path to Groen duration experiment CSV file",
    )
    parser.add_argument(
        "--groen-data-contrast",
        type=Path,
        help="Path to Groen contrast experiment CSV file",
    )

    # Output
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to save output figure",
    )

    # Focus layers (one per experiment, format: "interval+duration+contrast" e.g. "V1+V2+IT")
    parser.add_argument(
        "--focus-layers",
        type=str,
        default="*",
        help="Focus layers for each experiment as '+'-separated string (interval+duration+contrast), e.g. 'V1+V2+IT'. Use '*' for default (V1 for all).",
    )

    # Parameter specifications (different per experiment)
    parser.add_argument(
        "--parameter-interval",
        type=str,
        default="interval",
        help="Parameter column name for interval experiment",
    )
    parser.add_argument(
        "--parameter-duration",
        type=str,
        default="duration",
        help="Parameter column name for duration experiment",
    )
    parser.add_argument(
        "--parameter-contrast",
        type=str,
        default="contrast",
        help="Parameter column name for contrast experiment",
    )
    # Category specifications (same across all experiments)
    parser.add_argument(
        "--category",
        type=str,
        default="contrast",
        help="Category column name (same across all experiments, shown in horizontal legend)",
    )
    parser.add_argument(
        "--category2",
        type=str,
        help="Category-2 column name (same across all experiments, for Panel E)",
    )

    # Common parameters
    parser.add_argument("--dt", type=float, help="Time step in milliseconds")
    parser.add_argument(
        "--idle-timesteps",
        type=int,
        default=0,
        help="Number of idle timesteps before recorded data (shifts time axis)",
    )
    parser.add_argument("--palette", type=str, help="JSON color palette")
    parser.add_argument("--naming", type=str, help="JSON naming dict")
    parser.add_argument("--ordering", type=str, help="JSON ordering dict")

    args = parser.parse_args()

    # Helper to load and concatenate multiple CSV files
    def load_data_files(paths: List[Path]) -> pd.DataFrame:
        if not paths:
            return None
        if len(paths) > 1:
            logger.info(f"Loading and concatenating {len(paths)} files")
            dfs = [pd.read_csv(p) for p in paths]
            return pd.concat(dfs, ignore_index=True)
        else:
            logger.info(f"Loading data from: {paths[0]}")
            return pd.read_csv(paths[0])

    # Load main data
    df_interval = load_data_files(args.data_interval)
    df_duration = load_data_files(args.data_duration)
    df_contrast = load_data_files(args.data_contrast)

    # Load category2 data
    df_cat2_interval = (
        load_data_files(args.data_category2_interval)
        if args.data_category2_interval
        else None
    )
    df_cat2_duration = (
        load_data_files(args.data_category2_duration)
        if args.data_category2_duration
        else None
    )
    df_cat2_contrast = (
        load_data_files(args.data_category2_contrast)
        if args.data_category2_contrast
        else None
    )

    # Load config
    config = load_config_from_args(
        palette_str=args.palette,
        naming_str=args.naming,
        ordering_str=args.ordering,
    )

    # Parse focus layers (format: "interval+duration+contrast" or "*" for default)
    if args.focus_layers == "*":
        focus_layers = {"interval": "V1", "duration": "V1", "contrast": "V1"}
    else:
        layers = args.focus_layers.split("+")
        if len(layers) != 3:
            raise ValueError(
                f"focus-layers must be '*' or 3 '+'-separated values, got: {args.focus_layers}"
            )
        focus_layers = {
            "interval": layers[0],
            "duration": layers[1],
            "contrast": layers[2],
        }

    # Compute time offset from idle timesteps
    _dt = args.dt if args.dt is not None else 1.0
    time_offset = args.idle_timesteps * _dt

    # Create plot
    plot_all_dynamics_manuscript(
        df_interval=df_interval,
        df_duration=df_duration,
        df_contrast=df_contrast,
        df_category2_interval=df_cat2_interval,
        df_category2_duration=df_cat2_duration,
        df_category2_contrast=df_cat2_contrast,
        groen_data_interval=args.groen_data_interval,
        groen_data_duration=args.groen_data_duration,
        groen_data_contrast=args.groen_data_contrast,
        focus_layers=focus_layers,
        parameter_interval=args.parameter_interval,
        parameter_duration=args.parameter_duration,
        parameter_contrast=args.parameter_contrast,
        category=args.category,
        category2=args.category2,
        save_path=args.output,
        dt=args.dt,
        time_offset=time_offset,
        config=config,
    )


if __name__ == "__main__":
    main()
