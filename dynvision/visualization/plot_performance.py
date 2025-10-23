"""Plot performance traces with flexible dimension mapping.

This module provides a specialized plotting function for visualizing accuracy
and confidence traces across different experimental conditions with three flexible dimensions:
- Horizontal subplots 
- Vertical rows
- Hue (color coding)

Each dimension can represent: category, parameter, or experiment.
Supports multiple input files for different experiments.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union
import numpy as np

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Import functions and configurations from plot_responses
from dynvision.visualization.plot_responses import (
    FORMATTING as RESPONSES_FORMATTING,
    _filter_data_for_column,
    _format_legend_label,
    _get_colors_for_dimension,
    _get_dimension_key,
    _extract_dimension_values,
    _plot_accuracy_panel,
    _standardize_category_value,
    _validate_dimensions,
)

from dynvision.utils.visualization_utils import (
    calculate_label_indicator,
    get_color,
    get_display_name,
    get_ordering,
    load_config_from_args,
    order_layers,
    save_plot,
)

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel("INFO")

# Layout configuration for performance plots
PERFORMANCE_LAYOUT = {
    # Figure dimensions
    "subplot_width": 3,  # Width of each subplot in inches
    "subplot_height": 7.0,  # Height of each subplot in inches
    "subplot_spacing_x": 0.3,  # Horizontal spacing between subplots
    "subplot_spacing_y": 0.7,  # Vertical spacing between rows
    "peak_panel_width": 3.0,  # Width of peak height panel
    "peak_panel_spacing": 1,  # Spacing before peak panel
    "peak_time_panel_width": 3.0,  # Width of peak time panel
    "peak_time_panel_spacing": 1,  # Spacing before peak time panel
    # Margins
    "left_margin": 0.1,  # Left margin in inches
    "right_margin": 0.0,  # Right margin in inches
    "top_margin": 0.1,  # Top margin in inches
    "bottom_margin": 0.0,  # Bottom margin in inches
    # Title spacing
    "title_spacing": 0.08,
    # Panel letters
    "panel_letter_offset_x": -0.01,
    "panel_letter_offset_y": 0.03,
}

# Inherit formatting from plot_responses and override specific values
FORMATTING = {
    **RESPONSES_FORMATTING,  # Import all defaults from plot_responses
    "title_fontsize": 16,
    "fontsize_panel_label": 18,
}

# Errorbar configuration for lineplot
ERRORBAR_CONFIG = {
    "errorbar": None,  # ("ci", 99.999),  # Default: no errorbars
    "err_style": "bars",  # Style when errorbars are used
}


def _validate_dimensions(subplot_var: str, hue_var: str, row_var: str) -> None:
    """Validate that dimension variables are unique and supported."""
    supported_dims = ["category", "parameter", "experiment"]

    for dim_name, dim_var in [
        ("subplot", subplot_var),
        ("hue", hue_var),
        ("row", row_var),
    ]:
        if dim_var not in supported_dims:
            raise ValueError(
                f"{dim_name} dimension '{dim_var}' not supported. Must be one of: {supported_dims}"
            )

    used = [subplot_var, hue_var, row_var]
    if len(used) != len(set(used)):
        raise ValueError(
            f"Dimension variables must be unique. Got: subplot={subplot_var}, "
            f"hue={hue_var}, row={row_var}"
        )


def _filter_data_for_cell(
    df: pd.DataFrame,
    row_key: str,
    row_value: str,
    subplot_key: str,
    subplot_value: str,
) -> pd.DataFrame:
    """Filter data for a specific cell in the performance grid.

    Args:
        df: Input DataFrame
        row_key: Column name for row dimension
        row_value: Value for row dimension
        subplot_key: Column name for subplot dimension
        subplot_value: Value for subplot dimension

    Returns:
        Filtered DataFrame for the specific cell
    """
    # Filter data for this cell
    cell_data = df.copy()

    # Filter by row dimension
    if row_key in cell_data.columns:
        cell_data[row_key] = cell_data[row_key].apply(_standardize_category_value)
    row_value_std = _standardize_category_value(str(row_value))
    cell_data = cell_data[cell_data[row_key] == row_value_std]

    # Filter by subplot dimension
    if subplot_key in cell_data.columns:
        cell_data[subplot_key] = cell_data[subplot_key].apply(
            _standardize_category_value
        )
    subplot_value_std = _standardize_category_value(str(subplot_value))
    cell_data = cell_data[cell_data[subplot_key] == subplot_value_std]

    return cell_data


def _plot_accuracy_panel_with_ffonly(
    ax: plt.Axes,
    data: pd.DataFrame,
    hue_var: str,
    hue_key: str,
    hue_values: List[str],
    colors: Dict[str, str],
    dt: float,
    show_ylabel: bool,
    show_legend: bool,
    **kwargs,
) -> None:
    """Plot accuracy and confidence over time for both full and feedforward-only models.

    Args:
        ax: Matplotlib axes
        data: DataFrame with accuracy and confidence data (includes model_type column)
        hue_var: Variable for color coding
        hue_key: Column name for hue variable
        hue_values: Ordered list of hue values
        colors: Color mapping for hue values
        dt: Temporal resolution in ms
        show_ylabel: Whether to show y-axis label
        show_legend: Whether to show line style legend
        **kwargs: Override FORMATTING defaults
    """
    fmt = {**FORMATTING, **kwargs}

    # Extract errorbar settings from kwargs, with defaults from ERRORBAR_CONFIG
    errorbar_settings = {
        "errorbar": kwargs.get("errorbar", ERRORBAR_CONFIG["errorbar"]),
        "err_style": kwargs.get("err_style", ERRORBAR_CONFIG["err_style"]),
    }

    ax.patch.set_alpha(0)

    # Create time in ms
    data_plot = data.copy()
    data_plot["time_ms"] = data_plot["times_index"] * dt

    # Ensure hue column is standardized to match hue_values
    if hue_key in data_plot.columns:
        data_plot[hue_key] = data_plot[hue_key].apply(_standardize_category_value)

    # Check for required columns
    n_datapoints = len(data_plot)
    logger.info(f"Plotting accuracy panel with {n_datapoints} datapoints")

    has_accuracy = "accuracy" in data_plot.columns
    has_confidence = "confidence_avg" in data_plot.columns
    has_ffonly = (
        "model_type" in data_plot.columns
        and "ffonly" in data_plot["model_type"].values
    )

    if not has_accuracy and not has_confidence:
        logger.warning("Neither 'accuracy' nor 'confidence_avg' columns found in data")
        return

    # Split data by model type
    full_data = (
        data_plot[data_plot.get("model_type", "full") == "full"]
        if "model_type" in data_plot.columns
        else data_plot
    )
    ffonly_data = (
        data_plot[data_plot.get("model_type", "full") == "ffonly"]
        if has_ffonly
        else pd.DataFrame()
    )

    # Plot full model data (solid lines)
    if has_accuracy and len(full_data) > 0:
        logger.debug(
            f"Plotting full model accuracy with hue='{hue_key if hue_var != 'layers' else None}'"
        )
        sns.lineplot(
            data=full_data,
            x="time_ms",
            y="accuracy",
            hue=hue_key if hue_var != "layers" else None,
            hue_order=hue_values if hue_var != "layers" else None,
            palette=colors if hue_var != "layers" else None,
            ax=ax,
            legend=False,
            linewidth=fmt["linewidth_main"],
            marker="o",
            markersize=3,
            alpha=fmt["alpha_line"],
            linestyle="-",
            **errorbar_settings,
        )

    # Plot full model confidence (dotted lines)
    if has_confidence and len(full_data) > 0:
        logger.debug(
            f"Plotting full model confidence with hue='{hue_key if hue_var != 'layers' else None}'"
        )
        sns.lineplot(
            data=full_data,
            x="time_ms",
            y="confidence_avg",
            hue=hue_key if hue_var != "layers" else None,
            hue_order=hue_values if hue_var != "layers" else None,
            palette=colors if hue_var != "layers" else None,
            ax=ax,
            legend=False,
            linewidth=fmt["linewidth_main"],
            marker="s",
            markersize=2,
            alpha=fmt["alpha_line"],
            linestyle=":",
            **errorbar_settings,
        )

    # Plot feedforward-only data (dashed lines) with slightly reduced alpha
    if has_ffonly and len(ffonly_data) > 0:

        if has_accuracy:
            logger.debug(
                f"Plotting feedforward-only accuracy with hue='{hue_key if hue_var != 'layers' else None}'"
            )
            sns.lineplot(
                data=ffonly_data,
                x="time_ms",
                y="accuracy",
                hue=hue_key if hue_var != "layers" else None,
                hue_order=hue_values if hue_var != "layers" else None,
                palette=colors if hue_var != "layers" else None,
                ax=ax,
                legend=False,
                linewidth=fmt["linewidth_main"],
                marker="^",
                markersize=2,
                alpha=fmt["alpha_line"],
                linestyle="--",
                **errorbar_settings,
            )

        if has_confidence:
            logger.debug(
                f"Plotting feedforward-only confidence with hue='{hue_key if hue_var != 'layers' else None}'"
            )
            sns.lineplot(
                data=ffonly_data,
                x="time_ms",
                y="confidence_avg",
                hue=hue_key if hue_var != "layers" else None,
                hue_order=hue_values if hue_var != "layers" else None,
                palette=colors if hue_var != "layers" else None,
                ax=ax,
                legend=False,
                linewidth=fmt["linewidth_main"],
                marker="v",
                markersize=1,
                alpha=fmt["alpha_line"],
                linestyle="-.",
                **errorbar_settings,
            )

    if show_ylabel:
        ax.set_ylabel("Performance", fontsize=fmt["fontsize_axis"], fontweight="bold")
    else:
        ax.set_ylabel("")

    ax.set_xlabel("Time (ms)", fontsize=fmt["fontsize_axis"])
    ax.tick_params(labelsize=fmt["fontsize_tick"])

    # Add legend for line styles on first panel
    if show_legend:
        legend_elements = []

        if has_accuracy:
            legend_elements.append(
                plt.Line2D(
                    [0],
                    [0],
                    color="black",
                    linewidth=fmt["linewidth_main"],
                    linestyle="-",
                    label="Accuracy",
                    alpha=fmt["alpha_line"],
                )
            )
            if has_ffonly:
                legend_elements.append(
                    plt.Line2D(
                        [0],
                        [0],
                        color="black",
                        linewidth=fmt["linewidth_main"],
                        linestyle="--",
                        label="Accuracy (Feedforward)",
                        alpha=fmt["alpha_line"],
                    )
                )

        if has_confidence:
            legend_elements.append(
                plt.Line2D(
                    [0],
                    [0],
                    color="black",
                    linewidth=fmt["linewidth_main"],
                    linestyle=":",
                    label="Confidence",
                    alpha=fmt["alpha_line"],
                )
            )
            if has_ffonly:
                legend_elements.append(
                    plt.Line2D(
                        [0],
                        [0],
                        color="black",
                        linewidth=fmt["linewidth_main"],
                        linestyle="-.",
                        label="Confidence (Feedforward)",
                        alpha=fmt["alpha_line"],
                    )
                )

        if legend_elements:
            ax.legend(
                handles=legend_elements,
                loc="upper left",
                frameon=False,
                fontsize=fmt["fontsize_legend"],
                ncol=1,  # Single row
            )

    ax.grid(True, alpha=0.3)

    sns.despine(ax=ax, left=True)


def _plot_peak_height_panel(
    ax: plt.Axes,
    data: pd.DataFrame,
    row_key: str,
    row_value: str,
    subplot_key: str,
    subplot_values: List[str],
    hue_key: str,
    hue_values: List[str],
    colors: Dict[str, str],
    config: Dict,
    dt: float,
    show_ylabel: bool = False,
    show_legend: bool = True,
) -> None:
    """Plot max accuracy and confidence vs subplot value for each hue category."""
    # Filter data for this row
    row_data = data.copy()
    if row_key in row_data.columns:
        row_data[row_key] = row_data[row_key].apply(_standardize_category_value)
    row_value_std = _standardize_category_value(str(row_value))
    row_data = row_data[row_data[row_key] == row_value_std]

    if len(row_data) == 0:
        ax.text(0.5, 0.5, "No Data", ha="center", va="center", transform=ax.transAxes)
        return

    # Check for required columns
    has_accuracy = "accuracy" in row_data.columns
    has_confidence = "confidence_avg" in row_data.columns
    has_ffonly = (
        "model_type" in row_data.columns and "ffonly" in row_data["model_type"].values
    )

    if not has_accuracy and not has_confidence:
        ax.text(
            0.5,
            0.5,
            "No Performance Data",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return

    # Clean NaN values
    if has_accuracy:
        row_data = row_data.dropna(subset=["accuracy"])
    if has_confidence:
        row_data = row_data.dropna(subset=["confidence_avg"])

    if len(row_data) == 0:
        ax.text(
            0.5, 0.5, "No Valid Data", ha="center", va="center", transform=ax.transAxes
        )
        return

    logger.debug(
        f"Computing max performance metrics from accuracy and confidence columns"
    )

    # Standardize hue and subplot columns for consistent filtering
    if hue_key in row_data.columns:
        row_data[hue_key] = row_data[hue_key].apply(_standardize_category_value)
    if subplot_key in row_data.columns:
        row_data[subplot_key] = row_data[subplot_key].apply(
            _standardize_category_value
        )

    # Split data by model type
    full_data = (
        row_data[row_data.get("model_type", "full") == "full"]
        if "model_type" in row_data.columns
        else row_data
    )
    ffonly_data = (
        row_data[row_data.get("model_type", "full") == "ffonly"]
        if has_ffonly
        else pd.DataFrame()
    )

    # Prepare data for plotting - compute max of averages
    accuracy_plot_data = []
    confidence_plot_data = []
    ffonly_accuracy_plot_data = []

    for hue_val in hue_values:
        hue_val_std = _standardize_category_value(str(hue_val))

        # Full model data
        hue_data_full = (
            full_data[full_data[hue_key] == hue_val_std]
            if hue_key in full_data.columns
            else full_data
        )

        # Feedforward-only model data
        hue_data_ffonly = (
            ffonly_data[ffonly_data[hue_key] == hue_val_std]
            if hue_key in ffonly_data.columns and not ffonly_data.empty
            else ffonly_data
        )

        for subplot_val in subplot_values:
            subplot_val_std = _standardize_category_value(str(subplot_val))

            # Filter full model data
            subplot_data_full = (
                hue_data_full[hue_data_full[subplot_key] == subplot_val_std]
                if subplot_key in hue_data_full.columns
                else hue_data_full
            )

            # Filter feedforward-only model data
            subplot_data_ffonly = (
                hue_data_ffonly[hue_data_ffonly[subplot_key] == subplot_val_std]
                if subplot_key in hue_data_ffonly.columns and not hue_data_ffonly.empty
                else hue_data_ffonly
            )

            # Convert subplot value to numeric if possible for plotting
            try:
                x_val = float(subplot_val)
            except (ValueError, TypeError):
                # If not numeric, use index position
                x_val = subplot_values.index(subplot_val)

            # Process full model data
            if len(subplot_data_full) > 0:
                # Calculate max of the average accuracy/confidence across presentation labels
                if has_accuracy:
                    # Group by time and calculate mean across labels, then take max across time
                    if "times_index" in subplot_data_full.columns:
                        avg_accuracy_over_time = subplot_data_full.groupby(
                            "times_index"
                        )["accuracy"].mean()
                        # Check for NaN values and handle them
                        if (
                            not avg_accuracy_over_time.empty
                            and not avg_accuracy_over_time.isna().all()
                        ):
                            max_avg_accuracy = avg_accuracy_over_time.max()
                            if np.isfinite(max_avg_accuracy):
                                accuracy_plot_data.append(
                                    {
                                        "x": x_val,
                                        "y": max_avg_accuracy,
                                        "hue": hue_val,
                                        "subplot_val": subplot_val,
                                        "metric": "accuracy",
                                        "model_type": "full",
                                    }
                                )
                    else:
                        mean_accuracy = subplot_data_full["accuracy"].mean()
                        if np.isfinite(mean_accuracy):
                            accuracy_plot_data.append(
                                {
                                    "x": x_val,
                                    "y": mean_accuracy,
                                    "hue": hue_val,
                                    "subplot_val": subplot_val,
                                    "metric": "accuracy",
                                    "model_type": "full",
                                }
                            )

                # Calculate max of the average confidence across presentation labels
                if has_confidence:
                    # Group by time and calculate mean across labels, then take max across time
                    if "times_index" in subplot_data_full.columns:
                        avg_confidence_over_time = subplot_data_full.groupby(
                            "times_index"
                        )["confidence_avg"].mean()
                        # Check for NaN values and handle them
                        if (
                            not avg_confidence_over_time.empty
                            and not avg_confidence_over_time.isna().all()
                        ):
                            max_avg_confidence = avg_confidence_over_time.max()
                            if np.isfinite(max_avg_confidence):
                                confidence_plot_data.append(
                                    {
                                        "x": x_val,
                                        "y": max_avg_confidence,
                                        "hue": hue_val,
                                        "subplot_val": subplot_val,
                                        "metric": "confidence",
                                        "model_type": "full",
                                    }
                                )
                    else:
                        mean_confidence = subplot_data_full["confidence_avg"].mean()
                        if np.isfinite(mean_confidence):
                            confidence_plot_data.append(
                                {
                                    "x": x_val,
                                    "y": mean_confidence,
                                    "hue": hue_val,
                                    "subplot_val": subplot_val,
                                    "metric": "confidence",
                                    "model_type": "full",
                                }
                            )

            # Process feedforward-only model data
            if has_ffonly and len(subplot_data_ffonly) > 0:
                # Calculate max of the average accuracy across presentation labels
                if has_accuracy:
                    # Group by time and calculate mean across labels, then take max across time
                    if "times_index" in subplot_data_ffonly.columns:
                        avg_accuracy_over_time = subplot_data_ffonly.groupby(
                            "times_index"
                        )["accuracy"].mean()
                        # Check for NaN values and handle them
                        if (
                            not avg_accuracy_over_time.empty
                            and not avg_accuracy_over_time.isna().all()
                        ):
                            max_avg_accuracy = avg_accuracy_over_time.max()
                            if np.isfinite(max_avg_accuracy):
                                ffonly_accuracy_plot_data.append(
                                    {
                                        "x": x_val,
                                        "y": max_avg_accuracy,
                                        "hue": hue_val,
                                        "subplot_val": subplot_val,
                                        "metric": "accuracy",
                                        "model_type": "ffonly",
                                    }
                                )
                    else:
                        mean_accuracy = subplot_data_ffonly["accuracy"].mean()
                        if np.isfinite(mean_accuracy):
                            ffonly_accuracy_plot_data.append(
                                {
                                    "x": x_val,
                                    "y": mean_accuracy,
                                    "hue": hue_val,
                                    "subplot_val": subplot_val,
                                    "metric": "accuracy",
                                    "model_type": "ffonly",
                                }
                            )

    if (
        not accuracy_plot_data
        and not confidence_plot_data
        and not ffonly_accuracy_plot_data
    ):
        ax.text(
            0.5, 0.5, "No Valid Data", ha="center", va="center", transform=ax.transAxes
        )
        return

    # Plot full model accuracy data (solid lines with circles)
    if accuracy_plot_data:
        accuracy_df = pd.DataFrame(accuracy_plot_data)
        # Remove any remaining NaN values
        accuracy_df = accuracy_df.dropna(subset=["x", "y"])

        if not accuracy_df.empty:
            sns.lineplot(
                data=accuracy_df,
                x="x",
                y="y",
                hue="hue",
                hue_order=hue_values,
                palette=colors,
                ax=ax,
                legend=False,
                linewidth=FORMATTING["linewidth_main"],
                marker="o",
                markersize=6,
                alpha=FORMATTING["alpha_line"],
                linestyle="-",
                **ERRORBAR_CONFIG,
            )

    # Plot feedforward-only accuracy data (dashed lines with triangles)
    if has_ffonly and ffonly_accuracy_plot_data:
        ffonly_accuracy_df = pd.DataFrame(ffonly_accuracy_plot_data)
        # Remove any remaining NaN values
        ffonly_accuracy_df = ffonly_accuracy_df.dropna(subset=["x", "y"])

        if not ffonly_accuracy_df.empty:
            sns.lineplot(
                data=ffonly_accuracy_df,
                x="x",
                y="y",
                hue="hue",
                hue_order=hue_values,
                palette=colors,
                ax=ax,
                legend=False,
                linewidth=FORMATTING["linewidth_main"],
                marker="^",
                markersize=6,
                alpha=FORMATTING["alpha_line"],
                linestyle="--",
                **ERRORBAR_CONFIG,
            )

    # Plot full model confidence data (dotted lines with squares)
    if confidence_plot_data:
        confidence_df = pd.DataFrame(confidence_plot_data)
        # Remove any remaining NaN values
        confidence_df = confidence_df.dropna(subset=["x", "y"])

        if not confidence_df.empty:
            sns.lineplot(
                data=confidence_df,
                x="x",
                y="y",
                hue="hue",
                hue_order=hue_values,
                palette=colors,
                ax=ax,
                legend=False,
                linewidth=FORMATTING["linewidth_main"],
                marker="s",
                markersize=4,
                alpha=FORMATTING["alpha_line"],
                linestyle=":",
                **ERRORBAR_CONFIG,
            )

    # Add y-label
    if show_ylabel:
        ax.set_ylabel(
            "Performance", fontsize=FORMATTING["fontsize_axis"], fontweight="bold"
        )
    else:
        ax.set_ylabel("")

    subplot_display = get_display_name(subplot_key, config)
    ax.set_xlabel(subplot_display, fontsize=FORMATTING["fontsize_axis"])
    ax.tick_params(labelsize=FORMATTING["fontsize_tick"])

    # Set reasonable y-limits to handle potential axis issues
    try:
        current_ylim = ax.get_ylim()
        if not (np.isfinite(current_ylim[0]) and np.isfinite(current_ylim[1])):
            ax.set_ylim(0, 1)
    except:
        ax.set_ylim(0, 1)

    # Set x-axis labels if not numeric
    try:
        # Check if all subplot values are numeric
        numeric_vals = [float(v) for v in subplot_values]
    except (ValueError, TypeError):
        # Use categorical labels
        ax.set_xticks(range(len(subplot_values)))
        ax.set_xticklabels(
            [_format_legend_label(subplot_key, v, config, dt) for v in subplot_values],
            rotation=45,
            ha="right",
        )

    # Add legend for line styles (accuracy vs confidence vs feedforward)
    legend_elements = []

    if has_accuracy and show_legend:
        legend_elements.append(
            plt.Line2D(
                [0],
                [0],
                color="gray",
                linewidth=FORMATTING["linewidth_main"],
                marker="o",
                markersize=6,
                linestyle="-",
                label="Max Accuracy",
                alpha=FORMATTING["alpha_line"],
            )
        )

        if has_ffonly:
            legend_elements.append(
                plt.Line2D(
                    [0],
                    [0],
                    color="gray",
                    linewidth=FORMATTING["linewidth_main"],
                    marker="^",
                    markersize=6,
                    linestyle="--",
                    label="Max Accuracy (FF-only)",
                    alpha=FORMATTING["alpha_line"],
                )
            )

    if has_confidence and show_legend:
        legend_elements.append(
            plt.Line2D(
                [0],
                [0],
                color="gray",
                linewidth=FORMATTING["linewidth_main"],
                marker="s",
                markersize=4,
                linestyle=":",
                label="Max Confidence",
                alpha=FORMATTING["alpha_line"],
            )
        )

    if legend_elements:
        ax.legend(
            handles=legend_elements,
            loc="upper left",
            bbox_to_anchor=(0, 1 + PERFORMANCE_LAYOUT["panel_letter_offset_y"]),
            frameon=False,
            fontsize=FORMATTING["fontsize_legend"],
        )

    ax.grid(True, alpha=0.3)
    sns.despine(ax=ax)


def _plot_peak_time_panel(
    ax: plt.Axes,
    data: pd.DataFrame,
    row_key: str,
    row_value: str,
    subplot_key: str,
    subplot_values: List[str],
    hue_key: str,
    hue_values: List[str],
    colors: Dict[str, str],
    config: Dict,
    dt: float,
    show_ylabel: bool = False,
    show_legend: bool = True,
) -> None:
    """Plot time of max accuracy and confidence vs subplot value for each hue category.

    Args:
        ax: Matplotlib axes
        data: Full dataset
        row_key: Key for row dimension
        row_value: Value for this row
        subplot_key: Key for subplot dimension (x-axis)
        subplot_values: All subplot values for x-axis
        hue_key: Key for hue dimension (color coding)
        hue_values: All hue values for color coding
        colors: Color mapping for hue values
        config: Configuration dict
        dt: Temporal resolution
        show_ylabel: Whether to show y-axis label
    """
    # Filter data for this row
    row_data = data.copy()
    if row_key in row_data.columns:
        row_data[row_key] = row_data[row_key].apply(_standardize_category_value)
    row_value_std = _standardize_category_value(str(row_value))
    row_data = row_data[row_data[row_key] == row_value_std]

    if len(row_data) == 0:
        ax.text(0.5, 0.5, "No Data", ha="center", va="center", transform=ax.transAxes)
        return

    # Check for required columns
    has_accuracy = "accuracy" in row_data.columns
    has_confidence = "confidence_avg" in row_data.columns

    if not has_accuracy and not has_confidence:
        ax.text(
            0.5,
            0.5,
            "No Performance Data",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return

    logger.debug(f"Computing peak time metrics from accuracy and confidence columns")

    # Standardize hue and subplot columns for consistent filtering
    if hue_key in row_data.columns:
        row_data[hue_key] = row_data[hue_key].apply(_standardize_category_value)
    if subplot_key in row_data.columns:
        row_data[subplot_key] = row_data[subplot_key].apply(
            _standardize_category_value
        )

    # Prepare data for plotting - compute time of max of averages
    accuracy_plot_data = []
    confidence_plot_data = []

    for hue_val in hue_values:
        hue_val_std = _standardize_category_value(str(hue_val))
        hue_data = (
            row_data[row_data[hue_key] == hue_val_std]
            if hue_key in row_data.columns
            else row_data
        )

        for subplot_val in subplot_values:
            subplot_val_std = _standardize_category_value(str(subplot_val))
            subplot_data = (
                hue_data[hue_data[subplot_key] == subplot_val_std]
                if subplot_key in hue_data.columns
                else hue_data
            )

            if len(subplot_data) > 0:
                # Convert subplot value to numeric if possible for plotting
                try:
                    x_val = float(subplot_val)
                except (ValueError, TypeError):
                    # If not numeric, use index position
                    x_val = subplot_values.index(subplot_val)

                # Calculate time of max of the average accuracy/confidence across presentation labels
                if has_accuracy and "times_index" in subplot_data.columns:
                    # Group by time and calculate mean across labels, then find time of max
                    avg_accuracy_over_time = subplot_data.groupby("times_index")[
                        "accuracy"
                    ].mean()
                    peak_time_idx = avg_accuracy_over_time.idxmax()
                    peak_time_ms = peak_time_idx * dt

                    accuracy_plot_data.append(
                        {
                            "x": x_val,
                            "y": peak_time_ms,
                            "hue": hue_val,
                            "subplot_val": subplot_val,
                            "metric": "accuracy",
                        }
                    )

                # Calculate time of max of the average confidence across presentation labels
                if has_confidence and "times_index" in subplot_data.columns:
                    # Group by time and calculate mean across labels, then find time of max
                    avg_confidence_over_time = subplot_data.groupby("times_index")[
                        "confidence_avg"
                    ].mean()
                    peak_time_idx = avg_confidence_over_time.idxmax()
                    peak_time_ms = peak_time_idx * dt

                    confidence_plot_data.append(
                        {
                            "x": x_val,
                            "y": peak_time_ms,
                            "hue": hue_val,
                            "subplot_val": subplot_val,
                            "metric": "confidence",
                        }
                    )

    if not accuracy_plot_data and not confidence_plot_data:
        ax.text(
            0.5, 0.5, "No Valid Data", ha="center", va="center", transform=ax.transAxes
        )
        return

    # Plot accuracy data (solid lines with circles)
    if accuracy_plot_data:
        accuracy_df = pd.DataFrame(accuracy_plot_data)

        # Use seaborn lineplot with errorbar configuration
        sns.lineplot(
            data=accuracy_df,
            x="x",
            y="y",
            hue="hue",
            hue_order=hue_values,
            palette=colors,
            ax=ax,
            legend=False,
            linewidth=FORMATTING["linewidth_main"],
            marker="o",
            markersize=6,
            alpha=FORMATTING["alpha_line"],
            linestyle="-",
            **ERRORBAR_CONFIG,
        )

    # Plot confidence data (dashed lines with squares)
    if confidence_plot_data:
        confidence_df = pd.DataFrame(confidence_plot_data)

        # Use seaborn lineplot with errorbar configuration
        sns.lineplot(
            data=confidence_df,
            x="x",
            y="y",
            hue="hue",
            hue_order=hue_values,
            palette=colors,
            ax=ax,
            legend=False,
            linewidth=FORMATTING["linewidth_main"],
            marker="s",
            markersize=4,
            alpha=FORMATTING["alpha_line"],
            linestyle=":",
            **ERRORBAR_CONFIG,
        )

    # Styling
    if show_ylabel:
        ax.set_ylabel(
            "Peak Time (ms)", fontsize=FORMATTING["fontsize_axis"], fontweight="bold"
        )
    else:
        ax.set_ylabel("")

    subplot_display = get_display_name(subplot_key, config)
    ax.set_xlabel(subplot_display, fontsize=FORMATTING["fontsize_axis"])
    ax.tick_params(labelsize=FORMATTING["fontsize_tick"])

    # Set x-axis labels if not numeric
    try:
        # Check if all subplot values are numeric
        numeric_vals = [float(v) for v in subplot_values]
    except (ValueError, TypeError):
        # Use categorical labels
        ax.set_xticks(range(len(subplot_values)))
        ax.set_xticklabels(
            [_format_legend_label(subplot_key, v, config, dt) for v in subplot_values],
            rotation=45,
            ha="right",
        )

    # Add legend for line styles (accuracy vs confidence)
    if has_accuracy and has_confidence and show_legend:
        legend_elements = [
            plt.Line2D(
                [0],
                [0],
                color="gray",
                linewidth=FORMATTING["linewidth_main"],
                marker="o",
                markersize=6,
                linestyle="-",
                label="Peak Time (Accuracy)",
                alpha=FORMATTING["alpha_line"],
            ),
            plt.Line2D(
                [0],
                [0],
                color="gray",
                linewidth=FORMATTING["linewidth_main"],
                marker="s",
                markersize=4,
                linestyle=":",
                label="Peak Time (Confidence)",
                alpha=FORMATTING["alpha_line"],
            ),
        ]
        ax.legend(
            handles=legend_elements,
            loc="upper left",
            bbox_to_anchor=(0, 1 + PERFORMANCE_LAYOUT["panel_letter_offset_y"]),
            frameon=False,
            fontsize=FORMATTING["fontsize_legend"],
        )

    ax.grid(True, alpha=0.3)
    sns.despine(ax=ax)


def _add_panel_letters(
    fig: plt.Figure,
    perf_axes: List[List[plt.Axes]],
    peak_axes: List[plt.Axes],
    peak_time_axes: List[plt.Axes],
    layout: Dict,
) -> None:
    """Add panel letters A), B), C) to performance, peak, and peak time panels."""
    fmt = FORMATTING

    # Get positions from the first row of axes
    if not perf_axes or not perf_axes[0]:
        return

    # Panel A) - Above first performance subplot
    first_perf_ax = perf_axes[0][0]
    first_perf_ax.text(
        layout["panel_letter_offset_x"],
        1 + layout["panel_letter_offset_y"],
        "A)",
        fontsize=fmt["fontsize_panel_label"],
        fontweight="bold",
        ha="center",
        va="bottom",
        transform=first_perf_ax.transAxes,
    )

    # Panel B) - Above peak height panel
    if peak_axes:
        peak_ax = peak_axes[0]
        pos = peak_ax.get_position()
        peak_ax.text(
            layout["panel_letter_offset_x"],
            1 + layout["panel_letter_offset_y"],
            "B)",
            fontsize=fmt["fontsize_panel_label"],
            fontweight="bold",
            ha="center",
            va="bottom",
            transform=peak_ax.transAxes,
        )

    # Panel C) - Above peak time panel
    if peak_time_axes:
        peak_time_ax = peak_time_axes[0]
        pos = peak_time_ax.get_position()
        peak_time_ax.text(
            layout["panel_letter_offset_x"],
            1 + layout["panel_letter_offset_y"],
            "C)",
            fontsize=fmt["fontsize_panel_label"],
            fontweight="bold",
            ha="center",
            va="bottom",
            transform=peak_time_ax.transAxes,
        )


def _add_hue_legend(
    fig: plt.Figure,
    perf_axes: List[List[plt.Axes]],
    hue_key: str,
    hue_values: List[str],
    colors: Dict[str, str],
    config: Dict,
    dt: float,
) -> None:
    """Add horizontal hue legend at the top of performance plots."""
    if not hue_values or len(hue_values) <= 1 or not perf_axes or not perf_axes[0]:
        return

    # Use second performance subplot if available, otherwise first
    legend_ax = perf_axes[0][1] if len(perf_axes[0]) > 3 else perf_axes[0][-1]

    # Create legend elements
    legend_elements = []
    for hue_val in hue_values:
        color = colors.get(hue_val, "#000000")
        label = _format_legend_label(hue_key, hue_val, config, dt)
        legend_elements.append(
            plt.Line2D(
                [0],
                [0],
                color=color,
                linewidth=FORMATTING["linewidth_main"],
                label=label,
                alpha=FORMATTING["alpha_line"],
            )
        )

    # Add legend above the subplot
    legend_ax.legend(
        handles=legend_elements,
        loc="upper left",
        frameon=False,
        fontsize=FORMATTING["fontsize_legend"],
        ncol=len(hue_values),  # Horizontal layout
        title=get_display_name(hue_key, config),
        title_fontsize=FORMATTING["fontsize_legend"],
        # alignment="left",
    )


def plot_performance_grid(
    data_paths: List[Path],
    output: Path,
    subplot_var: Literal["category", "parameter", "experiment"],
    row_var: Literal["category", "parameter", "experiment"],
    hue_var: Literal["category", "parameter", "experiment"],
    category_key: Optional[str] = None,
    parameter_key: Optional[str] = None,
    experiment_names: Optional[List[str]] = None,
    data_ffonly_paths: Optional[List[Path]] = None,
    dt: float = 2.0,
    config: Optional[Dict] = None,
    subplot_filter: Optional[List[str]] = None,
    **kwargs,
) -> None:
    """Plot performance traces in a grid layout with flexible dimension mapping."""
    logger.info("=" * 60)
    logger.info("Starting performance grid plotting")
    logger.info("=" * 60)

    # Validation
    logger.info(
        f"Dimension mapping: subplot={subplot_var}, row={row_var}, hue={hue_var}"
    )
    _validate_dimensions(subplot_var=subplot_var, hue_var=hue_var, row_var=row_var)

    if config is None:
        config = {"palette": {}, "naming": {}, "ordering": {}}

    # Load and combine data from multiple files
    logger.info(f"Loading data from {len(data_paths)} full model files...")
    combined_data = []

    for i, data_path in enumerate(data_paths):
        logger.info(f"Loading full model file {i+1}/{len(data_paths)}: {data_path}")
        df = pd.read_csv(data_path)

        # Add experiment identifier
        if experiment_names and i < len(experiment_names):
            experiment_name = experiment_names[i]
        else:
            experiment_name = data_path.stem

        df["experiment"] = experiment_name
        df["model_type"] = "full"  # Mark as full model
        combined_data.append(df)
        logger.info(
            f"Loaded {len(df)} rows for full model experiment '{experiment_name}'"
        )

    # Load feedforward-only data if provided
    if data_ffonly_paths:
        logger.info(
            f"Loading data from {len(data_ffonly_paths)} feedforward-only files..."
        )

        for i, data_path in enumerate(data_ffonly_paths):
            logger.info(
                f"Loading feedforward-only file {i+1}/{len(data_ffonly_paths)}: {data_path}"
            )
            df = pd.read_csv(data_path)

            # Add experiment identifier (should match the corresponding full model)
            if experiment_names and i < len(experiment_names):
                experiment_name = experiment_names[i]
            else:
                experiment_name = data_path.stem

            df["experiment"] = experiment_name
            df["model_type"] = "ffonly"  # Mark as feedforward-only model
            combined_data.append(df)
            logger.info(
                f"Loaded {len(df)} rows for feedforward-only experiment '{experiment_name}'"
            )

    # Combine all data
    df = pd.concat(combined_data, ignore_index=True)
    logger.info(f"Combined data: {len(df)} rows, {len(df.columns)} columns")

    # Get dimension keys
    subplot_key = _get_dimension_key(
        dimension=subplot_var, category_key=category_key, parameter_key=parameter_key
    )
    row_key = _get_dimension_key(
        dimension=row_var, category_key=category_key, parameter_key=parameter_key
    )
    hue_key = _get_dimension_key(
        dimension=hue_var, category_key=category_key, parameter_key=parameter_key
    )
    logger.info(
        f"Keys: subplot_key='{subplot_key}', row_key='{row_key}', hue_key='{hue_key}'"
    )

    # Extract dimension values
    subplot_values_all = _extract_dimension_values(
        df=df, dimension=subplot_var, dimension_key=subplot_key, config=config
    )
    row_values = _extract_dimension_values(
        df=df, dimension=row_var, dimension_key=row_key, config=config
    )
    hue_values = _extract_dimension_values(
        df=df, dimension=hue_var, dimension_key=hue_key, config=config
    )

    # Filter subplot values for display if filter is provided
    if subplot_filter:
        subplot_values_display = [v for v in subplot_values_all if v in subplot_filter]
        logger.info(f"Applied subplot filter: {subplot_filter}")
    else:
        subplot_values_display = subplot_values_all

    if not subplot_values_display or not row_values or not hue_values:
        raise ValueError("No valid dimension values found")

    logger.info(
        f"Display subplot values ({len(subplot_values_display)}): {subplot_values_display}"
    )
    logger.info(
        f"All subplot values ({len(subplot_values_all)}): {subplot_values_all}"
    )
    logger.info(f"Row values ({len(row_values)}): {row_values}")
    logger.info(f"Hue values ({len(hue_values)}): {hue_values}")

    # Get colors for hue dimension
    colors = _get_colors_for_dimension(
        values=hue_values, dimension_key=hue_key, config=config
    )

    # Calculate figure size and layout
    layout = {
        **PERFORMANCE_LAYOUT,
        **{k: v for k, v in kwargs.items() if k in PERFORMANCE_LAYOUT},
    }

    n_subplots_display = len(subplot_values_display)
    n_rows = len(row_values)

    # Calculate total figure width using absolute spacing
    performance_section_width = (
        n_subplots_display * layout["subplot_width"]
        + (n_subplots_display - 1) * layout["subplot_spacing_x"]
    )

    fig_width = (
        layout["left_margin"]
        + performance_section_width
        + layout["peak_panel_spacing"]
        + layout["peak_panel_width"]
        + layout["peak_time_panel_spacing"]
        + layout["peak_time_panel_width"]
        + layout["right_margin"]
    )

    fig_height = (
        layout["top_margin"]
        + n_rows * layout["subplot_height"]
        + (n_rows - 1) * layout["subplot_spacing_y"]
        + layout["bottom_margin"]
    )

    logger.info(
        f'Creating figure: {fig_width:.2f}" x {fig_height:.2f}" ({n_subplots_display} performance + 1 peak + 1 peak time panel × {n_rows})'
    )

    # Create figure and calculate positions manually for better control
    fig = plt.figure(figsize=(fig_width, fig_height))

    # Calculate absolute positions for each panel type
    def calculate_panel_positions():
        """Calculate absolute positions for all panels."""
        positions = {"performance": [], "peak_height": [], "peak_time": []}

        # Performance panels positions
        perf_left_start = layout["left_margin"]
        for row_idx in range(n_rows):
            row_positions = []
            row_bottom = layout["bottom_margin"] + (n_rows - 1 - row_idx) * (
                layout["subplot_height"] + layout["subplot_spacing_y"]
            )

            for col_idx in range(n_subplots_display):
                left = perf_left_start + col_idx * (
                    layout["subplot_width"] + layout["subplot_spacing_x"]
                )
                row_positions.append(
                    [
                        left / fig_width,  # Convert to relative coordinates
                        row_bottom / fig_height,
                        layout["subplot_width"] / fig_width,
                        layout["subplot_height"] / fig_height,
                    ]
                )
            positions["performance"].append(row_positions)

        # Peak height panels positions
        peak_left = (
            perf_left_start + performance_section_width + layout["peak_panel_spacing"]
        )

        for row_idx in range(n_rows):
            row_bottom = layout["bottom_margin"] + (n_rows - 1 - row_idx) * (
                layout["subplot_height"] + layout["subplot_spacing_y"]
            )
            positions["peak_height"].append(
                [
                    peak_left / fig_width,
                    row_bottom / fig_height,
                    layout["peak_panel_width"] / fig_width,
                    layout["subplot_height"] / fig_height,
                ]
            )

        # Peak time panels positions
        peak_time_left = (
            peak_left + layout["peak_panel_width"] + layout["peak_time_panel_spacing"]
        )

        for row_idx in range(n_rows):
            row_bottom = layout["bottom_margin"] + (n_rows - 1 - row_idx) * (
                layout["subplot_height"] + layout["subplot_spacing_y"]
            )
            positions["peak_time"].append(
                [
                    peak_time_left / fig_width,
                    row_bottom / fig_height,
                    layout["peak_time_panel_width"] / fig_width,
                    layout["subplot_height"] / fig_height,
                ]
            )

        return positions

    # Get all panel positions
    panel_positions = calculate_panel_positions()

    # Create axes using calculated positions
    perf_axes = []
    peak_axes = []
    peak_time_axes = []

    for row_idx in range(n_rows):
        # Performance axes for this row
        perf_row = []
        for col_idx in range(n_subplots_display):
            pos = panel_positions["performance"][row_idx][col_idx]
            ax = fig.add_axes(pos)
            perf_row.append(ax)
        perf_axes.append(perf_row)

        # Peak height axis for this row
        pos = panel_positions["peak_height"][row_idx]
        peak_ax = fig.add_axes(pos)
        peak_axes.append(peak_ax)

        # Peak time axis for this row
        pos = panel_positions["peak_time"][row_idx]
        peak_time_ax = fig.add_axes(pos)
        peak_time_axes.append(peak_time_ax)

    # Plot performance panels (filtered subplot values)
    for row_idx, row_value in enumerate(row_values):
        for col_idx, subplot_value in enumerate(subplot_values_display):
            ax = perf_axes[row_idx][col_idx]

            logger.debug(
                f"Plotting performance [{row_idx}, {col_idx}]: {row_key}={row_value}, {subplot_key}={subplot_value}"
            )

            cell_data = _filter_data_for_cell(
                df=df,
                row_key=row_key,
                row_value=row_value,
                subplot_key=subplot_key,
                subplot_value=subplot_value,
            )

            logger.debug(f"  Cell data: {len(cell_data)} rows")

            if len(cell_data) == 0:
                logger.warning(f"No data for cell [{row_idx}, {col_idx}]")
                ax.text(
                    0.5,
                    0.5,
                    "No Data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                continue

            # Plot accuracy panel with hue coloring (now includes both full and ffonly data)
            _plot_accuracy_panel_with_ffonly(
                ax=ax,
                data=cell_data,
                hue_var=hue_var,
                hue_key=hue_key,
                hue_values=hue_values,
                colors=colors,
                dt=dt,
                show_ylabel=(col_idx == 0),
                show_legend=(
                    row_idx == 0 and col_idx == 0
                ),  # Legend only in first subplot
                **kwargs,
            )
            if col_idx == 0:
                sns.despine(ax=ax, left=False)

            # Add subplot title (only on top row)
            if row_idx == 0:
                title_text = get_display_name(subplot_key, config)
                title_text += " = " + _format_legend_label(
                    subplot_key, subplot_value, config, dt
                )
                ax.set_title(title_text, fontsize=FORMATTING["fontsize_axis"], pad=10)

        # Get experiment name from the row value based on row_var
        if row_var == "experiment":
            experiment_name = row_value
        else:
            # If row is not experiment, try to get experiment from first data file
            experiment_name = experiment_names[0] if experiment_names else "Experiment"

        row_label_text = get_display_name(f"{experiment_name}_experiment", config)
        if not row_label_text or row_label_text == f"{experiment_name}_experiment":
            row_label_text = experiment_name.replace("_", " ").title()

        perf_axes[row_idx][0].text(
            -0.5,
            0.5,
            row_label_text,
            rotation=90,
            ha="center",
            va="center",
            fontsize=FORMATTING["fontsize_axis"],
            fontweight="bold",
            transform=perf_axes[row_idx][0].transAxes,
        )

    # Plot peak height panels (using all subplot values for analysis)
    for row_idx, row_value in enumerate(row_values):
        ax = peak_axes[row_idx]

        logger.debug(f"Plotting peak height [{row_idx}]: {row_key}={row_value}")

        _plot_peak_height_panel(
            ax=ax,
            data=df,
            row_key=row_key,
            row_value=row_value,
            subplot_key=subplot_key,
            subplot_values=subplot_values_all,  # Use all values for comprehensive analysis
            hue_key=hue_key,
            hue_values=hue_values,
            colors=colors,
            config=config,
            dt=dt,
            show_ylabel=(row_idx == 0),  # Only show ylabel on first row
            show_legend=(row_idx == 0),
        )

    # Plot peak time panels (using all subplot values for analysis)
    for row_idx, row_value in enumerate(row_values):
        ax = peak_time_axes[row_idx]

        logger.debug(f"Plotting peak time [{row_idx}]: {row_key}={row_value}")

        _plot_peak_time_panel(
            ax=ax,
            data=df,
            row_key=row_key,
            row_value=row_value,
            subplot_key=subplot_key,
            subplot_values=subplot_values_all,  # Use all values for comprehensive analysis
            hue_key=hue_key,
            hue_values=hue_values,
            colors=colors,
            config=config,
            dt=dt,
            show_ylabel=(row_idx == 0),  # Only show ylabel on first row
            show_legend=(row_idx == 0),
        )

    # Synchronize y-limits and y-ticks across performance panels (Panel A) and peak height panels (Panel B)
    all_perf_axes = [ax for row in perf_axes for ax in row]
    all_peak_axes = peak_axes.copy()

    # Combine performance and peak height axes to determine global y-limits
    all_performance_related_axes = all_perf_axes + all_peak_axes

    # Get global y-limits from all performance-related panels
    try:
        global_ymin = min(axis.get_ylim()[0] for axis in all_performance_related_axes)
        global_ymax = max(axis.get_ylim()[1] for axis in all_performance_related_axes)
    except ValueError:
        # Fallback if no valid limits found
        global_ymin, global_ymax = 0.0, 1.0

    # Ensure reasonable bounds for performance data
    global_ymin = max(global_ymin, 0.0)  # Performance shouldn't go below 0
    global_ymax = min(global_ymax, 1.0)  # Performance shouldn't go above 1

    global_y_range = global_ymax - global_ymin

    # Determine appropriate tick spacing
    if global_y_range <= 0.2:
        tick_step = 0.05
    elif global_y_range <= 0.5:
        tick_step = 0.1
    else:
        tick_step = 0.2

    # Calculate y-ticks
    yticks = []
    tick = 0.0
    while tick <= 1.0:
        if tick >= global_ymin and tick <= global_ymax:
            yticks.append(tick)
        tick += tick_step

    # Apply consistent y-limits and y-ticks to performance panels and peak height panels
    for ax in all_performance_related_axes:
        ax.set_ylim(global_ymin, global_ymax)
        ax.set_yticks(yticks)

    # Hide y-tick labels for all performance panels except the leftmost column
    for row_idx in range(n_rows):
        for col_idx in range(n_subplots_display):
            ax = perf_axes[row_idx][col_idx]
            if col_idx > 0:  # Hide labels for all columns except the first
                ax.set_yticklabels([])

    # Add label indicators
    for row_idx, row_value in enumerate(row_values):
        for col_idx, subplot_value in enumerate(subplot_values_display):
            try:
                calculate_label_indicator(perf_axes[row_idx][col_idx])
            except Exception as e:
                logger.debug(f"Could not add label indicator: {e}")

    # Add hue legend in second performance subplot (if exists)
    _add_hue_legend(
        fig=fig,
        perf_axes=perf_axes,
        hue_key=hue_key,
        hue_values=hue_values,
        colors=colors,
        config=config,
        dt=dt,
    )

    # Add panel letters after all plotting is complete but before saving
    _add_panel_letters(
        fig=fig,
        perf_axes=perf_axes,
        peak_axes=peak_axes,
        peak_time_axes=peak_time_axes,
        layout=layout,
    )

    # Save
    logger.info(f"Saving figure to: {output}")
    save_plot(output)
    logger.info("Performance grid plotting complete")

    return fig


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Plot performance traces in grid layout"
    )
    parser.add_argument(
        "--data",
        type=Path,
        nargs="+",
        required=True,
        help="Paths to test_data.csv files",
    )
    parser.add_argument(
        "--data-ffonly",
        type=Path,
        nargs="*",
        help="Paths to feedforward-only test_data.csv files",
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Output figure path"
    )
    parser.add_argument(
        "--subplot",
        type=str,
        required=True,
        choices=["category", "parameter", "experiment"],
        help="Variable for horizontal subplots",
    )
    parser.add_argument(
        "--row",
        type=str,
        required=True,
        choices=["category", "parameter", "experiment"],
        help="Variable for vertical rows",
    )
    parser.add_argument(
        "--hue",
        type=str,
        required=True,
        choices=["category", "parameter", "experiment"],
        help="Variable for color coding",
    )
    parser.add_argument("--category-key", type=str, help="Category column name")
    parser.add_argument("--parameter-key", type=str, help="Parameter column name")
    parser.add_argument(
        "--experiment-names", type=str, nargs="*", help="Experiment names"
    )
    parser.add_argument("--dt", type=float, default=2.0, help="Time resolution (ms)")
    parser.add_argument("--palette", type=str, help="JSON color palette")
    parser.add_argument("--naming", type=str, help="JSON naming dict")
    parser.add_argument("--ordering", type=str, help="JSON ordering dict")
    parser.add_argument(
        "--subplot-filter",
        type=str,
        nargs="*",
        help="Filter subplot values to display in performance panels",
    )

    args = parser.parse_args()

    # Load config
    config = load_config_from_args(
        palette_str=args.palette,
        naming_str=args.naming,
        ordering_str=args.ordering,
    )

    # Plot
    plot_performance_grid(
        data_paths=args.data,
        output=args.output,
        subplot_var=args.subplot,
        row_var=args.row,
        hue_var=args.hue,
        category_key=args.category_key,
        parameter_key=args.parameter_key,
        experiment_names=args.experiment_names,
        data_ffonly_paths=args.data_ffonly,
        dt=args.dt,
        config=config,
        subplot_filter=args.subplot_filter,
    )


if __name__ == "__main__":
    main()
