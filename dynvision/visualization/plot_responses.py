"""Plot temporal ridge responses with flexible dimension mapping.

This module provides a generalized plotting function for visualizing temporal
dynamics of neural network layer responses with three flexible dimensions:
- Vertical subplots (ridge plots)
- Color hues
- Columns

Each dimension can represent: layers, category, or parameter.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from dynvision.utils.visualization_utils import (
    calculate_label_indicator,
    get_color,
    get_display_name,
    get_ordering,
    load_config_from_args,
    order_layers,
    save_plot,
)

# Configure logging - simplified to prevent duplicate output
logger = logging.getLogger(__name__)
logger.setLevel("INFO")


# Global layout configuration
LAYOUT = {
    # Figure dimensions
    "figure_height": 14,  # Total figure height in inches
    "column_width": 7,  # Width of each column in inches
    "column_spacing": 0.25,  # Spacing between columns in inches
    # Vertical layout in % of figure size (bot = bottom position, pad = whitespace below element)
    "title_bot": 0.96,
    "title_pad": 0.02,  # Space below title
    "accuracy_pad": 0.02,  # Space below accuracy panel
    "accuracy_height": 0.13,  # Fixed height for accuracy panel
    "legend_pad": 0.02,  # Space below legend
    "legend_height": 0.1,  # Fixed height for legend
    "ridge_height": 0.65,  # Total height for all ridge plots
    "ridge_overlap": 0.25,  # Overlap fraction between ridges
    # Margins
    "left_margin": 0.08,
    "right_margin": 0.02,
    "legend_margin": 0.02,
}

# Global formatting configuration
FORMATTING = {
    "fontsize_title": 18,
    "fontsize_axis": 16,
    "fontsize_tick": 14,
    "fontsize_legend": 16,
    "fontsize_label": 14,
    "linewidth_main": 2.5,
    "linewidth_indicator": 3,
    "alpha_line": 0.8,
    "alpha_indicator": 0.6,
    "layer_circle_colors": {
        "V1": "#ff69b4ff",
        "V2": "#dda0ddff",
        "V4": "#da70d6ff",
        "IT": "#ba55d3ff",
    },
    "min_global_ymin": -0.005,
    "max_global_ymax": 4,
}


def _format_legend_label(key: str, value: str, config: Dict, dt: float = 1.0) -> str:
    """Format legend label based on key type and config.

    Args:
        key: The parameter/category key
        value: The value to format
        config: Configuration dict with naming mappings
        dt: Temporal resolution in ms per timestep (for converting timestep values)

    Returns:
        Formatted label string
    """
    # First check if there's a direct naming translation for this value
    if config and "naming" in config:
        # Check both the value directly and the standardized value
        for check_val in [value, _standardize_category_value(value)]:
            translated_name = get_display_name(check_val, config)
            if translated_name and translated_name != check_val:
                return translated_name

    # Timestep parameters that should be converted to ms
    timestep_params = ["tsteps", "idle"]
    # Numerical parameters that are already in ms
    ms_params = ["tau", "trc", "tsk", "lossrt"]

    if key.lower() in timestep_params:
        try:
            timestep_value = int(float(value))
            ms_value = timestep_value * dt
            return f"{ms_value:.0f} ms"
        except (ValueError, TypeError):
            return f"{value} ms"
    elif key.lower() in ms_params:
        try:
            numeric_value = int(float(value))
            return f"{numeric_value} ms"
        except (ValueError, TypeError):
            return f"{value} ms"

    # For other parameters, check if boolean
    if str(value).lower() in ["true", "false"]:
        return str(value).capitalize()

    # Default: capitalize first letter
    return str(value).capitalize()


def _validate_dimensions(
    subplot_var: str, hue_var: str, column_var: Optional[str]
) -> None:
    """Validate that dimension variables are unique."""
    used = [subplot_var, hue_var]
    if column_var:
        used.append(column_var)

    if len(used) != len(set(used)):
        raise ValueError(
            f"Dimension variables must be unique. Got: subplot={subplot_var}, "
            f"hue={hue_var}, column={column_var}"
        )


def _get_dimension_key(
    dimension: str, category_key: Optional[str], parameter_key: Optional[str]
) -> str:
    """Get the dataframe column name for a dimension."""
    if dimension == "category":
        if not category_key:
            raise ValueError("category_key required when using 'category' dimension")
        return category_key
    elif dimension == "parameter":
        if not parameter_key:
            raise ValueError("parameter_key required when using 'parameter' dimension")
        return parameter_key
    elif dimension == "layers":
        return "layers"  # Special handling
    elif dimension == "experiment":
        return "experiment"  # Special handling for experiment dimension
    else:
        raise ValueError(f"Unknown dimension: {dimension}")


def _standardize_category_value(value: str) -> str:
    """Standardize category value formatting consistently across all processing."""
    value_str = str(value).strip()
    # Handle boolean-like values consistently
    if value_str.lower() in ["true", "false"]:
        return value_str.lower()
    return value_str


def _extract_dimension_values(
    df: pd.DataFrame,
    dimension: str,
    dimension_key: str,
    config: Dict,
) -> List[str]:
    """Extract and order unique values for a dimension."""
    if dimension == "layers":
        # Extract layer names from column headers
        layer_cols = [col for col in df.columns if col.endswith("_response_avg")]
        layers = [col.replace("_response_avg", "") for col in layer_cols]

        if not layers:
            logger.warning(
                f"No layer response columns found in data (looking for *_response_avg)"
            )
            return []

        ordered = order_layers(layers, config)
        logger.info(f"Found {len(ordered)} layers: {ordered}")
        return ordered
    else:
        # Check if column exists
        if dimension_key not in df.columns:
            logger.warning(
                f"Column '{dimension_key}' not found in data. Available columns: {list(df.columns)}"
            )
            return []

        # Get unique values from dataframe column and standardize them
        raw_values = df[dimension_key].unique()
        values = [_standardize_category_value(v) for v in sorted(raw_values)]
        # Remove duplicates while preserving order
        seen = set()
        values = [v for v in values if not (v in seen or seen.add(v))]

        if not values:
            logger.warning(f"No unique values found for dimension '{dimension_key}'")
            return []

        # Apply config ordering if available
        ordering = get_ordering(dimension_key, config)
        if ordering:
            # Standardize ordering values too
            standardized_ordering = [_standardize_category_value(v) for v in ordering]
            # Filter to only include values present in data
            ordered = [v for v in standardized_ordering if v in values]
            # Add any missing values
            for v in values:
                if v not in ordered:
                    ordered.append(v)
            logger.info(
                f"Found {len(ordered)} values for '{dimension_key}': {ordered} (config ordering applied)"
            )
            return ordered

        logger.info(f"Found {len(values)} values for '{dimension_key}': {values}")
        return values


def _get_colors_for_dimension(
    values: List[str], dimension_key: str, config: Dict
) -> Dict[str, str]:
    """Get color mapping for dimension values."""
    colors = {}
    for val in values:
        color = get_color(val, config)
        if color:
            colors[val] = color

    # Fill missing colors with viridis
    if len(colors) < len(values):
        viridis_colors = plt.cm.viridis(np.linspace(0, 1, len(values)))
        for i, val in enumerate(values):
            if val not in colors:
                colors[val] = plt.matplotlib.colors.to_hex(viridis_colors[i])

    return colors


def _filter_data_for_column(
    df: pd.DataFrame, column_var: Optional[str], column_value: Optional[str]
) -> pd.DataFrame:
    """Filter dataframe for a specific column value."""
    if column_var and column_value:
        # Standardize both the column values and the filter value for comparison
        df_copy = df.copy()
        if column_var in df_copy.columns:
            df_copy[column_var] = df_copy[column_var].apply(
                _standardize_category_value
            )
        standardized_column_value = _standardize_category_value(str(column_value))
        return df_copy[df_copy[column_var] == standardized_column_value].copy()
    return df.copy()


def _plot_accuracy_panel(
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
    """Plot accuracy and confidence over time.

    Args:
        ax: Matplotlib axes
        data: DataFrame with accuracy and confidence data
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

    if not has_accuracy and not has_confidence:
        logger.warning("Neither 'accuracy' nor 'confidence_avg' columns found in data")
        return

    # Plot accuracy (solid lines)
    if has_accuracy:
        logger.debug(
            f"Plotting accuracy with hue='{hue_key if hue_var != 'layers' else None}'"
        )
        sns.lineplot(
            data=data_plot,
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
            errorbar="ci",
            err_style="bars",
            # err_kws={"alpha": 0.2, "edgecolor": "none"}
        )
    else:
        logger.warning("'accuracy' column not found, skipping accuracy plot")

    # Plot confidence (dotted lines)
    if has_confidence:
        logger.debug(
            f"Plotting confidence with hue='{hue_key if hue_var != 'layers' else None}'"
        )
        sns.lineplot(
            data=data_plot,
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
            errorbar="ci",
            err_style="bars",
            # err_kws={"alpha": 0.2, "edgecolor": "none"}
        )
    else:
        logger.warning("'confidence_avg' column not found, skipping confidence plot")

    # Styling
    ax.set_ylim(-0.01, 1.01)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    if show_ylabel:
        ax.set_ylabel("Performance", fontsize=fmt["fontsize_axis"], fontweight="bold")
    else:
        ax.set_ylabel("")
        ax.set_yticklabels([])

    ax.set_xlabel("Time (ms)", fontsize=fmt["fontsize_axis"])
    ax.tick_params(labelsize=fmt["fontsize_tick"])

    # Add legend for line styles on first panel
    if show_legend and has_confidence:
        legend_elements = [
            plt.Line2D(
                [0],
                [0],
                color="black",
                linewidth=fmt["linewidth_main"],
                linestyle="-",
                label="Accuracy",
                alpha=fmt["alpha_line"],
            ),
            plt.Line2D(
                [0],
                [0],
                color="black",
                linewidth=fmt["linewidth_main"],
                linestyle=":",
                label="Confidence",
                alpha=fmt["alpha_line"],
            ),
        ]
        ax.legend(
            handles=legend_elements,
            loc="best",
            frameon=False,
            fontsize=fmt["fontsize_legend"] - 2,
        )

    # Add label indicator
    try:
        label_indicator_df = calculate_label_indicator(
            data,
            hue_key,
            ax.get_ylim(),
            0.1,
        )
        indicator_time = label_indicator_df["times_index"] * dt
        ax.plot(
            indicator_time,
            label_indicator_df["label_indicator"],
            color="dimgray",
            linewidth=fmt["linewidth_indicator"],
            drawstyle="steps-mid",
            alpha=fmt["alpha_indicator"],
        )
    except Exception as e:
        logger.debug(f"Could not calculate label indicator for subplot {hue_key}: {e}")

    ax.grid(True, alpha=0.3)
    sns.despine(ax=ax, left=True, bottom=True)


def _add_layer_circle(x, y, layer_name, ax=None, config=None, **kwargs):
    if ax is None:
        ax = plt.gca()

    pad = 0.5 if layer_name == "IT" else 0.4

    colors = _get_colors_for_dimension(
        values=[layer_name], dimension_key="layers", config=config
    )
    fmt = {**FORMATTING, **{k: v for k, v in kwargs.items() if k in FORMATTING}}

    ax.text(
        x,
        y,
        layer_name,
        horizontalalignment="center",
        verticalalignment="center",
        transform=ax.transAxes,
        bbox=dict(
            boxstyle=f"circle,pad={pad}",
            facecolor=colors.get(layer_name, "#808080ff"),
            edgecolor="#353535ff",
            linewidth=2,
            alpha=0.8,
        ),
        fontsize=fmt["fontsize_label"],
        fontweight="bold",
    )


def _plot_response_ridges(
    fig: plt.Figure,
    column_left: float,
    column_width: float,
    data: pd.DataFrame,
    subplot_var: str,
    subplot_key: str,
    subplot_values: List[str],
    hue_var: str,
    hue_key: str,
    hue_values: List[str],
    colors: Dict[str, str],
    dt: float,
    show_ylabel: bool,
    config: Dict,
    ridge_top: Optional[float] = None,
    **kwargs,
) -> List[plt.Axes]:
    """Plot ridge plots for responses.

    Args:
        fig: Matplotlib figure
        column_left: Left position of column
        column_width: Width of column
        data: DataFrame with response data
        subplot_var: Variable for subplots
        subplot_key: Column name for subplot variable
        subplot_values: Ordered list of subplot values
        hue_var: Variable for color coding
        hue_key: Column name for hue variable
        hue_values: Ordered list of hue values
        colors: Color mapping for hue values
        dt: Temporal resolution in ms
        show_ylabel: Whether to show y-axis label
        config: Configuration dict
        **kwargs: Override FORMATTING and LAYOUT defaults

    Returns:
        List of created axes
    """
    layout = {**LAYOUT, **{k: v for k, v in kwargs.items() if k in LAYOUT}}
    fmt = {**FORMATTING, **{k: v for k, v in kwargs.items() if k in FORMATTING}}

    # Check if focus_layer is specified in kwargs (for Panel B case)
    focus_layer = kwargs.get("focus_layer", None)

    n_subplots = len(subplot_values)
    if n_subplots == 0:
        logger.warning("No subplot values provided, cannot create ridge plots")
        return []

    logger.info(f"Creating {n_subplots} ridge subplots for dimension '{subplot_var}'")

    spacing = layout["ridge_height"] / n_subplots * (1 - layout["ridge_overlap"])
    plot_height = layout["ridge_height"] / n_subplots * 1.4

    # Prepare data
    data_plot = data.copy()
    data_plot["time_ms"] = data_plot["times_index"] * dt

    # Ensure categorical columns are standardized to match dimension values
    if hue_key in data_plot.columns and hue_key != "layers":
        data_plot[hue_key] = data_plot[hue_key].apply(_standardize_category_value)
    if subplot_key in data_plot.columns and subplot_key != "layers":
        data_plot[subplot_key] = data_plot[subplot_key].apply(
            _standardize_category_value
        )

    axes = []

    # Calculate ridge top position from other elements
    if ridge_top is None:
        ridge_top = (
            layout["title_bot"]
            - layout["title_pad"]
            - layout["accuracy_height"]
            - layout["accuracy_pad"]
            - layout["legend_height"]
            - layout["legend_pad"]
        )

    global_ymin, global_ymax = fmt["max_global_ymax"], fmt["min_global_ymin"]
    global_xmin, global_xmax = 0, 0

    for i, subplot_value in enumerate(subplot_values):
        # Position subplot
        top_pos = ridge_top - i * spacing
        bottom_pos = top_pos - plot_height
        ax = fig.add_axes([column_left, bottom_pos, column_width, plot_height])

        ax.patch.set_alpha(0)
        axes.append(ax)

        # Reference line
        ax.axhline(0, color="gray", linestyle=":", alpha=0.7, linewidth=1)

        # Filter data for this subplot
        if subplot_var == "layers":
            response_col = f"{subplot_value}_response_avg"
            if response_col not in data_plot.columns:
                logger.warning(
                    f"Response column '{response_col}' not found for layer '{subplot_value}', skipping"
                )
                continue
            plot_data = data_plot.copy()
            n_points = len(plot_data)
        else:
            plot_data = data_plot[data_plot[subplot_key] == subplot_value].copy()
            n_points = len(plot_data)
            if n_points == 0:
                logger.warning(
                    f"No data found for {subplot_key}={subplot_value}, skipping subplot"
                )
                continue

        logger.debug(
            f"Subplot {i+1}/{n_subplots} ({subplot_value}): {n_points} datapoints"
        )

        # Plot lines
        if subplot_var == "layers":
            # Plot layer response colored by hue
            sns.lineplot(
                data=plot_data,
                x="time_ms",
                y=response_col,
                hue=hue_key if hue_var != "layers" else None,
                hue_order=hue_values if hue_var != "layers" else None,
                palette=colors if hue_var != "layers" else None,
                ax=ax,
                legend=False,
                linewidth=fmt["linewidth_main"],
                marker=".",
                alpha=fmt["alpha_line"],
                errorbar="se",
                err_style="band",
                err_kws={"alpha": 0.2, "edgecolor": "none"},
            )

            _add_layer_circle(
                x=0.95,
                y=0.25,
                layer_name=subplot_value.upper(),
                ax=ax,
                config=config,
                **fmt,
            )

        else:
            # Handle parameter subplots - two cases:
            # 1. hue_var == "layers": plot different layers as different colored lines
            # 2. focus_layer specified: plot single layer response colored by category

            plotted_any = False

            if focus_layer and hue_var == "category":
                # Panel B case: plot focus layer response colored by category
                response_col = f"{focus_layer}_response_avg"
                if response_col in plot_data.columns:
                    sns.lineplot(
                        data=plot_data,
                        x="time_ms",
                        y=response_col,
                        hue=hue_key,
                        hue_order=hue_values,
                        palette=colors,
                        ax=ax,
                        legend=False,
                        linewidth=fmt["linewidth_main"],
                        marker=".",
                        alpha=fmt["alpha_line"],
                        errorbar="se",
                        err_style="band",
                        err_kws={"alpha": 0.2, "edgecolor": "none"},
                    )
                    plotted_any = True
                else:
                    logger.warning(
                        f"Focus layer response column '{response_col}' not found for parameter '{subplot_value}'"
                    )
            elif hue_var == "layers":
                # Plot different layers as hues
                for hue_val in hue_values:
                    response_col = f"{hue_val}_response_avg"
                    if response_col in plot_data.columns:
                        hue_data = plot_data.copy()
                        sns.lineplot(
                            data=hue_data,
                            x="time_ms",
                            y=response_col,
                            color=colors.get(hue_val, "#808080"),
                            ax=ax,
                            legend=False,
                            linewidth=fmt["linewidth_main"],
                            marker=".",
                            alpha=fmt["alpha_line"],
                        )
                        plotted_any = True
                    else:
                        logger.warning(
                            f"Response column '{response_col}' not found for hue '{hue_val}'"
                        )

            if not plotted_any:
                logger.warning(
                    f"No valid response columns found for any hue value in subplot {subplot_value}"
                )

            # Add parameter label
            display_name = get_display_name(subplot_key, config)
            label_text = f"{display_name}={subplot_value}"
            ax.text(
                0.95,
                0.25,
                label_text,
                horizontalalignment="right",
                verticalalignment="center",
                transform=ax.transAxes,
                fontsize=fmt["fontsize_label"],
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor="white",
                    edgecolor="gray",
                    alpha=0.8,
                ),
            )

        # Adjust limits to common scale across all subplots
        ymin, ymax = ax.get_ylim()
        global_ymin = max(min(global_ymin, ymin), fmt["min_global_ymin"])
        global_ymax = min(max(global_ymax, ymax), fmt["max_global_ymax"])
        xmin, xmax = ax.get_xlim()
        global_xmin = min(global_xmin, xmin)
        global_xmax = max(global_xmax, xmax)

        # Add label indicator
        try:
            label_indicator_df = calculate_label_indicator(
                plot_data,
                subplot_key if subplot_var != "layers" else hue_key,
                (0, global_ymax),
                0.1,
            )
            indicator_time = label_indicator_df["times_index"] * dt
            ax.plot(
                indicator_time,
                label_indicator_df["label_indicator"],
                color="dimgray",
                linewidth=fmt["linewidth_indicator"],
                drawstyle="steps-mid",
                alpha=fmt["alpha_indicator"],
            )
        except Exception as e:
            logger.debug(
                f"Could not calculate label indicator for subplot {subplot_value}: {e}"
            )

        # Y-label on leftmost panel, middle subplot
        if show_ylabel and i == len(subplot_values) // 2:
            ax.set_ylabel(
                "Avg Layer Response",
                fontsize=fmt["fontsize_axis"],
                fontweight="bold",
                labelpad=6,
            )
        else:
            ax.set_ylabel("")

        # X-axis only on bottom subplot
        if i < len(subplot_values) - 1:
            ax.set_xticklabels([])
            ax.set_xlabel("")
        else:
            ax.set_xlabel("Time (ms)", fontsize=fmt["fontsize_axis"])
            ax.tick_params(labelsize=fmt["fontsize_tick"])

        ax.set_yticks([0, 1])
        ax.set_yticklabels(["0", "1"], fontsize=fmt["fontsize_tick"])
        sns.despine(ax=ax, left=True, bottom=True)

    for ax in axes:
        ax.set_ylim(global_ymin, global_ymax)
        ax.set_xlim(global_xmin, global_xmax)

    logger.info(f"Created {len(axes)} ridge plot axes")
    return axes


def _add_horizontal_legend(
    fig: plt.Figure,
    hue_var: str,
    hue_key: str,
    hue_values: List[str],
    colors: Dict[str, str],
    config: Dict,
    legend_bot: float = None,
    dt: float = 2.0,
    **kwargs,
) -> None:
    """Add horizontal legend for hue dimension.

    Args:
        fig: Matplotlib figure
        column_left: Left position of column
        column_width: Width of column
        hue_var: Variable for color coding
        hue_key: Column name for hue variable
        hue_values: Ordered list of hue values (already ordered by config if available)
        colors: Color mapping for hue values
        config: Configuration dict with naming mappings
        dt: Temporal resolution in ms per timestep
        **kwargs: Override FORMATTING and LAYOUT defaults
    """
    layout = {**LAYOUT, **{k: v for k, v in kwargs.items() if k in LAYOUT}}
    fmt = {**FORMATTING, **{k: v for k, v in kwargs.items() if k in FORMATTING}}

    left = layout["left_margin"] + layout["legend_margin"]
    width = layout["column_width"] - 2 * layout["legend_margin"]

    # Calculate legend position from other elements
    if legend_bot is None:
        legend_bot = (
            layout["title_bot"]
            - layout["title_pad"]
            - layout["accuracy_height"]
            - layout["accuracy_pad"]
            - layout["legend_height"]
        )

    legend_ax = fig.add_axes([left, legend_bot, width, layout["legend_height"]])
    legend_ax.set_xlim(0, 1)
    legend_ax.set_ylim(0, 1)
    legend_ax.axis("off")
    legend_ax.patch.set_alpha(0)

    # Create legend elements using the ordered hue_values
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
                    marker=".",
                    markersize=8,
                    alpha=fmt["alpha_line"],
                )
            )

            if hue_var == "layers":
                label = val.upper()
            else:
                label = _format_legend_label(hue_key, val, config, dt)
            legend_labels.append(label)

    if legend_elements:
        # Get symbol for title
        symbol = get_display_name(hue_key, config)

        n_cols = min(len(legend_elements), 6)
        logger.debug(
            f"Adding legend with {len(legend_elements)} elements in {n_cols} columns"
        )
        legend = legend_ax.legend(
            legend_elements,
            legend_labels,
            loc="lower center",
            ncol=n_cols,
            frameon=False,
            fontsize=fmt["fontsize_legend"],
            handlelength=2,
            handletextpad=0.5,
            columnspacing=1.0,
            title=symbol if hue_var != "layers" else None,
            title_fontsize=fmt["fontsize_legend"],
        )
    else:
        logger.warning("No legend elements to display")


def plot_temporal_ridge_responses(
    data: Union[Path, pd.DataFrame],
    output: Path,
    subplot_var: Literal["layers", "category", "parameter"],
    hue_var: Literal["layers", "category", "parameter"],
    column_var: Optional[Literal["parameter"]] = None,
    category_key: Optional[str] = None,
    parameter_key: Optional[str] = None,
    experiment: Optional[str] = None,
    dt: float = 2.0,
    config: Optional[Dict] = None,
    **kwargs,
) -> None:
    """Plot temporal ridge responses with flexible dimension mapping.

    Args:
        data: Path to test_data.csv or DataFrame
        output: Path to save figure
        subplot_var: Variable for vertical subplots (ridge plots)
        hue_var: Variable for color coding
        column_var: Variable for columns (optional, must be 'parameter')
        category_key: Column name for category (e.g., 'rctype')
        parameter_key: Column name for parameter (e.g., 'duration', 'contrast')
        dt: Temporal resolution in ms per timestep
        config: Configuration dict with palette, naming, ordering
        **kwargs: Override LAYOUT and FORMATTING defaults
    """
    logger.info("=" * 60)
    logger.info("Starting temporal ridge response plotting")
    logger.info("=" * 60)

    # Validation
    logger.info(
        f"Dimension mapping: subplot={subplot_var}, hue={hue_var}, column={column_var}"
    )
    _validate_dimensions(
        subplot_var=subplot_var, hue_var=hue_var, column_var=column_var
    )

    if config is None:
        config = {"palette": {}, "naming": {}, "ordering": {}}
        logger.info("No config provided, using empty defaults")

    # Load data
    if isinstance(data, Path):
        logger.info(f"Loading data from: {data}")
        df = pd.read_csv(data)
        logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
        logger.debug(f"Columns: {list(df.columns)}")
    else:
        df = data.copy()
        logger.info(
            f"Using provided DataFrame with {len(df)} rows, {len(df.columns)} columns"
        )

    # Get dimension keys
    logger.info("Determining dimension keys...")
    subplot_key = _get_dimension_key(
        dimension=subplot_var, category_key=category_key, parameter_key=parameter_key
    )
    hue_key = _get_dimension_key(
        dimension=hue_var, category_key=category_key, parameter_key=parameter_key
    )
    column_key = (
        _get_dimension_key(
            dimension=column_var,
            category_key=category_key,
            parameter_key=parameter_key,
        )
        if column_var
        else None
    )
    logger.info(
        f"Keys: subplot_key='{subplot_key}', hue_key='{hue_key}', column_key='{column_key}'"
    )

    # Extract dimension values
    logger.info("Extracting dimension values...")
    subplot_values = _extract_dimension_values(
        df=df, dimension=subplot_var, dimension_key=subplot_key, config=config
    )
    hue_values = _extract_dimension_values(
        df=df, dimension=hue_var, dimension_key=hue_key, config=config
    )
    column_values = (
        _extract_dimension_values(
            df=df, dimension=column_var, dimension_key=column_key, config=config
        )
        if column_var
        else [None]
    )

    # Check if we have data to plot
    if not subplot_values or not hue_values:
        logger.error("Insufficient data for plotting: missing subplot or hue values")
        logger.error(f"Subplot values: {subplot_values}")
        logger.error(f"Hue values: {hue_values}")
        raise ValueError("Cannot create plot with empty dimension values")

    # Get colors for hue dimension
    logger.info("Assigning colors to hue dimension...")
    colors = _get_colors_for_dimension(
        values=hue_values, dimension_key=hue_key, config=config
    )
    logger.debug(f"Color mapping: {colors}")

    # Merge layout defaults with overrides
    layout = {**LAYOUT, **{k: v for k, v in kwargs.items() if k in LAYOUT}}
    fmt = {**FORMATTING, **{k: v for k, v in kwargs.items() if k in FORMATTING}}

    # Calculate figure size - use the actual inches for figure size
    n_columns = len(column_values)
    total_width = (
        n_columns * layout["column_width"] + (n_columns - 1) * layout["column_spacing"]
    )

    logger.info(
        f"Creating figure: {total_width:.2f}\" x {layout['figure_height']}\" with {n_columns} column(s)"
    )

    fig = plt.figure(
        figsize=(
            total_width,
            layout["figure_height"],
        )
    )
    sns.set_context("talk")

    # Plot each column
    all_ridge_axes = []
    for col_idx, column_value in enumerate(column_values):
        logger.info(
            "#" * 10
            + f" Processing column {col_idx+1}/{n_columns}"
            + (f" ({column_key}={column_value}) " if column_value else " ")
            + "#" * 10
        )

        # Calculate column position - now in figure units (0-1)
        relative_column_width = layout["column_width"] / total_width
        relative_column_spacing = layout["column_spacing"] / total_width
        column_left = col_idx * (relative_column_width + relative_column_spacing)

        # Filter data for this column
        column_data = _filter_data_for_column(
            df=df, column_var=column_key, column_value=column_value
        )
        logger.info(f"Column data: {len(column_data)} rows")

        # Add title
        if column_var:
            title_ax = fig.add_axes(
                [
                    column_left,
                    layout["title_bot"],
                    relative_column_width,
                    layout["title_pad"],
                ]
            )
            title_name = (
                get_display_name(key=f"{experiment}_experiment", config=config)
                if experiment
                else ""
            )
            title_symbol = get_display_name(key=column_key, config=config)
            if title_name:
                title_text = f"{title_name} ({title_symbol} = {column_value})"
            else:
                title_text = f"{title_symbol} = {column_value}"
            title_ax.text(
                0.5,
                0.5,
                title_text,
                ha="center",
                va="center",
                fontsize=fmt["fontsize_title"],
                fontweight="bold",
            )
            title_ax.set_xlim(0, 1)
            title_ax.set_ylim(0, 1)
            title_ax.axis("off")
            logger.debug(f"Added title: {title_text}")

        # Calculate accuracy panel position
        accuracy_bot = (
            layout["title_bot"] - layout["title_pad"] - layout["accuracy_height"]
        )

        # Plot accuracy panel
        logger.info("Creating accuracy panel...")
        acc_ax = fig.add_axes(
            [
                column_left,
                accuracy_bot,
                relative_column_width,
                layout["accuracy_height"],
            ]
        )
        _plot_accuracy_panel(
            ax=acc_ax,
            data=column_data,
            hue_var=hue_var,
            hue_key=hue_key,
            hue_values=hue_values,
            colors=colors,
            dt=dt,
            show_ylabel=(col_idx == 0),
            show_legend=(col_idx == 0),
            **kwargs,
        )

        # Plot response ridges
        logger.info("Creating ridge plots...")
        ridge_axes = _plot_response_ridges(
            fig=fig,
            column_left=column_left,
            column_width=relative_column_width,
            data=column_data,
            subplot_var=subplot_var,
            subplot_key=subplot_key,
            subplot_values=subplot_values,
            hue_var=hue_var,
            hue_key=hue_key,
            hue_values=hue_values,
            colors=colors,
            dt=dt,
            show_ylabel=(col_idx == 0),
            config=config,
            **kwargs,
        )
        all_ridge_axes.extend(ridge_axes)

        # Add legend
        logger.info("Adding legend...")
        _add_horizontal_legend(
            fig=fig,
            column_left=column_left,
            column_width=relative_column_width,
            hue_var=hue_var,
            hue_key=hue_key,
            hue_values=hue_values,
            colors=colors,
            config=config,
            dt=dt,
            **kwargs,
        )

        if all_ridge_axes:
            # Align xy-limits across all ridge axes
            global_ymin, global_ymax = fmt["max_global_ymax"], fmt["min_global_ymin"]
            global_xmin, global_xmax = 0, 0

            for ax in all_ridge_axes:
                ymin, ymax = ax.get_ylim()
                global_ymin = min(global_ymin, ymin)
                global_ymax = max(global_ymax, ymax)
                xmin, xmax = ax.get_xlim()
                global_xmin = min(global_xmin, xmin)
                global_xmax = max(global_xmax, xmax)

            global_ymax = min(global_ymax, fmt["max_global_ymax"])
            global_ymin = min(global_ymin, fmt["min_global_ymin"])

            for ax in all_ridge_axes:
                ax.set_ylim(global_ymin, global_ymax)
                ax.set_xlim(global_xmin, global_xmax)

        # Align y-labels
        fig.align_ylabels(all_ridge_axes)
        logger.info(f"Aligned y-labels for {len(all_ridge_axes)} axes")

    # Save
    logger.info(f"Saving figure to: {output}")
    save_plot(output)
    logger.info("=" * 60)
    logger.info("Plotting complete")
    logger.info("=" * 60)

    return fig


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Plot temporal ridge responses with flexible dimensions"
    )
    parser.add_argument(
        "--data", type=Path, required=True, help="Path to test_data.csv"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Output figure path"
    )
    parser.add_argument(
        "--subplot",
        type=str,
        required=True,
        choices=["layers", "category", "parameter"],
        help="Variable for vertical subplots",
    )
    parser.add_argument(
        "--hue",
        type=str,
        required=True,
        choices=["layers", "category", "parameter"],
        help="Variable for color hues",
    )
    parser.add_argument(
        "--column",
        type=str,
        default=None,
        choices=["parameter"],
        help="Variable for columns (optional)",
    )
    parser.add_argument("--category-key", type=str, help="Category column name")
    parser.add_argument("--parameter-key", type=str, help="Parameter column name")
    parser.add_argument("--experiment", type=str, help="Experiment name")
    parser.add_argument("--dt", type=float, default=2.0, help="Time resolution (ms)")
    parser.add_argument("--palette", type=str, help="JSON color palette")
    parser.add_argument("--naming", type=str, help="JSON naming dict")
    parser.add_argument("--ordering", type=str, help="JSON ordering dict")
    parser.add_argument(
        "--log-level",
        type=str,
        default="DEBUG",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    # Use parse_known_args to allow additional arguments
    args, unknown = parser.parse_known_args()

    # Configure logging - but don't add handlers if they already exist
    logging.basicConfig(
        level=getattr(logging, "DEBUG"),
        force=False,  # Don't override existing configuration
    )

    if unknown:
        logger.info(f"Ignoring unknown arguments: {unknown}")

    # Load config
    config = load_config_from_args(
        palette_str=args.palette,
        naming_str=args.naming,
        ordering_str=args.ordering,
    )

    # Plot
    plot_temporal_ridge_responses(
        data=args.data,
        output=args.output,
        subplot_var=args.subplot,
        hue_var=args.hue,
        column_var=args.column,
        category_key=args.category_key,
        parameter_key=args.parameter_key,
        experiment=args.experiment,
        dt=args.dt,
        config=config,
    )


if __name__ == "__main__":

    main()
