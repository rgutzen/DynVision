"""plot_dynamics.py
Unified plot showing layer dynamics across different parameter values with consistent layout.
 
This script creates a 4-panel visualization of neural network layer responses:
    - Panel A (upper left): Different layers for a reference parameter value
    - Panel B (lower left): Focus layer across different parameter values
    - Panel C (upper right): Empty panel for custom content
    - Panel D (lower right): Summary plot showing parameter effect on metrics

Example:
    ```bash
    python plot_dynamics.py --data responses.csv --output dynamics_plot.png --parameter tau \
        --category interval --focus-layer V1 --dt 1.0
    ```
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from dynvision.utils.visualization_utils import (
    save_plot,
    load_config_from_args,
    get_display_name,
    order_layers,
    peak_ratio,
    peak_time,
    peak_height,
)

# Import functions from plot_responses.py
from dynvision.visualization.plot_responses import (
    _plot_response_ridges,
    _add_horizontal_legend,
    _extract_dimension_values,
    _get_colors_for_dimension,
    _filter_data_for_column,
    _add_layer_circle,
    FORMATTING as RESPONSES_FORMATTING,
    LAYOUT as RESPONSES_LAYOUT,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Layout parameters for the 4-panel figure
DYNAMICS_LAYOUT = {
    "figure_width": 15,
    "figure_height": 10,
    "plots_start": 0.05,
    "left_plots_width": 0.6,
    "right_plots_width": 0.3,
    "gap_width": 0.1,
    # Vertical positions
    "upper_top": 0.98,
    "upper_bottom": 0.65,
    "lower_top": 0.48,
    "lower_bottom": 0.01,
    "legend_height": 0.1,
    "legend_margin": 0.02,
    # Positioning helpers
    "y_label_x_position": 0.01,
    "panel_label_x_offset": -0.04,
    "panel_label_y_offset": 0.02,
}

# Style constants
DYNAMICS_STYLE = {
    "fontsize_panel_labels": 16,
    "marker_edge_width": 1.2,
}


def determine_experiment_info(experiment):
    """Determine experiment type and corresponding labels from experiment name."""
    if "duration" in experiment.lower():
        return {
            "type": "duration",
            "label": "$D$",
            "full_label": "Duration ($D$)",
            "metric": "peak_height",
            "y_label": "Peak Response",
        }
    elif "contrast" in experiment.lower():
        return {
            "type": "contrast",
            "label": "$C$",
            "full_label": "Contrast ($C$)",
            "metric": "peak_time",
            "y_label": "Peak Time",
        }
    elif "interval" in experiment.lower():
        return {
            "type": "interval",
            "label": "$I$",
            "full_label": "Interval ($I$)",
            "metric": "peak_ratio",
            "y_label": "Peak Ratio",
        }
    else:
        return {
            "type": "unknown",
            "label": "$\\Delta_s$",
            "full_label": f"$\\Delta_s$ ({experiment})",
            "metric": "peak_height",
            "y_label": "Peak Height",
        }


def format_parameter_value(value, parameter, dt=None):
    """Format parameter values with appropriate units."""
    timestep_params = ["tsteps", "idle"]
    ms_params = ["tau", "trc", "tsk", "lossrt", "duration", "interval"]

    if dt is not None and parameter.lower() in timestep_params:
        return f"{int(float(value) * dt)} ms"
    elif parameter.lower() in ms_params:
        return f"{int(float(value))} ms"
    elif str(value).lower() in ["true", "false"]:
        return str(value).capitalize()
    elif parameter.lower() in ["contrast", "noiselevel"]:
        return f"{float(value):.2f}"
    else:
        return str(value).capitalize()


def add_panel_labels(fig):
    """Add panel labels A), B), C), D) to the figure."""
    panel_x_offset = DYNAMICS_LAYOUT["panel_label_x_offset"]
    panel_y_offset = DYNAMICS_LAYOUT["panel_label_y_offset"]
    plots_start = DYNAMICS_LAYOUT["plots_start"]
    left_width = DYNAMICS_LAYOUT["left_plots_width"]
    gap_width = DYNAMICS_LAYOUT["gap_width"]
    upper_top = DYNAMICS_LAYOUT["upper_top"]
    lower_top = DYNAMICS_LAYOUT["lower_top"]

    # Left side panel labels (A and B)
    fig.text(
        x=plots_start + panel_x_offset,
        y=upper_top + panel_y_offset,
        s="A)",
        fontsize=DYNAMICS_STYLE["fontsize_panel_labels"],
        fontweight="bold",
        ha="left",
        va="top",
    )
    fig.text(
        x=plots_start + panel_x_offset,
        y=lower_top + panel_y_offset,
        s="B)",
        fontsize=DYNAMICS_STYLE["fontsize_panel_labels"],
        fontweight="bold",
        ha="left",
        va="top",
    )

    # Right side panel labels (C and D)
    right_x = plots_start + left_width + gap_width + panel_x_offset
    fig.text(
        x=right_x,
        y=upper_top + panel_y_offset,
        s="C)",
        fontsize=DYNAMICS_STYLE["fontsize_panel_labels"],
        fontweight="bold",
        ha="left",
        va="top",
    )
    fig.text(
        x=right_x,
        y=lower_top + panel_y_offset,
        s="D)",
        fontsize=DYNAMICS_STYLE["fontsize_panel_labels"],
        fontweight="bold",
        ha="left",
        va="top",
    )


def calculate_metrics(df, focus_layer, parameter, category, metric, dt=None):
    """Calculate metrics (peak ratio, peak time, peak height) for summary plot."""
    df = df.copy()
    metric_col = f"{focus_layer}_{metric}"

    if metric_col in df.columns:
        return df

    df[metric_col] = np.nan

    # Group by parameter, category, AND first_label_index to calculate metrics per stimulus
    for group_key, group in df.groupby([parameter, category, "first_label_index"]):
        param_val, cat_val, pres_label = group_key
        logger.debug(f"Processing group: {param_val}, {cat_val}, {pres_label}")

        # Sort by times_index to ensure correct temporal order
        group = group.sort_values("times_index")

        response_col = f"{focus_layer}_response_avg"
        if response_col not in group.columns:
            continue

        response_values = group[response_col].values
        if len(response_values) == 0:
            continue

        try:
            # Reshape as [1, n_timesteps] for single stimulus response
            response_tensor = torch.tensor(response_values).reshape(1, -1)

            if metric == "peak_ratio":
                metric_value = float(peak_ratio(response_tensor, min_delay=5)[0])
            elif metric == "peak_time":
                peak_idx = int(peak_time(response_tensor)[0].item())
                metric_value = peak_idx * dt if dt is not None else peak_idx
            elif metric == "peak_height":
                metric_value = float(peak_height(response_tensor)[0].item())
            else:
                continue

            # Assign the calculated metric to all rows in this group
            for idx in group.index:
                df.at[idx, metric_col] = metric_value

        except Exception as e:
            logger.debug(
                f"Error calculating {metric} for {param_val}, {cat_val}, {pres_label}: {e}"
            )

    return df


def create_panel_a(
    fig,
    df,
    layer_names,
    parameter,
    category,
    param_value,
    hue_values,
    colors,
    dt,
    config,
):
    """Create Panel A: Ridge plot for layers at middle parameter value."""
    plots_start = DYNAMICS_LAYOUT["plots_start"]
    left_width = DYNAMICS_LAYOUT["left_plots_width"]
    upper_top = DYNAMICS_LAYOUT["upper_top"]
    upper_bottom = DYNAMICS_LAYOUT["upper_bottom"]

    # Filter data for middle parameter value
    panel_data = _filter_data_for_column(df, parameter, param_value)
    if dt is not None:
        panel_data["time_ms"] = panel_data["times_index"] * dt

    # Calculate layout parameters for ridge plots compatible with plot_responses
    panel_height = upper_top - upper_bottom

    # Create ridge plots for layers with proper layout parameters
    ridge_axes = _plot_response_ridges(
        fig=fig,
        column_left=plots_start,
        column_width=left_width,
        data=panel_data,
        subplot_var="layers",
        subplot_key="layers",
        subplot_values=layer_names,
        hue_var="category",
        hue_key=category,
        hue_values=hue_values,
        colors=colors,
        dt=dt or 1.0,
        show_ylabel=False,  # We'll add custom y-label
        config=config,
        # Override layout parameters to position correctly in Panel A
        title_bot=upper_top,
        title_pad=0.0,  # No title in this panel
        accuracy_height=0.0,  # No accuracy panel
        accuracy_pad=0.0,
        legend_height=0.0,  # No legend in individual panels
        legend_pad=0.0,
        ridge_top=upper_top,
        ridge_height=panel_height,
        ridge_overlap=0.2,
        **RESPONSES_FORMATTING,
    )

    # Add custom Y-axis label for the panel
    exp_info = determine_experiment_info(parameter)
    param_display = format_parameter_value(param_value, parameter, dt)
    fig.text(
        x=DYNAMICS_LAYOUT["y_label_x_position"],
        y=(upper_top + upper_bottom) / 2,
        s=f"Layer Responses ({exp_info['label']} = {param_display})",
        rotation=90,
        verticalalignment="center",
        fontsize=RESPONSES_FORMATTING["fontsize_label"],
        fontweight="bold",
    )

    return ridge_axes


def create_panel_b(
    fig,
    df,
    focus_layer,
    parameter,
    category,
    param_values,
    hue_values,
    colors,
    dt,
    config,
):
    """Create Panel B: Ridge plot for focus layer across all parameter values."""
    plots_start = DYNAMICS_LAYOUT["plots_start"]
    left_width = DYNAMICS_LAYOUT["left_plots_width"]
    lower_top = DYNAMICS_LAYOUT["lower_top"]
    lower_bottom = DYNAMICS_LAYOUT["lower_bottom"]

    # Prepare data with time conversion and ensure we have the focus layer response
    panel_data = df.copy()
    if dt is not None:
        panel_data["time_ms"] = panel_data["times_index"] * dt

    # Add focus layer response as the y-variable for each parameter subplot
    focus_response_col = f"{focus_layer}_response_avg"
    if focus_response_col not in panel_data.columns:
        logger.error(
            f"Focus layer response column '{focus_response_col}' not found in data"
        )
        return []

    # Calculate layout parameters
    panel_height = lower_top - lower_bottom

    # For Panel B, we plot parameter values as subplots and category as hue
    # But we need to use a single focus layer response column for all subplots
    ridge_axes = _plot_response_ridges(
        fig=fig,
        column_left=plots_start,
        column_width=left_width,
        data=panel_data,
        subplot_var="parameter",
        subplot_key=parameter,
        subplot_values=param_values,
        hue_var="category",
        hue_key=category,
        hue_values=hue_values,
        colors=colors,
        dt=dt or 1.0,
        show_ylabel=False,  # We'll add custom y-label
        config=config,
        # Override layout parameters to position correctly in Panel B
        title_bot=lower_top,
        title_pad=0.0,  # No title in this panel
        accuracy_height=0.0,  # No accuracy panel
        accuracy_pad=0.0,
        legend_height=0.0,  # No legend in individual panels
        legend_pad=0.0,
        ridge_top=lower_top,
        ridge_height=panel_height,
        ridge_overlap=0.2,
        # Add focus layer info so the function knows which response to plot
        focus_layer=focus_layer,
        **RESPONSES_FORMATTING,
    )

    # Add custom Y-axis label
    layer_display = get_display_name(focus_layer, config) or focus_layer.upper()
    fig.text(
        x=DYNAMICS_LAYOUT["y_label_x_position"],
        y=(lower_top + lower_bottom) / 2,
        s=f"Layer {layer_display} Response",
        rotation=90,
        verticalalignment="center",
        fontsize=RESPONSES_FORMATTING["fontsize_label"],
        fontweight="bold",
    )

    return ridge_axes


def create_panel_c(fig):
    """Create Panel C: Empty panel for custom content."""
    plots_start = DYNAMICS_LAYOUT["plots_start"]
    left_width = DYNAMICS_LAYOUT["left_plots_width"]
    gap_width = DYNAMICS_LAYOUT["gap_width"]
    right_width = DYNAMICS_LAYOUT["right_plots_width"]
    upper_top = DYNAMICS_LAYOUT["upper_top"]
    upper_bottom = DYNAMICS_LAYOUT["upper_bottom"]

    panel_height = upper_top - upper_bottom
    ax = fig.add_axes(
        [
            plots_start + left_width + gap_width,
            upper_bottom,
            right_width,
            panel_height,
        ]
    )
    ax.patch.set_alpha(0)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    return ax


def create_panel_d(
    fig, df, focus_layer, parameter, category, exp_info, hue_values, colors, dt, config
):
    """Create Panel D: Summary plot showing parameter effect on metrics."""
    plots_start = DYNAMICS_LAYOUT["plots_start"]
    left_width = DYNAMICS_LAYOUT["left_plots_width"]
    gap_width = DYNAMICS_LAYOUT["gap_width"]
    right_width = DYNAMICS_LAYOUT["right_plots_width"]
    lower_top = DYNAMICS_LAYOUT["lower_top"]
    lower_bottom = DYNAMICS_LAYOUT["lower_bottom"]

    summary_height = (lower_top - lower_bottom) * 0.6
    summary_center = lower_bottom + (lower_top - lower_bottom) / 2
    summary_bottom = summary_center - summary_height / 2
    summary_left = plots_start + left_width + gap_width

    summary_ax = fig.add_axes(
        [summary_left, summary_bottom, right_width, summary_height]
    )
    summary_ax.patch.set_alpha(0)

    # Add title
    layer_display = get_display_name(focus_layer, config) or focus_layer.upper()
    _add_layer_circle(
        x=0.5, y=1.1, ax=summary_ax, layer_name=focus_layer, config=config
    )

    # Calculate metrics
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

    df[category] = df[category].astype(str)
    hue_values_str = [str(val) for val in hue_values]

    if df[metric_col].notna().any():
        sns.lineplot(
            data=df,
            x=parameter,
            y=metric_col,
            ax=summary_ax,
            marker=None,
            hue=category,
            hue_order=hue_values_str,
            palette=colors,
            legend=False,
            errorbar="sd",
            err_style="bars",
            linewidth=RESPONSES_FORMATTING["linewidth_main"],
            alpha=RESPONSES_FORMATTING["alpha_line"],
        )

        for line in summary_ax.get_lines():
            line.set_markeredgecolor("white")
            line.set_markeredgewidth(DYNAMICS_STYLE["marker_edge_width"])

        param_values = df[parameter].unique()
        summary_ax.set_xticks(sorted(param_values))

        if dt is not None and parameter.lower() in ["tau", "interval", "duration"]:
            summary_ax.set_xticklabels(
                [f"{int(x * dt)}" for x in sorted(param_values)]
            )

        summary_ax.grid(True, which="major", linestyle=":", alpha=0.3)

        if exp_info["metric"] == "peak_ratio":
            summary_ax.axhline(
                y=1, color="gray", linestyle="--", linewidth=1, alpha=0.7
            )
    else:
        summary_ax.text(
            x=0.5,
            y=0.5,
            s="No summary data available",
            ha="center",
            va="center",
            fontsize=RESPONSES_FORMATTING["fontsize_label"],
            alpha=0.7,
        )

    summary_ax.set_xlabel(
        exp_info["full_label"], fontsize=RESPONSES_FORMATTING["fontsize_label"]
    )
    summary_ax.set_ylabel(
        exp_info["y_label"],
        fontsize=RESPONSES_FORMATTING["fontsize_label"],
        fontweight="bold",
    )
    summary_ax.tick_params(
        axis="both", labelsize=RESPONSES_FORMATTING["fontsize_tick"]
    )
    sns.despine(ax=summary_ax, left=True)

    return summary_ax


def plot_unified_dynamics(
    df,
    focus_layer,
    *,
    parameter="tau",
    category="interval",
    experiment=None,
    save_path=None,
    dt=None,
    config=None,
):
    """Create unified 4-panel dynamics visualization."""
    logger.info("=" * 60)
    logger.info("Starting unified dynamics plotting")
    logger.info("=" * 60)

    # Validate inputs
    if parameter not in df.columns:
        raise ValueError(f"Parameter column '{parameter}' not found")
    if category not in df.columns:
        raise ValueError(f"Category column '{category}' not found")

    # Initialize config if not provided
    if config is None:
        config = {"palette": {}, "naming": {}, "ordering": {}}

    # Determine experiment info
    experiment = experiment or parameter
    exp_info = determine_experiment_info(experiment)

    # Extract dimension values using plot_responses functions
    layer_names = _extract_dimension_values(df, "layers", "layers", config)
    hue_values = _extract_dimension_values(df, "category", category, config)
    param_values = _extract_dimension_values(df, "parameter", parameter, config)

    logger.info(f"Layers: {layer_names}")
    logger.info(f"Category values: {hue_values}")
    logger.info(f"Parameter values: {param_values}")

    # Get colors using plot_responses function
    colors = _get_colors_for_dimension(hue_values, category, config)

    # Select middle parameter value
    middle_idx = len(param_values) // 2
    middle_param_value = param_values[middle_idx]
    logger.info(f"Using middle parameter value: {middle_param_value}")

    # Convert time if needed
    if dt is not None:
        df = df.copy()
        df["time_ms"] = df["times_index"] * dt

    # Create figure
    fig = plt.figure(
        figsize=(DYNAMICS_LAYOUT["figure_width"], DYNAMICS_LAYOUT["figure_height"])
    )
    sns.set_style("ticks")
    sns.set_context("talk")

    # Add panel labels
    add_panel_labels(fig)

    # Create panels
    logger.info("Creating Panel A: Layers")
    panel_a_axes = create_panel_a(
        fig,
        df,
        layer_names,
        parameter,
        category,
        middle_param_value,
        hue_values,
        colors,
        dt,
        config,
    )

    logger.info("Creating Panel B: Parameter variation")
    panel_b_axes = create_panel_b(
        fig,
        df,
        focus_layer,
        parameter,
        category,
        param_values,
        hue_values,
        colors,
        dt,
        config,
    )

    logger.info("Creating Panel C: Empty panel")
    panel_c_ax = create_panel_c(fig)

    logger.info("Creating Panel D: Summary metrics")
    panel_d_ax = create_panel_d(
        fig,
        df,
        focus_layer,
        parameter,
        category,
        exp_info,
        hue_values,
        colors,
        dt,
        config,
    )

    # Add horizontal legend (positioned manually for dynamics layout)
    legend_left = DYNAMICS_LAYOUT["plots_start"]
    legend_width = DYNAMICS_LAYOUT[
        "left_plots_width"
    ]  # + DYNAMICS_LAYOUT["right_plots_width"]
    legend_height = DYNAMICS_LAYOUT["legend_height"]
    legend_bottom = DYNAMICS_LAYOUT["lower_top"] + DYNAMICS_LAYOUT["legend_margin"]

    _add_horizontal_legend(
        fig=fig,
        hue_var="category",
        hue_key=category,
        hue_values=hue_values,
        colors=colors,
        config=config,
        dt=dt or 1.0,
        legend_bot=legend_bottom,
        # Override legend positioning for dynamics layout
        **(
            RESPONSES_FORMATTING
            | {
                "legend_height": legend_height,
                "left_margin": legend_left,
                "column_width": legend_width,
            }
        ),
    )

    # Save figure
    if save_path:
        logger.info(f"Saving figure to {save_path}")
        save_plot(file_path=Path(save_path), dpi=300)

    logger.info("=" * 60)
    logger.info("Dynamics plotting complete")
    logger.info("=" * 60)

    return fig


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Create unified dynamics visualization"
    )
    parser.add_argument(
        "--data", type=Path, required=True, help="Path to response data CSV"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Path to save output figure"
    )
    parser.add_argument(
        "--parameter", type=str, required=True, help="Parameter column name"
    )
    parser.add_argument("--experiment", type=str, help="Experiment name")
    parser.add_argument(
        "--category", type=str, required=True, help="Category column name"
    )
    parser.add_argument(
        "--focus-layer", type=str, default="V1", help="Layer to focus on"
    )
    parser.add_argument("--dt", type=float, help="Time step in milliseconds")
    parser.add_argument("--palette", type=str, help="JSON color palette")
    parser.add_argument("--naming", type=str, help="JSON naming dict")
    parser.add_argument("--ordering", type=str, help="JSON ordering dict")

    args = parser.parse_args()

    # Load data
    df = pd.read_csv(args.data)

    # Load config
    config = load_config_from_args(
        palette_str=args.palette,
        naming_str=args.naming,
        ordering_str=args.ordering,
    )

    # Create plot
    plot_unified_dynamics(
        df=df,
        focus_layer=args.focus_layer,
        parameter=args.parameter,
        category=args.category,
        experiment=args.experiment,
        save_path=args.output,
        dt=args.dt,
        config=config,
    )


if __name__ == "__main__":
    main()
