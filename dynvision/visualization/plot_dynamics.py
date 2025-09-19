"""plot_dynamics.py
"""

import argparse
import json
import re
from pathlib import Path
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from dynvision.utils.visualization_utils import (
    save_plot,
    load_config_from_args,
    get_display_name,
    get_color,
    calculate_label_indicator,
    get_category_plotting_settings,
    order_layers,
)

# Global styling parameters
FONTSIZE_PANEL_LABELS = 18
FONTSIZE_AXIS_LABELS = 18
FONTSIZE_TICK_LABELS = 16
FONTSIZE_LEGEND = 18
FONTSIZE_TITLE = 20
LINEWIDTH_MAIN = 3
LINEWIDTH_INDICATOR = 3
ALPHA_LINES = 0.8
ALPHA_INDICATOR = 0.6
FIGURE_HEIGHT_PER_SUBPLOT = 4
SUBPLOT_SPACING = 0.3

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data", type=Path, required=True, help="Path to layer power CSV file"
)
parser.add_argument(
    "--output", type=Path, required=True, help="Path to output plot file"
)
parser.add_argument(
    "--parameter",
    type=str,
    required=True,
    help="Column name containing experiment parameter values",
)
parser.add_argument(
    "--experiment", type=str, required=True, help="Experiment name for title"
)
parser.add_argument(
    "--category",
    type=str,
    required=True,
    help="Column name containing model categories",
)
parser.add_argument("--focus_layer", type=str, default="V1", help="Layer to focus on")
parser.add_argument("--dt", type=float, help="Time step duration in milliseconds")
parser.add_argument("--palette", type=str, help="JSON formatted dictionary of colors")
parser.add_argument("--naming", type=str, help="JSON formatted naming dictionary")
parser.add_argument("--ordering", type=str, help="JSON formatted ordering dictionary")


def plot_dynamics(
    df,
    parameter,
    experiment,
    category,
    focus_layer="V1",
    config=None,
    dt=None,
    output_path=None,
):
    """Create dynamics plot showing layer responses over parameter values."""

    print(f"Plot data shape: {df.shape}")
    print(f"Available columns: {df.columns.tolist()}")

    # Validate that the specified columns exist
    if parameter not in df.columns:
        raise ValueError(
            f"Specified parameter column '{parameter}' not found in data. Available columns: {df.columns.tolist()}"
        )

    if category not in df.columns:
        raise ValueError(
            f"Specified category column '{category}' not found in data. Available columns: {df.columns.tolist()}"
        )

    print(f"Using parameter column: {parameter}")
    print(f"Using category column: {category}")

    # Get layer names from columns
    layer_cols = [col for col in df.columns if col.endswith("_response_avg")]
    layer_names = [col.replace("_response_avg", "") for col in layer_cols]

    # Order layers according to hierarchy
    ordered_layers = order_layers(layer_names, config)

    print(f"Using layers: {ordered_layers}")
    print(f"Focus layer: {focus_layer}")

    # Get plotting settings from config
    category_values = sorted(df[category].unique())
    model_order, colors = get_category_plotting_settings(
        category, category_values, config
    )

    # Get unique parameter values
    param_values = sorted(df[parameter].unique())
    n_params = len(param_values)
    n_layers = len(ordered_layers)

    # Convert time axis if dt is provided
    time_col = "times_index"
    if dt is not None:
        print(f"Converting time axis using dt={dt} ms")
        df = df.copy()
        df["time_ms"] = df["times_index"] * dt
        time_col = "time_ms"
        xlabel = "Time (ms)"
    else:
        xlabel = "Time Step"

    # Set seaborn style
    sns.set_style("whitegrid")
    sns.set_context("talk")

    # Create figure with subplots for each layer
    fig, axes = plt.subplots(
        n_layers, 1, figsize=(12, FIGURE_HEIGHT_PER_SUBPLOT * n_layers), sharex=True
    )

    # Ensure axes is always a list
    if n_layers == 1:
        axes = [axes]

    # Plot for each layer
    for i, layer in enumerate(ordered_layers):
        ax = axes[i]
        layer_col = f"{layer}_response_avg"

        if layer_col not in df.columns:
            print(f"Warning: {layer_col} not found in data")
            continue

        # Plot each category
        for cat_val in model_order:
            if cat_val in df[category].values:
                cat_data = df[df[category] == cat_val]

                # Group by parameter and time to get mean power
                plot_data = (
                    cat_data.groupby([parameter, time_col])[layer_col]
                    .mean()
                    .reset_index()
                )

                # Use seaborn lineplot
                sns.lineplot(
                    data=plot_data,
                    x=time_col,
                    y=layer_col,
                    hue=parameter,
                    palette="viridis",
                    linewidth=LINEWIDTH_MAIN,
                    alpha=ALPHA_LINES,
                    ax=ax,
                    legend=False,
                )

        # Add label indicator
        if len(df) > 0:
            y_min, y_max = ax.get_ylim()
            label_indicator_df = calculate_label_indicator(
                df, category, (y_min, y_max)
            )

            # Convert time for indicator if needed
            if dt is not None:
                indicator_time = label_indicator_df.times_index * dt
            else:
                indicator_time = label_indicator_df.times_index

            ax.plot(
                indicator_time,
                label_indicator_df.label_indicator,
                color="gray",
                linewidth=LINEWIDTH_INDICATOR,
                drawstyle="steps-mid",
                alpha=ALPHA_INDICATOR,
            )

        # Customize subplot
        layer_display_name = get_display_name(layer, config)
        ax.set_ylabel(f"{layer_display_name} Response", fontsize=FONTSIZE_AXIS_LABELS)
        ax.grid(True, alpha=0.3)

        # Highlight focus layer
        if layer == focus_layer:
            ax.set_facecolor("#f0f0f0")

        # Remove top and right spines
        sns.despine(ax=ax)

    # Set x-label only on the bottom subplot
    axes[-1].set_xlabel(xlabel, fontsize=FONTSIZE_AXIS_LABELS)

    # Add legend for parameter values
    param_legend_elements = []
    param_colors = plt.cm.viridis(np.linspace(0, 1, len(param_values)))
    for param_val, color in zip(param_values, param_colors):
        param_display = param_val
        if dt is not None and parameter.lower() in [
            "duration",
            "interval",
            "stim",
            "stimulus",
        ]:
            param_display = f"{int(param_val * dt)} ms"
        param_legend_elements.append(
            plt.Line2D(
                [0],
                [0],
                color=color,
                linewidth=LINEWIDTH_MAIN,
                label=f"{get_display_name(parameter, config)}={param_display}",
            )
        )

    # Position legend
    fig.legend(
        handles=param_legend_elements,
        title=f"{get_display_name(parameter, config)} Values",
        bbox_to_anchor=(1.02, 0.5),
        loc="center left",
        fontsize=FONTSIZE_LEGEND,
        title_fontsize=FONTSIZE_LEGEND,
    )

    # Overall title
    experiment_display = get_display_name(f"{experiment}_experiment", config)
    fig.suptitle(
        f"{experiment_display}: Layer Dynamics", fontsize=FONTSIZE_TITLE, y=0.98
    )

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.95, right=0.82, hspace=SUBPLOT_SPACING)

    return fig


if __name__ == "__main__":
    args = parser.parse_args()

    print(f"Loading layer power data from: {args.data}")
    df = pd.read_csv(args.data)

    # Load configuration from command line arguments
    config = load_config_from_args(
        palette_str=args.palette, naming_str=args.naming, ordering_str=args.ordering
    )

    # Check if data is empty
    if len(df) == 0:
        print("Warning: No data found in the input file")
        # Create an empty figure
        fig = plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, "No data available", ha="center", va="center", fontsize=16)
        plt.axis("off")
    else:
        # Generate the dynamics plot
        fig = plot_dynamics(
            df,
            parameter=args.parameter,
            experiment=args.experiment,
            category=args.category,
            focus_layer=args.focus_layer,
            config=config,
            dt=args.dt,
            output_path=args.output,
        )

    # Save the plot
    save_plot(args.output)

    print(f"Dynamics plot saved to: {args.output}")
