import argparse
import json
from pathlib import Path

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
FIGURE_HEIGHT_PER_SUBPLOT = 3
SUBPLOT_SPACING = 0.2

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
parser.add_argument("--experiment", type=str, help="Experiment name for title")
parser.add_argument(
    "--category",
    type=str,
    required=True,
    help="Column name containing model categories",
)
parser.add_argument("--palette", type=str, help="JSON formatted dictionary of colors")
parser.add_argument("--naming", type=str, help="JSON formatted naming dictionary")
parser.add_argument("--ordering", type=str, help="JSON formatted ordering dictionary")


def plot_response(
    df, parameter, category, experiment=None, config=None, output_path=None
):
    """Create response plot showing model responses across parameter values."""

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

    # Get plotting settings from config
    category_values = sorted(df[category].unique())
    model_order, colors = get_category_plotting_settings(
        category, category_values, config
    )

    # Get unique parameter values
    param_values = sorted(df[parameter].unique())
    n_params = len(param_values)
    n_layers = len(ordered_layers)

    # Set seaborn style
    sns.set_style("whitegrid")
    sns.set_context("talk")

    # Create figure with subplots: layers x parameters
    fig, axes = plt.subplots(
        n_layers,
        n_params,
        figsize=(4 * n_params, FIGURE_HEIGHT_PER_SUBPLOT * n_layers),
        sharex=True,
        sharey="row",
    )

    # Ensure axes is always 2D
    if n_layers == 1 and n_params == 1:
        axes = [[axes]]
    elif n_layers == 1:
        axes = [axes]
    elif n_params == 1:
        axes = [[ax] for ax in axes]

    # Plot for each layer and parameter combination
    for i, layer in enumerate(ordered_layers):
        layer_col = f"{layer}_response_avg"

        if layer_col not in df.columns:
            print(f"Warning: {layer_col} not found in data")
            continue

        for j, param_val in enumerate(param_values):
            ax = axes[i][j]
            param_data = df[df[parameter] == param_val]

            # Plot each category
            for cat_val in model_order:
                if cat_val in param_data[category].values:
                    cat_data = param_data[param_data[category] == cat_val]

                    # Group by time to get mean power
                    time_data = (
                        cat_data.groupby("times_index")[layer_col].mean().reset_index()
                    )

                    # Use seaborn lineplot
                    sns.lineplot(
                        data=time_data,
                        x="times_index",
                        y=layer_col,
                        color=colors[cat_val],
                        linewidth=LINEWIDTH_MAIN,
                        alpha=ALPHA_LINES,
                        ax=ax,
                        label=cat_val if i == 0 and j == 0 else "",
                    )

            # Add label indicator
            if len(param_data) > 0:
                y_min, y_max = ax.get_ylim()
                label_indicator_df = calculate_label_indicator(
                    param_data, category, (y_min, y_max)
                )

                ax.plot(
                    label_indicator_df.times_index,
                    label_indicator_df.label_indicator,
                    color="gray",
                    linewidth=LINEWIDTH_INDICATOR,
                    drawstyle="steps-mid",
                    alpha=ALPHA_INDICATOR,
                )

            # Customize subplot
            if i == 0:
                param_display = param_val
                param_symbol = get_display_name(parameter, config)
                ax.set_title(
                    f"{param_symbol}={param_display}", fontsize=FONTSIZE_TICK_LABELS
                )

            if j == 0:
                layer_display_name = get_display_name(layer, config)
                ax.set_ylabel(
                    f"{layer_display_name}\nResponse", fontsize=FONTSIZE_AXIS_LABELS
                )

            if i == n_layers - 1:
                ax.set_xlabel("Time Step", fontsize=FONTSIZE_AXIS_LABELS)

            ax.grid(True, alpha=0.3)

            # Remove top and right spines
            sns.despine(ax=ax)

    # Add legend only to the first subplot
    if n_layers > 0 and n_params > 0:
        axes[0][0].legend(
            title="Model Category",
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            fontsize=FONTSIZE_LEGEND,
            title_fontsize=FONTSIZE_LEGEND,
        )

    # Overall title
    param_display_name = get_display_name(parameter, config)
    fig.suptitle(
        f"Layer Responses across {param_display_name}", fontsize=FONTSIZE_TITLE, y=0.98
    )

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.95, right=0.85, hspace=SUBPLOT_SPACING, wspace=0.3)

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
        # Generate the response plot
        fig = plot_response(
            df,
            parameter=args.parameter,
            category=args.category,
            experiment=args.experiment,
            config=config,
            output_path=args.output,
        )

    # Save the plot
    save_plot(args.output)

    print(f"Response plot saved to: {args.output}")
