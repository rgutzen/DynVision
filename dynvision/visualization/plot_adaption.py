import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from dynvision.utils import replace_param_in_string
from dynvision.utils.visualization_utils import (
    save_plot,
    load_config_from_args,
    get_display_name,
    get_color,
    get_category_plotting_settings,
)

# Global styling parameters
FONTSIZE_PANEL_LABELS = 18
FONTSIZE_AXIS_LABELS = 18
FONTSIZE_TICK_LABELS = 16
FONTSIZE_LEGEND = 18
FONTSIZE_TITLE = 20
LINEWIDTH_MAIN = 3
ALPHA_LINES = 0.8
FIGURE_HEIGHT_PER_SUBPLOT = 4
SUBPLOT_SPACING = 0.3

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=Path, required=True, help="Path to pt files")
parser.add_argument("--output", type=Path, required=True, help="Path to directory")
parser.add_argument("--parameter", type=str, required=True, help="Parameter to plot")
parser.add_argument("--palette", type=str, help="JSON formatted dictionary of colors")
parser.add_argument("--naming", type=str, help="JSON formatted naming dictionary")
parser.add_argument("--ordering", type=str, help="JSON formatted ordering dictionary")


def plot_adaptation(data_path, parameter, config=None, output_path=None):
    """Create adaptation plot showing model adaptation over time."""

    print(f"Loading adaptation data from: {data_path}")

    # Find all .pt files in the directory
    pt_files = list(data_path.glob("*.pt"))
    print(f"Found {len(pt_files)} .pt files")

    if len(pt_files) == 0:
        raise ValueError(f"No .pt files found in {data_path}")

    # Extract parameter values and organize files
    file_data = []
    for pt_file in pt_files:
        try:
            param_value = replace_param_in_string(pt_file.stem, parameter, 0.0)
            file_data.append((param_value, pt_file))
        except Exception as e:
            print(f"Warning: Could not extract {parameter} from {pt_file.stem}: {e}")
            continue

    # Sort by parameter value
    file_data.sort(key=lambda x: x[0])
    param_values = [item[0] for item in file_data]
    files = [item[1] for item in file_data]

    print(f"Parameter values: {param_values}")

    # Load data and compute adaptation metrics
    adaptation_data = []

    for param_val, pt_file in tqdm(zip(param_values, files), desc="Processing files"):
        try:
            # Load tensor data
            data = torch.load(pt_file, map_location="cpu")

            # Compute adaptation metric (example: mean response over time)
            if isinstance(data, dict):
                # Multiple layers/components
                for layer_name, tensor in data.items():
                    if tensor.ndim >= 2:  # Has time dimension
                        # Compute mean response over spatial dimensions
                        mean_response = tensor.mean(dim=tuple(range(2, tensor.ndim)))

                        # Compute adaptation as change over time
                        if mean_response.shape[1] > 1:  # Multiple time steps
                            initial_response = mean_response[:, 0].mean().item()
                            final_response = mean_response[:, -1].mean().item()
                            adaptation_ratio = final_response / (
                                initial_response + 1e-8
                            )
                        else:
                            adaptation_ratio = 1.0

                        adaptation_data.append(
                            {
                                parameter: param_val,
                                "layer": layer_name,
                                "adaptation_ratio": adaptation_ratio,
                                "initial_response": initial_response,
                                "final_response": final_response,
                            }
                        )
            else:
                # Single tensor
                if data.ndim >= 2:
                    mean_response = data.mean(dim=tuple(range(2, data.ndim)))
                    if mean_response.shape[1] > 1:
                        initial_response = mean_response[:, 0].mean().item()
                        final_response = mean_response[:, -1].mean().item()
                        adaptation_ratio = final_response / (initial_response + 1e-8)
                    else:
                        adaptation_ratio = 1.0

                    adaptation_data.append(
                        {
                            parameter: param_val,
                            "layer": "response",
                            "adaptation_ratio": adaptation_ratio,
                            "initial_response": initial_response,
                            "final_response": final_response,
                        }
                    )

        except Exception as e:
            print(f"Error processing {pt_file}: {e}")
            continue

    if not adaptation_data:
        raise ValueError("No adaptation data could be computed")

    # Convert to DataFrame
    df = pd.DataFrame(adaptation_data)
    print(f"Computed adaptation data shape: {df.shape}")
    print(f"Available layers: {df['layer'].unique()}")

    # Get unique layers
    layers = sorted(df["layer"].unique())
    n_layers = len(layers)

    # Set seaborn style
    sns.set_style("whitegrid")
    sns.set_context("talk")

    # Create figure with subplots for each layer
    fig, axes = plt.subplots(
        n_layers, 1, figsize=(10, FIGURE_HEIGHT_PER_SUBPLOT * n_layers), sharex=True
    )

    # Ensure axes is always a list
    if n_layers == 1:
        axes = [axes]

    # Color mapping for layers
    layer_colors = plt.cm.Set1(np.linspace(0, 1, n_layers))
    layer_color_dict = dict(zip(layers, layer_colors))

    # Plot adaptation for each layer
    for i, layer in enumerate(layers):
        ax = axes[i]
        layer_data = df[df["layer"] == layer]

        # Plot adaptation ratio
        sns.lineplot(
            data=layer_data,
            x=parameter,
            y="adaptation_ratio",
            color=layer_color_dict[layer],
            linewidth=LINEWIDTH_MAIN,
            marker="o",
            markersize=8,
            alpha=ALPHA_LINES,
            ax=ax,
        )

        # Add horizontal line at y=1 (no adaptation)
        ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, linewidth=2)

        # Customize subplot
        layer_display_name = get_display_name(layer, config)
        ax.set_ylabel(
            f"{layer_display_name}\nAdaptation Ratio", fontsize=FONTSIZE_AXIS_LABELS
        )
        ax.grid(True, alpha=0.3)

        # Add text annotation for interpretation
        ax.text(
            0.02,
            0.98,
            (
                "Sensitization"
                if layer_data["adaptation_ratio"].mean() > 1
                else "Adaptation"
            ),
            transform=ax.transAxes,
            fontsize=FONTSIZE_TICK_LABELS,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
        )

        # Remove top and right spines
        sns.despine(ax=ax)

    # Set x-label only on the bottom subplot
    param_display_name = get_display_name(parameter, config)
    axes[-1].set_xlabel(param_display_name, fontsize=FONTSIZE_AXIS_LABELS)

    # Overall title
    fig.suptitle(
        f"Neural Adaptation across {param_display_name}",
        fontsize=FONTSIZE_TITLE,
        y=0.98,
    )

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.95, hspace=SUBPLOT_SPACING)

    return fig


if __name__ == "__main__":
    args = parser.parse_args()

    # Load configuration from command line arguments
    config = load_config_from_args(
        palette_str=args.palette, naming_str=args.naming, ordering_str=args.ordering
    )

    # Check if data directory exists
    if not args.data.exists():
        print(f"Warning: Data directory {args.data} does not exist")
        # Create an empty figure
        fig = plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, "No data available", ha="center", va="center", fontsize=16)
        plt.axis("off")
    else:
        # Generate the adaptation plot
        fig = plot_adaptation(
            args.data,
            parameter=args.parameter,
            config=config,
            output_path=args.output,
        )

    # Save the plot
    save_plot(args.output)

    print(f"Adaptation plot saved to: {args.output}")
