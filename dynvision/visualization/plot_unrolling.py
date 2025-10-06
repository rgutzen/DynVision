import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch

from dynvision.utils.visualization_utils import (
    save_plot,
    load_config_from_args,
    get_display_name,
    order_layers,
    tensor_to_numpy,
    layer_response_avg,
)

# Global styling parameters
FONTSIZE_PANEL_LABELS = 18
FONTSIZE_AXIS_LABELS = 18
FONTSIZE_TICK_LABELS = 16
FONTSIZE_LEGEND = 18
FONTSIZE_TITLE = 20
LINEWIDTH_MAIN = 3
ALPHA_LINES = 0.8
FIGURE_HEIGHT_PER_SUBPLOT = 3
SUBPLOT_SPACING = 0.2

parser = argparse.ArgumentParser()
parser.add_argument(
    "--engineering_time_data",
    type=Path,
    required=True,
    help="Path to engineering time response .pt file",
)
parser.add_argument(
    "--biological_time_data",
    type=Path,
    required=True,
    help="Path to biological time response .pt file",
)
parser.add_argument(
    "--t_feedforward",
    type=int,
    required=True,
    help="Time shift value for each successive layer",
)
parser.add_argument(
    "--output", type=Path, required=True, help="Path to output plot file"
)
parser.add_argument("--palette", type=str, help="JSON formatted dictionary of colors")
parser.add_argument("--naming", type=str, help="JSON formatted naming dictionary")
parser.add_argument("--ordering", type=str, help="JSON formatted ordering dictionary")


def load_and_process_responses(response_file):
    """Load response tensor file and calculate average responses for each layer.

    Args:
        response_file: Path to .pt file containing layer responses

    Returns:
        dict: Dictionary mapping layer_name -> averaged response array (samples x time)
    """
    print(f"Loading responses from: {response_file}")
    responses = torch.load(response_file, map_location=torch.device("cpu"))

    # Process each layer to get average responses
    processed_responses = {}

    for layer_name, tensor in responses.items():
        # Calculate layer response average across spatial/feature dimensions
        avg_response = layer_response_avg(tensor)  # Shape: (samples, time)
        processed_responses[layer_name] = tensor_to_numpy(avg_response)

    return processed_responses


def pad_responses_to_match(engineering_responses, biological_responses):
    """Pad response data to ensure both datasets have the same time dimensions.

    Args:
        engineering_responses: Dict of layer_name -> response arrays
        biological_responses: Dict of layer_name -> response arrays

    Returns:
        Tuple of (padded_engineering_responses, padded_biological_responses)
    """
    print("Padding responses to match time dimensions...")

    padded_eng = {}
    padded_bio = {}

    # Get common layers
    common_layers = set(engineering_responses.keys()) & set(
        biological_responses.keys()
    )

    for layer_name in common_layers:
        eng_data = engineering_responses[layer_name]
        bio_data = biological_responses[layer_name]

        # Get maximum timesteps for this layer
        eng_timesteps = eng_data.shape[1]
        bio_timesteps = bio_data.shape[1]
        max_timesteps = max(eng_timesteps, bio_timesteps)

        # Pad engineering data if needed
        if eng_timesteps < max_timesteps:
            pad_len = max_timesteps - eng_timesteps
            padded_eng_data = np.zeros((eng_data.shape[0], max_timesteps))
            padded_eng_data[:, pad_len:] = eng_data
            padded_eng[layer_name] = padded_eng_data
            print(
                f"  Padded {layer_name} engineering data: {eng_timesteps} -> {max_timesteps}"
            )
        else:
            padded_eng[layer_name] = eng_data

        # Pad biological data if needed
        if bio_timesteps < max_timesteps:
            pad_len = max_timesteps - bio_timesteps
            padded_bio_data = np.zeros((bio_data.shape[0], max_timesteps))
            padded_bio_data[:, pad_len:] = bio_data
            padded_bio[layer_name] = padded_bio_data
            print(
                f"  Padded {layer_name} biological data: {bio_timesteps} -> {max_timesteps}"
            )
        else:
            padded_bio[layer_name] = bio_data

    return padded_eng, padded_bio


def create_time_shifted_trace(response_data, shift_amount):
    """Create time-shifted version of response data.

    Args:
        response_data: Array of shape (samples, time)
        shift_amount: Number of time steps to shift forward

    Returns:
        Array with time-shifted data, padded with zeros at the beginning
    """
    if shift_amount <= 0:
        return response_data

    # Pad at the beginning with zeros
    n_samples, n_time = response_data.shape
    shifted_data = np.zeros((n_samples, n_time))

    if shift_amount < n_time:
        shifted_data[:, shift_amount:] = response_data[:, :-shift_amount]

    return shifted_data


def plot_unrolling(
    engineering_responses,
    biological_responses,
    t_feedforward,
    config=None,
    output_path=None,
):
    """Create unrolling plot comparing biological vs engineering time responses.

    Args:
        engineering_responses: Dict of layer_name -> response arrays from engineering time
        biological_responses: Dict of layer_name -> response arrays from biological time
        t_feedforward: Time shift amount for successive layers
        config: Configuration dictionary for styling
        output_path: Path to save the plot

    Returns:
        matplotlib Figure object
    """

    # Pad responses to ensure matching time dimensions
    engineering_responses, biological_responses = pad_responses_to_match(
        engineering_responses, biological_responses
    )

    # Get layer names and order them
    layer_names = list(engineering_responses.keys())
    ordered_layers = order_layers(layer_names, config)

    print(f"Plotting layers in order: {ordered_layers}")

    n_layers = len(ordered_layers)

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

    # Define colors and markers for the three traces
    bio_color = "#1b780eff"  # Green for biological
    eng_color = "#d3a60cff"  # Orange/yellow for engineering
    shift_color = "#b3860cff"  # Darker shade of engineering color

    # Plot each layer
    for i, layer in enumerate(ordered_layers):
        ax = axes[i]

        if layer not in engineering_responses or layer not in biological_responses:
            print(f"Warning: {layer} not found in response data")
            continue

        eng_data = engineering_responses[layer]
        bio_data = biological_responses[layer]

        # Calculate time shift for this layer (bottom-up, V2 gets 1x shift, V4 gets 2x, etc.)
        # Find layer index from bottom (V1=0, V2=1, V4=2, etc.)
        layer_hierarchy = ["V1", "V2", "V4", "IT", "classifier"]
        if layer in layer_hierarchy:
            layer_idx = layer_hierarchy.index(layer)
        else:
            layer_idx = i  # fallback to subplot index

        shift_amount = layer_idx * t_feedforward

        # Create time-shifted engineering data
        shifted_eng_data = create_time_shifted_trace(eng_data, shift_amount)

        # Calculate mean across samples for plotting
        eng_mean = np.mean(eng_data, axis=0)
        bio_mean = np.mean(bio_data, axis=0)
        shifted_eng_mean = np.mean(shifted_eng_data, axis=0)

        # Create time axis
        time_steps = np.arange(len(eng_mean))

        # Plot the three traces with different markers
        ax.plot(
            time_steps,
            bio_mean,
            color=bio_color,
            linewidth=LINEWIDTH_MAIN,
            alpha=ALPHA_LINES,
            label="Biological time",
            linestyle="-",
        )

        ax.plot(
            time_steps,
            eng_mean,
            color=eng_color,
            linewidth=LINEWIDTH_MAIN,
            alpha=ALPHA_LINES,
            label="Engineering time",
            linestyle="-",
        )

        ax.plot(
            time_steps,
            shifted_eng_mean,
            color=shift_color,
            linewidth=LINEWIDTH_MAIN + 1,  # Slightly wider than default
            alpha=ALPHA_LINES,
            label="Engineering time shifted",
            linestyle=":",
            marker="1",
            markevery=4,  # Show marker only every 4 steps
        )

        # Customize subplot
        layer_display_name = get_display_name(layer, config)
        ax.set_ylabel(f"{layer_display_name}\nResponse", fontsize=FONTSIZE_AXIS_LABELS)

        if i == n_layers - 1:
            ax.set_xlabel("Time Step", fontsize=FONTSIZE_AXIS_LABELS)

        ax.grid(True, alpha=0.3)

        # Remove top and right spines
        sns.despine(ax=ax)

        # Add label indicator to the bottom subplot
        if i == n_layers - 1:
            # Create a mock label indicator (assuming stimulus presentation at early time points)
            # You may need to adapt this based on your actual data structure
            y_min, y_max = ax.get_ylim()

            # Simple label indicator: assume stimulus is present for first 35 time steps
            stimulus_duration = 35
            label_indicator = np.full(len(time_steps), y_min)
            label_indicator[:stimulus_duration] = y_min + 0.1 * (y_max - y_min)

            ax.plot(
                time_steps,
                label_indicator,
                color="gray",
                linewidth=LINEWIDTH_MAIN,
                drawstyle="steps-mid",
                alpha=0.6,
            )

    # Add legend to the second subplot from the bottom (or top if only one layer)
    legend_ax_idx = max(0, n_layers - 2) if n_layers >= 2 else 0
    axes[legend_ax_idx].legend(
        loc="upper right", fontsize=FONTSIZE_LEGEND, frameon=False
    )

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(hspace=SUBPLOT_SPACING)

    return fig


if __name__ == "__main__":
    args = parser.parse_args()

    # Load configuration from command line arguments
    config = load_config_from_args(
        palette_str=args.palette, naming_str=args.naming, ordering_str=args.ordering
    )

    # Load and process response data
    engineering_responses = load_and_process_responses(args.engineering_time_data)
    biological_responses = load_and_process_responses(args.biological_time_data)

    # Check if data is empty
    if len(engineering_responses) == 0 or len(biological_responses) == 0:
        print("Warning: No data found in the input files")
        # Create an empty figure
        fig = plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, "No data available", ha="center", va="center", fontsize=16)
        plt.axis("off")
    else:
        # Generate the unrolling plot
        fig = plot_unrolling(
            engineering_responses,
            biological_responses,
            args.t_feedforward,
            config=config,
            output_path=args.output,
        )

    # Save the plot
    save_plot(args.output)

    print(f"Unrolling plot saved to: {args.output}")
