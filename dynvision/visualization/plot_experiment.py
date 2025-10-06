import argparse
import re
from pathlib import Path
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import numpy as np
import pandas as pd

from dynvision.utils import replace_param_in_string
from dynvision.utils.visualization_utils import save_plot

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=Path, required=True, help="Path to pt files")
parser.add_argument("--output", type=Path, required=True, help="Path to directory")
parser.add_argument("--parameter", type=str, required=True, help="Parameter to plot")
parser.add_argument("--category", type=str, required=True, help="Category to plot")
parser.add_argument(
    "--focus_layer", type=str, default="V4", help="Layer name to use as V4"
)


def order_layers(layer_names):
    """Order layers according to visual hierarchy: IT, V4, V2, V1 (top to bottom)"""
    # Define the preferred order mapping
    layer_order = {
        "layer1": "V1",
        "layer2": "V2",
        "layer3": "V4",
        "layer4": "IT",
        "classifier": "classifier",  # Will be filtered out
    }

    # Filter out classifier and sort by hierarchy (IT at top, V1 at bottom)
    hierarchy_order = ["IT", "V4", "V2", "V1"]

    # Filter out classifier layers
    filtered_layers = [
        layer for layer in layer_names if "classifier" not in layer.lower()
    ]

    # Sort layers according to hierarchy
    def get_sort_key(layer_name):
        mapped_name = layer_order.get(layer_name, layer_name)
        if mapped_name in hierarchy_order:
            return hierarchy_order.index(mapped_name)
        else:
            return len(hierarchy_order)  # Put unknown layers at the end

    ordered_layers = sorted(filtered_layers, key=get_sort_key)

    return ordered_layers


def calculate_label_indicator(
    df: pd.DataFrame, category: str, y_range: tuple
) -> pd.DataFrame:
    """
    Calculate label indicator (step function) at each time step.

    Args:
        df: DataFrame containing classifier responses
        category: Category column name
        y_range: Tuple of (y_min, y_max) for the current subplot

    Returns:
        DataFrame with times_index and label_indicator (relative to subplot height)
    """
    # Get first model's data to determine label validity at each time step
    first_model = df[category].iloc[0]
    model_data = df[df[category] == first_model]

    # Calculate step height as 10% of the y-axis range
    y_min, y_max = y_range
    step_height = (y_max - y_min) * 0.25

    indicator_data = []
    for time_step in sorted(model_data.times_index.unique()):
        time_data = model_data[model_data.times_index == time_step]

        # Check if any labels are valid (>= 0) at this time step
        valid_labels = (time_data.label_index >= 0).any()
        indicator_value = y_min + step_height if valid_labels else y_min

        indicator_data.append(
            {"times_index": time_step, "label_indicator": indicator_value}
        )

    return pd.DataFrame(indicator_data)


def get_category_plotting_settings(category, category_values=None):
    """
    Generalize plotting settings for different categories.

    Parameters:
    -----------
    category : str
        The category name
    category_values : list, optional
        List of unique category values. Required if category != 'rctype'.

    Returns:
    --------
    tuple: (order_list, colors_dict)
        - order_list: List of category values in the desired order
        - colors_dict: Dictionary mapping category values to hex colors
    """

    if category == "rctype":
        # Use predefined rctype settings
        rctype_model_order = ["full", "self", "depthpointwise", "pointdepthwise"]
        rctype_colors = {
            "full": "#DAA520",  # More golden
            "self": "#228B22",  # Forest green
            "depthpointwise": "#4682B4",  # Steel blue
            "pointdepthwise": "#9932CC",  # Dark orchid
        }
        return rctype_model_order, rctype_colors

    else:
        # Create settings for other categories
        if category_values is None:
            raise ValueError(
                "category_values must be provided for non-rctype categories"
            )

        unique_values = list(set(category_values))

        # Try to order by float values
        try:
            # Attempt to convert to float and sort
            float_values = [(float(val), val) for val in unique_values]
            float_values.sort(key=lambda x: x[0])
            ordered_values = [val for _, val in float_values]

            # Use sequential palette for numeric data
            colors = plt.cm.viridis_r(np.linspace(0, 1, len(ordered_values)))
            colors_dict = {
                val: mcolors.rgb2hex(color)
                for val, color in zip(ordered_values, colors)
            }

        except (ValueError, TypeError):
            # Can't convert to float, use alphabetical order
            ordered_values = sorted(unique_values)

            # Use categorical palette for non-numeric data
            if len(ordered_values) <= 10:
                colors = plt.cm.tab10(np.arange(len(ordered_values)))
            else:
                # Use larger palette for more categories
                colors = plt.cm.tab20(np.arange(len(ordered_values)) % 20)

            colors_dict = {
                val: mcolors.rgb2hex(color)
                for val, color in zip(ordered_values, colors)
            }

        return ordered_values, colors_dict


def plot_unified_adaption(
    df,
    data_arg_key="contrast",
    focus_layer="V4",
    category="rctype",
    output_path=None,
    experiment="duration",
):
    """
    Create unified figure with:
    - Upper: Different layers for middle key_value (switched from lower)
    - Middle: Peak height and peak time plots + legend (optional based on output path)
    - Lower: V4 layer power across different key_values (switched from upper)
    """

    print(f"Plot data shape: {df.shape}")
    print(f"Available columns: {df.columns.tolist()}")
    print(f"Unique {data_arg_key} values: {sorted(df[data_arg_key].unique())}")
    print(f"Labels: {df['label_index'].unique()}")
    # Print unique labels per timestep
    print("Unique labels per timestep:")
    for time_step in sorted(df["times_index"].unique()):
        time_data = df[df["times_index"] == time_step]
        unique_labels = sorted(time_data["label_index"].unique())
        print(f"  Time {time_step}: {unique_labels}")

    # Check if we should skip middle section
    skip_middle = output_path and "response" in str(output_path).lower()
    print(f"Skip middle section: {skip_middle}")

    # Get unique values and layers (exclude classifier)
    key_values = sorted(df[data_arg_key].unique())
    all_layer_names = [
        col.replace("_power", "") for col in df.columns if col.endswith("_power")
    ]
    layer_names = order_layers(all_layer_names)

    print(f"Key values: {key_values}")
    print(f"Layer names: {layer_names}")

    # Define rctype plotting settings
    category_values = sorted(df[category].unique())
    model_order, colors = get_category_plotting_settings(category, category_values)

    # Create figure
    fig = plt.figure(figsize=(8, 7))

    # Calculate subplot parameters for ridgeplot effect
    n_upper = len(layer_names)  # Switched: now using layer_names

    print(f"Upper section subset shape: {df.shape}")

    upper_spacing = 1 / max(n_upper, 1) * 0.8
    upper_plot_height = upper_spacing * 1.3

    print(
        f"Upper section: {n_upper} plots, spacing: {upper_spacing}, height: {upper_plot_height}"
    )

    # Store axes for y-axis sharing
    upper_axes = []

    for i, layer_name in enumerate(layer_names):
        # Calculate position from top to bottom
        top_pos = 1 - i * upper_spacing
        bottom_pos = top_pos - upper_plot_height

        print(
            f"Upper plot {i}: layer={layer_name}, top={top_pos}, bottom={bottom_pos}"
        )

        ax = fig.add_axes([0.15, bottom_pos, 0.75, upper_plot_height])
        ax.patch.set_alpha(0)  # Transparent background
        upper_axes.append(ax)

        # Add gray dotted y=0 line
        ax.axhline(0, color="gray", linestyle=":", alpha=0.7, linewidth=1)

        # Plot this layer's power
        power_col = f"{layer_name}_power"

        if power_col in df.columns and len(df) > 0:
            print(f"  Plotting {power_col}")
            sns.lineplot(
                data=df,
                x="times_index",
                y=power_col,
                ax=ax,
                marker=".",
                hue=category,
                hue_order=model_order,
                palette=colors,
                legend=False,
            )
        else:
            print(f"  Warning: No data for {power_col}")

        # Add step function for label indicator to the bottom subplot
        if i == len(layer_names) - 1:  # Bottom subplot
            y_min, y_max = ax.get_ylim()
            label_indicator_df = calculate_label_indicator(
                df, category, (y_min, y_max)
            )
            ax.plot(
                label_indicator_df.times_index,
                label_indicator_df.label_indicator,
                color="gray",
                linewidth=3,
                drawstyle="steps-mid",
                alpha=0.8,
            )

        ax.set_yticklabels([])

        # Formatting with label box positioned outside the plot area
        layer_colors = {
            "V1": "#ff69b4ff",
            "V2": "#dda0ddff",
            "V4": "#da70d6ff",
            "IT": "#ba55d3ff",
        }
        pad = 0.5 if layer_name == "IT" else 0.4
        ax.text(
            1.02,  # Position outside the plot area
            0.4,  # Position near bottom
            layer_name.upper(),
            horizontalalignment="left",  # Changed to left since it's outside
            verticalalignment="bottom",
            transform=ax.transAxes,
            va="center",
            bbox=dict(
                boxstyle=f"circle,pad={pad}",
                facecolor=layer_colors[layer_name.upper()],
                edgecolor="#353535ff",
                linewidth=2,
                alpha=0.8,
            ),
        )

        ax.set_ylabel("")
        ax.set_xlabel("")

        # Only show x-axis labels on bottom subplot
        if i < len(layer_names) - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("Time Index")

        sns.despine(ax=ax, left=True, bottom=True)

    # Share y-axis for upper section
    if upper_axes:
        # Get all y-limits
        all_y_limits = [ax.get_ylim() for ax in upper_axes]
        global_y_min = min(lim[0] for lim in all_y_limits)
        global_y_max = max(lim[1] for lim in all_y_limits)

        # Set same y-limits for all upper axes
        for ax in upper_axes:
            ax.set_ylim(global_y_min, global_y_max)

    # Legend as third component
    legend_ax = fig.add_axes([0.15, bottom_pos, 0.74, 0.8])
    legend_ax.set_xlim(0, 1)
    legend_ax.set_ylim(0, 1)
    # legend_ax.axis("off")
    legend_ax.set_ylabel(
        f"Avg Layer Power ($\\Delta_s = {key_values[0]}$)", labelpad=4
    )
    legend_ax.set_xticks([])
    legend_ax.set_yticks([])
    sns.despine(ax=legend_ax, left=True, bottom=True)
    legend_ax.patch.set_alpha(0)  # Transparent background

    # Create invisible plots just for the legend
    for i, (model, color) in enumerate(colors.items()):
        if model in model_order:
            legend_ax.plot([], [], color=color, label=model, linewidth=3, marker=".")

    legend = legend_ax.legend(
        loc="lower right",
        title=category,
        frameon=True,
        fontsize=16,
        handlelength=2,
        handletextpad=0.5,
    )
    frame = legend.get_frame()
    frame.set_facecolor("white")
    frame.set_edgecolor("white")
    frame.set_alpha(1)
    return fig


if __name__ == "__main__":
    args, unknown = parser.parse_known_args()

    print(f"Loading processed data from: {args.data}")
    df = pd.read_csv(args.data)

    # Set plotting style
    sns.set_context("talk")

    # Generate the unified figure
    fig = plot_unified_adaption(
        df,
        data_arg_key=args.parameter,
        focus_layer=args.focus_layer,
        category=args.category,
        output_path=args.output,  # Pass output path to check for 'response'
        experiment=args.data.parent.name,
    )

    # Save the plot
    save_plot(args.output)

    print(f"Unified plot saved to: {args.output}")
