import argparse
import re
from pathlib import Path
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from dynvision.utils import replace_param_in_string
from dynvision.utils.visualization_utils import save_plot, load_responses

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data", type=Path, required=True, help="Path to compiled CSV file"
)
parser.add_argument("--output", type=Path, required=True, help="Path to output file")
parser.add_argument("--parameter", type=str, required=True, help="Parameter to plot")
parser.add_argument("--experiment", type=str, required=True, help="Experiment to plot")
parser.add_argument("--category", type=str, default="rctype", help="Category to plot")
parser.add_argument(
    "--focus_layer", type=str, default="V4", help="Layer name to use as focus"
)


def determine_experiment_info(experiment):
    """Determine experiment type and corresponding labels from experiment name."""
    if "duration" in experiment.lower():
        return {
            "type": "duration",
            "label": "$D$",
            "full_label": "$D$ (duration)",
            "metric": "peak_height",
            "y_label": "Peak Power",
        }
    elif "contrast" in experiment.lower():
        return {
            "type": "contrast",
            "label": "$C$",
            "full_label": "$C$ (contrast)",
            "metric": "peak_time",
            "y_label": "Peak Time",
        }
    elif "interval" in experiment.lower():
        return {
            "type": "interval",
            "label": "$I$",
            "full_label": "$I$ (interval)",
            "metric": "peak_ratio",
            "y_label": "Peak Ratio",
        }
    else:
        # Default fallback
        return {
            "type": "unknown",
            "label": "$\\Delta_s$",
            "full_label": f"$\\Delta_s$ ({experiment})",
            "metric": "peak_height",
            "y_label": "Peak Height",
        }


def order_layers(layer_names):
    """Order layers according to visual hierarchy: IT, V4, V2, V1 (top to bottom)"""
    # Define the preferred order mapping
    layer_order = {
        "layer1": "V1",
        "layer2": "V2",
        "layer3": "V4",
        "layer4": "V4",
        "layer5": "IT",
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
    step_height = (y_max - y_min) * 0.1

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


def plot_unified_adaption(
    df,
    data_arg_key="stim",
    experiment="duration",
    focus_layer="V4",
    category="rctype",
):
    """
    Create unified figure with:
    - Upper: Different layers for middle key_value + empty space for image
    - Legend: Horizontal rctype legend between upper and lower
    - Lower: Focus layer power across different key_values + summary plot
    """

    print(f"Plot data shape: {df.shape}")
    print(f"Available columns: {df.columns.tolist()}")
    print(f"Unique {data_arg_key} values: {sorted(df[data_arg_key].unique())}")

    # Determine experiment information
    exp_info = determine_experiment_info(experiment)

    # Get unique values and layers (exclude classifier)
    key_values = sorted(df[data_arg_key].unique())
    all_layer_names = [
        col.replace("_power", "") for col in df.columns if col.endswith("_power")
    ]
    layer_names = order_layers(all_layer_names)

    print(f"Key values: {key_values}")
    print(f"Layer names: {layer_names}")
    print(f"Focus layer: {focus_layer}")
    print(f"Experiment info: {exp_info}")

    # Select middle key_value for upper subplot
    middle_key_value = key_values[len(key_values) // 2]

    # Define rctype plotting settings
    model_order = ["full", "self", "depthpointwise", "pointdepthwise"]
    colors = {
        "full": "#DAA520",  # More golden
        "self": "#228B22",  # Forest green
        "depthpointwise": "#4682B4",  # Steel blue
        "pointdepthwise": "#9932CC",  # Dark orchid
    }

    # Layer colors for formatting
    layer_colors = {
        "V1": "#ff69b4ff",
        "V2": "#dda0ddff",
        "V4": "#da70d6ff",
        "IT": "#ba55d3ff",
    }

    # Create figure
    fig = plt.figure(figsize=(15, 14))  # Made wider to accommodate increased spacing

    # Define section boundaries (reduced vertical spacing)
    upper_top = 0.92
    upper_bottom = 0.60  # Moved up from 0.60
    legend_top = 0.58  # Moved up from 0.56
    legend_bottom = 0.56  # Moved up from 0.52
    lower_top = 0.54  # Moved up from 0.48
    lower_bottom = 0.05

    # Calculate subplot parameters for ridgeplot effect
    n_upper = len(layer_names)
    n_lower = len(key_values)

    # Calculate dimensions for plots (increased separation and wider subplot D)
    left_plots_width = 0.50  # Slightly reduced to make room for increased gap
    right_plots_width = 0.27  # Increased by 50% from 0.18
    gap_width = 0.08  # Increased from 0.03 for more separation
    plots_start = 0.15  # Left margin

    # Add subplot labels positioned relative to y-label positions
    # A) Upper left (near y-label top)
    fig.text(0.06, upper_top - 0.02, "A)", fontsize=18, fontweight="bold")
    # B) Lower left (near y-label top)
    fig.text(0.06, lower_top - 0.02, "B)", fontsize=18, fontweight="bold")
    # C) Upper right (empty space)
    fig.text(
        plots_start + left_plots_width + gap_width - 0.02,
        upper_top - 0.02,
        "C)",
        fontsize=18,
        fontweight="bold",
    )
    # D) Lower right (summary plot)
    fig.text(
        plots_start + left_plots_width + gap_width - 0.02,
        lower_top - 0.02,
        "D)",
        fontsize=18,
        fontweight="bold",
    )

    # Upper section: Different layers for middle key_value
    fig.text(
        0.08,
        0.77,
        f"Avg Layer Power ({exp_info['label']} = {middle_key_value})",
        rotation=90,
        verticalalignment="center",
        fontsize=16,
        fontweight="bold",
    )

    subset_df = df[df[data_arg_key] == middle_key_value]
    print(f"Upper section subset shape: {subset_df.shape}")

    upper_height = upper_top - upper_bottom
    upper_spacing = upper_height / max(n_upper, 1) * 0.8
    upper_plot_height = upper_spacing * 1.2

    print(
        f"Upper section: {n_upper} plots, spacing: {upper_spacing}, height: {upper_plot_height}"
    )

    for i, layer_name in enumerate(layer_names):
        # Calculate position from top to bottom
        top_pos = upper_top - i * upper_spacing
        bottom_pos = top_pos - upper_plot_height

        print(
            f"Upper plot {i}: layer={layer_name}, top={top_pos}, bottom={bottom_pos}"
        )

        # Use left plots width to match lower section
        ax = fig.add_axes(
            [plots_start, bottom_pos, left_plots_width, upper_plot_height]
        )
        ax.patch.set_alpha(0)  # Transparent background

        # Add gray dotted y=0 line
        ax.axhline(0, color="gray", linestyle=":", alpha=0.7, linewidth=1)

        # Plot this layer's power
        power_col = f"{layer_name}_power"
        if power_col in subset_df.columns and len(subset_df) > 0:
            print(f"  Plotting {power_col}")
            sns.lineplot(
                data=subset_df,
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

        # Set y-ticks to only 0 and 1
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["0", "1"])

        # Formatting with label box positioned inside the plot area (right side, vertically centered)
        pad = 0.5 if layer_name == "IT" else 0.4
        ax.text(
            0.95,  # Position inside the plot area on the right
            0.5,  # Vertically centered
            layer_name.upper(),
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
            bbox=dict(
                boxstyle=f"circle,pad={pad}",
                facecolor=layer_colors[layer_name.upper()],
                edgecolor="#353535ff",
                linewidth=2,
                alpha=1,
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

    # Empty space for future image (upper right) - same size as summary plot
    # (This space is intentionally left empty for user to add image later)

    # Horizontal legend between upper and lower sections (only over left plots)
    legend_ax = fig.add_axes(
        [plots_start, legend_bottom, left_plots_width, legend_top - legend_bottom]
    )
    legend_ax.set_xlim(0, 1)
    legend_ax.set_ylim(0, 1)
    legend_ax.axis("off")
    legend_ax.patch.set_alpha(0)  # Transparent background

    # Create invisible plots just for the legend
    for i, (model, color) in enumerate(colors.items()):
        if model in model_order:
            legend_ax.plot([], [], color=color, label=model, linewidth=3, marker=".")

    legend = legend_ax.legend(
        loc="center",
        ncol=len(model_order),  # Horizontal layout
        frameon=False,
        fontsize=16,
        handlelength=2,
        handletextpad=0.5,
        # Removed title
    )

    # Lower section: Focus layer across different key_values
    fig.text(
        0.08,
        0.275,
        f"Layer {focus_layer.upper()} Power",
        rotation=90,
        verticalalignment="center",
        fontsize=16,
        fontweight="bold",
    )

    lower_height = lower_top - lower_bottom
    lower_spacing = lower_height / max(n_lower, 1) * 0.8
    lower_plot_height = lower_spacing * 1.2

    print(
        f"Lower section: {n_lower} plots, spacing: {lower_spacing}, height: {lower_plot_height}"
    )

    for i, key_value in enumerate(key_values):
        # Calculate position from top to bottom
        top_pos = lower_top - i * lower_spacing
        bottom_pos = top_pos - lower_plot_height

        print(
            f"Lower plot {i}: key_value={key_value}, top={top_pos}, bottom={bottom_pos}"
        )

        ax = fig.add_axes(
            [plots_start, bottom_pos, left_plots_width, lower_plot_height]
        )
        ax.patch.set_alpha(0)  # Transparent background

        # Filter data for this key_value
        subset_df = df[df[data_arg_key] == key_value]
        print(f"  Subset shape for {key_value}: {subset_df.shape}")

        # Add gray dotted y=0 line
        ax.axhline(0, color="gray", linestyle=":", alpha=0.7, linewidth=1)

        # Plot focus layer power
        power_col = f"{focus_layer}_power"
        if power_col in subset_df.columns and len(subset_df) > 0:
            print(f"  Plotting {power_col}")
            sns.lineplot(
                data=subset_df,
                x="times_index",
                y=power_col,
                ax=ax,
                marker=".",
                hue=category,
                hue_order=model_order,
                palette=colors,
                legend=False,
            )

            # Get y-axis range for relative step height
            y_min, y_max = ax.get_ylim()

            # Add step function for label indicator with relative height
            label_indicator_df = calculate_label_indicator(
                subset_df, category, (y_min, y_max)
            )
            ax.plot(
                label_indicator_df.times_index,
                label_indicator_df.label_indicator,
                color="gray",
                linewidth=3,
                drawstyle="steps-mid",
                alpha=0.8,
            )
        else:
            print(f"  Warning: No data for {power_col}")

        # Formatting with label box near bottom
        ax.text(
            0.95,
            0.15,
            f"{exp_info['label']}={key_value}",
            horizontalalignment="right",
            verticalalignment="bottom",
            transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

        ax.set_ylabel("")
        ax.set_xlabel("")

        # Only show x-axis labels on bottom subplot
        if i < len(key_values) - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("Time Index")

        sns.despine(ax=ax, left=True, bottom=True)

    # Summary statistics plot to the right of lower section (now 50% wider)
    summary_height = lower_height * 0.6  # 60% of lower section height
    summary_center = lower_bottom + lower_height / 2  # Center of lower section
    summary_bottom = summary_center - summary_height / 2  # Center the plot vertically
    summary_left = plots_start + left_plots_width + gap_width  # With increased gap

    summary_ax = fig.add_axes(
        [summary_left, summary_bottom, right_plots_width, summary_height]
    )
    summary_ax.patch.set_alpha(0)  # Transparent background

    # Plot the appropriate metric based on experiment type
    metric_col = f"{focus_layer}_{exp_info['metric']}"
    if exp_info["metric"] == "peak_ratio":
        # For peak ratio, we need to calculate it from the power data
        df[f"{focus_layer}_peak_ratio"] = 2 - df[f"{focus_layer}_peak_ratio"]
        summary_ax.axhline(1, color="gray", linestyle="--", linewidth=1, alpha=0.7)

    if metric_col in df.columns:
        sns.lineplot(
            data=df,
            x=data_arg_key,
            y=metric_col,
            ax=summary_ax,
            marker=".",
            hue=category,
            hue_order=model_order,
            palette=colors,
            legend=False,
        )
        summary_ax.set_title(f"Layer {focus_layer.upper()}", fontsize=14, pad=10)
        summary_ax.set_xlabel(exp_info["full_label"])
        summary_ax.set_ylabel(exp_info["y_label"])
        sns.despine(ax=summary_ax, left=True, bottom=True)

    return fig


if __name__ == "__main__":
    args, unknown = parser.parse_known_args()

    # Create output directory
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Load the compiled CSV data directly
    print(f"Loading compiled data from: {args.data}")
    df = pd.read_csv(args.data)

    # Extract layer names from columns
    layer_names = [
        col.replace("_power", "") for col in df.columns if col.endswith("_power")
    ]

    print(f"Loaded data shape: {df.shape}")
    print(f"Available columns: {df.columns.tolist()}")

    # Set plotting style
    sns.set_context("talk")

    print(f"Creating unified adaptation plot...")
    print(f"Available layers: {layer_names}")
    print(f"Focus layer set to: {args.focus_layer}")
    print(f"Parameter: {args.parameter}")

    # Generate the unified figure
    fig = plot_unified_adaption(
        df,
        data_arg_key=args.parameter,
        focus_layer=args.focus_layer,
        category=args.category,
        experiment=args.experiment,
    )

    # Save the plot
    save_plot(args.output)

    print(f"Unified plot saved to: {args.output}")
