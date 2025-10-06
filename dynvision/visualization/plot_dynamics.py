"""plot_dynamics.py
Unified plot showing layer dynamics across different parameter values with consistent layout.

This script creates a 4-panel visualization of neural network layer responses:
    - Panel A (upper left): Different layers for a reference parameter value
    - Panel B (lower left): Focus layer across different parameter values
    - Panel C (upper right): Layer relationship visualization
    - Panel D (lower right): Summary plot showing parameter effect on metrics

Example:
    ```bash
    python plot_dynamics.py --data responses.csv --output dynamics_plot.png --parameter tau \
        --category interval --focus-layer V1 --dt 1.0
    ```

Attributes:
    df (pandas.DataFrame): Input data containing layer responses and experiment parameters
    focus_layer (str): The layer to focus on for parameter variation plots (e.g., "V1")
    parameter (str): Column name with parameter values (e.g., "tau", "interval")
    category (str): Column name with category values (e.g., "interval", "duration")
    save_path (str, optional): Path to save the output figure
    dt (float, optional): Time step in milliseconds for x-axis conversion
    config (dict, optional): Configuration dictionary with colors, names, and ordering
"""

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from dynvision.utils.visualization_utils import (
    save_plot,
    load_config_from_args,
    get_display_name,
    get_color,
    calculate_label_indicator,
    get_category_plotting_settings,
    order_layers,
    peak_ratio,
    peak_time,
    peak_height,
)

# Set up logging with stream handler to ensure console output
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # Add explicit StreamHandler for console output
        logging.FileHandler("plot_dynamics.log"),  # Also log to file
    ],
)
logger = logging.getLogger(__name__)

# Style constants
STYLE = {
    "fontsize_title": 16,
    "fontsize_labels": 14,
    "fontsize_tick_labels": 12,
    "fontsize_legend": 12,
    "fontsize_panel_labels": 16,
    "fontsize_annotations": 12,
    "linewidth_main": 2.5,  # Increased for better visibility
    "linewidth_indicator": 2.0,
    "alpha_lines": 0.85,  # Slightly increased for better visibility
    "alpha_indicator": 0.6,  # Increased for better visibility
    "grid_linewidth": 0.5,
    "grid_alpha": 0.3,
    "badge_alpha": 0.85,
    "badge_linewidth": 1.5,
    "badge_pad_normal": 0.4,
    "badge_pad_small": 0.55,
    "marker_size": 8,
    "marker_edge_width": 1.2,
}

# Default colors for layers if not provided in config
DEFAULT_LAYER_COLORS = {
    "v1": "#ff69b4ff",  # hot pink
    "V1": "#ff69b4ff",
    "v2": "#dda0ddff",  # plum
    "V2": "#dda0ddff",
    "v4": "#da70d6ff",  # orchid
    "V4": "#da70d6ff",
    "it": "#ba55d3ff",  # medium orchid
    "IT": "#ba55d3ff",
}

# Layout parameters for the figure
LAYOUT = {
    "figure_width": 15,
    "figure_height": 10,
    "plots_start": 0.15,  # Starting x position for plots
    "left_plots_width": 0.5,
    "right_plots_width": 0.25,
    "gap_width": 0.05,
    "subplot_overlap": 0.88,  # Adjusted for better spacing (was 0.9)
    # Vertical positions
    "upper_top": 0.9,
    "upper_bottom": 0.6,
    "legend_top": 0.58,
    "legend_bottom": 0.53,
    "lower_top": 0.48,
    "lower_bottom": 0.15,
    # Positioning helpers
    "y_label_x_position": 0.03,
    "panel_label_x_offset": 0.06,
    "panel_label_y_offset": 0.02,
    "title_padding": 10,
}


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


def format_parameter_value(value, parameter, dt=None):
    """Format parameter values with appropriate units."""
    # Convert time parameters to milliseconds
    if dt is not None and parameter.lower() in [
        "tau",
        "time_constant",
        "duration",
        "interval",
        "stimulus",
        "stim",
        "lag",
    ]:
        return f"{int(float(value) * dt)} ms"
    # Format non-time parameters appropriately
    elif parameter.lower() in ["contrast", "amplitude"]:
        return f"{float(value):.2f}"
    elif parameter.lower() in ["frequency", "freq"]:
        return f"{float(value):.1f} Hz"
    else:
        return f"{value}"


def add_panel_labels(fig):
    """Add panel labels A), B), C), D) to the figure."""
    # Extract layout values for positioning
    panel_x_offset = LAYOUT["panel_label_x_offset"]
    panel_y_offset = LAYOUT["panel_label_y_offset"]
    plots_start = LAYOUT["plots_start"]
    left_width = LAYOUT["left_plots_width"]
    gap_width = LAYOUT["gap_width"]
    upper_top = LAYOUT["upper_top"]
    lower_top = LAYOUT["lower_top"]

    # Panel label styling
    fontsize = STYLE["fontsize_panel_labels"]
    badge_style = dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)

    # Left side panel labels (A and B)
    fig.text(
        x=panel_x_offset,
        y=upper_top - panel_y_offset,
        s="A)",
        fontsize=fontsize,
        fontweight="bold",
        ha="left",
        va="top",
        bbox=badge_style,
    )
    fig.text(
        x=panel_x_offset,
        y=lower_top - panel_y_offset,
        s="B)",
        fontsize=fontsize,
        fontweight="bold",
        ha="left",
        va="top",
        bbox=badge_style,
    )

    # Right side panel labels (C and D)
    right_x = plots_start + left_width + gap_width - 0.02
    fig.text(
        x=right_x,
        y=upper_top - panel_y_offset,
        s="C)",
        fontsize=fontsize,
        fontweight="bold",
        ha="left",
        va="top",
        bbox=badge_style,
    )
    fig.text(
        x=right_x,
        y=lower_top - panel_y_offset,
        s="D)",
        fontsize=fontsize,
        fontweight="bold",
        ha="left",
        va="top",
        bbox=badge_style,
    )


def create_horizontal_legend(fig, colors, model_order, category, config=None):
    """Create a horizontal legend between upper and lower sections."""
    # Extract layout parameters
    plots_start = LAYOUT["plots_start"]
    left_width = LAYOUT["left_plots_width"]
    right_width = LAYOUT["right_plots_width"]
    legend_top = LAYOUT["legend_top"]
    legend_bottom = LAYOUT["legend_bottom"]

    # Create axes for legend
    legend_ax = fig.add_axes(
        rect=[
            plots_start,
            legend_bottom,
            left_width + right_width,
            legend_top - legend_bottom,
        ]
    )
    legend_ax.set_xlim(0, 1)
    legend_ax.set_ylim(0, 1)
    legend_ax.axis("off")
    legend_ax.patch.set_alpha(0)  # Transparent background

    # Create invisible plots for the legend with improved formatting
    for model in model_order:
        color = colors.get(model, "#cccccc")

        # Format display name with consistent units
        if isinstance(model, (int, float)):
            if parameter.lower() in ["tau", "interval", "duration", "time"]:
                display_name = f"{model} ms" if dt is None else f"{int(model * dt)} ms"
            else:
                display_name = f"{model}"
        else:
            display_name = get_display_name(model, config) or model

        legend_ax.plot(
            [],
            [],
            color=color,
            label=display_name,
            linewidth=STYLE["linewidth_main"],
            marker=".",
            markersize=STYLE["marker_size"],
            alpha=STYLE["alpha_lines"],
        )

    # Create and customize the legend
    legend = legend_ax.legend(
        loc="center",
        ncol=min(len(model_order), 6),  # Horizontal layout, max 6 columns
        frameon=False,
        framealpha=0.8,
        fontsize=STYLE["fontsize_legend"],
        handlelength=2.5,
        handletextpad=0.8,
        columnspacing=1.5,  # Increased spacing between columns
    )

    return legend_ax


def calculate_metrics(df, focus_layer, parameter, category, metric, dt=None):
    """Calculate metrics (peak ratio, peak time, peak height) for summary plot."""
    df = df.copy()
    metric_col = f"{focus_layer}_{metric}"

    # Return if metric already exists
    if metric_col in df.columns:
        return df

    # Create new column for the metric
    df[metric_col] = np.nan

    # Group by parameter and category
    for group_key, group in df.groupby([parameter, category]):
        param_val, cat_val = group_key
        logger.info(f"Processing group: {param_val}, {cat_val}")

        if len(group) < 5:  # Need enough points for meaningful calculation
            logger.info(f"Skipping group with only {len(group)} points")
            continue

        # Get response column for focus layer
        response_col = f"{focus_layer}_response_avg"
        if response_col not in group.columns:
            logger.info(f"Skipping group: {response_col} not in columns")
            continue

        # Get response data
        response_values = group[response_col].values
        if len(response_values) == 0:
            continue

        # Convert to tensor and reshape for processing
        try:
            response_tensor = torch.tensor(response_values).reshape(1, -1)

            # Calculate metric based on type
            if metric == "peak_ratio":
                metric_value = float(peak_ratio(response_tensor, min_delay=3)[0])
                logger.info(f"Calculated peak ratio: {metric_value}")
            elif metric == "peak_time":
                peak_idx = int(peak_time(response_tensor)[0].item())
                metric_value = peak_idx * dt if dt is not None else peak_idx
            elif metric == "peak_height":
                metric_value = float(peak_height(response_tensor)[0].item())
            else:
                logger.info(f"Unknown metric: {metric}")
                continue

            # Update DataFrame with calculated metric
            for idx in group.index:
                df.at[idx, metric_col] = metric_value

        except Exception as e:
            logger.error(f"Error calculating {metric}: {e}")

    return df


def get_improved_color_palette(category_values, config=None, parameter=None):
    """Generate a perceptually uniform sequential color palette using viridis.

    This function adapts techniques from plot_response_triptych to ensure consistent
    color handling across visualizations. It prioritizes using colors defined in config.
    """
    # First check if all values are numeric (can be converted to float)
    are_numeric = True
    original_values = list(category_values)  # Keep original values
    numeric_values = []

    # Print diagnostic info
    logger.info(
        f"Generating color palette for {len(category_values)} values: {category_values}"
    )
    logger.info(
        f"Config palette has {len(config['palette']) if config and 'palette' in config else 0} entries"
    )

    for val in category_values:
        try:
            numeric_values.append(float(val))
        except (ValueError, TypeError):
            are_numeric = False
            break

    # Convert all category values to strings for consistent comparison
    category_values = [str(val) for val in category_values]

    # Try to get colors from config first
    if config and "palette" in config:
        colors = {}
        for val in category_values:
            color = get_color(val, config)
            if color:
                logger.info(f"Found color in config for '{val}': {color}")
                colors[val] = color
            else:
                logger.info(f"No color in config for '{val}'")

        if colors and len(colors) == len(category_values):
            # If values are numeric, sort numerically
            if are_numeric:
                # Sort by numeric value but keep as strings
                model_order = [str(x) for x in sorted(numeric_values)]
            else:
                model_order = sorted(category_values)
            logger.info(f"Using colors from config with model order: {model_order}")
            return model_order, colors

    # Use perceptually uniform sequential palette based on experiment type
    # If values are numeric, sort numerically
    if are_numeric:
        # Create mapping from numeric value to string representation
        value_to_str = {float(val): str(val) for val in original_values}
        # Sort numerically but return string representations in order
        model_order = [value_to_str[val] for val in sorted(numeric_values)]
    else:
        model_order = sorted(category_values)

    # Choose color map based on parameter
    if parameter and parameter.lower() in ["interval", "tau"]:
        # Use viridis for time-related parameters (blue to yellow)
        cmap = plt.cm.viridis
    elif parameter and parameter.lower() in ["contrast"]:
        # Use plasma for contrast (purple to yellow)
        cmap = plt.cm.plasma
    else:
        # Default to tab10 for categorical data or unknown parameters
        cmap = plt.cm.tab10

    # Generate colors
    if len(model_order) <= 10:
        color_values = cmap(np.linspace(0, 1, len(model_order)))
    else:
        # For many values, use full cycle of viridis
        color_values = cmap(np.linspace(0, 1, len(model_order)))

    hex_colors = [plt.matplotlib.colors.to_hex(color) for color in color_values]
    colors = dict(zip(model_order, hex_colors))

    logger.info(f"Generated color palette with model order: {model_order}")

    return model_order, colors


def plot_upper_layers_panel(
    fig,
    df,
    layer_names,
    parameter,
    category,
    param_value,
    model_order,
    colors,
    dt=None,
    config=None,
):
    """Create panel A showing different layers for a reference parameter value."""
    # Extract layout parameters
    upper_top = LAYOUT["upper_top"]
    upper_bottom = LAYOUT["upper_bottom"]
    plots_start = LAYOUT["plots_start"]
    left_width = LAYOUT["left_plots_width"]

    # Create vertical axis label
    exp_info = determine_experiment_info(parameter)
    param_display = format_parameter_value(param_value, parameter, dt)

    fig.text(
        x=LAYOUT["y_label_x_position"],
        y=(upper_top + upper_bottom) / 2,
        s=f"Avg Layer Response ({exp_info['label']} = {param_display})",
        rotation=90,
        verticalalignment="center",
        fontsize=STYLE["fontsize_labels"],
        fontweight="bold",
    )

    # Calculate subplot layout
    n_layers = len(layer_names)
    upper_height = upper_top - upper_bottom
    upper_spacing = upper_height / max(n_layers, 1) * LAYOUT["subplot_overlap"]
    upper_plot_height = upper_spacing * 1.2

    # Filter data for selected parameter value
    subset_df = df[df[parameter] == param_value].copy()
    logger.info(f"Upper section subset shape: {subset_df.shape}")

    # Setup time column
    time_col = (
        "time_ms"
        if dt is not None and "time_ms" in subset_df.columns
        else "times_index"
    )

    axes = []
    # Plot each layer
    for i, layer_name in enumerate(layer_names):
        # Calculate position from top to bottom
        top_pos = upper_top - i * upper_spacing
        bottom_pos = top_pos - upper_plot_height

        logger.info(
            f"Upper plot {i}: layer={layer_name}, top={top_pos}, bottom={bottom_pos}"
        )

        # Create subplot
        ax = fig.add_axes(
            rect=[plots_start, bottom_pos, left_width, upper_plot_height]
        )
        ax.patch.set_alpha(0)  # Transparent background
        axes.append(ax)

        # Add reference line at y=0
        ax.axhline(y=0, color="gray", linestyle=":", alpha=0.7, linewidth=1)

        # Plot layer response
        response_col = f"{layer_name}_response_avg"
        if response_col in subset_df.columns and len(subset_df) > 0:
            # Ensure model_order values are strings for consistent comparison
            string_model_order = [str(val) for val in model_order]
            # Ensure category values are strings
            subset_df[category] = subset_df[category].astype(str)

            sns.lineplot(
                data=subset_df,
                x=time_col,
                y=response_col,
                ax=ax,
                marker=".",
                markersize=3,  # Smaller markers for cleaner look
                hue=category,
                hue_order=string_model_order,
                palette=colors,
                legend=False,
                linewidth=STYLE["linewidth_main"],
                alpha=STYLE["alpha_lines"],
            )

            # Add grid to each subplot
            ax.grid(
                True,
                which="major",
                axis="both",
                linestyle=":",
                linewidth=STYLE["grid_linewidth"],
                alpha=STYLE["grid_alpha"],
            )

        else:
            logger.warning(f"No data for {response_col}")

        # Set y-ticks consistently
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["0", "1"])

        # Add layer badge
        layer_display = get_display_name(layer_name, config) or layer_name.upper()
        pad = (
            STYLE["badge_pad_small"]
            if layer_display == "IT"
            else STYLE["badge_pad_normal"]
        )

        # Get layer color from config
        layer_color = DEFAULT_LAYER_COLORS.get(layer_name, "#cccccc")
        if config and "palette" in config:
            layer_color = config["palette"].get(layer_name.upper(), layer_color)
            if not layer_color:
                layer_color = config["palette"].get(
                    layer_name.lower(), DEFAULT_LAYER_COLORS.get(layer_name, "#cccccc")
                )

        # Add circular badge with layer name
        ax.text(
            x=0.95,  # Position inside the plot area on the right
            y=0.5,  # Vertically centered
            s=layer_display,
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
            fontsize=STYLE["fontsize_panel_labels"],
            fontweight="bold",
            bbox=dict(
                boxstyle=f"circle,pad={pad}",
                facecolor=layer_color,
                edgecolor="#353535ff",
                linewidth=STYLE["badge_linewidth"],
                alpha=STYLE["badge_alpha"],
            ),
        )

        # Format axes
        ax.set_ylabel("")
        ax.set_xlabel("")

        # Only show x-axis labels on bottom subplot
        if i < len(layer_names) - 1:
            ax.set_xticklabels([])
        else:
            x_label = "Time (ms)" if dt is not None else "Time Step"
            ax.set_xlabel(x_label, fontsize=STYLE["fontsize_labels"])

        # Clean up spines
        sns.despine(ax=ax, left=True, bottom=True)

    return axes


def plot_parameter_variation_panel(
    fig,
    df,
    focus_layer,
    parameter,
    category,
    param_values,
    model_order,
    colors,
    dt=None,
    config=None,
):
    """Create panel B showing focus layer across different parameter values.

    This function has been refactored to match the style of plot_response_triptych.
    """
    # Extract layout parameters
    lower_top = LAYOUT["lower_top"]
    lower_bottom = LAYOUT["lower_bottom"]
    plots_start = LAYOUT["plots_start"]
    left_width = LAYOUT["left_plots_width"]

    # Add vertical axis label
    layer_display = get_display_name(focus_layer, config) or focus_layer.upper()
    fig.text(
        x=LAYOUT["y_label_x_position"],
        y=(lower_top + lower_bottom) / 2,
        s=f"Layer {layer_display} Response",
        rotation=90,
        verticalalignment="center",
        fontsize=STYLE["fontsize_labels"],
        fontweight="bold",
    )

    # Calculate subplot layout
    n_params = len(param_values)
    lower_height = lower_top - lower_bottom
    lower_spacing = lower_height / max(n_params, 1) * LAYOUT["subplot_overlap"]
    lower_plot_height = lower_spacing * 1.2

    # Create a time column in ms for proper x-axis scaling
    df_with_time = df.copy()
    if dt is not None:
        df_with_time["time_ms"] = df_with_time["times_index"] * dt

    # Ensure category values are strings for consistent comparison
    df_with_time[category] = df_with_time[category].astype(str)

    # Ensure model_order contains strings
    string_model_order = [str(val) for val in model_order]

    # Calculate global y-axis range for consistent scaling
    global_ymin, global_ymax = float("inf"), float("-inf")
    for param_value in param_values:
        subset_df = df_with_time[df_with_time[parameter] == param_value].copy()
        response_col = f"{focus_layer}_response_avg"

        if response_col in subset_df.columns and len(subset_df) > 0:
            ymin = subset_df[response_col].min()
            ymax = subset_df[response_col].max()
            global_ymin = min(global_ymin, ymin)
            global_ymax = max(global_ymax, ymax)

    # Add padding and ensure y=0 is included
    if global_ymin != float("inf") and global_ymax != float("-inf"):
        padding = (global_ymax - global_ymin) * 0.1
        global_ymin = min(global_ymin - padding, -0.05)  # Use -0.05 as in triptych
        global_ymax = global_ymax + padding

    axes = []
    # Plot each parameter value
    for i, param_value in enumerate(param_values):
        # Calculate position
        top_pos = lower_top - i * lower_spacing
        bottom_pos = top_pos - lower_plot_height

        logger.info(
            f"Lower plot {i}: param_value={param_value}, top={top_pos}, bottom={bottom_pos}"
        )

        # Create subplot
        ax = fig.add_axes(
            rect=[plots_start, bottom_pos, left_width, lower_plot_height]
        )
        ax.patch.set_alpha(0)  # Transparent background
        axes.append(ax)

        # Filter data for this parameter value
        subset_df = df_with_time[df_with_time[parameter] == param_value].copy()
        logger.info(f"Subset shape for {param_value}: {subset_df.shape}")

        # Set up time column
        time_col = (
            "time_ms"
            if dt is not None and "time_ms" in subset_df.columns
            else "times_index"
        )

        # Add reference line at y=0
        ax.axhline(y=0, color="gray", linestyle=":", alpha=0.7, linewidth=1)

        # Plot focus layer response for each category
        response_col = f"{focus_layer}_response_avg"
        if response_col in subset_df.columns:
            sns.lineplot(
                data=subset_df,
                x=time_col,
                y=response_col,
                ax=ax,
                marker=".",
                markersize=3,  # Smaller markers for cleaner look
                hue=category,
                hue_order=string_model_order,
                palette=colors,
                legend=False,
                linewidth=STYLE["linewidth_main"],
                alpha=STYLE["alpha_lines"],
            )

            # Add label indicator if available - styled like in triptych
            if "label_index" in subset_df.columns:
                # Calculate label indicator - use 10% of the plot height for indicator
                label_indicator_df = calculate_label_indicator(
                    df=subset_df,
                    category=category,
                    y_range=(global_ymin, global_ymax),
                    step_height=0.1,
                )

                # Convert time for indicator if needed
                if dt is not None and time_col == "time_ms":
                    indicator_time = label_indicator_df.times_index * dt
                else:
                    indicator_time = label_indicator_df.times_index

                # Plot indicator with improved styling
                ax.plot(
                    indicator_time,
                    label_indicator_df.label_indicator,
                    color="dimgray",  # Darker for better visibility (as in triptych)
                    linewidth=STYLE["linewidth_indicator"],
                    drawstyle="steps-mid",
                    alpha=STYLE["alpha_indicator"],
                )
        else:
            logger.warning(f"{response_col} not found in data for param={param_value}")

        # Format axes and add labels
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(
            axis="both", which="major", labelsize=STYLE["fontsize_tick_labels"]
        )

        # Y-axis label for leftmost plot only
        if i == 0:
            ax.set_ylabel("Activation", fontsize=STYLE["fontsize_labels"])

        # Parameter value label - improved positioning
        param_display = format_parameter_value(param_value, parameter, dt)
        ax.text(
            x=0.02,  # Left side
            y=0.92,  # Near the top
            s=param_display,
            transform=ax.transAxes,
            fontsize=STYLE["fontsize_annotations"],
            fontweight="bold",
        )

        # X-axis label for bottom plot only
        if i == len(param_values) - 1:
            x_label = "Time (ms)" if dt is not None else "Time Step"
            ax.set_xlabel(x_label, fontsize=STYLE["fontsize_labels"])

        # Apply consistent y-axis limits
        if global_ymin != float("inf") and global_ymax != float("-inf"):
            ax.set_ylim(global_ymin, global_ymax)

        # Format time axis if dt is provided
        if dt is not None and time_col == "time_ms":
            # Round to nearest multiple of 50ms for cleaner ticks
            x_min, x_max = ax.get_xlim()
            x_ticks = np.arange(int(x_min // 50) * 50, int(x_max // 50 + 1) * 50, 50)
            ax.set_xticks(x_ticks)
            ax.set_xticklabels([f"{x:.0f}" for x in x_ticks])

        # Add grid lines
        ax.grid(
            True,
            which="major",
            linestyle=":",
            linewidth=STYLE["grid_linewidth"],
            alpha=STYLE["grid_alpha"],
            axis="both",
        )

    return axes


def create_empty_panel(fig, title=None, config=None):
    """Create an empty panel C with optional title."""
    # Extract layout parameters
    upper_top = LAYOUT["upper_top"]
    upper_bottom = LAYOUT["upper_bottom"]
    plots_start = LAYOUT["plots_start"]
    left_width = LAYOUT["left_plots_width"]
    gap_width = LAYOUT["gap_width"]
    right_width = LAYOUT["right_plots_width"]

    # Create panel
    panel_height = upper_top - upper_bottom
    ax = fig.add_axes(
        rect=[
            plots_start + left_width + gap_width,
            upper_bottom,
            right_width,
            panel_height,
        ]
    )
    ax.patch.set_alpha(0)  # Transparent background

    # Add title if provided
    if title:
        ax.set_title(
            label=title, fontsize=STYLE["fontsize_title"], pad=LAYOUT["title_padding"]
        )

    # Make it empty
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    return ax


def plot_summary_panel(
    fig,
    df,
    focus_layer,
    parameter,
    category,
    exp_info,
    model_order,
    colors,
    dt=None,
    config=None,
):
    """Create panel D showing parameter effect on metrics."""
    # Extract layout parameters
    lower_top = LAYOUT["lower_top"]
    lower_bottom = LAYOUT["lower_bottom"]
    plots_start = LAYOUT["plots_start"]
    left_width = LAYOUT["left_plots_width"]
    gap_width = LAYOUT["gap_width"]
    right_width = LAYOUT["right_plots_width"]

    # Calculate panel dimensions
    summary_height = (lower_top - lower_bottom) * 0.6  # 60% of lower section height
    summary_center = (
        lower_bottom + (lower_top - lower_bottom) / 2
    )  # Center of lower section
    summary_bottom = summary_center - summary_height / 2  # Center the plot vertically
    summary_left = plots_start + left_width + gap_width  # With increased gap

    # Create panel
    summary_ax = fig.add_axes(
        rect=[summary_left, summary_bottom, right_width, summary_height]
    )
    summary_ax.patch.set_alpha(0)  # Transparent background

    # Add title
    layer_display = get_display_name(focus_layer, config) or focus_layer.upper()
    summary_ax.set_title(
        label=f"Layer {layer_display}",
        fontsize=STYLE["fontsize_title"],
        pad=LAYOUT["title_padding"],
    )

    # Get the metric to plot
    metric = exp_info["metric"]
    metric_col = f"{focus_layer}_{metric}"
    logger.info(f"Summary plot using metric: {metric_col}")

    # Calculate metric if needed
    if metric_col not in df.columns:
        df = calculate_metrics(
            df=df,
            focus_layer=focus_layer,
            parameter=parameter,
            category=category,
            metric=metric,
            dt=dt,
        )

    # Prepare summary data
    summary_data = []
    param_values = sorted(df[parameter].unique())

    # Convert category values to strings for consistent comparison
    df[category] = df[category].astype(str)

    for param_val in param_values:
        for cat_val in sorted(df[category].unique()):
            filtered = df[(df[parameter] == param_val) & (df[category] == cat_val)]
            if (
                len(filtered) > 0
                and metric_col in filtered.columns
                and not filtered[metric_col].isna().all()
            ):
                metric_value = filtered[metric_col].mean()
                summary_data.append(
                    {parameter: param_val, category: cat_val, metric_col: metric_value}
                )

    # Create plot
    if summary_data:
        # Create DataFrame for plot
        summary_df = pd.DataFrame(summary_data)
        logger.info(f"Summary data shape: {summary_df.shape}")

        # Ensure category values are strings and model_order is string-based
        summary_df[category] = summary_df[category].astype(str)
        string_model_order = [str(val) for val in model_order]

        # Plot summary data with improved styling
        sns.lineplot(
            data=summary_df,
            x=parameter,
            y=metric_col,
            ax=summary_ax,
            marker="o",
            hue=category,
            hue_order=string_model_order,
            palette=colors,
            legend=False,
            linewidth=STYLE["linewidth_main"],
            markersize=STYLE["marker_size"],
            alpha=STYLE["alpha_lines"],
        )

        # Improve marker appearance
        for line in summary_ax.get_lines():
            line.set_markeredgecolor("white")
            line.set_markeredgewidth(STYLE["marker_edge_width"])

        # Set x-ticks to match parameter values
        summary_ax.set_xticks(sorted(param_values))

        # If dt is provided and parameter is time-based, format x-tick labels with ms
        if dt is not None and parameter.lower() in [
            "tau",
            "interval",
            "duration",
            "time",
        ]:
            summary_ax.set_xticklabels(
                [f"{int(x * dt)}" for x in sorted(param_values)]
            )

        # Add grid with improved styling
        summary_ax.grid(
            True,
            which="major",
            linestyle=":",
            linewidth=STYLE["grid_linewidth"],
            alpha=STYLE["grid_alpha"],
            axis="both",
        )

        # Add reference line for peak ratio
        if exp_info["metric"] == "peak_ratio":
            summary_ax.axhline(
                y=1, color="gray", linestyle="--", linewidth=1, alpha=0.7
            )
    else:
        # Empty plot with message
        summary_ax.text(
            x=0.5,
            y=0.5,
            s="No summary data available",
            ha="center",
            va="center",
            fontsize=STYLE["fontsize_labels"],
            alpha=0.7,
        )

    # Add axis labels with improved formatting
    summary_ax.set_xlabel(exp_info["full_label"], fontsize=STYLE["fontsize_labels"])
    summary_ax.set_ylabel(exp_info["y_label"], fontsize=STYLE["fontsize_labels"])
    summary_ax.tick_params(axis="both", labelsize=STYLE["fontsize_tick_labels"])

    # Clean up spines
    sns.despine(ax=summary_ax)

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
    """
    Create unified figure with:
    - Panel A (upper left): Different layers for middle parameter value
    - Panel B (lower left): Focus layer across different parameter values
    - Panel C (upper right): Empty panel for custom image
    - Panel D (lower right): Summary plot showing parameter effect
    - Horizontal legend between upper and lower sections
    """
    # Validate inputs
    logger.info(f"Input data shape: {df.shape}")
    logger.info(f"Focus layer: {focus_layer}")
    logger.info(f"Parameter column: {parameter}")
    logger.info(f"Category column: {category}")

    if parameter not in df.columns:
        raise ValueError(
            f"Parameter column '{parameter}' not found in data columns: {df.columns.tolist()}"
        )
    if category not in df.columns:
        raise ValueError(
            f"Category column '{category}' not found in data columns: {df.columns.tolist()}"
        )

    # Determine experiment type for metric selection
    experiment = experiment or parameter
    exp_info = determine_experiment_info(experiment)
    logger.info(f"Experiment info: {exp_info}")

    # Get layer names from response columns
    layer_cols = [col for col in df.columns if col.endswith("_response_avg")]
    layer_names = [col.replace("_response_avg", "") for col in layer_cols]
    ordered_layers = order_layers(layer_names=layer_names, config=config)
    logger.info(f"Using layers: {ordered_layers}")

    # Get plotting settings from config using improved method
    category_values = sorted(df[category].unique())

    # Use improved color palette function to match triptych script
    model_order, colors = get_improved_color_palette(
        category_values=category_values, config=config, parameter=parameter
    )

    logger.info(f"Category values: {category_values}")
    logger.info(f"Model order: {model_order}")
    logger.info(f"Colors: {list(colors.items())}")

    # Get parameter values
    param_values = sorted(df[parameter].unique())

    # Convert time axis if dt is provided
    if dt is not None:
        logger.info(f"Converting time axis using dt={dt} ms")
        df = df.copy()
        df["time_ms"] = df["times_index"] * dt

    # Set seaborn style for a clean look
    sns.set_style("ticks")
    sns.set_context("talk")

    # Create figure with dimensions from LAYOUT
    fig = plt.figure(figsize=(LAYOUT["figure_width"], LAYOUT["figure_height"]))

    # Add panel labels
    add_panel_labels(fig=fig)

    # Select middle parameter value for upper subplot
    middle_key_idx = len(param_values) // 2
    middle_key_value = param_values[middle_key_idx]

    # PANEL A: Upper left - Different layers for middle parameter value
    logger.info(f"Creating Panel A: Layers plot for {parameter}={middle_key_value}")
    upper_left_axes = plot_upper_layers_panel(
        fig=fig,
        df=df,
        layer_names=ordered_layers,
        parameter=parameter,
        category=category,
        param_value=middle_key_value,
        model_order=model_order,
        colors=colors,
        dt=dt,
        config=config,
    )

    # PANEL B: Lower left - Focus layer across different parameter values
    # Select evenly spaced parameter values if we have too many
    n_show_params = min(5, len(param_values))
    if n_show_params < len(param_values):
        param_indices = np.round(
            np.linspace(0, len(param_values) - 1, n_show_params)
        ).astype(int)
        show_param_values = [param_values[i] for i in param_indices]
    else:
        show_param_values = param_values

    logger.info(
        f"Creating Panel B: Focus layer {focus_layer} across parameters: {show_param_values}"
    )
    lower_left_axes = plot_parameter_variation_panel(
        fig=fig,
        df=df,
        focus_layer=focus_layer,
        parameter=parameter,
        category=category,
        param_values=show_param_values,
        model_order=model_order,
        colors=colors,
        dt=dt,
        config=config,
    )

    # PANEL C: Upper right - Empty panel for custom image
    logger.info("Creating Panel C: Empty panel for custom visualization")
    upper_right_ax = create_empty_panel(fig=fig, title=None, config=config)

    # PANEL D: Lower right - Summary plot showing parameter effect
    logger.info(f"Creating Panel D: Summary plot showing {exp_info['metric']}")
    summary_ax = plot_summary_panel(
        fig=fig,
        df=df,
        focus_layer=focus_layer,
        parameter=parameter,
        category=category,
        exp_info=exp_info,
        model_order=model_order,
        colors=colors,
        dt=dt,
        config=config,
    )

    # Legend between upper and lower sections
    logger.info(f"Creating legend for {category} values")
    legend_ax = create_horizontal_legend(
        fig=fig,
        colors=colors,
        model_order=model_order,
        category=category,
        config=config,
    )

    # Adjust layout using consistent positioning with slightly more space at bottom
    plt.tight_layout(rect=[LAYOUT["plots_start"] - 0.1, 0.02, 0.95, 0.95])

    # Save figure if path provided
    if save_path:
        logger.info(f"Saving figure to {save_path}")
        save_path = Path(save_path)
        save_plot(file_path=save_path, dpi=300)

    return fig


def main():
    """Main function to handle command line arguments."""
    parser = argparse.ArgumentParser(
        description="Create unified dynamics visualization across parameter values and layers."
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
    parser.add_argument(
        "--experiment", type=str, help="Experiment name (defaults to parameter)"
    )
    parser.add_argument(
        "--category", type=str, required=True, help="Category column name"
    )
    parser.add_argument(
        "--focus-layer", type=str, default="V1", help="Layer to focus on"
    )
    parser.add_argument("--dt", type=float, help="Time step in milliseconds")
    parser.add_argument(
        "--palette", type=str, help="JSON formatted dictionary of colors"
    )
    parser.add_argument("--naming", type=str, help="JSON formatted naming dictionary")
    parser.add_argument(
        "--ordering", type=str, help="JSON formatted ordering dictionary"
    )

    args = parser.parse_args()

    # Load data
    logger.info(f"Loading data from: {args.data}")
    df = pd.read_csv(args.data)

    # Load configuration
    config = load_config_from_args(
        palette_str=args.palette, naming_str=args.naming, ordering_str=args.ordering
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
