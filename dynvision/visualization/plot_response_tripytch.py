"""plot_response_tripytch.py
"""

import argparse
import json
import re
import traceback
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path

from dynvision.utils.visualization_utils import (
    save_plot,
    load_config_from_args,
    get_display_name,
    get_color,
    calculate_label_indicator,
    get_category_plotting_settings,
    order_layers,
)

# Global layout configuration for consistent positioning
LAYOUT = {
    # Figure dimensions
    "figure_width": 20,
    "figure_height": 16,
    # Column layout
    "column_left_positions": [0.08, 0.39, 0.70],
    "column_width": 0.26,
    # Vertical layout: Adjusted to have visually pleasing whitespace between panels
    "title_bot": 0.98,  # Moved up to prevent overlap
    "title_height": 0.02,  # Reduced height
    # Upper part with better spacing between accuracy plots
    "accuracy_bot": 0.83,  # Slightly higher for better spacing from title
    "accuracy_height": 0.13,  # Slightly increased height
    "timestep_accuracy_bot": 0.64,  # Adjusted for clearer separation
    "timestep_accuracy_height": 0.13,  # Slightly increased height
    # Legend part with reduced gap to content
    "legend_bot": 0.52,  # Better positioned between panels
    "legend_height": 0.06,  # Reduced height for compact legend
    # Lower part - optimized spacing
    "dynamics_top": 0.50,  # Adjusted for better gap from legend
    "dynamics_height": 0.50,  # Increased height for better visibility
    # Panel letters positioning
    "column_letter_y": 0.99,  # Keep at top
    "subpanel_letter_x_offset": -0.02,
    # Legend centering within columns
    "legend_column_margin": 0.02,
}

# Styling constants
LINE_ALPHA = 0.8  # Consistent alpha for all line plots
FONTSIZE_PANEL_LABELS = 18
FONTSIZE_AXIS_LABELS = 18
FONTSIZE_TICK_LABELS = 16
FONTSIZE_LEGEND = 18
FONTSIZE_TITLE = 20
LINEWIDTH_MAIN = 3
LINEWIDTH_INDICATOR = 3  # Increased linewidth for indicator
ALPHA_INDICATOR = 0.6  # Slightly increased alpha for better visibility

parser = argparse.ArgumentParser(description="Create response triptych plots")
parser.add_argument("--data", type=Path, help="Path to first dataset")
parser.add_argument("--data2", type=Path, help="Path to second dataset")
parser.add_argument("--data3", type=Path, help="Path to third dataset")

# Support both single accuracy file (backward compatibility) and multiple accuracy files
parser.add_argument("--accuracy", type=Path, help="Path to accuracy CSV file (legacy)")
parser.add_argument("--accuracy1", type=Path, help="Path to first accuracy CSV file")
parser.add_argument("--accuracy2", type=Path, help="Path to second accuracy CSV file")
parser.add_argument("--accuracy3", type=Path, help="Path to third accuracy CSV file")

parser.add_argument("--output", type=Path, required=True, help="Path to output file")
parser.add_argument(
    "--parameter",
    type=str,
    required=True,
    help="Column name containing experiment parameter values",
)
parser.add_argument("--experiment", type=str, help="Experiment name for title")
parser.add_argument(
    "--category", type=str, required=True, help="Categories to plot (space-separated)"
)
parser.add_argument(
    "--dt", type=float, required=True, help="Temporal resolution in ms per timestep"
)
parser.add_argument(
    "--outlier_threshold",
    type=float,
    default=10.0,
    help="Y-axis values beyond ±this threshold are considered outliers (default: 10.0)",
)
parser.add_argument("--palette", type=str, help="JSON formatted dictionary of colors")
parser.add_argument("--naming", type=str, help="JSON formatted naming dictionary")
parser.add_argument("--ordering", type=str, help="JSON formatted ordering dictionary")


def get_title_and_symbol(category, config=None):
    """Get display title and mathematical symbol for a category."""
    # Try to get from config first
    symbol = get_display_name(category, config) if config else None

    # Enhanced mappings with full names
    if symbol is None:
        mappings = {
            "tau": ("Time Constant", r"$\tau$"),
            "trc": ("Recurrence Delay", r"$\Delta_{RC}$"),
            "tsk": ("Skip Delay", r"$\Delta_{SK}$"),
            "rctarget": ("Recurrence Target", "Target"),
            "lossrt": ("Loss Reaction Time", r"$t_{loss}$"),
            "feedback": ("Feedback", "Feedback"),
        }
        category_name, symbol = mappings.get(
            category, (category.capitalize(), category)
        )
    else:
        # If we have a symbol from config, use the enhanced name mappings
        name_mappings = {
            "tau": "Time Constant",
            "trc": "Recurrence Delay",
            "tsk": "Skip Delay",
            "rctarget": "Recurrence Target",
            "lossrt": "Loss Reaction Time",
            "feedback": "Feedback",
        }
        category_name = name_mappings.get(category, category.capitalize())

    # FIXED: Only add full name if symbol contains '$' (is a mathematical symbol)
    # This prevents duplicate symbols like "τ (τ)"
    if symbol and "$" in symbol:
        title = f"Varying {category_name} ({symbol})"
    else:
        # Use the proper category name, not the symbol
        title = f"Varying {category_name}"

    return title, symbol


def get_layer_colors_from_palette(config=None):
    """Generate colors for different brain areas/layers."""
    layer_colors = {
        "V1": "#1f77b4",  # blue
        "V2": "#ff7f0e",  # orange
        "V4": "#2ca02c",  # green
        "IT": "#d62728",  # red
        "CLASSIFIER": "#9467bd",  # purple
    }

    # Return default colors if no config
    if config is None:
        return layer_colors

    # Try to get colors from config
    for layer in layer_colors:
        color = get_color(layer, config)
        if color:
            layer_colors[layer] = color

    return layer_colors


def add_panel_letters(fig):
    """Add panel letters A), B), C) for columns and i), ii), iii) for plots within columns."""
    # Column letters A), B), C)
    column_letters = ["A)", "B)", "C)"]
    for i, left in enumerate(LAYOUT["column_left_positions"]):
        # Position column letter at top of each column
        fig.text(
            left + LAYOUT["subpanel_letter_x_offset"],
            LAYOUT["column_letter_y"],
            column_letters[i],
            fontsize=FONTSIZE_PANEL_LABELS,
            fontweight="bold",
            ha="center",
            va="center",
        )

        # Panel letters within each column: i), ii), iii) positioned at top-left of respective plots
        plot_letters = ["i)", "ii)", "iii)"]

        # Calculate y-positions based on actual plot positions, moved higher to upper left
        plot_y_positions = [
            LAYOUT["accuracy_bot"]
            + LAYOUT["accuracy_height"]
            + 0.01,  # Higher position above accuracy plot
            LAYOUT["timestep_accuracy_bot"]
            + LAYOUT["timestep_accuracy_height"]
            + 0.01,  # Higher position above timestep accuracy plot
            LAYOUT["dynamics_top"] + 0.01,  # Higher position above dynamics plot
        ]

        for j, (letter, y_pos) in enumerate(zip(plot_letters, plot_y_positions)):
            fig.text(
                left + LAYOUT["subpanel_letter_x_offset"] - 0.01,  # Moved more to left
                y_pos,
                letter,
                fontsize=FONTSIZE_PANEL_LABELS - 2,
                fontweight="bold",
                ha="left",  # Aligned left instead of center
                va="bottom",  # Aligned to bottom for better positioning
            )


def create_horizontal_legend(
    fig, colors, model_order, category_list, format_func, config=None
):
    """Create a horizontal legend centered within each column with mathematical symbols as titles."""
    for col_idx, (category, panel_colors, panel_order) in enumerate(
        zip(category_list, colors, model_order)
    ):
        if not panel_colors:
            continue

        # Calculate column-specific legend position - centered within column
        left = (
            LAYOUT["column_left_positions"][col_idx] + LAYOUT["legend_column_margin"]
        )
        width = LAYOUT["column_width"] - 2 * LAYOUT["legend_column_margin"]

        legend_ax = fig.add_axes(
            [left, LAYOUT["legend_bot"], width, LAYOUT["legend_height"]]
        )
        legend_ax.set_xlim(0, 1)
        legend_ax.set_ylim(0, 1)
        legend_ax.axis("off")
        legend_ax.patch.set_alpha(0)

        # Create legend elements for this column's category
        legend_elements = []
        legend_labels = []

        # Get the symbol for this category
        _, symbol = get_title_and_symbol(category, config)

        for val in panel_order:
            # Ensure val is a string for dictionary lookup
            val_str = str(val)
            if val_str in panel_colors:
                color = panel_colors[val_str]
                formatted_label = format_func(category, val, config)
                label = formatted_label

                legend_elements.append(
                    plt.Line2D(
                        [0],
                        [0],
                        color=color,
                        linewidth=LINEWIDTH_MAIN,
                        marker=".",
                        markersize=8,
                        alpha=LINE_ALPHA,
                    )
                )
                legend_labels.append(label)

        # Create horizontal legend - centered within column
        if legend_elements:
            # Calculate optimal number of columns for this legend
            n_items = len(legend_elements)
            n_cols = min(n_items, 3)  # Max 3 columns per panel

            legend = legend_ax.legend(
                legend_elements,
                legend_labels,
                loc="center",  # Center the legend within the axes
                ncol=n_cols,
                frameon=False,
                fontsize=FONTSIZE_LEGEND,
                handlelength=2,
                handletextpad=0.5,
                columnspacing=1.0,
                title=symbol,  # Add mathematical symbol as title
                title_fontsize=FONTSIZE_LEGEND,
            )

            # # Style the legend title
            # if legend.get_title():
            #     legend.get_title().set_fontweight("bold")

    return None


def create_empty_panel(column_index, fig, category, show_ylabel=False, config=None):
    """Create an empty panel with appropriate messaging."""
    left = LAYOUT["column_left_positions"][column_index]
    width = LAYOUT["column_width"]

    # Add title
    title, _ = get_title_and_symbol(category, config)
    title_ax = fig.add_axes([left, LAYOUT["title_bot"], width, LAYOUT["title_height"]])
    title_ax.text(
        0.5,
        0.5,
        title,
        ha="center",
        va="center",
        fontsize=FONTSIZE_TITLE,
        fontweight="bold",
    )
    title_ax.set_xlim(0, 1)
    title_ax.set_ylim(0, 1)
    title_ax.axis("off")

    # Create single empty plot area
    ax = fig.add_axes(
        [
            left,
            LAYOUT["dynamics_top"] - LAYOUT["dynamics_height"],
            width,
            LAYOUT["dynamics_height"],
        ]
    )
    ax.text(
        0.5, 0.5, "No data available", ha="center", va="center", fontsize=16, alpha=0.6
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    if show_ylabel:
        ax.set_ylabel(
            "Avg Layer Response",
            labelpad=8,
            fontsize=FONTSIZE_AXIS_LABELS,
            fontweight="bold",
        )
    else:
        ax.set_ylabel("")

    ax.set_xlabel("Time (ms)", fontsize=FONTSIZE_AXIS_LABELS, labelpad=8)
    ax.tick_params(labelsize=FONTSIZE_TICK_LABELS)
    sns.despine(ax=ax, left=True, bottom=True)

    return [ax]  # Return as list to maintain consistency


def plot_response_panel(
    df,
    parameter,
    category,
    column_index,
    fig,
    dt,
    accuracy_df=None,
    show_layer_indicators=False,
    show_ylabel=False,
    config=None,
):
    """Plot a single response panel within the triptych figure."""
    left = LAYOUT["column_left_positions"][column_index]
    width = LAYOUT["column_width"]

    # Check if data is valid
    if df is None or df.empty:
        print(f"Panel {column_index} - No valid data available")
        return (
            create_empty_panel(column_index, fig, category, show_ylabel, config),
            None,
            None,
        )

    # Check if required columns exist
    if category not in df.columns:
        print(
            f"Panel {column_index} - Category '{category}' not found in data columns"
        )
        return (
            create_empty_panel(column_index, fig, category, show_ylabel, config),
            None,
            None,
        )

    if parameter not in df.columns:
        print(
            f"Panel {column_index} - Parameter '{parameter}' not found in data columns"
        )
        print(f"Available columns: {df.columns.tolist()}")
        return (
            create_empty_panel(column_index, fig, category, show_ylabel, config),
            None,
            None,
        )

    response_columns = [col for col in df.columns if col.endswith("_response_avg")]
    if not response_columns:
        print(f"Panel {column_index} - No response columns found in data")
        return (
            create_empty_panel(column_index, fig, category, show_ylabel, config),
            None,
            None,
        )

    print(f"Panel {column_index} - Data shape: {df.shape}")
    print(f"Unique {category} values: {sorted(df[category].unique())}")

    # Print sample count
    if "sample_index" in df.columns:
        n_samples = df["sample_index"].nunique()
        print(f"Panel {column_index}: Averaging over {n_samples} sample indices")

    # Get layers and plotting settings
    all_layer_names = [
        col.replace("_response_avg", "")
        for col in df.columns
        if col.endswith("_response_avg")
    ]
    layer_names = order_layers(all_layer_names, config)

    if not layer_names:
        print(f"Panel {column_index} - No valid layer names found")
        return (
            create_empty_panel(column_index, fig, category, show_ylabel, config),
            None,
            None,
        )

    # Convert category values to strings to ensure consistent types
    category_values = [str(val) for val in sorted(df[category].unique())]
    try:
        # Use improved color palette (viridis)
        model_order, colors = get_improved_color_palette(category_values, config)
    except Exception as e:
        print(f"Panel {column_index} - Error getting plotting settings: {e}")
        return (
            create_empty_panel(column_index, fig, category, show_ylabel, config),
            None,
            None,
        )

    # Add title
    title, symbol = get_title_and_symbol(category, config)
    title_ax = fig.add_axes([left, LAYOUT["title_bot"], width, LAYOUT["title_height"]])
    title_ax.text(
        0.5,
        0.5,
        title,
        ha="center",
        va="center",
        fontsize=FONTSIZE_TITLE,
        fontweight="bold",
    )
    title_ax.set_xlim(0, 1)
    title_ax.set_ylim(0, 1)
    title_ax.axis("off")

    # Plot accuracy if available
    if accuracy_df is not None:
        accuracy_panel_df = extract_accuracy_data(accuracy_df, category)
        # Always create accuracy panel, even if empty (for consistent layout)
        plot_accuracy_panel(
            accuracy_panel_df,
            category,
            column_index,
            fig,
            colors,
            model_order,
            show_ylabel=(column_index == 0),
            show_legend=(column_index == 0),
            config=config,
        )

    # Plot timestep accuracy from classifier responses
    if df is not None and not df.empty:
        plot_timestep_accuracy_panel(
            df,
            category,
            column_index,
            fig,
            colors,
            model_order,
            dt,
            show_ylabel=(column_index == 0),
            config=config,
        )

    # Calculate subplot layout for ridgeplot effect
    n_layers = len(layer_names)
    overlap_factor = 0.25
    available_height = LAYOUT["dynamics_height"]
    spacing = available_height / n_layers * (1 - overlap_factor)
    plot_height = available_height / n_layers * 1.4
    start_top = LAYOUT["dynamics_top"]

    # Create a time column in ms for proper x-axis scaling
    df_with_time = df.copy()
    df_with_time["time_ms"] = df_with_time["times_index"] * dt

    # Ensure category values in dataframe are strings for consistent comparison
    if category in df_with_time.columns:
        df_with_time[category] = df_with_time[category].astype(str)

    axes = []
    for i, layer_name in enumerate(layer_names):
        # Position subplot with overlap
        top_pos = start_top - i * spacing
        bottom_pos = top_pos - plot_height

        ax = fig.add_axes([left, bottom_pos, width, plot_height])
        ax.patch.set_alpha(0)
        axes.append(ax)

        # Add reference line
        ax.axhline(0, color="gray", linestyle=":", alpha=0.7, linewidth=1)

        # Plot layer response with proper time scaling
        response_col = f"{layer_name}_response_avg"
        if response_col in df_with_time.columns:
            # Ensure model_order values are strings for consistent comparison
            string_model_order = [str(val) for val in model_order]

            sns.lineplot(
                data=df_with_time,
                x="time_ms",
                y=response_col,
                ax=ax,
                marker=".",
                hue=category,
                hue_order=string_model_order,
                palette=colors,
                legend=False,
                linewidth=LINEWIDTH_MAIN,
                # errorbar=None,
                alpha=LINE_ALPHA,  # Consistent alpha
            )

        # Add circular layer indicators to all lower subplots
        layer_colors = get_layer_colors_from_palette(config)
        pad = 0.5 if layer_name == "IT" else 0.4
        ax.text(
            0.95,  # Position at right side
            0.25,  # At 0.25 height as requested
            layer_name.upper(),
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
            bbox=dict(
                boxstyle=f"circle,pad={pad}",
                facecolor=layer_colors[layer_name.upper()],
                edgecolor="#353535ff",
                linewidth=2,
                alpha=0.8,
            ),
            fontsize=12,
            fontweight="bold",
        )

        # Y-label on leftmost panel, middle layer
        if show_ylabel and i == len(layer_names) // 2:
            ax.set_ylabel(
                "Avg Layer Response",
                labelpad=8,
                fontsize=FONTSIZE_AXIS_LABELS,
                fontweight="bold",
            )
        else:
            ax.set_ylabel("")

        # X-axis only on bottom subplot
        if i < len(layer_names) - 1:
            ax.set_xticklabels([])
            ax.set_xlabel("")
        else:
            ax.set_xlabel("Time (ms)", fontsize=FONTSIZE_AXIS_LABELS, labelpad=8)
            ax.tick_params(labelsize=FONTSIZE_TICK_LABELS)

        # Set y-ticks to only 0 and 1 for lower plots
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["0", "1"])

        sns.despine(ax=ax, left=True, bottom=True)

    return axes, colors, model_order


def get_improved_color_palette(category_values, config=None):
    """Generate a perceptually uniform sequential color palette."""
    # First check if all values are numeric (can be converted to float)
    are_numeric = True
    original_values = list(category_values)  # Keep original values for reference
    numeric_values = []

    for val in category_values:
        try:
            numeric_values.append(float(val))
        except (ValueError, TypeError):
            are_numeric = False
            break

    # Convert all category values to strings for consistent comparison
    category_values = [str(val) for val in category_values]

    # Try to get colors from config first
    if config:
        colors = {}
        for val in category_values:
            color = get_color(val, config)
            if color:
                colors[val] = color

        if colors and len(colors) == len(category_values):
            # If values are numeric, sort numerically
            if are_numeric:
                # Sort by numeric value but keep as strings
                model_order = [str(x) for x in sorted(numeric_values)]
            else:
                model_order = sorted(category_values)
            return model_order, colors

    # Use perceptually uniform sequential palette
    # If values are numeric, sort numerically
    if are_numeric:
        # Create mapping from numeric value to string representation
        value_to_str = {float(val): str(val) for val in original_values}
        # Sort numerically but return string representations in order
        model_order = [value_to_str[val] for val in sorted(numeric_values)]
    else:
        model_order = sorted(category_values)

    viridis_colors = plt.cm.viridis(np.linspace(0, 1, len(model_order)))
    hex_colors = [plt.matplotlib.colors.to_hex(color) for color in viridis_colors]
    colors = dict(zip(model_order, hex_colors))

    return model_order, colors


def extract_param_value(column_name, category, debug=False):
    """Extract parameter value from column name for given category (case-insensitive)."""
    pattern = f"{category}=([^+_]+)"
    match = re.search(pattern, column_name, re.IGNORECASE)
    if match:
        # Always return as string for consistent comparison
        value = str(match.group(1))
        return value

    # Debug: print pattern matching attempts for debugging
    if debug:
        print(
            f"  DEBUG: Pattern '{pattern}' in column '{column_name}' -> match: {match}"
        )
    return None


def format_legend_label(category, value, config=None):
    """Format legend labels based on parameter type."""
    # Special handling for feedback category
    if category == "feedback":
        if str(value).lower() == "true":
            return "Additive"
        elif str(value).lower() in ["mul", "multiplicative"]:
            return "Multiplicative"
        else:
            return str(value).capitalize()

    # Numerical parameters that should have "ms" unit
    numerical_with_ms = ["tau", "trc", "tsk", "lossrt"]

    if category in numerical_with_ms:
        try:
            numeric_value = int(float(value))
            return f"{numeric_value} ms"
        except (ValueError, TypeError):
            return f"{value} ms"
    else:
        # Categorical parameters - handle boolean values specially
        if str(value).lower() in ["true", "false"]:
            return str(value).capitalize()  # "True" or "False"
        else:
            # Other categorical values - capitalize first letter
            return str(value).capitalize()


def calculate_timestep_accuracy(df, category):
    """Calculate classification accuracy per timestep grouped by category with proper label handling."""
    if df is None or df.empty:
        return pd.DataFrame()

    # Check for required columns
    required_cols = [
        "times_index",
        category,
        "guess_index",
        "label_index",
        "sample_index",
    ]

    # If we don't have "guess_index", try using "accuracy" instead
    if "guess_index" not in df.columns and "accuracy" in df.columns:
        # We'll use the accuracy column directly
        timestep_accuracy_data = []

        for cat_value in df[category].unique():
            cat_data = df[df[category] == cat_value]

            for timestep in sorted(cat_data["times_index"].unique()):
                timestep_data = cat_data[cat_data["times_index"] == timestep]
                if not timestep_data.empty:
                    timestep_accuracy_data.append(
                        {
                            "times_index": timestep,
                            "timestep_accuracy": timestep_data["accuracy"].mean(),
                            category: cat_value,
                        }
                    )

        return pd.DataFrame(timestep_accuracy_data)

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"  Warning: Missing columns for timestep accuracy: {missing_cols}")
        return pd.DataFrame()

    try:
        # Step 1: Fix label_index values - replace -1 with actual label for each sample
        df_fixed = df.copy()

        for sample_idx in df_fixed["sample_index"].unique():
            sample_data = df_fixed[df_fixed["sample_index"] == sample_idx]

            # Find the non-negative label_index for this sample
            non_negative_labels = sample_data[sample_data["label_index"] >= 0][
                "label_index"
            ]
            if len(non_negative_labels) > 0:
                correct_label = non_negative_labels.iloc[0]
                # Update label_index where it was -1 for this sample
                df_fixed.loc[
                    (df_fixed["sample_index"] == sample_idx)
                    & (df_fixed["label_index"] < 0),
                    "label_index",
                ] = correct_label

        # Step 2: Calculate accuracy per timestep and category
        timestep_accuracy_data = []

        for cat_value in df_fixed[category].unique():
            cat_data = df_fixed[df_fixed[category] == cat_value]

            for timestep in sorted(cat_data["times_index"].unique()):
                timestep_data = cat_data[cat_data["times_index"] == timestep]
                if not timestep_data.empty:
                    # Calculate accuracy as correct / total
                    correct = (
                        timestep_data["guess_index"] == timestep_data["label_index"]
                    ).sum()
                    total = len(timestep_data)
                    accuracy = correct / total if total > 0 else 0

                    timestep_accuracy_data.append(
                        {
                            "times_index": timestep,
                            "timestep_accuracy": accuracy,
                            category: cat_value,
                        }
                    )

        return pd.DataFrame(timestep_accuracy_data)

    except Exception as e:
        print(f"Error calculating timestep accuracy: {e}")
        return pd.DataFrame()


def extract_accuracy_data(accuracy_df, category):
    """Extract accuracy data for the specified category variation."""
    if accuracy_df is None or accuracy_df.empty:
        return pd.DataFrame()

    print(f"  DEBUG: Extracting accuracy for category '{category}'")
    print(f"  DEBUG: Available columns: {list(accuracy_df.columns)}")

    accuracy_data = []

    try:
        # Find training and validation accuracy columns (exclude MIN/MAX)
        for acc_type, pattern in [
            ("Training", "train_accuracy"),
            ("Validation", "val_accuracy"),
        ]:
            # Find columns containing the pattern and not containing MIN/MAX
            matching_cols = [
                col
                for col in accuracy_df.columns
                if pattern in col and not any(x in col for x in ["__MIN", "__MAX"])
            ]

            print(
                f"  DEBUG: Found {len(matching_cols)} {acc_type.lower()} accuracy columns"
            )
            if matching_cols:
                print(
                    f"  DEBUG: Example {acc_type.lower()} columns: {matching_cols[:3]}"
                )

            for col in matching_cols:
                # Extract parameter value from column name
                param_value = extract_param_value(col, category, debug=True)
                print(f"  DEBUG: Column '{col}' -> parameter value: '{param_value}'")

                if param_value:
                    # Add data points for this column
                    for idx, row in accuracy_df.iterrows():
                        if not pd.isna(row[col]):
                            accuracy_data.append(
                                {
                                    "epoch": row["epoch"],
                                    "accuracy": row[col],
                                    "type": acc_type,
                                    category: param_value,
                                }
                            )

    except Exception as e:
        print(f"Error extracting accuracy data for category '{category}': {e}")
        import traceback

        traceback.print_exc()
        return pd.DataFrame()

    print(f"  DEBUG: Extracted {len(accuracy_data)} accuracy data points")
    if accuracy_data:
        result_df = pd.DataFrame(accuracy_data)
        unique_values = result_df[category].unique()
        print(f"  DEBUG: Unique {category} values found: {unique_values}")
        print(f"  DEBUG: Sample of extracted data:")
        print(result_df.head())
        return result_df
    else:
        print(f"  DEBUG: No accuracy data extracted for category '{category}'")
        return pd.DataFrame()


def plot_timestep_accuracy_panel(
    df,
    category,
    column_index,
    fig,
    colors,
    model_order,
    dt,
    show_ylabel=False,
    config=None,
):
    """Plot classification accuracy and confidence over timesteps with label indicator."""
    left = LAYOUT["column_left_positions"][column_index]
    width = LAYOUT["column_width"]

    ax = fig.add_axes(
        [
            left,
            LAYOUT["timestep_accuracy_bot"],
            width,
            LAYOUT["timestep_accuracy_height"],
        ]
    )
    ax.patch.set_alpha(0)

    # Calculate timestep accuracy
    timestep_accuracy_df = calculate_timestep_accuracy(df, category)

    # Check if we have valid data to plot
    if timestep_accuracy_df is None or timestep_accuracy_df.empty:
        ax.text(
            0.5,
            0.5,
            "No timestep accuracy data",
            ha="center",
            va="center",
            fontsize=12,
            alpha=0.6,
        )
    else:
        try:
            # Create time in ms column for plotting
            timestep_accuracy_df["time_ms"] = timestep_accuracy_df["times_index"] * dt

            # Ensure category values in dataframe are strings
            if category in timestep_accuracy_df.columns:
                timestep_accuracy_df[category] = timestep_accuracy_df[category].astype(
                    str
                )

            # Ensure model_order contains strings
            string_model_order = [str(val) for val in model_order]

            # Plot timestep accuracy (solid lines)
            sns.lineplot(
                data=timestep_accuracy_df,
                x="time_ms",
                y="timestep_accuracy",
                hue=category,
                hue_order=string_model_order,
                palette=colors,
                ax=ax,
                legend=False,
                linewidth=LINEWIDTH_MAIN,
                marker="o",
                markersize=4,
                # errorbar=None,
                alpha=LINE_ALPHA,
                linestyle="-",
            )

            # Plot confidence if available (dotted lines)
            if df is not None and "confidence_avg" in df.columns:
                # Calculate confidence data similar to accuracy
                confidence_data = []
                for cat_value in df[category].unique():
                    cat_data = df[df[category] == cat_value]
                    for timestep in sorted(cat_data["times_index"].unique()):
                        timestep_data = cat_data[cat_data["times_index"] == timestep]
                        if not timestep_data.empty:
                            confidence_data.append(
                                {
                                    "times_index": timestep,
                                    "confidence_avg": timestep_data[
                                        "confidence_avg"
                                    ].mean(),
                                    category: cat_value,
                                }
                            )

                if confidence_data:
                    confidence_df = pd.DataFrame(confidence_data)
                    confidence_df["time_ms"] = confidence_df["times_index"] * dt

                    # Ensure category values are strings
                    if category in confidence_df.columns:
                        confidence_df[category] = confidence_df[category].astype(str)

                    sns.lineplot(
                        data=confidence_df,
                        x="time_ms",
                        y="confidence_avg",
                        hue=category,
                        hue_order=string_model_order,  # Use the string version created earlier
                        palette=colors,
                        ax=ax,
                        legend=False,
                        linewidth=LINEWIDTH_MAIN,
                        marker="s",
                        markersize=3,
                        # errorbar=None,
                        alpha=LINE_ALPHA,
                        linestyle=":",
                    )

            # Set y-axis ticks with 0.2 intervals
            ax.set_ylim(0.0, 1.0)
            ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

            # Add label indicator if main data is available
            if df is not None and not df.empty:
                try:
                    # Use 0.1 (10%) of the plot height for the indicator
                    label_indicator_df = calculate_label_indicator(
                        df, category, (0.0, 1.0), 0.1
                    )
                    indicator_time = label_indicator_df.times_index * dt
                    ax.plot(
                        indicator_time,
                        label_indicator_df.label_indicator,
                        color="dimgray",  # Slightly darker for better visibility
                        linewidth=LINEWIDTH_INDICATOR,
                        drawstyle="steps-mid",
                        alpha=ALPHA_INDICATOR,
                    )
                except Exception as e:
                    print(f"Error adding label indicator to timestep accuracy: {e}")

        except Exception as e:
            print(f"Error plotting timestep accuracy: {e}")
            ax.text(
                0.5,
                0.5,
                "Error plotting timestep accuracy",
                ha="center",
                va="center",
                fontsize=12,
                alpha=0.6,
            )

    # Labels and formatting
    if show_ylabel:
        ax.set_ylabel(
            "Accuracy", fontsize=FONTSIZE_AXIS_LABELS, labelpad=4, fontweight="bold"
        )
    else:
        ax.set_ylabel("")
        ax.set_yticklabels([])

    ax.set_xlabel("Time (ms)", fontsize=FONTSIZE_AXIS_LABELS, labelpad=4)
    ax.tick_params(labelsize=FONTSIZE_TICK_LABELS)

    # FIXED: Add legend for line styles centered in subplot
    if show_ylabel and df is not None and "confidence_avg" in df.columns:
        legend_elements = [
            plt.Line2D(
                [0],
                [0],
                color="black",
                linewidth=LINEWIDTH_MAIN,
                linestyle="-",
                label="Accuracy",
                alpha=LINE_ALPHA,
            ),
            plt.Line2D(
                [0],
                [0],
                color="black",
                linewidth=LINEWIDTH_MAIN,
                linestyle=":",
                label="Confidence",
                alpha=LINE_ALPHA,
            ),
        ]
        ax.legend(
            handles=legend_elements,
            loc="center",  # FIXED: Centered in subplot
            frameon=False,
            fontsize=FONTSIZE_LEGEND - 2,
        )

    ax.grid(True, alpha=0.3)
    sns.despine(ax=ax, left=True, bottom=True)
    return ax


def plot_accuracy_panel(
    accuracy_df,
    category,
    column_index,
    fig,
    colors,
    model_order,
    show_ylabel=False,
    show_legend=False,
    config=None,
):
    """Plot training and validation accuracy at the top of each panel with category colors."""
    left = LAYOUT["column_left_positions"][column_index]
    width = LAYOUT["column_width"]

    ax = fig.add_axes([left, LAYOUT["accuracy_bot"], width, LAYOUT["accuracy_height"]])
    ax.patch.set_alpha(0)

    # Check if we have valid data to plot
    if accuracy_df is None or accuracy_df.empty:
        ax.text(
            0.5,
            0.5,
            "No accuracy data",
            ha="center",
            va="center",
            fontsize=14,
            alpha=0.6,
        )
        print(f"Panel {column_index} - No accuracy data to plot")
    else:
        print(
            f"Panel {column_index} - Plotting accuracy data with {len(accuracy_df)} rows"
        )
        print(
            f"Panel {column_index} - Unique categories in accuracy data: {accuracy_df[category].unique() if category in accuracy_df.columns else 'N/A'}"
        )

        try:
            # Ensure we have colors and model_order for accuracy plotting
            if colors is None or model_order is None:
                print(f"Panel {column_index} - Creating colors from accuracy data")
                # Get unique category values and create default colors
                if category in accuracy_df.columns:
                    category_values = sorted(accuracy_df[category].unique())
                    model_order, colors = get_improved_color_palette(
                        category_values, config
                    )
                    print(
                        f"Panel {column_index} - Created colors for values: {category_values}"
                    )
                else:
                    print(
                        f"Panel {column_index} - Category '{category}' not found in accuracy data columns: {list(accuracy_df.columns)}"
                    )
                    return ax
            else:
                # Ensure color keys and model_order values are strings to match the data
                string_model_order = [str(val) for val in model_order]
                string_colors = {str(k): v for k, v in colors.items()}
                model_order = string_model_order
                colors = string_colors
                print(
                    f"Panel {column_index} - Converting model_order to strings: {model_order}"
                )
                print(
                    f"Panel {column_index} - Converting color keys to strings: {list(colors.keys())}"
                )

            # FIXED: Ensure we have both training and validation data
            print(
                f"Panel {column_index} - Data types available: {accuracy_df['type'].unique() if 'type' in accuracy_df.columns else 'No type column'}"
            )

            # Plot training accuracy (solid lines)
            train_data = accuracy_df[accuracy_df["type"] == "Training"]
            if not train_data.empty:
                print(
                    f"Panel {column_index} - Plotting {len(train_data)} training points"
                )
                print(
                    f"Panel {column_index} - Training data category values: {train_data[category].unique()}"
                )

                sns.lineplot(
                    data=train_data,
                    x="epoch",
                    y="accuracy",
                    hue=category,
                    hue_order=model_order,
                    palette=colors,
                    ax=ax,
                    legend=False,
                    linewidth=LINEWIDTH_MAIN,
                    linestyle="-",
                    alpha=LINE_ALPHA,
                )
            else:
                print(f"Panel {column_index} - No training data found")

            # Plot validation accuracy (dotted lines)
            val_data = accuracy_df[accuracy_df["type"] == "Validation"]
            if not val_data.empty:
                print(
                    f"Panel {column_index} - Plotting {len(val_data)} validation points"
                )
                print(
                    f"Panel {column_index} - Validation data category values: {val_data[category].unique()}"
                )

                sns.lineplot(
                    data=val_data,
                    x="epoch",
                    y="accuracy",
                    hue=category,
                    hue_order=model_order,
                    palette=colors,
                    ax=ax,
                    legend=False,
                    linewidth=LINEWIDTH_MAIN,
                    linestyle=":",
                    alpha=LINE_ALPHA,
                )
            else:
                print(f"Panel {column_index} - No validation data found")

        except Exception as e:
            print(f"Panel {column_index} - Error plotting accuracy: {e}")
            import traceback

            traceback.print_exc()
            ax.text(
                0.5,
                0.5,
                f"Error plotting accuracy: {str(e)[:50]}...",
                ha="center",
                va="center",
                fontsize=12,
                alpha=0.6,
            )

    # Set consistent y-axis with 0.2 intervals
    if accuracy_df is not None and not accuracy_df.empty:
        current_ylim = ax.get_ylim()
        y_min = max(
            0.0, min(0.6, current_ylim[0] - 0.05)
        )  # Don't go below reasonable accuracy
        y_max = min(1.0, max(0.8, current_ylim[1] + 0.05))  # Don't go above 1.0
        ax.set_ylim(y_min, y_max)

        # Set ticks with 0.2 intervals
        tick_start = int(y_min * 5) * 0.2  # Round down to nearest 0.2
        tick_end = min(1.0, int(y_max * 5 + 1) * 0.2)  # Round up to nearest 0.2
        ticks = np.arange(
            tick_start, tick_end + 0.01, 0.2
        )  # +0.01 for floating point precision
        ax.set_yticks(ticks)

    # Labels and formatting
    if show_ylabel:
        ax.set_ylabel(
            "Accuracy", fontsize=FONTSIZE_AXIS_LABELS, labelpad=4, fontweight="bold"
        )
    else:
        ax.set_ylabel("")
        ax.set_yticklabels([])

    ax.set_xlabel("Epoch", fontsize=FONTSIZE_AXIS_LABELS, labelpad=4)
    ax.tick_params(labelsize=FONTSIZE_TICK_LABELS)

    # Add legend for line styles on first panel
    if show_legend:
        legend_elements = [
            plt.Line2D(
                [0],
                [0],
                color="black",
                linewidth=LINEWIDTH_MAIN,
                linestyle="-",
                label="Training",
                alpha=LINE_ALPHA,
            ),
            plt.Line2D(
                [0],
                [0],
                color="black",
                linewidth=LINEWIDTH_MAIN,
                linestyle=":",
                label="Validation",
                alpha=LINE_ALPHA,
            ),
        ]
        ax.legend(
            handles=legend_elements,
            loc="lower right",
            frameon=False,
            fontsize=FONTSIZE_LEGEND - 2,
        )

    ax.grid(True, alpha=0.3)
    sns.despine(ax=ax, left=True, bottom=True)
    return ax


def create_response_triptych(
    data_paths,
    parameter,
    category_list,
    dt,
    accuracy_paths=None,
    outlier_threshold=10.0,
    config=None,
):
    """Create a triptych (three-panel) figure with response plots."""
    print(f"Creating response triptych with {len(data_paths)} panels")

    sns.set_context("talk")
    fig = plt.figure(figsize=(LAYOUT["figure_width"], LAYOUT["figure_height"]))

    # Add panel letters A), B), C) and i), ii), iii)
    add_panel_letters(fig)

    all_axes = []
    all_y_limits = []
    panel_axes_list = []
    panel_colors_list = []
    panel_orders_list = []

    # Create each panel
    for i, (data_path, category) in enumerate(zip(data_paths, category_list)):
        print(f"\n=== Creating Panel {i+1}: {category} ===")
        print(f"Loading data from: {data_path}")

        # Load main data with error handling
        df = None
        try:
            if data_path and data_path.exists():
                df = pd.read_csv(data_path)
                if df.empty:
                    print(f"Warning: Empty data file: {data_path}")
            else:
                print(f"Warning: Data file not found: {data_path}")
        except Exception as e:
            print(f"Error loading data: {e}")

        # Load panel-specific accuracy data with error handling
        accuracy_df = None
        if accuracy_paths and i < len(accuracy_paths) and accuracy_paths[i]:
            try:
                print(f"Loading accuracy data from: {accuracy_paths[i]}")
                if accuracy_paths[i].exists():
                    accuracy_df = pd.read_csv(accuracy_paths[i])
                    if accuracy_df.empty:
                        print(f"Warning: Empty accuracy file: {accuracy_paths[i]}")
                else:
                    print(f"Warning: Accuracy file not found: {accuracy_paths[i]}")
            except Exception as e:
                print(f"Error loading accuracy data: {e}")

        # FIXED: Create panel - now handles accuracy independently even when main data is missing
        try:
            panel_axes, panel_colors, panel_order = plot_response_panel(
                df,
                parameter,
                category,
                i,
                fig,
                dt,
                accuracy_df=accuracy_df,  # This will be processed even if df is None
                show_ylabel=(i == 0),
                config=config,
            )
        except Exception as e:
            print(f"Error creating panel {i+1}: '{category}'")
            print(f"Exception: {e}")
            panel_axes = []
            panel_colors = None
            panel_order = None

        panel_axes_list.append(panel_axes)
        if panel_axes:
            all_axes.extend(panel_axes)

        panel_colors_list.append(panel_colors)
        panel_orders_list.append(panel_order)

        # FIXED: Only collect y-limits from panels with actual response data (robust calculation)
        if (
            df is not None
            and not df.empty
            and len([col for col in df.columns if col.endswith("_response_avg")]) > 0
            and panel_axes
            and len(panel_axes) > 0
            and panel_colors is not None  # Ensure this panel has valid colors
        ):
            for ax in panel_axes:
                try:
                    y_min, y_max = ax.get_ylim()
                    # Only add non-default y-limits
                    if not (y_min == 0.0 and y_max == 1.0):
                        all_y_limits.append((y_min, y_max))
                except:
                    continue

    # Create horizontal legend for panels with valid data
    valid_colors = [c for c in panel_colors_list if c is not None]
    valid_orders = [o for o in panel_orders_list if o is not None]
    valid_categories = [
        cat for cat, c in zip(category_list, panel_colors_list) if c is not None
    ]

    if valid_colors:
        create_horizontal_legend(
            fig,
            valid_colors,
            valid_orders,
            valid_categories,
            format_legend_label,
            config,
        )

    # FIXED: Apply robust global y-limits only to panels with valid response data
    if all_y_limits:
        # Filter out outliers to prevent extreme values from dominating the scale
        filtered_y_limits = []

        for y_min, y_max in all_y_limits:
            if abs(y_min) > outlier_threshold or abs(y_max) > outlier_threshold:
                # Cap extreme values to threshold
                filtered_y_min = max(-outlier_threshold, min(outlier_threshold, y_min))
                filtered_y_max = max(-outlier_threshold, min(outlier_threshold, y_max))
                filtered_y_limits.append((filtered_y_min, filtered_y_max))
            else:
                filtered_y_limits.append((y_min, y_max))

        if filtered_y_limits:
            # Find global min/max across all panels
            global_y_min = min(y_min for y_min, _ in filtered_y_limits)
            global_y_max = max(y_max for _, y_max in filtered_y_limits)

            # FIXED: Set minimum to exactly -0.05 for visibility of zero values
            global_y_min = -0.05  # Force exactly -0.05
            global_y_max = min(global_y_max, 4)  # Ensure at least 4 max

            # Apply common y-limits only to panels with actual response data
            for panel_idx, panel_axes in enumerate(panel_axes_list):
                if len(panel_axes) > 0 and panel_colors_list[panel_idx] is not None:
                    for ax in panel_axes:
                        ax.set_ylim(global_y_min, global_y_max)

            # FIXED: Add label indicators ONLY to the bottom plot with reduced height (10% of plot height)
            for panel_idx, (panel_axes, data_path, category) in enumerate(
                zip(panel_axes_list, data_paths, category_list)
            ):
                if (
                    data_path
                    and panel_axes
                    and len(panel_axes) > 0
                    and panel_colors_list[panel_idx] is not None
                ):
                    try:
                        df = pd.read_csv(data_path)
                        # Only add to the LAST (bottom) axis in the dynamics section
                        bottom_ax = panel_axes[-1]  # Last axis is the bottom one

                        label_indicator_df = calculate_label_indicator(
                            df, category, (global_y_min, global_y_max), 0.1
                        )
                        indicator_time = label_indicator_df.times_index * dt
                        bottom_ax.plot(
                            indicator_time,
                            label_indicator_df.label_indicator,
                            color="dimgray",  # Darker color for better visibility
                            linewidth=LINEWIDTH_INDICATOR,
                            drawstyle="steps-mid",
                            alpha=ALPHA_INDICATOR,
                        )
                    except Exception as e:
                        print(
                            f"Error adding label indicator for panel {panel_idx}: {e}"
                        )

    else:
        print("No valid y-limit data found - skipping global y-axis scaling")

    # Align y-labels only for panels with response data
    valid_axes = [
        ax
        for panel_idx, panel_axes in enumerate(panel_axes_list)
        for ax in panel_axes
        if len(panel_axes) > 0 and panel_colors_list[panel_idx] is not None
    ]
    if valid_axes:
        fig.align_ylabels(valid_axes)

    return fig


if __name__ == "__main__":
    args = parser.parse_args()

    # Load configuration from command line arguments
    config = load_config_from_args(
        palette_str=args.palette, naming_str=args.naming, ordering_str=args.ordering
    )

    category_list = args.category.split()
    print(f"Categories: {category_list}")
    print(f"Temporal resolution (dt): {args.dt} ms per timestep")
    print(f"Outlier threshold: ±{args.outlier_threshold}")

    if len(category_list) != 3:
        raise ValueError(
            f"Expected 3 categories, got {len(category_list)}: {category_list}"
        )

    # Handle accuracy files - support both single and multiple files
    if hasattr(args, "accuracy1") and args.accuracy1:
        accuracy_paths = [args.accuracy1, args.accuracy2, args.accuracy3]
        print("Using multiple accuracy files")
    elif args.accuracy:
        accuracy_paths = [args.accuracy] * 3
        print(f"Using single accuracy file for all panels: {args.accuracy}")
    else:
        accuracy_paths = None
        print("No accuracy files provided")

    # Validate data paths exist (but continue even if some don't)
    data_paths = [args.data, args.data2, args.data3]
    for i, path in enumerate(data_paths):
        if not path or not path.exists():
            print(f"Warning: Data file {i+1} not found: {path}")

    try:
        # Create and save the triptych
        fig = create_response_triptych(
            data_paths=data_paths,
            parameter=args.parameter,
            category_list=category_list,
            dt=args.dt,
            accuracy_paths=accuracy_paths,
            outlier_threshold=args.outlier_threshold,
            config=config,
        )

        save_plot(args.output)
        print(f"Response triptych saved to: {args.output}")
    except Exception as e:
        print(f"Error creating triptych: {e}")
        # Create a minimal figure with error message
        fig, ax = plt.subplots(
            figsize=(LAYOUT["figure_width"], LAYOUT["figure_height"])
        )
        ax.text(
            0.5,
            0.5,
            f"Error creating triptych:\n{str(e)}",
            ha="center",
            va="center",
            fontsize=16,
            alpha=0.6,
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        save_plot(args.output)
        print(f"Error plot saved to: {args.output}")
        raise
