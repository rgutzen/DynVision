"""plot_response_tripytch.py
"""

import argparse
import json
import re
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

# Global layout configuration for consistent positioning
LAYOUT = {
    # Figure dimensions
    "figure_width": 20,
    "figure_height": 16,
    # Column layout
    "column_left_positions": [0.08, 0.37, 0.66],
    "column_width": 0.26,
    # Vertical layout with improved spacing
    "title_top": 0.92,
    "title_height": 0.05,
    "accuracy_top": 0.82,
    "accuracy_height": 0.09,
    "accuracy_gap": 0.04,
    "timestep_accuracy_top": 0.69,
    "timestep_accuracy_height": 0.09,
    "timestep_accuracy_gap": 0.06,
    "legend_top": 0.54,
    "legend_height": 0.06,
    "legend_gap": 0.04,
    "dynamics_top": 0.44,
    "dynamics_height": 0.36,
    # Panel letters positioning
    "column_letter_y": 0.95,
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
LINEWIDTH_INDICATOR = 3
ALPHA_INDICATOR = 0.6

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
parser.add_argument("--parameter", type=str, required=True, help="Parameter to plot")
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

    if symbol is None:
        # Fallback mappings
        mappings = {
            "tau": ("Varying Time Constant", r"$\tau$"),
            "trc": ("Varying Recurrence Delay", r"$\Delta_{RC}$"),
            "tsk": ("Varying Skip Delay", r"$\Delta_{SK}$"),
            "rctarget": ("Varying Recurrence Target", "Target"),
            "lossrt": ("Varying Loss Reaction Time", r"$t_{loss}$"),
            "feedback": ("Varying Feedback", "Feedback"),
        }
        title, symbol = mappings.get(
            category, (f"Varying {category.upper()}", category)
        )
    else:
        title = f"Varying {category.capitalize()}"

    return title, symbol


def get_improved_color_palette(category_values, config=None):
    """Generate a perceptually uniform sequential color palette."""
    # Try to get colors from config first
    if config:
        colors = {}
        for val in category_values:
            color = get_color(val, config)
            if color:
                colors[val] = color

        if colors:
            model_order = sorted(category_values)
            return model_order, colors

    # Use perceptually uniform sequential palette
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
        value = match.group(1)
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
                actual_label = non_negative_labels.iloc[0]
                # Replace all -1 values for this sample with the actual label
                df_fixed.loc[
                    (df_fixed["sample_index"] == sample_idx)
                    & (df_fixed["label_index"] == -1),
                    "label_index",
                ] = actual_label

        # Step 2: Calculate accuracy per timestep and category
        timestep_accuracy_data = []

        for cat_value in df_fixed[category].unique():
            cat_data = df_fixed[df_fixed[category] == cat_value]

            for timestep in sorted(cat_data["times_index"].unique()):
                timestep_data = cat_data[cat_data["times_index"] == timestep]

                # Calculate accuracy for this timestep and category
                correct_predictions = (
                    timestep_data["guess_index"] == timestep_data["label_index"]
                ).sum()
                total_predictions = len(timestep_data)
                accuracy = (
                    correct_predictions / total_predictions
                    if total_predictions > 0
                    else 0
                )

                timestep_accuracy_data.append(
                    {
                        "times_index": timestep,
                        category: cat_value,
                        "timestep_accuracy": accuracy,
                    }
                )

        return pd.DataFrame(timestep_accuracy_data)

    except Exception as e:
        print(f"Error calculating timestep accuracy: {e}")
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
    """Plot classification accuracy over timesteps without error bars."""
    left = LAYOUT["column_left_positions"][column_index]
    width = LAYOUT["column_width"]

    ax = fig.add_axes(
        [
            left,
            LAYOUT["timestep_accuracy_top"],
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

            # Plot timestep accuracy without error bars
            sns.lineplot(
                data=timestep_accuracy_df,
                x="time_ms",
                y="timestep_accuracy",
                hue=category,
                hue_order=model_order,
                palette=colors,
                ax=ax,
                legend=False,
                linewidth=LINEWIDTH_MAIN,
                marker="o",
                markersize=4,
                errorbar=None,
                alpha=LINE_ALPHA,
            )

            # Set y-axis ticks for accuracy (max 1.0, 0.8, 0.6, 0.4, 0.2, 0.0)
            ax.set_ylim(0.0, 1.0)
            ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

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

    ax.grid(True, alpha=0.3)
    sns.despine(ax=ax, left=True, bottom=True)
    return ax


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
            columns = [
                col
                for col in accuracy_df.columns
                if pattern in col and not ("_MIN" in col or "_MAX" in col)
            ]

            print(f"  DEBUG: Found {len(columns)} {acc_type.lower()} accuracy columns")

            for col in columns:
                param_value = extract_param_value(col, category)
                if param_value:
                    # Clean the data and add to list
                    clean_data = accuracy_df[col].dropna()
                    for idx, acc_value in clean_data.items():
                        accuracy_data.append(
                            {
                                "epoch": idx,
                                "category": param_value,
                                "accuracy": acc_value,
                                "type": acc_type,
                            }
                        )

    except Exception as e:
        print(f"Error extracting accuracy data for category '{category}': {e}")
        return pd.DataFrame()

    print(f"  DEBUG: Extracted {len(accuracy_data)} accuracy data points")
    if accuracy_data:
        unique_values = pd.DataFrame(accuracy_data)[category].unique()
        print(f"  DEBUG: Unique {category} values found: {unique_values}")

    return pd.DataFrame(accuracy_data)


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
    """Plot training and validation accuracy at the top of each panel."""
    left = LAYOUT["column_left_positions"][column_index]
    width = LAYOUT["column_width"]

    ax = fig.add_axes([left, LAYOUT["accuracy_top"], width, LAYOUT["accuracy_height"]])
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
    else:
        try:
            # Plot training (solid) and validation (dotted) accuracy without error bars
            for acc_type, linestyle in [("Training", "-"), ("Validation", ":")]:
                type_data = accuracy_df[accuracy_df["type"] == acc_type]
                if not type_data.empty:
                    sns.lineplot(
                        data=type_data,
                        x="epoch",
                        y="accuracy",
                        hue=category,
                        hue_order=model_order,
                        palette=colors,
                        linestyle=linestyle,
                        ax=ax,
                        legend=False,
                        linewidth=LINEWIDTH_MAIN,
                        alpha=LINE_ALPHA,
                        errorbar=None,
                    )

        except Exception as e:
            print(f"Error plotting accuracy data: {e}")
            ax.text(
                0.5,
                0.5,
                "Error plotting accuracy",
                ha="center",
                va="center",
                fontsize=14,
                alpha=0.6,
            )

    # Allow y-axis to auto-scale for better data visibility
    # Only set lower bound if data exists
    if accuracy_df is not None and not accuracy_df.empty:
        # Let matplotlib auto-scale, but ensure we don't go above 1.0 for accuracy
        current_ylim = ax.get_ylim()
        ax.set_ylim(
            bottom=max(0.0, current_ylim[0] - 0.05),
            top=min(1.0, current_ylim[1] + 0.05),
        )

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
            fontsize=FONTSIZE_LEGEND,
        )

    ax.grid(True, alpha=0.3)
    sns.despine(ax=ax, left=True, bottom=True)
    return ax


def create_empty_panel(column_index, fig, category, show_ylabel=False, config=None):
    """Create an empty panel with appropriate messaging."""
    left = LAYOUT["column_left_positions"][column_index]
    width = LAYOUT["column_width"]

    # Add title
    title, _ = get_title_and_symbol(category, config)
    title_ax = fig.add_axes([left, LAYOUT["title_top"], width, LAYOUT["title_height"]])
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
            "Layer Power", labelpad=8, fontsize=FONTSIZE_AXIS_LABELS, fontweight="bold"
        )
    else:
        ax.set_ylabel("")

    ax.set_xlabel("Time (ms)", fontsize=FONTSIZE_AXIS_LABELS, labelpad=8)
    ax.tick_params(labelsize=FONTSIZE_TICK_LABELS)
    sns.despine(ax=ax, left=True, bottom=True)

    return [ax]  # Return as list to maintain consistency


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
            [left, LAYOUT["legend_top"], width, LAYOUT["legend_height"]]
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
            if val in panel_colors:
                color = panel_colors[val]
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

            # Style the legend title
            if legend.get_title():
                legend.get_title().set_fontweight("bold")

    return None


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

        # Calculate y-positions based on actual plot positions
        plot_y_positions = [
            LAYOUT["accuracy_top"]
            + LAYOUT["accuracy_height"]
            - 0.01,  # Top of accuracy plot
            LAYOUT["timestep_accuracy_top"]
            + LAYOUT["timestep_accuracy_height"]
            - 0.01,  # Top of timestep accuracy plot
            LAYOUT["dynamics_top"] - 0.01,  # Top of dynamics plot
        ]

        for j, (letter, y_pos) in enumerate(zip(plot_letters, plot_y_positions)):
            fig.text(
                left + LAYOUT["subpanel_letter_x_offset"],
                y_pos,
                letter,
                fontsize=FONTSIZE_PANEL_LABELS - 2,
                fontweight="bold",
                ha="center",
                va="center",
            )


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

    power_columns = [col for col in df.columns if col.endswith("_power")]
    if not power_columns:
        print(f"Panel {column_index} - No power columns found in data")
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
        col.replace("_power", "") for col in df.columns if col.endswith("_power")
    ]
    layer_names = order_layers(all_layer_names, config)

    if not layer_names:
        print(f"Panel {column_index} - No valid layer names found")
        return (
            create_empty_panel(column_index, fig, category, show_ylabel, config),
            None,
            None,
        )

    category_values = sorted(df[category].unique())
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
    title_ax = fig.add_axes([left, LAYOUT["title_top"], width, LAYOUT["title_height"]])
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

        # Plot layer power with proper time scaling
        power_col = f"{layer_name}_power"
        if power_col in df_with_time.columns:
            sns.lineplot(
                data=df_with_time,
                x="time_ms",
                y=power_col,
                ax=ax,
                marker=".",
                hue=category,
                hue_order=model_order,
                palette=colors,
                legend=False,
                linewidth=LINEWIDTH_MAIN,
                errorbar=None,
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
                "Layer Power",
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


def create_response_triptych(
    data_paths,
    parameter,
    category_list,
    dt,
    accuracy_paths=None,
    outlier_threshold=10.0,
    config=None,
):
    """Create a triptych (three-panel) figure with response plots.

    Args:
        outlier_threshold: Y-axis values beyond ±this threshold are considered outliers
                          and excluded from global y-axis scaling (default: 10.0)
        config: Configuration dictionary with palette, naming, and ordering
    """
    print(f"Creating response triptych with {len(data_paths)} panels")

    sns.set_context("talk")
    # Use global layout configuration
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
                    print(f"Warning: Data file is empty: {data_path}")
                    df = None
            else:
                print(f"Warning: Data file not found: {data_path}")
        except Exception as e:
            print(f"Error loading data from {data_path}: {e}")
            df = None

        # Load panel-specific accuracy data with error handling
        accuracy_df = None
        if accuracy_paths and i < len(accuracy_paths) and accuracy_paths[i]:
            try:
                if accuracy_paths[i].exists():
                    print(f"Loading accuracy data from: {accuracy_paths[i]}")
                    accuracy_df = pd.read_csv(accuracy_paths[i])
                    if accuracy_df.empty:
                        print(f"Warning: Accuracy file is empty: {accuracy_paths[i]}")
                        accuracy_df = None
                else:
                    print(f"Warning: Accuracy file not found: {accuracy_paths[i]}")
            except Exception as e:
                print(f"Error loading accuracy data from {accuracy_paths[i]}: {e}")
                accuracy_df = None

        # Create panel (will create empty panel if data is None/invalid)
        try:
            panel_axes, panel_colors, panel_order = plot_response_panel(
                df=df,
                parameter=parameter,
                category=category,
                column_index=i,
                fig=fig,
                dt=dt,
                accuracy_df=accuracy_df,
                show_layer_indicators=(
                    i == 2
                ),  # Still used for validation but all panels now have indicators
                show_ylabel=(i == 0),
                config=config,
            )
        except Exception as e:
            print(f"Error creating panel {i+1}: {e}")
            # Create empty panel as fallback
            panel_axes = create_empty_panel(
                i, fig, category, show_ylabel=(i == 0), config=config
            )
            panel_colors = None
            panel_order = None

        if panel_axes:
            all_axes.extend(panel_axes)
            panel_axes_list.append(panel_axes)
        else:
            panel_axes_list.append([])

        panel_colors_list.append(panel_colors)
        panel_orders_list.append(panel_order)

        # Only collect y-limits from panels with actual data
        if (
            df is not None
            and not df.empty
            and len([col for col in df.columns if col.endswith("_power")]) > 0
            and panel_axes
            and len(panel_axes) > 1  # Real data panel
        ):
            all_y_limits.extend([ax.get_ylim() for ax in panel_axes])

    # Create horizontal legend within each column
    if any(colors is not None for colors in panel_colors_list):
        create_horizontal_legend(
            fig,
            [c for c in panel_colors_list if c is not None],
            [o for o in panel_orders_list if o is not None],
            category_list,
            format_legend_label,
            config,
        )

    # Apply global y-limits only if we have valid data
    if all_y_limits:
        # Filter out outliers to prevent extreme values from dominating the scale
        filtered_y_limits = []

        for y_min, y_max in all_y_limits:
            # Only keep limits within reasonable range
            if abs(y_min) <= outlier_threshold and abs(y_max) <= outlier_threshold:
                filtered_y_limits.append((y_min, y_max))

        if filtered_y_limits:
            # Set global lower limit to 0, upper limit from filtered data
            global_y_min = 0.0
            global_y_max = max(lim[1] for lim in filtered_y_limits)

            print(f"Setting global y-limits: [{global_y_min:.3f}, {global_y_max:.3f}]")
            if len(filtered_y_limits) < len(all_y_limits):
                n_outliers = len(all_y_limits) - len(filtered_y_limits)
                print(
                    f"Filtered out {n_outliers} outlier subplot(s) with extreme y-limits (>±{outlier_threshold})"
                )
        else:
            print(
                f"All y-limits are outliers (>±{outlier_threshold}) - using default range [0, 5]"
            )
            global_y_min = 0.0
            global_y_max = 5.0

        # Set common y-ticks and apply to all panels
        if global_y_min is not None and global_y_max is not None:
            y_range = global_y_max - global_y_min
            if y_range > 0:  # Avoid division by zero
                # Apply to all panels with actual data (more than 1 axis = real data panel)
                for panel_idx, panel_axes in enumerate(panel_axes_list):
                    if (
                        len(panel_axes) > 1
                    ):  # Real data panel with multiple layer subplots
                        for subplot_idx, ax in enumerate(panel_axes):
                            # Apply global scaling to ALL subplots (including outliers)
                            ax.set_ylim(global_y_min, global_y_max)
                            # Set y-ticks to only 0 and 1 for lower plots
                            ax.set_yticks([0, 1])
                            ax.set_yticklabels(["0", "1"])

                            # Only show tick labels on leftmost panel
                            if panel_idx > 0:
                                ax.set_yticklabels([])

        # Add label indicator AFTER global y-scale adjustment
        for panel_idx, (panel_axes, data_path, category) in enumerate(
            zip(panel_axes_list, data_paths, category_list)
        ):
            if len(panel_axes) > 1:  # Real data panel
                try:
                    # Load data again for label indicator
                    if data_path and data_path.exists():
                        df = pd.read_csv(data_path)
                        if not df.empty and category in df.columns:
                            # Add label indicator to bottom subplot only
                            bottom_ax = panel_axes[-1]
                            y_min, y_max = bottom_ax.get_ylim()
                            label_indicator_df = calculate_label_indicator(
                                df, category, (y_min, y_max)
                            )
                            # Convert times_index to time_ms for label indicator
                            label_indicator_df["time_ms"] = (
                                label_indicator_df["times_index"] * dt
                            )
                            bottom_ax.plot(
                                label_indicator_df.time_ms,
                                label_indicator_df.label_indicator,
                                color="k",
                                linewidth=LINEWIDTH_INDICATOR,
                                drawstyle="steps-mid",
                                alpha=ALPHA_INDICATOR,
                            )
                except Exception as e:
                    print(
                        f"Warning: Could not add label indicator to panel {panel_idx}: {e}"
                    )

    else:
        print("No valid y-limit data found - skipping global y-axis scaling")

    # Align y-labels only for panels with data
    valid_axes = [
        ax
        for panel_axes in panel_axes_list
        for ax in panel_axes
        if len(panel_axes) > 1
    ]
    if valid_axes:
        fig.align_ylabels(valid_axes)

    return fig


if __name__ == "__main__":
    args, unknown = parser.parse_known_args()

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
