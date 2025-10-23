"""plot_response_tripytch.py

Refactored triptych plotting script using plot_responses.py functionality.
Creates three-column response plots with training accuracy, performance, and ridge plots.
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from dynvision.utils.visualization_utils import (
    save_plot,
    load_config_from_args,
    get_display_name,
)
from dynvision.visualization.plot_responses import (
    plot_temporal_ridge_responses,
    _extract_dimension_values,
    _get_colors_for_dimension,
    _format_legend_label,
    _plot_accuracy_panel,
    _add_horizontal_legend,
    LAYOUT as RESPONSES_LAYOUT,
    FORMATTING as RESPONSES_FORMATTING,
)

# Configure logging
logger = logging.getLogger(__name__)

# Triptych-specific layout configuration with relative positioning
TRIPTYCH_LAYOUT = {
    # Figure dimensions in inches
    "figure_width": 21,
    "figure_height": 16,
    # Column layout (relative coordinates 0-1)
    "n_columns": 3,
    "column_spacing": 0.04,  # Space between columns
    "left_margin": 0.06,
    "right_margin": 0.02,
    # Vertical layout (relative coordinates 0-1)
    "title_height": 0.03,
    "title_pad": 0.02,
    "training_accuracy_height": 0.13,  # Same height as performance panel
    "training_accuracy_pad": 0.07,
    "performance_height": 0.13,
    "performance_pad": 0.05,
    "legend_height": 0.08,
    "legend_pad": 0.01,
    "ridge_height": 0.5,  # Remaining space for ridge plots
    # Panel letters
    "panel_letter_offset_x": -0.04,
    "panel_letter_offset_y": 0.02,
}

# Formatting configuration
TRIPTYCH_FORMATTING = {
    **RESPONSES_FORMATTING,
    "fontsize_panel_label": 18,
}


def _calculate_column_positions(layout: Dict) -> List[float]:
    """Calculate left positions for each column."""
    available_width = 1 - layout["left_margin"] - layout["right_margin"]
    column_width = (
        available_width - (layout["n_columns"] - 1) * layout["column_spacing"]
    ) / layout["n_columns"]

    positions = []
    for i in range(layout["n_columns"]):
        left = layout["left_margin"] + i * (column_width + layout["column_spacing"])
        positions.append(left)

    return positions, column_width


def _get_title_and_symbol(
    category: str, config: Optional[Dict] = None
) -> Tuple[str, str]:
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
            "tsteps": ("Training Timesteps", r"$T$"),
            "idle": ("Initial Idle Time", r"$t_{idle}$"),
            "feedback": ("Feedback Connections", "Feedback"),
            "skip": ("Skip Connections", "Skip"),
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
            "tsteps": "Training Timesteps",
            "idle": "Initial Idle Time",
            "feedback": "Feedback Connections",
            "skip": "Skip Connections",
        }
        category_name = name_mappings.get(category, category.capitalize())

    # Only add full name if symbol contains '$' (is a mathematical symbol)
    if symbol and "$" in symbol:
        title = f"Varying {category_name} ({symbol})"
    else:
        title = f"Varying {category_name}"

    return title, symbol


def _add_panel_letters(
    fig: plt.Figure, column_positions: List[float], layout: Dict
) -> None:
    """Add panel letters A), B), C) for columns and i), ii), iii) for plots within columns."""
    fmt = TRIPTYCH_FORMATTING

    # Column letters A), B), C)
    column_letters = ["A)", "B)", "C)"]
    for i, left in enumerate(column_positions):
        fig.text(
            left + layout["panel_letter_offset_x"],
            1 - layout["panel_letter_offset_y"],
            column_letters[i],
            fontsize=fmt["fontsize_panel_label"],
            fontweight="bold",
            ha="center",
            va="top",
        )

        # Panel letters within each column: i), ii), iii)
        plot_letters = ["i)", "ii)", "iii)"]

        # Calculate y-positions for each subplot
        title_top = 1 - layout["title_height"]
        training_top = title_top - layout["title_pad"]
        performance_top = (
            training_top
            - layout["training_accuracy_height"]
            - layout["training_accuracy_pad"]
        )
        ridge_top = (
            performance_top
            - layout["performance_height"]
            - layout["performance_pad"]
            - layout["legend_height"]
            - layout["legend_pad"]
        )

        plot_y_positions = [
            training_top + layout["panel_letter_offset_y"],  # Above training accuracy
            performance_top + layout["panel_letter_offset_y"],  # Above performance
            ridge_top + layout["panel_letter_offset_y"],  # Above ridge plots
        ]

        for j, (letter, y_pos) in enumerate(zip(plot_letters, plot_y_positions)):
            fig.text(
                left + layout["panel_letter_offset_x"] + 0.01,
                y_pos,
                letter,
                fontsize=fmt["fontsize_panel_label"] - 2,
                fontweight="bold",
                ha="left",
                va="top",
            )


def _filter_first_parameter_value(
    df: pd.DataFrame, parameter_key: str
) -> pd.DataFrame:
    """Filter to use only the first parameter value (numerically sorted)."""
    if parameter_key not in df.columns:
        logger.warning(f"Parameter '{parameter_key}' not found in data")
        return df

    unique_params = df[parameter_key].unique()

    if len(unique_params) > 1:
        # Try to sort numerically
        try:
            sorted_params = sorted(unique_params, key=float)
            first_param = sorted_params[0]
            logger.warning(
                f"Multiple parameter values found: {unique_params}. "
                f"Using first value: {first_param}"
            )
        except (ValueError, TypeError):
            # Fall back to string sorting
            sorted_params = sorted(unique_params, key=str)
            first_param = sorted_params[0]
            logger.warning(
                f"Multiple parameter values found: {unique_params}. "
                f"Using first value (alphabetically): {first_param}"
            )

        # Filter to first parameter value only
        filtered_df = df[df[parameter_key] == first_param].copy()
        logger.info(f"Filtered from {len(df)} to {len(filtered_df)} rows")
        return filtered_df

    return df


def _standardize_category_value(value: str) -> str:
    """Standardize category value formatting consistently across all processing."""
    value_str = str(value).strip()
    # Handle boolean-like values consistently
    if value_str.lower() in ["true", "false"]:
        return value_str.lower()
    return value_str


def _extract_training_accuracy_data(
    accuracy_df: pd.DataFrame, category_key: str
) -> pd.DataFrame:
    """Extract training accuracy data for the specified category variation."""
    if accuracy_df is None or accuracy_df.empty:
        return pd.DataFrame()

    logger.debug(f"Extracting training accuracy for category '{category_key}'")
    logger.debug(f"Available columns: {list(accuracy_df.columns)}")

    accuracy_data = []

    try:
        # Find training and validation accuracy columns (exclude MIN/MAX)
        for acc_type, pattern in [
            ("Training", "train_accuracy"),
            ("Validation", "val_accuracy"),
        ]:
            matching_cols = [
                col
                for col in accuracy_df.columns
                if pattern in col and not any(x in col for x in ["__MIN", "__MAX"])
            ]

            logger.debug(
                f"Found {len(matching_cols)} {acc_type.lower()} accuracy columns"
            )

            for col in matching_cols:
                # Extract parameter value from column name using regex
                import re

                pattern_regex = f"{category_key}=([^+_]+)"
                match = re.search(pattern_regex, col, re.IGNORECASE)

                if match:
                    param_value = _standardize_category_value(match.group(1))
                    logger.debug(f"Column '{col}' -> parameter value: '{param_value}'")

                    # Add data points for this column
                    for idx, row in accuracy_df.iterrows():
                        if not pd.isna(row[col]):
                            accuracy_data.append(
                                {
                                    "epoch": row["epoch"],
                                    "accuracy": row[col],
                                    "type": acc_type,
                                    category_key: param_value,
                                }
                            )

    except Exception as e:
        logger.error(f"Error extracting training accuracy data: {e}")
        return pd.DataFrame()

    if accuracy_data:
        result_df = pd.DataFrame(accuracy_data)
        logger.info(f"Extracted {len(accuracy_data)} training accuracy data points")
        return result_df
    else:
        logger.warning(
            f"No training accuracy data extracted for category '{category_key}'"
        )
        return pd.DataFrame()


def _plot_training_accuracy_panel(
    fig: plt.Figure,
    column_left: float,
    column_width: float,
    bottom: float,
    height: float,
    accuracy_df: pd.DataFrame,
    category_key: str,
    hue_values: List[str],
    colors: Dict[str, str],
    show_ylabel: bool = False,
    show_legend: bool = False,
) -> None:
    """Plot training and validation accuracy panel."""
    ax = fig.add_axes([column_left, bottom, column_width, height])
    ax.patch.set_alpha(0)

    if accuracy_df is None or accuracy_df.empty:
        ax.text(
            0.5,
            0.5,
            "No training accuracy data",
            ha="center",
            va="center",
            fontsize=12,
            alpha=0.6,
        )
        logger.warning("No training accuracy data to plot")
    else:
        logger.info(f"Plotting training accuracy data with {len(accuracy_df)} rows")

        try:
            # Ensure category values are standardized and match hue_values
            if category_key in accuracy_df.columns:
                accuracy_df = accuracy_df.copy()
                accuracy_df[category_key] = accuracy_df[category_key].apply(
                    _standardize_category_value
                )

            # Plot training accuracy (solid lines)
            train_data = accuracy_df[accuracy_df["type"] == "Training"]
            if not train_data.empty:
                sns.lineplot(
                    data=train_data,
                    x="epoch",
                    y="accuracy",
                    hue=category_key,
                    hue_order=hue_values,
                    palette=colors,
                    ax=ax,
                    legend=False,
                    linewidth=TRIPTYCH_FORMATTING["linewidth_main"],
                    linestyle="-",
                    alpha=TRIPTYCH_FORMATTING["alpha_line"],
                )

            # Plot validation accuracy (dotted lines)
            val_data = accuracy_df[accuracy_df["type"] == "Validation"]
            if not val_data.empty:
                sns.lineplot(
                    data=val_data,
                    x="epoch",
                    y="accuracy",
                    hue=category_key,
                    hue_order=hue_values,
                    palette=colors,
                    ax=ax,
                    legend=False,
                    linewidth=TRIPTYCH_FORMATTING["linewidth_main"],
                    linestyle=":",
                    alpha=TRIPTYCH_FORMATTING["alpha_line"],
                )

        except Exception as e:
            logger.error(f"Error plotting training accuracy: {e}")
            ax.text(
                0.5,
                0.5,
                f"Error plotting training accuracy",
                ha="center",
                va="center",
                fontsize=12,
                alpha=0.6,
            )

    # Formatting
    if show_ylabel:
        ax.set_ylabel(
            "Train Accuracy",
            fontsize=TRIPTYCH_FORMATTING["fontsize_axis"],
            fontweight="bold",
        )
    else:
        ax.set_ylabel("")
        ax.set_yticklabels([])

    ax.set_xlabel("Epoch", fontsize=TRIPTYCH_FORMATTING["fontsize_axis"])
    ax.tick_params(labelsize=TRIPTYCH_FORMATTING["fontsize_tick"])

    # Add legend for line styles on first panel
    if show_legend:
        legend_elements = [
            plt.Line2D(
                [0],
                [0],
                color="black",
                linewidth=TRIPTYCH_FORMATTING["linewidth_main"],
                linestyle="-",
                label="Training",
                alpha=TRIPTYCH_FORMATTING["alpha_line"],
            ),
            plt.Line2D(
                [0],
                [0],
                color="black",
                linewidth=TRIPTYCH_FORMATTING["linewidth_main"],
                linestyle=":",
                label="Validation",
                alpha=TRIPTYCH_FORMATTING["alpha_line"],
            ),
        ]
        ax.legend(
            handles=legend_elements,
            loc="lower right",
            frameon=False,
            fontsize=TRIPTYCH_FORMATTING["fontsize_legend"] - 2,
        )

    ax.grid(True, alpha=0.3)
    sns.despine(ax=ax, left=True, bottom=True)
    return ax


def create_triptych_plot(
    data_paths: List[Path],
    category_list: List[str],
    parameter_key: str,
    dt: float,
    accuracy_paths: Optional[List[Path]] = None,
    experiment: Optional[str] = None,
    config: Optional[Dict] = None,
) -> plt.Figure:
    """Create a triptych plot using plot_responses.py functionality."""

    logger.info("=" * 60)
    logger.info("Creating triptych plot")
    logger.info("=" * 60)

    layout = TRIPTYCH_LAYOUT
    fmt = TRIPTYCH_FORMATTING

    # Calculate column positions
    column_positions, column_width = _calculate_column_positions(layout)

    # Create figure
    fig = plt.figure(figsize=(layout["figure_width"], layout["figure_height"]))
    sns.set_context("talk")

    # Add panel letters
    _add_panel_letters(fig, column_positions, layout)

    # Process each column
    all_ridge_axes = []
    all_accuracy_axes = []
    column_data_list = []

    for col_idx, (data_path, category_key) in enumerate(
        zip(data_paths, category_list)
    ):
        logger.info(f"Processing column {col_idx + 1}: {category_key}")

        column_left = column_positions[col_idx]

        # Load main data
        df = None
        if data_path and data_path.exists():
            try:
                df = pd.read_csv(data_path)
                logger.info(f"Loaded {len(df)} rows from {data_path}")

                # Filter to first parameter value only
                df = _filter_first_parameter_value(df, parameter_key)

                # Standardize category values in the dataframe to match dimension extraction
                if category_key in df.columns:
                    df[category_key] = df[category_key].apply(
                        _standardize_category_value
                    )

            except Exception as e:
                logger.error(f"Error loading data from {data_path}: {e}")

        # Load training accuracy data
        accuracy_df = None
        if (
            accuracy_paths
            and col_idx < len(accuracy_paths)
            and accuracy_paths[col_idx]
        ):
            try:
                accuracy_df = pd.read_csv(accuracy_paths[col_idx])
                logger.info(
                    f"Loaded training accuracy data from {accuracy_paths[col_idx]}"
                )
                accuracy_df = _extract_training_accuracy_data(
                    accuracy_df, category_key
                )
            except Exception as e:
                logger.error(f"Error loading training accuracy data: {e}")

        # Store data for later y-scaling
        column_data_list.append((df, accuracy_df, category_key))

        if df is None or df.empty:
            logger.warning(f"No valid data for column {col_idx + 1}")
            continue

        # Extract dimension values and colors
        try:
            hue_values = _extract_dimension_values(
                df=df,
                dimension="category",
                dimension_key=category_key,
                config=config or {},
            )
            colors = _get_colors_for_dimension(
                values=hue_values, dimension_key=category_key, config=config or {}
            )
            logger.info(f"Found {len(hue_values)} category values: {hue_values}")

        except Exception as e:
            logger.error(f"Error processing dimension values: {e}")
            continue

        # Calculate panel positions
        title_top = 1 - layout["title_height"]
        training_bottom = (
            title_top - layout["title_pad"] - layout["training_accuracy_height"]
        )
        performance_bottom = (
            training_bottom
            - layout["training_accuracy_pad"]
            - layout["performance_height"]
        )
        legend_bottom = (
            performance_bottom - layout["performance_pad"] - layout["legend_height"]
        )
        ridge_bottom = legend_bottom - layout["legend_pad"] - layout["ridge_height"]

        # Add title
        title, symbol = _get_title_and_symbol(category_key, config)
        title_ax = fig.add_axes(
            [column_left, title_top, column_width, layout["title_height"]]
        )
        title_ax.text(
            0.5,
            0.5,
            title,
            ha="center",
            va="center",
            fontsize=fmt["fontsize_title"],
            fontweight="bold",
        )
        title_ax.set_xlim(0, 1)
        title_ax.set_ylim(0, 1)
        title_ax.axis("off")

        # Plot training accuracy panel
        acc_ax = _plot_training_accuracy_panel(
            fig=fig,
            column_left=column_left,
            column_width=column_width,
            bottom=training_bottom,
            height=layout["training_accuracy_height"],
            accuracy_df=accuracy_df,
            category_key=category_key,
            hue_values=hue_values,
            colors=colors,
            show_ylabel=(col_idx == 0),
            show_legend=(col_idx == 0),
        )
        all_accuracy_axes.append(acc_ax)

        # Plot performance panel (accuracy & confidence from test data)
        performance_ax = fig.add_axes(
            [
                column_left,
                performance_bottom,
                column_width,
                layout["performance_height"],
            ]
        )
        _plot_accuracy_panel(
            ax=performance_ax,
            data=df,
            hue_var="category",
            hue_key=category_key,
            hue_values=hue_values,
            colors=colors,
            dt=dt,
            show_ylabel=(col_idx == 0),
            show_legend=(col_idx == 0),
        )
        performance_ax.set_ylim(-0.005, 1)

        # Add horizontal legend
        legend_ax = fig.add_axes(
            [column_left, legend_bottom, column_width, layout["legend_height"]]
        )
        legend_ax.set_xlim(0, 1)
        legend_ax.set_ylim(0, 1)
        legend_ax.axis("off")
        legend_ax.patch.set_alpha(0)

        # Create legend elements using the ordered hue_values
        legend_elements = []
        legend_labels = []

        for val in hue_values:
            if val in colors:
                legend_elements.append(
                    plt.Line2D(
                        [0],
                        [0],
                        color=colors[val],
                        linewidth=fmt["linewidth_main"],
                        marker=".",
                        markersize=8,
                        alpha=fmt["alpha_line"],
                    )
                )
                # Use the same formatting function as the main plot_responses module
                legend_labels.append(
                    _format_legend_label(category_key, val, config or {}, dt)
                )

        if legend_elements:
            n_cols = min(len(legend_elements), 5)
            legend = legend_ax.legend(
                legend_elements,
                legend_labels,
                loc="center",
                ncol=n_cols,
                frameon=False,
                fontsize=fmt["fontsize_legend"],
                handlelength=2,
                handletextpad=0.5,
                columnspacing=1.0,
                title=symbol,
                title_fontsize=fmt["fontsize_legend"],
            )

        # Plot ridge plots using plot_responses functionality
        try:
            # Get layer values
            layer_values = _extract_dimension_values(
                df, "layers", "layers", config or {}
            )
            logger.info(f"Found {len(layer_values)} layers: {layer_values}")

            if layer_values:
                from dynvision.visualization.plot_responses import (
                    _plot_response_ridges,
                )

                # Calculate ridge positioning manually since we're using custom layout
                ridge_top = ridge_bottom + layout["ridge_height"]

                ridge_axes = _plot_response_ridges(
                    fig=fig,
                    column_left=column_left,
                    column_width=column_width,
                    data=df,
                    subplot_var="layers",
                    subplot_key="layers",
                    subplot_values=layer_values,
                    hue_var="category",
                    hue_key=category_key,
                    hue_values=hue_values,
                    colors=colors,
                    dt=dt,
                    show_ylabel=(col_idx == 0),
                    config=config or {},
                    # Override LAYOUT values for our custom positioning
                    ridge_height=layout["ridge_height"],
                    title_bot=ridge_top,  # Use ridge_top as the starting point
                    title_pad=0,  # No title in ridge section
                    accuracy_height=0,  # No accuracy in ridge section
                    accuracy_pad=0,
                    legend_height=0,  # No legend in ridge section
                    legend_pad=0,
                )

                all_ridge_axes.extend(ridge_axes)
                logger.info(
                    f"Created {len(ridge_axes)} ridge axes for column {col_idx + 1}"
                )

        except Exception as e:
            logger.error(f"Error creating ridge plots for column {col_idx + 1}: {e}")
            import traceback

            traceback.print_exc()

    # Apply consistent y-scaling across all ridge axes
    if all_ridge_axes:
        # Calculate global y-limits
        global_ymin, global_ymax = float("inf"), float("-inf")
        for ax in all_ridge_axes:
            ymin, ymax = ax.get_ylim()
            global_ymin = min(global_ymin, ymin)
            global_ymax = max(global_ymax, ymax)

        # Apply consistent limits
        global_ymin = max(global_ymin, fmt["min_global_ymin"])
        global_ymax = min(global_ymax, fmt["max_global_ymax"])

        for ax in all_ridge_axes:
            ax.set_ylim(global_ymin, global_ymax)

        logger.info(
            f"Applied consistent y-scaling: [{global_ymin:.3f}, {global_ymax:.3f}]"
        )

    # Adjust y-scaling for accuracy panels if needed
    if all_accuracy_axes:
        global_acc_ymin = min([ax.get_ylim()[0] for ax in all_accuracy_axes])
        for ax in all_accuracy_axes:
            ax.set_ylim(global_acc_ymin, 1.0)
            # ax.set_xlim(0, 100)

    fig.align_ylabels()

    logger.info("Triptych plot creation complete")
    return fig


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Create response triptych plots using plot_responses.py functionality"
    )
    parser.add_argument("--data", type=Path, help="Path to first dataset")
    parser.add_argument("--data2", type=Path, help="Path to second dataset")
    parser.add_argument("--data3", type=Path, help="Path to third dataset")
    parser.add_argument(
        "--accuracy1", type=Path, help="Path to first training accuracy CSV file"
    )
    parser.add_argument(
        "--accuracy2", type=Path, help="Path to second training accuracy CSV file"
    )
    parser.add_argument(
        "--accuracy3", type=Path, help="Path to third training accuracy CSV file"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Path to output file"
    )
    parser.add_argument(
        "--parameter", type=str, required=True, help="Parameter column name"
    )
    parser.add_argument(
        "--category",
        type=str,
        required=True,
        help="Categories to plot (space-separated)",
    )
    parser.add_argument(
        "--dt",
        type=float,
        required=True,
        help="Temporal resolution in ms per timestep",
    )
    parser.add_argument("--experiment", type=str, help="Experiment name for title")
    parser.add_argument(
        "--palette", type=str, help="JSON formatted dictionary of colors"
    )
    parser.add_argument("--naming", type=str, help="JSON formatted naming dictionary")
    parser.add_argument(
        "--ordering", type=str, help="JSON formatted ordering dictionary"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s"
    )

    # Load configuration
    config = load_config_from_args(
        palette_str=args.palette,
        naming_str=args.naming,
        ordering_str=args.ordering,
    )

    # Parse categories
    category_list = args.category.split()
    if len(category_list) != 3:
        raise ValueError(
            f"Expected 3 categories, got {len(category_list)}: {category_list}"
        )

    # Prepare data paths
    data_paths = [args.data, args.data2, args.data3]
    accuracy_paths = [args.accuracy1, args.accuracy2, args.accuracy3]

    # Validate that we have some data
    valid_data_paths = [p for p in data_paths if p and p.exists()]
    if not valid_data_paths:
        raise ValueError("No valid data files found")

    logger.info(f"Processing {len(valid_data_paths)} data files")
    logger.info(f"Categories: {category_list}")
    logger.info(f"Parameter: {args.parameter}")
    logger.info(f"Temporal resolution: {args.dt} ms")

    try:
        # Create the triptych plot
        fig = create_triptych_plot(
            data_paths=data_paths,
            category_list=category_list,
            parameter_key=args.parameter,
            dt=args.dt,
            accuracy_paths=accuracy_paths,
            experiment=args.experiment,
            config=config,
        )

        # Save the plot
        save_plot(args.output)
        logger.info(f"Triptych plot saved to: {args.output}")

    except Exception as e:
        logger.error(f"Error creating triptych plot: {e}")
        raise


if __name__ == "__main__":
    main()
