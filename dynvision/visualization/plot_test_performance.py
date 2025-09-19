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
    calculate_label_indicator,
    get_category_plotting_settings,
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
FIGURE_HEIGHT_PER_SUBPLOT = 2.5
SUBPLOT_SPACING = 0.1

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data",
    type=Path,
    required=True,
    help="Path to processed test performance CSV file",
)
parser.add_argument(
    "--output", type=Path, required=True, help="Path to output plot file"
)
parser.add_argument("--topk", type=int, default=5, help="Top-k value to plot")
parser.add_argument("--palette", type=str, help="JSON formatted dictionary of colors")
parser.add_argument("--dt", type=float, help="Time step duration in milliseconds")
parser.add_argument("--naming", type=str, help="JSON formatted naming dictionary")
parser.add_argument("--ordering", type=str, help="JSON formatted ordering dictionary")
parser.add_argument(
    "--parameter", type=str, help="Column name containing experiment parameter values"
)
parser.add_argument("--experiment", type=str, help="name of the testing scenario")
parser.add_argument(
    "--category", type=str, help="Column name containing model categories"
)


def translate_name(name, config):
    """Translate name using the naming dictionary from config."""
    if config and "naming" in config:
        # Try exact match first
        if name in config["naming"]:
            return config["naming"][name]
        # Try string version
        if str(name) in config["naming"]:
            return config["naming"][str(name)]
    return name


def plot_accuracy_and_confidence(
    df,
    topk=5,
    config=None,
    dt=None,
    output_path=None,
    experiment=None,
    category=None,
    parameter=None,
):
    """Create a figure showing accuracy and confidence over time for each model category."""

    print(f"Plot data shape: {df.shape}")
    print(f"Available columns: {df.columns.tolist()}")

    # Check if we should plot top-k accuracy
    plot_topk = topk is not None and topk > 1

    if not plot_topk:
        print(
            f"Top-k plotting disabled (topk={topk}). Only plotting regular accuracy and confidence."
        )
        topk_col = None
        actual_k = None
    else:
        # Check which top-k accuracy columns are available
        topk_cols = [col for col in df.columns if col.startswith("accuracy_top")]
        print(f"Available top-k accuracy columns: {topk_cols}")

        # Select the top-k column to use
        topk_col = f"accuracy_top{topk}"
        if topk_col not in df.columns and topk_cols:
            topk_col = topk_cols[0]
            actual_k = int(topk_col.split("accuracy_top")[1])
            print(f"Requested top-{topk} not available, using {topk_col} instead")
        elif topk_col not in df.columns:
            print(f"No top-k accuracy columns found, plotting only standard accuracy")
            topk_col = None
            actual_k = None
        else:
            actual_k = topk
            print(f"Using {topk_col} for plotting")

    # Determine category and parameter columns
    expected_cols = [
        "presentation_label",
        "times_index",
        "label_index",
        "accuracy",
        "confidence_avg",
        "confidence_std",
    ]

    # Add all top-k columns to expected list
    topk_cols_all = [col for col in df.columns if col.startswith("accuracy_top")]
    expected_cols.extend(topk_cols_all)

    # Use CLI arguments to determine category and parameter columns
    if category is not None:
        if category not in df.columns:
            raise ValueError(
                f"Specified category column '{category}' not found in data. Available columns: {df.columns.tolist()}"
            )
        category_col = category
        print(f"Using category column from CLI args: {category_col}")
    else:
        # Fallback to original logic if no category specified
        category_cols = [col for col in df.columns if col not in expected_cols]
        if len(category_cols) < 2:
            raise ValueError(
                f"Expected at least 2 category/parameter columns, found: {category_cols}"
            )
        category_col = category_cols[-2]  # Second to last column
        print(f"Using category column from column order (fallback): {category_col}")

    if parameter is not None:
        if parameter not in df.columns:
            raise ValueError(
                f"Specified parameter column '{parameter}' not found in data. Available columns: {df.columns.tolist()}"
            )
        parameter_col = parameter
        print(f"Using parameter column from CLI args: {parameter_col}")
    else:
        # Fallback to original logic if no parameter specified
        category_cols = [col for col in df.columns if col not in expected_cols]
        if len(category_cols) < 2:
            raise ValueError(
                f"Expected at least 2 category/parameter columns, found: {category_cols}"
            )
        parameter_col = category_cols[-1]  # Last column
        print(f"Using parameter column from column order (fallback): {parameter_col}")

    print(f"Final selection - category: {category_col}, parameter: {parameter_col}")

    # Get plotting settings from config
    category_values = sorted(df[category_col].unique())
    model_order, colors = get_category_plotting_settings(
        category_col, category_values, config
    )

    # Get unique parameter values
    param_values = sorted(df[parameter_col].unique())
    n_params = len(param_values)

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

    # Create figure with reduced height and closer subplots
    fig, axes = plt.subplots(
        n_params,
        1,
        figsize=(10, FIGURE_HEIGHT_PER_SUBPLOT * n_params),
        sharex=True,
        sharey=True,
    )

    # Ensure axes is always a list
    if n_params == 1:
        axes = [axes]

    # Define metrics and their styles - updated markers
    metrics = ["Accuracy", "Confidence"]
    if topk_col and plot_topk:
        metrics.append(f"Top-{actual_k} Accuracy")

    line_styles = {
        "Accuracy": "-",  # solid
        "Confidence": ":",  # dotted
        f"Top-{actual_k} Accuracy": "--" if actual_k else None,  # dashed
    }

    markers = {
        "Accuracy": "o",
        "Confidence": "s",
        f"Top-{actual_k} Accuracy": "^" if actual_k else None,
    }

    # Prepare data for seaborn plotting
    all_plot_data = []

    for param_val in param_values:
        param_data = df[df[parameter_col] == param_val]

        for cat_val in model_order:
            if cat_val in param_data[category_col].values:
                cat_data = param_data[param_data[category_col] == cat_val]

                # Add data for accuracy
                for _, row in cat_data.iterrows():
                    all_plot_data.append(
                        {
                            time_col: row[time_col],
                            "value": row["accuracy"],
                            "metric": "Accuracy",
                            "category": cat_val,
                            "parameter_value": param_val,
                            "std": 0,
                        }
                    )

                # Add data for confidence
                for _, row in cat_data.iterrows():
                    all_plot_data.append(
                        {
                            time_col: row[time_col],
                            "value": row["confidence_avg"],
                            "metric": "Confidence",
                            "category": cat_val,
                            "parameter_value": param_val,
                            "std": row["confidence_std"],
                        }
                    )

                # Add data for top-k accuracy if available
                if topk_col and plot_topk and topk_col in cat_data.columns:
                    for _, row in cat_data.iterrows():
                        all_plot_data.append(
                            {
                                time_col: row[time_col],
                                "value": row[topk_col],
                                "metric": f"Top-{actual_k} Accuracy",
                                "category": cat_val,
                                "parameter_value": param_val,
                                "std": 0,
                            }
                        )

    # Convert to DataFrame for seaborn
    plot_df = pd.DataFrame(all_plot_data)

    # Plot for each parameter value
    for i, param_val in enumerate(param_values):
        param_data = df[df[parameter_col] == param_val]
        param_plot_data = plot_df[plot_df["parameter_value"] == param_val]
        ax = axes[i]

        print(f"Parameter {param_val}: {len(param_data)} rows")

        # Plot each metric using seaborn
        for metric in metrics:
            metric_data = param_plot_data[param_plot_data["metric"] == metric]

            if len(metric_data) == 0:
                continue

            # Use seaborn lineplot for each metric
            sns.lineplot(
                data=metric_data,
                x=time_col,
                y="value",
                hue="category",
                hue_order=model_order,
                palette=colors,
                linestyle=line_styles[metric],
                marker=markers[metric],
                markersize=6,
                linewidth=LINEWIDTH_MAIN,
                alpha=ALPHA_LINES,
                ax=ax,
                legend=False,
            )

            # Add error bands for confidence using fill_between
            if metric == "Confidence":
                for cat_val in model_order:
                    cat_metric_data = metric_data[metric_data["category"] == cat_val]
                    if len(cat_metric_data) > 0:
                        cat_metric_data = cat_metric_data.sort_values(time_col)
                        ax.fill_between(
                            cat_metric_data[time_col],
                            cat_metric_data["value"] - cat_metric_data["std"],
                            cat_metric_data["value"] + cat_metric_data["std"],
                            color=colors[cat_val],
                            alpha=0.2,
                        )

        # Add label indicator with updated styling
        if len(param_data) > 0:
            y_min, y_max = ax.get_ylim()
            label_indicator_df = calculate_label_indicator(
                param_data, category_col, (y_min, y_max)
            )

            # Convert time for indicator if needed
            if dt is not None:
                indicator_time = label_indicator_df.times_index * dt
            else:
                indicator_time = label_indicator_df.times_index

            ax.plot(
                indicator_time,
                label_indicator_df.label_indicator,
                color="k",  # Changed to 'k' (black)
                linewidth=LINEWIDTH_INDICATOR,
                drawstyle="steps-mid",
                alpha=0.6,  # Changed to 0.6
            )

        # Add parameter value as text box at the vertical center with naming translation
        param_symbol = translate_name(parameter_col, config)  # Use direct translation
        param_display = translate_name(param_val, config)  # Use direct translation

        # Fallback for time conversion if not translated
        if (
            param_display == param_val
            and dt is not None
            and parameter_col.lower()
            in [
                "duration",
                "interval",
                "stim",
                "stimulus",
            ]
        ):
            param_display = f"{int(param_val * dt)} ms"

        ax.text(
            0.98,
            0.5,  # Vertical center
            f"{param_symbol}={param_display}",
            horizontalalignment="right",
            verticalalignment="center",
            transform=ax.transAxes,
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                alpha=0.8,
                edgecolor="gray",
            ),
            fontsize=FONTSIZE_TICK_LABELS,
        )

        # Customize subplot - remove individual y-labels
        ax.set_ylabel("")  # Remove individual y-labels
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.1)  # Set lower limit a bit below 0
        ax.set_xlim(left=0)  # Start x-axis at 0

        # Remove top and right spines
        sns.despine(ax=ax)

    # Set x-label only on the bottom subplot
    axes[-1].set_xlabel(xlabel, fontsize=FONTSIZE_AXIS_LABELS)

    # Add single bold y-label at further distance from y-axis
    fig.text(
        0.05,  # Moved even further from y-axis (was 0.06)
        0.5,
        "Performance",
        rotation=90,
        verticalalignment="center",
        fontsize=FONTSIZE_AXIS_LABELS,
        fontweight="bold",
    )

    # Create combined legend with separate sections

    # Legend for model categories (colors)
    category_legend_elements = []
    for cat_val in model_order:
        category_legend_elements.append(
            plt.Line2D(
                [0],
                [0],
                color=colors[cat_val],
                linewidth=LINEWIDTH_MAIN,
                label=cat_val,
            )
        )

    # Legend for metrics (linestyles)
    metric_legend_elements = []
    for metric in metrics:
        metric_legend_elements.append(
            plt.Line2D(
                [0],
                [0],
                color="black",
                linestyle=line_styles[metric],
                marker=markers[metric],
                markersize=6,
                linewidth=LINEWIDTH_MAIN,
                label=metric,
            )
        )

    # Add stimulus indicator to metric legend
    metric_legend_elements.append(
        plt.Line2D(
            [0],
            [0],
            color="k",  # Changed to 'k' to match the plot
            linewidth=LINEWIDTH_INDICATOR,
            alpha=0.6,  # Changed to 0.6 to match the plot
            label="Stimulus",
        )
    )

    # Create combined legend
    all_legend_elements = []

    # Get category display name using naming dict translation
    category_display_name = translate_name(category_col, config)

    # Add category section with proper translation and visual centering using spaces
    all_legend_elements.append(
        plt.Line2D([0], [0], color="none", label=f"{category_display_name}")
    )
    all_legend_elements.extend(category_legend_elements)

    # Add spacing
    all_legend_elements.append(plt.Line2D([0], [0], color="none", label=""))

    # Add metric section with visual centering using spaces
    all_legend_elements.append(
        plt.Line2D([0], [0], color="none", label="Performance Metric")
    )
    all_legend_elements.extend(metric_legend_elements)

    # Position combined legend right next to subplots without frame
    legend = fig.legend(
        handles=all_legend_elements,
        bbox_to_anchor=(0.8, 0.5),  # Right next to subplots
        loc="center left",
        fontsize=FONTSIZE_LEGEND,
        frameon=False,  # Remove frame
    )

    # Style the section headers (bold) and visually center them using spaces
    legend_texts = legend.get_texts()
    if len(legend_texts) > 0:
        legend_texts[0].set_weight("bold")  # Category section header

    if len(legend_texts) > len(category_legend_elements) + 2:
        legend_texts[len(category_legend_elements) + 2].set_weight(
            "bold"
        )  # Metric section header

    # Note: Entries are left-aligned by default. To visually center section headers,
    # add leading/trailing spaces to their label strings when creating the legend.

    # Create enhanced title with parameter range and category
    title_parts = []

    # Derive experiment name from parameter column name or use experiment CLI arg as fallback
    experiment_name_for_title = None
    if experiment:
        # If experiment name was specified via CLI, use it for the title
        experiment_name_for_title = experiment
    else:
        # Try to derive experiment name from parameter column name
        if parameter_col.lower() in ["duration", "interval", "stim", "stimulus"]:
            experiment_name_for_title = "duration"
        elif "contrast" in parameter_col.lower():
            experiment_name_for_title = "contrast"
        elif "noise" in parameter_col.lower():
            experiment_name_for_title = "noise"
        elif "stability" in parameter_col.lower():
            experiment_name_for_title = "stability"
        elif "response" in parameter_col.lower():
            experiment_name_for_title = "response"

    if experiment_name_for_title:
        experiment_display = translate_name(
            f"{experiment_name_for_title}_experiment", config
        )
        if experiment_display != f"{experiment_name_for_title}_experiment":
            title_parts.append(experiment_display)
        else:
            title_parts.append(experiment_name_for_title.title())
    else:
        # Fallback: Extract experiment name from data path
        if output_path:
            path_parts = str(output_path).split("/")
            for part in path_parts:
                if "_experiment" in part or any(
                    exp in part
                    for exp in [
                        "duration",
                        "contrast",
                        "interval",
                        "stability",
                        "response",
                        "noise",
                        "uniformnoise",
                    ]
                ):
                    extracted_experiment_name = part.split("_")[0]
                    experiment_display = translate_name(
                        f"{extracted_experiment_name}_experiment", config
                    )
                    if experiment_display != f"{extracted_experiment_name}_experiment":
                        title_parts.append(experiment_display)
                    else:
                        title_parts.append(extracted_experiment_name.title())
                    break

    # Get parameter name using naming dict
    param_name_display = translate_name(parameter_col, config)

    # Create parameter value range string
    if len(param_values) > 1:
        param_min = min(param_values)
        param_max = max(param_values)

        # Translate min and max values if possible
        param_min_display = translate_name(param_min, config)
        param_max_display = translate_name(param_max, config)

        # Handle time conversion for parameter range if not translated
        if (
            param_min_display == param_min
            and dt is not None
            and parameter_col.lower()
            in [
                "duration",
                "interval",
                "stim",
                "stimulus",
            ]
        ):
            param_min_display = f"{int(param_min * dt)} ms"
            param_max_display = f"{int(param_max * dt)} ms"

        param_range = f"{param_min_display}-{param_max_display}"
    else:
        # Single parameter value
        param_val = param_values[0]
        param_range = translate_name(param_val, config)

        # Handle time conversion if not translated
        if (
            param_range == param_val
            and dt is not None
            and parameter_col.lower()
            in [
                "duration",
                "interval",
                "stim",
                "stimulus",
            ]
        ):
            param_range = f"{int(param_val * dt)} ms"

    # Format: <experiment> (<parameter_name> = [<parameter value range>] & <category name>)
    if title_parts:
        title = f"{title_parts[0]} ({parameter_col} ({param_name_display}) = [{param_range}]) & {category_display_name}"
    else:
        title = f"({param_name_display} = [{param_range}] & {category_display_name})"

    fig.suptitle(title, fontsize=FONTSIZE_TITLE, y=0.98)

    # Adjust layout with more space for y-label and right-positioned legend
    plt.tight_layout()
    plt.subplots_adjust(top=0.95, right=0.75, left=0.14, hspace=SUBPLOT_SPACING)

    return fig


if __name__ == "__main__":
    args = parser.parse_args()

    print(f"Loading processed test performance data from: {args.data}")
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
        # Generate the accuracy and confidence plot
        fig = plot_accuracy_and_confidence(
            df,
            topk=args.topk,
            config=config,
            dt=args.dt,
            output_path=args.output,
            experiment=args.experiment,
            category=args.category,
            parameter=args.parameter,
        )

    # Save the plot
    save_plot(args.output)

    print(f"Performance plot saved to: {args.output}")
