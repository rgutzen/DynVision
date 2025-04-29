"""Experiment output visualization for DynVision.

This module provides visualization tools for analyzing and comparing
experiment outputs across different conditions and models.
"""

import argparse
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .base_visualization import BaseVisualization
from .plot_classifier_responses import ClassifierResponseViz
from dynvision.utils import extract_param_from_string, load_df


class ExperimentOutputViz(BaseVisualization):
    """Visualization of experiment outputs and comparisons."""

    def __init__(
        self, data_arg_key: str = "contrast", category: str = "rctype", **kwargs
    ):
        """Initialize experiment output visualization.

        Args:
            data_arg_key: Key for data arguments
            category: Category for grouping
            **kwargs: Additional arguments for BaseVisualization
        """
        super().__init__(**kwargs)
        self.data_arg_key = data_arg_key
        self.category = category
        self.df = None

    def load_test_outputs(self, csv_files: List[Path]) -> None:
        """Load and process test outputs.

        Args:
            csv_files: List of CSV file paths
        """
        dfs = []

        for csv_file in csv_files:
            arg_value = extract_param_from_string(
                csv_file.stem, key=self.data_arg_key, value_type=float
            )
            rctype = extract_param_from_string(
                csv_file.stem, key="rctype", value_type=str
            )

            df = load_df(csv_file)
            df[self.data_arg_key] = arg_value
            df["recurrence_type"] = rctype

            dfs.append(df)

        self.df = pd.concat(dfs, axis=0)

    def plot_classifier_response(
        self,
        df: pd.DataFrame,
        ax: plt.Axes,
        show_accuracy: bool = True,
    ) -> None:
        """Plot classifier response for a specific condition.

        Args:
            df: DataFrame for specific condition
            ax: Matplotlib axes
            show_accuracy: Whether to show accuracy text
        """
        viz = ClassifierResponseViz()
        viz.load_data(df)
        viz.plot_responses(show_legend=False, ax=ax)

        if show_accuracy:
            accuracy = viz.calculate_accuracy()
            ax.text(
                0.05,
                0.8,
                f"{accuracy*100:.0f}%",
                transform=ax.transAxes,
            )

    def plot_experiment_grid(
        self,
        label_target: str = "8",
        save_path: Optional[Path] = None,
    ) -> Tuple[plt.Figure, np.ndarray]:
        """Plot grid of experiment outputs.

        Args:
            label_target: Target label to focus on
            save_path: Optional path to save figure

        Returns:
            Tuple of (Figure, Array of Axes)
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load_test_outputs first.")

        # Filter data for target label
        df_label_set = self.df[self.df.label_set.str.contains(str(label_target))]
        recurrence_types = df_label_set.recurrence_type.unique()
        data_arg_values = df_label_set[self.data_arg_key].unique()

        # Create figure grid
        fig, axes = plt.subplots(
            nrows=len(recurrence_types),
            ncols=len(data_arg_values),
            sharex=True,
            sharey=True,
            figsize=(len(data_arg_values) * 3.5, len(recurrence_types) * 3.5 + 2),
        )

        # Plot each condition
        for i, rctype in enumerate(recurrence_types):
            for j, value in enumerate(np.sort(data_arg_values)):
                ax = axes[i, j]
                plot_df = df_label_set[
                    (df_label_set[self.data_arg_key] == value)
                    & (df_label_set.recurrence_type == rctype)
                ]

                self.plot_classifier_response(plot_df, ax)

                # Add labels and titles
                if i == len(recurrence_types) - 1:
                    ax.set_xlabel("time step")
                if j == 0:
                    ax.set_ylabel(rctype, weight="bold")
                if i == 0:
                    ax.set_title(f"{self.data_arg_key}={value}")

        # Add legend to top-right subplot
        axes[0, -1].legend(
            title="class index",
            bbox_to_anchor=(0.93, 1),
            loc="upper left",
            frameon=False,
        )

        if save_path:
            self.save_figure(fig, save_path)

        return fig, axes

    def plot_performance_comparison(
        self,
        metric: str = "accuracy",
        save_path: Optional[Path] = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Plot performance comparison across conditions.

        Args:
            metric: Performance metric to plot
            save_path: Optional path to save figure

        Returns:
            Tuple of (Figure, Axes)
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load_test_outputs first.")

        # Calculate performance metric
        if metric == "accuracy":
            performance = self.df.groupby(
                ["recurrence_type", self.data_arg_key]
            ).apply(lambda x: (x.guess_index == x.label_index).mean())
        else:
            raise ValueError(f"Unsupported metric: {metric}")

        # Create figure
        fig, ax = self.create_figure()

        # Plot performance
        performance_df = performance.reset_index()
        sns.lineplot(
            data=performance_df,
            x=self.data_arg_key,
            y=0,
            hue="recurrence_type",
            marker="o",
            ax=ax,
        )

        # Style plot
        self.apply_common_styling(
            ax,
            title=f"{metric.capitalize()} vs {self.data_arg_key}",
            xlabel=self.data_arg_key,
            ylabel=metric.capitalize(),
        )

        if save_path:
            self.save_figure(fig, save_path)

        return fig, ax


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test_outputs",
        nargs="+",
        type=Path,
        required=True,
        help="Path to csv files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to output directory",
    )
    parser.add_argument(
        "--parameter",
        type=str,
        required=True,
        help="Parameter to plot",
    )
    parser.add_argument(
        "--category",
        type=str,
        required=True,
        help="Category to plot",
    )

    args = parser.parse_args()

    # Create visualizer
    viz = ExperimentOutputViz(
        data_arg_key=args.parameter,
        category=args.category,
    )

    # Load data
    viz.load_test_outputs(args.test_outputs)

    # Create output directory
    args.output.parent.mkdir(parents=True, exist_ok=True)
    data_identifier = args.output.parent.stem

    # Generate plots
    label_targets = viz.df.label_index.unique()
    label_targets = label_targets[label_targets != -1]

    for label_target in label_targets:
        fig, axes = viz.plot_experiment_grid(label_target=str(label_target))
        fig.suptitle(f"label={label_target}\n{data_identifier}")
        viz.save_figure(
            fig, args.output.parent / f"experiment_outputs_label{label_target}.png"
        )

        # Plot performance comparison
        fig, ax = viz.plot_performance_comparison()
        viz.save_figure(
            fig, args.output.parent / f"performance_comparison_label{label_target}.png"
        )


if __name__ == "__main__":
    main()
