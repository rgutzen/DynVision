"""Adaptation analysis visualization for DynVision.

This module provides visualization tools for analyzing how models
adapt to different stimuli and conditions over time.
"""

import argparse
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any

import matplotlib.pyplot as plt
import seaborn as sns

from .base_visualization import BaseVisualization
from dynvision.utils import parse_parameters, load_responses


class AdaptationViz(BaseVisualization):
    """Visualization of model adaptation patterns."""

    def __init__(
        self, data_arg_key: str = "contrast", category: str = "rctype", **kwargs
    ):
        """Initialize adaptation visualization.

        Args:
            data_arg_key: Key for data arguments
            category: Category for grouping
            **kwargs: Additional arguments for BaseVisualization

        Raises:
            ValueError: If data_arg_key or category is empty
        """
        if not data_arg_key:
            raise ValueError("data_arg_key cannot be empty")
        if not category:
            raise ValueError("category cannot be empty")

        super().__init__(**kwargs)
        self.data_arg_key = data_arg_key
        self.category = category
        self.df = None
        self.layer_names = []
        self._required_measures = {"power", "peak_height", "peak_time", "peak_ratio"}

    def load_data(
        self,
        responses: List[Path],
        test_outputs: List[Path],
        measures: List[str] = ["power", "peak_height", "peak_time"],
    ) -> None:
        """Load and process adaptation data.

        Args:
            responses: List of response file paths
            test_outputs: List of test output file paths
            measures: List of measures to compute

        Raises:
            ValueError: If file paths are invalid or measures are unsupported
        """
        # Validate inputs
        if not responses or not test_outputs:
            raise ValueError("responses and test_outputs cannot be empty")
        if not measures:
            raise ValueError("measures cannot be empty")

        invalid_measures = set(measures) - self._required_measures
        if invalid_measures:
            raise ValueError(
                f"Unsupported measures: {invalid_measures}. "
                f"Supported measures: {self._required_measures}"
            )
        # Load and process data
        self.df, self.layer_names = load_responses(
            responses,
            test_outputs,
            data_arg_key=self.data_arg_key,
            measures=measures,
            category=self.category,
        )

    def plot_adaptation_grid(
        self,
        layer_name: str,
        measures: List[str] = ["power", "peak_height", "peak_time"],
        label_target: Optional[int] = None,
        save_path: Optional[Path] = None,
    ) -> Tuple[plt.Figure, List[plt.Axes]]:
        """Plot grid of adaptation patterns.

        Args:
            layer_name: Name of layer to analyze
            measures: List of measures to plot
            label_target: Optional specific label to focus on
            save_path: Optional path to save figure

        Returns:
            Tuple of (Figure, List of Axes)

        Raises:
            ValueError: If data not loaded or parameters invalid
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load_data first.")
        if layer_name not in self.layer_names:
            raise ValueError(
                f"Invalid layer_name: {layer_name}. "
                f"Available layers: {', '.join(self.layer_names)}"
            )

        invalid_measures = set(measures) - self._required_measures
        if invalid_measures:
            raise ValueError(
                f"Unsupported measures: {invalid_measures}. "
                f"Supported measures: {self._required_measures}"
            )
        # Filter data for target label if specified
        plot_df = (
            self.df[self.df.label_set.str.contains(str(label_target))]
            if label_target is not None
            else self.df
        )
        if plot_df.empty:
            raise ValueError("No data to plot")
        # Get unique parameter values
        key_values = plot_df[self.data_arg_key].unique()

        # Create figure
        fig = plt.figure(figsize=(16, 10))
        ax_left = plt.subplot2grid((1, 2), (0, 0))
        ax_right = plt.subplot2grid((1, 2), (0, 1))
        ax_left.set_axis_off()
        ax_right.set_axis_off()

        axes = []

        # Plot power traces
        for i, key_value in enumerate(key_values):
            ax = fig.add_subplot(len(key_values), 2, 2 * i + 1, sharey=ax_left)
            axes.append(ax)

            sns.lineplot(
                data=plot_df[plot_df[self.data_arg_key] == key_value],
                x="times_index",
                y=f"{layer_name}_power",
                ax=ax,
                marker=".",
                palette="tab10",
                hue=self.category,
            )

            # Customize subplot
            if not i:
                ax.set_title(self.data_arg_key)
            ax.text(
                0.95,
                0.95,
                key_value,
                horizontalalignment="right",
                verticalalignment="top",
                transform=ax.transAxes,
            )
            if i == len(key_values) // 2:
                ax.set_ylabel("power")
            else:
                ax.set_ylabel("")
            ax.set_xlabel("")

            if i < len(key_values) - 1:
                ax.set_xticklabels([])

            ax.legend().remove()
            ax.xaxis.set_tick_params(which="both", labelbottom=True)
            sns.despine(ax=ax, left=True, bottom=True)

        axes[-1].set_xlabel("times index")

        # Plot measure comparisons
        for i, measure in enumerate(measures):
            ax = fig.add_subplot(len(measures), 2, 2 * i + 2)
            axes.append(ax)

            sns.lineplot(
                data=plot_df,
                x=self.data_arg_key,
                y=f"{layer_name}_{measure}",
                ax=ax,
                marker=".",
                palette="tab10",
                hue=self.category,
            )

            # Customize subplot
            ax.set_xlabel("")
            ax.set_ylabel(f"{measure.replace('_', ' ')}")
            sns.despine(ax=ax, left=True)

            if i:
                ax.legend().remove()
            else:
                ax.legend(frameon=False, title=self.category)

        axes[-1].set_xlabel(self.data_arg_key.replace("_", " "))
        fig.suptitle(layer_name)

        if save_path:
            self.save_figure(fig, save_path)

        return fig, axes

    def plot_layer_comparison(
        self,
        measure: str = "power",
        save_path: Optional[Path] = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Plot comparison of adaptation across layers.

        Args:
            measure: Measure to compare
            save_path: Optional path to save figure

        Returns:
            Tuple of (Figure, Axes)

        Raises:
            ValueError: If data not loaded or measure invalid
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load_data first.")
        if measure not in self._required_measures:
            raise ValueError(
                f"Unsupported measure: {measure}. "
                f"Supported measures: {self._required_measures}"
            )
        # Create figure
        fig, ax = self.create_figure(fig_size=(12, 6))

        # Plot layer comparisons
        for layer in self.layer_names:
            sns.lineplot(
                data=self.df,
                x=self.data_arg_key,
                y=f"{layer}_{measure}",
                label=layer,
                marker="o",
                ax=ax,
            )

        # Style plot
        self.apply_common_styling(
            ax,
            title=f"{measure.capitalize()} Adaptation Across Layers",
            xlabel=self.data_arg_key,
            ylabel=measure.capitalize(),
        )

        if save_path:
            self.save_figure(fig, save_path)

        return fig, ax


def create_argument_parser():
    """Main execution function."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--responses",
        nargs="+",
        type=Path,
        required=True,
        help="Path to pt files",
    )
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
        help="Path to directory",
    )
    parser.add_argument(
        "--parameter",
        type=str,
        required=True,
        help="Parameter to plot",
    )
    parser.add_argument(
        "--measures",
        nargs="+",
        type=str,
        required=True,
        help="Measure to plot",
    )
    parser.add_argument(
        "--category",
        type=str,
        required=True,
        help="Category to plot",
    )
    return parser


def main():
    parser = create_argument_parser()
    config = parse_parameters(parser)

    if "interval" in config.parameter:
        config.measures = [
            measure.replace("peak_time", "peak_ratio") for measure in config.measures
        ]

    # Create visualizer
    viz = AdaptationViz(
        data_arg_key=config.parameter, category=config.category, **vars(config)
    )

    # Load data
    viz.load_data(
        config.responses,
        config.test_outputs,
        measures=config.measures,
    )

    # Create output directory
    config.output.parent.mkdir(parents=True, exist_ok=True)

    # Generate plots
    label_targets = viz.df.label_index.unique()
    label_targets = label_targets[label_targets != -1]

    for layer_name in viz.layer_names:
        for label_target in [None] + list(label_targets):
            # Plot adaptation grid
            fig, _ = viz.plot_adaptation_grid(
                layer_name=layer_name,
                measures=config.measures,
                label_target=label_target,
            )

            # Save plot
            label_str = "" if label_target is None else f"_label{label_target}"
            viz.save_figure(
                fig,
                config.output.parent / f"{layer_name}{label_str}.png",
            )

        # Plot layer comparison
        fig, _ = viz.plot_layer_comparison()
        viz.save_figure(
            fig,
            config.output.parent / f"layer_comparison_{layer_name}.png",
        )

    # Create flag file
    config.output.touch()


if __name__ == "__main__":
    main()
