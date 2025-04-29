"""Plot classifier unit responses with focus on top active units.

This module provides visualization of classifier responses, focusing on
the most active units and their temporal dynamics.
"""

import argparse
import logging
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from dynvision.visualization.base_visualization import BaseVisualization
from dynvision.utils import load_df, parse_parameters

logger = logging.getLogger(__name__)


class ClassifierResponseViz(BaseVisualization):
    """Visualization of classifier unit responses."""

    def __init__(self, n_units: int = 10, batch_size: Optional[int] = None, **kwargs):
        """Initialize classifier response visualization.

        Args:
            n_units: Number of top units to analyze
            batch_size: Optional batch size for large datasets
            **kwargs: Additional arguments for BaseVisualization

        Raises:
            ValueError: If n_units < 1
        """
        if n_units < 1:
            raise ValueError(f"n_units must be >= 1, got {n_units}")

        super().__init__(**kwargs)
        self.n_units = n_units
        self.batch_size = batch_size
        self.df = None
        self._required_columns = {
            "label_set",
            "sample_index",
            "class_index",
            "label_index",
            "response",
            "times_index",
            "guess_index",
        }

    def validate_label_set(self, label_set: str) -> None:
        """Validate label set exists in data.

        Args:
            label_set: Label set to validate

        Raises:
            ValueError: If label set is invalid
        """
        if label_set not in self.df.label_set.unique():
            raise ValueError(
                f"Invalid label_set: {label_set}. "
                f"Available sets: {', '.join(map(str, self.df.label_set.unique()))}"
            )

    def get_label_sequence_info(self, label_set: str) -> Tuple[np.ndarray, List[int]]:
        """Extract label sequence information.

        Args:
            label_set: String identifier for label sequence

        Returns:
            Tuple of (changepoints, label sequence)

        Raises:
            ValueError: If data not loaded or label set invalid
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load_data first.")

        self.validate_label_set(label_set)

        try:
            # Filter DataFrame for the specific label set
            label_set_df = self.df[self.df.label_set == label_set]
            sample_index = label_set_df.sample_index.unique()[0]
            class_index = label_set_df.class_index.unique()[0]
            label_set_df = label_set_df[
                (label_set_df.sample_index == sample_index)
                & (label_set_df.class_index == class_index)
            ]
            label_indices = label_set_df.label_index.values

            if len(label_indices) == 0:
                raise ValueError(f"No data found for label_set: {label_set}")

            # Find points where label changes
            changepoints = np.where(np.diff(label_indices) != 0)[0]

            # Get sequence of labels
            sequence = [label_indices[0]]
            for point in changepoints:
                sequence.append(label_indices[point + 1])

            return changepoints, sequence
        except Exception as e:
            raise ValueError(
                f"Failed to extract sequence info for label_set {label_set}: {str(e)}"
            )

    def get_top_units(
        self,
        df: Optional[pd.DataFrame] = None,
        n_units: Optional[int] = None,
        use_ranks: bool = True,
    ) -> List[int]:
        """Identify top N classifier units based on activity or rank scores.

        Args:
            use_ranks: If True, select units based on average rank over time
                      If False, select based on overall mean activity (legacy behavior)

        Returns:
            List of class indices for top units
        """
        if df is None:
            if self.df is None:
                raise ValueError("No data loaded. Call load_data first.")
            else:
                df = self.df
        if n_units is None:
            n_units = self.n_units

        df = df[df.label_index > -1]
        df = (
            df.groupby(["label_set", "times_index", "class_index"])
            .mean()
            .reset_index(drop=False)
        )

        if use_ranks:
            # Group by time step and rank units within each time step
            ranked_df = df.groupby("times_index").apply(
                lambda x: x.assign(rank=x["response"].rank(method="average"))
            )

            # Calculate mean rank over time for each unit
            mean_ranks = ranked_df.groupby("class_index")["rank"].mean()
            return mean_ranks.nlargest(n_units).index.tolist()
        else:
            # Legacy behavior: use mean activity
            mean_responses = df.groupby("class_index")["response"].mean()
            return mean_responses.nlargest(n_units).index.tolist()

    def plot_responses(
        self,
        label_set: Optional[str] = None,
        show_accuracy: bool = True,
        save_path: Optional[Path] = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Plot response traces for top units.

        Args:
            label_set: Optional specific label set to plot
            show_accuracy: Whether to show accuracy text
            save_path: Optional path to save figure

        Returns:
            Tuple of (Figure, Axes)

        Raises:
            ValueError: If data not loaded or label set invalid
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load_data first.")

        # Get label set if not provided
        if label_set is None:
            label_set = self.df.label_set.unique()[0]
        else:
            self.validate_label_set(label_set)

        # Get top units and filter data
        plot_df = self.df[(self.df.label_set == label_set)].copy()
        top_units = self.get_top_units(plot_df)
        plot_df = plot_df[(plot_df.class_index.isin(top_units))].copy()

        if plot_df.empty:
            raise ValueError(f"No data to plot for label_set: {label_set}")

        # Create figure
        fig, ax = self.create_figure()

        # Create color palette
        colors = self.get_colors(top_units)
        # Plot responses
        sns.lineplot(
            data=plot_df,
            x="times_index",
            y="response",
            hue="class_index",
            hue_order=top_units,
            errorbar=("ci", 75),
            palette=colors,
            ax=ax,
            marker=".",
        )
        # Add changepoint lines
        changepoints, sequence = self.get_label_sequence_info(label_set)
        for changepoint in changepoints:
            ax.axvline(
                changepoint,
                color="0.6",
                linestyle="--",
                linewidth=1,
                alpha=0.5,
                label="_nolegend_",
            )

        # Add accuracy text
        if show_accuracy:
            accuracy = self.calculate_accuracy(plot_df)
            ax.text(
                0.05,
                0.95,
                f"Accuracy: {accuracy:.2f}",
                transform=ax.transAxes,
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
            )

        # Customize plot
        label_seq = "→".join(map(str, sequence))
        self.apply_common_styling(
            ax,
            title=f"Top {self.n_units} Most Active Units\nLabel Sequence: {label_seq}",
            xlabel="Time Step",
            ylabel="Classifier Response",
        )

        ax.legend(
            title="Class Index",
            bbox_to_anchor=(1.01, 1),
            loc="upper left",
            frameon=False,
        )

        if save_path:
            self.save_figure(fig, save_path)

        return fig, ax


def create_argument_parser():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Plot classifier unit responses.")
    parser.add_argument(
        "--input", type=Path, required=True, help="Path to input CSV file"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Path to output directory"
    )
    parser.add_argument(
        "--n_units", type=int, default=10, help="Number of top units to plot"
    )
    parser.add_argument(
        "--max_sets", type=int, default=5, help="Number of label sets to plot"
    )
    return parser


def main():
    parser = create_argument_parser()
    config = parse_parameters(parser)

    if not config.input.exists():
        raise FileNotFoundError(f"Input file not found: {config.input}")

    config.output.mkdir(parents=True, exist_ok=True)

    df = load_df(config.input)
    data_identifier = config.input.stem

    viz = ClassifierResponseViz(**vars(config))
    viz.load_dataframe(df)

    for label_set in df.label_set.unique()[: config.max_sets]:
        changepoints, sequence = viz.get_label_sequence_info(label_set)
        output_path = config.output / f"{data_identifier}_label{sequence}.png"
        viz.plot_responses(label_set=label_set, save_path=output_path)


if __name__ == "__main__":
    main()
