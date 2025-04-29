"""Enhanced weight distribution visualization for DynVision.

This module provides advanced visualization of neural network weight distributions
with additional features for analyzing layer connectivity and statistical properties.

Example:
    $ python enhanced_weight_distributions.py --model model.pt \
        --output_distributions dist.png --output_correlations corr.png
"""

import argparse
from typing import Optional, Dict, Any, Tuple, List, Union
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from dynvision.utils import parse_parameters
from dynvision.visualization.base_visualization import BaseVisualization

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_argument_parser():
    parser = argparse.ArgumentParser(
        description="Generate weight distribution visualizations"
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to model state dict (.pt file)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to save weight distribution plot",
    )
    return parser


class WeightDistributionViz(BaseVisualization):
    """Enhanced visualization of neural network weight distributions."""

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        state_dict: Optional[Dict[str, torch.Tensor]] = None,
        **kwargs,
    ):
        """Initialize weight distribution visualization.

        Args:
            model: Optional PyTorch model
            state_dict: Optional state dictionary
            **kwargs: Additional arguments for BaseVisualization
        """
        super().__init__(**kwargs)
        self.model = model
        self.weight_stats = {}
        if state_dict is not None:
            self.load_state_dict(state_dict)
        elif model is not None:
            self.load_state_dict(model.state_dict())

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """Load weights from state dictionary.

        Args:
            state_dict: PyTorch state dictionary
        """
        self.weight_stats = self._compute_weight_stats(state_dict)

    def _compute_weight_stats(
        self, state_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, Dict[str, Any]]:
        """Compute statistical properties of weights.

        Args:
            state_dict: PyTorch state dictionary

        Returns:
            Dictionary with weight statistics
        """
        stats = {}
        for key, params in state_dict.items():
            if "weight" in key:
                weights = params.cpu().flatten().numpy()
                stats[key] = {
                    "mean": np.mean(weights),
                    "std": np.std(weights),
                    "min": np.min(weights),
                    "max": np.max(weights),
                    "median": np.median(weights),
                    "quartiles": np.percentile(weights, [25, 75]),
                    "sparsity": np.mean(np.abs(weights) < 1e-6),
                    "layer_type": (
                        "recurrent" if "recurrence" in key else "feedforward"
                    ),
                }
        return stats

    def plot_weight_distributions(
        self,
        split_by_type: bool = True,
        save_path: Optional[Path] = None,
    ) -> Tuple[plt.Figure, List[plt.Axes]]:
        """Plot enhanced weight distributions.

        Args:
            figsize: Figure size
            split_by_type: Whether to split recurrent and feedforward weights
            show_stats: Whether to show statistical annotations
            save_path: Optional path to save figure

        Returns:
            Tuple of (Figure, List of Axes)
        """
        if not self.weight_stats:
            if self.model:
                self.load_state_dict(self.model.state_dict())
            else:
                raise ValueError(
                    "No weights loaded. Provide model or load state_dict."
                )

        # Create figure with two subplots for distributions and stats
        fig = plt.figure(figsize=self.figsize)
        gs = plt.GridSpec(2, 2, height_ratios=[3, 1])
        ax_dist = plt.subplot(gs[0, :])
        ax_stats = plt.subplot(gs[1, 0])
        ax_sparsity = plt.subplot(gs[1, 1])

        # Create DataFrame for plotting
        data = []
        for key, stats in self.weight_stats.items():
            layer_name = key.split(".")[0]
            data.append(
                {
                    "layer": layer_name,
                    "type": stats["layer_type"],
                    "mean": stats["mean"],
                    "std": stats["std"],
                    "sparsity": stats["sparsity"],
                    "min": stats["min"],
                    "max": stats["max"],
                }
            )
        df = pd.DataFrame(data)

        # Plot distributions
        if split_by_type:
            sns.violinplot(
                data=df,
                x="layer",
                y="mean",
                hue="type",
                split=True,
                ax=ax_dist,
                inner="quartile",
            )
        else:
            sns.violinplot(
                data=df,
                x="layer",
                y="mean",
                ax=ax_dist,
                inner="quartile",
            )
        ax_dist.axhline(0, color="0.7", linewidth=1.5)

        self.apply_common_styling(
            ax=ax_dist,
            ylabel="Weight",
        )

        # Plot statistics
        sns.barplot(
            data=df,
            x="layer",
            y="std",
            hue="type" if split_by_type else None,
            ax=ax_stats,
        )
        self.apply_common_styling(ax=ax_stats, ylabel="Weight std", legend=False)

        # Plot sparsity
        sns.barplot(
            data=df,
            x="layer",
            y="sparsity",
            hue="type" if split_by_type else None,
            ax=ax_sparsity,
        )
        self.apply_common_styling(
            ax=ax_sparsity, ylabel="Sparsity Ratio", legend=False
        )

        # Adjust layout
        plt.tight_layout()

        if save_path:
            self.save_figure(fig, save_path)

        return fig, [ax_dist, ax_stats, ax_sparsity]

    def print_weight_summary(self) -> None:
        """Print summary of weight statistics."""
        if not self.weight_stats:
            raise ValueError("No weights loaded")

        print("\nWeight Distribution Summary:")
        print("-" * 80)
        print(f"{'Layer':<30} {'Type':<12} {'Mean':>10} {'Std':>10} {'Sparsity':>10}")
        print("-" * 80)

        for key, stats in self.weight_stats.items():
            layer = key.split(".")[0]
            print(
                f"{layer:<30} {stats['layer_type']:<12} "
                f"{stats['mean']:>10.3f} {stats['std']:>10.3f} "
                f"{stats['sparsity']:>10.3f}"
            )
        print("-" * 80)


def main():
    parser = create_argument_parser()
    config = parse_parameters(parser)

    try:
        # Validate input file
        if not config.model.exists():
            raise FileNotFoundError(f"Model file not found: {config.model}")

        # Create output directories
        config.output.parent.mkdir(parents=True, exist_ok=True)

        # Load model state dict
        logger.info(f"Loading model from {config.model}")
        try:
            state_dict = torch.load(config.model)
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")

        # Create visualization instance with figure parameters
        viz = WeightDistributionViz(state_dict=state_dict, **vars(config))

        # Generate and save distribution plot
        logger.info("Generating weight distribution plot")
        try:
            fig, axes = viz.plot_weight_distributions(
                split_by_type=True,
                save_path=config.output,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to generate distribution plot: {str(e)}")
        plt.close(fig)

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise
    plt.close(fig)


if __name__ == "__main__":
    main()
