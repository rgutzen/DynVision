"""Base visualization components for DynVision.

This module provides base classes and utilities for standardized visualization
across the toolbox. It includes:
- Common plotting configurations
- Shared styling defaults
- Base visualization class
"""

from typing import Optional, Tuple, Dict, Any
import logging

import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)


class BaseVisualization:
    """Base class for DynVision visualizations.

    Provides common functionality for plot styling, saving, and configuration.
    Subclasses should implement specific visualization logic.
    """

    def __init__(
        self,
        style: str = "ticks",
        context: str = "talk",
        palette: str = "tab10",
        figsize: Tuple[int, int] = (10, 6),
        dpi: int = 300,
        **kwargs,
    ):
        """Initialize visualization with styling defaults.

        Args:
            style: Seaborn style name
            context: Seaborn context name
            palette: Color palette name
            figsize: Default figure size (width, height)
        """
        self.style = style
        self.context = context
        self.palette = palette
        self.figsize = figsize
        self.dpi = dpi
        self._setup_style()

    def load_dataframe(self, df: pd.DataFrame) -> None:
        """Load classifier response data.

        Args:
            df: DataFrame containing classifier responses

        Raises:
            ValueError: If required columns are missing or DataFrame is empty
        """
        # Validate DataFrame
        if df.empty:
            raise ValueError("DataFrame is empty")

        missing_cols = self._required_columns - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")

        # Process in batches if needed
        if self.batch_size and len(df) > self.batch_size:
            self.df = df.iloc[: self.batch_size].copy()
            logger.warning(
                f"Large DataFrame detected. Using first {self.batch_size} rows. "
                "Set batch_size=None to use full dataset."
            )
        else:
            self.df = df.copy()

    def calculate_accuracy(self, df: Optional[pd.DataFrame] = None) -> float:
        """Calculate classification accuracy.

        Returns:
            Classification accuracy as float
        """
        if df is None:
            if self.df is None:
                raise ValueError("No data loaded. Call load_dataframe first.")
            else:
                df = self.df

        valid_samples = df[df.label_index > -1]
        return (valid_samples.guess_index == valid_samples.label_index).mean()

    def _setup_style(self) -> None:
        """Configure plot styling."""
        sns.set_style(self.style)
        sns.set_context(self.context)
        plt.rcParams["figure.figsize"] = self.figsize

    def create_figure(
        self, figsize: Optional[Tuple[int, int]] = None, **kwargs
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Create new figure with consistent styling.

        Args:
            figsize: Optional custom figure size
            **kwargs: Additional arguments for plt.subplots

        Returns:
            Tuple of (Figure, Axes)
        """
        size = figsize if figsize else self.figsize
        fig, ax = plt.subplots(figsize=size, **kwargs)
        return fig, ax

    def save_figure(
        self,
        fig: plt.Figure,
        path: Path,
        dpi: Optional[int] = None,
        align_labels: bool = True,
        **kwargs,
    ) -> None:
        """Save figure with standard parameters.

        Args:
            fig: Figure to save
            path: Save path
            dpi: Resolution
            **kwargs: Additional arguments for savefig
        """
        if align_labels:
            fig.align_labels()
        dpi = self.dpi if dpi is None else dpi
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=int(dpi), bbox_inches="tight", **kwargs)
        plt.close(fig)

    def apply_common_styling(
        self,
        ax: plt.Axes,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        legend: bool = True,
        despine: bool = True,
    ) -> None:
        """Apply common styling to axes.

        Args:
            ax: Axes to style
            title: Optional title
            xlabel: Optional x-axis label
            ylabel: Optional y-axis label
            legend: Whether to show legend
            despine: Whether to remove top and right spines
        """
        if title:
            ax.set_title(title)

        if xlabel:
            ax.set_xlabel(xlabel)
        else:
            ax.set_xlabel("")

        if ylabel:
            ax.set_ylabel(ylabel)
        else:
            ax.set_ylabel("")

        if not legend:
            ax.get_legend().remove() if ax.get_legend() else None
        else:
            ax.legend(frameon=False)

        if despine:
            sns.despine(ax=ax)

    def get_colors(self, names: list, palette: Optional[str] = None) -> Dict[str, Any]:
        """Get consistent colors for visualization.

        Args:
            names: List of trace names
            palette: Optional color palette name

        Returns:
            Dictionary mapping trace names to colors
        """
        palette = self.palette if palette is None else palette
        colors = sns.color_palette(palette, n_colors=len(names))
        return dict(zip(names, colors))
