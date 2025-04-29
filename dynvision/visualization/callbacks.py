"""Training visualization callbacks for DynVision.

This module provides PyTorch Lightning callbacks for visualizing model
behavior during training, including architecture, weights, and dynamics.
"""

from typing import Optional, Dict, Any

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import wandb

from .base_visualization import BaseVisualization
from .plot_model_architecture import ModelArchitectureViz, plot_temporal_flow
from .plot_weight_distributions import WeightDistributionViz
from .plot_classifier_responses import ClassifierResponseViz


class ModelArchitectureCallback(pl.Callback, BaseVisualization):
    """Monitor and visualize model architecture."""

    def __init__(self, log_every_n_epochs: int = 1, **kwargs):
        """Initialize architecture visualization callback.

        Args:
            log_every_n_epochs: Epochs between visualizations
            **kwargs: Additional arguments for BaseVisualization
        """
        super().__init__(**kwargs)
        self.log_every_n_epochs = log_every_n_epochs

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Log initial architecture visualization.

        Args:
            trainer: PyTorch Lightning trainer
            pl_module: The model being trained
        """
        # Visualize model architecture
        arch_viz = ModelArchitectureViz(pl_module)
        fig, ax = arch_viz.plot_architecture()
        trainer.logger.experiment.log({"model/architecture": wandb.Image(fig)})
        plt.close(fig)

        # Log architecture summary
        arch_viz.print_architecture_summary()

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Log temporal flow visualization.

        Args:
            trainer: PyTorch Lightning trainer
            pl_module: The model being trained
        """
        if trainer.current_epoch % self.log_every_n_epochs != 0:
            return

        # Visualize temporal dynamics
        input_size = (1, pl_module.n_channels, pl_module.dim_y, pl_module.dim_x)
        fig, ax = plot_temporal_flow(
            pl_module, input_size=input_size, timesteps=pl_module.n_timesteps
        )
        trainer.logger.experiment.log({"model/temporal_flow": wandb.Image(fig)})
        plt.close(fig)


class WeightDistributionCallback(pl.Callback, BaseVisualization):
    """Monitor and visualize weight distributions."""

    def __init__(self, log_every_n_epochs: int = 1, **kwargs):
        """Initialize weight distribution callback.

        Args:
            log_every_n_epochs: Epochs between visualizations
            **kwargs: Additional arguments for BaseVisualization
        """
        super().__init__(**kwargs)
        self.log_every_n_epochs = log_every_n_epochs

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Log weight distribution visualizations.

        Args:
            trainer: PyTorch Lightning trainer
            pl_module: The model being trained
        """
        if trainer.current_epoch % self.log_every_n_epochs != 0:
            return

        # Create visualizer
        weight_viz = WeightDistributionViz(pl_module)

        # Plot distributions
        fig, axes = weight_viz.plot_weight_distributions(
            split_by_type=True, show_stats=True
        )
        trainer.logger.experiment.log({"weights/distributions": wandb.Image(fig)})
        plt.close(fig)

        # Plot correlations
        fig, ax = weight_viz.plot_weight_correlation_matrix(pl_module)
        trainer.logger.experiment.log({"weights/correlations": wandb.Image(fig)})
        plt.close(fig)

        # Log weight statistics
        weight_viz.print_weight_summary()


class ClassifierResponseCallback(pl.Callback, BaseVisualization):
    """Monitor and visualize classifier responses."""

    def __init__(self, log_every_n_epochs: int = 1, **kwargs):
        """Initialize response visualization callback.

        Args:
            log_every_n_epochs: Epochs between visualizations
            **kwargs: Additional arguments for BaseVisualization
        """
        super().__init__(**kwargs)
        self.log_every_n_epochs = log_every_n_epochs

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Log classifier response visualizations.

        Args:
            trainer: PyTorch Lightning trainer
            pl_module: The model being trained
        """
        if trainer.current_epoch % self.log_every_n_epochs != 0:
            return

        # Get classifier responses
        df = pl_module.get_classifier_dataframe()
        if not len(df):
            return

        # Create visualizer
        viz = ClassifierResponseViz()
        viz.load_data(df)

        # Plot individual responses
        fig, ax = viz.plot_responses()
        trainer.logger.experiment.log({"responses/classifier": wandb.Image(fig)})
        plt.close(fig)

        # Plot response summary
        fig, axes = viz.plot_response_summary()
        trainer.logger.experiment.log({"responses/summary": wandb.Image(fig)})
        plt.close(fig)


def get_default_callbacks(
    log_every_n_epochs: int = 1,
    monitor_architecture: bool = True,
    monitor_weights: bool = True,
    monitor_responses: bool = True,
    **kwargs
) -> list:
    """Get default visualization callbacks.

    Args:
        log_every_n_epochs: Epochs between visualizations
        monitor_architecture: Whether to monitor architecture
        monitor_weights: Whether to monitor weights
        monitor_responses: Whether to monitor responses
        **kwargs: Additional arguments for callbacks

    Returns:
        List of visualization callbacks
    """
    callbacks = []

    if monitor_architecture:
        callbacks.append(
            ModelArchitectureCallback(log_every_n_epochs=log_every_n_epochs, **kwargs)
        )

    if monitor_weights:
        callbacks.append(
            WeightDistributionCallback(log_every_n_epochs=log_every_n_epochs, **kwargs)
        )

    if monitor_responses:
        callbacks.append(
            ClassifierResponseCallback(log_every_n_epochs=log_every_n_epochs, **kwargs)
        )

    return callbacks
