"""Notebook generation utilities for DynVision.

This module provides utilities for creating and managing interactive
visualization notebooks for model analysis.
"""

import shutil
from pathlib import Path
from typing import Optional, Dict, Any, Union

import nbformat as nbf
from torch.nn import Module

from .base_visualization import BaseVisualization


class NotebookGenerator(BaseVisualization):
    """Generator for interactive visualization notebooks."""

    def __init__(
        self, output_dir: Path, template_path: Optional[Path] = None, **kwargs
    ):
        """Initialize notebook generator.

        Args:
            output_dir: Directory to save notebooks
            template_path: Optional path to template notebook
            **kwargs: Additional arguments for BaseVisualization
        """
        super().__init__(**kwargs)
        self.output_dir = Path(output_dir)
        self.template_path = template_path
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_notebook(
        self,
        model: Module,
        notebook_name: str,
        model_state_path: Optional[Path] = None,
        include_sections: Optional[Dict[str, bool]] = None,
    ) -> Path:
        """Generate interactive analysis notebook.

        Args:
            model: PyTorch model to analyze
            notebook_name: Name for the notebook
            model_state_path: Optional path to model state
            include_sections: Optional dict of sections to include

        Returns:
            Path to generated notebook
        """
        # Set default sections if not provided
        if include_sections is None:
            include_sections = {
                "architecture": True,
                "weights": True,
                "responses": True,
                "dynamics": True,
            }

        # Generate notebook path
        notebook_path = self.output_dir / f"{notebook_name}.ipynb"

        # Use template if available
        if self.template_path and self.template_path.exists():
            shutil.copy2(self.template_path, notebook_path)
            return notebook_path

        # Create new notebook
        nb = nbf.v4.new_notebook()
        cells = self._get_notebook_cells(model, model_state_path, include_sections)
        nb.cells.extend(cells)

        # Save notebook
        nbf.write(nb, str(notebook_path))
        return notebook_path

    def _get_notebook_cells(
        self,
        model: Module,
        model_state_path: Optional[Path],
        include_sections: Dict[str, bool],
    ) -> list:
        """Get cells for interactive notebook.

        Args:
            model: PyTorch model to analyze
            model_state_path: Optional path to model state
            include_sections: Dict of sections to include

        Returns:
            List of notebook cells
        """
        cells = [
            nbf.v4.new_markdown_cell(
                "# Interactive Model Analysis\n\n"
                "This notebook provides interactive visualizations for analyzing "
                "model behavior and performance."
            ),
            nbf.v4.new_code_cell(
                "import torch\n"
                "import matplotlib.pyplot as plt\n"
                "from ipywidgets import interact\n"
                "%matplotlib widget\n\n"
                "from dynvision.visualization import (\n"
                "    ModelArchitectureViz,\n"
                "    WeightDistributionViz,\n"
                "    ClassifierResponseViz\n"
                ")"
            ),
        ]

        # Add model loading code
        if model_state_path:
            cells.append(
                nbf.v4.new_code_cell(
                    "# Load model state\n"
                    f"model_state = '{model_state_path}'\n"
                    "model = torch.load(model_state)"
                )
            )
        else:
            cells.append(
                nbf.v4.new_code_cell(
                    "# Model is provided directly\n"
                    "model = None  # Replace with your model"
                )
            )

        # Add visualization sections
        if include_sections.get("architecture", True):
            cells.extend(self._get_architecture_cells())

        if include_sections.get("weights", True):
            cells.extend(self._get_weight_analysis_cells())

        if include_sections.get("responses", True):
            cells.extend(self._get_response_analysis_cells())

        if include_sections.get("dynamics", True):
            cells.extend(self._get_dynamics_analysis_cells())

        return cells

    def _get_architecture_cells(self) -> list:
        """Get cells for architecture visualization."""
        return [
            nbf.v4.new_markdown_cell(
                "## Model Architecture\n\n"
                "Visualize the model's architecture and connectivity."
            ),
            nbf.v4.new_code_cell(
                "# Create architecture visualization\n"
                "arch_viz = ModelArchitectureViz(model)\n\n"
                "@interact\n"
                "def plot_architecture(show_types=True):\n"
                "    fig, ax = arch_viz.plot_architecture(\n"
                "        show_types=show_types\n"
                "    )\n"
                "    plt.show()"
            ),
        ]

    def _get_weight_analysis_cells(self) -> list:
        """Get cells for weight analysis."""
        return [
            nbf.v4.new_markdown_cell(
                "## Weight Analysis\n\n"
                "Analyze weight distributions and correlations."
            ),
            nbf.v4.new_code_cell(
                "# Create weight visualization\n"
                "weight_viz = WeightDistributionViz(model)\n\n"
                "@interact\n"
                "def plot_weights(split_types=True, show_stats=True):\n"
                "    fig, axes = weight_viz.plot_weight_distributions(\n"
                "        split_by_type=split_types,\n"
                "        show_stats=show_stats\n"
                "    )\n"
                "    plt.show()\n\n"
                "# Plot weight correlations\n"
                "fig, ax = weight_viz.plot_weight_correlation_matrix(model)\n"
                "plt.show()"
            ),
        ]

    def _get_response_analysis_cells(self) -> list:
        """Get cells for response analysis."""
        return [
            nbf.v4.new_markdown_cell(
                "## Response Analysis\n\n" "Analyze classifier responses and patterns."
            ),
            nbf.v4.new_code_cell(
                "# Load responses\n"
                "df = model.get_classifier_dataframe()\n"
                "viz = ClassifierResponseViz()\n"
                "viz.load_data(df)\n\n"
                "@interact\n"
                "def plot_responses(show_accuracy=True, show_legend=True):\n"
                "    fig, ax = viz.plot_responses(\n"
                "        show_accuracy=show_accuracy,\n"
                "        show_legend=show_legend\n"
                "    )\n"
                "    plt.show()\n\n"
                "# Plot response summary\n"
                "fig, axes = viz.plot_response_summary()\n"
                "plt.show()"
            ),
        ]

    def _get_dynamics_analysis_cells(self) -> list:
        """Get cells for dynamics analysis."""
        return [
            nbf.v4.new_markdown_cell(
                "## Temporal Dynamics\n\n" "Analyze model behavior over time."
            ),
            nbf.v4.new_code_cell(
                "# Visualize temporal flow\n"
                "input_size = model.input_dims\n"
                "@interact\n"
                "def plot_temporal(timesteps=(5, 20)):\n"
                "    fig, ax = plot_temporal_flow(\n"
                "        model,\n"
                "        input_size=input_size,\n"
                "        timesteps=timesteps\n"
                "    )\n"
                "    plt.show()"
            ),
        ]


def generate_training_notebook(
    model: Module,
    output_dir: Path,
    epoch: int,
    template_path: Optional[Path] = None,
) -> Path:
    """Generate notebook for training-time analysis.

    Args:
        model: Model being trained
        output_dir: Directory to save notebook
        epoch: Current epoch number
        template_path: Optional template path

    Returns:
        Path to generated notebook
    """
    generator = NotebookGenerator(output_dir, template_path)
    return generator.generate_notebook(
        model,
        f"training_analysis_epoch{epoch:03d}",
        model_state_path=model.trainer.checkpoint_callback.best_model_path,
    )


def generate_analysis_notebook(
    model: Union[Module, Path],
    output_dir: Path,
    name: str = "model_analysis",
    template_path: Optional[Path] = None,
) -> Path:
    """Generate notebook for post-training analysis.

    Args:
        model: Model or path to model state
        output_dir: Directory to save notebook
        name: Name for the notebook
        template_path: Optional template path

    Returns:
        Path to generated notebook
    """
    generator = NotebookGenerator(output_dir, template_path)

    if isinstance(model, Path):
        model_state_path = model
        model = None
    else:
        model_state_path = None

    return generator.generate_notebook(model, name, model_state_path=model_state_path)
