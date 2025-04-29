"""Model architecture visualization for DynVision.

This module provides tools for visualizing neural network architectures,
with special focus on recurrent connections and layer hierarchies.
"""

import argparse
from typing import Optional, Dict, Any, List, Tuple
import torch
import torch.nn as nn
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
from dynvision.visualization.base_visualization import BaseVisualization
from dynvision.utils import parse_parameters, load_model_and_weights


class ModelArchitectureViz(BaseVisualization):
    """Visualize neural network architectures with focus on recurrent connections."""

    def __init__(self, model: nn.Module, **kwargs):
        """Initialize architecture visualization.

        Args:
            model: PyTorch model to visualize
            **kwargs: Additional arguments for BaseVisualization
        """
        super().__init__(**kwargs)
        self.model = model
        self.graph = nx.DiGraph()
        self._build_graph()

    def _build_graph(self) -> None:
        """Build networkx graph from model architecture.

        Handles two cases:
        1. Models with explicit layer_names and layer_operations attributes
        2. General PyTorch modules using hierarchical structure
        """
        if hasattr(self.model, "layer_names") and hasattr(
            self.model, "layer_operations"
        ):
            self._build_graph_from_layer_structure()
        else:
            self._build_graph_from_modules()

    def _build_graph_from_layer_structure(self) -> None:
        """Build graph using model's layer_names and layer_operations."""
        # Add layer nodes
        for layer_name in self.model.layer_names:
            self.graph.add_node(
                layer_name,
                type="Layer",
                recurrent=hasattr(getattr(self.model, layer_name), "recurrence"),
                recurrence_type=getattr(
                    getattr(self.model, layer_name), "recurrence_type", None
                ),
            )

            # Add operation nodes for this layer
            prev_op = layer_name
            for op in self.model.layer_operations:
                # Create operation node name with correct pattern
                node_name = f"{op}_{layer_name}"

                # Check for operation existence
                has_op = (
                    hasattr(self.model, node_name)  # layer-specific operation
                    or (
                        hasattr(self.model, op) and op not in ["record", "delay"]
                    )  # global operation
                    or op == "layer"  # special case for direct layer call
                )

                if has_op:
                    self.graph.add_node(
                        node_name,
                        type=op.capitalize(),
                        recurrent=(
                            op == "layer"
                            and hasattr(getattr(self.model, layer_name), "recurrent")
                        ),
                    )
                    # Connect operations in sequence
                    self.graph.add_edge(prev_op, node_name)
                    prev_op = node_name

            # Add recurrent connections with proper style
            module = getattr(self.model, layer_name)
            if hasattr(module, "recurrence") and module.recurrence is not None:
                self.graph.add_edge(
                    layer_name,
                    layer_name,
                    recurrence_type=module.recurrence_type,
                    style="solid" if module.recurrence_type == "full" else "dashed",
                )

        # Add sequential connections between layers
        for i in range(len(self.model.layer_names) - 1):
            curr_layer = self.model.layer_names[i]
            next_layer = self.model.layer_names[i + 1]
            self.graph.add_edge(curr_layer, next_layer)

        # Add skip connections if present
        for layer_name in self.model.layer_names:
            skip_name = f"addskip_{layer_name}"
            if hasattr(self.model, skip_name):
                skip_module = getattr(self.model, skip_name)
                if hasattr(skip_module, "source"):
                    # Find the layer name for the source module
                    source_name = None
                    for name in self.model.layer_names:
                        if getattr(self.model, name) is skip_module.source:
                            source_name = name
                            break
                    if source_name is None:
                        continue
                    self.graph.add_edge(
                        source_name,
                        layer_name,
                        color="blue",
                        style="dashed",
                    )

        # Add feedback connections if present
        for layer_name in self.model.layer_names:
            feedback_name = f"addfeedback_{layer_name}"
            if hasattr(self.model, feedback_name):
                feedback_module = getattr(self.model, feedback_name)
                if hasattr(feedback_module, "source"):
                    # Find the layer name for the source module
                    source_name = None
                    for name in self.model.layer_names:
                        if getattr(self.model, name) is feedback_module.source:
                            source_name = name
                            break
                    if source_name is None:
                        continue
                    self.graph.add_edge(
                        source_name,
                        layer_name,
                        color="green",
                        style="dotted",
                    )

    def _build_graph_from_modules(self) -> None:
        """Build graph from PyTorch module hierarchy."""
        # Add all nodes
        for name, module in self.model.named_modules():
            if name:  # Skip root module
                has_recurrence = (
                    hasattr(module, "recurrence") and module.recurrence is not None
                )
                recurrence_type = (
                    getattr(module, "recurrence_type", None)
                    if has_recurrence
                    else None
                )
                self.graph.add_node(
                    name,
                    type=type(module).__name__,
                    recurrent=has_recurrence,
                    recurrence_type=recurrence_type,
                )

        # Add hierarchical edges
        added_edges = set()
        for name in self.graph.nodes():
            # Add parent-child connections
            parts = name.split(".")
            for i in range(1, len(parts)):
                parent = ".".join(parts[:i])
                child = ".".join(parts[: i + 1])
                if parent in self.graph and child in self.graph:
                    self.graph.add_edge(parent, child)
                    added_edges.add((parent, child))

            # Add sequential connections within same parent
            parent = ".".join(parts[:-1])
            if parent:
                siblings = [
                    n
                    for n in self.graph.nodes()
                    if n.startswith(parent + ".") and n != name
                ]
                siblings.sort()
                if siblings:
                    idx = siblings.index(name) if name in siblings else -1
                    if idx >= 0 and idx < len(siblings) - 1:
                        self.graph.add_edge(name, siblings[idx + 1])
                        added_edges.add((name, siblings[idx + 1]))

        # Add recurrent connections
        for name in self.graph.nodes():
            if self.graph.nodes[name].get("recurrent"):
                self.graph.add_edge(name, name)
                added_edges.add((name, name))

        # Ensure root connection if graph is disconnected
        if not nx.is_connected(self.graph.to_undirected()):
            components = list(nx.connected_components(self.graph.to_undirected()))
            if len(components) > 1:
                main_component = max(components, key=len)
                main_node = min(main_component)
                for component in components:
                    if component != main_component:
                        node = min(component)
                        self.graph.add_edge(main_node, node)

    def plot_architecture(
        self,
        figsize: Optional[tuple] = (12, 8),
        layer_colors: Optional[Dict[str, Any]] = None,
        show_types: bool = True,
        save_path: Optional[Path] = None,
        min_node_size: int = 1000,
        max_node_size: int = 3000,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Plot model architecture.

        Args:
            figsize: Figure size
            layer_colors: Optional color mapping for layers
            show_types: Whether to show layer types
            save_path: Optional path to save figure
            min_node_size: Minimum node size for visualization
            max_node_size: Maximum node size for visualization

        Returns:
            Tuple of (Figure, Axes)

        Raises:
            ValueError: If graph is empty or disconnected
            RuntimeError: If layout computation fails
        """
        if not self.graph.nodes():
            raise ValueError("Graph is empty. No modules found in model.")

        if not nx.is_connected(self.graph.to_undirected()):
            raise ValueError("Graph contains disconnected components.")

        fig, ax = self.create_figure(figsize=figsize)

        # Try different layouts for best visualization
        layouts = [
            (nx.kamada_kawai_layout, {"weight": None}),
            (nx.spring_layout, {"k": 1.5, "iterations": 50}),
            (nx.shell_layout, {}),
        ]

        pos = None
        errors = []
        for layout_fn, kwargs in layouts:
            try:
                pos = layout_fn(self.graph, **kwargs)
                break
            except Exception as e:
                errors.append(f"{layout_fn.__name__}: {str(e)}")

        if pos is None:
            raise RuntimeError(
                "Failed to compute layout. Tried:\n" + "\n".join(errors)
            )

        # Calculate node sizes
        node_sizes = []
        node_colors = []
        info = self.get_layer_info()

        # Find max parameters among actual layers
        layer_params = [
            info[name]["parameters"]
            for name in info
            if name in getattr(self.model, "layer_names", [])
        ]
        max_params = max(layer_params) if layer_params else 1

        for node in self.graph.nodes():
            # Check if this is a layer node
            is_layer = node in getattr(self.model, "layer_names", [])
            # Check if this is a layer operation (with correct naming pattern)
            is_operation = "_" in node and any(
                node.startswith(op + "_")
                for op in getattr(self.model, "layer_operations", [])
            )

            if is_layer:
                # Layer nodes sized by parameter count
                param_count = info.get(node, {"parameters": 0})["parameters"]
                size = min_node_size + (max_node_size - min_node_size) * (
                    param_count / max_params
                )
                if self.graph.nodes[node]["recurrent"]:
                    color = "lightcoral"
                else:
                    color = "lightblue"
            elif is_operation:
                # Operation nodes are smaller
                size = min_node_size * 0.5
                if "layer" in node:  # Special case for layer operations
                    color = "lightgray"
                else:
                    color = "lightgreen"
            else:
                # Other nodes (e.g., from module hierarchy)
                size = min_node_size * 0.75
                if self.graph.nodes[node]["recurrent"]:
                    color = "lightcoral"
                else:
                    color = "lightblue"

            node_sizes.append(size)
            node_colors.append(color)

        # Draw nodes with scaled sizes
        nx.draw_networkx_nodes(
            self.graph,
            pos,
            node_color=node_colors,
            node_size=node_sizes,
            alpha=0.7,
            ax=ax,
        )

        # Draw edges by type
        edges_by_type = {"regular": [], "recurrent": [], "skip": [], "feedback": []}

        for u, v, data in self.graph.edges(data=True):
            if u == v:
                edges_by_type["recurrent"].append((u, v))
            elif data.get("style") == "dashed":
                edges_by_type["skip"].append((u, v))
            elif data.get("style") == "dotted":
                edges_by_type["feedback"].append((u, v))
            else:
                edges_by_type["regular"].append((u, v))

        # Draw regular edges
        nx.draw_networkx_edges(
            self.graph,
            pos,
            edgelist=edges_by_type["regular"],
            edge_color="gray",
            arrows=True,
            ax=ax,
        )

        # Draw recurrent edges by type
        recurrent_full = [
            (u, v)
            for (u, v, d) in self.graph.edges(data=True)
            if u == v and d.get("recurrence_type") == "full"
        ]
        recurrent_other = [
            (u, v)
            for (u, v, d) in self.graph.edges(data=True)
            if u == v and d.get("recurrence_type") != "full"
        ]

        # Draw full recurrent connections
        if recurrent_full:
            nx.draw_networkx_edges(
                self.graph,
                pos,
                edgelist=recurrent_full,
                edge_color="red",
                arrows=True,
                connectionstyle="arc3,rad=0.3",
                ax=ax,
            )

        # Draw other recurrent connections
        if recurrent_other:
            nx.draw_networkx_edges(
                self.graph,
                pos,
                edgelist=recurrent_other,
                edge_color="orange",
                style="dashed",
                arrows=True,
                connectionstyle="arc3,rad=0.2",
                ax=ax,
            )

        # Draw skip connections
        nx.draw_networkx_edges(
            self.graph,
            pos,
            edgelist=edges_by_type["skip"],
            edge_color="blue",
            style="dashed",
            arrows=True,
            ax=ax,
        )

        # Draw feedback connections
        nx.draw_networkx_edges(
            self.graph,
            pos,
            edgelist=edges_by_type["feedback"],
            edge_color="green",
            style="dotted",
            arrows=True,
            ax=ax,
        )

        # Add labels
        labels = {}
        for node in self.graph.nodes():
            label = node.split(".")[-1]
            if show_types:
                label += f"\n({self.graph.nodes[node]['type']})"
            labels[node] = label

        nx.draw_networkx_labels(self.graph, pos, labels, ax=ax)

        # Style plot
        ax.set_title("Model Architecture", pad=20)
        plt.axis("off")

        if save_path:
            self.save_figure(fig, save_path)

        return fig, ax

    def get_layer_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about model layers.

        Returns:
            Dictionary with layer information
        """
        info = {}
        for name, module in self.model.named_modules():
            if name:
                params = sum(p.numel() for p in module.parameters())
                has_recurrence = (
                    hasattr(module, "recurrence") and module.recurrence is not None
                )
                recurrence_type = (
                    getattr(module, "recurrence_type", None)
                    if has_recurrence
                    else None
                )
                info[name] = {
                    "type": type(module).__name__,
                    "parameters": params,
                    "recurrent": has_recurrence,
                    "recurrence_type": recurrence_type,
                }
        return info

    def print_architecture_summary(self) -> None:
        """Print summary of model architecture."""
        info = self.get_layer_info()
        print("\nModel Architecture Summary:")
        print("-" * 80)
        print(f"{'Layer Name':<30} {'Type':<20} {'Parameters':<12} {'Recurrent'}")
        print("-" * 80)

        for name, layer_info in info.items():
            print(
                f"{name:<30} {layer_info['type']:<20} "
                f"{layer_info['parameters']:<12} {layer_info['recurrent']}"
            )
        print("-" * 80)
        total_params = sum(info[name]["parameters"] for name in info)
        print(f"Total Parameters: {total_params:,}")


def create_argument_parser():
    """Main function for command line execution."""
    parser = argparse.ArgumentParser(description="Visualize model architecture")
    parser.add_argument("--model_name", type=str, required=True, help="Name of model")
    parser.add_argument(
        "--state_dict", type=str, required=True, help="Path to model state_dict"
    )
    parser.add_argument(
        "--output_static",
        type=str,
        required=True,
        help="Path for static architecture visualization",
    )
    parser.add_argument(
        "--output_temporal",
        type=str,
        required=True,
        help="Path for temporal flow visualization",
    )
    parser.add_argument(
        "--format", type=str, default="png", help="Output format (png, pdf, etc.)"
    )
    parser.add_argument(
        "--interactive",
        type=bool,
        default=True,
        help="Enable interactive visualization",
    )
    return parser


def main():
    parser = create_argument_parser()
    config = parse_parameters(parser)

    model = load_model_and_weights(
        model_name=config.model_name, state_dict_path=config.state_dict, config=config
    )

    # Create visualization instance
    viz = ModelArchitectureViz(model)

    viz.print_architecture_summary()

    # Generate and save static architecture plot
    static_path = Path(config.output_static)
    static_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = viz.plot_architecture(save_path=static_path)
    plt.close(fig)

    # Generate and save temporal flow plot
    temporal_path = Path(config.output_temporal)
    temporal_path.parent.mkdir(parents=True, exist_ok=True)

    # Use standard input size for temporal flow
    input_size = (1, 3, 224, 224)  # Standard image size
    fig, ax = plot_temporal_flow(
        model=model, input_size=input_size, save_path=temporal_path
    )
    plt.close(fig)


def plot_temporal_flow(
    model: nn.Module,
    input_size: tuple,
    timesteps: int = 5,
    save_path: Optional[Path] = None,
    device: Optional[torch.device] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Visualize temporal flow of activations through model layers.

    Args:
        model: PyTorch model
        input_size: Input tensor size (batch_size, channels, height, width)
        timesteps: Number of timesteps to visualize
        save_path: Optional path to save figure
        device: Optional device to run model on

    Returns:
        Tuple of (Figure, Axes)

    Raises:
        ValueError: If input size is invalid or timesteps < 1
        RuntimeError: If model execution fails
    """
    # Validate inputs
    if len(input_size) != 4:
        raise ValueError(
            f"Expected input_size with 4 dimensions (batch_size, channels, height, width), "
            f"got {len(input_size)} dimensions: {input_size}"
        )
    if timesteps < 1:
        raise ValueError(f"timesteps must be >= 1, got {timesteps}")
    viz = ModelArchitectureViz(model)
    fig, ax = viz.create_figure(figsize=(15, 8))

    # Use model's device if none specified
    if device is None:
        device = next(model.parameters()).device

    # Create dummy input and record activations
    dummy_input = torch.randn(input_size, device=device)
    activations = {}

    def hook_fn(name):
        def hook(module, input, output):
            if output is None:
                activations[name] = 0.0  # Default value for inactive/None outputs
                return

            if isinstance(output, (tuple, list)):
                output = output[0]
            activations[name] = output.detach().mean().item()

        return hook

    # Use context manager for hooks
    hooks = []
    try:
        for name, module in model.named_modules():
            if name:
                hooks.append(module.register_forward_hook(hook_fn(name)))

        # Run model for specified timesteps
        with torch.no_grad():
            for t in range(timesteps):
                model(dummy_input)
    finally:
        # Ensure hooks are removed
        for hook in hooks:
            hook.remove()

    # Plot temporal flow
    layers = list(activations.keys())
    times = range(timesteps)

    for layer in layers:
        plt.plot(
            times,
            [activations[layer]] * timesteps,
            label=layer.split(".")[-1],
            marker="o",
        )

    plt.xlabel("Timestep")
    plt.ylabel("Mean Activation")
    plt.title("Temporal Flow of Activations")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, linestyle="--", alpha=0.7)

    if save_path:
        viz.save_figure(fig, save_path)

    return fig, ax


if __name__ == "__main__":
    main()
