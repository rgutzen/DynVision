# DynVision Visualization Module

This module provides comprehensive visualization capabilities for analyzing neural network models, with special focus on recurrent architectures and temporal dynamics.

## Features

### 1. Model Architecture Visualization
- Graph-based architecture visualization
- Special handling of recurrent connections
- Temporal flow analysis
- Interactive exploration in Jupyter notebooks

```python
from dynvision.visualization.plot_model_architecture import ModelArchitectureViz

# Create visualizer
viz = ModelArchitectureViz(model)

# Plot architecture
fig, ax = viz.plot_architecture()

# Show temporal dynamics
fig, ax = viz.plot_temporal_flow(input_size=(1, 3, 224, 224))
```

### 2. Enhanced Weight Analysis
- Detailed statistical analysis
- Layer-wise distribution comparisons
- Weight correlation visualization
- Sparsity analysis

```python
from dynvision.visualization.enhanced_weight_distributions import WeightDistributionViz

# Create visualizer
viz = WeightDistributionViz(model)

# Plot distributions
fig, axes = viz.plot_weight_distributions(split_by_type=True)

# Show correlations
fig, ax = viz.plot_weight_correlation_matrix(model)
```

### 3. Interactive Analysis
The module includes Jupyter notebooks for interactive analysis:
- Model architecture exploration
- Weight distribution analysis
- Temporal dynamics visualization

To generate an interactive notebook:
```bash
snakemake -j1 generate_interactive_notebook model_name=YourModel
```

### 4. Existing Visualizations
- Confusion matrix generation
- Classifier response analysis
- Experiment result plotting
- Adaptation analysis

## Usage with Snakemake

The visualization module integrates with the Snakemake workflow system. Common tasks:

```bash
# Generate all visualizations for a model
snakemake -j1 plot_model_architecture model_name=YourModel

# Create enhanced weight analysis
snakemake -j1 plot_enhanced_weights model_name=YourModel

# Generate interactive notebook
snakemake -j1 generate_interactive_notebook model_name=YourModel
```

## Customization

### Styling
All visualizations use a consistent style based on the BaseVisualization class:
```python
from dynvision.visualization.base_visualization import BaseVisualization

# Create custom visualization
class CustomViz(BaseVisualization):
    def __init__(self, style="ticks", context="talk", **kwargs):
        super().__init__(style=style, context=context, **kwargs)
```

### Output Formats
Visualizations support multiple output formats:
- PNG (default)
- SVG for vector graphics
- PDF for publication-quality figures

## Examples

### Model Architecture Analysis
```python
# Visualize model architecture with temporal flow
viz = ModelArchitectureViz(model)
fig, ax = viz.plot_architecture(show_types=True)
viz.print_architecture_summary()

# Analyze temporal dynamics
fig, ax = viz.plot_temporal_flow(timesteps=10)
```

### Weight Analysis
```python
# Analyze weight distributions
viz = WeightDistributionViz(model)
viz.print_weight_summary()

# Plot enhanced distributions
fig, axes = viz.plot_weight_distributions(
    split_by_type=True,
    show_stats=True
)
```

## Integration with Experiment Workflow

The visualization module integrates seamlessly with the experiment workflow:

1. During Training:
   - Monitor weight distributions
   - Track classifier responses

2. Post-Training Analysis:
   - Generate architecture visualizations
   - Analyze weight distributions
   - Create interactive notebooks

3. Experiment Comparison:
   - Compare architectures across models
   - Analyze adaptation patterns
   - Visualize performance differences

## Contributing

When adding new visualizations:

1. Inherit from BaseVisualization for consistent styling
2. Add corresponding Snakemake rules
3. Update the interactive notebook template
4. Document usage in this README

## Dependencies

- matplotlib
- seaborn
- networkx (for architecture visualization)
- ipywidgets (for interactive features)
- jupyter (for notebook support)