# DynVision Organization

This reference document explains the core organizational principles and structure of the DynVision toolbox.

## Core Principles

DynVision is organized around three fundamental principles:

1. **Modular Architecture**: Each component is designed as a self-contained module with clear responsibilities and interfaces.
2. **Biological Plausibility**: The structure reflects the hierarchical and recurrent nature of biological visual systems.
3. **Experimental Workflow**: The organization supports systematic experimentation from data preparation to result analysis.

## Component Organization

The toolbox is structured into specialized modules, each serving a distinct purpose:

```
dynvision/
├── model_components/     # Neural building blocks
├── models/              # Complete architectures
├── data/                # Data management
├── losses/              # Training objectives
├── runtime/             # Execution handling
├── workflow/            # Experiment management
├── visualization/       # Analysis tools
├── utils/               # Shared utilities
├── configs/             # Configuration system
└── project_paths.py     # Centralized path handling
```

## Module Structure and Purpose

### Model Components Module

The `model_components` module provides the fundamental building blocks for neural networks:

1. **Base Classes**: Abstract classes defining common interfaces
   - `LightningBase`: PyTorch Lightning integration
   - `UtilityBase`: Core neural network utilities

2. **Neural Components**:
   - `dynamics_solver.py`: numerical integration methods
   - `recurrence.py`: recurrent connection implementations
   - `topographical_recurrence`: spatially-constraint recurrent connections
   - `layer_connections.py`: skip and feedback connectivity patterns
   - `supralinearity.py`: nonlinear activation functions

For details, see [Model Components Reference](model-components.md).

### Models Module

The `models` module implements complete neural architectures:

1. **Core Implementations**:
   - Research models (DyRCNN)
   - Standard architectures (e.g. ResNet, AlexNet CorNetRT)
   - Custom architectures

2. **Model Organization**:
   - Each model in a separate file
   - Consistent inheritance from base classes
   - Standardized configuration interface

### Data Module

The `data` module manages all data-related operations:

1. **Data Loading with Pytorch**:
   - `datasets.py`: Dataset implementations
   - `dataloader.py`: PyTorch data loaders

2. **Data Loading with FFCV**:
   - `ffcv_datasets.py`: Optimized dataset compression
   - `ffcv_dataloader.py`: Optimized loading pipelines

2. **Processing Pipeline**:
   - `transforms.py`: Data transformations
   - `operations.py`, `ffcv_operations.py`: Processing operations
   - `get_data.py`: Dataset acquisition

### Losses Module

The `losses` module implements training objectives:

1. **Loss Functions**:
   - `base_loss.py`: Abstract base classes
   - Task-specific losses (classification, energy)
   - Custom biological constraints

2. **Organization**:
   - Modular implementation
   - Configurable parameters
   - Composition support

### Runtime Module

The `runtime` module handles execution:

1. **Core Components**:
   - `init_model.py`: Model initialization
   - `train_model.py`: Training procedures
   - `test_model.py`: Evaluation routines

2. **Integration**:
   - PyTorch Lightning integration
   - Experiment tracking
   - Resource management

### Workflow Module

The workflow system orchestrates experiments through Snakemake:

1. **Core Workflows**:
   - `snake_data.smk`: Data preparation
   - `snake_runtime.smk`: Execution
   - `snake_visualizations.smk`: Visualization
   - `snake_experiments.smk`: Running testing sets
   - `Snakefile`: Main workflow putting it all together

2. **Organization**:
   - Modular rule definitions
   - Dependency management
   - Resource allocation

For usage details, see the [Workflows Guide](../user-guide/workflows.md).

### Visualization Module

The `visualization` module provides analysis tools:

1. **Plot Types**:
   - Model responses
   - Training dynamics
   - Network analysis
   - Result visualization

2. **Components**:
   - `callbacks.py`: Runtime visualization
   - `plot_*.py`: Specialized plotting functions
   - Analysis utilities

### Utils Module

The `utils` module provides shared functionality:

1. **Utility Categories**:
   - `config_utils.py`: Configuration handling
   - `data_utils.py`: Data operations
   - `model_utils.py`: Model operations
   - `torch_utils.py`: PyTorch helpers
   - `type_utils.py`: Type checking
   - `visualization_utils.py`: Plotting helpers

2. **Organization**:
   - Function-specific files
   - Consistent interfaces
   - Shared type definitions

### Configuration Module

The configuration system manages all settings:

1. **Config Files**:
   - `config_defaults.yaml`: Base settings
   - `config_data.yaml`: Dataset settings
   - `config_experiments.yaml`: Experiment parameters
   - `config_workflow.yaml`: Workflow settings

2. **Organization**:
   - Hierarchical structure
   - Override system
   - Environment adaptation

See the [Configuration Reference](configuration.md) for details.

## Extension Points

DynVision can be extended through several mechanisms:

1. **New Models**: Inherit from base classes in `model_components`
2. **Custom Components**: Add modules following the component interface patterns
3. **New Experiments**: Add specialized dataloaders and/or parameter sweeps in `config_experiments.yaml`
3. **Additional Workflows**: Define new Snakemake rules
4. **Visualization Tools**: Implement new analysis capabilities
5. **Loss Functions**: Add new training objectives in the `losses` module
6. **Utility Functions**: Contribute shared functionality to the `utils` module

For implementation details, refer to the [Custom Models Guide](../user-guide/custom-models.md).