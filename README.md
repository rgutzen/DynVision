# DynVision
> A modular modeling toolbox for constructing and evaluating recurrent convolutional neural networks (RCNNs) with biologically inspired dynamics.

## Overview

DynVision provides a flexible framework for building and analyzing biologically plausible RCNNs, incorporating key properties of the visual cortex:
- Realistic recurrent architectures
- Activity evolution governed by dynamical systems equations
- Structured connectivity reflecting cortical arrangements
- Computational efficiency and scalability

Built on modern deep learning infrastructure:
- PyTorch for performative tensor operations
- PyTorch Lightning for training organization
- FFCV for optimized data loading
- Snakemake for workflow management
- YAML for configuration

## Features

- **Modular Architecture**: Easily extensible with new models and components
- **Biological Plausibility**: Incorporates key visual cortex properties
- **Performance Optimized**: Efficient data loading and GPU utilization
- **Reproducible Research**: Structured workflows and configurations
- **Comprehensive Analysis**: Built-in visualization and analysis tools

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/dynvision.git
cd dynvision
```

2. Create and activate a conda environment:
```bash
conda create -n dynvision python=3.11
conda activate dynvision
```

3. Install dependencies:
```bash
# Core dependencies
pip install -e .

# Development dependencies
pip install -e ".[dev]"

# Documentation dependencies
pip install -e ".[doc]"
```

## Quick Start

1. Prepare your dataset:
```bash
# Download and prepare CIFAR-10
snakemake -j1 get_data --config data_name=cifar10

# Build optimized FFCV dataset
snakemake -j1 build_ffcv_datasets --config data_name=cifar10
```

2. Train a model:
```bash
# Initialize model
snakemake -j1 init_model \
    --config model_name=DyRCNNx4 \
    seed=0001 \
    model_args="rctype=full,tau=8"

# Train model
snakemake -j1 train_model \
    --config model_name=DyRCNNx4 \
    seed=0001 \
    epochs=200

# Evaluate model
snakemake -j1 test_model \
    --config model_name=DyRCNNx4 \
    seed=0001
```

3. Visualize results:
```bash
# Plot confusion matrix
snakemake -j1 plot_confusion_matrix \
    --config model_name=DyRCNNx4

# Analyze responses
snakemake -j1 plot_classifier_responses \
    --config model_name=DyRCNNx4
```

## Project Organization

```
├── LICENSE            <- Open-source license
├── Makefile           <- Makefile with convenience commands
├── README.md          <- The top-level README for developers
├── data
│   ├── external       <- Data from third party sources
│   ├── interim        <- Intermediate data that has been transformed
│   ├── processed      <- The final, canonical data sets for modeling
│   └── raw           <- The original, immutable data dump
│
├── docs               <- Documentation (mkdocs)
├── models             <- Trained and serialized models
├── notebooks          <- Jupyter notebooks
├── pyproject.toml     <- Project configuration and dependencies
├── references         <- Data dictionaries and explanatory materials
├── reports           <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures       <- Generated graphics and figures
├── setup.cfg         <- Configuration for development tools
│
└── dynvision         <- Source code package
    ├── cluster       <- Utility scripts for cluster execution
    |   └── profiles  <- cluster config for specific queueing systems (e.g. slurm)
    ├── data          <- Data loading and processing
    ├── models        <- Model implementations
    ├── model_components <- Modules to use in model implementations
    ├── runtime       <- init, train, and test models
    ├── utils         <- collection of utility functions
    ├── losses        <- Loss function implementations
    ├── visualization <- Analysis and visualization tools
    ├── workflow      <- Workflow management
    |   └── tests     <- unit tests to validate workflow
    └── project_paths.py  <- paths handling across the project
```

## Development

### Environment Setup

1. Install development dependencies:
```bash
pip install -e ".[dev]"
```

2. Install pre-commit hooks:
```bash
pre-commit install
```

### Code Style

- We use Black for code formatting
- Follow PEP 8 guidelines
- Use type hints
- Write docstrings in NumPy format

### Testing

Run the test suite:
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_models.py

# Run with coverage
pytest --cov=dynvision
```

### Documentation

Build the documentation:
```bash
# Install documentation dependencies
pip install -e ".[doc]"

# Build docs
cd docs
mkdocs build

# Serve locally
mkdocs serve
```

## Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and ensure they pass
5. Submit a pull request

Please read our [Contributing Guidelines](CONTRIBUTING.md) for more details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use DynVision in your research, please cite:

```bibtex
@software{dynvision2024,
  title = {DynVision: A Toolbox for Modeling Dynamical Vision},
  author = {Author, A.},
  year = {2024},
  url = {https://github.com/yourusername/dynvision}
}
```

## Acknowledgments

This project builds on several key technologies:
- [PyTorch](https://pytorch.org/)
- [PyTorch Lightning](https://lightning.ai/)
- [FFCV](https://ffcv.io/)
- [Snakemake](https://snakemake.readthedocs.io/)