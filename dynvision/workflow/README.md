# Workflow Documentation

This document provides comprehensive documentation for the rhythmic visual attention workflow system, including usage guides, examples, best practices, and troubleshooting information.

## Table of Contents
- [Overview](#overview)
- [Getting Started](#getting-started)
- [Workflow Components](#workflow-components)
- [Usage Examples](#usage-examples)
- [Configuration Guide](#configuration-guide)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)
- [Advanced Topics](#advanced-topics)

## Overview

The workflow system manages the complete pipeline for training and evaluating neural networks for rhythmic visual attention, including:
- Data preparation and preprocessing
- Model training and evaluation
- Result visualization and analysis
- Quality control and validation

### Key Features
- Modular workflow components
- Automated data processing
- Configurable model training
- Comprehensive visualization
- Quality control checks
- Resource management

## Getting Started

### Prerequisites
```bash
# Install dependencies
pip install snakemake pytorch pillow numpy pandas matplotlib

# Optional: Install FFCV for faster data loading
pip install ffcv
```

### Basic Usage
```bash
# Run complete workflow
snakemake --cores all

# Run specific component
snakemake -j1 train_model

# Run with cluster configuration
snakemake --profile profiles/slurm
```

## Workflow Components

### Data Processing (`snake_data.smk`)
Handles dataset preparation and organization:
```bash
# Create dataset
snakemake -j1 create_gabordetect dimension=48 data_subset=train

# Build FFCV datasets
snakemake -j1 build_ffcv_datasets data_name=cifar10
```

### Model Training (`snake_models.smk`)
Manages model training and evaluation:
```bash
# Initialize model
snakemake -j1 init_model model_name=DyRCNNx4 seed=0001

# Train model
snakemake -j1 train_model model_name=DyRCNNx4 seed=0001

# Test model
snakemake -j1 test_model model_name=DyRCNNx4 seed=0001
```

### Visualization (`snake_visualizations.smk`)
Creates analysis visualizations:
```bash
# Generate confusion matrix
snakemake -j1 plot_confusion_matrix model_name=DyRCNNx4

# Analyze responses
snakemake -j1 plot_classifier_responses model_name=DyRCNNx4
```

## Usage Examples

### Training a New Model
```bash
# 1. Prepare dataset
snakemake -j1 build_ffcv_datasets \
    --config data_name=cifar10

# 2. Initialize model
snakemake -j1 init_model \
    --config model_name=DyRCNNx4 \
    seed=0001 \
    model_args="rctype=full,tau=8"

# 3. Train model
snakemake -j1 train_model \
    --config model_name=DyRCNNx4 \
    seed=0001 \
    epochs=200

# 4. Evaluate model
snakemake -j1 test_model \
    --config model_name=DyRCNNx4 \
    seed=0001 \
    data_group=all
```

### Running Experiments
```bash
# Run duration experiment
snakemake -j1 experiment \
    --config experiment=duration \
    model_name=DyRCNNx4

# Analyze results
snakemake -j1 plot_experiments \
    --config experiment=duration \
    model_name=DyRCNNx4
```

## Configuration Guide

### Configuration Files
- `config_defaults.yaml`: Default parameter values
- `config_data.yaml`: Dataset configurations
- `config_workflow.yaml`: Workflow execution settings
- `config_experiments.yaml`: Experiment definitions

### Example Configuration
```yaml
# Model configuration
model_name: DyRCNNx4
seed: "0001"
model_args:
  rctype: full
  tau: 8
  trc: 6

# Training configuration
epochs: 200
batch_size: 256
learning_rate: 0.001
loss:
  - CrossEntropyLoss

# Data configuration
data_name: cifar10
data_group: all
```

## Best Practices

### Resource Management
- Use appropriate batch sizes for your GPU memory
- Enable gradient accumulation for large models
- Monitor GPU memory usage
- Use FFCV datasets for faster loading

### Model Training
- Start with small experiments
- Monitor validation metrics
- Use early stopping
- Save checkpoints regularly
- Track training progress

### Data Processing
- Validate datasets before training
- Check class distributions
- Monitor data quality metrics
- Use appropriate preprocessing

### Workflow Organization
- Keep configurations versioned
- Use meaningful experiment names
- Organize results systematically
- Clean up temporary files

## Troubleshooting

### Common Issues

#### Data Processing
```
Issue: FFCV dataset building fails
Solution: Check disk space and file permissions
```

#### Model Training
```
Issue: Out of memory errors
Solution: Reduce batch size or enable gradient accumulation
```

#### Visualization
```
Issue: Missing plots
Solution: Check output directories and permissions
```

### Error Messages
- `ValueError: Invalid model name`: Check model_name in config
- `RuntimeError: CUDA out of memory`: Reduce batch size
- `FileNotFoundError`: Check file paths and permissions

## Advanced Topics

### Custom Models
```python
# Define model in models/
class CustomModel(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        # Model definition

# Use in workflow
snakemake -j1 train_model --config model_name=CustomModel
```

### Resource Configuration
```yaml
# profiles/slurm/config.yaml
cluster:
  mkdir -p logs/slurm
  sbatch
    --cpus-per-task={threads}
    --mem={resources.mem_mb}
    --time={resources.time}
    --gres=gpu:{resources.gpu}
default-resources:
  - mem_mb=16000
  - time=60
  - gpu=1
```

### Quality Control
```bash
# Run quality checks
snakemake -j1 check_data_quality \
    --config data_name=cifar10

# Generate quality report
snakemake -j1 plot_quality_metrics
```

### Performance Optimization
- Use FFCV for data loading
- Enable GPU memory optimization
- Use gradient accumulation
- Monitor resource usage

### Testing
```bash
# Run workflow tests
python workflow/tests/run_tests.py

# Run specific test
pytest workflow/tests/test_workflow.py -k test_data_processing
```

## Additional Resources

### Documentation
- [Snakemake Documentation](https://snakemake.readthedocs.io/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [FFCV Documentation](https://ffcv.io/)

### References
- Project paper: [Link to paper]()
- Related work: [Links to related papers]()

### Support
For issues and questions:
- Open an issue on GitHub
- Contact the maintainers
- Check the FAQ section

## Contributing
See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines on:
- Code style
- Pull requests
- Testing requirements
- Documentation standards