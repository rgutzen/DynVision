# DynVision Documentation

## Introduction

DynVision is a modular modeling toolbox for constructing and evaluating recurrent convolutional neural networks (RCNNs) with biologically inspired dynamics. The toolbox provides a flexible framework for investigating neural mechanisms of visual perception while maintaining computational efficiency.

## Technical Overview

### Architecture

DynVision is built on a modular architecture with several key components:

1. **Model Components**
   - Dynamics solvers for neural activity evolution
   - Recurrent connection implementations
   - Layer connections (skip, feedback)
   - Retina preprocessing
   - Supralinearity functions

2. **Model Implementations**
   - DyRCNN (base implementation)
   - CORDSNet (cortically organized)
   - CorNet (various versions)
   - Standard models (ResNet, AlexNet)

3. **Training Infrastructure**
   - PyTorch Lightning integration
   - FFCV data loading optimization
   - Distributed training support
   - Checkpoint management

4. **Workflow Management**
   - Snakemake-based pipelines
   - Cluster execution support
   - Configuration management
   - Experiment tracking

### Key Features

1. **Biological Plausibility**
   - Realistic recurrent architectures
   - Dynamical systems integration
   - Structured connectivity patterns
   - Biologically-inspired loss functions

2. **Performance Optimization**
   - FFCV data loading
   - GPU memory optimization
   - Batch processing
   - Resource management

3. **Analysis Tools**
   - Response visualization
   - Weight distribution analysis
   - Confusion matrix generation
   - Temporal dynamics analysis

## Getting Started

### Installation

1. **Environment Setup**
   ```bash
   conda create -n dynvision python=3.11
   conda activate dynvision
   ```

2. **Install Dependencies**
   ```bash
   # Core installation
   pip install -e .
   
   # Development tools
   pip install -e ".[dev]"
   
   # Documentation
   pip install -e ".[doc]"
   ```

### Basic Usage

1. **Data Preparation**
   ```python
   from dynvision.data import get_dataset
   
   # Load dataset
   dataset = get_dataset(
       path="path/to/data",
       data_transform="cifar10_train"
   )
   ```

2. **Model Creation**
   ```python
   from dynvision.models import DyRCNNx4
   
   # Initialize model
   model = DyRCNNx4(
       input_dims=(20, 3, 32, 32),
       n_classes=10,
       recurrence_type="full",
       tau=8.0
   )
   ```

3. **Training**
   ```python
   from pytorch_lightning import Trainer
   
   # Configure trainer
   trainer = Trainer(
       max_epochs=200,
       accelerator="gpu",
       devices=1
   )
   
   # Train model
   trainer.fit(model, train_dataloader, val_dataloader)
   ```

## API Reference

### Model Components

- [Dynamics Solver](api/dynamics_solver.md)
- [Layer Connections](api/layer_connections.md)
- [Recurrence](api/recurrence.md)
- [Retina](api/retina.md)
- [Supralinearity](api/supralinearity.md)

### Models

- [DyRCNN](api/models/dyrcnn.md)
- [CORDSNet](api/models/cordsnet.md)
- [CorNet](api/models/cornet.md)
- [ResNet](api/models/resnet.md)

### Data Handling

- [Datasets](api/data/datasets.md)
- [Transforms](api/data/transforms.md)
- [FFCV Integration](api/data/ffcv.md)

### Training

- [Lightning Base](api/training/lightning_base.md)
- [Loss Functions](api/training/losses.md)
- [Callbacks](api/training/callbacks.md)

### Visualization

- [Response Analysis](api/visualization/responses.md)
- [Weight Analysis](api/visualization/weights.md)
- [Performance Metrics](api/visualization/metrics.md)

## Advanced Topics

### Custom Models

Learn how to create custom models by extending the base classes:

```python
from dynvision.models import DyRCNN

class CustomModel(DyRCNN):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Custom initialization
        
    def _define_architecture(self):
        # Define model architecture
        pass
```

### Workflow Configuration

Configure experiments using YAML:

```yaml
model_name: DyRCNNx4
seed: "0001"
model_args:
  rctype: full
  tau: 8
  trc: 6

training:
  epochs: 200
  batch_size: 256
  learning_rate: 0.001
  loss:
    - CrossEntropyLoss
```

### Cluster Execution

Run on compute clusters:

```bash
# Submit job
sbatch cluster/snakejob.sh

# Monitor progress
tail -f logs/slurm/*.out
```

## Contributing

See our [Contributing Guide](contributing.md) for details on:
- Code style
- Pull request process
- Testing requirements
- Documentation standards

## Support

- [Issue Tracker](https://github.com/yourusername/dynvision/issues)
- [Discussion Forum](https://github.com/yourusername/dynvision/discussions)
- [Email Support](mailto:support@dynvision.org)