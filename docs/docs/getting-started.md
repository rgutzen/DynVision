# Getting Started with DynVision

This guide will help you get up and running with DynVision, walking through installation, basic usage, and common workflows.

## Prerequisites

Before installing DynVision, ensure you have:

- Python 3.11 or later
- CUDA-capable GPU (recommended)
- Git
- Make (optional, for convenience commands)

## Installation

### 1. Environment Setup

First, create and activate a conda or mamba environment. Mamba is a faster alternative to conda and is now the recommended environment handler. The mamba and conda commands are interchangeable.
See [https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html] for a guide how to install them.

#### Using Mamba/Conda:

```bash
# Create environment
mamba create -n dynvision python=3.11
mamba activate dynvision

# Install CUDA dependencies
mamba install -c conda-forge cudatoolkit=11.3
```

Both methods will set up an isolated environment with the required Python version and CUDA dependencies.

### 2. Install DynVision

Clone and install the repository:

```bash
# Clone repository
git clone https://github.com/yourusername/dynvision.git
cd DynVision

# Install package
pip install -e .
```

With this installation type, you may edit the codebase and have any changes directly applied to the installed dynvision package without reinstallation.

**Troubleshooting:** In case of package or dependency issues, it can help to instead install the package that cause the environment error with `conda install pkg_name`, comment it out in the pyproject.toml, and run `pip install -e .` again.


### 3. Verify Installation (optional)

Run the test suite to verify your installation:

```bash
pytest
```

## Basic Usage

### 1. Data Preparation

DynVision supports several standard datasets and custom data formats:

```python
from dynvision.data import get_dataset, get_data_loader

# Load standard dataset
dataset = get_dataset(
    path="path/to/data",
    data_transform="cifar10_train"
)

# Create optimized data loader
dataloader = get_data_loader(
    dataset,
    batch_size=32,
    num_workers=8
)
```

### 2. Model Creation

Create a model with your desired configuration:

```python
from dynvision.models import DyRCNNx4

# Initialize model
model = DyRCNNx4(
    input_dims=(20, 3, 32, 32),  # (timesteps, channels, height, width)
    n_classes=10,
    recurrence_type="full",
    tau=8.0,
    dt=2.0
)
```

### 3. Training

Train your model using PyTorch Lightning:

```python
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

# Configure checkpointing
checkpoint_callback = ModelCheckpoint(
    monitor="val_accuracy",
    dirpath="models",
    filename="model-{epoch:02d}-{val_accuracy:.2f}",
    save_top_k=1,
    mode="max",
)

# Configure trainer
trainer = Trainer(
    max_epochs=200,
    accelerator="gpu",
    devices=1,
    callbacks=[checkpoint_callback]
)

# Train model
trainer.fit(model, train_dataloader, val_dataloader)
```

### 4. Evaluation

Evaluate your trained model:

```python
# Test model
trainer.test(model, test_dataloader)

# Analyze responses
from dynvision.visualization import plot_classifier_responses

responses_df = model.get_classifier_dataframe()
plot_classifier_responses(responses_df)
```

## Common Workflows

### Using Snakemake

DynVision uses Snakemake for workflow management:

```bash
# Prepare dataset
snakemake -j1 get_data --config data_name=cifar10

# Build optimized dataset
snakemake -j1 build_ffcv_datasets --config data_name=cifar10

# Train model
snakemake -j1 train_model \
    --config model_name=DyRCNNx4 \
    seed=0001 \
    epochs=200
```

### Cluster Execution

Run on compute clusters:

```bash
# Submit job
sbatch cluster/snakejob.sh

```

## Configuration

### Model Configuration

Configure model parameters in YAML:

```yaml
model_name: DyRCNNx4
seed: "0001"
model_args:
  rctype: full
  tau: 8
  trc: 6
  dt: 2

training:
  epochs: 200
  batch_size: 256
  learning_rate: 0.001
```

### Data Configuration

Configure dataset parameters:

```yaml
data_name: cifar10
data_resolution: 32
train_ratio: 0.9
data_groups:
  cifar10:
    all: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    animals: [2, 3, 4, 5, 6, 7]
    vehicles: [0, 1, 8, 9]
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size
   - Enable gradient accumulation
   - Use mixed precision training

2. **Slow Data Loading**
   - Increase num_workers
   - Use FFCV datasets
   - Check disk I/O

3. **NaN Losses**
   - Check learning rate
   - Verify input normalization
   - Monitor gradient norms

### Getting Help

If you encounter issues:

1. Check the [FAQ](faq.md)
2. Search [existing issues](https://github.com/yourusername/dynvision/issues)
3. Create a new issue with:
   - Error message
   - Minimal reproduction code
   - Environment details

## Next Steps

- [API Reference](api/index.md)
- [Advanced Topics](advanced/index.md)
- [Contributing Guide](contributing.md)