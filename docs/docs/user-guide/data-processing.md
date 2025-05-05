# Data Processing and Management

This guide explains how to use DynVision's data processing pipeline, including dataset preparation, management, and loading for experiments.

## Overview

DynVision implements a comprehensive data management system that:

1. Handles standard image datasets (CIFAR10, CIFAR100, MNIST, ImageNet)
2. Organizes data into groups for specialized experiments
3. Uses symbolic links for efficient storage and organization
4. Provides performance-optimized data loading with FFCV
5. Implements specialized data loaders for neuroscience experiments

## Data Directory Structure

DynVision organizes data into a structured hierarchy:

```
data/
├── raw/                 # Original datasets in original format
│   ├── cifar10/
│   ├── cifar100/
│   └── mnist/
├── external/            # External data not part of standard datasets
├── interim/             # Processed data, organized by experiments
│   ├── cifar100/
│   │   ├── train_all/
│   │   ├── train_invertebrates/
│   │   └── test_invertebrates/
│   └── mnist/
└── processed/           # Final data ready for model consumption
    ├── cifar100/
    │   ├── train_all/
    │   │   ├── train.beton     # FFCV-formatted data
    │   │   └── val.beton
    │   └── test_invertebrates/
    └── mnist/
```

## Downloading and Preparing Datasets

DynVision uses Snakemake rules to download and prepare datasets:

```bash
# Download and prepare CIFAR100 dataset
snakemake -j1 project_paths.data.raw/cifar100/train
```

This command triggers the `get_data` rule, which downloads CIFAR100 and organizes it in the raw data directory.

For custom datasets, place them in the `data/raw` directory following the same structure.

## Creating Data Groups

DynVision supports organizing datasets into groups of categories, which is useful for specialized experiments:

### Configuring Data Groups

Data groups are defined in `config_data.yaml`:

```yaml
data_groups:
    cifar100:
        invertebrates:
            - 6  # bee
            - 7  # beetle
            - 14 # butterfly
            - 18 # caterpillar
            - 24 # cockroach
            - 26 # crab
            - 45 # lobster
            - 77 # snail
            - 79 # spider
            - 99 # worm
    mnist:
        '01':
            - 0
            - 1
        '89':
            - 8
            - 9
```

### Creating Data Group Symlinks

To create symlinks for a data group:

```bash
# Create symlinks for CIFAR100 invertebrates group
snakemake -j1 project_paths.data.interim/cifar100/train_invertebrates/folder.link
```

This command creates symbolic links to the specified categories, making them appear as a cohesive dataset.

## Converting to FFCV Format

DynVision uses FFCV for optimized data loading. To convert a dataset to FFCV format:

```bash
# Convert CIFAR100 to FFCV format
snakemake -j1 project_paths.data.processed/cifar100/train_all/train.beton
```

This creates `.beton` files that can be loaded more efficiently during training.

## Data Loaders

DynVision provides several specialized data loaders for neuroscience experiments:

### Standard Data Loader

The basic data loader with performance optimizations:

```python
from dynvision.data.dataloader import get_data_loader
from torchvision.datasets import CIFAR100

# Create dataset
dataset = CIFAR100(root='./data/raw', train=True, download=True)

# Create data loader
loader = get_data_loader(
    dataset,
    batch_size=32,
    num_workers=8,
    pin_memory=True,
    prefetch_factor=2,
    n_timesteps=20  # Repeat data over time dimension
)
```

### Stimulus Duration Loader

For experiments varying stimulus duration:

```python
from dynvision.data.dataloader import StimulusDurationDataLoader
from torchvision.datasets import CIFAR100

# Create dataset
dataset = CIFAR100(root='./data/raw', train=False, download=True)

# Create stimulus duration loader
loader = StimulusDurationDataLoader(
    dataset,
    batch_size=32,
    n_timesteps=100,        # Total sequence length
    stimulus_duration=20,   # Duration of the stimulus
    intro_duration=2,       # Duration before stimulus
)
```

This loader presents a stimulus for a specified duration, with intro and outro periods of void values.

### Stimulus Interval Loader

For experiments testing effects of repeated stimuli with varying intervals:

```python
from dynvision.data.dataloader import StimulusIntervalDataLoader
from torchvision.datasets import CIFAR100

# Create dataset
dataset = CIFAR100(root='./data/raw', train=False, download=True)

# Create stimulus interval loader
loader = StimulusIntervalDataLoader(
    dataset,
    batch_size=32,
    n_timesteps=100,        # Total sequence length
    stimulus_duration=5,    # Duration of each stimulus
    intro_duration=1,       # Duration before first stimulus
    interval_duration=10,   # Interval between stimuli
)
```

This loader presents a stimulus twice with a specified interval between presentations.

### Stimulus Contrast Loader

For experiments varying stimulus contrast:

```python
from dynvision.data.dataloader import StimulusContrastDataLoader
from torchvision.datasets import CIFAR100

# Create dataset
dataset = CIFAR100(root='./data/raw', train=False, download=True)

# Create stimulus contrast loader
loader = StimulusContrastDataLoader(
    dataset,
    batch_size=32,
    n_timesteps=100,        # Total sequence length
    stimulus_duration=15,   # Duration of the stimulus
    intro_duration=2,       # Duration before stimulus
    stimulus_contrast=0.5,  # Contrast of the stimulus (0-1)
)
```

This loader presents a stimulus with adjustable contrast.

### FFCV Data Loader

For optimal performance using FFCV:

```python
from dynvision.data.ffcv_dataloader import get_ffcv_dataloader

# Create FFCV data loader
loader = get_ffcv_dataloader(
    path='data/processed/cifar100/train_all/train.beton',
    batch_size=128,
    n_timesteps=20,
    num_workers=8,
    resolution=32,
    normalize=([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]),
    order='random',
    dtype=torch.float16
)
```

## Using Data Loaders in Experiments

DynVision's workflow system sets up appropriate data loaders based on experiment configurations:

```yaml
# In config_experiments.yaml
experiment_config:
  contrast:
    status: trained
    parameter: contrast
    data_loader: StimulusContrast
    data_args:
      tsteps: 100
      stim: 15
      contrast: [0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]  
```

To run this experiment:

```bash
snakemake -j1 experiment --config experiment=contrast
```

The system automatically sets up the `StimulusContrastDataLoader` with the specified parameters.

## Custom Data Transforms

DynVision provides configurable data transformations:

```python
from dynvision.data.transforms import get_data_transform

# Get a predefined transform
transform = get_data_transform('train')  # Standard training augmentations

# Get dataset-specific transform
transform = get_data_transform('train', data_name='cifar100')

# Create custom transform combination
transform = get_data_transform(['train', 'cifar100'])
```

Available transform presets include:
- `train`: Standard training augmentations (random flips, etc.)
- `test`: Standard test transformations
- `ffcv_train`: Optimized transforms for FFCV training
- `ffcv_test`: Optimized transforms for FFCV testing
- Dataset-specific transforms (e.g., `mnist`, `imagenet`)

## Target Transforms

For data grouping, target transforms map original labels to new indices:

```python
from dynvision.data.transforms import get_target_transform

# Transform for CIFAR100 invertebrates group
target_transform = get_target_transform('cifar100_invertebrates')
```

## Handling Large Datasets

DynVision implements several optimizations for large datasets:

### Memory-Efficient Processing

For large datasets like ImageNet:

```python
# In config_data.yaml
mounted_datasets:
    - imagenet
```

This configuration tells DynVision to use the dataset from its mounted location rather than copying it.

### Automatic Optimization

When working with large datasets, DynVision automatically adjusts parameters:

```python
# For large datasets (high resolution or many timesteps)
optimized_params = DatasetParams(
    batch_size=min(batch_size, MAX_BATCH_SIZE),    # Reduce batch size
    batches_ahead=min(batches_ahead, 2),           # Reduce prefetching
    order=OrderOption.QUASI_RANDOM,                # Better memory efficiency
    os_cache=False,                                # Disable OS cache
    dtype=torch.float16,                           # Use mixed precision
    num_workers=min(num_workers, MAX_WORKERS_LARGE_DATASET)  # Limit workers
)
```

### Custom Dataset Integration

To use a custom dataset with DynVision:

1. Place your dataset in `data/raw/your_dataset/`

2. Create a PyTorch dataset class:

```python
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        # Load your data
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, index):
        # Load and process a single sample
        return sample, target
```

3. Update configuration in `config_data.yaml`:

```yaml
data_resolution:
    your_dataset: 64
    
data_statistics:
    your_dataset:
        mean: [0.5, 0.5, 0.5]
        std: [0.25, 0.25, 0.25]
        
data_groups:
    your_dataset:
        your_group:
            - 0
            - 1
            - 2
```

4. Create data loader in your experiment:

```python
from dynvision.data.dataloader import get_data_loader
from your_module import CustomDataset

dataset = CustomDataset(
    root='data/raw/your_dataset',
    transform=get_data_transform('train')
)

loader = get_data_loader(
    dataset,
    batch_size=32,
    n_timesteps=20
)
```

## Best Practices

### Memory Management

1. **Store large datasets externally**: Use `mounted_datasets` for large datasets
2. **Use FFCV**: Enable `use_ffcv: True` for efficient data loading
3. **Adjust batch size**: Use smaller batches for large models/datasets
4. **Use mixed precision**: Set `dtype=torch.float16` for data loading

### Performance Optimization

1. **Optimize worker count**: Set `num_workers` based on CPU cores (typically CPU cores / 2)
2. **Enable pin memory**: Use `pin_memory=True` for faster GPU transfer
3. **Adjust prefetch factor**: Use `prefetch_factor=2` for balanced loading
4. **Use channels-last memory format**: For GPU optimization

### Data Organization

1. **Use symbolic links**: Organize data with symlinks instead of duplicating
2. **Keep raw data intact**: Don't modify original datasets
3. **Create appropriate data groups**: Organize categories that make sense together
4. **Use consistent naming**: Follow the established naming pattern

## Troubleshooting

### Data Loading Errors

If you encounter errors loading data:

1. Check if the dataset exists in the expected location
2. Verify that symlinks are correctly created
3. Ensure transforms are appropriate for the dataset
4. Check for sufficient disk space and memory

### FFCV Issues

If FFCV data loading fails:

1. Verify FFCV is correctly installed with required dependencies
2. Check that `.beton` files exist and are not corrupted
3. Try reducing the number of workers
4. Set `use_ffcv: False` temporarily to isolate the issue

### Memory Errors

If you encounter memory errors:

1. Reduce batch size
2. Store responses on CPU with `store_responses_on_cpu: True`
3. Use mixed precision with `dtype=torch.float16`
4. Reduce the number of timesteps

## Conclusion

DynVision's data processing pipeline provides flexible and efficient data management for neuroscience experiments. By combining symbolic linking, FFCV optimization, and specialized data loaders, DynVision enables researchers to efficiently work with various datasets and experimental conditions.

For more information on specific data loaders, see the [API Reference](../reference/data-loaders.md).
