"""Test data generation utilities for workflow testing.

This module provides utilities for generating test datasets
and configurations for workflow testing.

Example:
    ```python
    from test_data import TestDataGenerator
    
    generator = TestDataGenerator()
    test_data = generator.create_test_dataset()
    ```
"""

import logging
from pathlib import Path
import numpy as np
from PIL import Image
import yaml
import torch
import torch.nn as nn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('workflow.test_data')

class TestModel(nn.Module):
    """Simple test model for workflow validation.
    
    This model provides a minimal implementation for testing
    the training and evaluation workflow.
    """
    
    def __init__(
        self,
        input_size: int = 32 * 32 * 3,
        hidden_size: int = 64,
        n_classes: int = 2
    ):
        """Initialize test model.
        
        Args:
            input_size: Input dimension
            hidden_size: Hidden layer size
            n_classes: Number of output classes
        """
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, n_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor
            
        Returns:
            torch.Tensor: Output predictions
        """
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        return self.fc2(x)

class TestDataGenerator:
    """Generate test data for workflow validation."""
    
    def __init__(
        self,
        n_classes: int = 2,
        n_samples: int = 10,
        image_size: tuple = (32, 32),
        channels: int = 3
    ):
        """Initialize generator.
        
        Args:
            n_classes: Number of classes
            n_samples: Samples per class
            image_size: Image dimensions
            channels: Number of channels
        """
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.image_size = image_size
        self.channels = channels
    
    def create_test_dataset(
        self,
        output_dir: Path,
        splits: list = ['train', 'val', 'test']
    ) -> Path:
        """Create test dataset.
        
        Args:
            output_dir: Output directory
            splits: Dataset splits to create
            
        Returns:
            Path: Dataset directory
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for split in splits:
            split_dir = output_dir / split
            split_dir.mkdir(exist_ok=True)
            
            for class_idx in range(self.n_classes):
                class_dir = split_dir / f'class_{class_idx}'
                class_dir.mkdir(exist_ok=True)
                
                for sample_idx in range(self.n_samples):
                    self._create_sample_image(
                        class_dir / f'sample_{sample_idx}.png',
                        class_idx
                    )
        
        logger.info(f"Created test dataset in {output_dir}")
        return output_dir
    
    def _create_sample_image(
        self,
        output_path: Path,
        class_idx: int
    ) -> None:
        """Create a single test image.
        
        Args:
            output_path: Output path
            class_idx: Class index for the image
        """
        # Create base image
        img_array = np.random.randint(
            0, 255,
            (*self.image_size, self.channels),
            dtype=np.uint8
        )
        
        # Add class-specific pattern
        pattern = np.zeros_like(img_array)
        pattern[class_idx::self.n_classes] = 255
        img_array = (img_array * 0.8 + pattern * 0.2).astype(np.uint8)
        
        # Save image
        img = Image.fromarray(img_array)
        img.save(output_path)

    def create_test_model(
        self,
        output_dir: Path,
        model_name: str = 'TestModel'
    ) -> Path:
        """Create test model checkpoint.
        
        Args:
            output_dir: Output directory
            model_name: Model name
            
        Returns:
            Path: Model checkpoint path
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create model
        model = TestModel(
            input_size=self.image_size[0] * self.image_size[1] * self.channels,
            n_classes=self.n_classes
        )
        
        # Save checkpoint
        checkpoint_path = output_dir / f'{model_name}_init.pt'
        torch.save(model.state_dict(), checkpoint_path)
        
        logger.info(f"Created test model checkpoint at {checkpoint_path}")
        return checkpoint_path

    def create_test_config(
        self,
        output_dir: Path,
        model_name: str = 'TestModel'
    ) -> Path:
        """Create test configuration.
        
        Args:
            output_dir: Output directory
            model_name: Model name
            
        Returns:
            Path: Configuration file path
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create config
        config = {
            'model_name': model_name,
            'data_name': 'test_data',
            'seed': '0000',
            'epochs': 1,
            'batch_size': 2,
            'learning_rate': 0.001,
            'model_args': {
                'input_size': self.image_size[0] * self.image_size[1] * self.channels,
                'hidden_size': 64,
                'n_classes': self.n_classes
            },
            'data_resolution': {
                'test_data': max(self.image_size)
            }
        }
        
        # Save config
        config_path = output_dir / 'test_config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        logger.info(f"Created test configuration at {config_path}")
        return config_path

def create_test_environment(
    base_dir: Path,
    n_classes: int = 2,
    n_samples: int = 10
) -> tuple:
    """Create complete test environment.
    
    Args:
        base_dir: Base directory
        n_classes: Number of classes
        n_samples: Samples per class
        
    Returns:
        Tuple containing:
            - Dataset directory
            - Model checkpoint path
            - Configuration path
    """
    generator = TestDataGenerator(
        n_classes=n_classes,
        n_samples=n_samples
    )
    
    # Create directory structure
    base_dir = Path(base_dir)
    data_dir = base_dir / 'data'
    model_dir = base_dir / 'models'
    config_dir = base_dir / 'configs'
    
    # Generate test data
    dataset_dir = generator.create_test_dataset(data_dir)
    model_path = generator.create_test_model(model_dir)
    config_path = generator.create_test_config(config_dir)
    
    return dataset_dir, model_path, config_path

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate test data")
    parser.add_argument('output_dir', type=Path, help="Output directory")
    parser.add_argument(
        '--n_classes',
        type=int,
        default=2,
        help="Number of classes"
    )
    parser.add_argument(
        '--n_samples',
        type=int,
        default=10,
        help="Samples per class"
    )
    args = parser.parse_args()
    
    try:
        dataset_dir, model_path, config_path = create_test_environment(
            args.output_dir,
            args.n_classes,
            args.n_samples
        )
        logger.info("Test environment created successfully")
        logger.info(f"Dataset: {dataset_dir}")
        logger.info(f"Model: {model_path}")
        logger.info(f"Config: {config_path}")
    except Exception as e:
        logger.error(f"Failed to create test environment: {str(e)}")