"""Test suite for Snakemake workflow validation.

This module provides comprehensive tests for the workflow including:
- Data processing validation
- Model training verification
- Visualization testing
- Configuration validation
- Resource management checks

Example:
    $ python -m pytest workflow/tests/test_workflow.py -v
"""

import os
import shutil
import subprocess
from pathlib import Path
import pytest
import yaml
import numpy as np
from PIL import Image

# Test configuration
TEST_DATA = {
    'n_classes': 2,
    'n_samples': 10,
    'image_size': (32, 32),
    'channels': 3
}

@pytest.fixture(scope='session')
def test_data_dir(tmp_path_factory):
    """Create test dataset structure.
    
    Returns:
        Path: Directory containing test data
    """
    base_dir = tmp_path_factory.mktemp('data')
    
    # Create class directories
    for class_idx in range(TEST_DATA['n_classes']):
        class_dir = base_dir / f'class_{class_idx}'
        class_dir.mkdir()
        
        # Create sample images
        for sample_idx in range(TEST_DATA['n_samples']):
            img_array = np.random.randint(
                0, 255,
                (*TEST_DATA['image_size'], TEST_DATA['channels']),
                dtype=np.uint8
            )
            img = Image.fromarray(img_array)
            img.save(class_dir / f'sample_{sample_idx}.png')
    
    return base_dir

@pytest.fixture(scope='session')
def test_config():
    """Create test configuration.
    
    Returns:
        dict: Test configuration
    """
    return {
        'model_name': 'TestModel',
        'data_name': 'test_data',
        'seed': '0000',
        'epochs': 1,
        'batch_size': 2,
        'learning_rate': 0.001,
        'model_args': {
            'n_layers': 2,
            'hidden_size': 64
        }
    }

def test_data_processing(test_data_dir):
    """Test data processing workflow.
    
    Args:
        test_data_dir: Test data directory
    """
    # Run data processing workflow
    result = subprocess.run([
        'snakemake',
        '--cores', '1',
        '--config',
        f'data_dir={test_data_dir}',
        'build_ffcv_datasets'
    ], capture_output=True)
    
    assert result.returncode == 0, \
        f"Data processing failed: {result.stderr.decode()}"
    
    # Verify outputs
    assert (test_data_dir / 'processed' / 'train.beton').exists()
    assert (test_data_dir / 'processed' / 'val.beton').exists()

def test_model_training(test_data_dir, test_config):
    """Test model training workflow.
    
    Args:
        test_data_dir: Test data directory
        test_config: Test configuration
    """
    # Create test config file
    config_path = test_data_dir / 'test_config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(test_config, f)
    
    # Run training workflow
    result = subprocess.run([
        'snakemake',
        '--cores', '1',
        '--config',
        f'config_path={config_path}',
        'train_model'
    ], capture_output=True)
    
    assert result.returncode == 0, \
        f"Model training failed: {result.stderr.decode()}"
    
    # Verify outputs
    model_path = Path('models') / test_config['model_name']
    assert model_path.exists()
    assert list(model_path.glob('*.pt'))

def test_visualization(test_data_dir, test_config):
    """Test visualization workflow.
    
    Args:
        test_data_dir: Test data directory
        test_config: Test configuration
    """
    # Run visualization workflow
    result = subprocess.run([
        'snakemake',
        '--cores', '1',
        '--config',
        f'data_dir={test_data_dir}',
        f'model_name={test_config["model_name"]}',
        'plot_confusion_matrix'
    ], capture_output=True)
    
    assert result.returncode == 0, \
        f"Visualization failed: {result.stderr.decode()}"
    
    # Verify outputs
    assert (Path('figures') / 'confusion_matrix.png').exists()

def test_resource_management():
    """Test resource management configuration."""
    # Test GPU memory calculation
    from workflow.snake_models import get_gpu_requirements
    
    class MockWildcards:
        model_name = 'DyRCNNx4'
    
    mem = get_gpu_requirements(MockWildcards(), attempt=1)
    assert mem == 8192, "Incorrect GPU memory calculation"
    
    # Test scaling with attempts
    mem_retry = get_gpu_requirements(MockWildcards(), attempt=2)
    assert mem_retry == 2 * mem, "Incorrect retry scaling"

def test_config_validation(test_config):
    """Test configuration validation.
    
    Args:
        test_config: Test configuration
    """
    from workflow.validate_config import validate_configuration
    
    # Test valid config
    validate_configuration(test_config)
    
    # Test missing required parameter
    invalid_config = test_config.copy()
    del invalid_config['model_name']
    with pytest.raises(ValueError):
        validate_configuration(invalid_config)
    
    # Test invalid parameter type
    invalid_config = test_config.copy()
    invalid_config['epochs'] = 'invalid'
    with pytest.raises(TypeError):
        validate_configuration(invalid_config)

def test_data_quality(test_data_dir):
    """Test data quality analysis.
    
    Args:
        test_data_dir: Test data directory
    """
    from workflow.data_quality import DataQualityAnalyzer
    
    analyzer = DataQualityAnalyzer(test_data_dir)
    report = analyzer.generate_quality_report()
    
    assert 'dataset_stats' in report
    assert 'class_stats' in report
    assert 'quality_metrics' in report
    assert len(report['warnings']) == 0

def test_workflow_cleanup(test_data_dir):
    """Test workflow cleanup operations.
    
    Args:
        test_data_dir: Test data directory
    """
    # Create test files
    temp_files = [
        test_data_dir / 'temp.txt',
        test_data_dir / 'checkpoint.ckpt'
    ]
    for file in temp_files:
        file.touch()
    
    # Run cleanup
    result = subprocess.run([
        'snakemake',
        '--cores', '1',
        'cleanup'
    ], capture_output=True)
    
    assert result.returncode == 0, \
        f"Cleanup failed: {result.stderr.decode()}"
    
    # Verify cleanup
    for file in temp_files:
        assert not file.exists()

def test_workflow_integration(test_data_dir, test_config):
    """Test complete workflow integration.
    
    Args:
        test_data_dir: Test data directory
        test_config: Test configuration
    """
    # Run complete workflow
    result = subprocess.run([
        'snakemake',
        '--cores', '1',
        '--config',
        f'data_dir={test_data_dir}',
        f'model_name={test_config["model_name"]}',
        'all'
    ], capture_output=True)
    
    assert result.returncode == 0, \
        f"Workflow integration failed: {result.stderr.decode()}"
    
    # Verify all outputs exist
    assert (Path('models') / test_config['model_name']).exists()
    assert (Path('figures')).exists()
    assert (Path('reports')).exists()

if __name__ == '__main__':
    pytest.main([__file__, '-v'])