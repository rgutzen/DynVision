"""Test runner for workflow validation.

This script provides a convenient way to run workflow tests,
including setup and cleanup of test environments.

Example:
    $ python run_tests.py --test-data-dir /tmp/test_data
"""

import argparse
import logging
import shutil
import sys
from pathlib import Path
import pytest

from test_data import create_test_environment

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('workflow.test_runner')

def setup_test_environment(
    base_dir: Path,
    n_classes: int = 2,
    n_samples: int = 10
) -> tuple:
    """Set up test environment.
    
    Args:
        base_dir: Base directory for test data
        n_classes: Number of classes
        n_samples: Samples per class
        
    Returns:
        Tuple containing:
            - Dataset directory
            - Model checkpoint path
            - Configuration path
    """
    logger.info("Setting up test environment...")
    
    try:
        return create_test_environment(
            base_dir,
            n_classes,
            n_samples
        )
    except Exception as e:
        logger.error(f"Failed to set up test environment: {str(e)}")
        raise

def cleanup_test_environment(base_dir: Path) -> None:
    """Clean up test environment.
    
    Args:
        base_dir: Base directory to clean up
    """
    logger.info("Cleaning up test environment...")
    
    try:
        if base_dir.exists():
            shutil.rmtree(base_dir)
    except Exception as e:
        logger.error(f"Failed to clean up test environment: {str(e)}")
        raise

def run_workflow_tests(
    test_data_dir: Path,
    cleanup: bool = True
) -> int:
    """Run workflow tests.
    
    Args:
        test_data_dir: Test data directory
        cleanup: Whether to clean up after tests
        
    Returns:
        int: Test exit code
    """
    logger.info("Running workflow tests...")
    
    try:
        # Set up environment
        dataset_dir, model_path, config_path = setup_test_environment(
            test_data_dir
        )
        
        # Run tests
        exit_code = pytest.main([
            'test_workflow.py',
            '-v',
            f'--test-data-dir={dataset_dir}',
            f'--model-path={model_path}',
            f'--config-path={config_path}'
        ])
        
        # Clean up if requested
        if cleanup:
            cleanup_test_environment(test_data_dir)
        
        return exit_code
        
    except Exception as e:
        logger.error(f"Test execution failed: {str(e)}")
        return 1

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run workflow tests")
    parser.add_argument(
        '--test-data-dir',
        type=Path,
        default=Path('/tmp/workflow_tests'),
        help="Directory for test data"
    )
    parser.add_argument(
        '--n-classes',
        type=int,
        default=2,
        help="Number of classes in test data"
    )
    parser.add_argument(
        '--n-samples',
        type=int,
        default=10,
        help="Samples per class in test data"
    )
    parser.add_argument(
        '--no-cleanup',
        action='store_true',
        help="Don't clean up test data after running"
    )
    args = parser.parse_args()
    
    try:
        exit_code = run_workflow_tests(
            args.test_data_dir,
            cleanup=not args.no_cleanup
        )
        sys.exit(exit_code)
        
    except Exception as e:
        logger.error(f"Test runner failed: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()