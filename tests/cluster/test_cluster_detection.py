"""Unit tests for cluster execution detection.

Tests the is_cluster_execution() function that automatically detects
whether code is running on an HPC cluster vs. local workstation.

See: docs/development/planning/cluster-execution.md
"""

import os
import sys
from pathlib import Path

import pytest


# Import the function under test
# Note: Since snake_utils.smk is a Snakefile, we need to load it as a module
def get_is_cluster_execution():
    """
    Import is_cluster_execution() from snake_utils.smk.

    This is a helper to load the function from the Snakefile module.
    We can't import directly since .smk files aren't standard Python modules.
    """
    # Find the project root
    test_dir = Path(__file__).parent
    project_root = test_dir.parent.parent

    # Read and execute the snake_utils.smk file to get the function
    snake_utils_path = project_root / 'dynvision' / 'workflow' / 'snake_utils.smk'

    # Create a module-like namespace to execute the code in
    namespace = {
        '__file__': str(snake_utils_path),
        '__name__': 'snake_utils',
    }

    # Execute the file to load the function
    with open(snake_utils_path, 'r') as f:
        # Read up to the function definition (we don't need the whole file)
        # This avoids executing Snakemake-specific code
        code_lines = []
        in_function = False
        indent_level = 0

        for line in f:
            # Start capturing at the function definition
            if 'def is_cluster_execution()' in line:
                in_function = True
                indent_level = len(line) - len(line.lstrip())

            if in_function:
                code_lines.append(line)

                # Stop when we hit the next function or top-level code
                if line.strip() and not line.strip().startswith('#'):
                    current_indent = len(line) - len(line.lstrip())
                    if current_indent <= indent_level and 'def is_cluster_execution()' not in line:
                        # We've exited the function
                        break

    # Remove the last line (which triggered the break)
    if code_lines:
        code_lines.pop()

    # Add necessary imports
    imports = """
import os
import logging

# Mock logger for testing
class MockLogger:
    def info(self, msg): pass
    def debug(self, msg): pass
    def warning(self, msg): pass

pylogger = MockLogger()
"""

    full_code = imports + '\n' + ''.join(code_lines)

    # Execute the code
    exec(full_code, namespace)

    return namespace['is_cluster_execution']


@pytest.fixture(scope='module')
def is_cluster_execution():
    """Fixture providing the is_cluster_execution function."""
    return get_is_cluster_execution()


class TestClusterDetection:
    """Test suite for cluster execution detection."""

    def test_slurm_job_id_detection(self, is_cluster_execution, monkeypatch):
        """Test detection via SLURM_JOB_ID variable."""
        # Clear any existing cluster vars
        self._clear_cluster_vars(monkeypatch)

        # Set SLURM_JOB_ID
        monkeypatch.setenv('SLURM_JOB_ID', '12345')

        assert is_cluster_execution() is True

    def test_slurm_jobid_alternative(self, is_cluster_execution, monkeypatch):
        """Test detection via SLURM_JOBID (alternative spelling)."""
        self._clear_cluster_vars(monkeypatch)
        monkeypatch.setenv('SLURM_JOBID', '67890')

        assert is_cluster_execution() is True

    def test_pbs_detection(self, is_cluster_execution, monkeypatch):
        """Test detection via PBS_JOBID variable."""
        self._clear_cluster_vars(monkeypatch)
        monkeypatch.setenv('PBS_JOBID', '1234.server')

        assert is_cluster_execution() is True

    def test_lsf_detection(self, is_cluster_execution, monkeypatch):
        """Test detection via LSB_JOBID variable."""
        self._clear_cluster_vars(monkeypatch)
        monkeypatch.setenv('LSB_JOBID', '5678')

        assert is_cluster_execution() is True

    def test_sge_detection(self, is_cluster_execution, monkeypatch):
        """Test detection via SGE_TASK_ID variable."""
        self._clear_cluster_vars(monkeypatch)
        monkeypatch.setenv('SGE_TASK_ID', '1')

        assert is_cluster_execution() is True

    def test_sge_with_job_id(self, is_cluster_execution, monkeypatch):
        """Test SGE detection with both JOB_ID and SGE_TASK_ID."""
        self._clear_cluster_vars(monkeypatch)
        monkeypatch.setenv('JOB_ID', '111')
        monkeypatch.setenv('SGE_TASK_ID', '1')

        assert is_cluster_execution() is True

    def test_job_id_alone_insufficient(self, is_cluster_execution, monkeypatch):
        """Test that JOB_ID alone is not sufficient (too generic)."""
        self._clear_cluster_vars(monkeypatch)
        monkeypatch.setenv('JOB_ID', '111')

        # Should NOT detect cluster without SGE_TASK_ID
        assert is_cluster_execution() is False

    def test_local_execution(self, is_cluster_execution, monkeypatch):
        """Test that local execution is detected when no cluster vars present."""
        self._clear_cluster_vars(monkeypatch)

        assert is_cluster_execution() is False

    def test_multiple_cluster_vars(self, is_cluster_execution, monkeypatch):
        """Test detection when multiple cluster variables are set."""
        self._clear_cluster_vars(monkeypatch)
        monkeypatch.setenv('SLURM_JOB_ID', '12345')
        monkeypatch.setenv('PBS_JOBID', '67890')  # Shouldn't happen, but test anyway

        assert is_cluster_execution() is True

    def test_empty_cluster_var(self, is_cluster_execution, monkeypatch):
        """Test that empty cluster variable still triggers detection."""
        self._clear_cluster_vars(monkeypatch)
        monkeypatch.setenv('SLURM_JOB_ID', '')

        # Empty string is still "present" in environment
        assert is_cluster_execution() is True

    def test_unrelated_env_vars(self, is_cluster_execution, monkeypatch):
        """Test that unrelated environment variables don't trigger detection."""
        self._clear_cluster_vars(monkeypatch)
        monkeypatch.setenv('HOME', '/home/user')
        monkeypatch.setenv('PATH', '/usr/bin')
        monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0')

        assert is_cluster_execution() is False

    # Helper methods

    @staticmethod
    def _clear_cluster_vars(monkeypatch):
        """Clear all cluster-related environment variables."""
        cluster_vars = [
            'SLURM_JOB_ID',
            'SLURM_JOBID',
            'PBS_JOBID',
            'LSB_JOBID',
            'SGE_TASK_ID',
            'JOB_ID',
            'PBS_ARRAYID',
            'LOADL_STEP_ID',
        ]

        for var in cluster_vars:
            monkeypatch.delenv(var, raising=False)


class TestClusterDetectionEdgeCases:
    """Test edge cases and unusual scenarios."""

    def test_case_sensitivity(self, is_cluster_execution, monkeypatch):
        """Test that variable names are case-sensitive."""
        self._clear_cluster_vars(monkeypatch)

        # Lowercase should NOT trigger detection
        monkeypatch.setenv('slurm_job_id', '12345')
        assert is_cluster_execution() is False

        # Uppercase should trigger detection
        monkeypatch.setenv('SLURM_JOB_ID', '12345')
        assert is_cluster_execution() is True

    def test_partial_variable_names(self, is_cluster_execution, monkeypatch):
        """Test that partial matches don't trigger detection."""
        self._clear_cluster_vars(monkeypatch)

        # These should NOT trigger detection
        monkeypatch.setenv('MY_SLURM_JOB_ID', '12345')
        assert is_cluster_execution() is False

        monkeypatch.setenv('SLURM_JOB_ID_EXTRA', '12345')
        assert is_cluster_execution() is False

    # Helper methods
    @staticmethod
    def _clear_cluster_vars(monkeypatch):
        """Clear all cluster-related environment variables."""
        cluster_vars = [
            'SLURM_JOB_ID',
            'SLURM_JOBID',
            'PBS_JOBID',
            'LSB_JOBID',
            'SGE_TASK_ID',
            'JOB_ID',
            'slurm_job_id',  # Also clear lowercase versions
            'MY_SLURM_JOB_ID',
            'SLURM_JOB_ID_EXTRA',
        ]

        for var in cluster_vars:
            monkeypatch.delenv(var, raising=False)


class TestClusterDetectionIntegration:
    """Integration tests for realistic scenarios."""

    def test_typical_slurm_environment(self, is_cluster_execution, monkeypatch):
        """Test detection in a typical SLURM job environment."""
        self._clear_cluster_vars(monkeypatch)

        # Set typical SLURM variables
        monkeypatch.setenv('SLURM_JOB_ID', '123456')
        monkeypatch.setenv('SLURM_JOB_NAME', 'test_job')
        monkeypatch.setenv('SLURM_SUBMIT_DIR', '/home/user/project')
        monkeypatch.setenv('SLURM_NODELIST', 'node001')

        assert is_cluster_execution() is True

    def test_typical_pbs_environment(self, is_cluster_execution, monkeypatch):
        """Test detection in a typical PBS job environment."""
        self._clear_cluster_vars(monkeypatch)

        # Set typical PBS variables
        monkeypatch.setenv('PBS_JOBID', '1234.server.domain')
        monkeypatch.setenv('PBS_JOBNAME', 'test_job')
        monkeypatch.setenv('PBS_O_WORKDIR', '/home/user/project')

        assert is_cluster_execution() is True

    def test_local_development_environment(self, is_cluster_execution, monkeypatch):
        """Test detection in typical local development environment."""
        self._clear_cluster_vars(monkeypatch)

        # Set typical local development variables (no cluster vars)
        monkeypatch.setenv('USER', 'developer')
        monkeypatch.setenv('HOME', '/home/developer')
        monkeypatch.setenv('CONDA_DEFAULT_ENV', 'dynvision')
        monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0,1')

        assert is_cluster_execution() is False

    # Helper methods
    @staticmethod
    def _clear_cluster_vars(monkeypatch):
        """Clear all cluster-related environment variables."""
        cluster_vars = [
            'SLURM_JOB_ID',
            'SLURM_JOBID',
            'SLURM_JOB_NAME',
            'SLURM_SUBMIT_DIR',
            'SLURM_NODELIST',
            'PBS_JOBID',
            'PBS_JOBNAME',
            'PBS_O_WORKDIR',
            'LSB_JOBID',
            'SGE_TASK_ID',
            'JOB_ID',
        ]

        for var in cluster_vars:
            monkeypatch.delenv(var, raising=False)


if __name__ == '__main__':
    # Allow running tests directly
    pytest.main([__file__, '-v'])
