"""Utility functions and configuration management for the workflow.

This module provides common utility functions and configuration management for
the workflow, including:
- Path management and system configuration
- Argument parsing and string manipulation
- Dataset class management
- Environment detection and setup

The utilities in this module are used by other workflow components to ensure
consistent behavior and reduce code duplication.
"""

import sys
import inspect
import logging
from pathlib import Path
from types import SimpleNamespace
from itertools import product
import os
import subprocess
import re
from typing import Dict, List, Optional, Union, Any
from datetime import datetime

import yaml

# Add the parent directory to the system path
package_dir = Path(inspect.getfile(lambda: None)).parents[2].resolve()
sys.path.insert(0, str(package_dir))

from dynvision.project_paths import project_paths

pylogger = logging.getLogger('workflow.utils')

def is_cluster_execution() -> bool:
    """
    Detect if running on a compute cluster via scheduler environment variables.

    Checks for presence of environment variables set by common HPC job schedulers.
    This enables automatic adaptation of execution commands without config changes.

    Supported Schedulers:
        - SLURM (SLURM_JOB_ID, SLURM_JOBID)
        - PBS/Torque (PBS_JOBID)
        - LSF (LSB_JOBID)
        - SGE/UGE (JOB_ID when SGE_TASK_ID also present)

    Returns:
        True if running in a cluster job, False if running locally

    Examples:
        >>> # On cluster node within SLURM job
        >>> is_cluster_execution()
        True

        >>> # On local workstation
        >>> is_cluster_execution()
        False

    Notes:
        - Detection happens at Snakemake parse time (when rule params are evaluated)
        - For cluster execution, params are evaluated on the submit node
        - Result is consistent for all jobs in a workflow run
        - Logging helps debug unexpected detection results

    See:
        docs/development/planning/cluster-execution.md for design rationale
    """
    # Environment variables indicating cluster execution
    # Listed in order of prevalence (most common first)
    cluster_indicators = [
        'SLURM_JOB_ID',    # SLURM (most common in academic HPC)
        'SLURM_JOBID',     # SLURM alternative spelling
        'PBS_JOBID',       # PBS/Torque
        'LSB_JOBID',       # IBM LSF
        'SGE_TASK_ID',     # SGE/UGE (combined with JOB_ID check)
    ]

    # Check each indicator
    detected = any(var in os.environ for var in cluster_indicators)

    # SGE special case: JOB_ID is too generic, require SGE_TASK_ID as well
    if not detected and 'JOB_ID' in os.environ and 'SGE_TASK_ID' in os.environ:
        detected = True

    # Log detection for debugging
    if detected:
        detected_vars = [var for var in cluster_indicators if var in os.environ]
        pylogger.info(f"Cluster execution detected via: {', '.join(detected_vars)}")
    else:
        pylogger.debug("Local execution detected (no cluster scheduler variables)")

    return detected

# Load and freeze configuration files to prevent mid-workflow changes
# See: docs/development/planning/snakecharm-config-stability-issue.md
def _load_and_freeze_config(cli_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Load configuration files once and freeze them for the entire workflow.

    This prevents mid-workflow config changes from affecting running jobs
    when using cluster execution with snakemake-executor-plugin.

    In cluster mode, Snakemake re-parses the workflow for each job submission.
    Using Snakemake's configfile: directive would cause configs to be re-read
    from disk each time, allowing changed files to affect running workflows.

    Instead, we load the YAML files manually once, freeze the merged config
    in memory, and use that frozen version throughout the workflow run.

    Note on cluster execution:
        Cluster detection (singularity/conda wrapper) is handled via
        environment variables (see is_cluster_execution()), NOT config.
        This ensures detection works correctly with frozen config and
        separates execution context from experiment configuration.

    Args:
        cli_config: Optional dictionary of CLI config overrides from Snakemake --config

    Returns:
        Merged configuration dictionary frozen at workflow start

    See:
        - docs/development/planning/snakecharm-config-stability-issue.md
        - docs/development/planning/cluster-execution.md
    """
    config_files = [
        'config_defaults.yaml',
        'config_data.yaml',
        'config_visualization.yaml',
        'config_experiments.yaml',
        'config_modes.yaml',
        'config_workflow.yaml',
    ]

    merged_config = {}
    configs_dir = project_paths.scripts.configs

    for config_file in config_files:
        config_path = configs_dir / config_file
        if config_path.exists():
            pylogger.debug(f"Loading config: {config_path}")
            with config_path.open('r', encoding='utf-8') as f:
                file_config = yaml.safe_load(f) or {}
                merged_config.update(file_config)
        else:
            pylogger.warning(f"Config file not found: {config_path}")

    # Merge with any --config overrides from Snakemake CLI
    if cli_config:
        pylogger.info(f"Applying {len(cli_config)} CLI config overrides: {list(cli_config.keys())}")
        merged_config.update(cli_config)
    else:
        pylogger.debug("No CLI config overrides detected")

    pylogger.info(f"Config frozen at workflow start with {len(merged_config)} keys")
    return merged_config

# Load config ONCE and freeze it for the entire workflow
# Pass the Snakemake config dict which contains CLI overrides from --config
# Snakemake injects 'config' dict into global scope before parsing workflow files
try:
    _frozen_config = _load_and_freeze_config(cli_config=config)
except NameError:
    # If config doesn't exist (e.g., when testing modules in isolation)
    pylogger.warning("Snakemake config not found - CLI overrides will not be applied")
    _frozen_config = _load_and_freeze_config(cli_config=None)

wildcard_constraints:
    model_name = r'[a-zA-Z0-9]+',
    data_name = r'[a-z0-9]+',
    data_subset = r'[a-z]+',
    data_group = r'[a-z0-9]+',
    data_group_not_all = r'(?!all$)[a-z0-9]+',
    data_loader = r'[a-zA-Z]+',
    status = r'(init|trained|trained-[a-z0-9\. =]+)',
    seed = r'\d+',
    seeds = r'[\d\.]+',
    category_str = r'([a-z0-9]+=\*|\s?)',
    model_args = r'(:[a-z,;:\+=\d\.\*]+|\s?)',
    data_args = r'(:[a-zTF,;:\+=\d\.]+|\s?)',
    args = r'([a-z,;:\+=\d\.]+|\s?)',
    args1 = r'([a-z,;:\+=\d\.]+[,;:\+]|\s?)',
    args2 = r'([a-z,;:\+=\d\.]+|\s?)',
    experiment = r'[a-z]+',
    layer_name = r'(layer1|layer2|V1|V2|V4|IT)',
    model_identifier = r'([\w:+=,\*\#]+)'

localrules: all, symlink_data_subsets, symlink_data_groups
ruleorder: symlink_data_subsets > symlink_data_groups > train_model_distributed > train_model > process_test_data > test_model

def _write_base_config_file(config_payload: Dict[str, Any]) -> Path:
    """Persist the fully merged Snakemake config for reuse by runtime scripts.

    The config written here is FROZEN at workflow start. Changes to source
    config files will NOT affect this workflow run. To use updated configs,
    start a new workflow run.
    """

    config_dir = project_paths.large_logs / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base_config_path = config_dir / f"workflow_config_{timestamp}.yaml"

    header = [
        "# DynVision workflow base configuration",
        f"# Generated at: {timestamp}",
        "#",
        "# WARNING: This config is FROZEN for the duration of this workflow run.",
        "# Changes to source config files will NOT affect this workflow.",
        "# To use updated configs, start a new workflow run.",
    ]

    with base_config_path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(header) + "\n\n")
        yaml.safe_dump(config_payload, handle, default_flow_style=False, sort_keys=False)

    pylogger.info(f"Persisted FROZEN workflow config to {base_config_path}")
    return base_config_path

# Set up logging
def setup_logger():
    # create logger
    logger = logging.getLogger('project')
    logger.setLevel(getattr(logging, config.log_level.upper(), logging.INFO))

    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logger.level)

    # create formatter
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')

    # add formatter to ch
    ch.setFormatter(formatter)

    # add ch to logger
    logger.addHandler(ch)
    
def get_param(key, default=None) -> callable:
    """Get a parameter value from the config.

    Args:
        key: Parameter key to retrieve

    Returns:
        Value of the parameter or None if not found
    """
    return lambda w: getattr(config, key, default)


def parse_arguments(
    wildcards: Any,
    args_key: str = 'model_args',
    delimiter: str = '+',
    assigner: str = '=',
    prefix: str = ":"
) -> str:
    """Parse argument string into command line arguments.

    Args:
        wildcards: Snakemake wildcards object
        args_key: Key for arguments in wildcards
        delimiter: Character separating arguments
        assigner: Character separating key and value
        prefix: Prefix character for argument string

    Returns:
        Formatted command line arguments string
    """
    args = getattr(wildcards, args_key, '')
    args = args.lstrip(prefix).split(delimiter)

    if len(args) == 1 and not args[0]:
        return ""

    args_dict = {
        arg.split(assigner)[0]: arg.split(assigner)[1]
        for arg in args
    }

    return " ".join(f"--{key} {value}" for key, value in args_dict.items())

def args_product(
    args_dict: Optional[Dict] = None,
    delimiter: str = '+',
    assigner: str = '=',
    prefix: str = ':'
) -> List[str]:
    """Generate product of argument combinations.

    Args:
        args_dict: Dictionary of argument options
        delimiter: Character separating arguments
        assigner: Character separating key and value
        prefix: Prefix character for argument string

    Returns:
        List of argument combination strings
    """
    if not args_dict:
        return ['']

    # Convert single values to lists
    args_dict = {
        key: [value] if not isinstance(value, list) else value
        for key, value in args_dict.items()
    }

    # Generate combinations
    args_combinations = product(*args_dict.values())
    return [
        prefix + delimiter.join(
            f'{key}{assigner}{value}'
            for key, value in zip(args_dict.keys(), combo)
        )
        for combo in args_combinations
    ]

def replace_param_in_string(
    s: str,
    key: str = "contrast",
    value_type: Optional[type] = None,
    new_value: str = "*"
) -> str:
    """Replace parameter value in string with new value.

    Args:
        s: Input string
        key: Parameter key to replace
        value_type: Type of value to match
        new_value: New value to insert

    Returns:
        Modified string

    Raises:
        ValueError: If parameter not found or invalid value type
    """
    patterns = {
        int: rf"{key}=(\d+)",
        float: rf"{key}=(\d+(\.\d+)?)",
        str: rf"{key}=([a-z]+)",
        None: rf"{key}=([\da-z\.]+)"
    }

    if value_type not in patterns:
        raise ValueError(f"Invalid value type: {value_type}")

    pattern = patterns[value_type]
    match = re.search(pattern, s)
    
    if not match:
        raise ValueError(f"No {key} value found in string: {s}")
        
    return s.replace(match.group(0), f"{key}={new_value}")

def dict_poped(d: Dict, keys: Union[str, List[str]]) -> Dict:
    """Create new dictionary with specified keys removed.

    Args:
        d: Input dictionary
        keys: Key or list of keys to remove

    Returns:
        New dictionary without specified keys
    """
    dc = d.copy()
    if isinstance(keys, list):
        for key in keys:
            dc.pop(key, None)
    else:
        dc.pop(keys, None)
    return dc

def compute_hash(*args, length: int = 8) -> str:
    """Compute deterministic hash from model_args and seed.

    Creates a short hash representation to avoid filesystem length limits.
    Idempotent - returns input unchanged if already a hash.

    Args:
        *args: Components to hash (model_args, seed - NOT data_name)
        length: Hash length in hex characters (default: 8 = 32 bits)

    Returns:
        Hash-prefixed string (e.g., ':hash=a7f3c9d4')

    Examples:
        >>> compute_hash('tsteps=20+dt=2+...', '42')
        ':hash=a7f3c9d4'

        >>> compute_hash(':hash=a7f3c9d4')  # Idempotent
        ':hash=a7f3c9d4'

    Notes:
        - Uses MD5 for speed (cryptographic strength not required)
        - Deterministic (same inputs → same output)
        - 8 hex chars = ~4 billion combinations
        - Collision probability negligible for typical use (<1000 models)
        - Hash does NOT include data_name (training data in subfolder)
    """
    import hashlib

    # Idempotent: if any arg already contains 'hash=', return first such arg
    for arg in args:
        if 'hash=' in str(arg):
            return str(arg)

    # Combine all arguments
    combined = '_'.join(str(arg).lstrip(':') for arg in args)

    # Compute MD5 hash
    hash_obj = hashlib.md5(combined.encode())
    hash_val = hash_obj.hexdigest()[:length]

    return f':hash={hash_val}'

def get_conda_env() -> tuple[Optional[str], str]:
    """Get information about the current conda environment.

    Returns:
        Tuple containing:
            - Environment name (or None if not in conda)
            - Environment status string
    """
    try:
        env_name = os.environ['CONDA_DEFAULT_ENV']
        env_status = subprocess.check_output(
            ['conda', 'info', '--envs']
        ).decode('utf-8')
        return env_name, env_status
    except (KeyError, subprocess.CalledProcessError):
        return None, "No conda environment active"



def build_execution_command(script_path, use_distributed=False):
    """
    Build the execution command with conditional wrappers.

    Automatically detects cluster execution and wraps commands with
    executor_wrapper.sh (singularity + conda) when running on cluster nodes.

    Args:
        script_path: Path to the Python script to execute
        use_distributed: Whether to use distributed/multi-node setup

    Returns:
        String containing the complete execution command

    Execution Modes:
        Local:
            python script.py

        Cluster (single-node):
            executor_wrapper.sh python script.py

        Cluster (distributed):
            source setup_distributed_execution.sh &&
            executor_wrapper.sh torchrun --nproc_per_node=... script.py

    Environment Detection:
        Uses is_cluster_execution() to automatically detect cluster jobs.
        No configuration needed - detection is based on scheduler env vars.

    See:
        docs/development/planning/cluster-execution.md for design details
    """
    cmd_parts = []

    # Automatic cluster detection (replaces use_executor config)
    use_executor = is_cluster_execution()

    if use_distributed:
        setup_script = SCRIPTS / 'cluster' / 'setup_distributed_execution.sh'
        cmd_parts.append(f"source {setup_script} &&")

    # Add executor wrapper if on cluster
    if use_executor:
        executor_script = SCRIPTS / 'cluster' / 'executor_wrapper.sh'
        cmd_parts.append(str(executor_script))

    if use_distributed:
        cmd_parts.append(
            "torchrun "
            "--nproc_per_node=${GPU_PER_NODE:-2} "
            "--nnodes=${NUM_NODES:-1} "
            "--node_rank=${NODE_RANK:-0} "
            "--master_addr=${MASTER_ADDR} "
            "--master_port=${MASTER_PORT} "
            f"{script_path}"
        )
    else:
        cmd_parts.append(f"python {script_path}")
    
    return "\\\n        ".join(cmd_parts)

# Initialize environment information
env_name, env_status = get_conda_env()
pylogger.info(f"Conda environment: {env_name or 'None'}")

# Use frozen config for all downstream processing
# This ensures that changes to config files on disk do not affect the running workflow
_raw_config = _frozen_config.copy()

# Write snapshot to disk for runtime scripts
WORKFLOW_CONFIG_PATH = _write_base_config_file(_raw_config)

# Convert to SimpleNamespace for dot notation access in rules
config = SimpleNamespace(**_raw_config)

# Log the snapshot location for debugging
pylogger.info(f"Workflow config snapshot: {WORKFLOW_CONFIG_PATH}")

SCRIPTS = project_paths.scripts_path

setup_logger()
