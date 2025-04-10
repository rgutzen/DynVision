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
from collections import defaultdict
from itertools import product
import json
import os
import subprocess
import re
from typing import Dict, List, Optional, Union, Any

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
pylogger = logging.getLogger('workflow.utils')

# Add the parent directory to the system path
package_dir = Path(inspect.getfile(lambda: None)).parents[2].resolve()
sys.path.insert(0, str(package_dir))

from dynvision.project_paths import project_paths

# Load configuration files in priority order
configfile: project_paths.scripts.configs / 'config_defaults.yaml'
configfile: project_paths.scripts.configs / 'config_data.yaml'
# configfile: project_paths.scripts.configs / 'config_visualization.yaml'
configfile: project_paths.scripts.configs / 'config_workflow.yaml'
configfile: project_paths.scripts.configs / 'config_experiments.yaml'

# Convert to SimpleNamespace for dot notation access
config = SimpleNamespace(**config)

# Save runtime configuration
runtime_config = project_paths.scripts.configs / 'config_runtime.yaml'
with open(runtime_config, 'w') as f:
    f.write("# This is an automatically compiled file. Do not edit manually!\n")
    json.dump(config.__dict__, f, indent=4)

# Print configuration summary
pylogger.info("Loaded configurations:")
for key, value in config.__dict__.items():
    pylogger.info(f"\t{key}: {value}")

CONFIGS = project_paths.scripts.configs / 'config_runtime.yaml'
SCRIPTS = project_paths.scripts_path

wildcard_constraints:
    model_name = r'[a-zA-Z0-9]+',
    data_name = r'[a-z0-9]+',
    data_subset = r'[a-z]+',
    data_group = r'[a-z0-9]+',
    data_loader = r'[a-zA-Z]+',
    status = r'[a-z]+',
    seed = r'\d+',
    condition = r'(withwave_|nowave_|\s?)',
    category = r'(?!folder)[a-z0-9]+',
    model_args = r'(:[a-z,;:\+=\d\.\*]+|\s?)',
    data_args = r'(:[a-zTF,;:\+=\d\.]+|\s?)',
    args = r'([a-z,;:\+=\d\.]+|\s?)',
    args1 = r'([a-z,;:\+=\d\.]+[,;:\+]|\s?)',
    args2 = r'([a-z,;:\+=\d\.]+|\s?)',
    parameter = r'(contrast|duration|interval)',
    experiment = r'[a-z]+',
    layer_name = r'(layer1|layer2|V1|V2|V4|IT)'

localrules: all, symlink_data_subsets, symlink_data_groups, experiment, plot_adaption, plot_experiments_on_models, all_experiments
ruleorder: symlink_data_groups > symlink_data_subsets

def get_imagenet_classes(tiny: bool = False) -> tuple[list, dict]:
    """Get ImageNet class information.

    Args:
        tiny: Whether to use TinyImageNet classes

    Returns:
        Tuple containing:
            - List of class names
            - Dictionary mapping indices to class information
    """
    index_file = "tinyimagenet_class_index" if tiny else "imagenet_class_index"
    try:
        with open(project_paths.references / f"{index_file}.json") as f:
            class_dict = json.load(f)
            imagenet_classes = [v[0] for k, v in class_dict.items()]
        return imagenet_classes
    except FileNotFoundError:
        raise ValueError(f"Class index file not found: {index_file}")

def get_category(data_name: str, data_group: str) -> List[str]:
    """Get category information for a dataset.

    Args:
        data_name: Name of the dataset
        data_group: Group within the dataset

    Returns:
        List of category names

    Raises:
        ValueError: If data_name is unknown
    """
    category_handlers = {
        'imagenet': lambda: get_imagenet_classes(tiny=('tiny' in data_name)),
        'cifar10': lambda: [str(i) for i in range(10)],
        'cifar100': lambda: [str(i) for i in range(100)],
        'snakenet': lambda: [
            'n01729322', 'n01740131', 'n01744401',
            'n01753488', 'n01755581', 'n01756291'
        ],
        'imagenette': lambda: ['n01440764',  'n02979186',  'n03028079',  'n03417042',  'n03445777', 'n02102040',  'n03000684',  'n03394916',  'n03425413',  'n03888257'],
        'mnist': lambda: [str(i) for i in range(10)],
    }

    if data_name not in category_handlers:
        raise ValueError(f"Unknown dataset: {data_name}")

    if data_group == 'all':
        return category_handlers[data_name]()
    
    try:
        group_indices = config.data_groups[data_name][data_group]
        if 'imagenet' in data_name:
            return [category_handlers['imagenet']()[i] for i in group_indices]
        else:
            return group_indices
    except KeyError:
        raise ValueError(f"Unknown data group '{data_group}' for {data_name}")

def get_data_location(wildcards: Any) -> Path:
    """Get the data directory for given wildcards.

    Args:
        wildcards: Snakemake wildcards object

    Returns:
        Path to data directory
    """
    data_name = wildcards.data_name
    data_subset = 'val' if data_name == 'imagenet' and wildcards.data_subset == 'test' else wildcards.data_subset
    category = wildcards.category

    base_dir = get_data_base_dir(wildcards)
    return base_dir / data_subset / category

def get_data_base_dir(wildcards: Any) -> Path:
    """Get the base directory for a dataset.

    Args:
        wildcards: Snakemake wildcards object

    Returns:
        Path to base directory
    """
    data_name = wildcards.data_name
    if data_name in config.mounted_datasets and project_paths.iam_on_cluster():
        return Path(f'/{data_name}')
    return project_paths.data.raw / data_name

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

# Initialize environment information
env_name, env_status = get_conda_env()
pylogger.info(f"Conda environment: {env_name or 'None'}")

rule checkpoint_to_statedict:
    """Convert Lightning checkpoints to state dictionaries."""
    input:
        checkpoint_dir = project_paths.large_logs / 'checkpoints',
        script = project_paths.scripts.utils / 'checkpoint_to_statedict.py'
    output:
        temp(project_paths.models / '{model_identifier}.ckpt2pt')
    # log:
    #     project_paths.logs / "checkpoint_to_statedict_{model_identifier}.log"
    shell:
        """
        {config.executor_start}
        python {input.script:q} \
            --checkpoint_dir {input.checkpoint_dir:q} \
            --output {output:q} \
            > {log} 2>&1
        {config.executor_close}
        """