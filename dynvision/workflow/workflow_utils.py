"""
Workflow utility functions for Snakemake parsing.

This module contains pure utility functions with minimal dependencies
that are used during Snakemake workflow parsing (DAG building time).

IMPORTANT: This module is imported during Snakemake parsing, which happens
in a minimal environment (e.g., on cluster login nodes). Therefore:
- Only use standard library imports (+ yaml, which is a Snakemake dependency)
- Do NOT import: torch, lightning, ffcv, or any DynVision modules
- Keep functions pure and testable

Functions in this module are used for:
- Wildcard expansion (config-based and filesystem-based)
- Argument parsing and string manipulation
- Path pattern processing
"""

import re
import glob
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from itertools import product

# Use module-level logger
logger = logging.getLogger('workflow.utils')


def expand_filesystem_pattern(
    pattern: str,
    wildcard_values: Dict[str, Union[str, List[str]]],
    base_path: Optional[Path] = None
) -> Dict[str, List[str]]:
    """
    Expand filesystem wildcards (?) by globbing existing files.

    Scans filesystem for files matching the pattern and extracts parameter
    values from actual filenames. Supports mixed wildcards:
    - '?' : Match from filesystem (e.g., 'seed=5?' finds seed=5000, seed=5001)
    - Concrete values: Used as-is
    - Lists: Used as-is

    Args:
        pattern: Path pattern with {wildcards} and parameters with ?
                 Example: 'models/{model_name}/{model_name}:{model_args}_{seed}_{data_name}_trained.pt'
        wildcard_values: Dict mapping wildcard names to values or patterns
                        Example: {'model_name': 'DyRCNNx8', 'seed': '5?', 'model_args': ':tau=5'}
        base_path: Optional base path to resolve relative patterns

    Returns:
        Dict with '?' wildcards replaced by lists of found values
        Example: {'model_name': ['DyRCNNx8'], 'seed': ['5000', '5001', '5023'], ...}

    Raises:
        ValueError: If no files match the pattern

    Example:
        >>> expand_filesystem_pattern(
        ...     'models/{model_name}/*_{seed}_*_trained.pt',
        ...     {'model_name': 'DyRCNNx8', 'seed': '5?'}
        ... )
        {'model_name': ['DyRCNNx8'], 'seed': ['5000', '5001', '5023']}

    Notes:
        - The '?' character matches any alphanumeric characters including dots
        - Matching stops at delimiters: '_', '+', ':', '='
        - Values are extracted, deduplicated, and sorted
    """
    # Identify which wildcards have ? patterns
    fs_wildcards = {k: v for k, v in wildcard_values.items()
                    if isinstance(v, str) and '?' in v}

    if not fs_wildcards:
        # No filesystem wildcards, return input as lists
        return {k: [v] if not isinstance(v, list) else v
                for k, v in wildcard_values.items()}

    # Build glob pattern by substituting wildcards
    glob_pattern = str(pattern)

    # First, substitute concrete wildcards and convert ? to *
    for key, value in wildcard_values.items():
        if isinstance(value, str):
            if '?' in value:
                # Convert ? to * for globbing
                glob_value = value.replace('?', '*')
            else:
                glob_value = value
            glob_pattern = glob_pattern.replace(f'{{{key}}}', glob_value)
        elif isinstance(value, list):
            # For lists, use first value for globbing
            glob_pattern = glob_pattern.replace(f'{{{key}}}', str(value[0]))

    # Resolve base path
    if base_path:
        glob_pattern = str(base_path / glob_pattern)

    logger.debug(f"Filesystem glob pattern: {glob_pattern}")

    # Find matching files
    matching_files = glob.glob(glob_pattern)

    if not matching_files:
        raise ValueError(
            f"No files found matching pattern: {glob_pattern}\n"
            f"Filesystem wildcards: {fs_wildcards}\n"
            f"Original pattern: {pattern}"
        )

    logger.info(f"Found {len(matching_files)} files matching pattern")

    # Extract values for each ? wildcard
    result = {}

    for key, pattern_value in wildcard_values.items():
        if isinstance(pattern_value, str) and '?' in pattern_value:
            # Extract values from filenames
            # Build regex to match parameter value
            # Pattern like '5?' becomes regex to match '5' followed by word chars/dots
            prefix = pattern_value.split('?')[0]
            # Match: prefix followed by word chars and dots, until delimiter
            value_regex = re.compile(f'{re.escape(prefix)}([\\w\\.]+)')

            found_values = set()
            for filepath in matching_files:
                filename = Path(filepath).name
                matches = value_regex.finditer(filename)
                for match in matches:
                    full_match = prefix + match.group(1)
                    # Validate it's a standalone value (not part of larger string)
                    # Check it's preceded/followed by delimiter or start/end
                    idx = filename.index(full_match)
                    before = filename[idx-1] if idx > 0 else '_'
                    after_idx = idx + len(full_match)
                    after = filename[after_idx] if after_idx < len(filename) else '_'

                    # Delimiters that indicate parameter boundaries
                    if before in '_+:=' and after in '_+:.':
                        found_values.add(full_match)

            values_list = sorted(list(found_values))

            if not values_list:
                logger.warning(
                    f"No values extracted for {key}={pattern_value} from {len(matching_files)} files"
                )

            result[key] = values_list
            logger.info(f"Filesystem expansion: {key}={pattern_value} -> {values_list}")
        else:
            # Not a filesystem wildcard, keep as-is
            result[key] = [pattern_value] if not isinstance(pattern_value, list) else pattern_value

    return result


def args_product(
    args_dict: Optional[Dict] = None,
    delimiter: str = '+',
    assigner: str = '=',
    prefix: str = ':',
    config_categories: Optional[Dict] = None,
    enable_fs_wildcards: bool = True,
    fs_pattern: Optional[str] = None,
    fs_base_path: Optional[Path] = None
) -> List[str]:
    """
    Generate product of argument combinations with optional wildcard expansion.

    Supports mixed wildcard types:
    - '*' : Expand from config_categories (if provided)
    - '?' : Expand from filesystem (if enable_fs_wildcards=True and fs_pattern provided)
    - Concrete values: Use as-is
    - Lists: Use as-is

    Args:
        args_dict: Dictionary of argument options
        delimiter: Character separating arguments (default '+')
        assigner: Character separating key and value (default '=')
        prefix: Prefix character for argument string (default ':')
        config_categories: Dict of config categories for '*' expansion
        enable_fs_wildcards: Enable filesystem wildcard expansion
        fs_pattern: Pattern template for filesystem search (e.g., 'models/{model_name}/*.pt')
        fs_base_path: Base path for filesystem search

    Returns:
        List of argument combination strings

    Example:
        >>> args_product(
        ...     {'tau': '*', 'seed': '5?', 'rctype': 'full'},
        ...     config_categories={'tau': [3, 5, 9]},
        ...     enable_fs_wildcards=True,
        ...     fs_pattern='models/DyRCNNx8/*.pt'
        ... )
        [':tau=3+seed=5000+rctype=full',
         ':tau=5+seed=5000+rctype=full',
         ':tau=3+seed=5001+rctype=full',
         ':tau=5+seed=5001+rctype=full']

    Notes:
        - Empty or None args_dict returns ['']
        - Filesystem expansion requires both fs_pattern and enable_fs_wildcards=True
        - Config expansion requires config_categories dict
    """
    if not args_dict:
        return ['']

    # Make a copy to avoid modifying input
    args_dict = dict(args_dict)

    # Expand filesystem wildcards if enabled and present
    if enable_fs_wildcards and any(isinstance(v, str) and '?' in v for v in args_dict.values()):
        if not fs_pattern:
            logger.warning(
                "Filesystem wildcards present but no fs_pattern provided. "
                "Wildcards will be treated as literals."
            )
        else:
            # Expand filesystem wildcards
            try:
                expanded_dict = expand_filesystem_pattern(
                    pattern=fs_pattern,
                    wildcard_values=args_dict,
                    base_path=fs_base_path
                )
                # Update args_dict with expanded values
                for key, values in expanded_dict.items():
                    if isinstance(args_dict.get(key), str) and '?' in args_dict[key]:
                        args_dict[key] = values
            except ValueError as e:
                logger.error(f"Filesystem wildcard expansion failed: {e}")
                raise

    # Expand config wildcards ('*')
    if config_categories:
        for key, value in list(args_dict.items()):
            if value == '*':
                if key in config_categories:
                    args_dict[key] = config_categories[key]
                    logger.debug(f"Config expansion: {key}=* -> {config_categories[key]}")
                else:
                    logger.warning(f"Config wildcard '{key}=*' not found in config_categories")

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


def parse_arguments(
    args_str: str,
    delimiter: str = '+',
    assigner: str = '=',
    prefix: str = ":"
) -> str:
    """
    Parse argument string into command line arguments.

    Args:
        args_str: Argument string (e.g., ':tau=5+rctype=full')
        delimiter: Character separating arguments (default '+')
        assigner: Character separating key and value (default '=')
        prefix: Prefix character for argument string (default ':')

    Returns:
        Formatted command line arguments string

    Example:
        >>> parse_arguments(':tau=5+rctype=full')
        '--tau 5 --rctype full'
    """
    args = args_str.lstrip(prefix).split(delimiter)

    if len(args) == 1 and not args[0]:
        return ""

    args_dict = {
        arg.split(assigner)[0]: arg.split(assigner)[1]
        for arg in args
        if assigner in arg
    }

    return " ".join(f"--{key} {value}" for key, value in args_dict.items())


def replace_param_in_string(
    s: str,
    key: str = "contrast",
    value_type: Optional[type] = None,
    new_value: str = "*"
) -> str:
    """
    Replace parameter value in string with new value.

    Args:
        s: Input string
        key: Parameter key to replace
        value_type: Type of value to match (int, float, str, or None for any)
        new_value: New value to insert

    Returns:
        Modified string

    Raises:
        ValueError: If parameter not found or invalid value type

    Example:
        >>> replace_param_in_string('tau=5+rctype=full', 'tau', int, '*')
        'tau=*+rctype=full'
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
    """
    Create new dictionary with specified keys removed.

    Args:
        d: Input dictionary
        keys: Key or list of keys to remove

    Returns:
        New dictionary without specified keys

    Example:
        >>> dict_poped({'a': 1, 'b': 2, 'c': 3}, ['a', 'c'])
        {'b': 2}
    """
    dc = d.copy()
    if isinstance(keys, list):
        for key in keys:
            dc.pop(key, None)
    else:
        dc.pop(keys, None)
    return dc
