"""Data preparation and processing workflow.

This workflow handles all data-related operations including:
- Dataset creation and preprocessing
- Data organization and linking
- FFCV dataset building
- Data validation and verification

The workflow supports multiple dataset types:
- Standard datasets (MNIST, CIFAR10, CIFAR100, TinyImageNet)
- External datasets (ImageNet)
"""

import json
import logging
import shutil
from pathlib import Path
from types import SimpleNamespace

logger = logging.getLogger('workflow.data')

_CLASS_INDEX_CACHE = {}


def _to_dict(entry):
    if isinstance(entry, SimpleNamespace):
        return entry.__dict__
    return entry


def _load_class_index(data_name: str):
    mapping_files = _to_dict(getattr(config, 'class_index_files', {})) or {}
    mapping_path = mapping_files.get(data_name)
    if mapping_path is None:
        return {}
    path = Path(mapping_path)
    if not path.is_absolute():
        path = project_paths.references / mapping_path
    if not path.exists():
        raise FileNotFoundError(f"Class index file not found for {data_name}: {path}")
    with open(path, 'r', encoding='utf-8') as handle:
        raw_mapping = json.load(handle)

    mapping = {}
    if isinstance(raw_mapping, dict):
        items = raw_mapping.items()
    else:
        items = enumerate(raw_mapping)

    for key, value in items:
        if isinstance(value, list) and value:
            mapping[str(key)] = str(value[0])
        elif isinstance(value, dict):
            preferred = value.get('id') or value.get('name') or next(iter(value.values()), None)
            if preferred is None:
                raise ValueError(f"Unable to parse mapping entry for {data_name}: {value}")
            mapping[str(key)] = str(preferred)
        else:
            mapping[str(key)] = str(value)
    return mapping


def _resolve_class_name(data_name: str, entry):
    mapping = _CLASS_INDEX_CACHE.get(data_name)
    if mapping is None:
        mapping = _load_class_index(data_name)
        _CLASS_INDEX_CACHE[data_name] = mapping

    if mapping:
        key = str(entry)
        if key not in mapping:
            raise ValueError(f"Class index '{entry}' not defined for {data_name}")
        return mapping[key]
    return str(entry)


def _safe_symlink(link_path: Path, target_path: Path) -> None:
    if link_path.is_symlink() or link_path.exists():
        try:
            if link_path.resolve() == target_path:
                return
        except FileNotFoundError:
            pass
        if link_path.is_dir() and not link_path.is_symlink():
            shutil.rmtree(link_path)
        else:
            link_path.unlink()
    link_path.symlink_to(target_path)


def _raw_subset_path(data_name: str, data_subset: str) -> Path:
    mounted = getattr(config, 'mounted_datasets', []) or []
    subset_key = 'val' if data_name == 'imagenet' and data_subset == 'test' else data_subset
    if data_name in mounted and project_paths.iam_on_cluster():
        return Path(f'/{data_name}') / subset_key
    return project_paths.data.raw / data_name / subset_key


rule get_data:
    """Download and prepare standard datasets.
    
    This rule handles the download and initial preparation of
    standard datasets like MNIST, CIFAR10, and CIFAR100.

    Input:
        script: Python script for dataset preparation
    
    Output:
        Directory containing prepared dataset

    Parameters:
        data_name: Name of the dataset
        data_subset: Dataset split (train/test)
        ext: Image file extension
    """
    input:
        script = SCRIPTS / 'data' / 'get_data.py'
    params:
        raw_data_path = lambda w: project_paths.data.raw / f'{w.data_name}',
        output = lambda w: project_paths.data.raw \
            / f'{w.data_name}' \
            / f'{w.data_subset}',
        ext = 'png',
        data_name = lambda w: ''.join([c.upper() if c.isalpha() else c for c in w.data_name]),
        execution_cmd = lambda w, input: build_execution_command(
            script_path=input.script,
            use_distributed=False,
        ),
    output:
        flag = directory(project_paths.data.raw \
            / '{data_name}' \
            / '{data_subset}' )
    wildcard_constraints:
        data_name = r"cifar10|cifar100|mnist"
    benchmark:
        project_paths.benchmarks / 'get_data_{data_name}_{data_subset}.txt'
    shell:
        """
        {params.execution_cmd} \
            --output {params.output:q} \
            --data_name {params.data_name} \
            --raw_data_path {params.raw_data_path:q} \
            --subset {wildcards.data_subset} \
            --ext {params.ext}
        """

rule symlink_data_subsets:
    """Materialize canonical subset directories with direct symlinks to raw data."""
    input:
        subset_folder = lambda w: _raw_subset_path(w.data_name, w.data_subset)
    output:
        subset_dir = directory(project_paths.data.interim / '{data_name}' / '{data_subset}_all'),
        ready = project_paths.data.interim / '{data_name}' / '{data_subset}_all.ready'
    group: "dataprep"
    run:
        raw_subset = Path(input.subset_folder)
        subset_dir = Path(output.subset_dir)
        ready_file = Path(output.ready)

        subset_dir.mkdir(parents=True, exist_ok=True)
        ready_file.parent.mkdir(parents=True, exist_ok=True)

        available = []
        for entry in sorted(raw_subset.iterdir()):
            if not entry.is_dir():
                continue
            available.append(entry.name)
            _safe_symlink(subset_dir / entry.name, entry.resolve())

        keep = set(available)
        for entry in subset_dir.iterdir():
            if entry.name not in keep and entry.is_symlink():
                entry.unlink()

        ready_file.write_text('ready\n', encoding='utf-8')


rule symlink_data_groups:
    """Create group directories by selecting classes from canonical subsets."""
    input:
        subset_dir = project_paths.data.interim / '{data_name}' / '{data_subset}_all',
        subset_ready = project_paths.data.interim / '{data_name}' / '{data_subset}_all.ready'
    output:
        group_dir = directory(project_paths.data.interim / '{data_name}' / '{data_subset}_{data_group_not_all}'),
        ready = project_paths.data.interim / '{data_name}' / '{data_subset}_{data_group_not_all}.ready'
    group: "dataprep"
    run:
        subset_dir = Path(input.subset_dir)
        group_dir = Path(output.group_dir)
        ready_file = Path(output.ready)

        group_dir.mkdir(parents=True, exist_ok=True)
        ready_file.parent.mkdir(parents=True, exist_ok=True)

        defined_groups = _to_dict(config.data_groups)
        group_entries = None
        if defined_groups and wildcards.data_name in defined_groups:
            group_entries = defined_groups[wildcards.data_name].get(wildcards.data_group_not_all)

        if group_entries:
            class_names = []
            for entry in group_entries:
                class_name = _resolve_class_name(wildcards.data_name, entry)
                class_names.append(class_name)
            # Preserve order but drop duplicates
            seen = set()
            class_names = [x for x in class_names if not (x in seen or seen.add(x))]
        else:
            class_names = sorted(entry.name for entry in subset_dir.iterdir() if entry.is_symlink())

        subset_targets = {}
        for entry in subset_dir.iterdir():
            if entry.is_symlink():
                subset_targets[entry.name] = entry.resolve()

        missing = [cls for cls in class_names if cls not in subset_targets]
        if missing:
            raise ValueError(
                f"Classes {missing} not available for {wildcards.data_name}/{wildcards.data_subset}."
            )

        for class_name in class_names:
            _safe_symlink(group_dir / class_name, subset_targets[class_name])

        keep = set(class_names)
        for entry in group_dir.iterdir():
            if entry.is_symlink() and entry.name not in keep:
                entry.unlink()

        ready_file.write_text('ready\n', encoding='utf-8')

rule build_ffcv_datasets:
    """Build FFCV datasets for faster data loading.

    This rule converts prepared datasets into FFCV format
    for improved training performance.

    Input:
        script: Python script for FFCV conversion
        data: Prepared dataset directory
    
    Output:
        train: Training dataset in FFCV format
        val: Validation dataset in FFCV format

    Parameters:
        train_ratio: Split ratio for training data
        max_resolution: Maximum image resolution
    """
    input: 
        script = SCRIPTS / 'data' / 'ffcv_datasets.py',
        data_ready = project_paths.data.interim / '{data_name}' / 'train_all.ready'
    params:
        train_ratio = get_param('train_ratio'),
        max_resolution = lambda w: config.data_resolution[w.data_name],
        base_config_path = WORKFLOW_CONFIG_PATH,
        data_dir = lambda w: project_paths.data.interim / w.data_name / 'train_all',
        execution_cmd = lambda w, input: build_execution_command(
            script_path=input.script,
            use_distributed=False,
        ),
        seed = get_param("seed")[0] if isinstance(get_param("seed"), list) else get_param("seed"),
    output:
        train = project_paths.data.processed / '{data_name}' / 'train_all' / 'train.beton',
        val = project_paths.data.processed / '{data_name}' / 'train_all' / 'val.beton'
    benchmark:
        project_paths.benchmarks / 'build_ffcv_datasets_{data_name}.txt'
    shell:
        """
        {params.execution_cmd} \
            --config_path {params.base_config_path:q} \
            --input {params.data_dir:q} \
            --output_train {output.train:q} \
            --output_val {output.val:q} \
            --train_ratio {params.train_ratio} \
            --data_name {wildcards.data_name} \
            --max_resolution {params.max_resolution} \
            --seed {params.seed}
        """

# Log workflow initialization
logger.info("Data workflow initialized")
logger.info(f"Data directory: {project_paths.data}")