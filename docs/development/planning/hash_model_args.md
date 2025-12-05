# Hierarchical File Organization and Model Identifier Hashing

**Status:** Planning
**Created:** 2025-01-06
**Authors:** Robin Gutzen, Claude (AI Assistant)

## Overview

This document describes a comprehensive reorganization of the DynVision file structure to:
1. **Solve filesystem limitations** through model identifier hashing
2. **Improve conceptual clarity** by separating model and test attributes hierarchically
3. **Enable scalability** for large parameter sweeps and experiments

The new paradigm introduces **hierarchical path organization** where model attributes and test/data attributes are clearly separated at the directory level, making the workflow more maintainable and avoiding filename length issues.

## Problem Statement

### Filesystem Length Limitations

The workflow encounters filesystem errors when model parameter strings become too long:

```
OSError: [Errno 36] File name too long:
'/home/.../logs/slurm/rule_test_model/StimulusNoise_DyRCNNx8_:tsteps=20+dt=2+tau=5+
tff=0+trc=6+tsk=0+lossrt=4+energyloss=0.1+pattern=1+rctype=full+rctarget=output+
skip=true+feedback=false+dloader=torch_6000_imagenette_trained-epoch=149_
:dsteps=20+noisetype=uniform+ssnr=0.6+tempmode=dynamic+idle=20_all'
```

**Root causes:**
1. Filesystem limits: 255 bytes per path component
2. Long parameter combinations: Model + seed + data + status exceeds limits
3. SLURM plugin uses wildcards for log paths

### Conceptual Issues with Flat Structure

The previous flat file organization mixed model and test attributes:
```
# Old structure (flat, mixed attributes)
models/DyRCNNx8/DyRCNNx8:tsteps=20+..._{seed}_{data}_trained.pt
                └─ model params, seed, data, status all in filename

reports/DyRCNNx8:tsteps=20+..._{seed}_{data}_trained_StimulusNoise:dsteps=20+...all_test_outputs.csv
        └─ model params, seed, data, status, data_loader, data params, group all in one filename
```

**Problems:**
1. Long filenames exceed 255-char filesystem limits
2. Conceptually unclear: model attributes mixed with test attributes
3. Difficult to navigate: all information in a single directory level
4. Hard to query: finding all tests for a model requires parsing filenames

## Proposed Solution

### New Paradigm: Hierarchical Separation of Concerns

The solution introduces a **hierarchical file organization** that separates model attributes from test/data attributes at the directory structure level.

#### Core Architectural Principles

1. **Hierarchical organization**: Separate model attributes from test attributes via directory structure
2. **Conceptual clarity**: Model properties in parent directories, test properties in subdirectories
3. **Filesystem compatibility**: Short filenames, long information in directory paths
4. **Polymorphic identifiers**: `{model_identifier}` matches full or hashed forms
5. **Forward-only hashing**: Deterministic hash function, no reverse lookups needed

#### Hierarchical Path Organization

**Models: `{model_identifier}/{status}.pt`**
```
models/
  DyRCNNx8:tsteps=20+dt=2+tau=5+...._42_imagenette/  ← Model identifier (full)
    ├── init.pt                                       ← Status: initialization
    ├── trained.pt                                    ← Status: trained
    └── a7f3c9d4.hash                                ← Hash documentation

  DyRCNNx8:hash=a7f3c9d4/                            ← Model identifier (hashed)
    ├── init.pt                                       ← (symlink)
    └── trained.pt                                    ← (symlink)
```

**Test Results: `{data_loader}/{model_identifier}_{status}/{data_loader}:{data_args}_{data_group}/<outputs>`**
```
reports/
  StimulusNoise/                                      ← Data loader (level 1)
    DyRCNNx8:hash=a7f3c9d4_trained/                  ← Model + status (level 2)
      StimulusNoise:dsteps=20+noisetype=uniform+...all/  ← Data params + group (level 3)
        ├── test_outputs.csv                         ← Test results
        └── test_responses.pt                        ← Layer responses
```

**Processed Experiment Data: `{experiment}/{model_identifier}_{status}/{data_group}_test_data.csv`**
```
reports/
  uniformnoise/                                       ← Experiment name (level 1)
    DyRCNNx8:hash=a7f3c9d4_trained/                  ← Model + status (level 2)
      ├── all_test_data.csv                          ← Processed for all samples
      └── subset_test_data.csv                       ← Processed for subset
```

**Visualization: `{experiment}/{model_identifier}_{status}/{data_group}_<plot>.png`**
```
figures/
  uniformnoise/                                       ← Experiment name (level 1)
    DyRCNNx8:hash=a7f3c9d4_trained/                  ← Model + status (level 2)
      ├── all_performance.png                        ← Plot for all samples
      └── all_responses.png                          ← Response plot
```

#### Separation of Concerns

The hierarchy cleanly separates:

| Level | Contains | Represents |
|-------|----------|------------|
| **Models** | | |
| 1. Model identifier | `model_name:model_args_seed_data` | What was trained |
| 2. Status file | `init.pt`, `trained.pt` | Training stage |
| **Test Results** | | |
| 1. Data loader | `StimulusNoise`, `torch` | How data was loaded |
| 2. Model + status | `model_identifier_status` | What model was tested |
| 3. Data config | `data_loader:data_args_data_group` | What data was used |
| 4. Output files | `test_outputs.csv` | Test results |
| **Processed Data** | | |
| 1. Experiment | `uniformnoise`, `rctarget` | What experiment |
| 2. Model + status | `model_identifier_status` | What model |
| 3. Data group | `{group}_test_data.csv` | What samples |
| **Visualizations** | | |
| 1. Experiment | `uniformnoise`, `rctarget` | What experiment |
| 2. Model + status | `model_identifier_status` | What model |
| 3. Plot files | `{group}_performance.png` | What visualization |

#### Benefits of Hierarchical Organization

1. **Conceptual clarity**:
   - Model attributes (args, seed, data) separated from test attributes (data_loader, data_args)
   - Natural grouping: "All tests for this model" or "All models in this experiment"

2. **Filesystem compatibility**:
   - Short filenames: `test_outputs.csv`, `all_test_data.csv`, `performance.png`
   - Long identifiers in directory paths (not limited to 255 chars)

3. **Navigability**:
   - Browse by experiment: `reports/uniformnoise/*/`
   - Browse by data loader: `reports/StimulusNoise/*/`
   - Browse by model: `models/DyRCNNx8:hash=*/`

4. **Queryability**:
   - Find all tests for a model: `reports/*/DyRCNNx8:hash=a7f3c9d4_*/`
   - Find all experiment results: `reports/uniformnoise/*/*.csv`

5. **Scalability**:
   - Adding new test configurations doesn't change model directories
   - Adding new models doesn't pollute test result directories

### Hash-Based Model Identifiers

To handle long parameter combinations, model identifiers can use hash form:

**Polymorphic `{model_identifier}` wildcard** matches EITHER:
- Full form: `:tsteps=20+dt=2+tau=5+...._42_imagenette`
- Hash form: `:hash=a7f3c9d4`

**Key principles:**
- Training creates folder symlink: `hash=XXXXXXXX/` → `tsteps=20+...._42_imagenette/`
- Downstream rules use `{model_identifier}` (accepts both forms)
- Checkpoint ensures symlinks exist before downstream execution
- Hash includes model_args + seed + data_name (unique per model)
- No registry needed (forward-only transformation)

### Directory Structure

```
models/
  DyRCNNx8:tsteps=20+dt=2+tau=5+...._42_imagenette/
    init.pt
    trained.pt
    a7f3c9d4.hash              # Documents hash value

  DyRCNNx8:hash=a7f3c9d4/      # Symlink → DyRCNNx8:tsteps=20+...._42_imagenette/
    init.pt                    # (accessible via symlink)
    trained.pt                 # (accessible via symlink)

reports/
  DyRCNNx8:hash=a7f3c9d4_trained/
    StimulusNoise:dsteps=20+.../
      test_outputs.csv
```

### Polymorphic Wildcard Pattern

The `{model_identifier}` wildcard in test/processing rules matches:

```python
# Pattern 1: Full identifier (during DAG construction, before checkpoint)
{model_identifier} = "tsteps=20+dt=2+tau=5+...._42_imagenette"

# Pattern 2: Hashed identifier (after checkpoint creates symlink)
{model_identifier} = "hash=a7f3c9d4"
```

Snakemake's pattern matching handles both forms transparently.

## Implementation Details

### Component 1: `compute_hash()` Utility Function

**Location:** `dynvision/workflow/snake_utils.smk`

```python
def compute_hash(*args, length: int = 8) -> str:
    """Compute deterministic hash from multiple arguments.

    Creates a short hash representation to avoid filesystem length limits.
    Idempotent - returns input unchanged if already a hash.

    Args:
        *args: Components to hash (model_args, seed, data_name, etc.)
        length: Hash length in hex characters (default: 8 = 32 bits)

    Returns:
        Hash-prefixed string (e.g., ':hash=a7f3c9d4')

    Examples:
        >>> compute_hash('tsteps=20+dt=2+...', '42', 'imagenette')
        ':hash=a7f3c9d4'

        >>> compute_hash(':hash=a7f3c9d4')  # Idempotent
        ':hash=a7f3c9d4'

    Notes:
        - Uses MD5 for speed (cryptographic strength not required)
        - Deterministic (same inputs → same output)
        - 8 hex chars = ~4 billion combinations
        - Collision probability negligible for typical use (<1000 models)
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
```

### Component 2: Model Rules (snake_runtime.smk)

#### `init_model` (unchanged structure)

```python
rule init_model:
    """Initialize model with specified configuration.

    Creates model folder with full identifier (not hashed).
    """
    input:
        script = SCRIPTS / 'runtime' / 'init_model.py',
        dataset_ready = project_paths.data.interim / '{data_name}' / 'train_all.ready'
    params:
        base_config_path = WORKFLOW_CONFIG_PATH,
        model_arguments = lambda w: parse_arguments(w, 'model_args'),
        dataset_path = lambda w: project_paths.data.interim / w.data_name / 'train_all',
        execution_cmd = lambda w, input: build_execution_command(
            script_path=input.script,
            use_distributed=False,
        ),
    output:
        model_state = project_paths.models \
            / '{model_name}{model_args}_{seed}_{data_name}' \
            / 'init.pt'
    shell:
        """
        {params.execution_cmd} \
            --config_path {params.base_config_path:q} \
            --model_name {wildcards.model_name} \
            --dataset_path {params.dataset_path:q} \
            --data_name {wildcards.data_name} \
            --seed {wildcards.seed} \
            --output {output.model_state:q} \
            {params.model_arguments}
        """
```

#### `train_model` (checkpoint, creates symlink)

```python
checkpoint train_model:
    """Train model and create hashed folder symlink.

    This checkpoint rule trains the model and creates a symlink with hashed
    identifier for filesystem compatibility. The checkpoint triggers DAG
    re-evaluation so downstream rules can find the symlink.

    Side effects:
        - Creates {hash_value}.hash file documenting the hash
        - Creates symlink: DyRCNNx8:hash=XXXXXXXX/ → DyRCNNx8:args_seed_data/
    """
    input:
        model_state = project_paths.models \
            / '{model_name}{model_args}_{seed}_{data_name}' \
            / 'init.pt',
        dataset_ready = project_paths.data.interim / '{data_name}' / 'train_all.ready',
        dataset_train = lambda w: project_paths.data.processed \
            / w.data_name / 'train_all' / 'train.beton' if config.use_ffcv else [],
        dataset_val = lambda w: project_paths.data.processed \
            / w.data_name / 'train_all' / 'val.beton' if config.use_ffcv else [],
        script = SCRIPTS / 'runtime' / 'train_model.py'
    params:
        base_config_path = WORKFLOW_CONFIG_PATH,
        data_group = "all",
        model_arguments = lambda w: parse_arguments(w, 'model_args'),
        dataset_link = lambda w: project_paths.data.interim / w.data_name / 'train_all',
        resolution = lambda w: config.data_resolution[w.data_name],
        normalize = lambda w: json.dumps((
            config.data_statistics[w.data_name]['mean'],
            config.data_statistics[w.data_name]['std']
        )),
        execution_cmd = lambda w, input: build_execution_command(
            script_path=input.script,
            use_distributed=get_param('use_distributed_mode', False)(w),
        ),
        # Symlink parameters
        model_folder = lambda w: project_paths.models \
            / f'{w.model_name}{w.model_args}_{w.seed}_{w.data_name}',
        symlink_folder = lambda w: project_paths.models \
            / f'{w.model_name}{compute_hash(w.model_args, w.seed, w.data_name)}',
        hash_file = lambda w: project_paths.models \
            / f'{w.model_name}{w.model_args}_{w.seed}_{w.data_name}' \
            / f'{compute_hash(w.model_args, w.seed, w.data_name).lstrip(":")}.hash',
    priority: 2
    output:
        model_state = project_paths.models \
            / '{model_name}{model_args}_{seed}_{data_name}' \
            / 'trained.pt'
    shell:
        """
        # Run training
        {params.execution_cmd} \
            --config_path {params.base_config_path:q} \
            --input_model_state {input.model_state:q} \
            --output_model_state {output.model_state:q} \
            --model_name {wildcards.model_name} \
            --dataset_link {params.dataset_link:q} \
            --dataset_train {input.dataset_train:q} \
            --dataset_val {input.dataset_val:q} \
            --data_name {wildcards.data_name} \
            --data_group {params.data_group} \
            --seed {wildcards.seed} \
            --resolution {params.resolution} \
            --normalize {params.normalize:q} \
            {params.model_arguments}

        # Create hash documentation file
        touch {params.hash_file}

        # Create folder symlink (relative for portability)
        ln -sf $(basename {params.model_folder}) {params.symlink_folder}

        echo "Created symlink: {params.symlink_folder} -> {params.model_folder}"
        """
```

#### `test_model` (uses polymorphic wildcard)

```python
rule test_model:
    """Evaluate model on test data.

    Accepts both full and hashed model identifiers via {model_identifier} wildcard.
    The checkpoint in train_model ensures hashed symlinks exist before this runs.
    """
    input:
        model_state = project_paths.models \
            / '{model_name}{model_identifier}' \
            / '{status}.pt',
        dataset_ready = project_paths.data.interim / '{data_name}' / 'test_{data_group}.ready',
        script = SCRIPTS / 'runtime' / 'test_model.py'
    params:
        base_config_path = WORKFLOW_CONFIG_PATH,
        model_arguments = lambda w: parse_arguments(w, 'model_args'),
        data_arguments = lambda w: parse_arguments(w, 'data_args'),
        dataset_path = lambda w: project_paths.data.interim / w.data_name / f'test_{w.data_group}',
        normalize = lambda w: (
            config.normalize if hasattr(config, 'normalize') else json.dumps((
                config.data_statistics[w.data_name]['mean'],
                config.data_statistics[w.data_name]['std']
            ))
        ),
        batch_size = config.test_batch_size,
        enable_progress_bar = True,
        execution_cmd = lambda w, input: build_execution_command(
            script_path=input.script,
            use_distributed=False,
        ),
    priority: 1
    output:
        responses = project_paths.reports \
            / '{model_name}{model_identifier}_{status}' \
            / '{data_loader}{data_args}_{data_group}' \
            / 'test_responses.pt',
        results = project_paths.reports \
            / '{model_name}{model_identifier}_{status}' \
            / '{data_loader}{data_args}_{data_group}' \
            / 'test_outputs.csv'
    log:
        project_paths.logs / "slurm" / "rule_test_model" \
            / '{model_name}{model_identifier}_{status}' \
            / '{data_loader}{data_args}_{data_group}.log'
    shell:
        """
        {params.execution_cmd} \
            --config_path {params.base_config_path:q} \
            --input_model_state {input.model_state:q} \
            --output_results {output.results:q} \
            --output_responses {output.responses:q} \
            --model_name {wildcards.model_name} \
            --data_name {wildcards.data_name} \
            --dataset_path {params.dataset_path:q} \
            --data_loader {wildcards.data_loader} \
            --data_group {wildcards.data_group} \
            --seed {wildcards.seed} \
            --normalize {params.normalize:q} \
            --enable_progress_bar {params.enable_progress_bar} \
            {params.model_arguments} \
            {params.data_arguments} \
            --batch_size {params.batch_size}
        """
```

**Note:** The `{model_identifier}` wildcard will match either:
- Full: `tsteps=20+..._42_imagenette`
- Hash: `hash=a7f3c9d4`

Snakemake resolves this based on which files exist after checkpoint.

### Component 3: Processing Rules

#### `process_test_data` (triggers checkpoint, uses hash)

```python
rule process_test_data:
    """Process test data combining layer responses and performance metrics.

    Requires models (full form) which triggers train_model checkpoint.
    References test outputs using hashed identifiers for shorter paths.
    """
    input:
        # Full-form model paths trigger train_model checkpoint
        models = expand(
            project_paths.models
            / '{{model_name}}:{{args1}}{category}{category_value}{{args2}}_{{seed}}_{{data_name}}'
            / '{status}.pt',
            category = lambda w: w.category_str.strip('*'),
            category_value = lambda w: config.experiment_config['categories'].get(
                w.category_str.strip('=*'), ''
            ) if w.category_str else "",
            status = lambda w: config.experiment_config[w.experiment].get('status', w.status),
        ),

        # Hashed model identifiers for test outputs (shorter paths)
        responses = expand(
            project_paths.reports
            / '{{model_name}}{hashed_identifier}_{status}'
            / '{data_loader}{data_args}_{{data_group}}'
            / 'test_responses.pt',
            hashed_identifier = lambda w: compute_hash(
                f'{{args1}}{category}{category_value}{{args2}}'.format(
                    category=w.category_str.strip('*'),
                    category_value=config.experiment_config['categories'].get(
                        w.category_str.strip('=*'), ''
                    ) if w.category_str else ""
                ),
                w.seed,
                w.data_name
            ),
            status = lambda w: config.experiment_config[w.experiment].get('status', w.status),
            data_loader = lambda w: config.experiment_config[w.experiment]['data_loader'],
            data_args = lambda w: args_product(config.experiment_config[w.experiment]['data_args']),
        ),

        # Similar for test_outputs...

        script = SCRIPTS / 'visualization' / 'process_test_data.py'
    params:
        measures = ['response_avg', 'response_std', 'guess_confidence',
                   'first_label_confidence', 'accuracy_top3', 'accuracy_top5'],
        parameter = lambda w: config.experiment_config[w.experiment]['parameter'],
        category = lambda w: w.category_str.strip('=*'),
        additional_parameters = 'epoch',
        batch_size = 1,
        remove_input_responses = True,
        fail_on_missing_inputs = False,
        sample_resolution = 'sample',
        execution_cmd = lambda w, input: build_execution_command(
            script_path=input.script,
            use_distributed=False,
        ),
    priority: 3
    output:
        test_data = project_paths.reports \
            / '{experiment}' \
            / '{experiment}_{model_name}:{args1}{category_str}{args2}_{seed}_{data_name}_{status}' \
            / 'test_data_{data_group}.csv'
    shell:
        """
        {params.execution_cmd} \
            --responses {input.responses:q} \
            --test_outputs {input.test_outputs:q} \
            --output {output.test_data:q} \
            --parameter {params.parameter} \
            --category {params.category} \
            --measures {params.measures} \
            --batch_size {params.batch_size} \
            --sample_resolution {params.sample_resolution} \
            --additional_parameters {params.additional_parameters} \
            --remove_input_responses {params.remove_input_responses} \
            --fail_on_missing_inputs {params.fail_on_missing_inputs}
        """
```

## Execution Flow

### DAG Construction and Checkpoint Behavior

```
1. User requests:
   process_test_data: experiment_DyRCNNx8:tsteps=20+..._{seed}_{data}_{status}/test_data_{group}.csv

2. Snakemake builds DAG:
   process_test_data needs:
     - models: DyRCNNx8:tsteps=20+..._{seed}_{data}/trained.pt
     - responses: DyRCNNx8:hash=a7f3c9d4_trained/StimulusNoise:.../test_responses.pt

3. To get models:
   train_model checkpoint needs to run

4. train_model executes:
   - Trains model → creates trained.pt
   - Creates a7f3c9d4.hash file
   - Creates symlink: DyRCNNx8:hash=a7f3c9d4/ → DyRCNNx8:tsteps=20+..._{seed}_{data}/

5. Checkpoint completes → DAG re-evaluated

6. Now Snakemake sees:
   - Symlink DyRCNNx8:hash=a7f3c9d4/ exists
   - Can resolve test_model input: DyRCNNx8:hash=a7f3c9d4/trained.pt

7. test_model executes:
   - Input: DyRCNNx8:hash=a7f3c9d4/trained.pt (via symlink)
   - Output: DyRCNNx8:hash=a7f3c9d4_trained/StimulusNoise:.../test_outputs.csv

8. process_test_data collects results
```

## Implementation Plan

### Phase 1: Core Utilities (1 hour)

**File:** `dynvision/workflow/snake_utils.smk`

1. Add `compute_hash()` function after `dict_poped()`
2. Write unit tests for hash function
3. Test determinism and idempotence

**Deliverables:**
- `compute_hash()` implementation
- Unit tests in `tests/workflow/test_hash.py`

### Phase 2: Update Model Rules (2 hours)

**File:** `dynvision/workflow/snake_runtime.smk`

1. Update `init_model` to use folder structure (no wildcard constraints needed)
2. Convert `train_model` to checkpoint
3. Add symlink creation logic to train_model
4. Update `test_model` to use `{model_identifier}` wildcard
5. Keep hierarchical log paths

**Deliverables:**
- Modified init_model (folder structure)
- train_model as checkpoint with symlink creation
- test_model accepting polymorphic wildcard

### Phase 3: Update Processing Rules (2 hours)

**File:** `dynvision/workflow/snake_runtime.smk`

1. Update `process_test_data` inputs:
   - Models: use full identifiers (triggers checkpoint)
   - Test outputs: use hashed identifiers (shorter paths)
2. Update output paths if needed

**Deliverables:**
- Modified process_test_data with dual identifier handling

### Phase 4: Integration Testing (2 hours)

1. Dry-run with one experiment
2. Verify symlinks created correctly
3. Check DAG resolution with checkpoint
4. Validate output paths

**Test commands:**
```bash
# Dry run
snakemake --config experiment=rctarget -n

# Force checkpoint re-evaluation
snakemake --config experiment=rctarget --forcerun train_model -n

# Check created files
ls -l models/DyRCNNx8:hash=*/
```

**Deliverables:**
- Integration test results
- Documentation of any issues

### Phase 5: Documentation (1 hour)

1. Update developer docs
2. Add examples of hash usage
3. Document troubleshooting

**Deliverables:**
- Updated docs/development/guides/claude-guide.md
- Examples in this planning doc

## Testing Strategy

### Unit Tests

**File:** `tests/workflow/test_hash_compression.py`

```python
import pytest
from dynvision.workflow.snake_utils import compute_hash

def test_compute_hash_deterministic():
    """Hash function is deterministic."""
    hash1 = compute_hash(':tsteps=20+dt=2', '42', 'imagenette')
    hash2 = compute_hash(':tsteps=20+dt=2', '42', 'imagenette')
    assert hash1 == hash2
    assert hash1.startswith(':hash=')
    assert len(hash1) == len(':hash=') + 8

def test_compute_hash_idempotent():
    """Hashing a hash returns it unchanged."""
    args = ':tsteps=20+dt=2'
    hash1 = compute_hash(args, '42', 'imagenette')
    hash2 = compute_hash(hash1, '42', 'imagenette')
    assert hash1 == hash2

def test_compute_hash_variadic():
    """Works with any number of arguments."""
    hash1 = compute_hash('arg1', 'arg2')
    hash2 = compute_hash('arg1', 'arg2', 'arg3')
    assert hash1 != hash2  # Different args → different hash

def test_compute_hash_strips_colon():
    """Handles colons in arguments."""
    hash1 = compute_hash(':tsteps=20', '42', 'imagenette')
    hash2 = compute_hash('tsteps=20', '42', 'imagenette')
    assert hash1 == hash2  # Colon stripped before hashing
```

### Integration Tests

**Manual testing workflow:**

```bash
# 1. Create test model configuration
cat > test_config.yaml <<EOF
model_name: DyRCNNx8
model_args:
  tsteps: 20
  dt: 2
  tau: 5
seed: 42
data_name: imagenette
EOF

# 2. Run init_model
snakemake models/DyRCNNx8:tsteps=20+dt=2+tau=5_42_imagenette/init.pt -n

# 3. Run train_model (checkpoint)
snakemake models/DyRCNNx8:tsteps=20+dt=2+tau=5_42_imagenette/trained.pt -f

# 4. Verify symlink created
ls -la models/DyRCNNx8:hash=*/
readlink models/DyRCNNx8:hash=*/

# 5. Verify hash file exists
cat models/DyRCNNx8:tsteps=20+dt=2+tau=5_42_imagenette/*.hash

# 6. Test that test_model can find hashed model
snakemake reports/DyRCNNx8:hash=*_trained/StimulusNoise:*/test_outputs.csv -n
```

## Success Criteria

### Functional Requirements

- ✅ No filesystem length errors
- ✅ train_model creates symlink correctly
- ✅ test_model resolves both full and hashed identifiers
- ✅ process_test_data triggers checkpoint and finds results
- ✅ Existing workflows continue to work

### Performance Requirements

- ✅ Checkpoint overhead < 5 seconds
- ✅ Hash computation < 1ms
- ✅ No significant DAG construction slowdown

### Quality Requirements

- ✅ Unit tests pass (100% coverage for compute_hash)
- ✅ Integration tests verify end-to-end flow
- ✅ Documentation complete
- ✅ No breaking changes

## Known Limitations

### Filesystem Compatibility

**Symlinks may not work on:**
- Windows (without Developer Mode)
- Some network filesystems

**Mitigation:** Can detect and use hard links or copies as fallback

### Hash Collisions

**Probability:** 8 hex chars = 4B combinations
- <1000 models: collision probability <0.01%
- Can increase to 10-12 chars if needed

### Path Component Limits

**Still possible (though unlikely):**
- If hashed identifier + status + data_loader + data_args > 255 chars
- Can apply hashing to data_args too if needed

## Migration from Current State

### Old vs New Structure Comparison

#### Models

**Old (flat):**
```
models/DyRCNNx8/DyRCNNx8:tsteps=20+dt=2+tau=5+..._{seed}_{data}_init.pt
models/DyRCNNx8/DyRCNNx8:tsteps=20+dt=2+tau=5+..._{seed}_{data}_trained.pt
```

**New (hierarchical):**
```
models/DyRCNNx8:tsteps=20+dt=2+tau=5+..._{seed}_{data}/init.pt
models/DyRCNNx8:tsteps=20+dt=2+tau=5+..._{seed}_{data}/trained.pt
models/DyRCNNx8:tsteps=20+dt=2+tau=5+..._{seed}_{data}/a7f3c9d4.hash
models/DyRCNNx8:hash=a7f3c9d4/  → symlink
```

**Migration:** No migration needed - new structure independent of old

#### Test Results

**Old (flat, single directory):**
```
reports/torch/DyRCNNx8:tsteps=20+..._{seed}_{data}_trained_torch:_all/test_outputs.csv
```

**New (hierarchical, separated concerns):**
```
reports/torch/                                     ← Data loader
  DyRCNNx8:hash=a7f3c9d4_trained/                 ← Model + status
    torch:_all/                                    ← Data config
      test_outputs.csv                             ← Result file
```

**Migration:** Existing files work; new files use hierarchical structure

#### Processed Experiment Data

**Old (mixed in with other reports):**
```
reports/uniformnoise/uniformnoise_DyRCNNx8:tsteps=20+..._{seed}_{data}_trained_all/test_data.csv
```

**New (clear experiment organization):**
```
reports/uniformnoise/                              ← Experiment
  DyRCNNx8:hash=a7f3c9d4_trained/                 ← Model + status
    all_test_data.csv                              ← Processed data
```

**Migration:** Process rules updated to use new paths

#### Visualizations

**Old:**
```
figures/uniformnoise/uniformnoise_DyRCNNx8:tsteps=20+..._{seed}_{data}_trained_all/performance.png
```

**New:**
```
figures/uniformnoise/                              ← Experiment
  DyRCNNx8:hash=a7f3c9d4_trained/                 ← Model + status
    all_performance.png                            ← Plot
```

**Migration:** Plotting rules updated to use new paths

### Backward Compatibility Strategy

1. **Existing data files:** No need to move or rename
2. **New workflow runs:** Use new hierarchical structure
3. **Coexistence:** Old and new structures can coexist
4. **Gradual transition:**
   - Implement hierarchical structure in rules
   - New runs automatically use new structure
   - Old data accessible but not reorganized
   - Optional cleanup script to migrate old data if desired

### Rule Changes Required

#### Updated Rules

1. **`init_model`**: Output to `{model_identifier}/init.pt` instead of `{model_identifier}_init.pt`
2. **`train_model`**:
   - Output to `{model_identifier}/trained.pt`
   - Becomes checkpoint
   - Creates hash symlink
3. **`test_model`**:
   - Output to `{data_loader}/{model_identifier}_{status}/{data_loader}:{data_args}_{data_group}/`
   - Uses polymorphic `{model_identifier}` wildcard
4. **`process_test_data`**:
   - Input from `{data_loader}/{model_identifier}_{status}/...`
   - Output to `{experiment}/{model_identifier}_{status}/{data_group}_test_data.csv`
5. **Visualization rules**:
   - Input from `{experiment}/{model_identifier}_{status}/{data_group}_test_data.csv`
   - Output to `{experiment}/{model_identifier}_{status}/{data_group}_{plot}.png`

#### Unchanged Rules

- Data preparation rules (no model dependencies)
- Utility rules

### Gradual Rollout Plan

**Phase 1: Core infrastructure (Week 1)**
1. Implement `compute_hash()` utility
2. Update model rules (init, train, test)
3. Test with single model/experiment

**Phase 2: Processing pipeline (Week 2)**
1. Update `process_test_data` rule
2. Update visualization rules
3. Test full experiment workflow

**Phase 3: Validation (Week 3)**
1. Run parallel workflows (old vs new structure)
2. Validate outputs match
3. Performance testing

**Phase 4: Full deployment (Week 4)**
1. Update all experiment configurations
2. Document new structure
3. Optional: migrate existing data

### Benefits Summary

The hierarchical reorganization provides:

**Immediate benefits:**
- ✅ Solves filesystem length errors
- ✅ Clearer separation of model vs test attributes
- ✅ Easier navigation and file discovery

**Long-term benefits:**
- ✅ Scales to larger parameter sweeps
- ✅ Simplifies result organization
- ✅ Enables better experiment tracking
- ✅ Facilitates automated analysis (all tests for model X)

## References

- Snakemake checkpoints: https://snakemake.readthedocs.io/en/stable/snakefiles/rules.html#data-dependent-conditional-execution
- Filesystem limits: https://en.wikipedia.org/wiki/Comparison_of_file_systems#Limits
- MD5 collision resistance: https://en.wikipedia.org/wiki/MD5

## Timeline

- **Phase 1 (Utilities):** 1 hour
- **Phase 2 (Model rules):** 2 hours
- **Phase 3 (Processing rules):** 2 hours
- **Phase 4 (Testing):** 2 hours
- **Phase 5 (Documentation):** 1 hour
- **Total:** ~8 hours

## Change Log

- 2025-01-06: Initial planning document created
- 2025-01-06: Updated with polymorphic wildcard approach and simplified checkpoint
