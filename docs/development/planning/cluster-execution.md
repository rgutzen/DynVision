# Cluster Execution Detection - Implementation Plan

**Date**: 2025-11-22
**Status**: Planning → Implementation → Documentation
**Priority**: High (blocks cluster workflows)

## Executive Summary

Replace config-based cluster detection (`use_executor`) with automatic environment variable detection to enable robust, self-configuring execution across local and cluster environments. This resolves the conflict between frozen config requirements and dynamic environment detection.

## Problem Statement

### Current Broken Behavior

When running workflows via `snakecharm.sh` on a compute cluster:

1. `_snakecharm.sh` passes `--config use_executor=True` to Snakemake
2. With frozen config loading, CLI config overrides must be merged at workflow start
3. Rules use `get_param('use_executor', False)(w)` to decide whether to use `executor_wrapper.sh`
4. If `use_executor=False` (default), python runs **without** singularity container
5. **Result**: `ModuleNotFoundError: No module named 'torch'` (not in host environment)

### Root Cause

**Architectural misalignment**: Execution environment (cluster vs. local) is a property of the **runtime context**, not experiment **configuration**.

- Experiment configs should define *what* to compute (hyperparameters, datasets, models)
- Execution environment should be detected from *where* the code runs (cluster scheduler vars)

Conflating these via `use_executor` config creates fragility:
- Requires manual config synchronization with execution mode
- Breaks with frozen config (config determined at workflow start, not job submission)
- Violates separation of concerns (configuration vs. execution context)

## Solution: Environment Variable Detection

### Design Principles

**Building blocks philosophy**:
- Execution environment detection is an independent, reusable utility
- No coupling to experiment configuration
- Composes cleanly with existing `build_execution_command()`
- Can be used across different workflow systems

**Scientific rigor**:
- Deterministic: same environment always detected the same way
- Explicit: log detection results for reproducibility
- Robust: handle multiple cluster schedulers

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│ Snakemake Rule (e.g., init_model)                       │
│   params:                                                │
│     execution_cmd = build_execution_command(...)         │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│ build_execution_command(script_path, use_distributed)   │
│   ├─ is_cluster_execution()  ← NEW                      │
│   │    └─ Check env vars (SLURM_JOB_ID, etc.)          │
│   ├─ use_executor = detected automatically              │
│   └─ Build command with/without executor_wrapper.sh     │
└────────────────────┬────────────────────────────────────┘
                     │
                     ├─ Local: python script.py
                     │
                     └─ Cluster: executor_wrapper.sh python script.py
                              └─ singularity exec ... conda activate ... python
```

## Implementation Plan

### Phase 1: Create Detection Utility

**File**: `dynvision/workflow/snake_utils.smk`
**Lines**: Insert after imports (~line 35)

```python
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
```

**Testing**:
```python
# Unit test cases
def test_is_cluster_execution():
    # Test SLURM detection
    os.environ['SLURM_JOB_ID'] = '12345'
    assert is_cluster_execution() == True
    del os.environ['SLURM_JOB_ID']

    # Test PBS detection
    os.environ['PBS_JOBID'] = '67890'
    assert is_cluster_execution() == True
    del os.environ['PBS_JOBID']

    # Test local execution
    assert is_cluster_execution() == False
```

### Phase 2: Update build_execution_command

**File**: `dynvision/workflow/snake_utils.smk`
**Function**: `build_execution_command` (currently at line 316)

**Changes**:

```python
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
```

**Diff summary**:
```diff
  def build_execution_command(script_path, use_distributed=False):
-     """
-     Build the execution command with conditional wrappers.
-
-     Args:
-         script_path: Path to the Python script to execute
-         use_distributed: Whether to use distributed setup
-         use_executor: Whether to use executor wrapper
-
-     Returns:
-         String containing the complete execution command
-     """
+     """Build execution command with automatic cluster detection."""
      cmd_parts = []

+     # Automatic cluster detection (replaces use_executor config)
+     use_executor = is_cluster_execution()
+
      if use_distributed:
          setup_script = SCRIPTS / 'cluster' / 'setup_distributed_execution.sh'
          cmd_parts.append(f"source {setup_script} &&")
```

### Phase 3: Update Rule Calls

**Files to update**: All Snakemake rule files that call `build_execution_command`

**Current pattern** (across all rules):
```python
params:
    execution_cmd = lambda w, input: build_execution_command(
        script_path=input.script,
        use_distributed=False,
        use_executor=get_param('use_executor', False)(w)  # ← REMOVE
    ),
```

**New pattern**:
```python
params:
    execution_cmd = lambda w, input: build_execution_command(
        script_path=input.script,
        use_distributed=False,
    ),
```

**Files to modify**:
1. `dynvision/workflow/snake_runtime.smk`
   - Line 37-41: `rule init_model`
   - Line 96-100: `rule train_model`
   - Line 173-177: `rule train_model_distributed`
   - Line 212-216: `rule test_model`
   - Line 235-239: `rule test_model_dataloader`

2. `dynvision/workflow/snake_data.smk`
   - Line 125-129: `rule get_data`
   - Line 259-263: `rule build_ffcv_datasets`

3. `dynvision/workflow/snake_visualizations.smk`
   - Line 41-45: `rule process_test_data`
   - Line 80-84: `rule plot_responses_to_stimulus`
   - Line 117-121: `rule plot_responses_of_model`
   - Line 178-182: `rule plot_performance_by_category`
   - Line 207-211: `rule plot_experiments_on_categories`
   - Line 245-249: `rule plot_experiments_on_models`
   - Line 294-298: `rule plot_experiments_over_time`
   - Line 336-340: `rule plot_model_performance_over_time`
   - Line 378-382: `rule plot_model_comparison`
   - Line 421-425: `rule plot_time_evolution`

**Automated replacement** (for efficiency):
```bash
# Find and replace across all workflow files
sed -i 's/use_executor=get_param.*,$//' dynvision/workflow/snake_*.smk

# Verify changes
git diff dynvision/workflow/snake_*.smk
```

### Phase 4: Remove Deprecated Config

**File**: `dynvision/configs/config_defaults.yaml`
**Line**: 240

**Change**:
```diff
- use_executor: False  # automatic setting (don't change)
+ # use_executor removed: cluster execution now auto-detected via environment variables
+ # See: docs/development/planning/cluster-execution.md
```

**File**: `dynvision/cluster/_snakecharm.sh`
**Lines**: 38-39

**Change**:
```diff
- # Always append use_executor=True to config
- args+=("use_executor=True")
+ # Cluster execution auto-detected via environment variables (SLURM_JOB_ID, etc.)
+ # No config override needed
```

### Phase 5: Update Frozen Config Loading

**File**: `dynvision/workflow/snake_utils.smk`
**Function**: `_load_and_freeze_config()`

**Change**: Add note about environment detection

```python
def _load_and_freeze_config() -> Dict[str, Any]:
    """
    Load configuration files once and freeze them for the entire workflow.

    This prevents mid-workflow config changes from affecting running jobs
    when using cluster execution with snakemake-executor-plugin.

    Note on cluster execution:
        Cluster detection (singularity/conda wrapper) is handled via
        environment variables, NOT config. This ensures detection works
        correctly with frozen config and separates execution context from
        experiment configuration.

    Returns:
        Merged configuration dictionary frozen at workflow start
    """
    # ... existing implementation
```

## Testing Strategy

### Unit Tests

```python
# tests/workflow/test_cluster_detection.py
import os
import pytest
from dynvision.workflow.snake_utils import is_cluster_execution

class TestClusterDetection:
    def test_slurm_detection(self, monkeypatch):
        monkeypatch.setenv('SLURM_JOB_ID', '12345')
        assert is_cluster_execution() == True

    def test_pbs_detection(self, monkeypatch):
        monkeypatch.setenv('PBS_JOBID', '67890')
        assert is_cluster_execution() == True

    def test_local_detection(self, monkeypatch):
        # Ensure no cluster vars present
        for var in ['SLURM_JOB_ID', 'PBS_JOBID', 'LSB_JOBID']:
            monkeypatch.delenv(var, raising=False)
        assert is_cluster_execution() == False

    def test_sge_detection(self, monkeypatch):
        monkeypatch.setenv('JOB_ID', '111')
        monkeypatch.setenv('SGE_TASK_ID', '1')
        assert is_cluster_execution() == True

    def test_job_id_alone_not_sufficient(self, monkeypatch):
        # JOB_ID alone is too generic
        monkeypatch.setenv('JOB_ID', '111')
        assert is_cluster_execution() == False
```

### Integration Tests

**Test 1: Local execution**
```bash
# On local workstation (no SLURM vars)
cd /path/to/DynVision
snakemake init_model --config model_name=DyRCNNx4 data_name=cifar10 -n

# Verify: execution_cmd should be "python /path/to/init_model.py"
# Should NOT include executor_wrapper.sh
```

**Test 2: Cluster execution via snakecharm**
```bash
# On cluster login node
./dynvision/cluster/snakecharm.sh init_model --config model_name=DyRCNNx4 data_name=cifar10

# Verify in job output:
# - "Cluster execution detected via: SLURM_JOB_ID"
# - execution_cmd includes "executor_wrapper.sh"
# - Python runs inside singularity with conda activated
```

**Test 3: Manual SLURM job**
```bash
# On cluster, submit manual job
sbatch --wrap="cd /path/to/DynVision && snakemake init_model --config model_name=DyRCNNx4"

# Verify same behavior as Test 2
```

### Validation Criteria

✅ **Correctness**:
- Local execution: Python runs directly, no singularity
- Cluster execution: Python runs via executor_wrapper.sh with singularity+conda
- Detection logged in workflow output

✅ **Robustness**:
- Works with frozen config (no config dependency)
- Works across SLURM, PBS, LSF schedulers
- No false positives on local workstations

✅ **Backward compatibility**:
- Existing workflows run without changes
- Frozen config still functions correctly

## Migration Guide

### For Users

**No action required** - cluster execution is now automatic!

**Before** (manual config):
```bash
# Had to manually set use_executor
./snakecharm.sh train_model --config use_executor=True model_name=DyRCNNx4
```

**After** (automatic):
```bash
# Just run - cluster detection is automatic
./snakecharm.sh train_model --config model_name=DyRCNNx4
```

**Verification**: Check logs for detection message:
```
[INFO] Cluster execution detected via: SLURM_JOB_ID
```

### For Developers

**Config changes**:
- `use_executor` removed from `config_defaults.yaml`
- No longer passed via `_snakecharm.sh`

**Code changes**:
- `build_execution_command()` signature simplified (no `use_executor` parameter)
- All rule calls updated to remove `use_executor=get_param(...)(w)`
- New utility: `is_cluster_execution()` for environment detection

**Testing**:
- Add unit tests for `is_cluster_execution()`
- Verify both local and cluster execution work correctly

## Rollout Plan

### Phase 1: Implementation (Estimated: 2 hours)
1. ✅ Create implementation plan (this document)
2. ⏳ Implement `is_cluster_execution()` utility
3. ⏳ Update `build_execution_command()`
4. ⏳ Remove `use_executor` from all rule calls (automated sed)
5. ⏳ Update config files and _snakecharm.sh

### Phase 2: Testing (Estimated: 1 hour)
1. ⏳ Write unit tests for cluster detection
2. ⏳ Test local execution (workstation)
3. ⏳ Test cluster execution (via snakecharm)
4. ⏳ Verify logging output

### Phase 3: Documentation (Estimated: 1 hour)
1. ⏳ Update user guides with automatic detection
2. ⏳ Update troubleshooting guide
3. ⏳ Update parameter-processing.md
4. ⏳ Add note to CHANGELOG.md

### Phase 4: Deployment (Estimated: 30 min)
1. ⏳ Create git commits (implementation, tests, docs)
2. ⏳ Test on actual cluster
3. ⏳ Merge to dev branch

## Documentation Updates Required

### User-Facing Documentation

**1. docs/user-guide/cluster-integration.md**
- Add section on automatic cluster detection
- Explain environment variable mechanism
- Remove references to `use_executor` config

**2. docs/user-guide/troubleshooting.md**
- Add "Cluster not detected" troubleshooting section
- How to verify detection (check logs for "Cluster execution detected")
- Common issues (no scheduler vars, wrong profile)

**3. docs/getting-started.md**
- Update cluster execution quickstart
- Simplify instructions (no manual config needed)

### Developer Documentation

**1. docs/development/guides/parameter-processing.md**
- Update config freezing section
- Explain why `use_executor` was removed
- Document environment-based detection approach

**2. docs/development/guides/claude-guide.md**
- Add cluster detection to "Common Patterns"
- Document `is_cluster_execution()` utility

**3. docs/reference/configuration.md**
- Remove `use_executor` from config reference
- Add note about automatic detection

## Success Metrics

✅ **Functional**:
- [ ] Local workflows run without singularity wrapper
- [ ] Cluster workflows run with singularity wrapper
- [ ] No ModuleNotFoundError on cluster
- [ ] All tests pass

✅ **Code Quality**:
- [ ] No config dependency for environment detection
- [ ] Clean separation: execution context vs. experiment config
- [ ] Reusable building block (`is_cluster_execution()`)
- [ ] Comprehensive logging for debugging

✅ **User Experience**:
- [ ] No manual config required
- [ ] Automatic and transparent
- [ ] Clear error messages if detection fails
- [ ] Documented behavior in user guides

## Known Limitations

1. **Assumes standard schedulers**: SLURM, PBS, LSF, SGE. Custom schedulers may need additional environment variables added to `cluster_indicators`.

2. **Detection at parse time**: Environment vars must be present when Snakemake parses the workflow. For cluster execution, this is typically on the submit node where vars are set by the job submission itself.

3. **No override mechanism**: If detection is incorrect, users cannot manually override. Could add `--config force_local=True` if needed.

## Future Enhancements

1. **Configuration override** (if needed):
   ```python
   # Allow manual override for edge cases
   use_executor = config.get('force_executor', is_cluster_execution())
   ```

2. **Additional schedulers**: Add more cluster indicators as users report other environments.

3. **Container auto-detection**: Extend to detect which container to use based on available images.

4. **Environment validation**: Check that detected environment has required resources (GPUs, conda, etc.).

## Related Files

**Implementation**:
- `dynvision/workflow/snake_utils.smk`: Core detection and command building
- `dynvision/workflow/snake_runtime.smk`: Rule updates
- `dynvision/workflow/snake_data.smk`: Rule updates
- `dynvision/workflow/snake_visualizations.smk`: Rule updates

**Configuration**:
- `dynvision/configs/config_defaults.yaml`: Remove `use_executor`
- `dynvision/cluster/_snakecharm.sh`: Remove `use_executor=True` override

**Testing**:
- `tests/workflow/test_cluster_detection.py`: New unit tests

**Documentation**:
- `docs/user-guide/cluster-integration.md`: Update for automatic detection
- `docs/user-guide/troubleshooting.md`: Add detection troubleshooting
- `docs/development/guides/parameter-processing.md`: Document design rationale

## References

- **Config freezing**: `docs/development/planning/snakecharm-config-stability-issue.md`
- **Executor wrapper**: `dynvision/cluster/executor_wrapper.sh`
- **Building blocks philosophy**: `docs/development/guides/ai-style-guide.md`
- **SLURM environment**: https://slurm.schedmd.com/sbatch.html#SECTION_INPUT-ENVIRONMENT-VARIABLES
- **PBS environment**: https://help.altair.com/2022.1.0/PBS%20Professional/PBSReferenceGuide2022.1.pdf

---

## Appendix A: Environment Variables by Scheduler

| Scheduler | Primary Variable | Alternative Variables | Notes |
|-----------|-----------------|----------------------|-------|
| **SLURM** | `SLURM_JOB_ID` | `SLURM_JOBID` | Most common in academic HPC |
| **PBS/Torque** | `PBS_JOBID` | `PBS_JOBID`, `PBS_ARRAYID` | Also sets PBS_ENVIRONMENT=PBS_BATCH |
| **LSF** | `LSB_JOBID` | `LSB_BATCH_JID` | IBM Platform LSF |
| **SGE/UGE** | `SGE_TASK_ID` + `JOB_ID` | - | JOB_ID alone too generic |
| **LoadLeveler** | `LOADL_STEP_ID` | - | Older IBM system |

## Appendix B: Example Log Output

**Local execution**:
```
[DEBUG] workflow.utils: Local execution detected (no cluster scheduler variables)
[INFO] workflow.utils: Config frozen at workflow start with 127 keys
[INFO] workflow.utils: Workflow config snapshot: logs/configs/workflow_config_20251122-143000.yaml
```

**Cluster execution**:
```
[INFO] workflow.utils: Cluster execution detected via: SLURM_JOB_ID
[INFO] workflow.utils: Config frozen at workflow start with 127 keys
[INFO] workflow.utils: Workflow config snapshot: logs/configs/workflow_config_20251122-143000.yaml
```

**Rule execution (cluster)**:
```
[INFO] Building job: init_model
[DEBUG] Execution command: executor_wrapper.sh python /path/to/init_model.py
============= Container Environment =============
WORLD_SIZE: 1
RANK: 0
LOCAL_RANK: 0
CUDA_VISIBLE_DEVICES: 0
================================================
Conda environment: rva
Python version: Python 3.10.12
```
