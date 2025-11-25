# Roadmap: Filesystem-Based Wildcard Matching for Snakemake

**Feature**: Add support for `?` wildcard character in Snakemake target patterns that uses `glob_wildcards()` to match only existing files in the filesystem.

**Status**: ✅ Implementation Complete
**Created**: 2025-11-24
**Completed**: 2025-11-24

## Summary

Successfully implemented filesystem-based wildcard expansion for Snakemake workflows:

- ✅ Created `workflow_utils.py` with pure functions (minimal dependencies)
- ✅ Implemented `expand_filesystem_pattern()` core function
- ✅ Created `expand_mixed()` drop-in replacement for `expand()`
- ✅ Updated `process_test_data` rule to support `?` wildcards
- ✅ Integrated with Snakemake checkpoint mechanism for dynamic file discovery
- ✅ Updated user and developer documentation
- ✅ Maintained cluster environment compatibility

**Key Achievement**: Users can now use `?` in target paths to match only existing files, enabling partial result analysis and resuming interrupted experiments without config changes.

---

## Overview

### Objective
Enable users to specify Snakemake targets with a `?` wildcard that expands to only existing files on the filesystem, complementing the existing `*` operator which expands to all configured parameter values from YAML configs.

### Motivation
Currently, users can only expand wildcards using values from `config_experiments.yaml`. When testing a subset of already-trained models (e.g., only seeds starting with 5), users must manually list all matching seeds or modify the config. Filesystem-based expansion allows:
- Testing only completed model runs without config changes
- Resuming interrupted parameter sweeps
- Exploratory analysis of partial results
- Avoiding re-runs of failed experiments

### Example Usage

**Current behavior** (config-based expansion):
```bash
# Expands to ALL seeds in config: [0001, 0002, 0003, ...]
snakemake reports/duration/duration_DyRCNNx8:tsteps=5+seed=*_trained_all/test_data.csv
```

**New behavior** (filesystem-based expansion):
```bash
# Expands to ONLY existing seeds matching pattern: [5000, 5001, 5023]
snakemake reports/duration/duration_DyRCNNx8:tsteps=5+seed=5?_trained_all/test_data.csv
```

---

## Technical Context

### Current System

1. **Parameter Expansion (`*`)**:
   - Defined in `dynvision/workflow/snake_utils.smk:275-309` (`args_product()`)
   - Expands wildcards using values from config files
   - Used with Snakemake's `expand()` function
   - Examples: `tau=*` → `tau=3`, `tau=5`, `tau=9` (from config)

2. **File Path Format**:
   ```
   {model_name}{model_args}_{seed}_{data_name}_{status}_{data_loader}{data_args}_{data_group}

   Example:
   DyRCNNx8:tsteps=20+rctype=full+tau=5_0040_imagenette_trained_StimulusDuration:stim=20_all
   ```

3. **Wildcard Constraints**:
   - Defined in `snake_utils.smk:159-177`
   - Regex patterns like `model_args = r'(:[a-z,;:\+=\d\.\*]+|\s?)'`
   - Current pattern allows `*` but not `?`

4. **Snakemake's Built-in Wildcards**:
   - `*` = matches any string (in shell context)
   - `?` = matches single character (in shell context)
   - `**` = recursive directory matching (in shell context)

### Key Design Questions

1. **When to process `?` patterns?**
   - Before Snakemake DAG building? (preprocessing)
   - During rule input function evaluation? (dynamic)
   - Through checkpoint rules? (re-evaluation)

2. **How to distinguish between patterns?**
   - `seed=*` → all config values (existing behavior)
   - `seed=5?` → filesystem glob for files with seed=5*
   - `seed=50??` → filesystem glob for seed=50* (any 4-digit seed starting with 50)

3. **What if no files match?**
   - Error with helpful message
   - Warning and skip
   - Fall back to config expansion

4. **How to handle mixed patterns?**
   - `tau=*+seed=5?` → `tau` from config, `seed` from filesystem
   - Need to combine both expansion methods

---

## Investigation Phase

### Dependencies Review

**Relevant imports in snake_utils.smk**:
- `from pathlib import Path` - for filesystem operations
- `from itertools import product` - for cartesian products
- `import re` - for pattern matching
- Built-in Snakemake functions: `glob_wildcards()`, `expand()`

**Snakemake features to use**:
- `glob_wildcards(pattern)` - extract wildcard values from existing files
- Input functions - evaluate dynamically at runtime
- Checkpoint rules - re-evaluate DAG after filesystem changes (if needed)

### Existing Code Analysis

**`args_product()` function** (snake_utils.smk:275-309):
- Takes dictionary of parameter options
- Generates cartesian product of all combinations
- Returns list of formatted argument strings
- **Does NOT check filesystem**

**Wildcard constraints** (snake_utils.smk:159-177):
- Need to update `model_args` and `data_args` patterns to allow `?`
- Current: `model_args = r'(:[a-z,;:\+=\d\.\*]+|\s?)'`
- Need: `model_args = r'(:[a-z,;:\+=\d\.\*\?]+|\s?)'`

**Example `expand()` usage** (snake_experiments.smk:36):
```python
expand(project_paths.figures / "{experiment}" /
       "{experiment}_DyRCNNx8:tsteps=20+rctype=*+tau=5_{seed}_imagenette_trained_all/dynamics_{layer}.png",
       experiment=['duration', 'contrast'],
       rctype=['full', 'self', 'depthpointwise'],  # from config
       seed=SEED,
       layer=['V1', 'V2', 'V4', 'IT'])
```

### Design Constraints

1. **Backward Compatibility**:
   - Existing `*` expansion must continue to work
   - No breaking changes to current workflows

2. **Performance**:
   - Filesystem globbing can be slow on large directories
   - Should cache results where possible
   - Only glob when `?` is present

3. **Clarity**:
   - Error messages must clearly distinguish between:
     - Missing files (filesystem issue)
     - Invalid config values (config issue)
     - Pattern syntax errors (user error)

4. **Composition**:
   - Must work with other Snakemake features (checkpoints, input functions, etc.)
   - Should compose with existing `args_product()`

---

## Design Decision

### Chosen Approach: Preprocessing Function + Dynamic Input Functions

**Rationale**:
- Most flexible and composable
- Works with existing Snakemake mechanisms
- Can be used in both command-line targets and rule definitions
- Maintains clear separation between config-based and filesystem-based expansion

### Architecture

1. **New function: `expand_filesystem_wildcards()`**
   - Location: `snake_utils.smk`
   - Purpose: Scan filesystem and extract matching parameter values
   - Input: Pattern string with `?` characters
   - Output: List of actual values found in filesystem

2. **New function: `resolve_mixed_wildcards()`**
   - Combines config-based (`*`) and filesystem-based (`?`) expansion
   - Handles mixed patterns like `tau=*+seed=5?`
   - Returns list of fully expanded argument strings

3. **Modify: `args_product()`**
   - Add optional `filesystem_patterns` parameter
   - Integrate filesystem expansion with config expansion

4. **Update: Wildcard constraints**
   - Allow `?` in `model_args` and `data_args` patterns

### Implementation Details

#### Function: `expand_filesystem_wildcards(pattern, search_dir, prefix=':', delimiter='+', assigner='=')`

```python
def expand_filesystem_wildcards(pattern, search_dir, prefix=':', delimiter='+', assigner='='):
    """
    Expand wildcards in pattern by globbing filesystem.

    Detects parameters with '?' and uses glob_wildcards() to find
    matching files, then extracts parameter values.

    Args:
        pattern: Argument string like ':tau=5+seed=5?+rctype=full'
        search_dir: Directory to search (e.g., project_paths.models)
        prefix: Argument prefix character (default ':')
        delimiter: Argument separator (default '+')
        assigner: Key-value separator (default '=')

    Returns:
        dict: Parameter name -> list of values found in filesystem

    Example:
        pattern = ':tau=5+seed=5?'
        search_dir = 'models/DyRCNNx8'

        # Finds files like:
        # - DyRCNNx8:tau=5+seed=5000_imagenette_trained.pt
        # - DyRCNNx8:tau=5+seed=5001_imagenette_trained.pt

        # Returns: {'seed': ['5000', '5001']}
    """
```

**Algorithm**:
1. Parse pattern to identify parameters with `?`
2. For each `?` parameter:
   - Construct glob pattern (replace `?` with `*` for globbing)
   - Use `glob.glob()` or `glob_wildcards()` to find matching files
   - Extract parameter values using regex
   - Deduplicate and sort values
3. Return dictionary of parameter -> values

#### Function: `resolve_mixed_wildcards(args_dict, config, search_dir)`

```python
def resolve_mixed_wildcards(args_dict, config, search_dir):
    """
    Resolve mixed * (config) and ? (filesystem) wildcards.

    Args:
        args_dict: Dict with values that may be '*', '5?', or concrete values
        config: Config object with parameter categories
        search_dir: Directory to search for filesystem wildcards

    Returns:
        List of fully expanded argument strings

    Example:
        args_dict = {'tau': '*', 'seed': '5?', 'rctype': 'full'}

        # tau=* from config: [3, 5, 9]
        # seed=5? from filesystem: [5000, 5001]
        # rctype=full: concrete value

        # Returns:
        [':tau=3+seed=5000+rctype=full',
         ':tau=3+seed=5001+rctype=full',
         ':tau=5+seed=5000+rctype=full',
         ':tau=5+seed=5001+rctype=full',
         ':tau=9+seed=5000+rctype=full',
         ':tau=9+seed=5001+rctype=full']
    """
```

**Algorithm**:
1. Separate parameters into three categories:
   - Config wildcards (`*`)
   - Filesystem wildcards (contains `?`)
   - Concrete values (no wildcards)
2. Expand config wildcards from `config.experiment_config.categories`
3. Expand filesystem wildcards using `expand_filesystem_wildcards()`
4. Compute cartesian product of all expansions
5. Format as argument strings

---

## Implementation Plan

### Phase 1: Core Functionality
- [x] Create roadmap document
- [ ] Implement `expand_filesystem_wildcards()` function
- [ ] Implement `resolve_mixed_wildcards()` function
- [ ] Update wildcard constraints to allow `?`
- [ ] Add detection logic for `?` patterns

### Phase 2: Integration
- [ ] Modify `args_product()` to use new functions when `?` detected
- [ ] Update rules that use wildcard expansion
- [ ] Add caching for filesystem glob results
- [ ] Handle edge cases (no matches, invalid patterns)

### Phase 3: Testing
- [ ] Create test files with known patterns
- [ ] Test single parameter with `?`: `seed=5?`
- [ ] Test multiple `?` parameters: `seed=5?+tau=?`
- [ ] Test mixed `*` and `?`: `tau=*+seed=5?`
- [ ] Test no matches found scenario
- [ ] Test with different directory structures

### Phase 4: Documentation & Polish
- [ ] Add docstrings to new functions
- [ ] Update @docs/user-guide/workflows.md with `?` wildcard examples
- [ ] Update @docs/development/guides/claude-guide.md workflow section
- [ ] Add error messages for common mistakes
- [ ] Add examples to developer guide

---

## Documentation Updates Required

After implementation is complete, the following documentation sections must be updated:

### 1. @docs/user-guide/workflows.md
**Section: "Working with Wildcards" (lines ~193-208)**
- Add explanation of `?` wildcard for filesystem-based expansion
- Add examples comparing `*` vs `?` expansion
- Add usage patterns like `seed=5?`, `tau=*+seed=5?`

**New section to add: "Filesystem-Based Wildcard Expansion"**
- Explain when to use `?` vs `*`
- Show example command lines
- Explain how values are discovered from existing files
- Add troubleshooting tips for no matches

### 2. @docs/development/guides/claude-guide.md
**Section: "Experiment Wildcards" (lines ~417-433)**
- Update wildcard format documentation
- Add `?` to the list of special characters
- Provide examples of filesystem-based expansion in experimental workflows

**Section: "Workflow System" (lines ~246-277)**
- Update description of wildcard-based path patterns
- Add note about filesystem-based expansion capabilities

### 3. New documentation file (optional)
**Create: @docs/user-guide/advanced-workflows.md**
- Detailed guide on advanced wildcard usage
- Filesystem-based expansion patterns
- Combining config and filesystem expansion
- Performance considerations
- Troubleshooting guide

---

## Design Alternatives Considered

### Alternative 1: Wrapper Script
**Approach**: Create a wrapper script that preprocesses targets before calling Snakemake.

**Pros**:
- Clean separation of concerns
- No modification to Snakemake workflow

**Cons**:
- Extra step for users
- Harder to integrate with existing workflows
- Doesn't work well with Snakemake's DAG visualization

**Decision**: Rejected - too intrusive for users

### Alternative 2: Checkpoint Rules
**Approach**: Use Snakemake checkpoint rules to dynamically re-evaluate targets.

**Pros**:
- Native Snakemake feature
- Automatic re-evaluation

**Cons**:
- Checkpoints are for rules, not command-line targets
- More complex implementation
- Harder to reason about

**Decision**: Rejected - checkpoints solve a different problem

### Alternative 3: Custom Snakemake Plugin
**Approach**: Create a Snakemake plugin to extend wildcard functionality.

**Pros**:
- Most powerful and flexible
- Could add other wildcard types

**Cons**:
- Much more complex
- Requires understanding Snakemake internals
- Installation and maintenance overhead

**Decision**: Rejected - overengineered for this use case

---

## Implementation Progress

### 2025-11-24: Initial Investigation
- ✅ Read AI Style Guide and Claude Guide
- ✅ Analyzed existing parameter expansion system
- ✅ Reviewed Snakemake wildcard handling
- ✅ Identified key files: `snake_utils.smk`, `snake_runtime.smk`, `config_experiments.yaml`
- ✅ Understood current `args_product()` function
- ✅ Created roadmap document
- ✅ Understanding Snakemake's command-line target processing

### 2025-11-24: Architecture & Design
- ✅ Clarified design questions with user
- ✅ Decided on Option 3 (Hybrid approach): `workflow_utils.py` + wrappers in `snake_utils.smk`
- ✅ Designed `expand_filesystem_pattern()` algorithm
- ✅ Planned integration with existing `args_product()` and `expand()`

### 2025-11-24: Implementation
- ✅ Created `workflow_utils.py` with pure functions (minimal dependencies)
- ✅ Implemented `expand_filesystem_pattern()` core function
- ✅ Implemented `args_product()` with mixed wildcard support
- ✅ Implemented helper functions: `parse_arguments()`, `replace_param_in_string()`, `dict_poped()`
- ✅ Created `expand_mixed()` wrapper for Snakemake's `expand()`
- ✅ Updated `snake_utils.smk` to import from `workflow_utils.py`
- ✅ Created wrapper functions that integrate with Snakemake config
- ✅ Updated wildcard constraints to allow `?` character
- ✅ Updated roadmap with implementation details

### 2025-11-24: Integration & Documentation
- ✅ Updated `process_test_data` rule to use `expand_mixed()`
- ✅ Updated user documentation (`docs/user-guide/workflows.md`)
  - Added "Wildcard Expansion Types" section
  - Provided usage examples for `*` and `?`
  - Explained when to use each type
- ✅ Updated developer documentation (`docs/development/guides/claude-guide.md`)
  - Added technical details about wildcard expansion
  - Documented implementation architecture
  - Added references to `workflow_utils.py`

### Next Steps (Optional)
1. Write formal pytest tests for `workflow_utils.py` functions
2. Test with actual Snakemake workflow execution
3. Performance testing with large file sets
4. Consider future enhancements (regex patterns, caching)

---

## Open Questions

1. **Should `?` match any number of characters or just one?**
   - Shell glob: `?` = single character
   - Proposal: `?` = any number of characters (like `*` but filesystem-based)
   - Reasoning: More useful for patterns like `seed=5?` to match `5000`, `5001`, `5023`
   - **Decision**: Use `?` as "any characters" for semantic meaning, distinct from shell globbing

2. **How to handle parameters in different positions?**
   - `seed=5?` at end of args: easy to extract
   - `seed=5?` in middle: need careful regex
   - **Strategy**: Parse entire argument string, handle position-independent matching

3. **Should we cache glob results?**
   - Filesystem won't change during workflow execution (usually)
   - Caching would improve performance
   - But could cause issues with interrupted workflows
   - **Decision**: Cache within single Snakemake invocation, not across invocations

4. **What about directory structure?**
   - Models in: `models/{model_name}/`
   - Reports in: `reports/{data_loader}/`
   - Need to construct correct search paths based on target type
   - **Strategy**: Infer search directory from target path structure

---

## Risk Assessment

### Low Risk
- Backward compatibility: `*` expansion unchanged
- Performance: Only glob when `?` present

### Medium Risk
- Complex regex patterns for parameter extraction
- Edge cases with special characters in parameter values
- Integration with existing `expand()` calls in rules

### High Risk
- None identified yet

---

## Success Criteria

1. **Functional**:
   - `seed=5?` expands to only existing seed values starting with 5
   - `tau=*+seed=5?` correctly mixes config and filesystem expansion
   - No matches returns clear error message

2. **Performance**:
   - Filesystem globbing completes in < 1 second for typical project
   - No noticeable slowdown for workflows not using `?`

3. **Usability**:
   - Clear documentation with examples
   - Helpful error messages for common mistakes
   - Works seamlessly from command line

4. **Maintainability**:
   - Well-documented code with docstrings
   - Unit tests covering edge cases
   - Follows existing code patterns in `snake_utils.smk`

---

## Implementation Details

### Architecture: Option 3 (Hybrid Approach)

**Structure**:
```
dynvision/workflow/
├── workflow_utils.py          # Pure functions (minimal deps: stdlib + yaml)
│   └── Core logic for wildcard expansion
│
└── snake_utils.smk            # Snakemake integration
    ├── Import from workflow_utils
    ├── Wrapper functions with config integration
    └── Environment detection & execution commands
```

**Rationale**:
- Pure functions can be tested independently
- Minimal dependencies work in cluster minimal environment
- Clean separation of concerns
- Easy to maintain and extend

### Files Created/Modified

**New Files**:
1. `dynvision/workflow/workflow_utils.py` - Core utility functions with minimal dependencies

**Modified Files**:
1. `dynvision/workflow/snake_utils.smk`
   - Added imports from `workflow_utils.py`
   - Replaced duplicate functions with wrappers
   - Updated wildcard constraints to allow `?`

### Functions Implemented

**In `workflow_utils.py`**:
- `expand_filesystem_pattern()`: Core filesystem globbing and value extraction
- `args_product()`: Parameter combination generation with wildcard support
- `parse_arguments()`: Argument string parsing
- `replace_param_in_string()`: Parameter value replacement
- `dict_poped()`: Dictionary manipulation utility

**In `snake_utils.smk`**:
- `args_product()`: Wrapper that integrates with Snakemake config
- `parse_arguments()`: Wrapper that extracts from Snakemake wildcards
- `expand_mixed()`: Drop-in replacement for Snakemake's `expand()`

### Usage Examples

#### Example 1: Command-line usage - Analyze only completed runs

**Scenario**: You've trained models with seeds 5000, 5001, 5023, but seeds 5002-5022 failed.

**Before** (had to list all seeds manually or modify config):
```bash
snakemake reports/duration/duration_DyRCNNx8:tau=5_5000_imagenette_trained_all/test_data.csv
snakemake reports/duration/duration_DyRCNNx8:tau=5_5001_imagenette_trained_all/test_data.csv
snakemake reports/duration/duration_DyRCNNx8:tau=5_5023_imagenette_trained_all/test_data.csv
```

**After** (use ? wildcard):
```bash
snakemake reports/duration/duration_DyRCNNx8:tau=5_5?_imagenette_trained_all/test_data.csv
```

The `process_test_data` rule automatically finds and processes only existing files.

#### Example 2: Mixed wildcards - Config + Filesystem

**Scenario**: Test all tau values from config, but only completed seeds starting with 5.

```bash
snakemake reports/contrast/contrast_DyRCNNx8:tau=*_5?_imagenette_trained_all/test_data.csv
```

Expands to:
- `tau=*` → `tau=3`, `tau=5`, `tau=9` (from config)
- `seed=5?` → `5000`, `5001`, `5023` (from filesystem)

Result: 9 combinations (3 tau values × 3 seeds).

#### Example 3: In workflow rules (developer use)

When writing custom rules, use `expand_mixed()` instead of `expand()`:

```python
rule my_analysis:
    input:
        data = expand_mixed(
            project_paths.models / '{model}/{model}_{seed}_{data}_trained.pt',
            model='DyRCNNx8',
            seed='5?',  # Filesystem wildcard
            data='imagenette'
        )
    output:
        'reports/my_analysis.csv'
    shell:
        "python analyze.py {input.data} {output}"
```

#### Example 4: Pattern matching with prefixes

```python
# Different prefix patterns
seed='1?'    # Matches: 1000, 1001, 1234
seed='50?'   # Matches: 5000, 5001, 5023, 5099
seed='123?'  # Matches: 1230, 1231, 1234
```

#### Example 5: Checkpoint-generated files - Intermediate training checkpoints

**Scenario**: Analyze intermediate model checkpoints created during training.

The `intermediate_checkpoint_to_statedict` checkpoint rule converts Lightning checkpoints to state dicts. To analyze these checkpoint-generated files:

```bash
# Modify experiment config to use trained-epoch=? for status
# config_experiments.yaml:
#   responseintermediate:
#     status: trained-epoch=?  # or specific pattern like trained-epoch=4?

# Run workflow with checkpoint awareness
snakemake --config experiment=responseintermediate \
  --allowed-rules intermediate_checkpoint_to_statedict test_model process_test_data \
  reports/responseintermediate/responseintermediate_DyRCNNx8_5?_imagenette_trained-epoch=?_all/test_data.csv
```

**How it works**:
1. `process_test_data` uses input functions (`lambda w: expand_mixed(...)`)
2. Input functions defer filesystem globbing until runtime
3. After `intermediate_checkpoint_to_statedict` checkpoint completes, Snakemake re-evaluates input functions
4. `expand_mixed()` now sees newly created `trained-epoch={epoch}.pt` files
5. Workflow proceeds with discovered checkpoint files

**Note**: The input function pattern (`lambda w: expand_mixed(...)`) is essential for checkpoint awareness. Direct calls to `expand_mixed()` are evaluated at parse time and won't see checkpoint-generated files.

### Common Use Cases

**Resume interrupted parameter sweep**:
```bash
# Original sweep targeting all seeds from config
snakemake reports/duration/duration_DyRCNNx8:tau=*_*_imagenette_trained_all/test_data.csv

# Resume by targeting only existing files
snakemake reports/duration/duration_DyRCNNx8:tau=*_?_imagenette_trained_all/test_data.csv
```

**Exploratory analysis without config changes**:
```bash
# Quick check: only seeds 5000-5999
snakemake reports/contrast/contrast_Model:args_5?_data_trained_all/test_data.csv

# Compare two different seed ranges
snakemake reports/contrast/contrast_Model:args_5?_data_trained_all/test_data.csv
snakemake reports/contrast/contrast_Model:args_6?_data_trained_all/test_data.csv
```

**Analyze checkpoint-generated intermediate models**:
```bash
# Test all available intermediate training epochs
snakemake --config experiment=responseintermediate \
  --allowed-rules intermediate_checkpoint_to_statedict test_model process_test_data \
  reports/responseintermediate/responseintermediate_Model_seed_data_trained-epoch=?_all/test_data.csv

# Test specific epoch range (e.g., epochs 40-49, 140-149)
snakemake --config experiment=responseintermediate \
  reports/responseintermediate/responseintermediate_Model_seed_data_trained-epoch=?4?_all/test_data.csv
```

### Comparison: `*` vs `?`

| Feature | `*` (Config) | `?` (Filesystem) |
|---------|-------------|------------------|
| **Source** | YAML config | Existing files |
| **Use case** | Full parameter sweep | Partial/completed runs |
| **Example** | `tau=*` → 3,5,9 | `seed=5?` → 5000,5001 |
| **When to use** | Test all combinations | Resume interrupted work |
| **Requires** | Config file update | Files must exist |
| **Checkpoint support** | N/A | Yes (via input functions) |
| **Evaluation time** | Parse time | Runtime (when using input functions) |

### Troubleshooting

**Error: No files found**
```
ValueError: No files found matching pattern: models/DyRCNNx8/*_5*_imagenette_trained.pt
```

**Solutions**:
1. Verify files exist: `ls models/DyRCNNx8/*_5*_imagenette_trained.pt`
2. Check pattern syntax: `seed=5?` matches `seed=5`, `seed=50`, `seed=5000`, etc.
3. Use more specific pattern: `seed=50?` instead of `seed=5?`
4. Use dry-run to debug: `snakemake -n <target>` to see what Snakemake will attempt

**Unexpected matches**:
- `seed=5?` matches: 5, 50, 500, 5000, 5999, 59999 (any digits after 5)
- `seed=50?` matches: 500, 5000, 5001, 5099 (more specific)

**Best practices**:
1. Start specific, then generalize: Try `seed=500?` before `seed=5?`
2. Organize seeds by ranges: 5000-5999, 6000-6999 for easier filtering
3. Test with `ls` or `find` first to verify matches
4. Use `snakemake -n` dry-run to preview actions

---

## References

- [Snakemake Wildcards Documentation](https://snakemake.readthedocs.io/en/stable/snakefiles/rules.html#wildcards)
- [Snakemake glob_wildcards](https://snakemake.readthedocs.io/en/stable/snakefiles/rules.html#glob-wildcards)
- [Python glob module](https://docs.python.org/3/library/glob.html)
- AI Style Guide: `docs/development/guides/ai-style-guide.md`
- Claude Guide: `docs/development/guides/claude-guide.md`
- DynVision workflow structure: `dynvision/workflow/`
- Implementation: `dynvision/workflow/workflow_utils.py`
- Integration: `dynvision/workflow/snake_utils.smk`

---

## Notes

- This feature is inspired by shell glob patterns but with semantic meaning specific to DynVision's parameter system
- The `?` character was chosen to avoid conflict with other wildcard syntax
- The hybrid architecture ensures minimal environment compatibility for cluster execution
- Pure functions in `workflow_utils.py` can be tested independently
- Consider future enhancements: regex patterns, caching, better error messages
