# Parameter Processing System - Developer Guide

> **Note**: This document replaces the previous `mode-specific-parameters-*.md` files with an updated, comprehensive guide to the parameter processing system.

## Overview

The DynVision parameter processing system handles the complex flow of configuration data from YAML files through the Snakemake workflow to runtime script execution. It implements a three-level precedence hierarchy (Source → Scope → Alias) to ensure predictable parameter resolution.

## Architecture

### System Components

```
┌────────────────────────────────────────────────────────────────┐
│                    Configuration Layer                          │
│  config_defaults.yaml → config_data.yaml → config_experiments   │
│  (Later files override earlier files)                           │
└──────────────────────┬─────────────────────────────────────────┘
                       │
                       ▼
┌────────────────────────────────────────────────────────────────┐
│                  ConfigHandler (snake_utils.smk)                │
│  • Applies mode overrides (debug, large_dataset, distributed)   │
│  • Injects wildcards from Snakemake                             │
│  • Generates per-rule runtime config files                      │
└──────────────────────┬─────────────────────────────────────────┘
                       │
                       ▼
┌────────────────────────────────────────────────────────────────┐
│           Runtime Scripts (init_model.py, etc.)                 │
│  • Receives config_path pointing to runtime config              │
│  • Receives CLI args from Snakemake shell command               │
│  • Calls CompositeParams.from_cli_and_config()                  │
└──────────────────────┬─────────────────────────────────────────┘
                       │
                       ▼
┌────────────────────────────────────────────────────────────────┐
│            CompositeParams (composite_params.py)                │
│  • Separates config and CLI parameter sources                   │
│  • Applies scope-aware precedence within each source            │
│  • Merges with CLI taking priority                              │
│  • Routes parameters to component classes                       │
└──────────────────────┬─────────────────────────────────────────┘
                       │
                       ▼
┌────────────────────────────────────────────────────────────────┐
│         Component Instantiation (ModelParams, etc.)             │
│  • Pydantic validation of parameters                            │
│  • Type checking and constraint enforcement                     │
│  • Computed properties and cross-validation                     │
└────────────────────────────────────────────────────────────────┘
```

## Three-Level Precedence Hierarchy

### Level 1: Source Precedence (Primary)

**Rule: CLI arguments always beat config file values**

This is the highest-level precedence rule. Regardless of how a parameter is scoped or aliased, if it comes from the CLI it will override the same parameter from the config.

**Implementation** (`_separate_component_configs_two_sources`):
```python
# Separate config params (scoped > unscoped within config)
config_components = cls._separate_single_source(config_params)

# Separate CLI params (scoped > unscoped within CLI)
cli_components = cls._separate_single_source(cli_params)

# Merge: CLI always wins over config
for comp_name in component_classes:
    final_configs[comp_name] = composite_base.copy()
    final_configs[comp_name].update(config_components.get(comp_name, {}))
    final_configs[comp_name].update(cli_components.get(comp_name, {}))  # CLI wins
```

**Examples:**
- `--seed 42` (CLI) beats `seed: 100` (config)
- `--model_name X` (CLI unscoped) beats `model.model_name: Y` (config scoped)

### Level 2: Scope Precedence (Secondary)

**Rule: Within each source, more specific scope beats less specific**

Scoped parameters use dot notation to target specific components or modes:
- **3-part keys**: `mode.component.param` (e.g., `init.model.store_responses`)
- **2-part keys**: `component.param` or `mode.param` (e.g., `model.model_name`, `init.batch_size`)
- **1-part keys**: `param` (unscoped, applies to all matching components)

**Implementation** (`_separate_single_source`):
```python
# Phase 1: Classify parameters by scope
for key, value in params.items():
    parts = key.split(".")
    
    if len(parts) == 3:  # mode.component.param
        if mode and parts[0] == mode and parts[1] in component_classes:
            level_2_mode_component[parts[1]][parts[2]] = value
            
    elif len(parts) == 2:  # component.param or mode.param
        if mode and parts[0] == mode:
            level_3_mode[parts[1]] = value
        elif parts[0] in component_classes:
            level_4_component[parts[0]][parts[1]] = value
            
    else:  # Unscoped
        if key in base_fields:
            composite_base[key] = value
        level_5_base[key] = value  # Also add to base for routing

# Phase 2: Apply precedence (higher level overrides lower)
for comp_name in component_classes:
    comp_config = {}
    
    # Level 5: Unscoped (lowest)
    comp_config.update({k: v for k, v in level_5_base.items() 
                       if k in comp_fields and (comp_name, k) not in explicitly_scoped})
    
    # Level 4: Component-scoped
    comp_config.update(level_4_component[comp_name])
    
    # Level 3: Mode-scoped
    comp_config.update({k: v for k, v in level_3_mode.items() if k in comp_fields})
    
    # Level 2: Mode+Component-scoped (highest)
    comp_config.update(level_2_mode_component[comp_name])
```

**Examples:**
- Within config: `model.model_name: X` beats `model_name: Y`
- Within CLI: `--model.batch_size 32` beats `--batch_size 64`
- Mode-specific: `init.model.store_responses: 0` beats `model.store_responses: 100`

### Level 3: Alias Precedence (Tertiary)

**Rule: Within same source and scope, short aliases beat long forms**

Aliases provide convenient short-forms for commonly used parameters. When both an alias and its target exist at the same scope level within the same source, the alias takes precedence.

**Implementation** (`_resolve_aliases_with_precedence`):
```python
# Group parameters by scope level (number of dots)
by_scope = {0: {}, 1: {}, 2: {}}  # 0=unscoped, 1=one dot, 2=two dots
for key, value in params.items():
    scope_level = key.count('.')
    by_scope[scope_level][key] = value

# Resolve aliases within each scope level
for scope_level in [0, 1, 2]:
    scope_params = by_scope[scope_level]
    
    for alias, full_name in aliases.items():
        # Only process if both alias and full_name are at this scope level
        if alias.count('.') == scope_level and full_name.count('.') == scope_level:
            if alias in scope_params:
                # Alias exists - use it for the full name
                scope_params[full_name] = scope_params[alias]
                del scope_params[alias]

# Merge back with higher scope levels overriding lower ones
resolved = {}
resolved.update(by_scope[0])  # Unscoped
resolved.update(by_scope[1])  # One dot
resolved.update(by_scope[2])  # Two dots (highest)
```

**Examples:**
- `tff: 100` beats `t_feedforward: 200` (both unscoped in same source)
- `model.tff: 100` beats `model.t_feedforward: 200` (both scoped to model)
- But: `model.t_feedforward: 100` beats `tff: 200` (scope precedence applies first)

**Common Aliases:**
- `tff` → `t_feedforward`
- `trc` → `t_recurrent`
- `bs` → `batch_size`
- `lr` → `learning_rate`

## Shared Fields Across Components

### Problem Statement

Some parameters like `seed` and `log_level` are defined in multiple Pydantic classes:
- `BaseParams` has `seed` and `log_level`
- `ModelParams` has `seed` and `log_level`
- `DataParams` has `seed` and `log_level`
- `TrainerParams` has `seed`

When a user provides `--seed 42`, this value should propagate to all components that accept it.

### Solution

**Phase 1: Dual Routing**

Single-part keys are added to BOTH `composite_base` AND `level_5_base`:

```python
# Single-part key (Level 5 or composite base)
if key in base_fields:
    composite_base[key] = value

# Also add to level_5_base for component routing
level_5_base[key] = value
```

This ensures shared fields can route to components while also being stored at the composite level.

**Phase 2: Component Merging**

When merging config and CLI sources, shared fields from `composite_base` are applied first:

```python
for comp_name in component_classes:
    # Start with composite base (includes shared fields)
    final_configs[comp_name] = composite_base.copy()
    
    # Add config values
    final_configs[comp_name].update(config_components.get(comp_name, {}))
    
    # Override with CLI values
    final_configs[comp_name].update(cli_components.get(comp_name, {}))
```

**Component-Specific Overrides:**

Users can provide different values for different components:

```bash
# All components get seed=1, except model gets 42, data gets 99
python script.py --seed 1 --model.seed 42 --data.seed 99
```

This works because:
1. Unscoped `--seed 1` goes to all components via composite_base
2. Scoped `--model.seed 42` overrides the model component specifically
3. Scoped `--data.seed 99` overrides the data component specifically

## Mode-Specific Parameter Overrides

### Mode System Overview

Modes provide context-specific parameter adjustments without modifying base configuration files. They're particularly useful for:
- **Debug mode**: Reduce dataset size, store more responses, increase logging
- **Large dataset mode**: Enable FFCV, adjust batch sizes, configure memory optimization
- **Distributed mode**: Set DDP strategy, configure gradient accumulation, enable sync_batchnorm

### Mode Configuration Structure

Modes are configured in `config_modes.yaml` with two sections:

1. **Mode Activation** (`use_*_mode`):
```yaml
use_debug_mode: auto          # Auto-detect or explicit true/false
use_large_dataset_mode: auto
use_distributed_mode: false
```

2. **Mode Parameters**:
```yaml
debug:
  log_level: "DEBUG"
  epochs: 5
  init:
    data:
      batch_size: 8
    model:
      store_responses: 10
  test:
    model:
      store_responses: 100
```

### Mode-Scoped Parameters

Mode-scoped parameters use the pattern `mode.component.param` and only apply when that mode is active:

```yaml
# In init mode, model stores 10 responses
init.model.store_responses: 10

# In test mode, model stores 100 responses
test.model.store_responses: 100
```

The mode name comes from the `CompositeParams` subclass:
```python
class InitParams(CompositeParams):
    mode_name: ClassVar[str] = "init"  # Enables init.* parameters

class TestingParams(CompositeParams):
    mode_name: ClassVar[str] = "test"  # Enables test.* parameters
```

### ConfigHandler Integration

The `ConfigHandler` applies mode overrides during `process_configs()`:

```python
def process_configs(self, config: Any, wildcards: Any = None) -> Path:
    """Process configuration with modes and wildcards for a specific rule."""
    
    # Add wildcards to config
    working_config = self._add_wildcards_to_config(config, wildcards)
    
    # Apply active modes
    working_config = self._apply_modes(working_config)
    
    # Write runtime config file
    config_path = self.config_dir / self._generate_unique_filename(wildcards)
    with open(config_path, 'w') as f:
        yaml.dump(working_config, f)
    
    return config_path
```

Mode parameters are merged into the config before it's saved, so when the runtime script loads it, mode overrides are already applied.

## Implementation Details

### Class Hierarchy

```
BaseParams
├── ModelParams
├── TrainerParams  
├── DataParams
└── CompositeParams
    ├── InitParams (mode_name="init")
    ├── TrainingParams (mode_name="train")
    └── TestingParams (mode_name="test")
```

### Key Methods

#### `from_cli_and_config` (CompositeParams)

Entry point for parameter resolution:

```python
@classmethod
def from_cli_and_config(
    cls,
    config_path: Optional[str] = None,
    override_kwargs: Optional[Dict[str, Any]] = None,
    args: Optional[List[str]] = None,
) -> "CompositeParams":
    # Get config and CLI params separately
    config_params, cli_params = cls._get_config_and_cli_params_separate(
        config_path=config_path,
        override_kwargs=override_kwargs,
        args=args
    )
    
    # Separate and merge with proper precedence
    separated = cls._separate_component_configs_two_sources(
        config_params=config_params,
        cli_params=cli_params
    )
    
    return cls(**separated)
```

#### `_separate_single_source` (CompositeParams)

Processes a single parameter source with scope-aware precedence:

```python
@classmethod
def _separate_single_source(
    cls, params: Dict[str, Any]
) -> Dict[str, Dict[str, Any]]:
    # Resolve aliases with scope precedence
    params = cls._resolve_aliases_with_precedence(params)
    
    # Phase 1: Classify parameters by scope
    # ... (see Scope Precedence section)
    
    # Phase 2: Apply precedence hierarchy
    # ... (higher levels override lower)
    
    # Phase 3: Handle unscoped parameters
    # ... (route to matching components)
    
    # Phase 4: Add composite base fields
    # ... (propagate shared fields)
    
    return component_configs
```

#### `_resolve_aliases_with_precedence` (CompositeParams)

Scope-aware alias resolution:

```python
@classmethod
def _resolve_aliases_with_precedence(
    cls, params: Dict[str, Any]
) -> Dict[str, Any]:
    aliases = cls.get_aliases()
    
    # Group by scope level
    by_scope = {0: {}, 1: {}, 2: {}}
    for key, value in params.items():
        scope_level = key.count('.')
        by_scope[scope_level][key] = value
    
    # Resolve aliases within each scope level
    for scope_level in [0, 1, 2]:
        scope_params = by_scope[scope_level]
        
        for alias, full_name in aliases.items():
            if (alias.count('.') == scope_level and 
                full_name.count('.') == scope_level and
                alias in scope_params):
                scope_params[full_name] = scope_params[alias]
                del scope_params[alias]
    
    # Merge with scope precedence
    resolved = {}
    resolved.update(by_scope[0])  # Unscoped
    resolved.update(by_scope[1])  # One dot
    resolved.update(by_scope[2])  # Two dots
    
    return resolved
```

## Type Coercion for CLI Arguments

### The Problem

CLI arguments are always strings. When Snakemake passes wildcards or when users provide values via command line:

```bash
python script.py --seed 42 --batch_size 32
```

Both `"42"` and `"32"` arrive as strings, but Pydantic expects `int` types.

### The Solution

**Pydantic v2 `strict=False`** enables automatic type coercion:

```python
class BaseParams(BaseModel):
    seed: int = Field(description="Random seed")
    
    model_config = ConfigDict(
        strict=False,  # ✅ Allows "42" → 42 coercion
        # ... other config
    )
```

With `strict=False`:
- String `"42"` → `int` 42
- String `"3.14"` → `float` 3.14
- String `"true"` → `bool` True
- String `"false"` → `bool` False

### Config Path Extraction

When `config_path` comes from CLI args (Snakemake scenario):

```python
# _get_config_and_cli_params_separate extracts it first
if args:
    cli_args = cls._parse_cli_args(args)
    if not config_path and "config_path" in cli_args:
        config_path = cli_args.pop("config_path")  # Extract for loading
    else:
        cli_args.pop("config_path", None)  # Remove if already provided
```

This allows runtime scripts to work with both:
- Direct parameter: `InitParams.from_cli_and_config(config_path="config.yaml")`
- CLI argument: `InitParams.from_cli_and_config(args=["--config_path", "config.yaml"])`

## Snakemake Integration

### Workflow Pattern

1. **Rule Definition** (`snake_runtime.smk`):
```python
rule init_model:
    params:
        config_path = lambda w: process_configs(config, wildcards=w),
    shell:
        """
        python {input.script} \\
            --config_path {params.config_path} \\
            --model_name {wildcards.model_name} \\
            --seed {wildcards.seed} \\
            --output {output.model_state}
        """
```

2. **Config Processing** (`process_configs`):
   - Merges all YAML config files
   - Applies mode overrides
   - Injects wildcards
   - Writes runtime config file

3. **Script Execution** (`init_model.py`):
```python
def main():
    # Loads runtime config + CLI args
    params = InitParams.from_cli_and_config()
    
    # Use validated parameters
    model = initialize_model(params.model)
```

### Wildcard Injection

Wildcards from Snakemake rules are added to the config by `ConfigHandler`:

```python
def _add_wildcards_to_config(self, config: Dict, wildcards: Any) -> Dict:
    config["wildcards"] = {}
    
    for attr_name in dir(wildcards):
        if not attr_name.startswith("_"):
            attr_value = getattr(wildcards, attr_name)
            if isinstance(attr_value, (str, int, float)):
                config["wildcards"][attr_name] = attr_value
                config[attr_name] = attr_value  # Also add to top level
    
    return config
```

This allows wildcards to participate in the precedence system as config values.

## Testing and Validation

### Unit Tests

Test the precedence rules:

```python
def test_cli_beats_config():
    """Test that CLI arguments override config values."""
    config_params = {'model_name': 'ConfigValue'}
    cli_params = {'model_name': 'CliValue'}
    
    params = InitParams.from_cli_and_config(
        config_path='config.yaml',
        override_kwargs=cli_params
    )
    
    assert params.model.model_name == 'CliValue'

def test_scoped_beats_unscoped():
    """Test that scoped params beat unscoped within same source."""
    params = InitParams.from_cli_and_config(
        config_path='config.yaml',
        override_kwargs={
            'model_name': 'Unscoped',
            'model.model_name': 'Scoped',
        }
    )
    
    assert params.model.model_name == 'Scoped'

def test_shared_fields_propagate():
    """Test that shared fields propagate to all components."""
    params = InitParams.from_cli_and_config(
        config_path='config.yaml',
        override_kwargs={'seed': 42}
    )
    
    assert params.seed == 42
    assert params.model.seed == 42
    assert params.data.seed == 42
```

### Integration Tests

Test the complete workflow:

```python
def test_snakemake_workflow():
    """Test parameter flow through Snakemake workflow."""
    # 1. Simulate ConfigHandler creating runtime config
    config_handler = ConfigHandler(base_config, project_paths)
    runtime_config_path = config_handler.process_configs(
        config, wildcards={'seed': 123, 'model_name': 'DyRCNNx8'}
    )
    
    # 2. Simulate script receiving config + CLI args
    params = InitParams.from_cli_and_config(
        config_path=runtime_config_path,
        override_kwargs={
            'seed': 123,
            'model_name': 'DyRCNNx8',
            'output': '/tmp/model.pt',
        }
    )
    
    # 3. Verify correct resolution
    assert params.model.model_name == 'DyRCNNx8'
    assert params.seed == 123
```

## Best Practices

### For Users

1. **Use config files for defaults**: Put all standard parameters in `config_defaults.yaml`
2. **Use scoped parameters for clarity**: `model.learning_rate` is clearer than relying on routing
3. **Use CLI for run-specific values**: Seeds, output paths, and experiment-specific overrides
4. **Use modes for context**: Debug, large dataset, distributed modes handle common scenarios
5. **CLI args are always strings**: All CLI arguments are parsed as strings but automatically coerced to the correct type (e.g., `--seed 42` becomes `int(42)`)

### For Developers

1. **Enable type coercion**: Set `strict=False` in `model_config` to allow automatic type coercion from strings (CLI args) to typed fields
2. **Define shared fields consistently**: If a field appears in multiple classes, document it
3. **Use ClassVar for mode_name**: Prevents it from being treated as a Pydantic field
4. **Test precedence rules**: Ensure new parameter classes respect the three-level hierarchy
5. **Document aliases**: Make it clear which short-forms map to which long-forms
6. **Validate cross-component consistency**: Use Pydantic validators to check parameter compatibility
7. **Handle config_path in CLI**: The system automatically extracts `--config_path` from CLI args when present

## Common Pitfalls

### Pitfall 1: Expecting unscoped to override scoped within same source

**Wrong assumption:**
```bash
# User expects model_name to win because it comes "later"
python script.py --model.model_name X --model_name Y
```

**Actual behavior:** `model.model_name = X` (scoped beats unscoped)

### Pitfall 2: Forgetting source precedence beats scope

**Wrong assumption:**
```yaml
# config.yaml
model.model_name: ConfigScoped  # User expects this to beat CLI unscoped
```

```bash
python script.py --config config.yaml --model_name CliUnscoped
```

**Actual behavior:** `model.model_name = CliUnscoped` (CLI source beats config, regardless of scope)

### Pitfall 3: Not propagating shared fields

**Wrong implementation:**
```python
# Missing the dual routing for shared fields
if key in base_fields:
    composite_base[key] = value
else:
    level_5_base[key] = value  # ❌ Should be unconditional
```

**Correct implementation:**
```python
if key in base_fields:
    composite_base[key] = value

# Always add to level_5_base for component routing
level_5_base[key] = value  # ✅
```

### Pitfall 4: Hard-coding defaults in Pydantic classes

**Wrong:**
```python
class ModelParams(BaseParams):
    learning_rate: float = 0.001  # ❌ Hard-coded default
```

**Right:**
```python
class ModelParams(BaseParams):
    learning_rate: float  # ✅ No default, forces config to provide it

# In config_defaults.yaml:
learning_rate: 0.001
```

## Migration from Old System

If migrating from the old mode-specific parameter system:

1. **Update mode_name to ClassVar:**
```python
# Old
class InitParams(CompositeParams):
    mode_name = "init"  # ❌ Pydantic treats this as a field

# New  
class InitParams(CompositeParams):
    mode_name: ClassVar[str] = "init"  # ✅ Class variable
```

2. **Use two-source separation:**
```python
# Old (single merged dict)
params = cls.get_params_from_cli_and_config(...)
separated = cls._separate_component_configs(params)

# New (config and CLI kept separate)
config_params, cli_params = cls._get_config_and_cli_params_separate(...)
separated = cls._separate_component_configs_two_sources(
    config_params=config_params,
    cli_params=cli_params
)
```

3. **Add shared field dual routing:**
```python
# Add to Phase 1 of _separate_single_source
if key in base_fields:
    composite_base[key] = value

# Always add to level_5_base (new)
level_5_base[key] = value
```

## Summary

The parameter processing system implements a **Source → Scope → Alias** precedence hierarchy:

1. **Source**: CLI beats config (always)
2. **Scope**: More specific beats less specific (within each source)
3. **Alias**: Short form beats long form (within same source+scope)

This provides predictable, testable parameter resolution that scales from simple single-parameter overrides to complex multi-source, multi-component configurations. The system handles shared fields across components, mode-specific overrides, and integration with Snakemake workflows while maintaining type safety through Pydantic validation.
