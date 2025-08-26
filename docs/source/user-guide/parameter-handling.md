# Parameter Handling

DynVision uses a sophisticated parameter management system that ensures type safety, validation, and consistency across all experimental workflows. This system integrates configuration files, command-line interfaces, and runtime validation to provide a robust foundation for reproducible research.

## Overview

The parameter handling system follows a four-layer architecture that progressively refines and validates neural network parameters:

1. **Configuration Layer**: YAML configuration files with operational mode management
2. **Validation Layer**: Pydantic-based type checking and constraint enforcement  
3. **Composition Layer**: Script-specific parameter combinations with computed properties
4. **Runtime Layer**: Model, trainer, and dataloader instantiation with validated parameters

This architecture separates configuration management from model implementation, ensuring flexibility and maintainability while providing comprehensive validation and automatic parameter derivation.

## Parameter Flow Architecture

### Complete Parameter Flow with Precedence

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Config Files  │    │ Snakemake CLI    │    │ ConfigMode      │
│   (Priority 1)  │───▶│ --config param=X │───▶│ Manager         │
│                 │    │ (Priority 2)     │    │ (Mode Override) │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                        │
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Python Script   │    │ Pydantic Classes │    │ Model/Trainer   │
│ CLI Arguments   │◀───│ Validation &     │◀───│ Component       │
│ (Priority 3)    │    │ Derivation       │    │ Instantiation   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                       ┌─────────────────┐
                       │ Direct Override │
                       │ kwargs          │
                       │ (Priority 4)    │
                       └─────────────────┘
```

**Parameter Precedence (lowest to highest priority):**
1. **YAML Configuration Files**: Base parameter values from multiple config files
2. **Snakemake CLI Arguments**: `snakemake --config param=value` overrides config files
3. **Python Script CLI Arguments**: Arguments passed to individual scripts (e.g., `train_model.py --learning_rate 0.002`)
4. **Direct Override kwargs**: Programmatic parameter overrides (highest priority)

### Key Distinction: Snakemake CLI vs Python Script CLI

- **Snakemake CLI**: `snakemake train_model --config learning_rate=0.002` - Sets parameters in Snakemake's config namespace
- **Python Script CLI**: Arguments passed to the actual Python script within a Snakemake rule via shell commands
- The Snakemake config values become shell command arguments that are then parsed by Pydantic classes

## Configuration Mode System

### How Config Modes Work

The `ConfigModeManager` provides automatic parameter adjustment based on operational contexts. Modes are defined in `config_modes.yaml` and can be enabled explicitly or through auto-detection:

```yaml
# config_modes.yaml
use_debug_mode: auto          # Auto-detect based on log_level and epochs
use_large_dataset_mode: auto  # Auto-detect based on data_name
use_distributed_mode: false   # Explicitly disabled

debug:
  log_level: "DEBUG"
  epochs: 5
  batch_size: 8
  store_responses: 10

large_dataset:
  use_ffcv: true
  batch_size: 128
  accumulate_grad_batches: 4

distributed:
  strategy: "ddp"
  precision: "16-mixed"
  sync_batchnorm: true
```

### Mode Detection Logic

The system automatically detects appropriate modes based on context:

- **Debug Mode**: Triggered when `log_level="DEBUG"` or `epochs <= 5`
- **Large Dataset Mode**: Activated for datasets like ImageNet, COCO, or OpenImages
- **Distributed Mode**: Must be explicitly enabled (`use_distributed_mode: true`)

When a mode is active, its parameter overrides are applied before Pydantic validation, allowing for context-appropriate parameter adjustments.

## Parameter Classes Architecture

### Component Hierarchy

DynVision organizes parameters into a hierarchical structure:

- **BaseParams**: Foundation class with common functionality (CLI parsing, config loading, alias resolution)
- **Component Classes**: Specialized parameter groups
  - **ModelParams**: Neural architecture, biological parameters, optimizer configuration
  - **TrainerParams**: PyTorch Lightning trainer settings, system configuration
  - **DataParams**: Dataset specification, data loading, preprocessing options
- **Composite Classes**: Script-specific combinations
  - **TrainingParams**: ModelParams + TrainerParams + DataParams + training-specific paths
  - **InitParams**: ModelParams + minimal DataParams for model initialization
  - **TestingParams**: ModelParams + DataParams for evaluation

### Computed Properties and Validation

Parameter classes automatically derive additional values and perform context-aware validation:

- **Biological Feasibility**: Checks for realistic neural time constants, integration steps, and delays
- **Cross-Component Consistency**: Validates parameter compatibility between model, trainer, and data components
- **Automatic Scaling**: Adjusts learning rates and batch sizes for distributed training
- **Derived Parameters**: Computes delay timesteps, stability ratios, and effective batch sizes

## Actionable Interaction Points

### How to Change a Parameter Value

**Method 1: Configuration File**
Add or modify parameter in the appropriate config file:
```yaml
# In config_defaults.yaml or config_experiments.yaml
learning_rate: 0.002
batch_size: 64
```

**Method 2: Snakemake CLI Override**
```bash
snakemake train_model --config learning_rate=0.002 batch_size=64
```

**Method 3: Script-Specific Arguments**
Parameters can also be passed directly to Python scripts through Snakemake rule definitions.

### How to Add a New Parameter

**Step 1: Add to Appropriate Parameter Class**
```python
# In dynvision/hyperparameters/model_params.py
class ModelParams(BaseParams):
    # Add your new parameter
    custom_parameter: float = Field(
        default=1.0, 
        description="Description of the parameter",
        gt=0.0  # Validation constraint
    )
```

**Step 2: Add Validation (if needed)**
```python
@field_validator("custom_parameter")
def validate_custom_parameter(cls, v):
    if v > 10.0:
        raise ValueError("custom_parameter should not exceed 10.0")
    return v
```

**Step 3: Use in Model Classes**
The parameter becomes automatically available in model kwargs through `get_model_kwargs()`.

### How to Add a Parameter Alias

**Step 1: Add to Class Aliases**
```python
# In the appropriate parameter class
@classmethod
def get_aliases(cls) -> Dict[str, str]:
    aliases = super().get_aliases()
    aliases.update({
        "custom": "custom_parameter",  # alias -> full_name
        "cp": "custom_parameter",
    })
    return aliases
```

**Step 2: Use Alias in CLI or Config**
```bash
snakemake train_model --config custom=2.0  # Instead of custom_parameter=2.0
```

### How to Add a Derived Parameter

**Add as Property to Parameter Class**
```python
class ModelParams(BaseParams):
    @property
    def derived_value(self) -> float:
        """Compute derived value from base parameters."""
        return self.custom_parameter * self.learning_rate
    
    def get_computation_summary(self) -> Dict[str, Any]:
        """Get summary including derived parameters."""
        return {
            "base": {"custom_parameter": self.custom_parameter},
            "derived": {"derived_value": self.derived_value},
        }
```

### How to Add a Config Mode

**Step 1: Define Mode Detection**
```python
# In dynvision/hyperparameters/config_mode_manager.py
class ConfigModeManager:
    def _detect_custom_mode(self) -> bool:
        """Auto-detect if custom mode should be enabled."""
        return self.config.get("model_name") == "CustomModel"
```

**Step 2: Add Mode Configuration**
```yaml
# In config_modes.yaml
use_custom_mode: auto

custom:
  learning_rate: 0.005
  batch_size: 16
  precision: "16-mixed"
```

**Step 3: Use Mode**
The mode will be automatically applied when detection conditions are met, or can be explicitly enabled with `use_custom_mode: true`.

### How to Add Cross-Component Validation

**Add to Composite Parameter Class**
```python
# In dynvision/hyperparameters/training_params.py
class TrainingParams(BaseParams):
    @model_validator(mode="after")
    def validate_custom_constraints(self) -> "TrainingParams":
        # Example: Ensure batch size is compatible with model timesteps
        if self.data.batch_size > self.model.n_timesteps:
            logger.warning(
                f"Batch size ({self.data.batch_size}) exceeds timesteps "
                f"({self.model.n_timesteps}) - may cause memory issues"
            )
        
        return self
```

## Common Usage Patterns

### Training with Parameter Modifications

**Quick Parameter Override**
```bash
snakemake train_model --config \
    model_name=DyRCNNx4 \
    learning_rate=0.002 \
    epochs=50
```

**Complex Parameter Combinations**
```bash
snakemake train_model --config \
    model_args="{rctype:full,lr:0.001,tsteps:20}" \
    trainer_args="{epochs:100,devices:2}"
```

### Direct Script Usage

For advanced users who want to bypass Snakemake:

```python
from dynvision.params import TrainingParams

# Load with automatic validation and mode detection
params = TrainingParams.from_cli_and_config()

# Apply scaling for distributed training
params.apply_parameter_scaling()

# Create model with validated parameters
model = create_model(**params.model.get_model_kwargs())
```

### Parameter Inspection and Debugging

```python
# Get comprehensive parameter summary
timing_info = params.model.get_timing_summary()

# Export complete configuration for reproducibility
params.export_full_config("experiment_config.yaml")
```

## Integration with Existing Components

### Snakemake Workflow Integration

The parameter system integrates seamlessly with Snakemake workflows. Configuration modes are applied during workflow initialization, and parameter validation occurs within individual rule executions. This ensures consistent parameter handling across the entire experimental pipeline.

### PyTorch Lightning Integration

Validated parameters are automatically filtered and passed to PyTorch Lightning components. The system handles parameter translation between DynVision's biological parameter names and PyTorch Lightning's expected arguments.

### Model Class Integration

Existing model classes continue to work through the `@alias_kwargs` decorator system. New parameter classes provide additional type safety and validation while maintaining backward compatibility with existing model implementations.

## Benefits and Design Principles

### Type Safety and Validation
- **Comprehensive validation**: Parameter types, ranges, and constraints are enforced at parse time
- **Context-aware rules**: Different validation logic for initialization, training, and testing scenarios
- **Biological plausibility**: Specialized validation for neuroscience parameters and constraints

### Flexibility and Composability
- **Modular design**: Parameters are organized into logical components that can be mixed and matched
- **Computed properties**: Derived parameters are calculated automatically based on base parameters
- **Mode-aware configuration**: Operational modes automatically adjust parameters for different contexts

### Reproducibility and Maintainability
- **Complete configuration export**: All parameters (base, derived, and computed) are saved for reproducibility
- **Parameter precedence tracking**: Clear hierarchy shows which source provided each parameter value
- **Extensible architecture**: New parameters, validations, and modes can be added without breaking existing code

This parameter handling system provides a robust foundation for managing the complexity of neural network research while maintaining the flexibility needed for experimental workflows.