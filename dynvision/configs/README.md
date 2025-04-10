# Configuration System

This directory contains the configuration files for the rhythmic visual attention project. The configuration system uses a hierarchical structure with clear precedence rules and validation.

## Configuration Files

The configuration files are loaded in the following order (later files override earlier ones):

1. `config_defaults.yaml`: Base configuration with default values
2. `config_data.yaml`: Dataset-specific configurations
3. `config_workflow.yaml`: Workflow execution parameters
4. `config_experiments.yaml`: Experiment-specific settings
5. `config_visualization.yaml`: (Optional) Visualization parameters

## Usage Examples

### Basic Training Configuration
```yaml
# config_workflow.yaml
model_name: DyRCNNx4
seed: "0001"
model_args:
  rctype: ['self', 'full']
  dt: 2
  tau: 8
  tff: 12
  trc: 6
data_name: cifar10
data_group: "all"
```

## Configuration Validation

The configuration system includes validation to ensure:
- Required parameters are present
- Parameter types are correct
- Values are within valid ranges
- Dependencies between parameters are satisfied

### Validation Rules
1. Required Parameters:
   - model_name
   - data_name
   - seed
   - experiment

2. Type Validation:
   - Numeric parameters (e.g., learning_rate) must be numbers
   - List parameters must be lists
   - String parameters must be strings

3. Value Validation:
   - Learning rate must be positive
   - Batch size must be positive
   - Epochs must be positive
   - Time parameters must be non-negative

4. Dependency Validation:
   - Dataset must exist if specified
   - Model architecture must exist
   - Experiment type must be valid

## Adding New Configurations

To add new configuration parameters:

1. Add default values to `config_defaults.yaml` or to the config file that seems most suitable
2. Document the parameters in this README
3. Update the validation schema if needed
4. Test the configuration with different values
