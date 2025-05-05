# Workflow Management with Snakemake

DynVision uses Snakemake to manage complex workflows. This guide explains how to use Snakemake to run experiments, parameter sweeps, scale and port between environments with DynVision.

## Introduction to Snakemake

Snakemake is a workflow management system that allows for the creation of reproducible and scalable data analyses. It is based on a Python-like language that defines rules to create output files from input files.

In DynVision, Snakemake is used to:
- Organize computational pipelines into discrete steps
- Automatically determine dependencies between tasks
- Enable parameter sweeps 
- Provide consistent execution across different environments

## DynVision's Workflow Structure

DynVision's workflow system is organized into several Snakemake files:

```
dynvision/workflow/
├── Snakefile              # Main entry point
├── snake_utils.smk        # Utility functions and configuration
├── snake_data.smk         # Data preparation and processing
├── snake_runtime.smk      # Model training and evaluation
├── snake_experiments.smk  # Complex testing scenarios
└── snake_visualizations.smk # Result visualization and analysis
```

Each file contains rules that define specific parts of the overall workflow:

1. **Snakefile**: The main entry point that includes the other files and defines the top-level targets.
2. **snake_utils.smk**: Utility functions, path management, and configuration loading.
3. **snake_data.smk**: Rules for dataset acquisition, organization, and preprocessing.
4. **snake_runtime.smk**: Rules for model initialization, training, and evaluation.
5. **snake_experiments.smk**: Rules for running suites of tests
5. **snake_visualizations.smk**: Rules for visualizing model responses and analyzing results.

## Basic Workflow Commands

### Running a Single Experiment

To run a single experiment with a specific model and dataset:

```bash
# Navigate to the DynVision directory
cd dynvision

# Run an experiment with default parameters
snakemake -j1 <project_paths.reports>/'experiment_<experiment_name>.done'
```

which is equivalent to

```bash
snakemake -j1 --config experiment=<experiment_name>
```

### Specifying Model and Parameters

You can override default parameters using the `--config` option:

```bash
# Train a DyRCNNx4 model with full recurrence on CIFAR100
snakemake -j1 experiment --config model_name=DyRCNNx4 data_name=cifar100 model_args="{rctype:full}"
```

### Running Multiple Experiments

To run multiple experiments defined in the configuration:

```bash
# Run all experiments defined in config_experiments.yaml
snakemake -j1
```

### Visualizing Results

To generate visualizations for experiments:

```bash
# Generate plots for all experiments
snakemake -j1 plot_experiments_on_models
```

## Parameter Sweeps

DynVision makes it easy to run parameter sweeps by specifying multiple values for parameters:

```bash
# Run experiments with different recurrence types
snakemake -j4 all_experiments --config experiment=contrast model_args="{rctype:[full,self,depthpointwise,pointdepthwise]}"
```

This will run the contrast experiment for each recurrence type in parallel.

## Understanding Workflow Rules

Let's examine some of the key Snakemake rules in DynVision:

### Data Preparation Rules

```python
rule get_data:
    """Download and prepare standard datasets."""
    input:
        script = SCRIPTS / 'data' / 'get_data.py'
    params:
        raw_data_path = lambda w: project_paths.data.raw / f'{w.data_name}',
        # Additional parameters...
    output:
        flag = directory(project_paths.data.raw \
            / '{data_name}' \
            / '{data_subset}' )
    shell:
        """
        python {input.script:q} \
            --output {params.output:q} \
            --data_name {params.data_name} \
            # Additional parameters...
        """
```

This rule downloads and prepares a standard dataset for use in experiments.

### Model Initialization Rule

```python
rule init_model:
    """Initialize a model with specified configuration."""
    input:
        script = SCRIPTS / 'runtime' / 'init_model.py',
        dataset = project_paths.data.interim \
            / '{data_name}' \
            / 'train_all' \
            / 'folder.link'
    params:
        config_path = CONFIGS,
        model_arguments = lambda w: parse_arguments(w, 'model_args'),
        init_with_pretrained = config.init_with_pretrained,
    output:
        model_state = project_paths.models \
            / '{model_name}' \
            / '{model_name}{model_args}_{seed}_{data_name}_init.pt'
    shell:
        """
        python {input.script:q} \
            # Command-line arguments...
        """
```

This rule initializes a model with the specified configuration.

### Model Training Rule

```python
rule train_model:
    """Train a model on specified dataset."""
    input:
        model_state = project_paths.models \
            / '{model_name}' \
            / '{model_name}{model_args}_{seed}_{data_name}_init.pt',
        dataset_train = project_paths.data.processed \
            / '{data_name}' \
            / 'train_all' \
            / 'train.beton',
        dataset_val = project_paths.data.processed \
            / '{data_name}' \
            / 'train_all' \
            / 'val.beton',
        script = SCRIPTS / 'runtime' / 'train_model.py'
    params:
        # Various parameters...
    output:
        model_state = project_paths.models \
            / '{model_name}' \
            / '{model_name}{model_args}_{seed}_{data_name}_trained.pt'
    shell:
        """
        python {input.script:q} \
            # Command-line arguments...
        """
```

This rule trains a model on the specified dataset.

### Model Evaluation Rule

```python
rule test_model:
    """Evaluate a trained model on test data."""
    input:
        model_state = project_paths.models \
            / '{model_name}' \
            / '{model_name}{model_args}_{seed}_{data_name}_{status}.pt',
        dataset = project_paths.data.interim \
            / '{data_name}' \
            / 'test_{data_group}' \
            / 'folder.link',
        script = SCRIPTS / 'runtime' / 'test_model.py'
    params:
        # Various parameters...
    output:
        responses = project_paths.models \
            / '{model_name}' \
            / '{model_name}{model_args}_{seed}_{data_name}_{status}_{data_loader}{data_args}_{data_group}_test_responses.pt',
        results = project_paths.reports \
            / '{model_name}' \
            / '{model_name}{model_args}_{seed}_{data_name}_{status}_{data_loader}{data_args}_{data_group}_test_outputs.csv'
    shell:
        """
        python {input.script:q} \
            # Command-line arguments...
        """
```

This rule evaluates a trained model on test data.

## Working with Wildcards

Snakemake uses wildcards to generalize rules. In DynVision, wildcards are extensively used to enable flexible workflows. Common wildcards include:

- `{model_name}`: Name of the model (e.g., DyRCNNx4, AlexNet)
- `{data_name}`: Name of the dataset (e.g., cifar100, mnist)
- `{model_args}`: Model arguments (e.g., `:rctype=full+tsteps=20`)
- `{data_loader}`: Data loader name (e.g., StimulusDuration)
- `{data_args}`: Data loader arguments (e.g., `:tsteps=100+stim=5`)
- `{data_group}`: Named subset of classes for testing
- `{seed}`: Random seed for reproducibility
- `{status}`: Either `init` or `trained` 
- `{experiment}`: Experiment name (e.g., contrast, duration)

These wildcards are used to specify which files to generate and how to connect the different steps of the workflow.

## Configuring Workflows

DynVision workflows are configured through YAML files and command-line overrides:

1. **YAML Configuration**:
   - `config_defaults.yaml`: Default parameters for all components
   - `config_data.yaml`: Dataset-specific configurations
   - `config_workflow.yaml`: Workflow execution parameters
   - `config_experiments.yaml`: Experiment-specific settings

2. **Command-Line Overrides**:
   - Directly override parameters with `--config key=value`
   - Specify complex parameters using Python-like syntax: `--config model_args="{rctype:full}"`

See the [Configuration Reference](../reference/configuration.md) for detailed information about configuration parameters.

## Output Organization

DynVision organizes workflow outputs according to a consistent pattern:

- **Models**: `/models/{model_name}/{model_name}{model_args}_{seed}_{data_name}_{status}.pt`
- **Responses**: `/models/{model_name}/{model_name}{model_args}_{seed}_{data_name}_{status}_{data_loader}{data_args}_{data_group}_test_responses.pt`
- **Results**: `/reports/{model_name}/{model_name}{model_args}_{seed}_{data_name}_{status}_{data_loader}{data_args}_{data_group}_test_outputs.csv`
- **Figures**: `/reports/figures/{experiment}/{experiment}_{model_name}{model_args}_{seed}_{data_name}_{status}_{data_group}/{plot}.png`

This organization ensures that outputs can be easily located and associated with the parameters that generated them.

## Running Workflows on Clusters

DynVision provides robust support for executing workflows on high-performance computing (HPC) clusters, enabling efficient parallel processing of large-scale experiments. This section explains how to configure and run DynVision workflows on cluster environments.

### Prerequisites

Before running workflows on a cluster, ensure:

1. **Environment Setup**:
   ```bash
   # Load required modules (example for typical HPC setup)
   module load anaconda3/2020.07
   
   # Activate the conda environment with Snakemake
   source activate snake-env
   
   # Verify environment
   python -c "import snakemake; print(snakemake.__version__)"
   ```

2. **Data Access**:
   - Verify that input datasets are accessible from compute nodes
   - Check that output directories have sufficient permissions
   - Consider using mounted filesystems for large datasets

3. **Resource Requirements**:
   - Estimate memory needs for different workflow stages
   - Plan GPU requirements for training tasks
   - Consider storage requirements for experiment outputs

For more details about environment setup, see the [Installation Guide](installation.md).

### Cluster Configuration

DynVision uses Snakemake's cluster profiles for efficient job management. The configuration is located in `dynvision/cluster/`:

```
dynvision/cluster/
├── snakecharm.sh         # Main cluster execution wrapper
├── _snakecharm.sh        # Core cluster execution script
└── profiles/
    └── slurm/            # SLURM cluster profile
        └── config.yaml   # Cluster-specific settings
```

The SLURM profile (`config.yaml`) defines default resource requirements:

```yaml
# Example SLURM profile configuration
cluster:
  mkdir -p logs/slurm/{rule} &&
  sbatch
    --partition={resources.partition}
    --cpus-per-task={threads}
    --mem={resources.mem_mb}
    --time={resources.time}
    --output=logs/slurm/{rule}/{jobid}.out
    --error=logs/slurm/{rule}/{jobid}.err
    --gres=gpu:{resources.gpu}  # For GPU jobs

# Default resources for all rules
default-resources:
  - partition=cpu
  - mem_mb=32000
  - time="24:00:00"
  - gpu=0

# Rule-specific resource overrides
rule-specific-resources:
  train_model:
    - partition=gpu
    - mem_mb=64000
    - time="48:00:00"
    - gpu=1
```

### Basic Cluster Execution

To run workflows on a cluster using the provided SLURM profile:

```bash
# Navigate to the DynVision directory
cd dynvision

# Run all experiments with 100 parallel jobs
./cluster/snakecharm.sh -j100 all_experiments

# Run specific experiment with cluster configuration
./cluster/snakecharm.sh -j100 experiment --config \
    model_name=DyRCNNx4 \
    data_name=cifar100 \
    model_args="{rctype:full}"
```

The `snakecharm.sh` wrapper script handles:
1. Environment setup and activation
2. Cluster-specific parameter configuration
3. Job submission and monitoring
4. Log file management and organization

### Resource Management

DynVision automatically manages computational resources through Snakemake's cluster integration:

1. **Rule-Specific Resources**:
   ```python
   rule train_model:
       resources:
           partition="gpu",     # GPU partition for training
           mem_mb=64000,       # 64GB memory
           time="48:00:00",    # 48-hour time limit
           gpu=1               # Request 1 GPU
   ```

2. **Automatic Scaling**:
   - CPU/Memory allocation based on rule requirements
   - GPU scheduling for training tasks
   - Job dependencies and parallel execution
   - Automatic job resubmission on failure

3. **Environment Detection**:
   ```python
   # Automatically adjust settings for cluster environment
   batch_size = config.batch_size if project_paths.iam_on_cluster() else 3
   enable_progress_bar = not project_paths.iam_on_cluster()
   ```

### Job Monitoring and Logging

The workflow automatically organizes logs for easy monitoring:

```bash
# Log directory structure
logs/
└── slurm/
    ├── train_model/           # Logs for training jobs
    │   ├── job_12345.out     # Standard output
    │   └── job_12345.err     # Error messages
    ├── test_model/           # Logs for evaluation jobs
    └── snakecharm_exp1.log   # Workflow execution log
```

Monitor jobs using standard cluster commands:
```bash
# View all running jobs
squeue -u $USER

# Check specific job status
sacct -j <job_id> --format=JobID,State,Elapsed,MaxRSS,MaxVMSize

# View job output in real-time
tail -f logs/slurm/train_model/job_12345.out

# Monitor GPU usage
nvidia-smi
```

### Troubleshooting Cluster Execution

Common issues and solutions:

1. **Environment Problems**:
   ```bash
   # Verify environment activation
   which python
   python -c "import snakemake; print(snakemake.__version__)"
   
   # Check module availability
   module list
   module avail cuda  # For GPU support
   ```

2. **Resource Limits**:
   ```bash
   # Monitor memory usage
   sstat -j <job_id> --format=MaxRSS,MaxVMSize
   
   # Check GPU utilization
   nvidia-smi -l 1
   
   # View partition limits
   sinfo -o "%20P %5D %14F"
   ```

3. **Job Failures**:
   ```bash
   # Rerun failed jobs
   ./cluster/snakecharm.sh -j100 all_experiments --rerun-incomplete
   
   # Debug specific rule
   ./cluster/snakecharm.sh -j1 train_model --debug
   
   # Check job error logs
   less logs/slurm/train_model/job_12345.err
   ```

4. **Data Access**:
   ```bash
   # Verify paths and permissions
   ls -l /path/to/data
   df -h /path/to/output  # Check disk space
   
   # Test file system mounts
   cd /path/to/mounted/data && touch test.txt
   ```

For more advanced cluster usage, see:
- [Snakemake Cluster Documentation](../development/snakemake.md#cluster-execution)
- [SLURM User Guide](https://slurm.schedmd.com/documentation.html)
- [GPU Computing Guide](../reference/configuration.md#gpu-configuration)

## Advanced Workflow Usage

### Dry Runs

To see what Snakemake would do without actually executing commands:

```bash
snakemake -n all_experiments
```

### Creating Workflow Graphs

Generate a visual representation of the workflow:

```bash
snakemake --dag all_experiments | dot -Tpdf > workflow.pdf
```

### Resuming Interrupted Workflows

If a workflow is interrupted, you can resume from where it left off:

```bash
snakemake -j1 all_experiments --rerun-incomplete
```

### Force Rerunning Rules

To force Snakemake to rerun a particular rule:

```bash
snakemake -j1 all_experiments --forcerun test_model
```

## Custom Workflow Extensions

You can extend DynVision's workflows by adding new rules to the existing Snakemake files or creating new ones.

### Adding a New Experiment Type

To add a new experiment type:

1. Define the experiment in `config_experiments.yaml`:
   ```yaml
   experiment_config:
     my_experiment:
       status: trained
       parameter: my_parameter
       data_loader: MyDataLoader
       data_args:
         tsteps: 100
         my_parameter: [1, 2, 3, 4]
   ```

2. Create the corresponding data loader in `dataloader.py`

3. Add visualization rules if needed

### Creating Custom Analysis Workflows

You can create custom analysis workflows by defining new rules:

```python
rule my_custom_analysis:
    input:
        responses = project_paths.models \
            / '{model_name}' \
            / '{model_name}{model_args}_{seed}_{data_name}_{status}_{data_loader}{data_args}_{data_group}_test_responses.pt',
        script = SCRIPTS / 'analysis' / 'my_custom_analysis.py'
    output:
        results = project_paths.reports \
            / 'analysis' \
            / '{model_name}' \
            / 'my_custom_analysis_{model_name}{model_args}_{seed}_{data_name}_{status}_{data_group}.csv'
    shell:
        """
        python {input.script:q} \
            --responses {input.responses:q} \
            --output {output.results:q}
        """
```

## Common Workflow Patterns

### Initializing and Training Multiple Models

```bash
# Initialize and train models with different recurrence types
snakemake -j4 --config \
  model_name=DyRCNNx4 \
  model_args="{rctype: [full, self, depthpointwise, pointdepthwise]}" \
  data_name=cifar100 \
  seed=0001 \
  expand(project_paths.models / '{model_name}' / '{model_name}{model_args}_{seed}_{data_name}_trained.pt', \
    model_name=config.model_name, \
    model_args=args_product(config.model_args), \
    seed=config.seed, \
    data_name=config.data_name)
```

### Running a Cross-Validation Experiment

```bash
# Run 5-fold cross-validation
for i in {1..5}; do
  snakemake -j1 experiment --config \
    model_name=DyRCNNx4 \
    seed=000$i \
    experiment=contrast
done
```

### Comparing Pre-trained Models

```bash
# Test standard pre-trained models on a dataset
snakemake -j4 --config \
  model_name="[AlexNet,CorNetRT,ResNet18,DyRCNNx4]" \
  data_name=cifar100 \
  data_group=invertebrates \
  experiment=contrast \
  test_standard_models
```

## Troubleshooting Workflows

### Missing Input Files

If Snakemake reports missing input files, check:
1. If the dataset has been downloaded (`get_data` rule)
2. If the data paths are correct in `project_paths.py`
3. If all required symbolic links have been created

### Rule Execution Errors

If a rule fails to execute:
1. Check the error message in the log file
2. Ensure all dependencies are installed
3. Try running the individual script with the same parameters

### Performance Issues

If workflows are running slowly:
1. Enable FFCV data loading with `use_ffcv: True`
2. Adjust the number of threads (`-j` parameter)
3. Use mixed precision training with `precision: "16-mixed"`

## Further Reading

For more information about Snakemake, see the [official Snakemake documentation](https://snakemake.readthedocs.io/).
