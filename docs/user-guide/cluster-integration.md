# Cluster Integration Guide

This guide explains how to run DynVision workflows on high-performance computing (HPC) clusters. It covers setup, configuration, and best practices for efficient cluster execution.

## Prerequisites

Before running workflows on a cluster:

1. **Environment Setup**
```bash
# Load required modules (example for typical HPC setup)
module load anaconda3/2020.07
source activate snake-env

# Verify environment
python -c "import snakemake; print(snakemake.__version__)"
```

2. **Data Access**
- Ensure datasets are accessible from compute nodes
- Check storage quotas and permissions
- Consider using mounted filesystems for large datasets

3. **Resource Planning**
- Estimate memory requirements per task
- Plan GPU allocation for training
- Consider storage needs for outputs

## Basic Cluster Execution

DynVision provides a wrapper script for cluster execution:

```bash
# Navigate to DynVision directory
cd dynvision

# Run all experiments using cluster profile
./cluster/snakecharm.sh -j100 all_experiments

# Run specific experiment
./cluster/snakecharm.sh -j100 experiment --config \
    model_name=DyRCNNx4 \
    data_name=cifar100
```

The wrapper script:
1. Sets up the environment
2. Configures cluster parameters
3. Submits and monitors jobs
4. Manages log files

## Cluster Configuration

DynVision uses Snakemake's cluster profiles for job management:

```
dynvision/cluster/
├── snakecharm.sh         # Wrapper script
├── _snakecharm.sh        # Core execution script
└── profiles/
    └── slurm/            # SLURM cluster profile
        └── config.yaml   # Cluster settings
```

### Basic Configuration

Example SLURM configuration (`profiles/slurm/config.yaml`):
```yaml
cluster:
  mkdir -p logs/slurm/{rule} &&
  sbatch
    --partition={resources.partition}
    --cpus-per-task={threads}
    --mem={resources.mem_mb}
    --time={resources.time}
    --output=logs/slurm/{rule}/{jobid}.out
    --error=logs/slurm/{rule}/{jobid}.err

default-resources:
  - partition=cpu
  - mem_mb=32000
  - time="24:00:00"
  - gpu=0
```

### Rule-Specific Resources

Configure resources per rule type:
```yaml
# In config.yaml
rule-specific-resources:
  train_model:
    - partition=gpu
    - mem_mb=64000
    - time="48:00:00"
    - gpu=1
  
  test_model:
    - partition=cpu
    - mem_mb=32000
    - time="24:00:00"
```

## Job Management

### Monitoring Jobs

Track job status:
```bash
# View all running jobs
squeue -u $USER

# Check specific job status
sacct -j <job_id> --format=JobID,State,Elapsed,MaxRSS,MaxVMSize

# View job output in real-time
tail -f logs/slurm/train_model/job_12345.out
```

### Resource Usage

Monitor resource utilization:
```bash
# Check memory usage
sstat -j <job_id> --format=MaxRSS,MaxVMSize

# Monitor GPU usage
nvidia-smi -l 1

# View partition limits
sinfo -o "%20P %5D %14F"
```

## Advanced Configuration

### Environment Detection

DynVision automatically adjusts settings for cluster environments:

```python
# In snake_runtime.smk
batch_size = config.batch_size if project_paths.iam_on_cluster() else 3
enable_progress_bar = not project_paths.iam_on_cluster()
```

### Data Management

Configure data paths for cluster environments:

```python
# In snake_utils.smk
def get_data_base_dir(wildcards):
    """Get dataset directory based on environment."""
    if wildcards.data_name in config.mounted_datasets and \
       project_paths.iam_on_cluster():
        return Path(f'/{wildcards.data_name}')
    return project_paths.data.raw / wildcards.data_name
```

## Troubleshooting

### Common Issues

1. **Environment Problems**
```bash
# Verify environment
which python
module list
module avail cuda  # For GPU support
```

2. **Resource Limits**
```bash
# Check partition limits
sinfo -o "%20P %5D %14F"

# Monitor memory usage
sstat -j <job_id> --format=MaxRSS,MaxVMSize
```

3. **Job Failures**
```bash
# Rerun failed jobs
./cluster/snakecharm.sh -j100 all_experiments --rerun-incomplete

# Debug specific rule
./cluster/snakecharm.sh train_model --debug

# Check error logs
less logs/slurm/train_model/job_12345.err
```

### Best Practices

1. **Resource Allocation**
   - Request appropriate resources per rule
   - Consider job duration and memory needs
   - Use GPU partitions only when needed

2. **Data Management**
   - Use fast storage for frequently accessed data
   - Clean up temporary files
   - Monitor disk quotas

3. **Job Organization**
   - Use meaningful job names
   - Maintain organized log directories
   - Monitor long-running jobs

## Further Reading

- [Snakemake Cluster Documentation](https://snakemake.readthedocs.io/en/stable/executing/cluster.html)
- [SLURM Documentation](https://slurm.schedmd.com/documentation.html)
- [DynVision Configuration Reference](../reference/configuration.md)