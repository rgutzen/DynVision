# Cluster Integration Guide

This guide explains how to run DynVision workflows on high-performance computing (HPC) clusters. It covers setup, configuration, and best practices for efficient cluster execution.

## Overview

DynVision provides two methods for running workflows on HPC clusters:

1. **Basic Execution**: Submit a single job script that handles environment setup and runs the entire workflow sequentially.
2. **Advanced Execution**: Use Snakemake's built-in cluster capabilities to submit and manage individual jobs in parallel.

## Prerequisites

Before running workflows on a cluster, ensure:

1. **Cluster Access**
   You have access to a compute cluster and a directory with read and write access.

2. **Getting DynVision**
   Download or sync the DynVision folder into you cluster directory

3. **Data Access**
   If you are working with non-standard datasets that can be automatically downloaded within the workflow execution, make sure they are available.

4. **Path Mangement**
   Review your `dynvision.project_paths.py` file and set the alternative cluster paths according to your needs.

5. **Environment Setup**
   Follow you HPC documentation to setup and environment that has dynvision and all its dependencies installed. Depending on you system, this may for example also involve setting up a docker or singularity image.
   See the [Installation Guide](installation.md).

## Basic Execution (Single Job)

The basic method uses a single job script to run the entire workflow:

1. **Create Job Script**
   Use the provided template in `dynvision/cluster/snakejob_example.sh`:

   ```slurm
   #!/usr/bin/env bash
   #SBATCH -o /path/to/logs/slurm/%j.out
   #SBATCH -e /path/to/logs/slurm/%j.err
   #SBATCH --time=24:00:00
   #SBATCH --mem=80G
   #SBATCH --nodes=1
   #SBATCH --ntasks-per-node=1
   #SBATCH --cpus-per-task=16
   #SBATCH --gres=gpu:1"
   
   module purge
   
   # If using Singularity
   singularity exec --nv \
       --overlay /path/to/overlay.ext3:ro \
       /path/to/container.sif \
       bash -c "
   conda activate myenv
   
   cd ../workflow/
   snakemake \
       --cores 16 \
       --resources gpu=1 \
       --config \
           model_name=DyRCNNx4 \
           data_name=cifar100
   "
   ```

2. **Submit Job**
   ```bash
   sbatch snakejob_example.sh
   ```

3. **Monitor Job**

   ```bash
   squeue -u <user-name>
   ```

   It is also recommended to make use of the Weights & Biases integration to monitor progress online.

This method is simpler but less efficient for large workflows as it runs tasks sequentially.

## Advanced Execution (Parallel Jobs)

The advanced method uses Snakemake's cluster capabilities to run tasks in parallel:


1. Profile Configuration

   DynVision uses Snakemake's cluster profiles for job management:

   ```
   dynvision/cluster/
   ├── snakecharm.sh         # Wrapper script
   ├── _snakecharm.sh        # Core execution script
   └── profiles/
      └── slurm/            # SLURM cluster profile
         └── config.yaml   # Cluster settings
   ```

   The SLURM profile (`profiles/slurm/config.yaml`) defines default resources and rule-specific settings:

   ```yaml
   # General settings
   executor: slurm
   jobs: 150
   default-resources:
   runtime: 60
   mem_mb: 16000
   cpus_per_task: 16

   # Rule-specific resources
   set-resources:
   train_model:
      mem: 46000
      runtime: 1440
      gpu: 1
      constraint: "a100|h100"
      slurm_extra: "'--gres=gpu:1'"
   ```

   For more details see the [Snakemake Executor Documentation](https://snakemake.github.io/snakemake-plugin-catalog/plugins/executor/slurm.html).

2. **Executor settings**
   ```yaml
   # in config_defaults.yaml
   executor_start: "singularity exec --nv \
   --overlay /scratch/rg5022/images/rva.ext3:ro \
   --overlay /vast/work/public/ml-datasets/imagenet/imagenet-train.sqf:ro \
   --overlay /vast/work/public/ml-datasets/imagenet/imagenet-val.sqf:ro \
   --overlay /vast/work/public/ml-datasets/imagenet/imagenet-test.sqf:ro \
   /scratch/work/public/singularity/cuda12.2.2-cudnn8.9.4-devel-ubuntu22.04.3.sif \
   bash -c '\
   source /ext3/env.sh; \
   conda activate rva; \
   "
   executor_close: "'"
   ```

   For cluster executions any python call is wrapped in these executor settings:

   ```bash
   {executor_start}
   python script.py --arg1 value
   {executor_close}
   ```

   the above settings are an example, adapt them to your system.

3. **Executor Environment**
   This advanced execution may require the creation of an additional environment if the commands of the scheduler (e.g. SLURM) are not available in the main environment (e.g. because of using singularity).
   Therefore, create a new environment (let's call it 'snake-env') in the cluster's native filesystem (so that commands like `sbatch` and `srun` are available) and install snakemake (`pip install snakemake`).

4. **Adapt Executor Scripts**

   Adapt the executor script `dynvision/cluster/_snakecharm.sh` as needed to activate the environment created in **3.**
   ```bash
   module load anaconda3/2020.07  # conda needs to be available
   source activate snake-env  # environment that contains snakemake
   ```

5. **Use Snakecharm Wrapper**
   DynVision provides `snakecharm.sh` to manage cluster execution:

   ```bash
   # Navigate to DynVision directory
   cd dynvision
   
   # Run all experiments
   ./cluster/snakecharm.sh
   
   # Run specific experiment
   ./cluster/snakecharm.sh "--config experiment=duration model_name=DyRCNNx4 data_name=cifar100"
   ```

   The wrapper script:
   - Sets up the environment
   - Configures cluster parameters
   - Submits and monitors jobs
   - Manages log files

   You may pass any commandline arguments to the snakecharm.sh script as you would do with a regular snakemake command

6. **Monitor Execution**
   - Check logs in the configured `slurm-logdir`
   - Monitor job status with `squeue`
   - View detailed logs for each rule


## Environment Adaptation

DynVision automatically adapts to cluster environments:

1. **Environment Detection**
   ```python
   # In project_paths.py
   def iam_on_cluster(self):
       host_name = os.popen("hostname").read()
       cluster_names = ["hpc", "greene", "slurm", "compute"]
       return any(x in host_name for x in cluster_names)
   ```

2. **Path Management**
   - Large data directories move to scratch partitions
   - Logs redirect to appropriate locations
   - Container mounts configured automatically
   ```python
   # in project_paths.py
   if self.iam_on_cluster():
      # move large folders to scratch partition
      self.data.raw = Path("/scratch") / self.user_name / "data"
      self.models = (
            Path("/scratch") / self.user_name / self.project_name / "models"
      )
      self.reports = (
            Path("/scratch") / self.user_name / self.project_name / "reports"
      )
      self.large_logs = (
            Path("/scratch") / self.user_name / self.project_name / "logs"
      )
   ```

3. **Resource Scaling**
   - Batch sizes adjust for development vs. production
   - Progress bars disable on compute nodes
   - Memory limits scale based on available resources
   ```yaml
   # Debugging settings
   debug_batch_size: 3
   debug_check_val_every_n_epoch: 1
   debug_log_every_n_steps: 1
   debug_accumulate_grad_batches: 1
   debug_enable_progress_bar: True
   ```


## Related Resources

- [Snakemake Cluster Documentation](https://snakemake.readthedocs.io/en/stable/executing/cluster.html)
- [SLURM Documentation](https://slurm.schedmd.com/documentation.html)
- [DynVision Configuration Reference](../reference/configuration.md)
- [Workflow Organization](workflows.md)