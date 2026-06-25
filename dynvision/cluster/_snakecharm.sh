#!/usr/bin/env bash
# Cluster execution script for the snakemake workflow using a cluster profile (e.g. slurm).
# Is being wrapped by snakecharm.sh, so call snakecharm.sh instead of this script directly.

# Save args immediately before source/module commands clobber $@
args=("$@")
echo "DEBUG: $# args received"; printf 'DEBUG arg: [%s]\n' "${args[@]}"

source ~/.bashrc
# source ~/.bash_profile  # make sure custom commands are available
# module load anaconda3/2020.07  # conda needs to be available
module purge
# Load your conda module (adjust to your system)
module load <YOUR_CONDA_MODULE>  # e.g. anaconda3/2025.06
# module load bioinformatics/20251223  # disabled: using conda env's snakemake instead
# source activate snake-env  # environment that contains snakemake
conda activate <YOUR_SNAKEMAKE_ENV>  # e.g. /scratch/<user>/environments/snake-env

# Set paths
current_dir=$(pwd)
echo "current directory: $current_dir"

# Load project paths from project_paths.py
workflow_dir=$(python3 -c "
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join('..')))
from project_paths import project_paths
print(project_paths.scripts.workflow)
")

echo "workflow directory: $workflow_dir"
cd "$workflow_dir"

echo -n 'running workflow with snakemake version: '
snakemake --version

# Cluster execution auto-detected via environment variables (SLURM_JOB_ID, etc.)
# No config override needed - see docs/development/planning/cluster-execution.md

echo "Final command will be:"
echo "snakemake ${args[@]} --profile $current_dir/profiles/slurm --latency-wait 90 --resources disk_mb=300000"
echo ""

snakemake --unlock --cores=1
# snakemake --dry-run --printshellcmds "${args[@]}" --profile $current_dir/profiles/slurm --latency-wait 30
snakemake "${args[@]}" --profile $current_dir/profiles/slurm --latency-wait 90 --resources disk_mb=300000

echo -e '\n Snakecharm ended!'
