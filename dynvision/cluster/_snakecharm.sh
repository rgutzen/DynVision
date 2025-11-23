#!/usr/bin/env bash
# Cluster execution script for the snakemake workflow using a cluster profile (e.g. slurm). Is being wrapped by snakecharm.sh, so call snakecharm.sh instead of this script directly.

source ~/.bashrc
source ~/.bash_profile  # make sure custom commands are available
module load anaconda3/2020.07  # conda needs to be available
source activate snake-env  # environment that contains snakemake
source ./clean_env_vars.sh

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
    

# Process command line arguments
args=("$@")  # Start with all original arguments

# Check if --config exists, if not add it
if [[ ! " $* " =~ " --config " ]]; then
    args+=("--config")
fi

# Cluster execution auto-detected via environment variables (SLURM_JOB_ID, etc.)
# No config override needed - see docs/development/planning/cluster-execution.md

echo "Final command will be:"
echo "snakemake ${args[@]} --profile $current_dir/profiles/slurm"
echo ""

snakemake --unlock --cores=1
# snakemake --dry-run --printshellcmds "${args[@]}" --profile $current_dir/profiles/slurm --latency-wait 30
snakemake "${args[@]}" --profile $current_dir/profiles/slurm --latency-wait 30

echo -e '\n Snakecharm ended!'
