#!/usr/bin/env bash

# Load project paths from project_paths.py
slurm_logdir=$(python3 -c "
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join('..')))
from project_paths import project_paths
print(project_paths.logs / 'slurm')
")
echo "slurm log directory: $slurm_logdir"

# Ensure the log directory exists
mkdir -p "$slurm_logdir"

# Remove backslashes from arguments for the file name
if [[ "$@" == *"/"* ]]; then
    id=$(echo "$@" | tr '/' '-')
else
    id="$@"
fi

# Use nohup to run the script in the background and redirect both stdout and stderr to the log file
nohup bash -c "
sh ./_snakecharm.sh $@
" "$@" > "$slurm_logdir/snakecharm_$id.log" 2>&1 &

# nohup bash ./_snakecharm.sh "$@" > "$slurm_logdir/snakecharm_$id.log" 2>&1 &

pid=$!  # Process ID of the last background command (stop with `kill $pid`)
echo "PID: $pid" >> "$slurm_logdir/snakecharm_$id.log"

echo "Workflow is running in the background (PID: $pid). Check $slurm_logdir/snakecharm_$id.log for details."