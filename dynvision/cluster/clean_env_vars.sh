#!/bin/bash

# Quick fix script for PyTorch Lightning rank environment variable issue
# Run this before executing DynVision scripts

# List of environment variables that can cause issues
rank_vars=("RANK" "LOCAL_RANK" "NODE_RANK" "SLURM_PROCID" "SLURM_LOCALID" "WORLD_SIZE" "LOCAL_WORLD_SIZE")

# Clean up empty variables
for var in "${rank_vars[@]}"; do
    if [[ -v $var ]] && [[ -z "${!var}" ]]; then
        echo "   Unsetting empty variable: $var"
        unset "$var"
    fi
done

# Set single-node defaults for local execution if not in SLURM environment
if [[ -z "$SLURM_JOB_ID" ]]; then
    export RANK=${RANK:-0}
    export LOCAL_RANK=${LOCAL_RANK:-0} 
fi

# Test import
# echo "🧪 Testing PyTorch Lightning import..."
# python3 -c "
# try:
#     import pytorch_lightning as pl
#     print(f'✅ Successfully imported PyTorch Lightning v{pl.__version__}')
# except Exception as e:
#     print(f'❌ {e}')
#     exit(1)
# "
