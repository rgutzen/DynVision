#!/bin/bash
# executor_wrapper_script.sh

# Check if arguments were provided
if [ $# -eq 0 ]; then
    echo "Error: No command provided to executor wrapper"
    exit 1
fi

singularity exec --nv \
    --env WORLD_SIZE="$WORLD_SIZE" \
    --env RANK="$RANK" \
    --env LOCAL_RANK="$LOCAL_RANK" \
    --env MASTER_ADDR="$MASTER_ADDR" \
    --env MASTER_PORT="$MASTER_PORT" \
    --env USE_DISTRIBUTED="$USE_DISTRIBUTED" \
    --env NCCL_DEBUG="$NCCL_DEBUG" \
    --env NCCL_SOCKET_IFNAME="$NCCL_SOCKET_IFNAME" \
    --env NCCL_TIMEOUT="$NCCL_TIMEOUT" \
    --overlay /scratch/rg5022/images/rva.ext3:ro \
    --overlay /vast/work/public/ml-datasets/imagenet/imagenet-train.sqf:ro \
    --overlay /vast/work/public/ml-datasets/imagenet/imagenet-val.sqf:ro \
    --overlay /vast/work/public/ml-datasets/imagenet/imagenet-test.sqf:ro \
    /scratch/work/public/singularity/cuda12.2.2-cudnn8.9.4-devel-ubuntu22.04.3.sif \
    bash -c '

    if [ "${WORLD_SIZE:-0}" -gt 1 ]; then
        echo "============= Container Environment Check =============";
        echo "WORLD_SIZE: ${WORLD_SIZE:-not_set}";
        echo "RANK: ${RANK:-not_set}";
        echo "LOCAL_RANK: ${LOCAL_RANK:-not_set}";
        echo "MASTER_ADDR: ${MASTER_ADDR:-not_set}";
        echo "MASTER_PORT: ${MASTER_PORT:-not_set}";
        echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not_set}";
        echo "NCCL_DEBUG: ${NCCL_DEBUG:-not_set}";
        echo "PWD: $PWD";
    fi
  
    # Setup environment
    source /ext3/env.sh || { echo "Failed to source /ext3/env.sh"; exit 1; }
    conda activate rva || { echo "Failed to activate conda environment"; exit 1; }

    echo "Conda environment: $CONDA_DEFAULT_ENV";
    echo "Python version: $(python --version)";

    exec "$@"' -- "$@"

exit_code=$?
echo "Exit code: $exit_code"
exit $exit_code