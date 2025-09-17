#!/bin/bash
# Singularity executor wrapper for DynVision training
# Handles environment setup, distributed configuration, and container execution

set -eo pipefail  # Exit on error, preserve exit codes in pipes, but allow unset variables

# Configuration
readonly SCRIPT_NAME="$(basename "$0")"
readonly CONTAINER_IMAGE="/scratch/work/public/singularity/cuda12.2.2-cudnn8.9.4-devel-ubuntu22.04.3.sif"
readonly CONDA_ENV="rva"

# Overlay mounts - organized for clarity
readonly OVERLAYS=(
    "/scratch/rg5022/images/rva.ext3:ro"
    "/vast/work/public/ml-datasets/imagenet/imagenet-train.sqf:ro"
    "/vast/work/public/ml-datasets/imagenet/imagenet-val.sqf:ro"
    "/vast/work/public/ml-datasets/imagenet/imagenet-test.sqf:ro"
)

# Logging functions
log_info() {
    echo "[INFO] $*" >&2
}

log_error() {
    echo "[ERROR] $*" >&2
}

log_debug() {
    if [[ "${DEBUG:-false}" == "true" ]]; then
        echo "[DEBUG] $*" >&2
    fi
}

# Validation
validate_args() {
    if [[ $# -eq 0 ]]; then
        log_error "No command provided to $SCRIPT_NAME"
        echo "Usage: $SCRIPT_NAME <command> [args...]" >&2
        exit 1
    fi
}

# Distributed environment detection and setup
is_distributed_mode() {
    local world_size="${WORLD_SIZE:-1}"
    local use_distributed="${USE_DISTRIBUTED:-false}"
    
    [[ "$world_size" -gt 1 ]] || [[ "$use_distributed" == "true" ]]
}

setup_distributed_env() {
    if is_distributed_mode; then
        log_info "Distributed mode (WORLD_SIZE=${WORLD_SIZE:-1})"
        return 0
    fi
        
    # Set proper single-node values
    export WORLD_SIZE="1"
    export RANK="0" 
    export LOCAL_RANK="0"
    
    # Ensure single GPU visibility
    if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        export CUDA_VISIBLE_DEVICES="0"
    fi
    
    log_info "Single-node mode configured (WORLD_SIZE=1, RANK=0, LOCAL_RANK=0)"
}

# Environment variable collection for container
collect_env_vars() {
    local env_vars=(
        "WORLD_SIZE" "RANK" "LOCAL_RANK"
        "MASTER_ADDR" "MASTER_PORT" "USE_DISTRIBUTED" 
        "CUDA_VISIBLE_DEVICES"
        "NCCL_DEBUG" "NCCL_SOCKET_IFNAME" "NCCL_TIMEOUT"
    )
    
    for var in "${env_vars[@]}"; do
        echo "--env"
        echo "${var}=${!var:-}"
    done
}

# Build overlay arguments
build_overlay_args() {
    for overlay in "${OVERLAYS[@]}"; do
        echo "--overlay"
        echo "$overlay"
    done
}

# Container execution with environment setup
execute_container() {
    local cmd=("$@")
    
    # Collect environment and overlay arguments
    local env_args=()
    local overlay_args=()
    
    mapfile -t env_args < <(collect_env_vars)
    mapfile -t overlay_args < <(build_overlay_args)
    
    log_info "Executing container with command: ${cmd[*]}"
    
    # Execute singularity with all arguments - use exec to replace the shell process
    exec singularity exec --nv \
        "${env_args[@]}" \
        "${overlay_args[@]}" \
        "$CONTAINER_IMAGE" \
        bash -c '
            # Environment diagnostics
            echo "============= Container Environment ============="
            echo "WORLD_SIZE: ${WORLD_SIZE:-not_set}"
            echo "RANK: ${RANK:-not_set}"  
            echo "LOCAL_RANK: ${LOCAL_RANK:-not_set}"
            echo "MASTER_ADDR: ${MASTER_ADDR:-not_set}"
            echo "MASTER_PORT: ${MASTER_PORT:-not_set}"
            echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not_set}"
            if [[ "${WORLD_SIZE:-0}" -gt 1 ]]; then
                echo "NCCL_DEBUG: ${NCCL_DEBUG:-not_set}"
                echo "NCCL_SOCKET_IFNAME: ${NCCL_SOCKET_IFNAME:-not_set}"
                echo "NCCL_TIMEOUT: ${NCCL_TIMEOUT:-not_set}"
            fi
            echo "PWD: $PWD"
            echo "================================================"
            
            # Setup conda environment
            if ! source /ext3/env.sh; then
                echo "Failed to source /ext3/env.sh" >&2
                exit 1
            fi
            
            if ! conda activate '"$CONDA_ENV"'; then
                echo "Failed to activate conda environment: '"$CONDA_ENV"'" >&2
                exit 1
            fi
            
            echo "Conda environment: $CONDA_DEFAULT_ENV"
            echo "Python version: $(python --version)"
            
            # Execute the provided command
            exec "$@"
        ' -- "${cmd[@]}"
}

# Main execution flow
main() {
    validate_args "$@"
    setup_distributed_env
    execute_container "$@"
}

# Execute main function - no trap needed since we use exec
main "$@"