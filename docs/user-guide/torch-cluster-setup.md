# Torch Cluster Setup 
For general cluster concepts, see the [Cluster Integration Guide](https://github.com/rgutzen/DynVision/blob/dev/docs/user-guide/cluster-integration.md) in the DynVision documentation, and the [Torch Documentation](https://services.rt.nyu.edu/docs/hpc/getting_started/intro/).

## Torch Cluster Overview

- **H200 nodes**: 8 GPUs/node, 128 cores, ~2TB RAM
- **L40S nodes**: 4 GPUs/node, 128 cores, ~512GB RAM
- **Do not specify partitions manually** --- SLURM auto-assigns based on requested resources
- Account: `torch_pr_628_general` (check with `sacctmgr show associations user=$USER`)

## Step 1: Create the snake-env

This lightweight environment lives on the native filesystem (outside any container) and only contains snakemake + the SLURM plugin.

```bash
mkdir /scratch/$USER/environments/
module load anaconda3/2025.06
conda create -p /scratch/$USER/environments/snake-env \
    -c conda-forge -c bioconda \
    python=3.12 snakemake snakemake-executor-plugin-slurm -y
```

> Use Python 3.12, not newer. Python 3.13+ has compatibility issues with some snakemake dependencies.

## Step 2: Configure `_snakecharm.sh`

Edit `dynvision/cluster/_snakecharm.sh`. The environment activation block should look like:

```bash
source ~/.bashrc
module purge
module load anaconda3/2025.06
conda activate /scratch/$USER/environments/snake-env
```

- Use `conda activate`, never `source activate`

## Step 3: Configure `executor_wrapper.sh`

Edit `dynvision/cluster/executor_wrapper.sh` and set the paths at the top:

```bash
readonly CONTAINER_IMAGE="/share/apps/images/cuda12.2.2-cudnn8.9.4-devel-ubuntu22.04.3.sif"
readonly CONDA_ENV="rva"

readonly OVERLAYS=(
    "/scratch/$USER/images/rva.ext3:ro"  # or where you have stored you singularity image
)
```

This script is called automatically for each SLURM job. It launches Singularity, mounts overlays, activates the conda environment inside the container, and runs the rule's command.

## Step 4: Configure SLURM Profile

Edit `dynvision/cluster/profiles/slurm/config.yaml`:

- Set `slurm-logdir` to your log path
- Set `slurm_account` to your account in both `default-resources` and each entry under `set-resources`

## Step 5: Run

```bash
cd DynVision/dynvision/cluster
snakecharm.sh                                          # run all defaults
snakecharm.sh manuscript_figures --config test_batch=32 # specific target
snakecharm.sh -n                                       # dry run
```

> Don't put the arguments after snakecharm.sh in quotes as it was done on Greene!

## Migration Notes (Greene/Previous Cluster -> Torch)

### Runtime values are in seconds

The snakemake SLURM executor plugin divides `runtime` by 60 before passing to SLURM's `--time`. So config values must be in **seconds**:

| Intended time | Config value | SLURM receives |
|---|---|---|
| 30 minutes | `runtime: 1800` | `--time=30` |
| 2 hours | `runtime: 7200` | `--time=120` |
| 13 hours | `runtime: 48000` | `--time=800` |

If your jobs are timing out immediately (e.g., after 1 minute), check that you didn't carry over minute-based values from the old config.

### GPU allocation: use `--gpus-per-node`, not `gpu` or `--gres`

Torch does **not** support SLURM's `--gpus` flag (job-level GPU allocation), which is what the snakemake plugin generates from the `gpu` resource key. It also blocks `--gres` in `slurm_extra`. The workaround is to use `--gpus-per-node` via `slurm_extra`:

```yaml
# Old Greene config (causes error on Torch)
gpu: 1
tasks_per_gpu: 0
# or
slurm_extra: "'--gres=gpu:1'"

# New Torch config
slurm_extra: "'--gpus-per-node=1'"
```

For distributed training with multiple GPUs, combine flags in a single `slurm_extra`:
```yaml
slurm_extra: "'--gpus-per-node=4 --ntasks-per-node=4'"
```

Do **not** use the `gpu`, `tasks_per_gpu`, or `cpus_per_gpu` resource keys --- they all trigger the unsupported `--gpus` flag.

### No `constraint` for GPU types

Torch has H200 and L40S GPUs. The old `constraint: "a100|h100"` is invalid here. Remove `constraint` entries --- SLURM auto-assigns appropriate hardware.

### Don't specify partitions

Torch auto-assigns partitions. Do not set `slurm_partition` in the profile.

### Argument passing (no outer quotes)

Pass snakecharm arguments as separate shell words:

```bash
# Correct
./cluster/snakecharm.sh manuscript_figures --config test_batch=32

# Wrong (was required on some older setups)
./cluster/snakecharm.sh "manuscript_figures --config test_batch=32"
```

### Snakemake 9.x CLI changes

Some CLI flags changed in snakemake 9.x:

| Old (snakemake 7/8) | New (snakemake 9) |
|---|---|
| `--dryrun` / `-n` | `-n` still works, also `-e dryrun` |
| `--touch` | `-e touch` |

### `conda activate` instead of `source activate`

`source activate` is deprecated and may cause `command not found` errors. Always use `conda activate`.

### VS Code Remote Connection
Add these settings to maintain a stable SSH connection during VS Code Server installation:

**In VS Code settings** (`settings.json`):
```json
"remote.SSH.connectTimeout": 120
```

**In your SSH config** (typically `~/.ssh/config` under your Torch profile):
```
ServerAliveInterval 120
```

These settings prevent SSH timeout while VS Code Server installs in your cluster home directory.