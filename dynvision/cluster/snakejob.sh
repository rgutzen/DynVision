#!/usr/bin/env bash
#SBATCH -o <YOUR_PROJECT_PATH>/logs/slurm/%j.out
#SBATCH -e <YOUR_PROJECT_PATH>/logs/slurm/%j.err
#SBATCH --time=2:00:00
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --job-name=dynvision
#SBATCH --mail-type=END,FAIL
#SBATCH --gres=gpu:1
#SBATCH --account=<YOUR_SLURM_ACCOUNT>

module purge

singularity exec --nv \ 
    --overlay <YOUR_OVERLAY_PATH>/rva.ext3:ro \
    --overlay /projects/work/public/ml-datasets/imagenet/imagenet-train.sqf:ro \
    --overlay /projects/work/public/ml-datasets/imagenet/imagenet-val.sqf:ro \
    --overlay /projects/work/public/ml-datasets/imagenet/imagenet-test.sqf:ro \
    <YOUR_CONTAINER_IMAGE_PATH> \
    bash -c \
"
# source /ext3/env.sh
source /ext3/miniforge3/etc/profile.d/conda.sh
conda activate <YOUR_CONDA_ENV>

cd ../workflow
snakemake --unlock --cores=1
snakemake $@ \
          --jobname '{jobid}.{rulename}' \
	      --latency-wait 90 \
          --keep-going \
          --nolock \
          --rerun-incomplete \
          --cores=16 \
          --jobs=1 \
          --resources gpu=1 cpu=16 

exit
"