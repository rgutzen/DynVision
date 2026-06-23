#!/usr/bin/env bash
#SBATCH -o /home/rg5022/Modeling_Dynamical_Vision/logs/slurm/%j.out
#SBATCH -e /home/rg5022/Modeling_Dynamical_Vision/logs/slurm/%j.err
#SBATCH --time=2:00:00
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --job-name=rva
#SBATCH --mail-type=END,FAIL
#SBATCH --gres=gpu:1
#SBATCH --account=torch_pr_628_general

module purge

singularity exec --nv \ 
    --overlay /scratch/rg5022/images/rva.ext3:ro \
    --overlay /projects/work/public/ml-datasets/imagenet/imagenet-train.sqf:ro \
    --overlay /projects/work/public/ml-datasets/imagenet/imagenet-val.sqf:ro \
    --overlay /projects/work/public/ml-datasets/imagenet/imagenet-test.sqf:ro \
    /share/apps/images/cuda12.2.2-cudnn8.9.4-devel-ubuntu22.04.3.sif \
    bash -c \
"
# source /ext3/env.sh
source /ext3/miniforge3/etc/profile.d/conda.sh
conda activate rva

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
    # --overlay /vast/work/public/ml-datasets/imagenet/imagenet-train.sqf:ro \
    # --overlay /vast/work/public/ml-datasets/imagenet/imagenet-val.sqf:ro \
    # --overlay /vast/work/public/ml-datasets/imagenet/imagenet-test.sqf:ro \