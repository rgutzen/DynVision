#!/usr/bin/env bash
#SBATCH -o /home/rg5022/rhythmic_visual_attention/logs/slurm/%j.out
#SBATCH -e /home/rg5022/rhythmic_visual_attention/logs/slurm/%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=80G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --job-name=rva
#SBATCH --mail-type=END,FAIL
#SBATCH --gres=gpu:1 -C "a100|h100"

module purge

singularity exec --nv \
    --overlay /scratch/rg5022/images/rva.ext3:ro \
    --overlay /vast/work/public/ml-datasets/imagenet/imagenet-train.sqf:ro \
    --overlay /vast/work/public/ml-datasets/imagenet/imagenet-val.sqf:ro \
    --overlay /vast/work/public/ml-datasets/imagenet/imagenet-test.sqf:ro \
    /scratch/work/public/singularity/cuda12.2.2-cudnn8.9.4-devel-ubuntu22.04.3.sif \
    bash -c \
"
source /ext3/env.sh
conda activate rva

cd ../workflow
snakemake --unlock --cores=1
snakemake \
          --jobname '{jobid}.{rulename}' \
	      --latency-wait 90 \
          --keep-going \
          --nolock \
          --rerun-incomplete \
          --cores=16 \
          --jobs=1 \
          --resources gpu=1 cpu=16 \

exit
"