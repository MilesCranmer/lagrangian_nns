#!/bin/bash
#SBATCH -N1
#SBATCH -n 1
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem 50000
#SBATCH --cpus-per-task=7

srun bash -c "echo $(hostname) $SLURM_PROCID && $HOME/miniconda3/envs/main2/bin/python example.py"

