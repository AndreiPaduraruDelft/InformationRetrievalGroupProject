#!/bin/bash
#SBATCH --job-name=genqr_xxl
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%j.out

module load java/11
source activate your_env

python run_experiment.py --model google/flan-t5-xxl --device cuda
