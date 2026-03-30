#!/bin/bash

#SBATCH --job-name=genqr_xxl
#SBATCH --partition=gpu-a100
#SBATCH --time=6:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=12G
#SBATCH --account=Education-EEMCS-Courses-DSAIT4095
#SBATCH --output=logs/%j.out
#SBATCH --mail-type=START,END,FAIL

source $(conda info --base)/etc/profile.d/conda.sh
conda activate IR

export HF_HOME=/scratch/knikolaevskii/hf_cache
export IR_DATASETS_HOME=/scratch/knikolaevskii/ir_datasets

python run_experiment.py \
  --model google/flan-t5-xxl \
  --device cuda \
  --datasets msmarco-passage/trec-dl-2019/judged \
  --num_samples 20 \
  --use_cache \
  --log_reformulations
