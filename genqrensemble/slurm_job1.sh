#!/bin/bash

#SBATCH --job-name=genqr_xxl
#SBATCH --partition=gpu-a100
#SBATCH --time=6:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=7G
#SBATCH --account=Education-EEMCS-Courses-DSAIT4095
#SBATCH --output=logs/%j.out
#SBATCH --mail-type=START,END,FAIL

module load 2025
module load cuda/12.9
module load miniconda3
conda activate IR

export HF_HOME=/scratch/knikolaevskii/hf_cache
export IR_DATASETS_HOME=/scratch/knikolaevskii/ir_datasets
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

python reformulate.py --queries msmarco-passage_trec-dl-2019_judged_queries.json \
                          --model google/flan-t5-xxl \
                          --device cuda \
                          --output cache/google_flan-t5-xxl__msmarco-passage_trec-dl-2019_judged.json