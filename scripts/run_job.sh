#!/bin/bash
#SBATCH --time=0:10:00
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=a100:1
#SBATCH --job-name=short test

# ---------------- Setup ----------------
module --force purge
module load Python/3.12.3-GCCcore-13.3.0
source $HOME/NN_Drumming_Project/NNvenv/bin/activate


python -m main


