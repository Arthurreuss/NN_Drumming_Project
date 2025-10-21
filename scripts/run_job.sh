#!/bin/bash
#SBATCH --time=2:00:00
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=a100:1

# ---------------- Setup ----------------
module --force purge
module load Python/3.12.3-GCCcore-13.3.0
source $HOME/NN_Drumming_Project/NNvenv/bin/activate


python -m main --model $1 --learning_rate $2


