#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --partition=gpu1
#SBATCH --gpus-per-node=a100:1
#SBATCH --job-name=llms
#SBATCH --mem=50G

module purge
module load CUDA/11.7.0
module load cuDNN/8.4.1.50-CUDA-11.7.0
module load NCCL/2.12.12-GCCcore-11.3.0-CUDA-11.7.0
module load Python/3.11.3-GCCcore-12.3.0
module load GCC/11.3.0

source $HOME/venvs/first_env/bin/activate

python BERTscore.py

deactivate