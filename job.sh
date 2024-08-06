#!/bin/bash
#SBATCH --job-name=transfer-learning
#SBATCH --output=transfer-learning.out
#SBATCH --error=transfer-learning.error
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:4
#SBATCH --time=4:00:00
#SBATCH --exclude=t01pdscgpu01
 
 
eval "$(/lustre1/tier2/users/aymane.elfirdoussi/miniconda3/bin/conda shell.bash hook)"
conda activate venv
 
python3 fine_tuning.py