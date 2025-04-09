#!/bin/bash
#SBATCH -A r00066
#SBATCH -J train
#SBATCH -p gpu
#SBATCH --mail-type=ALL
#SBATCH --mail-user=xxx
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=1
#SBATCH --time=4:00:00
#SBATCH --mem=100G

module load miniconda
module load cudatoolkit
source activate /N/slate/zwa2/conda_envs/mia

python -u train_mlp.py
