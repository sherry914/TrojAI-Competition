#!/bin/bash
#SBATCH -A r00066
#SBATCH -J train
#SBATCH -p gpu
#SBATCH --mail-type=ALL
#SBATCH --mail-user=xxx
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=6
#SBATCH --gpus-per-node=1
#SBATCH --time=2:00:00
#SBATCH --mem=100G

module load miniconda
source activate /N/slate/zwa2/conda_envs/mia

python extractor.py 

