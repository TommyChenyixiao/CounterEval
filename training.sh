#!/bin/bash

#SBATCH --job-name=Soccer
#SBATCH --time=8:00:00
#SBATCH --partition=gpu
#SBATCH --mail-type=ALL
#SBATCH --gpus=a100:1

module purge
module load miniconda
export PATH=$HOME/miniconda3/bin:$PATH
source $(conda info --base)/etc/profile.d/conda.sh
conda activate sds625_soccer

python graph_train.py
