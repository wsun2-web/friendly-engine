#!/bin/bash

#SBATCH --job-name=K_means_clustering
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --cluster=chip-cpu
#SBATCH --account=hpcf-scales
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=00:10:00
#SBATCH --mem=1G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

# Activate PyVenv
module load Python/3.12.3-GCCcore-13.3.0
python -m venv /umbc/rs/hpcf-scales/users/wsun2/PythonEnv
source /umbc/rs/hpcf-scales/users/wsun2/PythonEnv/bin/activate

python K_Means_Clustering.py
