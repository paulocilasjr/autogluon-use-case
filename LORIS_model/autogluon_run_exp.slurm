#!/bin/bash
#SBATCH --job-name=autglu_MIL               # Job name
#SBATCH --output=ag.out  # Output file with job ID
#SBATCH --error=ag.out   # Error file with job ID
#SBATCH --ntasks=2                    # Number of tasks
#SBATCH --cpus-per-task=10              # Number of CPU cores
#SBATCH --mem=1000G                     # Memory allocation
#SBATCH --time=5-02:00:00               # Time limit (days-HH:MM:SS)

export LD_LIBRARY_PATH=/home/4475918/.conda/envs/pytorch_env/lib:$LD_LIBRARY_PATH

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate ag_env

# --- Run experiment ---
python LORIS_autogluon_exp.py
