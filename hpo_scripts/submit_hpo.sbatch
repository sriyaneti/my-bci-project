#!/bin/bash
#SBATCH --job-name=bci_hpo
#SBATCH --output=hpo_slurm_%j.out
#SBATCH --error=hpo_slurm_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

set -euo pipefail

# Activate your env -- MAKE SURE YOU ARE RUNNING THE RIGHT ENV
source /sailhome/neti/miniconda3/etc/profile.d/conda.sh
conda activate b2txt25

# (Optional) help torch choose good CPU threads
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

cd /juice6/u/neti/my_bci_project

bash hpo_scripts/run_hpo_worker.sh