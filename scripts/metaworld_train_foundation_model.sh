#!/bin/bash
#SBATCH --partition=cpu,cpu_il
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --job-name=train_metaworld_model
#SBATCH --time=04:00:00
#SBATCH --output=logs/train_metaworld_%j.out

source /pfs/data5/home/kn/kn_kn/kn_pop257914/multi-type-feedback/venv/bin/activate

python scripts/train_metaworld_foundation_model_intrinsic.py

/pfs/data5/home/kn/kn_kn/kn_pop257914/multi-type-feedback/venv/bin/deactivate