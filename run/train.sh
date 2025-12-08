#!/bin/bash
#SBATCH --job-name=breast_cancer_xai
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mem=64G
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:1

# module load python # Uncomment if needed on cluster
# cd .. # The user template had this, implying script is run from run/ but CWD should be project root for python execution

# Get the directory of this script to reliably find the project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$SCRIPT_DIR/.."
cd "$PROJECT_ROOT"

# Ensure logs directory exists for local runs (optional, but good practice if not on Slurm)
mkdir -p logs

python scripts/training.py fit \
  --data.ndim=3 \
  --data.batch_size=1 \
  --data.num_workers=12 \
  --model.n_estimators=100 \
  --model.max_iter=1000 \
  --model.svm_kernel=rbf \
  --trainer.max_epochs=1000 \
  --trainer.accelerator=gpu
