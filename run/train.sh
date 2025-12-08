#!/bin/bash

# Ensure script halts on error
set -e

# Get the directory of this script to reliably find the project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$SCRIPT_DIR/.."

# Navigate to project root
cd "$PROJECT_ROOT"

echo "Starting training pipeline from $PROJECT_ROOT"

# Run the training script with all steps
python scripts/training.py --steps all

echo "Done."
