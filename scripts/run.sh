#!/bin/bash
#SBATCH --job-name=sleep_train_eval
#SBATCH --partition=long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/train_eval_%j.out
#SBATCH --error=logs/train_eval_%j.err

module load PyTorch/2.10.0-foss-2025b-CUDA-12.9.1
source venv/bin/activate

echo "Starting training..."
python -u sleepnn/train.py

if [ $? -eq 0 ]; then
    echo ""
    echo "Training complete. Starting evaluation..."
    echo ""
    python -u sleepnn/eval.py
    echo ""
    echo "Evaluation complete."
    echo "Output files: ./outputs/"
else
    echo ""
    echo "Training failed. Skipping evaluation."
    exit 1
fi
