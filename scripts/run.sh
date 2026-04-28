#!/bin/bash
#SBATCH --job-name=sleep_train_single
#SBATCH --partition=long
#SBATCH --gres=gpu:4        
#SBATCH --cpus-per-task=24   
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

module load PyTorch/2.10.0-foss-2025b-CUDA-12.9.1
source venv/bin/activate

echo "Sleep Stage Classification"


# NCCL settings
export NCCL_TIMEOUT_MIN=30
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL


torchrun --nproc_per_node=4 sleepnn/train.py

if [ $? -eq 0 ]; then
    echo ""
    echo "Training successful. Starting evaluation..."
    echo ""
    python -u sleepnn/eval.py
    echo ""
else
    echo ""
    echo "Training failed!"
    exit 1
fi