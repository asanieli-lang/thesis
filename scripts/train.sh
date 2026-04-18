#!/bin/bash
#SBATCH --job-name=sleep_train
#SBATCH --partition=long         # Use the partition intended for GPUs (check your university's cluster docs if it's named differently)
#SBATCH --gres=gpu:1             # CRITICAL: This line explicitly requests 1 GPU
#SBATCH --cpus-per-task=4        # 4 CPU cores to feed data to the GPU quickly
#SBATCH --mem=32G                # 32 GB RAM for data loading
#SBATCH --time=04:00:00          # 4 hours is a very safe buffer
#SBATCH --output=logs/eval_%j.out   
#SBATCH --error=logs/eval_%j.err     

module load PyTorch/2.10.0-foss-2025b-CUDA-12.9.1
source venv/bin/activate

echo "Zahajuji evaluovani neuronove site..."
srun python -u src/eval.py      # We are now executing the training loop!

echo "Dokoncen!"