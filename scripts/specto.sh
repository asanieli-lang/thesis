#!/bin/bash
#SBATCH --job-name=generate_specs
#SBATCH --partition=long
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=logs/generate_specs_%j.out
#SBATCH --error=logs/generate_specs_%j.err

module load PyTorch/2.10.0-foss-2025b-CUDA-12.9.1
source venv/bin/activate

echo "Starting spectrogram generation..."
python -u preprocessing/generate_spectograms.py

if [ $? -eq 0 ]; then
    echo ""
    echo "Spectrograms generated successfully! ✓"
else
    echo ""
    echo "Spectrogram generation failed!"
    exit 1
fi
