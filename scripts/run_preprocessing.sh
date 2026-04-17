#!/bin/bash
#SBATCH --job-name=data_test
#SBATCH --partition=long         
#SBATCH --cpus-per-task=4        # Zde nepotřebujeme GPU, vezmeme si silný CPU
#SBATCH --mem=32G                # Poprosíme o 32 GB RAM pro čtení dat
#SBATCH --time=04:00:00          # Dáme mu na to štědré 4 hodiny
#SBATCH --output=logs/prep_%j.out   
#SBATCH --error=logs/prep_%j.err     

module load PyTorch/2.10.0-foss-2025b-CUDA-12.9.1
source venv/bin/activate
echo "Zahajuji testovani dat..."
srun python -u data/test_dataset.py

echo "Dokončen!"