# Sleep Stage Classification using EEG

## Overview

This project classifies sleep stages (Wake, NREM, REM) from EEG signals using a CNN-LSTM-Attention architecture. Training is distributed across multiple GPUs using PyTorch DDP.

## Architecture

- **CNN**: Multi-scale feature extraction (kernels 7×3, 3×7, 5×5) with residual blocks
- **LSTM**: Temporal modeling with 128 hidden units, 2 layers
- **Attention**: Multi-head attention (4 heads) over LSTM outputs
- **Classifier**: Dropout → Dense layers → 3-class output

## Data Processing

- Input: EDF files with raw EEG signals (500 Hz sampling)
- Processing: 4-second windows, z-score normalization, subject-wise split
- Spectrogram: STFT with N_FFT=256, HOP_LENGTH=50

## Requirements

- PyTorch >= 1.12
- MNE (for EDF reading)
- NumPy, Pandas, Scikit-learn, Matplotlib, UMAP

Install: `pip install -r requirements.txt`

## Usage

### Preprocessing

```bash
python preprocessing/preprocessing.py  # Convert EDF → tensors
python preprocessing/generate_spectrograms.py  # Compute STFT
```

### Training

```bash
export SLEEPNN_RUN_ID=12345
export SLEEPNN_DATA_DIR=/path/to/processed_data
python -m torch.distributed.launch --nproc_per_node=4 sleepnn/train.py
```

### Evaluation

```bash
export SLEEPNN_RUN_ID=12345
python sleepnn/eval.py
```

## Project Structure

```
sleepnn/
  model.py         - Architecture definition
  dataset.py       - Data loading with caching
  train.py         - Training loop (distributed)
  eval.py          - Evaluation and metrics

preprocessing/
  preprocessing.py - EDF to tensor conversion
  generate_spectrograms.py - STFT computation

outputs/           - Models, metrics, visualizations
logs/              - Training logs
```

## Training Details

- Loss: Focal loss with label smoothing
- Optimizer: Adam with learning rate scheduling
- Batch size: 16
- Epochs: 100 (early stopping on F1 score)
- Class balancing: Sqrt-normalized weights

## Results

Results are saved to `outputs/`:
- `best_model_checkpoint_{RUN_ID}.pth` - Best model
- `loss_history_lstm_{RUN_ID}.csv` - Metrics per epoch
- `hypnogram_{RUN_ID}.png` - Predictions vs ground truth
- `per_subject_f1_{RUN_ID}.png` - Per-subject generalization
- `umap_2d/3d_{RUN_ID}.png` - Latent space visualization

