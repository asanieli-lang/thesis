import torch
from pathlib import Path

# STFT parameters for spectrogram computation
N_FFT = 256
HOP_LENGTH = 50

input_dir = Path("/mnt/scratch/temporary/asanieli_data/processed_pt")
output_dir = input_dir

input_dir.mkdir(parents=True, exist_ok=True)
output_dir.mkdir(parents=True, exist_ok=True)


def signal_to_spectrogram(signal):
    """Compute magnitude spectrogram via STFT with z-score normalization."""
    # Normalize signal via z-score
    signal = (signal - signal.mean()) / (signal.std() + 1e-8)
    # Compute magnitude spectrogram via STFT
    window = torch.hann_window(N_FFT)
    stft_result = torch.stft(signal, n_fft=N_FFT, hop_length=HOP_LENGTH, window=window, return_complex=True)
    return torch.abs(stft_result)


def process_file(input_file):
    """Load signal file and compute per-channel spectrograms using STFT."""
    data = torch.load(input_file, map_location='cpu', weights_only=True)
    signals = data['signals']
    labels = data['labels']
    
    # Compute spectrogram per window and channel
    n_windows, n_channels, _ = signals.shape
    spectrograms_list = []
    
    for i in range(n_windows):
        spec_per_window = []
        for ch in range(n_channels):
            spec = signal_to_spectrogram(signals[i, ch])
            spec_per_window.append(spec)
        spectrograms_list.append(torch.stack(spec_per_window))
    
    all_spectrograms = torch.stack(spectrograms_list)
    
    output_file = output_dir / input_file.name.replace('_processed', '_spec')
    torch.save({'spectrograms': all_spectrograms, 'labels': labels}, output_file)


if __name__ == "__main__":
    # Process all input files and generate spectrograms
    input_files = sorted(input_dir.glob('*_processed.pt'))
    
    for i, f in enumerate(input_files, 1):
        print(f"[{i}/{len(input_files)}] Processing {f.name}...")
        process_file(f)
