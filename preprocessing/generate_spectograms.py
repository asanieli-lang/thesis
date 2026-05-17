import torch
from pathlib import Path

N_FFT = 256
HOP_LENGTH = 50

input_dir = Path("/mnt/scratch/temporary/asanieli_data/processed_pt")
output_dir = input_dir

input_dir.mkdir(parents=True, exist_ok=True)
output_dir.mkdir(parents=True, exist_ok=True)


def signal_to_spectrogram(signal):
    signal = (signal - signal.mean()) / (signal.std() + 1e-8)
    window = torch.hann_window(N_FFT)
    stft_result = torch.stft(signal, n_fft=N_FFT, hop_length=HOP_LENGTH, window=window, return_complex=True)
    return torch.abs(stft_result)


def process_file(input_file):
    data = torch.load(input_file, map_location='cpu', weights_only=True)
    signals = data['signals']
    labels = data['labels']
    
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
    input_files = sorted(input_dir.glob('*_processed.pt'))
    
    for i, f in enumerate(input_files, 1):
        print(f"[{i}/{len(input_files)}] Processing {f.name}...")
        process_file(f)
