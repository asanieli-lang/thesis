import os
import bisect
import torch
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, data_dir: str, split: str = 'train', split_ratio: float = 0.8):
        self.data_dir = data_dir
        self.split = split
        self.split_ratio = split_ratio
        
        all_files = sorted([
            os.path.join(data_dir, f) for f in os.listdir(data_dir) 
            if f.endswith('.pt')
        ])
        
        split_idx = int(len(all_files) * split_ratio)
        self.files = all_files[:split_idx] if split == 'train' else all_files[split_idx:]
        
        self.cumulative_lengths = []
        total_samples = 0
        
        for f in self.files:
            data = torch.load(f, map_location='cpu', weights_only=True)
            length = len(data['labels'])
            total_samples += length
            self.cumulative_lengths.append(total_samples)
        
        print(f"[{split.upper()}] Loaded {len(self.files)} files with {total_samples} samples")

    def __len__(self):
        return self.cumulative_lengths[-1] if self.cumulative_lengths else 0

    def __getitem__(self, idx):
        file_idx = bisect.bisect_right(self.cumulative_lengths, idx)
        local_idx = idx if file_idx == 0 else idx - self.cumulative_lengths[file_idx - 1]
        file_path = self.files[file_idx]
        
        spec_file_path = str(file_path).replace('_processed.pt', '_spec.pt')
        
        if os.path.exists(spec_file_path):
            data = torch.load(spec_file_path, map_location='cpu', mmap=True, weights_only=True)
            spectrogram = data['spectrograms'][local_idx]
        else:
            data = torch.load(file_path, map_location='cpu', mmap=True, weights_only=True)
            signal = data['signals'][local_idx]
            signal = (signal - signal.mean()) / (signal.std() + 1e-8)
            window = torch.hann_window(256)
            spectrogram = torch.abs(torch.stft(signal, n_fft=256, hop_length=50, window=window, return_complex=True))
        
        label = data['labels'][local_idx]
        return spectrogram, int(label)