import os
import bisect
import torch
import re
import json
import numpy as np
from torch.utils.data import Dataset
from typing import Dict, List, Tuple

# STFT parameters for spectrogram computation (must match preprocessing)
N_FFT = 256
HOP_LENGTH = 50

class MyDataset(Dataset):
    """Single-sample EEG dataset with stratified subject-level split."""
    def __init__(self, data_dir: str, split: str = 'train', split_ratio: float = 0.8):
        """Load and prepare single-sample dataset with stratified subject split."""
        self.data_dir = data_dir
        self.split = split
        self.split_ratio = split_ratio
        
        all_files = sorted([
            os.path.join(data_dir, f) for f in os.listdir(data_dir) 
            if f.endswith('_processed.pt')
        ])
        
        self.files = self._stratified_subject_split(all_files, split)
        
        # Map file paths to subject IDs for statistics computation
        self.file_to_subject = {}
        for f in self.files:
            subject_id, _ = self._extract_subject_and_substance(f)
            self.file_to_subject[f] = subject_id
        
        self.subject_to_files = {}
        for f, subject_id in self.file_to_subject.items():
            if subject_id not in self.subject_to_files:
                self.subject_to_files[subject_id] = []
            self.subject_to_files[subject_id].append(f)
        
        self._load_cumulative_lengths()
        self._compute_subject_statistics()

    def _stratified_subject_split(self, all_files: List[str], split: str) -> List[str]:
        """Split files by subject while ensuring all conditions are in test set."""
        # Group files by subject and experimental conditions
        subject_groups: Dict[str, List[str]] = {}
        subject_substances: Dict[str, set] = {} 
        subject_file_counts: Dict[str, int] = {}  
        
        for file_path in all_files:
            subject_id, substance = self._extract_subject_and_substance(file_path)
            
            if subject_id not in subject_groups:
                subject_groups[subject_id] = []
                subject_substances[subject_id] = set()
                subject_file_counts[subject_id] = 0
            
            subject_groups[subject_id].append(file_path)
            subject_substances[subject_id].add(substance)
            subject_file_counts[subject_id] += 1  
        
        # Represent all experimental conditions in test set
        all_substances = set()
        for substs in subject_substances.values():
            all_substances.update(substs)

        sorted_subjects_by_size = sorted(
            subject_groups.keys(), 
            key=lambda s: subject_file_counts[s]
        )

        # Allocate subjects with unique substances to test set
        eval_subjects = []
        remaining_subjects = set(subject_groups.keys())
        substances_covered_in_eval = set()

        for substance in sorted(all_substances):
            if substance not in substances_covered_in_eval:
                for subject in sorted_subjects_by_size:
                    if subject in remaining_subjects and substance in subject_substances[subject]:
                        eval_subjects.append(subject)
                        remaining_subjects.remove(subject)
                        substances_covered_in_eval.update(subject_substances[subject])
                        break

        # Fill test set with remaining smaller subjects
        eval_target_count = int(len(subject_groups) * (1 - self.split_ratio))
        
        remaining_sorted = sorted(list(remaining_subjects), key=lambda s: subject_file_counts[s])
        for subject in remaining_sorted:
            if len(eval_subjects) < eval_target_count:
                eval_subjects.append(subject)
                remaining_subjects.remove(subject)

        train_subjects = [s for s in subject_groups.keys() if s not in eval_subjects]

        eval_files = [f for s in eval_subjects for f in subject_groups[s]]
        train_files = [f for s in train_subjects for f in subject_groups[s]]

        return eval_files if split == 'test' else train_files
        
    def _extract_subject_and_substance(self, filename: str) -> Tuple[str, str]:
        """Extract subject ID and experimental conditions from filename."""
        basename = os.path.basename(filename)
        
        subject_match = re.search(r'\d+', basename)
        subject_id = subject_match.group() if subject_match else "unknown"
        
        substances = []
        keywords = {
            'mdl': r'mdl',
            'psilo': r'psilo|psilocybin',
            'saline': r'saline',
            'depr': r'depr|deprivation',
        }
        
        for keyword, pattern in keywords.items():
            if re.search(pattern, basename.lower()):
                substances.append(keyword)
        
        substance = '_'.join(sorted(substances)) if substances else 'unknown'
        return subject_id, substance
    

    def _load_cumulative_lengths(self):
        """Cache cumulative sample indices for fast index-to-file lookup."""
        # Cache cumulative sample indices for fast lookup
        cache_path = os.path.join(os.path.dirname(self.data_dir), f"lengths_cache_{self.split}_{len(self.files)}.json")
        
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    self.cumulative_lengths = json.load(f)
                return
            except Exception:
                pass
        
        self.cumulative_lengths = []
        total_samples = 0
        
        for f in self.files:
            data = torch.load(f, map_location='cpu', weights_only=True)
            length = len(data['labels'])
            total_samples += length
            self.cumulative_lengths.append(total_samples)
            del data
        
        try:
            with open(cache_path, 'w') as f:
                json.dump(self.cumulative_lengths, f)
        except Exception:
            pass

    def _compute_subject_statistics(self):
        """Computes mean and std for spectrograms to enable subject-wise normalization."""
        stats_cache_path = os.path.join(
            os.path.dirname(self.data_dir), 
            f"subject_stats_{self.split}_{len(self.files)}.json"
        )
        
        if os.path.exists(stats_cache_path):
            try:
                with open(stats_cache_path, 'r') as f:
                    stats_dict = json.load(f)
                    self.subject_stats = {k: v for k, v in stats_dict.items()}
                return
            except Exception:
                pass
        
        # Compute per-subject statistics across all files
        self.subject_stats = {}
        for subject_id, subject_files in self.subject_to_files.items():
            all_specs = []
            for file_path in subject_files:
                spec_path = str(file_path).replace('_processed.pt', '_spec.pt')
                try:
                    if os.path.exists(spec_path):
                        data = torch.load(spec_path, map_location='cpu', weights_only=True)
                        all_specs.append(data['spectrograms'])
                    else:
                        data = torch.load(file_path, map_location='cpu', weights_only=True)
                        all_specs.append(data['signals'])
                except Exception:
                    pass
            
            if all_specs:
                all_data = torch.cat(all_specs, dim=0)
                self.subject_stats[subject_id] = {
                    'mean': all_data.mean().item(),
                    'std': all_data.std().item()
                }
        
        try:
            with open(stats_cache_path, 'w') as f:
                json.dump(self.subject_stats, f)
        except Exception:
            pass

    def __len__(self):
        """Return total number of samples across all files."""
        return self.cumulative_lengths[-1] if self.cumulative_lengths else 0

    def __getitem__(self, idx):
        """Load spectrogram and label for single sample with subject normalization."""
        # Find file containing this index via binary search
        file_idx = bisect.bisect_right(self.cumulative_lengths, idx)
        local_idx = idx if file_idx == 0 else idx - self.cumulative_lengths[file_idx - 1]
        file_path = self.files[file_idx]
        
        # Load precomputed spectrogram or compute STFT on-the-fly
        spec_path = str(file_path).replace('_processed.pt', '_spec.pt')
        if os.path.exists(spec_path):
            data = torch.load(spec_path, map_location='cpu', weights_only=True, mmap=True)
            spectrogram = data['spectrograms'][local_idx]
            label = data['labels'][local_idx]
        else:
            # Fallback: compute STFT when precomputed spectrogram unavailable
            data = torch.load(file_path, map_location='cpu', weights_only=True, mmap=True)
            signal = data['signals'][local_idx]
            window = torch.hann_window(N_FFT)
            spectrogram = torch.abs(torch.stft(signal, n_fft=N_FFT, hop_length=HOP_LENGTH, window=window, return_complex=True))
            label = data['labels'][local_idx]
        
        subject_id = self.file_to_subject[file_path]
        if subject_id in self.subject_stats:
            stats = self.subject_stats[subject_id]
            if stats['std'] > 0:
                spectrogram = (spectrogram - stats['mean']) / stats['std']
            
        return spectrogram, int(label)
    


class SequenceDataset(Dataset):  
    """Sliding-window sequence dataset for temporal modeling."""
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        split_ratio: float = 0.8,
        sequence_length: int = 20,
        stride: int = 1
    ):
        """Initialize sliding-window sequence dataset with caching."""
        self.sequence_length = sequence_length
        self.stride = stride
        
        # Use MyDataset as base single-sample access layer
        self.base_dataset = MyDataset(
            data_dir=data_dir,
            split=split,
            split_ratio=split_ratio
        )
        
        self.total_samples = len(self.base_dataset)
        # Precompute valid sequence starts for fast indexing
        self.sequence_starts = self._find_valid_sequence_starts()
        
        # Cache with LRU eviction to manage memory efficiently
        self._data_cache = {}
        self._cache_order = []
        self._max_cache_size = 5

    def _find_valid_sequence_starts(self):
        """Find all valid starting positions for sequences across files."""
        sequence_starts = []
        cumsum = self.base_dataset.cumulative_lengths
        
        for file_idx, file_path in enumerate(self.base_dataset.files):
            if file_idx == 0:
                file_start = 0
                file_end = cumsum[0]
            else:
                file_start = cumsum[file_idx - 1]
                file_end = cumsum[file_idx]
            
            num_samples_in_file = file_end - file_start
            max_start_in_file = num_samples_in_file - self.sequence_length
            
            if max_start_in_file >= 0:
                for local_pos in range(0, max_start_in_file + 1, self.stride):
                    global_pos = file_start + local_pos
                    sequence_starts.append(global_pos)
        
        return sequence_starts
    
    def __len__(self):
        """Return total number of valid sequence windows."""
        return len(self.sequence_starts)
    
    def _load_data_file(self, spec_path):
        """Load file from cache or disk with LRU eviction strategy."""
        if spec_path in self._data_cache:
            return self._data_cache[spec_path]
        
        data = torch.load(spec_path, map_location='cpu', weights_only=True, mmap=True)
        
        # Evict oldest entry when cache reaches max size
        if len(self._cache_order) >= self._max_cache_size:
            old_path = self._cache_order.pop(0)
            del self._data_cache[old_path]
        
        self._data_cache[spec_path] = data
        self._cache_order.append(spec_path)
        
        return data
    
    def __getitem__(self, idx):
        """Fetch sequence of spectrograms with subject normalization and augmentation."""
        global_start = self.sequence_starts[idx]
        file_idx = bisect.bisect_right(self.base_dataset.cumulative_lengths, global_start)
        file_start = 0 if file_idx == 0 else self.base_dataset.cumulative_lengths[file_idx - 1]
        local_start = global_start - file_start
        
        file_path = self.base_dataset.files[file_idx]
        spec_path = str(file_path).replace('_processed.pt', '_spec.pt')
        subject_id = self.base_dataset.file_to_subject[file_path]
        
        data = self._load_data_file(spec_path)
        
        # Extract sequence window; use final label as target
        spectrograms = data['spectrograms'][local_start:local_start + self.sequence_length].clone()
        labels = data['labels'][local_start:local_start + self.sequence_length]
        target_label = int(labels[-1])
               
        # Apply subject-wise z-score normalization
        if subject_id in self.base_dataset.subject_stats:
            stats = self.base_dataset.subject_stats[subject_id]
            if stats['std'] > 0:
                spectrograms = (spectrograms - stats['mean']) / stats['std']
        
        # Apply augmentations during training only
        if self.base_dataset.split == 'train':
            # SpecAugment: mask random frequency bands (1-9 bins) and time frames (1-4 frames)
            freq_mask_size = torch.randint(1, 9, (1,)).item()
            freq_start = torch.randint(0, spectrograms.shape[-2] - freq_mask_size, (1,)).item()
            spectrograms[..., freq_start:freq_start + freq_mask_size, :] = 0
            
            time_mask_size = torch.randint(1, 4, (1,)).item()
            time_start = torch.randint(0, spectrograms.shape[-1] - time_mask_size, (1,)).item()
            spectrograms[..., time_start:time_start + time_mask_size] = 0
            
            # Amplitude scaling: simulate hardware gain variance with uniform distribution U(0.8, 1.2)
            amplitude_scale = torch.FloatTensor(1).uniform_(0.8, 1.2).item()
            spectrograms = spectrograms * amplitude_scale
            
            # Additive Gaussian noise with standard deviation σ=0.05
            gaussian_noise = torch.randn_like(spectrograms) * 0.05
            spectrograms = spectrograms + gaussian_noise
        
        subject_id_int = int(subject_id) if subject_id.isdigit() else 0
        return spectrograms, target_label, subject_id_int