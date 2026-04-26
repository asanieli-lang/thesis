import os
import bisect
import torch
import re
import json
import numpy as np
from torch.utils.data import Dataset
from typing import Dict, List, Tuple

class MyDataset(Dataset):
    def __init__(self, data_dir: str, split: str = 'train', split_ratio: float = 0.8):
        self.data_dir = data_dir
        self.split = split
        self.split_ratio = split_ratio
        
        all_files = sorted([
            os.path.join(data_dir, f) for f in os.listdir(data_dir) 
            if f.endswith('_processed.pt')
        ])
        
        self.files = self._stratified_subject_split(all_files, split)
        
        self._load_cumulative_lengths()
        
    def _extract_subject_and_substance(self, filename: str) -> Tuple[str, str]:
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
    
    def _stratified_subject_split(self, all_files: List[str], split: str) -> List[str]:
        subject_groups: Dict[str, List[str]] = {}
        subject_substances: Dict[str, set] = {} 
        
        for file_path in all_files:
            subject_id, substance = self._extract_subject_and_substance(file_path)
            
            if subject_id not in subject_groups:
                subject_groups[subject_id] = []
                subject_substances[subject_id] = set()
            
            subject_groups[subject_id].append(file_path)
            subject_substances[subject_id].add(substance)
        
        all_substances = set()
        for subj in sorted(subject_groups.keys()):
            all_substances.update(subject_substances[subj])

        eval_subjects = []
        train_subjects = []
        substances_in_eval = set()

        remaining_subjects = set(subject_groups.keys())
        
        for substance in sorted(all_substances):
            for subject in sorted(remaining_subjects):
                if substance in subject_substances[subject]:
                    eval_subjects.append(subject)
                    substances_in_eval.update(subject_substances[subject])
                    remaining_subjects.remove(subject)
                    break
        
        eval_target = int(len(subject_groups) * (1 - self.split_ratio))
        
        for subject in sorted(remaining_subjects):
            if len(eval_subjects) < eval_target:
                eval_subjects.append(subject)
            else:
                train_subjects.append(subject)

        eval_files = []
        train_files = []
        
        for subject in eval_subjects:
            eval_files.extend(subject_groups[subject])
        
        for subject in train_subjects:
            train_files.extend(subject_groups[subject])
        
        return eval_files if split == 'test' else train_files
    
    def _load_cumulative_lengths(self):
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

    def __len__(self):
        return self.cumulative_lengths[-1] if self.cumulative_lengths else 0

    def __getitem__(self, idx):
        file_idx = bisect.bisect_right(self.cumulative_lengths, idx)
        local_idx = idx if file_idx == 0 else idx - self.cumulative_lengths[file_idx - 1]
        file_path = self.files[file_idx]
        
        spec_path = str(file_path).replace('_processed.pt', '_spec.pt')
        if os.path.exists(spec_path):
            data = torch.load(spec_path, map_location='cpu', weights_only=True, mmap=True)
            spectrogram = data['spectrograms'][local_idx]
            label = data['labels'][local_idx]
        else:
            data = torch.load(file_path, map_location='cpu', weights_only=True, mmap=True)
            signal = data['signals'][local_idx]
            window = torch.hann_window(256)
            spectrogram = torch.abs(torch.stft(signal, n_fft=256, hop_length=50, window=window, return_complex=True))
            label = data['labels'][local_idx]
            
        return spectrogram, int(label)
    


class SequenceDataset(Dataset):  
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        split_ratio: float = 0.8,
        sequence_length: int = 15,
        stride: int = 1
    ):
        self.sequence_length = sequence_length
        self.stride = stride
        
        self.base_dataset = MyDataset(
            data_dir=data_dir,
            split=split,
            split_ratio=split_ratio
        )
        
        self.total_samples = len(self.base_dataset)
        self.sequence_starts = self._find_valid_sequence_starts()
        
        self._data_cache = {}
        self._cache_order = []
        self._max_cache_size = 5

    def _find_valid_sequence_starts(self):
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
        return len(self.sequence_starts)
    
    def _load_data_file(self, spec_path):
        if spec_path in self._data_cache:
            return self._data_cache[spec_path]
        
        data = torch.load(spec_path, map_location='cpu', weights_only=True, mmap=True)
        
        if len(self._cache_order) >= self._max_cache_size:
            old_path = self._cache_order.pop(0)
            del self._data_cache[old_path]
        
        self._data_cache[spec_path] = data
        self._cache_order.append(spec_path)
        
        return data
    
    def __getitem__(self, idx):
        global_start = self.sequence_starts[idx]
        file_idx = bisect.bisect_right(self.base_dataset.cumulative_lengths, global_start)
        file_start = 0 if file_idx == 0 else self.base_dataset.cumulative_lengths[file_idx - 1]
        local_start = global_start - file_start
        
        file_path = self.base_dataset.files[file_idx]
        spec_path = str(file_path).replace('_processed.pt', '_spec.pt')
        
        # Používat cache místo přímého torch.load
        data = self._load_data_file(spec_path)
        
        spectrograms = data['spectrograms'][local_start:local_start + self.sequence_length]
        labels = data['labels'][local_start:local_start + self.sequence_length]
        
        target_label = int(labels[-1])
        
        if self.base_dataset.split == 'train':
            noise = torch.randn_like(spectrograms) * 0.01 #zvysuju z  0.01
            spectrograms = spectrograms + noise
        
        return spectrograms, target_label