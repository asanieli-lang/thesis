
import re
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import torch

try:
    import mne
except ImportError:
    mne = None


class EEGPreprocessor:
    """Convert raw EDF files to 4-second windowed PyTorch tensors."""
    CHANNEL_NAMES: list[str] = ['EEG-0', 'EEG-1', 'EEG-2', 'EMG']
    SAMPLING_RATE: int = 500
    WINDOW_DURATION: int = 4
    WINDOW_SAMPLES: int = SAMPLING_RATE * WINDOW_DURATION
    SLEEP_STAGE_MAP: dict[str, int] = {
        'Wake': 0, 'W': 0, 'WAKE': 0,
        'NREM': 1, 'N': 1,
        'REM': 2, 'R': 2,
    }

    def __init__(
            self,
            data_dir: Optional[Path] = None,
            output_dir: Optional[Path] = None,
            channel_indices: Optional[list[int]] = None,
        ) -> None:
            """Initialize preprocessor with input/output directories."""
            self.data_dir: Path = (
                Path(data_dir) if data_dir
                else Path('/mnt/scratch/temporary/asanieli_data/data')
            )
            self.output_dir: Path = (
                Path(output_dir) if output_dir
                else Path('/mnt/scratch/temporary/asanieli_data/processed_pt')
            )
            self.channel_indices: Optional[list[int]] = channel_indices
            self.output_dir.mkdir(parents=True, exist_ok=True)


    def find_file_pairs(self) -> list[tuple[Path, Optional[Path]]]:
        """Match EEG and annotation files by subject ID and experimental conditions."""
        eeg_dir = self.data_dir / 'eeg'
        eeg_files = sorted(eeg_dir.glob('*.edf'))


        labels_dir = self.data_dir / 'labels'
        if not labels_dir.exists():
            return [(f, None) for f in eeg_files]

        label_files = sorted(labels_dir.glob('*.edf'))
        
        # Extract subject ID and conditions for reliable matching
        def extract_signature(filename: str) -> str:
            """Generate unique signature from filename for pair matching."""
            subject_match = re.search(r'\d+', filename)
            if not subject_match:
                return ""
            subject_id = subject_match.group()
            keywords = {
                'mdl': r'mdl',
                'psilo': r'psilo|psilocybin',
                'saline': r'saline',
                'depr': r'depr|deprivation',
                'sleep': r'sleep',
            }

            conditions = []
            for keyword, pattern in keywords.items():
                if re.search(pattern, filename.lower()):
                    conditions.append(keyword)
            if conditions:
                conditions_str = '_'.join(sorted(conditions))
                return f"{subject_id}_{conditions_str}"
            else:
                return subject_id

        eeg_signatures = {extract_signature(f.stem): f for f in eeg_files}
        label_signatures = {extract_signature(f.stem): f for f in label_files}

        pairs = []
        for signature, eeg_file in eeg_signatures.items():
            label_file = label_signatures.get(signature)
            pairs.append((eeg_file, label_file))

        return pairs

    def read_edf_file(self, edf_path: Path) -> tuple[np.ndarray, int, list[str]]:
        """Load EDF file using MNE library with error handling."""
        if mne is None:
            raise ImportError("mne library is required to read EDF files")
        
        # Suppress MNE verbose output
            warnings.simplefilter("ignore")
            raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)

        signal = raw.get_data()
        sampling_rate = int(raw.info['sfreq'])
        channel_names = raw.ch_names

        return signal, sampling_rate, channel_names

    def select_channels(
        self,
        signal: np.ndarray,
        channel_names: list[str],
        n_channels: int = 4
    ) -> np.ndarray:
        """Extract target channels (EEG + EMG) with fallback strategy."""
        if self.channel_indices is not None:
            indices = self.channel_indices[:n_channels]
            return signal[indices, :]

        # Match channel names using case-insensitive substring search
        indices = []
        for target_name in self.CHANNEL_NAMES[:n_channels]:
            found = False
            for i, ch_name in enumerate(channel_names):
                if target_name.lower() in ch_name.lower():
                    indices.append(i)
                    found = True
                    break

        # Fallback: use first available channels if matching fails
        if len(indices) < n_channels:
            indices = list(range(min(n_channels, signal.shape[0])))

        return signal[indices[:n_channels], :]

    def read_annotations(
        self,
        annotation_path: Path
    ) -> Optional[mne.Annotations]:
        """Load MNE annotations from EDF file with error handling."""
        if annotation_path is None or not annotation_path.exists():
            return None
        try:
            annotations = mne.read_annotations(str(annotation_path))
            return annotations
        except Exception:
            return None

    def parse_sleep_stages(
        self,
        annotations: Optional[mne.Annotations],
        signal_duration: float,
        sampling_rate: int
    ) -> np.ndarray:
        """Convert MNE annotations to per-sample sleep stage labels."""
        n_samples = int(signal_duration * sampling_rate)
        # Initialize array with unlabeled marker
        stages = np.full(n_samples, -1, dtype=np.int32)  # Initialize with unlabeled (-1)

        if annotations is None:
            return stages

        try:
            # Map annotations to per-sample stage labels
            for annotation in annotations:
                onset_sec = annotation['onset']
                duration_sec = annotation['duration']
                description = annotation['description'].strip().upper()
                start_idx = int(onset_sec * sampling_rate)
                end_idx = int((onset_sec + duration_sec) * sampling_rate)
                # Map stage label from predefined dictionary
                stage = self.SLEEP_STAGE_MAP.get(description, -1)
                start_idx = max(0, min(start_idx, n_samples))
                end_idx = max(0, min(end_idx, n_samples))
                if end_idx > start_idx:
                    stages[start_idx:end_idx] = stage
        except Exception:
            pass

        return stages

    def slice_signal(
        self,
        signal: np.ndarray,
        stages: np.ndarray
    ) -> list[dict[str, torch.Tensor | int]]:
        """Partition signal into 4-second non-overlapping windows with stage labels."""
        epochs = []
        n_samples = signal.shape[1]

        # Process non-overlapping windows
        for start_idx in range(0, n_samples - self.WINDOW_SAMPLES + 1, self.WINDOW_SAMPLES):
            end_idx = start_idx + self.WINDOW_SAMPLES
            window_signal = signal[:, start_idx:end_idx]
            window_stages = stages[start_idx:end_idx]

            # Skip windows with unlabeled samples
            if np.any(window_stages == -1):
                continue

            # Assign majority vote label when multiple stages exist
            label = int(np.bincount(window_stages).argmax())
            signal_tensor = torch.from_numpy(window_signal).float()
            epoch = {'signal': signal_tensor, 'label': label}
            epochs.append(epoch)

        return epochs

    def save_epochs(
        self,
        epochs: list[dict[str, torch.Tensor | int]],
        output_prefix: str
    ) -> None:
        """Stack epochs and save to tensor file."""
        # Stack epochs and save output file
        all_signals = torch.stack([e['signal'] for e in epochs])
        all_labels = torch.tensor([e['label'] for e in epochs], dtype=torch.long)
        output_data = {'signals': all_signals, 'labels': all_labels}
        filename = self.output_dir / f"{output_prefix}_processed.pt"
        torch.save(output_data, filename)



    def process_single_file(
        self,
        eeg_path: Path,
        annotation_path: Optional[Path]
    ) -> int:
        """Load EEG file, extract channels, segment, and save as tensor."""
        try:
            signal, sampling_rate, channel_names = self.read_edf_file(eeg_path)
            signal = self.select_channels(signal, channel_names)
            annotations = self.read_annotations(annotation_path)
            signal_duration = signal.shape[1] / sampling_rate
            stages = self.parse_sleep_stages(
                annotations,
                signal_duration,
                sampling_rate
            )
            epochs = self.slice_signal(signal, stages)
            if len(epochs) == 0:
                return 0
            output_prefix = eeg_path.stem
            self.save_epochs(epochs, output_prefix)
            return len(epochs)
        except Exception:
            return 0

    def run(self) -> dict[str, int]:
        """Process all paired EDF files and generate statistics. """
        pairs = self.find_file_pairs()
        total_epochs = 0
        processed_files = 0
        failed_files = 0

        for eeg_file, annotation_file in pairs:
            n_epochs = self.process_single_file(eeg_file, annotation_file)
            if n_epochs > 0:
                processed_files += 1
                total_epochs += n_epochs
            else:
                failed_files += 1

        return {
            'processed_files': processed_files,
            'failed_files': failed_files,
            'total_epochs': total_epochs,
        }


def main() -> dict[str, int]:
    preprocessor = EEGPreprocessor()
    stats = preprocessor.run()
    return stats


if __name__ == '__main__':
    main()
