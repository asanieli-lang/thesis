"""EEG Data Preprocessing Script.

Main objective: Process .edf files and split them into 4-second windows
with corresponding sleep stage annotations.

Output: data/processed/ folder with PyTorch tensors (.pt files)
Each .pt file contains a dictionary:
    {
        'signal': torch.tensor([4, 2000]),  # 4 channels, 2000 samples (4 sec @ 500Hz)
        'label': int  # 0=Wake, 1=NREM, 2=REM
    }
"""

import logging
import re
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# For reading EDF files
try:
    import mne
except ImportError:
    print("Warning: mne library not installed. Please install: pip install mne")
    mne = None

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EEGPreprocessor:
    """Preprocessor class for EEG data."""

    # Standard channels for rat data (3x EEG, 1x EMG)
    CHANNEL_NAMES: list[str] = ['EEG-0', 'EEG-1', 'EEG-2', 'EMG']

    # Parameters for slicing
    SAMPLING_RATE: int = 500  # Hz
    WINDOW_DURATION: int = 4  # seconds
    WINDOW_SAMPLES: int = SAMPLING_RATE * WINDOW_DURATION  # 2000 samples

    # Sleep stage mapping to numbers
    SLEEP_STAGE_MAP: dict[str, int] = {
        'Wake': 0,
        'W': 0,
        'WAKE': 0,
        'NREM': 1,
        'N': 1,
        'REM': 2,
        'R': 2,
    }

    def __init__(
            self,
            data_dir: Optional[Path] = None,
            output_dir: Optional[Path] = None,
            channel_indices: Optional[list[int]] = None,
        ) -> None:
            
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
        """Find paired file couples using regex-based signature matching.

        Extracts subject ID and experimental conditions from filenames,
        creates a sorted signature, and pairs files with matching signatures.

        Returns:
            List of tuples (eeg_file, annotation_file)
        """
        eeg_dir = self.data_dir / 'eeg'
        eeg_files = sorted(eeg_dir.glob('*.edf'))
        logger.info(f"Found {len(eeg_files)} EDF files in {eeg_dir}")


        labels_dir = self.data_dir / 'labels'
        if not labels_dir.exists():
            logger.warning(f"Labels directory not found at {labels_dir}")
            # Return pairs with only EEG files
            return [(f, None) for f in eeg_files]

        label_files = sorted(labels_dir.glob('*.edf'))
        logger.info(f"Found {len(label_files)} label files in {labels_dir}")


        # Build a signature dictionary for each file
        def extract_signature(filename: str) -> str:
            """Extract subject ID and conditions from filename.

            Returns a sorted signature string like '11_depr_mdl'.
            """
            # Extract subject ID (first sequence of digits)
            subject_match = re.search(r'\d+', filename)
            if not subject_match:
                return ""

            subject_id = subject_match.group()

            # Extract condition keywords
            conditions = []
            keywords = {
                'mdl': r'mdl',
                'psilo': r'psilo|psilocybin',
                'saline': r'saline',
                'depr': r'depr|deprivation',
                'sleep': r'sleep',
            }

            for keyword, pattern in keywords.items():
                if re.search(pattern, filename.lower()):
                    conditions.append(keyword)

            # Create sorted signature
            if conditions:
                conditions_str = '_'.join(sorted(conditions))
                return f"{subject_id}_{conditions_str}"
            else:
                return subject_id

        # Build signature dictionary for all files
        eeg_signatures = {extract_signature(f.stem): f for f in eeg_files}
        label_signatures = {extract_signature(f.stem): f for f in label_files}

        logger.debug(f"EEG signatures: {list(eeg_signatures.keys())}")
        logger.debug(f"Label signatures: {list(label_signatures.keys())}")

        # Pair files by matching signatures
        pairs = []
        for signature, eeg_file in eeg_signatures.items():
            label_file = label_signatures.get(signature)
            pairs.append((eeg_file, label_file))

            if label_file is None:
                logger.warning(
                    f"No matching label file for EEG file {eeg_file.name} "
                    f"(signature: {signature})"
                )
            else:
                logger.debug(
                    f"Paired: {eeg_file.name} <-> {label_file.name} "
                    f"(signature: {signature})"
                )

        return pairs

    def read_edf_file(self, edf_path: Path) -> tuple[np.ndarray, int, list[str]]:
        """Read EDF file and return signal with channel information.

        Args:
            edf_path: Path to EDF file

        Returns:
            Tuple of:
            - signal: ndarray [channels, samples]
            - sampling_rate: int
            - channel_names: list[str]
        """
        if mne is None:
            raise ImportError("mne library is required to read EDF files")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)

        signal = raw.get_data()  # [channels, samples]
        sampling_rate = int(raw.info['sfreq'])
        channel_names = raw.ch_names

        logger.debug(
            f"Loaded file {edf_path.name}: {signal.shape}, "
            f"sampling_rate={sampling_rate}Hz"
        )
        logger.debug(f"Channels: {channel_names}")

        return signal, sampling_rate, channel_names

    def select_channels(
        self,
        signal: np.ndarray,
        channel_names: list[str],
        n_channels: int = 4
    ) -> np.ndarray:
        """Select relevant channels (3x EEG, 1x EMG).

        Logic: If channel_indices are defined, use them. Otherwise, try
        to find channels by name or take first n_channels.

        Args:
            signal: Original signal [channels, samples]
            channel_names: Channel names
            n_channels: Number of channels to select (default 4)

        Returns:
            Selected signal [n_channels, samples]
        """
        if self.channel_indices is not None:
            # If indices are predefined, use them
            indices = self.channel_indices[:n_channels]
            return signal[indices, :]

        # Try to find channels by name
        indices = []
        for target_name in self.CHANNEL_NAMES[:n_channels]:
            found = False
            for i, ch_name in enumerate(channel_names):
                if target_name.lower() in ch_name.lower():
                    indices.append(i)
                    found = True
                    break

            if not found:
                logger.warning(f"Channel {target_name} not found")

        # If we couldn't find channels by name, take first n_channels
        if len(indices) < n_channels:
            logger.warning(
                f"Found only {len(indices)} desired channels, "
                f"taking first {n_channels} channels from signal"
            )
            indices = list(range(min(n_channels, signal.shape[0])))

        selected_signal = signal[indices[:n_channels], :]
        logger.debug(
            f"Selected {selected_signal.shape[0]} channels, "
            f"signal shape: {selected_signal.shape}"
        )

        return selected_signal

    def read_annotations(
        self,
        annotation_path: Path
    ) -> Optional[mne.Annotations]:
        """Read annotation file (EDF format) using mne.read_annotations.

        Args:
            annotation_path: Path to annotation EDF file

        Returns:
            mne.Annotations object or None if file doesn't exist or read fails
        """
        if annotation_path is None or not annotation_path.exists():
            return None

        try:
            annotations = mne.read_annotations(str(annotation_path))
            logger.debug(f"Loaded annotations from {annotation_path.name}")
            logger.debug(f"Annotations contain {len(annotations)} events")
            return annotations

        except Exception as e:
            logger.warning(f"Error reading annotations {annotation_path.name}: {e}")
            return None

    def parse_sleep_stages(
        self,
        annotations: Optional[mne.Annotations],
        signal_duration: float,
        sampling_rate: int
    ) -> np.ndarray:
        """Parse mne.Annotations and return stage for each sample.

        Converts annotation onset and duration to array indices and assigns
        integer labels based on sleep stage descriptions.

        Args:
            annotations: mne.Annotations object from EDF file
            signal_duration: Duration of signal in seconds
            sampling_rate: Sampling frequency

        Returns:
            ndarray [n_samples] with values 0/1/2 or -1 for unknown
        """
        n_samples = int(signal_duration * sampling_rate)
        stages = np.full(n_samples, -1, dtype=np.int32)  # -1 = unknown stage

        if annotations is None:
            logger.warning("No annotations provided, all samples have stage=-1")
            return stages

        # Iterate over mne.Annotations
        try:
            for annotation in annotations:
                # Extract onset (start time in seconds) and duration (in seconds)
                onset_sec = annotation['onset']  # seconds
                duration_sec = annotation['duration']  # seconds
                description = annotation['description'].strip().upper()

                # Convert to sample indices
                start_idx = int(onset_sec * sampling_rate)
                end_idx = int((onset_sec + duration_sec) * sampling_rate)

                # Map description to sleep stage
                stage = self.SLEEP_STAGE_MAP.get(description, -1)

                # Clamp indices to valid range
                start_idx = max(0, min(start_idx, n_samples))
                end_idx = max(0, min(end_idx, n_samples))

                # Assign stage to samples
                if end_idx > start_idx:
                    stages[start_idx:end_idx] = stage
                    logger.debug(
                        f"Annotation: {description} at {onset_sec}s "
                        f"(samples {start_idx}-{end_idx}) -> stage {stage}"
                    )

        except Exception as e:
            logger.warning(f"Error parsing annotations: {e}")

        return stages

    def slice_signal(
        self,
        signal: np.ndarray,
        stages: np.ndarray
    ) -> list[dict[str, torch.Tensor | int]]:
        """Slice signal into 4-second windows with corresponding labels.

        Args:
            signal: ndarray [4, n_samples]
            stages: ndarray [n_samples] with values 0/1/2 or -1

        Returns:
            List of dicts with 'signal' (torch.Tensor) and 'label' (int)
        """
        epochs = []
        n_samples = signal.shape[1]

        # Iterate through signal with 4-second step
        for start_idx in range(0, n_samples - self.WINDOW_SAMPLES + 1, self.WINDOW_SAMPLES):
            end_idx = start_idx + self.WINDOW_SAMPLES

            # Extract window
            window_signal = signal[:, start_idx:end_idx]
            window_stages = stages[start_idx:end_idx]

            # Check if window is valid (has assigned stage)
            if np.any(window_stages == -1):
                # If some sample doesn't have stage, skip window
                logger.debug(f"Window {start_idx}-{end_idx} has unknown stages, skipping")
                continue

            # Determine dominant stage for this window
            label = int(np.bincount(window_stages).argmax())

            # Convert to PyTorch tensor
            signal_tensor = torch.from_numpy(window_signal).float()

            epoch = {
                'signal': signal_tensor,
                'label': label
            }
            epochs.append(epoch)

        logger.info(f"Created {len(epochs)} valid windows from {n_samples} samples")
        return epochs

    def save_epochs(
        self,
        epochs: list[dict[str, torch.Tensor | int]],
        output_prefix: str
    ) -> None:
        """Save epochs to .pt files.

        Args:
            epochs: List of dicts with 'signal' and 'label'
            output_prefix: Prefix for output filenames (without .pt)
        """

        all_signals = torch.stack([e['signal'] for e in epochs])
        all_labels = torch.tensor([e['label'] for e in epochs], dtype=torch.long)

        output_data = {
            'signals': all_signals,
            'labels': all_labels
        }
        
        filename = self.output_dir / f"{output_prefix}_processed.pt"
        torch.save(output_data, filename)

        logger.info(f"Saved {len(epochs)} windows in {filename.name}")

    def process_single_file(
        self,
        eeg_path: Path,
        annotation_path: Optional[Path]
    ) -> int:
        """Process a single EDF file.

        Args:
            eeg_path: Path to EDF file
            annotation_path: Path to annotation file (if exists)

        Returns:
            Number of saved windows
        """
        logger.info(f"Processing file: {eeg_path.name}")

        try:
            # 1. Read EDF file
            signal, sampling_rate, channel_names = self.read_edf_file(eeg_path)

            # Check sampling rate
            if sampling_rate != self.SAMPLING_RATE:
                logger.warning(
                    f"Sampling rate {sampling_rate}Hz differs from "
                    f"expected {self.SAMPLING_RATE}Hz"
                )

            # 2. Select relevant channels
            signal = self.select_channels(signal, channel_names)

            # 3. Read annotations (mne.Annotations object)
            annotations = self.read_annotations(annotation_path)

            # 4. Parse sleep stages
            signal_duration = signal.shape[1] / sampling_rate
            stages = self.parse_sleep_stages(
                annotations,
                signal_duration,
                sampling_rate
            )

            # 5. Slice into windows
            epochs = self.slice_signal(signal, stages)

            if len(epochs) == 0:
                logger.warning(f"File {eeg_path.name} contains no valid windows!")
                return 0

            # 6. Save epochs
            output_prefix = eeg_path.stem
            self.save_epochs(epochs, output_prefix)

            return len(epochs)

        except Exception as e:
            logger.error(f"Error processing {eeg_path.name}: {e}", exc_info=True)
            return 0

    def run(self) -> dict[str, int]:
        """Main method. Process all EDF files.

        Returns:
            Dictionary with statistics (number of processed files, etc.)
        """
        logger.info("=" * 70)
        logger.info("Starting EEG data preprocessing")
        logger.info("=" * 70)

        pairs = self.find_file_pairs()

        total_epochs = 0
        processed_files = 0
        failed_files = 0

        for i, (eeg_file, annotation_file) in enumerate(pairs, 1):
            logger.info(f"\n[{i}/{len(pairs)}] File: {eeg_file.name}")

            n_epochs = self.process_single_file(eeg_file, annotation_file)

            if n_epochs > 0:
                processed_files += 1
                total_epochs += n_epochs
            else:
                failed_files += 1

        logger.info("\n" + "=" * 70)
        logger.info("Preprocessing completed!")
        logger.info(f"Processed: {processed_files} files")
        logger.info(f"Failed: {failed_files} files")
        logger.info(f"Total windows created: {total_epochs}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info("=" * 70)

        return {
            'processed_files': processed_files,
            'failed_files': failed_files,
            'total_epochs': total_epochs,
        }


def main() -> dict[str, int]:
    """Main entry point."""
    # Initialize preprocessor
    preprocessor = EEGPreprocessor()

    # Run preprocessing
    stats = preprocessor.run()

    return stats


if __name__ == '__main__':
    main()
