from dataset import EEGBatchedDataset
from torch.utils.data import DataLoader

print("--- Testing Dataset ---")
# Point it to your processed data folder
data_path = "/mnt/scratch/temporary/asanieli_data/processed_pt"

# Initialize the dataset
dataset = EEGBatchedDataset(data_path)

print(f"Total dataset length: {len(dataset)}")

# Fetch the very first sample directly
signal, label = dataset[0]
print(f"Sample 0 - Signal shape: {signal.shape}, Label: {label}")

# Test if PyTorch DataLoader can batch it properly (e.g., 32 windows at once)
print("\n--- Testing DataLoader ---")
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Grab exactly one batch
for batch_signals, batch_labels in dataloader:
    print(f"Batch signals shape: {batch_signals.shape}") # Should be [32, 4, 2000]
    print(f"Batch labels shape: {batch_labels.shape}")   # Should be [32]
    break