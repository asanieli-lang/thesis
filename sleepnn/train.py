import csv
import json
import torch
import torch.nn as nn
import torch.optim as optim
import datetime
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

from dataset import SequenceDataset
from model import  SequenceCNN

os.makedirs("outputs", exist_ok=True)

def compute_class_weights(train_labels, num_classes=3):
    unique, counts = np.unique(train_labels, return_counts=True)
    total = len(train_labels)
    weights = torch.zeros(num_classes)
    for i, count in enumerate(counts):
        weights[i] = total / (num_classes * count)
    weights = weights / weights.sum() * num_classes
    return weights

def get_or_compute_class_weights(data_dir: str, split_ratio: float = 0.8, num_classes: int = 3):
    cache_file = "outputs/class_weights_cache.json"
    
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            weights_dict = json.load(f)
            weights = torch.tensor(weights_dict['weights'], dtype=torch.float32)
            return weights
    
    from dataset import MyDataset
    train_dataset = MyDataset(data_dir, split='train', split_ratio=split_ratio)
    all_train_labels = []
    
    for f in train_dataset.files:
        data = torch.load(f, map_location='cpu', weights_only=True)
        all_train_labels.extend(data['labels'].numpy())
        del data
    
    weights = compute_class_weights(np.array(all_train_labels), num_classes=num_classes)
    
    with open(cache_file, 'w') as f:
        json.dump({'weights': weights.tolist(), 'split_ratio': split_ratio}, f)
    
    return weights

class EarlyStopping:
    def __init__(self, patience=5):
        self.patience = patience
        self.counter = 0
        self.best_loss = None
        self.best_model_state = None
    
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        elif val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            self.best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            torch.save(self.best_model_state, "outputs/best_model_checkpoint.pth")
            print(f"  Best model saved (loss: {val_loss:.4f})")
        else:
            self.counter += 1
        return self.counter >= self.patience


def main():
    dist.init_process_group(backend='nccl', timeout=datetime.timedelta(minutes=60))
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    
    log_on_rank_0 = rank == 0
    
    if log_on_rank_0:
        print(f"\n{'='*60}")
        print(f"Distributed Training: {world_size} GPUs")
        print(f"{'='*60}\n")
    
    sequence_length = 10
    BATCH_SIZE = 64
    
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 30
    DATA_DIR = "/mnt/scratch/temporary/asanieli_data/processed_pt"
    TRAIN_RATIO = 0.8
    NUM_WORKERS = 4
    
    train_dataset = SequenceDataset(
            DATA_DIR,
            split='train',
            split_ratio=TRAIN_RATIO,
            sequence_length=sequence_length,
            stride=1
        )

    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=42
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True if NUM_WORKERS > 0 else False
    )
    
    if log_on_rank_0:
        print(f"Train: {len(train_dataset)} sequences")

    test_dataset = SequenceDataset(
            DATA_DIR,
            split='test',
            split_ratio=TRAIN_RATIO,
            sequence_length=sequence_length,
            stride=1
        )

    test_sampler = DistributedSampler(
        test_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        sampler=test_sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True if NUM_WORKERS > 0 else False
    )
    
    if log_on_rank_0:
        print(f"Test: {len(test_dataset)} sequences\n")
    
    if rank == 0:
        class_weights = get_or_compute_class_weights(DATA_DIR, split_ratio=TRAIN_RATIO, num_classes=3)
    else:
        class_weights = torch.zeros(3)

    class_weights = class_weights.to(device)

    if world_size > 1:
        dist.broadcast(class_weights, src=0)
    
    model = SequenceCNN(
            channels=4,
            num_classes=3,
            lstm_hidden=128,
            sequence_length=sequence_length
        ).to(device)
    
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    model_name = "CNN+LSTM"
    output_model_path = "outputs/sleep_lstm_weights.pth"
    output_csv = "outputs/loss_history_lstm.csv"
    output_plot = "outputs/training_metrics_lstm.png"

    
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smooting=0.1)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3) 
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)
    
    if log_on_rank_0:
        print(f"Model initialized: {model_name}\n")
        print(f"{'='*60}")
        print("Starting training...")
        print(f"{'='*60}\n")
    
    epoch_losses = []
    test_accuracies = []
    early_stopping = EarlyStopping(patience=7)
    for epoch in range(NUM_EPOCHS):
        train_sampler.set_epoch(epoch)
        
        model.train()
        running_loss = 0.0
        epoch_loss_sum = 0.0 
        
        for batch_idx, (signals, labels) in enumerate(train_loader):
            signals = signals.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(signals)
            loss = criterion(outputs, labels)
            loss.backward() 
            optimizer.step()
            
            loss_val = loss.item()
            running_loss += loss_val
            epoch_loss_sum += loss_val  
            
            if log_on_rank_0 and (batch_idx + 1) % 500 == 0:
                avg_loss = running_loss / 500
                print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Batch {batch_idx+1}/{len(train_loader)}: {avg_loss:.4f}")
                running_loss = 0.0 
        epoch_loss_tensor = torch.tensor([epoch_loss_sum], device=device)
        dist.all_reduce(epoch_loss_tensor, op=dist.ReduceOp.SUM)
        avg_epoch_loss = epoch_loss_tensor.item() / (len(train_loader) * world_size)
        epoch_losses.append(avg_epoch_loss)
        
        model.eval()
        

        local_metrics = torch.zeros(3, device=device)  # [loss_sum, correct, total]
        
        with torch.no_grad():
            for signals, labels in test_loader:
                signals = signals.to(device)
                labels = labels.to(device)
                outputs = model(signals)
                loss = criterion(outputs, labels)
                _, predicted = torch.max(outputs.data, 1)
                
                local_metrics[0] += loss.item() * labels.size(0)
                local_metrics[1] += (predicted == labels).sum().item()
                local_metrics[2] += labels.size(0)
        
        dist.all_reduce(local_metrics, op=dist.ReduceOp.SUM)
        
        avg_val_loss = local_metrics[0].item() / local_metrics[2].item() if local_metrics[2].item() > 0 else 0.0
        test_accuracy = 100 * local_metrics[1].item() / local_metrics[2].item() if local_metrics[2].item() > 0 else 0.0
        
        test_accuracies.append(test_accuracy)
        
        if log_on_rank_0:
            print(f"Epoch {epoch+1}: Loss={avg_epoch_loss:.4f} | Val Loss={avg_val_loss:.4f} | Acc={test_accuracy:.2f}%")
        
        if rank == 0:
            should_stop = early_stopping(avg_val_loss, model.module)
        else:
            should_stop = False
        
        should_stop_tensor = torch.tensor(int(should_stop), device=device)
        dist.broadcast(should_stop_tensor, src=0)
        
        if should_stop_tensor.item() == 1:
            if rank == 0:
                print(f"\nEarly stopping at epoch {epoch+1}")
            break
        
        avg_val_loss_tensor = torch.tensor(
        [avg_val_loss if rank == 0 else 0.0], 
        device=device)
        dist.broadcast(avg_val_loss_tensor, src=0)
        scheduler.step(avg_val_loss_tensor.item())

    if log_on_rank_0:
        print(f"\n{'='*60}")
        print("Training complete!")
        print(f"{'='*60}\n")
    
    dist.barrier()
    
    if log_on_rank_0:
        if early_stopping.best_model_state is not None:
            model.module.load_state_dict(early_stopping.best_model_state)
        torch.save(model.module.state_dict(), output_model_path)
        print(f"Model saved to {output_model_path}")
        
        with open(output_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "test_accuracy"])  
            for i, loss_val in enumerate(epoch_losses):
                writer.writerow([i + 1, loss_val, test_accuracies[i]])
        print(f"Results saved to {output_csv}")
        
        df = pd.read_csv(output_csv)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        ax1.plot(df['epoch'], df['train_loss'], marker='o', color='#1f77b4', linewidth=2, label='Loss')
        ax1.set_title(f'Training Loss ({model_name})', fontsize=12)
        ax1.set_xlabel('Epoch', fontsize=10)
        ax1.set_ylabel('Loss', fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(df['epoch'])
        ax1.legend()

        ax2.plot(df['epoch'], df['test_accuracy'], marker='s', color='#2ca02c', linewidth=2, label='Accuracy')
        ax2.set_title(f'Test Accuracy ({model_name})', fontsize=12)
        ax2.set_xlabel('Epoch', fontsize=10)
        ax2.set_ylabel('Accuracy (%)', fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(df['epoch'])
        ax2.set_ylim([0, 100])
        ax2.legend()

        plt.tight_layout()
        plt.savefig(output_plot, dpi=300)
        print(f"Plot saved to {output_plot}\n")
    
    dist.destroy_process_group()

if __name__ == "__main__":
    main()