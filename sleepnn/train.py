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
from sklearn.metrics import f1_score

from dataset import SequenceDataset
from model import  SequenceCNN

RUN_ID = os.environ.get("SLEEPNN_RUN_ID")
SEQ_LEN = 32
STRIDE = 5
LSTM_HIDDEN = 128
NUM_HEADS = 4
ATTN_DROPOUT = 0.1 
LSTM_LAYERS = 2
LSTM_DROPOUT = 0.3   
CLASSIFIER_DROPOUT1 = 0.5
CLASSIFIER_DROPOUT2 = 0.3
LABEL_SMOOTH = 0.02
WEIGHT_DECAY = 5e-4
LEARNING_RATE = 1e-4
WARMUP_EPOCHS = 3
PATIENCE = 25

USE_CLASS_WEIGHTS = True
os.makedirs("outputs", exist_ok=True)

def compute_class_weights(train_labels, num_classes=3):
    """Compute class weights using sqrt normalization to handle imbalance."""
    counts = np.bincount(train_labels.astype(int), minlength=num_classes).astype(np.float32)
    counts = np.clip(counts, 1.0, None)
    counts_t = torch.tensor(counts, dtype=torch.float32)
    total = counts_t.sum()
    weights = torch.sqrt(total / (num_classes * counts_t))
    weights = weights / weights.sum() * num_classes
    return weights

def get_or_compute_class_weights(data_dir: str, split_ratio: float = 0.8, num_classes: int = 3):
    """Load cached class weights or compute and cache them.    """
    cache_file = f"outputs/class_weights_cache_{RUN_ID}_sqrt.json"
    
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
    
    weights = compute_class_weights(np.array(all_train_labels), num_classes=num_classes)
    
    with open(cache_file, 'w') as f:
        json.dump({'weights': weights.tolist(), 'split_ratio': split_ratio}, f)
    
    return weights

class EarlyStopping:
    """Stop training when F1 score plateaus to prevent overfitting."""
    def __init__(self, patience):
        """Initialize early stopping tracker with patience threshold."""
        self.patience = patience
        self.counter = 0
        self.best_f1 = None
        self.best_model_state = None
    
    def __call__(self, f1, model):
        """Check stopping condition and update best model."""
        if self.best_f1 is None:
            self.best_f1 = f1
            self.best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        elif f1 > self.best_f1:
            self.best_f1 = f1
            self.counter = 0
            self.best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.save_best_checkpoint()
        else:
            self.counter += 1
        return self.counter >= self.patience

    def save_best_checkpoint(self):
        """Save best model state to checkpoint file."""
        global RUN_ID
        torch.save(self.best_model_state, f"outputs/best_model_checkpoint_{RUN_ID}.pth")


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance. """
    def __init__(self, weight=None, gamma=2.0, label_smoothing=0.02):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.label_smoothing = label_smoothing
    
    def forward(self, inputs, targets):
        """Compute focal loss with label smoothing for hard negative focus."""
        ce_loss = nn.functional.cross_entropy(
            inputs, targets, weight=self.weight, 
            label_smoothing=self.label_smoothing, 
            reduction='none')
        # Downweight easy examples for hard negative focus
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


def setup_distributed():
    """Initialize distributed training and return rank, world_size, device."""
    dist.init_process_group(backend='nccl', timeout=datetime.timedelta(minutes=60))
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    return rank, world_size, local_rank, device


def create_dataloaders(data_dir: str, rank: int, world_size: int, batch_size: int = 16, num_workers: int = 4):
    """Create train and test dataloaders with distributed sampling."""
    train_dataset = SequenceDataset(data_dir, split='train', split_ratio=0.8, sequence_length=SEQ_LEN, stride=STRIDE)
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=42)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, 
                              num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0)
    
    test_dataset = SequenceDataset(data_dir, split='test', split_ratio=0.8, sequence_length=SEQ_LEN, stride=SEQ_LEN)
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler, 
                             num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0)
    
    return train_loader, test_loader, train_sampler, train_dataset, test_dataset


def create_model_and_loss(rank: int, world_size: int, device, data_dir: str):
    """Create model, optimizer, criterion with distributed setup."""
    # Compute class weights on rank 0 and broadcast
    if rank == 0:
        class_weights = get_or_compute_class_weights(data_dir, split_ratio=0.8, num_classes=3)
    else:
        class_weights = torch.zeros(3)
    
    class_weights = class_weights.to(device)
    if world_size > 1:
        dist.broadcast(class_weights, src=0)
    
    # Initialize model
    model = SequenceCNN(channels=4, num_classes=3, lstm_hidden=LSTM_HIDDEN, attn_dropout=ATTN_DROPOUT,
                        num_heads=NUM_HEADS, lstm_layers=LSTM_LAYERS, lstm_dropout=LSTM_DROPOUT,
                        classifier_dropout1=CLASSIFIER_DROPOUT1, classifier_dropout2=CLASSIFIER_DROPOUT2).to(device)
    model = DDP(model, device_ids=[int(os.environ.get('LOCAL_RANK', 0))], output_device=int(os.environ.get('LOCAL_RANK', 0)))
    
    # Loss, optimizer, scheduler
    criterion = FocalLoss(weight=class_weights, gamma=2.0, label_smoothing=LABEL_SMOOTH)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=PATIENCE//2, factor=0.5)
    
    return model, criterion, optimizer, scheduler


def train_epoch(model, train_loader, criterion, optimizer, device, epoch, log_on_rank_0, first_epoch_losses, num_epochs):
    """Train for one epoch and return average loss."""
    model.train()
    epoch_running_loss = 0.0
    epoch_train_total = 0.0
    running_loss = 0.0
    train_total = 0.0
    
    for batch_idx, (signals, labels, _) in enumerate(train_loader):
        signals, labels = signals.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(signals)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        loss_val = loss.item()
        running_loss += loss_val * labels.size(0)
        train_total += labels.size(0)
        epoch_running_loss += loss_val * labels.size(0)
        epoch_train_total += labels.size(0)
        
        if epoch == 0:
            first_epoch_losses.append(loss_val)
        
        if log_on_rank_0 and (batch_idx + 1) % 500 == 0:
            avg_loss = running_loss / train_total
            print(f"Epoch {epoch+1}/{num_epochs} - Batch {batch_idx+1}/{len(train_loader)}: {avg_loss:.4f}")
            running_loss = 0.0
            train_total = 0.0
    
    return epoch_running_loss / epoch_train_total if epoch_train_total > 0 else 0.0


def evaluate_epoch(model, test_loader, criterion, device):
    """Evaluate on test set and return metrics."""
    model.eval()
    test_loss = 0.0
    test_correct = 0.0
    test_total = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for signals, labels, _ in test_loader:
            signals, labels = signals.to(device), labels.to(device)
            outputs = model(signals)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            
            test_loss += loss.item() * labels.size(0)
            test_correct += (predicted == labels).sum().item()
            test_total += labels.size(0)
            
            all_preds.append(predicted.cpu())
            all_labels.append(labels.cpu())
    
    avg_test_loss = test_loss / test_total if test_total > 0 else 0.0
    test_accuracy = 100 * test_correct / test_total if test_total > 0 else 0.0
    
    f1_macro = 0.0
    if test_total > 0:
        final_preds = torch.cat(all_preds).numpy()
        final_labels = torch.cat(all_labels).numpy()
        f1_macro = f1_score(final_labels, final_preds, average='macro')
    
    return avg_test_loss, test_accuracy, f1_macro


def save_results(model, train_losses, test_losses, test_accuracies, epoch_f1_scores, first_epoch_losses):
    """Save model, metrics, and plots."""
    output_model_path = f"outputs/sleep_lstm_weights_{RUN_ID}.pth"
    output_csv = f"outputs/loss_history_lstm_{RUN_ID}.csv"
    output_plot = f"outputs/training_metrics_lstm_{RUN_ID}.png"
    
    if hasattr(model, 'module'):
        torch.save(model.module.state_dict(), output_model_path)
    else:
        torch.save(model.state_dict(), output_model_path)
    print(f"Model saved to {output_model_path}")
    
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "test_loss", "test_accuracy", "f1_macro"])
        for i, train_loss_val in enumerate(train_losses):
            writer.writerow([i + 1, train_loss_val, test_losses[i], test_accuracies[i], epoch_f1_scores[i]])
    print(f"Results saved to {output_csv}")
    
    # Plot first epoch losses
    if first_epoch_losses:
        plt.figure(figsize=(10, 4))
        plt.plot(first_epoch_losses, linewidth=1.5, color='#1f77b4')
        plt.xlabel('Batch Index')
        plt.ylabel('Loss')
        plt.title('Training Loss During First Epoch (Batch-level)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"outputs/first_epoch_loss_{RUN_ID}.png", dpi=300)
        plt.close()
    
    # Plot training metrics
    df = pd.read_csv(output_csv)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(df['epoch'], df['train_loss'], marker='o', color='#1f77b4', linewidth=2, label='Train Loss')
    ax1.plot(df['epoch'], df['test_loss'], marker='s', color='#ff7f0e', linewidth=2, label='Test Loss')
    ax1.set_title('Training & Test Loss (CNN+LSTM)', fontsize=12)
    ax1.set_xlabel('Epoch', fontsize=10)
    ax1.set_ylabel('Loss', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(df['epoch'])
    ax1.legend()
    
    ax2.plot(df['epoch'], df['test_accuracy'], marker='s', color='#2ca02c', linewidth=2, label='Accuracy')
    ax2.set_title('Test Accuracy (CNN+LSTM)', fontsize=12)
    ax2.set_xlabel('Epoch', fontsize=10)
    ax2.set_ylabel('Accuracy (%)', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(df['epoch'])
    ax2.set_ylim([0, 100])
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(output_plot, dpi=300)
    print(f"Plot saved to {output_plot}\n")


def set_optimizer_lr(optimizer, lr):
    """Update optimizer learning rate across all parameter groups."""
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def main():
    """Orchestrate distributed multi-GPU training with early stopping."""
    rank, world_size, local_rank, device = setup_distributed()
    log_on_rank_0 = rank == 0
    
    data_dir = os.environ.get("SLEEPNN_DATA_DIR", "/mnt/scratch/temporary/asanieli_data/processed_pt")
    train_loader, test_loader, train_sampler, train_dataset, test_dataset = create_dataloaders(data_dir, rank, world_size)
    model, criterion, optimizer, scheduler = create_model_and_loss(rank, world_size, device, data_dir)
    
    train_losses = []
    test_losses = []
    test_accuracies = []
    epoch_f1_scores = []
    early_stopping = EarlyStopping(patience=PATIENCE)
    first_epoch_losses = []
    
    for epoch in range(100):
        # Apply linear learning rate warmup
        if epoch < WARMUP_EPOCHS:
            warmup_lr = LEARNING_RATE * (epoch + 1) / WARMUP_EPOCHS
            set_optimizer_lr(optimizer, warmup_lr)
        
        train_sampler.set_epoch(epoch)
        avg_train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch, log_on_rank_0, first_epoch_losses, 100)
        train_losses.append(avg_train_loss)
        
        avg_test_loss, test_accuracy, f1_macro = evaluate_epoch(model, test_loader, criterion, device)
        test_losses.append(avg_test_loss)
        test_accuracies.append(test_accuracy)
        epoch_f1_scores.append(f1_macro)
        
        if log_on_rank_0:
            print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f} | Test Loss={avg_test_loss:.4f} | Acc={test_accuracy:.2f}% | F1: {f1_macro:.4f}")
        
        # Early stopping check
        if rank == 0:
            should_stop = early_stopping(f1_macro, model.module if hasattr(model, 'module') else model)
        else:
            should_stop = False
        
        should_stop_tensor = torch.tensor(int(should_stop), device=device)
        dist.broadcast(should_stop_tensor, src=0)
        
        if should_stop_tensor.item() == 1:
            if log_on_rank_0:
                print(f"\nEarly stopping at epoch {epoch+1}")
            break
        
        if (epoch + 1) > WARMUP_EPOCHS:
            scheduler.step(avg_test_loss)
    
    if log_on_rank_0:
        print(f"\n{'='*60}\nTraining complete!\n{'='*60}\n")
    
    dist.barrier()
    
    if log_on_rank_0:
        if early_stopping.best_model_state is not None:
            if hasattr(model, 'module'):
                model.module.load_state_dict(early_stopping.best_model_state)
            else:
                model.load_state_dict(early_stopping.best_model_state)
        save_results(model, train_losses, test_losses, test_accuracies, epoch_f1_scores, first_epoch_losses)
    
    dist.destroy_process_group()

if __name__ == "__main__":
    main()