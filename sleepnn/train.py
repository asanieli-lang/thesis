import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

from dataset import MyDataset
from model import SleepCNN

os.makedirs("outputs", exist_ok=True)

def compute_class_weights(train_labels, num_classes=3):
    unique, counts = np.unique(train_labels, return_counts=True)
    total = len(train_labels)
    weights = torch.zeros(num_classes)
    for i, count in enumerate(counts):
        weights[i] = total / (num_classes * count)
    weights = weights / weights.sum() * num_classes
    return weights

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    BATCH_SIZE = 512
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 20
    DATA_DIR = "/mnt/scratch/temporary/asanieli_data/processed_pt"
    TRAIN_RATIO = 0.8
    NUM_WORKERS = 4

    print("Loading train dataset...")
    train_dataset = MyDataset(DATA_DIR, split='train', split_ratio=TRAIN_RATIO)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    print(f"Train samples: {len(train_dataset)}, batches: {len(train_loader)}\n")

    print("Loading test dataset...")
    test_dataset = MyDataset(DATA_DIR, split='test', split_ratio=TRAIN_RATIO)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    print(f"Test samples: {len(test_dataset)}\n")
    
    print("Computing class weights...")
    all_train_labels = []
    for _, labels in train_loader:
        all_train_labels.extend(labels.numpy())
    class_weights = compute_class_weights(np.array(all_train_labels), num_classes=3)
    class_weights = class_weights.to(device)
    print(f"Class weights: {class_weights}\n")

    print("Initializing model...")
    model = SleepCNN().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-3) 
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)
    print("Model ready.\n")
    
    print("=" * 60)
    print("Starting training...")
    print("=" * 60 + "\n")
    
    epoch_losses = []
    test_accuracies = []
    
    for epoch in range(NUM_EPOCHS):
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
            
            if (batch_idx + 1) % 500 == 0:
                avg_loss = running_loss / 500
                print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Batch [{batch_idx+1}/{len(train_loader)}] Loss: {avg_loss:.4f}")
                running_loss = 0.0 

        avg_epoch_loss = epoch_loss_sum / len(train_loader)
        epoch_losses.append(avg_epoch_loss)
        print(f"Epoch {epoch+1} avg loss: {avg_epoch_loss:.4f}")

        model.eval()
        correct = 0
        total = 0
        val_loss_sum = 0.0
        with torch.no_grad():
            for signals, labels in test_loader:
                signals = signals.to(device)
                labels = labels.to(device)
                outputs = model(signals)
                loss = criterion(outputs, labels)
                val_loss_sum += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_val_loss = val_loss_sum / len(test_loader)
        test_accuracy = 100 * correct / total
        test_accuracies.append(test_accuracy)
        print(f"Test accuracy: {test_accuracy:.2f}% | Validation loss: {avg_val_loss:.4f}")
        scheduler.step(avg_val_loss)
        print()

    print("=" * 60)
    print("Training complete!")
    print("=" * 60 + "\n")
    
    print("Saving model and results...")
    torch.save(model.state_dict(), "outputs/sleep_cnn_weights.pth")
    print("✓ Model saved to outputs/sleep_cnn_weights.pth")
    
    with open("outputs/loss_history.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "test_accuracy"])  
        for i, loss_val in enumerate(epoch_losses):
            writer.writerow([i + 1, loss_val, test_accuracies[i]])
    print("✓ History saved to outputs/loss_history.csv")
    
    print("Generating plots...")
    df = pd.read_csv('outputs/loss_history.csv')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(df['epoch'], df['train_loss'], marker='o', color='#1f77b4', linewidth=2, label='Loss')
    ax1.set_title('Training Loss', fontsize=12)
    ax1.set_xlabel('Epoch', fontsize=10)
    ax1.set_ylabel('Loss', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(df['epoch'])
    ax1.legend()

    ax2.plot(df['epoch'], df['test_accuracy'], marker='s', color='#2ca02c', linewidth=2, label='Accuracy')
    ax2.set_title('Test Accuracy', fontsize=12)
    ax2.set_xlabel('Epoch', fontsize=10)
    ax2.set_ylabel('Accuracy (%)', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(df['epoch'])
    ax2.set_ylim([0, 100])
    ax2.legend()

    plt.tight_layout()
    plt.savefig('outputs/training_metrics.png', dpi=300)
    print("✓ Plots saved to outputs/training_metrics.png")
    print("\nAll done! ✨")

if __name__ == "__main__":
    main()