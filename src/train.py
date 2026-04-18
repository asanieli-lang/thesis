import csv  # NOVÉ: Vestavěná knihovna pro ukládání tabulek
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt

from dataset import MyDataset
from model import SleepCNN

def compute_class_weights(train_labels, num_classes=3):
    """
    Vypočítá váhy pro třídy na základě jejich reprezentace v datasetu.
    Méně reprezentované třídy dostanou vyšší váhu.
    """
    unique, counts = np.unique(train_labels, return_counts=True)
    total = len(train_labels)
    
    weights = torch.zeros(num_classes)
    for i, count in enumerate(counts):
        weights[i] = total / (num_classes * count)
    
    # Normalizace
    weights = weights / weights.sum() * num_classes
    
    print(f"Třídní váhy: {weights.numpy()}")
    return weights

def main():
    print("=== Inicializace trénovacího procesu ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Výpočty budou probíhat na: {device}")

    BATCH_SIZE = 512
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 20
    DATA_DIR = "/mnt/scratch/temporary/asanieli_data/processed_pt"
    TRAIN_RATIO = 0.8  # 80% trénování, 20% testování

    print("Načítám trénovací dataset...")
    train_dataset = MyDataset(DATA_DIR, split='train', split_ratio=TRAIN_RATIO)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    print(f"Trénovací dataset připraven. Počet dávek (batches): {len(train_loader)}")

    print("\nNačítám testovací dataset...")
    test_dataset = MyDataset(DATA_DIR, split='test', split_ratio=TRAIN_RATIO)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    print(f"Testovací dataset připraven. Počet vzorků: {len(test_dataset)}")

    # Vypočítání třídních vah na základě trénovacích dat
    print("\nVýpočet třídních vah...")
    all_train_labels = []
    for _, labels in train_loader:
        all_train_labels.extend(labels.numpy())
    class_weights = compute_class_weights(np.array(all_train_labels), num_classes=3)
    class_weights = class_weights.to(device)

    model = SleepCNN().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("\n=== Zahajuji trénování ===")
    
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
                print(f"Epocha [{epoch+1}/{NUM_EPOCHS}] | Dávka [{batch_idx+1}/{len(train_loader)}] | Ztráta (Loss): {avg_loss:.4f}")
                running_loss = 0.0 

        avg_epoch_loss = epoch_loss_sum / len(train_loader)
        epoch_losses.append(avg_epoch_loss)
        print(f"--- Konec epochy {epoch+1} | Celková průměrná ztráta: {avg_epoch_loss:.4f} ---")

        # Validace na testovacích datech
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for signals, labels in test_loader:
                signals = signals.to(device)
                labels = labels.to(device)
                outputs = model(signals)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        test_accuracy = 100 * correct / total
        test_accuracies.append(test_accuracy)
        print(f"    Testovací přesnost (Accuracy): {test_accuracy:.2f}%")

    print("=== Trénování úspěšně dokončeno! ===")
    
    torch.save(model.state_dict(), "sleep_cnn_weights.pth")
    print("Naučené váhy byly uloženy do souboru 'sleep_cnn_weights.pth'.")
    

    with open("loss_history.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "test_accuracy"])  
        for i, loss_val in enumerate(epoch_losses):
            writer.writerow([i + 1, loss_val, test_accuracies[i]])
            
    print("Historie učení byla uložena do 'loss_history.csv'.")
    df = pd.read_csv('loss_history.csv')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    ax1.plot(df['epoch'], df['train_loss'], marker='o', color='#1f77b4', linewidth=2, label='Training Loss')
    ax1.set_title('Training Loss Convergence: 1D-CNN Sleep Classifier', fontsize=14)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Cross-Entropy Loss', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.set_xticks(df['epoch'])
    ax1.legend()

    # Test accuracy plot
    ax2.plot(df['epoch'], df['test_accuracy'], marker='s', color='#ff7f0e', linewidth=2, label='Test Accuracy')
    ax2.set_title('Test Accuracy: 1D-CNN Sleep Classifier', fontsize=14)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.set_xticks(df['epoch'])
    ax2.legend()

    plt.tight_layout()
    plt.savefig('training_metrics.png', dpi=300)
    print("Grafy byly uloženy do 'training_metrics.png'.")

if __name__ == "__main__":
    main()