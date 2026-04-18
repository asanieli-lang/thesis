import csv  # NOVÉ: Vestavěná knihovna pro ukládání tabulek
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import pandas as pd
import matplotlib.pyplot as plt

from dataset import MyDataset
from model import SleepCNN

def main():
    print("=== Inicializace trénovacího procesu ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Výpočty budou probíhat na: {device}")

    BATCH_SIZE = 512
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 20
    DATA_DIR = "/mnt/scratch/temporary/asanieli_data/processed_pt"

    print("Načítám dataset...")
    dataset = MyDataset(DATA_DIR)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    print(f"Dataset připraven. Počet dávek (batches) v jedné epoše: {len(dataloader)}")

    model = SleepCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("\n=== Zahajuji trénování ===")
    
    epoch_losses = [] 
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        
        running_loss = 0.0
        epoch_loss_sum = 0.0 
        
        for batch_idx, (signals, labels) in enumerate(dataloader):
            
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
                print(f"Epocha [{epoch+1}/{NUM_EPOCHS}] | Dávka [{batch_idx+1}/{len(dataloader)}] | Ztráta (Loss): {avg_loss:.4f}")
                running_loss = 0.0 


        avg_epoch_loss = epoch_loss_sum / len(dataloader)
        epoch_losses.append(avg_epoch_loss)
        print(f"--- Konec epochy {epoch+1} | Celková průměrná ztráta: {avg_epoch_loss:.4f} ---")

    print("=== Trénování úspěšně dokončeno! ===")
    
    torch.save(model.state_dict(), "sleep_cnn_weights.pth")
    print("Naučené váhy byly uloženy do souboru 'sleep_cnn_weights.pth'.")
    

    with open("loss_history.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "loss"])  
        for i, loss_val in enumerate(epoch_losses):
            writer.writerow([i + 1, loss_val])
            
    print("Historie učení byla uložena do 'loss_history.csv'.")
    df = pd.read_csv('loss_history.csv')

    plt.figure(figsize=(10, 6))
    plt.plot(df['epoch'], df['loss'], marker='o', color='#1f77b4', linewidth=2, label='Training Loss')

    plt.title('Training Loss Convergence: 1D-CNN Sleep Classifier', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Cross-Entropy Loss', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(df['epoch']) # Show every epoch number
    plt.legend()

    plt.savefig('training_loss_plot.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    main()