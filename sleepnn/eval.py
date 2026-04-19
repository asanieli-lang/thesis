import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np

from dataset import MyDataset
from model import SleepCNN

os.makedirs("outputs", exist_ok=True)

def evaluate_model(model, test_loader, device):
    """
    Vyhodnotí model na testovacích datech.
    """
    model.eval() 
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for signals, labels in test_loader:
            signals = signals.to(device)
            labels = labels.to(device)
            
            outputs = model(signals)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Vypočet metrik
    accuracy = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)

    print(f"\n=== VÝSLEDKY EVALUACE ===")
    print(f"Přesnost (Accuracy): {accuracy:.4f}")
    print(f"\nKlasifikační report:")
    print(classification_report(all_labels, all_preds, target_names=['Wake', 'NREM', 'REM']))

    # Confusion matrix: Ukazuje, jak dobře se síť orientuje v každé třídě
    # Diagonála = správné predikce. Off-diagonála = chyby (co si síť spletla).
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Absolutní počty
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['Wake', 'NREM', 'REM'], 
                yticklabels=['Wake', 'NREM', 'REM'], ax=ax1,
                cbar_kws={'label': 'Počet vzorků'})
    ax1.set_xlabel('Predikce sítě', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Skutečnost (Expert)', fontsize=11, fontweight='bold')
    ax1.set_title('Confusion Matrix: Absolutní počty\n(Diagonála = správné, ostatní = chyby)', 
                  fontsize=12, fontweight='bold')
    
    # Normalizovaná matrice (procentuálně)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    sns.heatmap(cm_norm, annot=True, fmt='.1f', cmap='RdYlGn', cbar=True, vmin=0, vmax=100,
                xticklabels=['Wake', 'NREM', 'REM'], 
                yticklabels=['Wake', 'NREM', 'REM'], ax=ax2,
                cbar_kws={'label': 'Procenta (%)'})
    ax2.set_xlabel('Predikce sítě', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Skutečnost (Expert)', fontsize=11, fontweight='bold')
    ax2.set_title('Confusion Matrix: Normalizovaná (Recall %)\n(Ukazuje přesnost klasifikace každé třídy)', 
                  fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('outputs/confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("Confusion matrix uložena do 'outputs/confusion_matrix.png'")
    
    return accuracy, cm

def main():
    print("=== Inicializace evaluačního procesu ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Výpočty budou probíhat na: {device}")

    MODEL_PATH = "outputs/sleep_cnn_weights.pth"
    TEST_DATA_DIR = "/mnt/scratch/temporary/asanieli_data/processed_pt"
    BATCH_SIZE = 512

    print(f"\nLoading model from {MODEL_PATH}...")
    model = SleepCNN().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print("Model loaded.")

    print(f"\nLoading test data...")
    test_dataset = MyDataset(TEST_DATA_DIR, split='test', split_ratio=0.8)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    print(f"Test samples: {len(test_dataset)}")

    print("\nRunning evaluation...")
    accuracy, cm = evaluate_model(model, test_loader, device)
    
    print("\nEvaluation complete.")

if __name__ == "__main__":
    main()