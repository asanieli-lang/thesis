import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import sys

from dataset import MyDataset
from model import SleepCNN

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

    # Confusion matrix
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Wake', 'NREM', 'REM'], 
                yticklabels=['Wake', 'NREM', 'REM'])
    plt.xlabel('Predikce sítě')
    plt.ylabel('Skutečnost (Expert)')
    plt.title('Confusion Matrix - Baseline 1D-CNN')
    plt.savefig('confusion_matrix.png')
    print("Confusion matrix uložena do 'confusion_matrix.png'")
    
    return accuracy, cm

def main():
    print("=== Inicializace evaluačního procesu ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Výpočty budou probíhat na: {device}")

    # Cesty a parametry
    MODEL_PATH = "outputs/sleep_cnn_weights.pth"  # Cesta k naučenému modelu
    TEST_DATA_DIR = "/mnt/scratch/temporary/asanieli_data/processed_pt"  # Testovací data
    BATCH_SIZE = 512
    TRAIN_RATIO = 0.8  # Stejný poměr jako při trénování!

    # Načtení modelu
    print(f"\nNačítám model z '{MODEL_PATH}'...")
    model = SleepCNN().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print("Model úspěšně načten.")

    # Načtení testovacích dat (POUZE test split!)
    print(f"\nNačítám testovací data z '{TEST_DATA_DIR}'...")
    test_dataset = MyDataset(TEST_DATA_DIR, split='test', split_ratio=TRAIN_RATIO)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    print(f"Testovací data připravena. Počet vzorků: {len(test_dataset)}")

    # Evaluace
    print("\n=== Zahajuji evaluaci ===")
    accuracy, cm = evaluate_model(model, test_loader, device)
    
    print("\n=== Evaluace dokončena! ===")

if __name__ == "__main__":
    main()