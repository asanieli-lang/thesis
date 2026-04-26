import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np

from dataset import SequenceDataset
from model import SequenceCNN

os.makedirs("outputs", exist_ok=True)

def evaluate_model(model, test_loader, device):
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

    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(classification_report(all_labels, all_preds, target_names=['Wake', 'NREM', 'REM']))
    print(f"Overall Accuracy: {accuracy_score(all_labels, all_preds):.4f}")
    print(f"F1 Score (Macro): {f1_score(all_labels, all_preds, average='macro'):.4f}\n")

    cm = confusion_matrix(all_labels, all_preds)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=True, fmt='.1f', cmap='Blues', 
                xticklabels=['Wake', 'NREM', 'REM'], 
                yticklabels=['Wake', 'NREM', 'REM'])
    plt.title('Confusion Matrix (%)')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('outputs/final_eval_cm.png')
    print(f"Confusion matrix saved to outputs/final_eval_cm.png")
    
    return accuracy_score(all_labels, all_preds), cm

def main():
    print("\n" + "="*50)
    print("EVALUATION")
    print("="*50 + "\n")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    MODEL_PATH = "outputs/sleep_lstm_weights.pth"
    TEST_DATA_DIR = "/mnt/scratch/temporary/asanieli_data/processed_pt"
    BATCH_SIZE = 64
    NUM_WORKERS = 4

    print(f"Loading model from {MODEL_PATH}...")
    model = SequenceCNN(channels=4, num_classes=3, lstm_hidden=128, sequence_length=10).to(device)
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    print("Model loaded.\n")

    print("Loading test data...")
    test_dataset = SequenceDataset(TEST_DATA_DIR, split='test', split_ratio=0.8, sequence_length=10)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    print(f"Test sequences: {len(test_dataset)}\n")

    print("Evaluating...")
    accuracy, cm = evaluate_model(model, test_loader, device)
    
    print(f"{'='*50}")
    print("Done!\n")

if __name__ == "__main__":
    main()