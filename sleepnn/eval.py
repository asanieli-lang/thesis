import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score, f1_score,
    precision_score, recall_score, cohen_kappa_score, log_loss, explained_variance_score
)
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np

from dataset import SequenceDataset
from model import SequenceCNN

RUN_ID = 7

os.makedirs("outputs", exist_ok=True)

def plot_hypnogram(all_labels, all_preds, run_id, num_samples=1000):
    plt.figure(figsize=(15, 5))
    plt.plot(all_labels[:num_samples], label='Ground Truth', alpha=0.7, color='black', linewidth=2)
    plt.plot(all_preds[:num_samples], label='Predicted', alpha=0.5, color='red', linestyle='--', linewidth=1.5)
    plt.yticks([0, 1, 2], ['Wake', 'NREM', 'REM'])
    plt.xlabel('Sequence Index')
    plt.ylabel('Sleep Stage')
    plt.title(f'Hypnogram (First {num_samples} sequences)')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"outputs/hypnogram_{run_id}.png", dpi=300)
    print(f"Hypnogram saved to outputs/hypnogram_{run_id}.png")

def plot_f1_per_class(all_labels, all_preds, run_id):
    f1_per_class = f1_score(all_labels, all_preds, average=None)
    
    plt.figure(figsize=(8, 5))
    bars = plt.bar(['Wake', 'NREM', 'REM'], f1_per_class, color=['#1f77b4', '#2ca02c', '#ff7f0e'], alpha=0.7)
    plt.ylabel('F1 Score', fontsize=12)
    plt.title('F1 Score per Sleep Stage', fontsize=14, fontweight='bold')
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3, axis='y')
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"outputs/f1_per_class_{run_id}.png", dpi=300)
    print(f"F1 per-class chart saved to outputs/f1_per_class_{run_id}.png")

def plot_validation_metrics(all_labels, all_preds, all_probs, run_id):
    """Plot all validation metrics (Precision, Recall, F1, Accuracy, Cohen Kappa, Log Loss, Explained Variance)"""
    precision_macro = precision_score(all_labels, all_preds, average='macro')
    recall_macro = recall_score(all_labels, all_preds, average='macro')
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    accuracy = accuracy_score(all_labels, all_preds)
    kappa = cohen_kappa_score(all_labels, all_preds)
    
    logloss = log_loss(all_labels, all_probs)
    
    explained_var = explained_variance_score(all_labels, all_preds, multioutput='raw_values')
    explained_var_mean = np.mean(explained_var)
    
    metrics = {
        'Precision': precision_macro,
        'Recall': recall_macro,
        'F1-Score': f1_macro,
        'Accuracy': accuracy,
        'Cohen Kappa': kappa,
        'Log Loss': logloss,
        'Explained Variance': explained_var_mean
    }
    
    fig, ax = plt.subplots(figsize=(10, 6))
    metric_names = list(metrics.keys())
    metric_values = list(metrics.values())
    
    bars = ax.bar(metric_names, metric_values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2'], alpha=0.7)
    
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title('Validation Metrics', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xticklabels(metric_names, rotation=45, ha='right')
    
    for i, (bar, name) in enumerate(zip(bars, metric_names)):
        height = bar.get_height()
        if name == 'Log Loss':
            display_value = logloss
        else:
            display_value = metrics[name]
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{display_value:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"outputs/validation_metrics_{run_id}.png", dpi=300)
    print(f"Validation metrics chart saved to outputs/validation_metrics_{run_id}.png")
    
    print(f"\nDetailed Metrics:")
    print(f"  Precision (macro):      {precision_macro:.4f}")
    print(f"  Recall (macro):         {recall_macro:.4f}")
    print(f"  F1-Score (macro):       {f1_macro:.4f}")
    print(f"  Accuracy:               {accuracy:.4f}")
    print(f"  Cohen Kappa:            {kappa:.4f}")
    print(f"  Log Loss:               {logloss:.4f}")
    print(f"  Explained Variance:     {explained_var_mean:.4f}")

def evaluate_model(model, test_loader, device):
    model.eval() 
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for signals, labels in test_loader:
            signals = signals.to(device)
            labels = labels.to(device)
            
            outputs = model(signals)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

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
    plt.tight_layout()
    plt.savefig(f"outputs/final_eval_cm_{RUN_ID}.png", dpi=300)
    print(f"Confusion matrix saved to outputs/final_eval_cm_{RUN_ID}.png")
    
    plot_hypnogram(all_labels, all_preds, RUN_ID)
    plot_f1_per_class(all_labels, all_preds, RUN_ID)
    plot_validation_metrics(all_labels, all_preds, all_probs, RUN_ID)
    
    return accuracy_score(all_labels, all_preds), cm

def main():
    print("\n" + "="*50)
    print("EVALUATION")
    print("="*50 + "\n")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    MODEL_PATH = f"outputs/sleep_lstm_weights_{RUN_ID}.pth"
    TEST_DATA_DIR = "/mnt/scratch/temporary/asanieli_data/processed_pt"
    BATCH_SIZE = 64
    NUM_WORKERS = 4

    print(f"Loading model from {MODEL_PATH}...")
    model = SequenceCNN(channels=4, num_classes=3, lstm_hidden=64).to(device)
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    print("Model loaded.\n")

    print("Loading test data...")
    test_dataset = SequenceDataset(TEST_DATA_DIR, split='test', split_ratio=0.8)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    print(f"Test sequences: {len(test_dataset)}\n")

    print("Evaluating...")
    accuracy, cm = evaluate_model(model, test_loader, device)
    
    print(f"{'='*50}")
    print("Done!\n")

if __name__ == "__main__":
    main()