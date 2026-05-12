import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score, f1_score,
    precision_score, recall_score, cohen_kappa_score, log_loss, roc_curve, auc
)
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.manifold import TSNE
from dataset import SequenceDataset
from model import SequenceCNN


RUN_ID = os.environ.get("SLEEPNN_RUN_ID")
SEQ_LEN = 32
LSTM_HIDDEN = 128
NUM_HEADS = 4
ATTN_DROPOUT = 0.1
LSTM_LAYERS = 2
LSTM_DROPOUT = 0.3
CLASSIFIER_DROPOUT1 = 0.5
CLASSIFIER_DROPOUT2 = 0.3

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

def plot_scatter_confidence(all_labels, all_probs, run_id):
    """Scatter: confidence modelu pro správné vs. špatné predikce."""
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    max_probs = all_probs.max(axis=1)
    all_preds = all_probs.argmax(axis=1)
    correct = (all_preds == all_labels)
    
    class_names = ['Wake', 'NREM', 'REM']
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for cls_idx, (ax, name) in enumerate(zip(axes, class_names)):
        mask = all_labels == cls_idx
        if mask.sum() == 0:
            continue
        
        correct_conf = max_probs[mask & correct]
        wrong_conf = max_probs[mask & ~correct]
        
        ax.scatter(range(len(correct_conf)), 
                   np.sort(correct_conf)[::-1],
                   alpha=0.3, s=5, color='green', label='Correct')
        ax.scatter(range(len(wrong_conf)),
                   np.sort(wrong_conf)[::-1],
                   alpha=0.3, s=5, color='red', label='Incorrect')
        ax.set_title(f'{name}\n'
                     f'Correct: {len(correct_conf)} '
                     f'| Wrong: {len(wrong_conf)}')
        ax.set_xlabel('Sample rank')
        ax.set_ylabel('Max predicted probability')
        ax.set_ylim(0, 1)
        ax.legend(markerscale=3)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Model Confidence: Correct vs. Incorrect Predictions',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"outputs/scatter_confidence_{run_id}.png", dpi=300)
    print(f"Scatter plot saved to outputs/scatter_confidence_{run_id}.png")

def plot_eval_metrics(all_labels, all_preds, all_probs, run_id):
    """Plot evaluation metrics (Precision, Recall, F1, Accuracy, Cohen Kappa, Log Loss, Explained Variance)."""
    precision_macro = precision_score(all_labels, all_preds, average='macro')
    recall_macro = recall_score(all_labels, all_preds, average='macro')
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    accuracy = accuracy_score(all_labels, all_preds)
    kappa = cohen_kappa_score(all_labels, all_preds)
    
    logloss = log_loss(all_labels, all_probs)
    
    metrics = {
        'Precision': precision_macro,
        'Recall': recall_macro,
        'F1-Score': f1_macro,
        'Accuracy': accuracy,
        'Cohen Kappa': kappa,
        'Log Loss': logloss
    }
    
    fig, ax = plt.subplots(figsize=(10, 6))
    metric_names = list(metrics.keys())
    metric_values = list(metrics.values())
    x = np.arange(len(metric_names))
    
    bars = ax.bar(x, metric_values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'], alpha=0.7)
    
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title('Evaluation Metrics', fontsize=14, fontweight='bold')
    min_val = min(metric_values)
    max_val = max(metric_values)
    pad = (max_val - min_val) * 0.1 if max_val != min_val else 0.1
    ax.set_ylim(min_val - pad, max_val + pad)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xticks(x)
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
    plt.savefig(f"outputs/eval_metrics_{run_id}.png", dpi=300)
    print(f"Evaluation metrics chart saved to outputs/eval_metrics_{run_id}.png")
    
    print(f"\nDetailed Metrics:")
    print(f"  Precision (macro):      {precision_macro:.4f}")
    print(f"  Recall (macro):         {recall_macro:.4f}")
    print(f"  F1-Score (macro):       {f1_macro:.4f}")
    print(f"  Accuracy:               {accuracy:.4f}")
    print(f"  Cohen Kappa:            {kappa:.4f}")
    print(f"  Log Loss:               {logloss:.4f}")

def plot_roc_curves(all_labels, all_probs, run_id):
    """Plot ROC curves (One-vs-Rest) for each sleep stage."""
    fig, ax = plt.subplots(figsize=(8, 6))
    class_names = ['Wake', 'NREM', 'REM']
    colors = ['#1f77b4', '#2ca02c', '#ff7f0e']
    
    for i, (name, color) in enumerate(zip(class_names, colors)):
        y_bin = (np.array(all_labels) == i).astype(int)
        fpr, tpr, _ = roc_curve(y_bin, np.array(all_probs)[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f'{name} (AUC = {roc_auc:.3f})')
    
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves (One-vs-Rest)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"outputs/roc_curves_{run_id}.png", dpi=300)
    print(f"ROC curves saved to outputs/roc_curves_{run_id}.png")

def plot_per_subject_f1(all_labels, all_preds, all_subject_ids, run_id):
    """Per-subject F1 score analýza – pro identifikaci problémových subjektů."""
    subjects = sorted(set(all_subject_ids))
    f1_scores = []
    
    for subj in subjects:
        mask = np.array(all_subject_ids) == subj
        if mask.sum() < 10:
            continue
        f1 = f1_score(
            np.array(all_labels)[mask],
            np.array(all_preds)[mask],
            average='macro'
        )
        f1_scores.append((subj, f1))
    
    f1_scores.sort(key=lambda x: x[1])
    subjects_sorted = [str(x[0]) for x in f1_scores]
    scores_sorted = [x[1] for x in f1_scores]
    
    plt.figure(figsize=(12, 5))
    colors = ['red' if s < 0.75 else 'steelblue' for s in scores_sorted]
    plt.bar(range(len(subjects_sorted)), scores_sorted, color=colors, alpha=0.7)
    plt.axhline(y=np.mean(scores_sorted), color='black', linestyle='--', 
                label=f'Mean F1: {np.mean(scores_sorted):.3f}')
    plt.xticks(range(len(subjects_sorted)), subjects_sorted, rotation=45)
    plt.xlabel('Subject ID', fontsize=12)
    plt.ylabel('Macro F1', fontsize=12)
    plt.title('Per-Subject Classification Performance', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f"outputs/per_subject_f1_{run_id}.png", dpi=300)
    print(f"Per-subject F1 chart saved to outputs/per_subject_f1_{run_id}.png")

def extract_latent_features(model, test_loader, device):
    model.eval()
    all_features = []
    all_labels = []
    all_subject_ids = []

    features_hook = []
    
    def hook_fn(module, input, output):
        features_hook.append(input[0].detach().cpu())
    
    hook = model.classifier[-1].register_forward_hook(hook_fn)
    
    try:
        with torch.no_grad():
            for signals, labels, subject_ids in test_loader:
                signals = signals.to(device)
                labels = labels.to(device)
                features_hook.clear()
                _ = model(signals)
                if features_hook:
                    all_features.append(features_hook[0])
                
                all_labels.extend(labels.cpu().numpy())
                all_subject_ids.extend(subject_ids)
    
    finally:
        hook.remove()

    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.array(all_labels)
    
    print(f"Extrahováno {all_features.shape[0]} latentních features")
    print(f"Dimenze features: {all_features.shape[1]}")
    
    return all_features, all_labels, all_subject_ids

def plot_latent_features_tsne(features, labels, subject_ids, run_id):    
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    features_2d = tsne.fit_transform(features)
    class_colors = {0: '#1f77b4', 1: '#2ca02c', 2: '#ff7f0e'}  
    class_names = {0: 'Wake', 1: 'NREM', 2: 'REM'}
    
    plt.figure(figsize=(10, 8))
    for class_id in sorted(set(labels)):
        mask = labels == class_id
        plt.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                   c=class_colors[class_id], label=class_names[class_id], 
                   alpha=0.6, s=30, edgecolors='black', linewidth=0.5)
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.title('Latent Features - t-SNE (by Sleep Stage)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"outputs/tsne_features_by_class_{run_id}.png", dpi=300)
    unique_subjects = sorted(set(subject_ids))
    num_subjects = len(unique_subjects)
    colors = plt.cm.tab20(np.linspace(0, 1, num_subjects))
    subject_to_color = {subj: colors[i] for i, subj in enumerate(unique_subjects)}
    
    plt.figure(figsize=(12, 8))
    for subj in unique_subjects:
        mask = np.array(subject_ids) == subj
        plt.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                   c=[subject_to_color[subj]], label=f'Subject {subj}', 
                   alpha=0.6, s=30, edgecolors='black', linewidth=0.5)
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.title('Latent Features - t-SNE (by Subject)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.tight_layout()
    plt.savefig(f"outputs/tsne_features_by_subject_{run_id}.png", dpi=300, bbox_inches='tight')

def evaluate_model(model, test_loader, device):
    model.eval() 
    all_preds = []
    all_labels = []
    all_probs = []
    all_subject_ids = []

    with torch.no_grad():
        for signals, labels, subject_ids in test_loader:
            signals = signals.to(device)
            labels = labels.to(device)
            
            outputs = model(signals)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_subject_ids.extend(subject_ids)

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
    plot_scatter_confidence(all_labels, all_probs, RUN_ID)
    plot_roc_curves(all_labels, all_probs, RUN_ID)
    plot_per_subject_f1(all_labels, all_preds, all_subject_ids, RUN_ID)
    plot_eval_metrics(all_labels, all_preds, all_probs, RUN_ID)
    
    print("\nExtracting and visualizing latent features...")
    latent_features, feat_labels, feat_subject_ids = extract_latent_features(
        model, test_loader, device)
    plot_latent_features_tsne(
        latent_features, feat_labels, feat_subject_ids, RUN_ID)
    
    return accuracy_score(all_labels, all_preds), cm

def main():
    print("\n" + "="*50)
    print("EVALUATION")
    print("="*50 + "\n")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    MODEL_PATH = f"outputs/sleep_lstm_weights_{RUN_ID}.pth"
    TEST_DATA_DIR = os.environ.get("SLEEPNN_DATA_DIR", "/mnt/scratch/temporary/asanieli_data/processed_pt")
    BATCH_SIZE = 16
    NUM_WORKERS = 4

    print(f"Loading model from {MODEL_PATH}...")
    model = SequenceCNN(
        channels=4,
        num_classes=3,
        lstm_hidden=LSTM_HIDDEN,
        attn_dropout=ATTN_DROPOUT,
        num_heads=NUM_HEADS,
        lstm_layers=LSTM_LAYERS,
        lstm_dropout=LSTM_DROPOUT,
        classifier_dropout1=CLASSIFIER_DROPOUT1,
        classifier_dropout2=CLASSIFIER_DROPOUT2
    ).to(device)
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    print("Model loaded.\n")

    print("Loading test data...")
    test_dataset = SequenceDataset(
        TEST_DATA_DIR,
        split='test',
        split_ratio=0.8,
        sequence_length=SEQ_LEN,
        stride=SEQ_LEN
    )
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    print(f"Test sequences: {len(test_dataset)}\n")

    print("Evaluating...")
    accuracy, cm = evaluate_model(model, test_loader, device)
    
    print(f"{'='*50}")
    print("Done!\n")

if __name__ == "__main__":
    main()