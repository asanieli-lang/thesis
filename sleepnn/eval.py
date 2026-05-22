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
    """Compare predicted vs ground truth sleep stages as time series."""
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
    plt.close()

def plot_f1_per_class(all_labels, all_preds, run_id):
    """Show per-stage F1 scores to identify imbalanced class performance."""
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
    plt.close()

def plot_roc_curves(all_labels, all_probs, run_id):
    """Generate one-vs-rest ROC curves for each sleep stage."""
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
    plt.close()

def plot_per_subject_f1(all_labels, all_preds, all_subject_ids, run_id):
    """Evaluate generalization by showing per-subject macro-F1 scores."""
    subjects = np.unique(all_subject_ids)
    f1_scores = []
    valid_subjects = []

    for subj in subjects:
        idx = (all_subject_ids == subj)
        if np.sum(idx) > 0:
            score = f1_score(all_labels[idx], all_preds[idx], average='macro')
            f1_scores.append(score)
            valid_subjects.append(str(subj))

    plt.figure(figsize=(14, 6))
    colors = ['#d62728' if s < 0.75 else '#1f77b4' for s in f1_scores]
    bars = plt.bar(valid_subjects, f1_scores, color=colors, edgecolor='black', alpha=0.8)
    plt.axhline(y=0.75, color='red', linestyle='--', alpha=0.5, label='Threshold 0.75')
    
    plt.ylim(0, 1.0)
    plt.xlabel('Subject ID (Potkan)', fontsize=12)
    plt.ylabel('Macro-F1 Score', fontsize=12)
    plt.title('Model Performance Across Individual Unseen Subjects', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"outputs/per_subject_f1_{run_id}.png", dpi=300)
    plt.close()

def extract_latent_features(model, test_loader, device):
    """Extract features from classification layer using forward hook."""
    model.eval()
    all_features = []
    all_labels = []
    all_subject_ids = []

    features_hook = []
    
    # Hook to capture activations before final classification layer
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
    return all_features, all_labels, all_subject_ids

def plot_umap_3d(features, labels, subject_ids, run_id):
    """Visualize learned representations using UMAP 2D and 3D dimensionality reduction."""
    try:
        import umap
    except ImportError:
        return
    
    # Subsample to 20k samples for UMAP efficiency
    n = len(features)
    if n > 20000:
        idx = np.random.choice(n, 20000, replace=False)
        features = features[idx]
        labels = labels[idx]
        subject_ids = [subject_ids[i] for i in idx]
    reducer = umap.UMAP(n_components=3, n_neighbors=30, min_dist=0.1,
                        random_state=42, verbose=False)
    emb = reducer.fit_transform(features)
    
    class_colors = {0: '#1f77b4', 1: '#2ca02c', 2: '#ff7f0e'}
    class_names = {0: 'Wake', 1: 'NREM', 2: 'REM'}
    
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    for class_id in sorted(set(labels)):
        mask = labels == class_id
        ax.scatter(emb[mask, 0], emb[mask, 1], emb[mask, 2],
                   c=class_colors[class_id], label=class_names[class_id],
                   alpha=0.4, s=8, edgecolors='none')
    
    ax.set_xlabel('UMAP 1', labelpad=8)
    ax.set_ylabel('UMAP 2', labelpad=8)
    ax.set_zlabel('UMAP 3', labelpad=8)
    ax.set_title('Latent Space – 3D UMAP by Sleep Stage',
                 fontsize=13, fontweight='bold')
    ax.legend(markerscale=4, fontsize=11)
    fig.savefig(f"outputs/umap_3d_{run_id}.png", dpi=200, bbox_inches='tight')
    plt.close(fig)
    
    reducer2d = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.1,
                          random_state=42, verbose=False)
    emb2d = reducer2d.fit_transform(features)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    for class_id in sorted(set(labels)):
        mask = labels == class_id
        ax.scatter(emb2d[mask, 0], emb2d[mask, 1],
                   c=class_colors[class_id], label=class_names[class_id],
                   alpha=0.4, s=8, edgecolors='none')
    ax.set_xlabel('UMAP Dimension 1', fontsize=12)
    ax.set_ylabel('UMAP Dimension 2', fontsize=12)
    ax.set_title('Latent Space – 2D UMAP by Sleep Stage',
                 fontsize=13, fontweight='bold')
    ax.legend(markerscale=4, fontsize=11)
    ax.grid(True, alpha=0.2)
    fig.savefig(f"outputs/umap_2d_{run_id}.png", dpi=200, bbox_inches='tight')
    plt.close(fig)

def evaluate_model(model, test_loader, device):
    """Perform comprehensive evaluation with metrics and visualization plots."""
    model.eval() 
    all_preds = []
    all_labels = []
    all_probs = []
    all_subject_ids = []

    # Collect predictions and probabilities over test set
    with torch.no_grad():
        for signals, labels, subject_ids in test_loader:
            signals = signals.to(device)
            
            # Forward pass to get class probabilities
            outputs = model(signals)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_subject_ids.extend(subject_ids.cpu().numpy() if isinstance(subject_ids, torch.Tensor) else subject_ids)

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_subject_ids = np.array(all_subject_ids)

    # Display standard classification metrics
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(classification_report(all_labels, all_preds, target_names=['Wake', 'NREM', 'REM']))
    print(f"Overall Accuracy: {accuracy_score(all_labels, all_preds):.4f}")
    print(f"F1 Score (Macro): {f1_score(all_labels, all_preds, average='macro'):.4f}\n")

    # Generate and visualize confusion matrix
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2])  # 0=Wake, 1=NREM, 2=REM
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=True, fmt='.1f', cmap='Blues', cbar_kws={'label': 'Percentage (%)'}, 
                xticklabels=['Wake', 'NREM', 'REM'], 
                yticklabels=['Wake', 'NREM', 'REM'])
    plt.title('Confusion Matrix (%)')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(f"outputs/final_eval_cm_{RUN_ID}.png", dpi=300)
    plt.close()
    
    # Generate evaluation plots
    plot_hypnogram(all_labels, all_preds, RUN_ID)
    plot_f1_per_class(all_labels, all_preds, RUN_ID)
    plot_roc_curves(all_labels, all_probs, RUN_ID)
    plot_per_subject_f1(all_labels, all_preds, all_subject_ids, RUN_ID)
    
    # Extract and visualize learned latent representations
    latent_features, feat_labels, feat_subject_ids = extract_latent_features(
        model, test_loader, device)
    plot_umap_3d(latent_features, feat_labels, feat_subject_ids, RUN_ID)
    
    return accuracy_score(all_labels, all_preds), cm

def main():
    """Load checkpoint, evaluate on test set, and generate comprehensive metrics."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model and configuration
    MODEL_PATH = f"outputs/best_model_checkpoint_{RUN_ID}.pth"
    TEST_DATA_DIR = os.environ.get("SLEEPNN_DATA_DIR", "/mnt/scratch/temporary/asanieli_data/processed_pt")
    BATCH_SIZE = 16  # Must match train.py batch size
    NUM_WORKERS = 4
    TEST_STRIDE = SEQ_LEN  # Non-overlapping evaluation windows
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
    test_dataset = SequenceDataset(
        TEST_DATA_DIR,
        split='test',
        split_ratio=0.8,
        sequence_length=SEQ_LEN,
        stride=TEST_STRIDE  # Non-overlapping to avoid double-counting samples
    )
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    accuracy, cm = evaluate_model(model, test_loader, device)

if __name__ == "__main__":
    main()