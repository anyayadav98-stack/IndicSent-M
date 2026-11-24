"""
evaluate.py - Test Set Evaluation Pipeline for IndicSent-M
Evaluates trained models on data/processed/test.csv.
Computes:
- Overall Accuracy & Macro F1
- Per-class Precision, Recall, and F1-score
- Confusion Matrix (raw counts and normalized percentages)
Saves:
- outputs/metrics_<model>.json
- outputs/figures/confusion_matrix_<model>.png
"""

import os
os.environ['MPLCONFIGDIR'] = os.path.join(os.getcwd(), 'outputs')
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from dataset import MemeDataset
from models import TextOnlyModel, ImageOnlyModel, MultimodalConcatModel, MultimodalAttentionModel

os.environ['MPLCONFIGDIR'] = os.path.join(os.getcwd(), 'outputs')

CLASS_NAMES = ["Negative", "Neutral", "Positive"]

def plot_confusion_matrix(cm, model_name, out_path):
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots(figsize=(6, 5))
    
    # Custom colored grid
    im = ax.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
           title=f'Confusion Matrix - {model_name.upper()}',
           ylabel='True Label',
           xlabel='Predicted Label')

    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")

    thresh = cm_norm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]}\n({cm_norm[i, j]*100:.1f}%)",
                    ha="center", va="center",
                    color="white" if cm_norm[i, j] > thresh else "black",
                    fontsize=10, fontweight="bold")
    
    fig.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix plot to {out_path}")

def evaluate_model(model_type, device, test_csv, checkpoints_dir, figures_dir, outputs_dir, batch_size=16):
    ckpt_path = os.path.join(checkpoints_dir, f"best_{model_type}.pt")
    if not os.path.exists(ckpt_path):
        print(f"[WARNING] Checkpoint {ckpt_path} not found. Skipping evaluation for {model_type}.")
        return None

    print("\n" + "=" * 60)
    print(f"Evaluating {model_type.upper()} on Test Set ({test_csv})...")
    print("=" * 60)

    # Tokenizer
    if model_type in ['text', 'concat', 'attention']:
        tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    else:
        tokenizer = None

    test_ds = MemeDataset(test_csv, tokenizer=tokenizer, is_train=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # Model definition
    if model_type == 'text':
        model = TextOnlyModel(num_classes=3)
    elif model_type == 'image':
        model = ImageOnlyModel(num_classes=3)
    elif model_type == 'concat':
        model = MultimodalConcatModel(num_classes=3)
    elif model_type == 'attention':
        model = MultimodalAttentionModel(num_classes=3)

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()

    all_preds, all_targets = [], []
    all_probs = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Evaluating {model_type}"):
            labels = batch['label'].to(device)
            if model_type == 'text':
                logits = model(batch['input_ids'].to(device), batch['attention_mask'].to(device))
            elif model_type == 'image':
                logits = model(batch['image'].to(device))
            elif model_type in ['concat', 'attention']:
                logits = model(batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['image'].to(device))

            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            all_probs.extend(probs.tolist())
            all_preds.extend(preds.tolist())
            all_targets.extend(labels.cpu().numpy().tolist())

    # Calculate metrics
    acc = accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    weighted_f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
    
    p_per_class = precision_score(all_targets, all_preds, average=None, zero_division=0)
    r_per_class = recall_score(all_targets, all_preds, average=None, zero_division=0)
    f1_per_class = f1_score(all_targets, all_preds, average=None, zero_division=0)

    cm = confusion_matrix(all_targets, all_preds)

    per_class_metrics = {}
    for idx, cname in enumerate(CLASS_NAMES):
        per_class_metrics[cname] = {
            "precision": float(round(p_per_class[idx], 4)),
            "recall": float(round(r_per_class[idx], 4)),
            "f1_score": float(round(f1_per_class[idx], 4)),
            "support": int(np.sum(np.array(all_targets) == idx))
        }

    metrics = {
        "model": model_type,
        "accuracy": float(round(acc, 4)),
        "macro_f1": float(round(macro_f1, 4)),
        "weighted_f1": float(round(weighted_f1, 4)),
        "per_class": per_class_metrics,
        "confusion_matrix": cm.tolist()
    }

    # Save metrics JSON
    metrics_path = os.path.join(outputs_dir, f"metrics_{model_type}.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved test metrics to {metrics_path}")

    # Plot Confusion Matrix
    cm_plot_path = os.path.join(figures_dir, f"confusion_matrix_{model_type}.png")
    plot_confusion_matrix(cm, model_type, cm_plot_path)

    print(f"Test Accuracy: {acc*100:.2f}% | Test Macro F1: {macro_f1*100:.2f}%")
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Evaluate IndicSent-M models on test set.")
    parser.add_argument('--model', type=str, default='all', choices=['text', 'image', 'concat', 'attention', 'all'],
                        help="Model to evaluate or 'all'.")
    parser.add_argument('--batch_size', type=int, default=16)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    processed_dir = os.path.join('data', 'processed')
    test_csv = os.path.join(processed_dir, 'test.csv')
    checkpoints_dir = os.path.join('outputs', 'checkpoints')
    figures_dir = os.path.join('outputs', 'figures')
    outputs_dir = 'outputs'

    models_to_eval = ['text', 'image', 'concat', 'attention'] if args.model == 'all' else [args.model]

    for m in models_to_eval:
        evaluate_model(m, device, test_csv, checkpoints_dir, figures_dir, outputs_dir, args.batch_size)

if __name__ == "__main__":
    main()

