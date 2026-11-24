"""
train.py - PyTorch Training Pipeline for IndicSent-M
Supports: text, image, concat, attention models.
Features:
- RTX 2050 4GB memory optimizations (mixed precision fp16, batch size 8, gradient accumulation)
- Class-weighted Cross-Entropy loss for handling class imbalance
- AdamW optimizer & linear/cosine scheduler
- Checkpointing best model based on validation macro F1 score
- Generates loss and F1 training curves in outputs/figures/
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
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoTokenizer

from dataset import MemeDataset
from models import TextOnlyModel, ImageOnlyModel, MultimodalConcatModel, MultimodalAttentionModel

os.environ['MPLCONFIGDIR'] = os.path.join(os.getcwd(), 'outputs')

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def compute_weights(train_csv):
    df = pd.read_csv(train_csv)
    labels = df['label'].values
    weights = compute_class_weight(class_weight='balanced', classes=np.array([0, 1, 2]), y=labels)
    return torch.tensor(weights, dtype=torch.float)

def plot_curves(history, model_name, out_path):
    epochs = range(1, len(history['train_loss']) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    
    # Loss plot
    axes[0].plot(epochs, history['train_loss'], 'o-', label='Train Loss', color='#3B82F6', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 's--', label='Val Loss', color='#EF4444', linewidth=2)
    axes[0].set_title(f'{model_name.upper()} - Loss Curve', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].grid(True, linestyle='--', alpha=0.5)
    axes[0].legend()

    # Macro F1 plot
    axes[1].plot(epochs, history['val_f1'], 'd-', label='Val Macro F1', color='#10B981', linewidth=2)
    axes[1].set_title(f'{model_name.upper()} - Validation Macro F1', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Macro F1')
    axes[1].grid(True, linestyle='--', alpha=0.5)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved training curves to {out_path}")

def main():
    parser = argparse.ArgumentParser(description="Train IndicSent-M sentiment models.")
    parser.add_argument('--model', type=str, required=True, choices=['text', 'image', 'concat', 'attention'],
                        help="Model architecture to train.")
    parser.add_argument('--epochs', type=int, default=3, help="Number of training epochs.")
    parser.add_argument('--batch_size', type=int, default=8, help="Batch size (8 recommended for 4GB RTX 2050).")
    parser.add_argument('--lr', type=float, default=2e-5, help="Learning rate.")
    parser.add_argument('--subset', type=int, default=None, help="Sample N rows for quick test.")
    parser.add_argument('--max_length', type=int, default=128, help="Max sequence length for XLM-R.")
    parser.add_argument('--accum_steps', type=int, default=2, help="Gradient accumulation steps.")
    args = parser.parse_args()

    set_seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 60)
    print(f"IndicSent-M Training: Model [{args.model.upper()}] on [{device}]")
    if torch.cuda.is_available():
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
        print(f"Allocated VRAM: {torch.cuda.memory_allocated() / 1e6:.1f} MB")
    print("=" * 60)

    # Directories
    processed_dir = os.path.join('data', 'processed')
    train_csv = os.path.join(processed_dir, 'train.csv')
    val_csv = os.path.join(processed_dir, 'val.csv')
    
    checkpoints_dir = os.path.join('outputs', 'checkpoints')
    figures_dir = os.path.join('outputs', 'figures')
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    # Tokenizer
    if args.model in ['text', 'concat', 'attention']:
        print("Loading multilingual XLM-RoBERTa tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    else:
        tokenizer = None

    # Datasets & Loaders
    train_ds = MemeDataset(train_csv, tokenizer=tokenizer, max_length=args.max_length, is_train=True, subset=args.subset)
    val_ds = MemeDataset(val_csv, tokenizer=tokenizer, max_length=args.max_length, is_train=False, subset=args.subset // 4 if args.subset else None)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    print(f"Training Samples: {len(train_ds)} | Validation Samples: {len(val_ds)}")

    # Model instantiation
    print(f"Initializing {args.model.upper()} architecture...")
    if args.model == 'text':
        model = TextOnlyModel(num_classes=3)
    elif args.model == 'image':
        model = ImageOnlyModel(num_classes=3)
    elif args.model == 'concat':
        model = MultimodalConcatModel(num_classes=3)
    elif args.model == 'attention':
        model = MultimodalAttentionModel(num_classes=3)

    model.to(device)

    # Class-weighted loss
    class_weights = compute_weights(train_csv).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Differential learning rate: lower for backbone, higher for classification head
    head_params = []
    backbone_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if "classifier" in name or "proj" in name or "cross_attention" in name:
                head_params.append(param)
            else:
                backbone_params.append(param)

    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': args.lr},
        {'params': head_params, 'lr': args.lr * 5}
    ], weight_decay=0.01)

    scaler = GradScaler(enabled=(device.type == 'cuda'))

    best_val_f1 = 0.0
    history = {'train_loss': [], 'val_loss': [], 'val_f1': [], 'val_acc': []}
    best_checkpoint_path = os.path.join(checkpoints_dir, f"best_{args.model}.pt")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()

        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]")
        for step, batch in enumerate(loop):
            labels = batch['label'].to(device)

            with autocast(enabled=(device.type == 'cuda')):
                if args.model == 'text':
                    logits = model(batch['input_ids'].to(device), batch['attention_mask'].to(device))
                elif args.model == 'image':
                    logits = model(batch['image'].to(device))
                elif args.model in ['concat', 'attention']:
                    logits = model(batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['image'].to(device))

                loss = criterion(logits, labels)
                loss = loss / args.accum_steps

            scaler.scale(loss).backward()

            if (step + 1) % args.accum_steps == 0 or (step + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_loss += loss.item() * args.accum_steps
            loop.set_postfix({'loss': f"{loss.item() * args.accum_steps:.4f}"})

        avg_train_loss = total_loss / len(train_loader)

        # Validation step
        model.eval()
        val_loss = 0.0
        val_preds, val_targets = [], []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} [Val]"):
                labels = batch['label'].to(device)

                with autocast(enabled=(device.type == 'cuda')):
                    if args.model == 'text':
                        logits = model(batch['input_ids'].to(device), batch['attention_mask'].to(device))
                    elif args.model == 'image':
                        logits = model(batch['image'].to(device))
                    elif args.model in ['concat', 'attention']:
                        logits = model(batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['image'].to(device))

                    v_loss = criterion(logits, labels)

                val_loss += v_loss.item()
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_targets.extend(labels.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        val_acc = accuracy_score(val_targets, val_preds)
        val_macro_f1 = f1_score(val_targets, val_preds, average='macro', zero_division=0)

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_f1'].append(val_macro_f1)
        history['val_acc'].append(val_acc)

        print(f"--> Epoch {epoch}: Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f} | Val Macro F1: {val_macro_f1:.4f}")

        # Checkpoint if best macro F1
        if val_macro_f1 > best_val_f1:
            best_val_f1 = val_macro_f1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_f1': val_macro_f1,
                'val_acc': val_acc,
                'args': vars(args)
            }, best_checkpoint_path)
            print(f"*** New best checkpoint saved with Val Macro F1: {val_macro_f1:.4f} -> {best_checkpoint_path}")

    # Save curves and history
    curve_path = os.path.join(figures_dir, f"training_curves_{args.model}.png")
    plot_curves(history, args.model, curve_path)

    history_path = os.path.join('outputs', f"history_{args.model}.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    print("\nTraining completed successfully!")
    print(f"Best Val Macro F1: {best_val_f1:.4f}")

if __name__ == "__main__":
    main()

