import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

os.environ['MPLCONFIGDIR'] = os.path.join(os.getcwd(), 'outputs')

figures_dir = os.path.join('outputs', 'figures')
os.makedirs(figures_dir, exist_ok=True)

# 1. Class Distribution Plot
train_df = pd.read_csv('data/processed/train.csv')
val_df = pd.read_csv('data/processed/val.csv')
test_df = pd.read_csv('data/processed/test.csv')

classes = ['Positive', 'Neutral', 'Negative']
colors = ['#10B981', '#3B82F6', '#EF4444']

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
splits = [('Train Set (N=6,961)', train_df), ('Validation Set (N=740)', val_df), ('Test Set (N=741)', test_df)]

for ax, (title, df) in zip(axes, splits):
    counts = [df['sentiment'].value_counts().get(c, 0) for c in classes]
    bars = ax.bar(classes, counts, color=colors, edgecolor='#1E293B', linewidth=1.2, alpha=0.9)
    ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
    ax.set_ylabel('Count', fontsize=11)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    for bar in bars:
        h = bar.get_height()
        p = (h / len(df)) * 100
        ax.annotate(f'{h}\n({p:.1f}%)',
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords='offset points',
                    ha='center', va='bottom', fontsize=9, fontweight='semibold')

plt.tight_layout()
dist_path = os.path.join(figures_dir, 'class_distribution.png')
plt.savefig(dist_path, dpi=300)
plt.close()
print(f'Saved class distribution plot to {dist_path}')

# 2. Sample Memes Grid
sample_rows = train_df.sample(6, random_state=42)
fig, axes = plt.subplots(2, 3, figsize=(14, 9))
axes = axes.flatten()

for ax, (_, row) in zip(axes, sample_rows.iterrows()):
    img_path = row['image_path']
    try:
        img = Image.open(img_path).convert('RGB')
        ax.imshow(img)
    except Exception as e:
        ax.text(0.5, 0.5, 'Image load error', ha='center', va='center')
    
    text_preview = row['text'][:55] + '...' if len(row['text']) > 55 else row['text']
    sent = row['sentiment']
    color = '#10B981' if sent == 'Positive' else ('#3B82F6' if sent == 'Neutral' else '#EF4444')
    ax.set_title(f'Sentiment: {sent}\n"{text_preview}"', fontsize=9, color=color, fontweight='bold')
    ax.axis('off')

plt.suptitle('IndicSent-M Sample Multimodal Memes (Memotion 3)', fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout()
grid_path = os.path.join(figures_dir, 'sample_memes_grid.png')
plt.savefig(grid_path, dpi=300)
plt.close()
print(f'Saved sample memes grid to {grid_path}')

