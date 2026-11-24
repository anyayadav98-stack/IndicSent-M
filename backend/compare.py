"""
compare.py - Model Comparison Pipeline for IndicSent-M
Aggregates metrics from all evaluated models into outputs/comparison.json
and generates outputs/figures/model_comparison.png (grouped bar chart).
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.environ['MPLCONFIGDIR'] = os.path.join(os.getcwd(), 'outputs')

MODELS = ['text', 'image', 'concat', 'attention']
DISPLAY_NAMES = {
    'text': 'Text-Only (XLM-R)',
    'image': 'Image-Only (ResNet50)',
    'concat': 'Multimodal (Concat)',
    'attention': 'Multimodal (Attention)'
}

CLASSES = ['Negative', 'Neutral', 'Positive']

def main():
    outputs_dir = 'outputs'
    figures_dir = os.path.join(outputs_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)

    comparison_data = {}
    
    for m in MODELS:
        mpath = os.path.join(outputs_dir, f"metrics_{m}.json")
        if os.path.exists(mpath):
            with open(mpath, 'r') as f:
                comparison_data[m] = json.load(f)
        else:
            print(f"[INFO] metrics_{m}.json not found.")

    if not comparison_data:
        print("[WARNING] No model metrics found to compare. Run backend/evaluate.py first.")
        return

    # Save outputs/comparison.json
    comp_json_path = os.path.join(outputs_dir, 'comparison.json')
    with open(comp_json_path, 'w') as f:
        json.dump(comparison_data, f, indent=2)
    print(f"Aggregated comparison saved to {comp_json_path}")

    # Prepare plot data
    present_models = [m for m in MODELS if m in comparison_data]
    labels = [DISPLAY_NAMES[m] for m in present_models]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 1. Grouped Bar Chart: F1 per Class
    x = np.arange(len(CLASSES))
    width = 0.18
    palette = ['#6366F1', '#EC4899', '#3B82F6', '#10B981']

    for i, m in enumerate(present_models):
        f1_vals = [comparison_data[m]['per_class'][c]['f1_score'] * 100 for c in CLASSES]
        axes[0].bar(x + (i - len(present_models)/2 + 0.5) * width, f1_vals, width, label=DISPLAY_NAMES[m], color=palette[i % len(palette)], edgecolor='#1E293B', alpha=0.9)

    axes[0].set_title('Per-Class F1-Score Comparison (%)', fontsize=13, fontweight='bold', pad=12)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(CLASSES, fontsize=11, fontweight='semibold')
    axes[0].set_ylabel('F1 Score (%)', fontsize=11)
    axes[0].set_ylim(0, 100)
    axes[0].grid(axis='y', linestyle='--', alpha=0.5)
    axes[0].legend(loc='upper right', framealpha=0.95)

    # 2. Overall Accuracy and Macro F1 Comparison
    x2 = np.arange(len(present_models))
    w2 = 0.35
    accs = [comparison_data[m]['accuracy'] * 100 for m in present_models]
    macro_f1s = [comparison_data[m]['macro_f1'] * 100 for m in present_models]

    rects1 = axes[1].bar(x2 - w2/2, accs, w2, label='Accuracy', color='#0284C7', edgecolor='#1E293B', alpha=0.9)
    rects2 = axes[1].bar(x2 + w2/2, macro_f1s, w2, label='Macro F1', color='#0D9488', edgecolor='#1E293B', alpha=0.9)

    axes[1].set_title('Overall Test Performance: Accuracy vs Macro F1 (%)', fontsize=13, fontweight='bold', pad=12)
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels([m.upper() for m in present_models], fontsize=11, fontweight='semibold')
    axes[1].set_ylabel('Percentage (%)', fontsize=11)
    axes[1].set_ylim(0, 100)
    axes[1].grid(axis='y', linestyle='--', alpha=0.5)
    axes[1].legend(loc='lower right', framealpha=0.95)

    for rect in rects1:
        h = rect.get_height()
        axes[1].annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width() / 2, h),
                         xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')
    for rect in rects2:
        h = rect.get_height()
        axes[1].annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width() / 2, h),
                         xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    chart_path = os.path.join(figures_dir, 'model_comparison.png')
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"Saved model comparison grouped bar chart to {chart_path}")

    # Print summary table
    print("\n" + "=" * 70)
    print(f"{'Model':<26} | {'Accuracy':<10} | {'Macro F1':<10} | {'Weighted F1':<12}")
    print("-" * 70)
    for m in present_models:
        acc = comparison_data[m]['accuracy'] * 100
        mf1 = comparison_data[m]['macro_f1'] * 100
        wf1 = comparison_data[m]['weighted_f1'] * 100
        print(f"{DISPLAY_NAMES[m]:<26} | {acc:>8.2f}% | {mf1:>8.2f}% | {wf1:>10.2f}%")
    print("=" * 70)

if __name__ == "__main__":
    main()

