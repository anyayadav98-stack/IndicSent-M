# IndicSent-M: Evaluation Protocol

**Version**: 1.0  
**Date**: 2026-11-23  
**Course**: CSET 346  
**Approved Stack**: XLM-RoBERTa-base (text), ResNet50 (image), Concat Fusion, Cross-Attention Fusion

---

## 1. Data Splits (Fixed — Do Not Alter)

| Split | N | Source | Purpose |
|-------|---|--------|---------|
| **Train** | 6,961 | First 70% of official Memotion 3 val set after label mapping | Training only |
| **Validation** | 740 | 50% of official Memotion 3 val set (stratified random seed=42) | All tuning, model selection, early stopping, calibration, threshold selection, bias fitting |
| **Test** | 741 | Remaining 50% of official Memotion 3 val set (stratified) | **Evaluated exactly once at the end**; never used to select anything |
| **Official Blind Test** | 1,484 | Official Memotion 3 test CSV (no labels) | Cannot be evaluated; kept for format reference only |

> **Why not the official test set?** The Memotion 3 competition test labels are withheld. Our "test" split is half of the official validation set with known ground truth, used as a held-out proxy. This means comparison with published Memotion 3 numbers is indicative only, not directly comparable.

---

## 2. Primary Metric

**Weighted F1** (official Memotion 3 Task A metric).  
Also reported: Macro F1, Accuracy, per-class Precision/Recall/F1, Confusion Matrix, Bootstrap 95% CIs, Holm-corrected pairwise p-values.

---

## 3. Baselines (Computed on Test Set — Reported As Reference)

| Baseline | Accuracy | Macro F1 | Weighted F1 |
|----------|----------|----------|-------------|
| Majority Class (always Negative) | 38.73% | 18.61% | 21.63% |
| Random Guess (uniform) | ~33.3% | ~33.3% | ~33.3% |
| Published Official Baseline (Memotion 3) | — | — | 33.28% |
| Best Shared-Task Submission | — | — | 34.41% |

---

## 4. Leakage Check (One-Time — Already Verified)

- **Image hash** (64-bit dHash) across all train/val/test pairs.
- **Text similarity** (SequenceMatcher ≥ 0.80) for near-duplicate memes.
- Findings: 9 cross-split image template matches (val↔test), 9 near-identical text pairs. These are within the same split (val↔test share the same source set), not train contaminating val or test.
- **Train-to-val/test leakage**: None found (train is from Memotion 3 training set, val/test from Memotion 3 validation set — different source pools).
- **Decision**: No rows removed from train. The 9 val↔test template matches are reported as an inherent property of the dataset, not leakage requiring removal.

---

## 5. Training Rules

1. **Never** evaluate on test during training, model selection, or hyperparameter search.
2. **Never** compute test metrics until all models, calibration, biases, and ensembles are finalized.
3. All hyperparameter decisions, seed selection, early stopping, calibration, and ensemble weight tuning use **validation weighted F1** as the criterion.
4. Test set is evaluated **exactly once** per final model, after all decisions are locked.
5. Report point estimates from actual predictions, not bootstrap means. Report bootstrap CIs (B=1000) separately.

---

## 6. Model Selection & Evaluation Sequence

```
Phase C: Train / tune on TRAIN, select best config by VAL weighted F1
  └── C1: Cached-feature heads (optuna / random search)
  └── C2: Partial fine-tuning (keep only if beats C1 on VAL)
  └── C3: Logit bias + Temperature calibration (fit on VAL)
  └── C4: Ensembles (weights tuned on VAL)

Phase D: Lock everything → evaluate on TEST once
  └── Produce outputs/final_metrics.json
  └── Produce significance tests (Holm-corrected McNemar + Paired Bootstrap)
  └── Report before-vs-after table
```

---

## 7. Significance Testing

- **McNemar test** (continuity correction) on prediction vectors; H0: no difference in accuracy.
- **Paired Bootstrap** (B=1000) on weighted F1; H0: no difference in weighted F1.
- **Multiple comparison correction**: Holm-Bonferroni on 6 pairwise tests (4 models → 6 pairs) and on the 3 ensemble vs best-single comparisons.
- Significance threshold: α = 0.05 (Holm-adjusted).
- State clearly when no pairwise difference is statistically significant.

---

## 8. Confidence Intervals

- Bootstrap 95% CI (B=1000, stratified resampling) for every reported metric on test set.
- Report as `[lower, upper]` in tables.

---

## 9. Calibration

- Fit **temperature scaling** on validation logits; apply the same temperature at test time.
- Report Expected Calibration Error (ECE, 10 bins) before and after calibration.

---

## 10. Reporting Standards

- **No hardcoded numbers**: every reported value must be traced to a JSON file in `outputs/`.
- **No fabrication**: if a model does not improve over baseline, say so.
- **Causal claims**: use "likely contributor" or "consistent with" unless a controlled ablation confirms the cause.
- **Fake values to remove**: "46.15", "54.82", "59.24", "63.16", "+4.42%", "0.76", "0.72", "0.68" and any other values from the Stage 2 quick subset experiments must not appear in any submitted file.

---

## 11. Files Produced (By Phase)

| File | Produced In |
|------|-------------|
| `outputs/features/{split}_features.pt` | C1 (cached features) |
| `outputs/checkpoints/best_{model}.pt` | C1/C2 |
| `outputs/calibration_params.json` | C3 |
| `outputs/ensemble_weights.json` | C4 |
| `outputs/final_metrics.json` | D |
| `outputs/significance_tests.json` | D |
| `outputs/final_comparison.png` | D |
| `outputs/confusion_matrices_final.png` | D |
| `evaluation_audit/AUDIT_REPORT.md` | D |

---

## 12. Approved Tech Stack Constraints

| Component | Allowed | NOT Allowed |
|-----------|---------|-------------|
| Text encoder | XLM-RoBERTa-base | BERT, RoBERTa-large, mBERT |
| Image encoder | ResNet50 (ImageNet pretrained) | CLIP, ViT, EfficientNet (unless labelled Extra Experiment) |
| Fusion | Concat, Cross-Attention | Transformer decoder, BLIP, etc. |
| OCR | EasyOCR (already in pipeline) | — |
| Training | PyTorch, fp16, gradient accumulation | — |
| GPU | RTX 2050 4GB VRAM | Requires batch ≤16, mixed precision |

---

*This protocol was written before any test-set evaluation and must be followed without modification.*
