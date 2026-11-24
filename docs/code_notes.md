# IndicSent-M: Code Walkthrough

**For students who need to understand and explain the codebase during a viva.**

---

## Project Purpose

IndicSent-M is a multimodal sentiment analysis system for Indian/Hinglish internet memes. It uses both the **text** (extracted by OCR) and the **image** of a meme to predict whether the sentiment is Negative, Neutral, or Positive.

This is built for the **Memotion 3** shared task (AAAI-23 Workshop). The task is known to be very hard — the official best submission reached only 34.41% Weighted F1.

---

## Repository Map

```
IndicSent-M/
├── backend/              # Python server and all ML code
├── frontend/             # React web application
├── data/                 # Dataset (not in package; see data/README.md)
├── notebooks/            # Three Jupyter notebooks for demonstration
├── outputs/              # Trained models, metrics, figures
├── scripts/              # Training, evaluation, and reporting scripts
├── docs/                 # Papers, protocols, documentation
├── tests/                # Automated tests
├── start_all.bat         # One-click startup script (Windows)
├── stop_all.bat          # One-click stop script (Windows)
└── README.md             # Setup and usage guide
```

---

## Backend Files (`backend/`)

### `preprocess.py`
**Purpose**: Cleans and prepares the raw Memotion 3 dataset.  
**What it does**:
- Reads `train.csv`, `val.csv`, `test.csv` from the Memotion 3 dataset.
- Maps the original multi-task labels to 3 sentiment classes: Negative (0), Neutral (1), Positive (2).
- Removes rows with missing images or labels.
- Saves clean CSVs to `data/processed/`.
- **Run once** before any training.

### `dataset.py`
**Purpose**: PyTorch Dataset class that loads memes during training.  
**Key class: `MemeDataset`**  
- Accepts a CSV file path.
- Loads each image with PIL and applies ResNet50 transforms (resize to 224×224, ImageNet normalization).
- Tokenizes the text with XLM-RoBERTa tokenizer (max 64 tokens).
- Returns a dictionary with `input_ids`, `attention_mask`, `image`, `label`.

### `models.py`
**Purpose**: Defines the 4 neural network architectures.

| Class | Input | Architecture |
|-------|-------|-------------|
| `TextOnlyModel` | Text tokens | XLM-RoBERTa-base → [CLS] → Dropout → Linear(768→3) |
| `ImageOnlyModel` | Image tensor | ResNet50 → AvgPool → Dropout → Linear(2048→3) |
| `MultimodalConcatModel` | Text + Image | XLM-RoBERTa [CLS] + ResNet50 pool → LayerNorm → Gated fusion (256-dim) → MLP → 3 |
| `MultimodalAttentionModel` | Text seq + Image patches | XLM-RoBERTa seq → Cross-Attention over 9 spatial image regions → masked pool → MLP → 3 |

**Key design choices**:
- **Gated Concat**: A sigmoid gate `g = σ(W[t;i])` decides how much text vs. image to use. Text gets an extra residual: `g*t + (1-g)*i + 0.25*t` to prevent the 2048-dim image from dominating the 768-dim text.
- **Cross-Attention**: Text tokens attend over 9 downsampled visual regions (from 49 spatial patches via `AdaptiveAvgPool1d(9)`). This prevents memorizing noisy pixel-level details.

### `train.py`
**Purpose**: Simple training script for quick experiments (not the main pipeline).  
Supports `--model text|image|concat|attention`, `--epochs`, `--batch_size`.  
Uses class-weighted Cross-Entropy, AdamW, fp16, gradient accumulation.

### `train_and_evaluate_all.py`
**Purpose**: Main production training + evaluation script.  
**What it does**:
1. Loads frozen XLM-RoBERTa and ResNet50, extracts features for train/val/test, caches to disk.
2. Trains lightweight head networks on cached features (fast — no GPU memory for backbones).
3. Runs 3 random seeds per model, keeps best checkpoint by validation Weighted F1.
4. Evaluates on the test set **exactly once**.
5. Saves checkpoints, comparison.json, figures.

### `evaluate.py`
**Purpose**: Standalone evaluation on any split.  
Loads checkpoints, runs inference, prints accuracy/F1/confusion matrix.

### `inference.py`
**Purpose**: Production inference engine used by the FastAPI server.  
**Key functions**:
- `load_all_models()`: Loads all 4 checkpoints into GPU memory at startup.
- `predict()`: Takes an image (upload or URL) + optional text, runs OCR if text not provided, runs all models or a selected model, returns sentiment + confidence + explanation.
- `generate_explanations()`: Runs Grad-CAM for image saliency and LIME for text highlights.
- `TEST_METRICS`: Real accuracy numbers read from JSON (never hardcoded).

### `app.py`
**Purpose**: FastAPI web server.  
**Endpoints**:
- `GET /health`: Server health check.
- `POST /predict`: Main prediction endpoint (multipart/form-data with image file).
- `GET /metrics`: Returns all model accuracy numbers from JSON.
- `GET /compare`: Returns side-by-side model comparison data.
- `POST /explain`: Progressive explainability (Grad-CAM + LIME).

### `ocr_engine.py`
**Purpose**: Text extraction from meme images using EasyOCR.  
**Key functions**:
- `extract_meme_text()`: Runs EasyOCR, deduplicates text blocks, returns cleaned text.
- `compute_image_hash()`: Computes 64-bit dHash for near-duplicate detection.

### `explain.py`
**Purpose**: Explainability utilities.  
- `GradCAM`: Hooks into ResNet50's last conv layer, computes gradient-weighted activation maps, overlays as heatmaps.
- `lime_text_explain()`: Perturbs text tokens, calls model, identifies which words most influence prediction.

---

## Frontend Files (`frontend/`)

Built with React 18 + Vite + Tailwind CSS.

| File | Purpose |
|------|---------|
| `src/App.jsx` | Main app with React Router — routes to 5 pages |
| `src/pages/Home.jsx` | Landing page with project summary |
| `src/pages/Analyze.jsx` | Upload a meme and get a prediction + explanation |
| `src/pages/Compare.jsx` | Side-by-side model accuracy comparison table |
| `src/pages/Dataset.jsx` | Dataset stats and class distribution charts |
| `src/pages/Results.jsx` | Detailed per-model metrics and confusion matrices |
| `src/pages/About.jsx` | Architecture diagrams and research background |
| `src/components/Navbar.jsx` | Navigation bar |

---

## Scripts (`scripts/`)

| Script | Purpose |
|--------|---------|
| `phase_c_full_pipeline.py` | Random search + calibration + ensemble + final test eval |
| `build_report_tables.py` | Reads JSON → writes audit report and results tables |
| `build_submission_package.py` | Builds clean submission folder |
| `update_inference_metrics.py` | Syncs final_metrics.json → inference.py |
| `audit_dataset_and_splits.py` | Verifies leakage / near-duplicates |
| `compute_rigorous_statistics.py` | McNemar, bootstrap CI, Holm correction |
| `generate_complete_audit.py` | Full audit pipeline |

---

## Training Pipeline Flow

```
raw data
   │
   ▼ preprocess.py
data/processed/train.csv, val.csv, test.csv
   │
   ▼ train_and_evaluate_all.py (Phase 1: feature extraction)
outputs/features/{train,val,test}_features.pt
   │
   ▼ train_and_evaluate_all.py (Phase 2: head training)
   │  → 3 seeds × 4 models on cached features
   │  → early stopping on val Weighted F1
   │  → test eval ONCE
outputs/checkpoints/best_{text,image,concat,attention}.pt
outputs/comparison.json, outputs/final_metrics.json
   │
   ▼ build_report_tables.py
evaluation_audit/AUDIT_REPORT.md, docs/Results_Table.md
   │
   ▼ update_inference_metrics.py
backend/inference.py (TEST_METRICS updated)
```

---

## Viva Questions & Suggested Answers

**Q: Why use XLM-RoBERTa instead of BERT?**  
A: XLM-RoBERTa is trained on 100 languages including Hindi and handles multilingual Hinglish/code-mixed text better. BERT is primarily English.

**Q: Why freeze most of the backbone?**  
A: Our dataset has only ~7,000 training memes. Fine-tuning all 125M XLM-R parameters would overfit badly. Freezing lower layers and training only the last few layers + head gives better generalization.

**Q: What is the Gated Concat fusion doing differently from plain concatenation?**  
A: Plain concat just joins 768-dim text and 2048-dim image vectors. The 2048-dim image dominates due to magnitude difference. The gate `g = σ(W[t;i])` learns a sample-specific balance. The residual `+0.25*t` ensures text sentiment is never ignored.

**Q: Why use 9 spatial regions instead of 49 for Cross-Attention?**  
A: 49 noisy patches (7×7) lead to memorization of background pixels. Pooling to 9 regions (3×3 approximate) forces attention to coarser semantic regions, improving generalization.

**Q: Why are accuracy numbers below the majority-class baseline (38.73%)?**  
A: The majority baseline predicts only Negative (38.73% of the test set), so it trivially gets 38.73% accuracy. Our models predict all 3 classes, distributing errors across classes. The official metric is Weighted F1, where our models score 32–36% vs the baseline's 21.6%.

**Q: Why are no models significantly better than each other?**  
A: Memotion 3 Task A is an extremely hard task with high label noise (subjective sentiment), code-mixed OCR text, and only ~7,000 training examples. This is well-documented — the best published submission was only 34.41% Weighted F1.
