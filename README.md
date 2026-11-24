# IndicSent-M: Multimodal Sentiment Analysis for Indian Memes

**Course**: CSET 346 - Natural Language Processing  
**Instructor**: Dr. Shubham Choudhary  
**Team**:
- Anya Yadav  
- Pragya Singh  
- Priyanshu Thakur  
- Ronit Jaiswal  
- Sparshman Gupta  

---

## What is this project?

Indian internet memes use a mix of Hindi and English (called Hinglish) in both Roman and Devanagari scripts. Normal NLP models like BERT don't handle this well because they're English-only and also completely ignore the image part of a meme. The goal of this project is to build a system that looks at BOTH the text caption and the image together to figure out if a meme is Positive, Negative, or Neutral.

We used the **Memotion 3.0** dataset from AAAI-23 (around 10,000 Hinglish memes with labels) and built 4 models to compare:

1. Text only — using XLM-RoBERTa (handles multilingual/code-mixed text)
2. Image only — using ResNet50 (pretrained on ImageNet)
3. Multimodal with simple feature concat
4. Multimodal with cross-attention (text attends to image regions)

We also added explainability: LIME for text tokens and Grad-CAM for image heatmaps.

---

## Folder Structure

```
IndicSent M/
├── backend/
│   ├── app.py              # FastAPI server
│   ├── models.py           # All 4 model architectures
│   ├── train.py            # Training loop
│   ├── evaluate.py         # Evaluation metrics
│   ├── inference.py        # Inference pipeline
│   ├── explain.py          # LIME + Grad-CAM
│   ├── dataset.py          # PyTorch Dataset class
│   ├── preprocess.py       # Data cleaning
│   ├── compare.py          # Model comparison
│   ├── plot_dataset.py     # Dataset plots
│   ├── test_api.py         # Quick API test
│   └── requirements.txt
├── data/
│   ├── memotion3/          # Raw dataset images
│   └── processed/          # Cleaned CSVs (train/val/test)
├── docs/
│   ├── dataset_documentation.md
│   ├── paper_draft.md
│   ├── demo_script.md
│   ├── api_examples.md
│   └── project_report_summary.md
├── frontend/               # React + Vite + Tailwind dashboard
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_training_and_comparison.ipynb
│   └── 03_explainability_demo.ipynb
├── outputs/
│   ├── checkpoints/        # Saved model weights (.pt files)
│   ├── figures/            # Plots and graphs
│   └── comparison.json
├── start_all.bat
├── stop_all.bat
└── README.md
```

---

## How to Run

### Option 1: One-click (Windows)

Just double-click `start_all.bat` or run it in cmd:
```
start_all.bat
```
This starts:
- Backend (FastAPI) at `http://localhost:8000`
- Frontend (React) at `http://localhost:5173`

To stop everything:
```
stop_all.bat
```

### Option 2: Manual (Two Terminals)

**Terminal 1 — Backend:**
```powershell
.\.venv\Scripts\activate
python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

**Terminal 2 — Frontend:**
```powershell
cd frontend
npm run dev
```

Then open `http://localhost:5173` in your browser.

---

## Results Summary

Evaluated on 741 test samples (held-out from Memotion 3 val set since official test has no labels):

| Model | Accuracy | Weighted F1 |
|---|---|---|
| Image-Only (ResNet50) | 32.12% | 32.46% |
| Text-Only (XLM-R) | 37.25% | 35.42% |
| Multimodal Concat | 33.60% | 34.10% |
| Multimodal Cross-Attention | 33.33% | 33.51% |

Text-only performed best on this dataset mostly because Memotion 3 OCR text carries most of the sentiment signal. Cross-attention overfitted on training (90%+ accuracy) but generalized poorly — something we tried to address with better regularization.

---

## Setup Notes

- Python 3.12, PyTorch, XLM-RoBERTa (via HuggingFace), ResNet50 (torchvision)
- Trained on RTX 2050 (4GB VRAM) — batch size 8 with mixed precision (fp16)
- Frontend: React 18, Vite, TailwindCSS, Recharts
- `.venv/` already set up. If not, run `pip install -r backend/requirements.txt` inside a venv.

VS Code users: `.vscode/settings.json` already points to `.venv`. Do `Ctrl+Shift+P` → Reload Window to pick up the interpreter.

---

## Notes / Known Issues

- Official test.csv from Memotion 3 has no labels (blind challenge set), so we split the validation set 50-50 for val/test evaluation.
- EasyOCR for Devanagari detection is slow — first run will download model weights (~200MB).
- Cross-attention model trains well but overfits noticeably — we didn't have time to do full hyperparameter search.
