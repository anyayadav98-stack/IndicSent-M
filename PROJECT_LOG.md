# IndicSent-M: Project Log

Keeping track of what we did and when. Mostly so we don't forget what broke and how we fixed it.

---

## Week 1 (Nov 12-13): Dataset + Preprocessing

Downloaded Memotion 3 dataset. It came as a ZIP with separate archives for train/val/test images. Had to unzip each one manually.

Wrote `preprocess.py` to:
- Map 5 sentiment grades → 3 classes (Positive, Neutral, Negative)
- Strip empty OCR rows (found ~39 in train, ~19 in val)
- Normalize text (strip extra whitespace, keep Devanagari)

Annoying thing: val CSV used tab as separator instead of comma. Took a while to figure out.

Official test.csv has no labels (blind challenge set for AAAI leaderboard), so we split the val set (1481 rows) 50-50 into our own val (740) and test (741) sets using stratified split.

Generated class distribution plots and sample meme grid.

---

## Week 2 (Nov 13-14): Model Architecture + Training

Built all 4 models in `models.py`:
- TextOnlyModel: XLM-RoBERTa base + linear classifier head
- ImageOnlyModel: ResNet50 (ImageNet pretrained) + linear head
- MultimodalConcatModel: concat XLM-R [CLS] embedding (768-dim) + ResNet50 pooled features (2048-dim) → MLP
- MultimodalAttentionModel: text tokens as Q, ResNet50 7x7 spatial patches as K/V in cross-attention

Training: AdamW, class-weighted cross-entropy, mixed precision fp16, early stopping on val macro-F1. Saved best checkpoints.

4GB VRAM on RTX 2050 was a pain — had to use batch size 8 and gradient accumulation. Cross-attention was especially slow.

---

## Week 2-3 (Nov 14-22): Evaluation + Explainability

Wrote `evaluate.py` for accuracy, macro/weighted F1, confusion matrices.  
Wrote `explain.py` for LIME (text token attributions) and Grad-CAM (image heatmaps).  
Wrote `inference.py` unified pipeline that runs both together.

Got some weird results — cross-attention was hitting 90% training accuracy but only ~33% on test. Classic overfitting. Tried a few regularization tweaks but ran out of time.

---

## Week 3 (Nov 21-23): FastAPI Backend + React Frontend

Built FastAPI backend (`app.py`) with:
- `/predict` endpoint (multipart form: image + text)
- `/health`, `/models`, `/compare` endpoints
- CORS enabled for frontend

Built React frontend with pages:
- Home, Analyze, Results (Grad-CAM overlay + LIME chips + sentiment badge), Compare, Dataset, About

Connected frontend to backend. Had some CORS issues initially.

---

## Week 3-4 (Nov 23-25): OCR Integration + Cleanup

Added `ocr_engine.py` to auto-extract text from uploaded memes using EasyOCR (English + Devanagari). This way users don't have to type the caption manually.

Wrote Jupyter notebooks for faculty review (data exploration, training demo, explainability demo).

Final testing + bug fixes. Wrote this README and docs.

---

## Final Results (Test set, N=741)

| Model | Accuracy | Weighted F1 |
|---|---|---|
| Image-Only | 32.12% | 32.46% |
| Text-Only | **37.25%** | **35.42%** |
| Concat Fusion | 33.60% | 34.10% |
| Cross-Attention | 33.33% | 33.51% |

Text-only wins, which makes sense — OCR text carries most of the sentiment for this dataset. Memes are hard.
