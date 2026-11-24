# IndicSent-M: Project Summary

**Course**: CSET 346 - Natural Language Processing  
**Faculty**: Dr. Shubham Choudhary  
**Team**: Anya Yadav, Pragya Singh, Priyanshu Thakur, Ronit Jaiswal, Sparshman Gupta

---

## Problem

Indian memes on social media use Hinglish — a mix of Hindi (Roman + Devanagari script) and English. Normal English-only models like BERT can't handle this. Also, memes are multimodal — the image and caption together convey the sentiment, and sometimes they contradict each other (e.g., a funny-looking image with a sad caption).

We built **IndicSent-M** to handle both issues by combining multilingual text understanding with image features.

---

## Approach

- **Text**: XLM-RoBERTa (100-language multilingual transformer) for the meme caption
- **Image**: ResNet50 (ImageNet pretrained) for visual features
- **Fusion approaches compared**:
  1. Text only
  2. Image only
  3. Feature concatenation (simple fusion)
  4. Cross-attention (text tokens attend to image spatial patches)
- **Explainability**: LIME (word attributions) + Grad-CAM (image heatmaps)
- **Dataset**: Memotion 3.0 (10,000 Hinglish memes, AAAI-23 benchmark)

---

## System Architecture

```
Input Meme
    |
    +----- Caption text ------> XLM-RoBERTa --> text features (768-dim)
    |                                               |
    +----- Meme image --------> ResNet50 ---------> visual features (2048-dim / 7x7 patches)
                                                    |
                                       Concat or Cross-Attention Fusion
                                                    |
                                             Sentiment (Pos/Neu/Neg)
```

---

## Results (Test set, N=741)

| Model | Accuracy | Weighted F1 |
|---|---|---|
| Text-Only (XLM-R) | **37.25%** | **35.42%** |
| Multimodal Concat | 33.60% | 34.10% |
| Multimodal Cross-Attn | 33.33% | 33.51% |
| Image-Only (ResNet50) | 32.12% | 32.46% |

Text-only actually performed best — which we think is because the Memotion 3 OCR captions carry most of the sentiment information directly. The images are mostly standard meme templates.

Cross-attention achieved 90%+ training accuracy but generalized poorly (overfitting issue with limited data).

---

## Deliverables

- Source code: `backend/` (PyTorch models, FastAPI server), `frontend/` (React dashboard)
- Trained model checkpoints: `outputs/checkpoints/`
- Jupyter notebooks for review: `notebooks/`
- Research paper draft: `docs/paper_draft.md`
- API docs: `docs/api_examples.md`
- Dataset documentation: `docs/dataset_documentation.md`
