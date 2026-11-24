# IndicSent-M: Viva Prep Notes

Quick notes for the demo + expected questions.

---

## Demo Flow (3 minutes)

**0:00 – 0:45 | Introduction**

Introduce the project — Indian memes use Hinglish (Hindi + English mixed) and standard BERT/text-only models don't handle this well. Also memes are multimodal — sometimes the image expression completely flips the caption's meaning. So we built a system that uses both text and image together.

Dataset: Memotion 3 (10,000 memes, Hinglish, AAAI-23 benchmark).  
Task: 3-class sentiment — Positive, Neutral, Negative.

---

**0:45 – 1:45 | Live Dashboard Demo**

Open `http://localhost:5173`

1. Go to Analyze page, upload any meme or use a preset example
2. Click Analyze
3. Show the Results page:
   - Final sentiment badge (e.g., Neutral, 56% confidence)
   - Grad-CAM heatmap on the image (shows which regions the model focused on)
   - LIME word chips (green = pushed toward positive, red = pushed negative)

---

**1:45 – 2:30 | Comparison**

Go to Compare tab. Show bar chart of Weighted F1 for all 4 models:
- Text-only: best (35.42%)
- Cross-attention: 33.51%
- Concat: 34.10%
- Image-only: 32.46%

Mention that these match the ballpark of published Memotion 3 baselines (33–34% range). Pairwise significance tests show no significant difference between text-only and multimodal — the OCR text just carries most of the sentiment signal in this dataset.

---

**2:30 – 3:00 | Wrap up**

Summarize: end-to-end system, 4 architectures compared, dual explainability (LIME + Grad-CAM), live web dashboard.

---

## Expected Viva Questions

**Q: Why XLM-RoBERTa and not BERT?**  
BERT is English-only. Hinglish memes have code-mixed text — Hindi words in Roman script (like "bhai", "yaar", "kya kar diya"). XLM-RoBERTa was trained on 100 languages including Hindi, so it handles subword tokenization for Romanized Hindi much better.

**Q: Why ResNet50?**  
ResNet50 gives 2048-dim residual features, works well off-the-shelf with ImageNet pretraining. The last conv layer gives a 7x7 spatial feature map (49 patches) which is exactly what we need for Grad-CAM and cross-attention. Something like ViT would be more modern but harder to train on 4GB VRAM.

**Q: Difference between concat fusion and cross-attention?**  
Concat just stacks the pooled text vector (768) and image vector (2048) → 2816-dim → MLP. No interaction, just merging.  
Cross-attention lets text tokens attend to image patches — so a word like "sad" can pay attention to sad-looking regions in the image. More expressive in theory.

**Q: Why did cross-attention overfit?**  
It has way more parameters (attention heads + projection layers) and we only had 6961 training samples. We used dropout and class weighting but didn't have time for a full hyperparameter sweep. Standard overfitting problem with small data + complex model.

**Q: How does class imbalance affect training?**  
Memotion 3 has more Negative and Neutral than Positive. Without correction, models just predict the majority class. We used class-weighted cross-entropy loss — weights inversely proportional to class frequencies.

**Q: How does LIME work here?**  
Mask individual words one at a time and see how the predicted probability changes. Words that cause big drops when removed get high attribution. Positive attribution = pushed toward the predicted class.

**Q: How does Grad-CAM work?**  
Take gradient of predicted class score w.r.t. the last ResNet conv layer. Average these gradients to get channel weights, then do weighted sum of feature maps and ReLU. Overlay onto original image.

**Q: Why did text-only outperform multimodal?**  
Most of the sentiment in Memotion 3 is carried in the OCR text (captions are quite direct). The images are mostly stock/meme template backgrounds. Cross-modal fusion didn't help because the visual information wasn't adding much signal over the text. This is a known issue with Memotion 3 specifically.

**Q: Limitations?**  
- OCR errors especially for stylized or heavily decorated text
- Sarcasm and cultural jokes that need external context
- Cross-attention overfits on small data
- No fine-tuning of ResNet50 backbone (kept frozen due to VRAM)
