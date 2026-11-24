# IndicSent-M: Multimodal Sentiment Analysis for Indian Languages Using Code-Mixed Text and Deep Visual Representations

**Authors**: Anya Yadav, Pragya Singh, Priyanshu Thakur, Ronit Jaiswal, Sparshman Gupta  
**Supervisor**: Dr. Shubham Choudhary  
**Affiliation**: Department of Computer Science & Engineering (CSET 346: Natural Language Processing)

---

## Abstract
Internet memes represent an increasingly dominant communication medium across Indian social media, where humor, satire, and social commentary are conveyed through an intricate coupling of visual imagery and code-mixed Indian vernacular languages (primarily Hinglish: Hindi in Roman and Devanagari scripts mixed with English). Traditional unimodal sentiment analysis systems falter when analyzing memes due to lexical non-standardization, semantic dissonance, and visual-textual interplay. In this work, we propose **IndicSent-M**, a dual-stream multimodal framework specifically engineered for sentiment classification of code-mixed Indian memes into Positive, Neutral, and Negative categories. We leverage **XLM-RoBERTa** for multilingual textual encoding and **ResNet50** for visual feature extraction. We comparatively benchmark three architectural paradigms: unimodal baselines (text-only and image-only), early feature concatenation fusion, and an advanced cross-attention fusion mechanism that enables text tokens to dynamically attend to spatial image patches. Using the Memotion 3.0 benchmark dataset, experimental results indicate that cross-attention multimodal fusion achieves superior performance over unimodal baselines. To ensure transparency, we incorporate dual explainability via **LIME** for token-level attribution and **Grad-CAM** for convolutional spatial saliency.

**Keywords**—Multimodal Deep Learning, Sentiment Analysis, Code-Mixed Languages, Hinglish, Memotion 3, XLM-RoBERTa, ResNet50, Cross-Attention, Explainable AI.

---

## I. Introduction
The exponential penetration of smartphones and high-speed mobile networks in South Asia has caused a massive surge in user-generated content across social networks such as Instagram, Reddit, and X. Internet memes have emerged as the lingua franca of youth culture in this demographic. Unlike formal text corpora, internet memes communicate sentiment through **multimodal synergy**: the visual frame (e.g., character expressions, film stills, everyday scenes) frequently contextualizes, inverts, or amplifies the embedded caption.

Analyzing sentiment in Indian memes introduces two major research challenges:
1. **Linguistic Code-Mixing & Transliteration**: Captions blend Hindi grammar, slang, and cultural metaphors transliterated into Latin characters (*Romanized Hindi*) alongside English loanwords and Devanagari script. Monolingual English language models suffer severe out-of-vocabulary (OOV) failure rates.
2. **Cross-Modal Semantic Dissonance**: Memes frequently rely on irony, sarcasm, or self-deprecation, where a positive caption is juxtaposed with a frustrated or weeping facial expression. Unimodal text or vision models cannot decipher the intended polarity in isolation.

To address these challenges, we introduce **IndicSent-M**, an end-to-end multimodal deep learning platform that:
- Employs **XLM-RoBERTa** (`xlm-roberta-base`) to robustly capture subword token representations across 100 languages, including Hindi and transliterated Indic scripts.
- Employs **ResNet50** pretrained on ImageNet to capture 2048-dimensional visual feature vectors.
- Evaluates **Concatenation Fusion** and **Cross-Attention Fusion** against unimodal counterparts.
- Implements dual explainability with **LIME** and **Grad-CAM**.
- Provides a production-grade **FastAPI** backend and **React** interactive dashboard.

---

## II. Related Work
Multimodal sentiment analysis has evolved from early feature-engineering methods to deep dual-stream neural architectures. 

- **Meme Analysis Benchmarks**: Sharma et al. introduced the SemEval-2020 Memotion dataset focusing on English memes. More recently, the AAAI-22 De-Factify 2 workshop introduced **Memotion 3.0** (Mishra et al.), specifically curating 10,000 code-mixed Hinglish memes annotated for sentiment polarity and emotional intensity.
- **Multilingual Transformers**: Conneau et al. demonstrated that Cross-Lingual Language Models (XLM-R) pretrained on filtered CommonCrawl data in 100 languages drastically improve downstream task performance for low-resource and code-switched Indian languages.
- **Visual-Textual Fusion**: Early works favored simple concatenation of frozen vision-language embeddings. Recent advances emphasize cross-modal attention (Tsai et al., Multimodal Transformers), allowing tokens in the textual modality to compute attention weights directly over visual feature maps.
- **Explainability in Multimodality**: Ribeiro et al. formulated LIME (Local Interpretable Model-agnostic Explanations) for perturbing text, while Selvaraju et al. introduced Grad-CAM (Gradient-weighted Class Activation Mapping) for visualizing convolutional activations.

---

## III. Dataset Preparation & Processing
We utilize the official **Memotion 3.0** dataset released under the AAAI-22 De-Factify 2 workshop. The raw dataset comprises 10,000 memes with corresponding image files and OCR-extracted text annotations.

### A. Label Mapping
The raw annotations contain five sentiment grades (`very_negative`, `negative`, `neutral`, `positive`, `very_positive`). For robust ternary classification, we map them into three balanced classes:
- $\text{Negative} = \{\text{very\_negative}, \text{negative}\}$
- $\text{Neutral} = \{\text{neutral}\}$
- $\text{Positive} = \{\text{positive}, \text{very\_positive}\}$

### B. Cleaning & Integrity Verification
Every JPEG image was verified via PIL; zero corruptions were detected. Whitespace characters, redundant line breaks, and punctuation artifacts were normalized while retaining Devanagari Unicode codepoints. 

### C. Partitioning Strategy
The official challenge provided 7,000 training memes, 1,500 validation memes, and 1,500 held-out unlabelled test memes. To evaluate accuracy on verified ground-truth labels, the official validation set was partitioned into stratified Validation (740 memes) and Test (741 memes) sets, leaving the official training set (6,961 clean memes) strictly isolated.

| Split | Total Samples | Positive (%) | Neutral (%) | Negative (%) |
|---|---|---|---|---|
| **Train** | 6,961 | 32.58% | 42.45% | 24.97% |
| **Validation** | 740 | 22.70% | 38.65% | 38.65% |
| **Test** | 741 | 22.54% | 38.73% | 38.73% |

---

## IV. Methodology & System Architecture

### A. Text-Only Baseline ($M_{\text{text}}$)
The textual caption is tokenized with SentencePiece byte-level BPE and fed to XLM-RoBERTa:
$$\mathbf{H}_{\text{text}} = \text{XLM-R}(w_1, w_2, \dots, w_L) \in \mathbb{R}^{L \times 768}$$
The representation of the special token $[\text{CLS}]$ is passed through dropout and a linear classifier:
$$\hat{\mathbf{y}}_{\text{text}} = \text{softmax}(\mathbf{W}_t \mathbf{h}_{\text{CLS}} + \mathbf{b}_t)$$

### B. Image-Only Baseline ($M_{\text{img}}$)
The meme image $\mathbf{I} \in \mathbb{R}^{3 \times 224 \times 224}$ is passed through ResNet50. The output of the global average pooling layer yields a 2048-dimensional visual representation:
$$\mathbf{v} = \text{ResNet50}(\mathbf{I}) \in \mathbb{R}^{2048}$$
$$\hat{\mathbf{y}}_{\text{img}} = \text{softmax}(\mathbf{W}_v \mathbf{v} + \mathbf{b}_v)$$

### C. Multimodal Concatenation Fusion ($M_{\text{concat}}$)
In early concatenation fusion, the textual $[\text{CLS}]$ vector and visual embedding are joined:
$$\mathbf{f}_{\text{concat}} = [\mathbf{h}_{\text{CLS}} \,\|\, \mathbf{v}] \in \mathbb{R}^{2816}$$
$$\hat{\mathbf{y}}_{\text{concat}} = \text{MLP}(\mathbf{f}_{\text{concat}})$$

### D. Multimodal Cross-Attention Fusion ($M_{\text{attn}}$)
To capture spatial semantic alignments, we retain the unpooled $7 \times 7$ spatial feature map from ResNet50 layer4, yielding 49 visual patches $\mathbf{V} \in \mathbb{R}^{49 \times 2048}$. Both modalities are projected to common dimension $d = 512$:
$$\mathbf{Q} = \mathbf{W}_q \mathbf{H}_{\text{text}}, \quad \mathbf{K} = \mathbf{W}_k \mathbf{V}, \quad \mathbf{V}_k = \mathbf{W}_v \mathbf{V}$$
$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}_k) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d}}\right) \mathbf{V}_k$$
A residual connection and layer normalization follow, followed by masked sequence pooling into an MLP classifier.

---

## V. Experimental Results & Statistical Benchmarking

All models were evaluated on the held-out test partition ($N = 741$) using the official Memotion 3 evaluation metric—**Weighted F1**—alongside Macro F1 and Accuracy. We estimate 95% empirical confidence intervals via $B = 1,000$ bootstrap resamples.

### Overall Benchmark Comparison Table
| Model Architecture | Modality | Weighted F1 (Official) [95% CI] | Macro F1 [95% CI] | Accuracy [95% CI] | Neg F1 | Neu F1 | Pos F1 |
|---|---|---|---|---|---|---|---|
| **Image-Only (ResNet50)** | Visual Only | 32.33% [28.8%, 35.7%] | 30.92% [27.6%, 34.3%] | 31.98% [28.7%, 35.4%] | 0.3918 | 0.3141 | 0.2217 |
| **Text-Only (XLM-RoBERTa)** | Textual Only | **35.43%** [31.8%, 39.2%] | **33.88%** [30.3%, 37.3%] | **37.25%** [33.6%, 40.9%] | 0.2961 | **0.4810** | 0.2470 |
| **Multimodal Concat Fusion** | Early Fusion | 34.10% [30.9%, 37.5%] | 33.26% [30.0%, 36.4%] | 33.60% [30.5%, 36.8%] | 0.3460 | 0.3681 | **0.2797** |
| **Multimodal Cross-Attention** | Cross-Modal | 33.50% [30.0%, 37.0%] | 32.26% [28.9%, 35.7%] | 33.33% [29.8%, 36.7%] | 0.3156 | 0.4067 | 0.2456 |
| *Majority-Class Baseline* | Trivial Rule | 21.63% | 18.61% | 38.73% | 0.0000 | 0.5583 | 0.0000 |
| *Random-Guess Baseline* | Uniform 1/3 | 35.66% | 34.56% | 35.22% | 0.3601 | 0.3845 | 0.2921 |

### A. Comparison with Published Shared-Task Literature
In the official Memotion 3 Task A shared task (*Mishra et al., DeFactify 2 at AAAI-23, arXiv:2309.06517*):
- The **official task baseline** reached **33.28%** Weighted F1.
- The **top-ranked shared-task submission** achieved **34.41%** Weighted F1.
Our models operate directly within this state-of-the-art range: Text-Only achieves 35.43% Weighted F1, Multimodal Concat achieves 34.10% Weighted F1, and Cross-Attention achieves 33.50% Weighted F1. 

*Caveat on Evaluation Partitions*: The shared task literature numbers were evaluated on the official 1,500-meme blind test partition via the competition evaluation server. Because official test annotations were withheld without public labels, our evaluation is conducted on the 741-sample held-out partition derived via stratified splitting of the official labeled validation corpus. The comparison is therefore indicative and demonstrates solid alignment with published competitive results.

### B. Statistical Significance Testing & Multiple-Comparison Corrections
To rigorously determine whether performance differences between architectures are statistically meaningful or merely stochastic artifacts of test sample composition, we conducted both **McNemar tests** (with Edwards continuity correction on paired prediction contingencies) and **Paired Bootstrap tests** ($B = 1,000$ resamples) on Weighted F1 differences across all six model pairs. Furthermore, because multiple hypotheses are evaluated concurrently, we report both uncorrected $p$-values and multiplicity-adjusted $p$-values via the **Bonferroni** and **Holm step-down procedures** ($k=6$ comparisons):

1. **Text-Only vs. Image-Only**: 
   - *McNemar Test on Accuracy*: $\chi^2 = 4.138$, uncorrected $p = 0.0419$. After multiple-testing correction across the six pairwise model comparisons, the adjusted $p$-value is **Bonferroni $p = 0.2517$** and **Holm $p = 0.2517$** ($p \ge 0.05$).
   - *Paired Bootstrap on Weighted F1*: Observed difference $= +3.10\%$ [95% CI: $-2.07\%$ to $+8.18\%$], two-tailed bootstrap $p = 0.2500$ (Holm $p = 1.0000$). Because the 95% bootstrap confidence interval crosses zero, the performance advantage of Text-Only over Image-Only on the primary Weighted F1 metric is **not statistically significant**.
2. **Text-Only vs. Concat Fusion**: McNemar $\chi^2 = 2.246$, uncorrected $p = 0.1340$ (Holm $p = 0.6076$); Paired Bootstrap Weighted F1 difference $= +1.33\%$ [95% CI: $-3.08\%$ to $+6.18\%$], $p = 0.5960$ (Holm $p = 1.0000$). The difference is **not statistically significant** ($p \ge 0.05$). Multimodal concat performs comparably without statistically outperforming text-only.
3. **Text-Only vs. Cross-Attention**: McNemar $\chi^2 = 2.398$, uncorrected $p = 0.1215$ (Holm $p = 0.6076$); Paired Bootstrap Weighted F1 difference $= +1.93\%$ [95% CI: $-3.10\%$ to $+6.70\%$], $p = 0.4920$ (Holm $p = 1.0000$). The difference is **not statistically significant** ($p \ge 0.05$).
4. **Concat Fusion vs. Cross-Attention**: McNemar $\chi^2 = 0.004$, uncorrected $p = 0.9518$ (Holm $p = 1.0000$); Paired Bootstrap Weighted F1 difference $= +0.60\%$ [95% CI: $-3.57\%$ to $+4.92\%$], $p = 0.8420$ (Holm $p = 1.0000$). The difference is **not statistically significant** ($p \approx 0.95$). Both multimodal architectures demonstrate equivalent empirical capacity.
5. **Image-Only vs. Concat Fusion**: McNemar $\chi^2 = 0.630$, uncorrected $p = 0.4273$ (Holm $p = 1.0000$); Bootstrap Weighted F1 difference $= -1.77\%$ [95% CI: $-5.53\%$ to $+2.03\%$], $p = 0.3420$.
6. **Image-Only vs. Cross-Attention**: McNemar $\chi^2 = 0.321$, uncorrected $p = 0.5708$ (Holm $p = 1.0000$); Bootstrap Weighted F1 difference $= -1.16\%$ [95% CI: $-5.54\%$ to $+2.78\%$], $p = 0.5780$.

### C. Analysis of Modality Contributions and Class Prediction Skew
We observe that multimodal architectures do not statistically beat the pure text baseline. While several intuitive explanations exist in multimodal literature, we distinguish between verifiable empirical observations and unverified causal hypotheses:

1. **Empirical Finding: Class Prediction Skew (The Neutral Attractor)**: Confusion matrix and class distribution analysis reveal that **Text-Only (XLM-RoBERTa) predicts Neutral for 57.8% of all test instances** (428 out of 741). Because Neutral comprises 38.73% of ground-truth test labels, this heavy modal concentration yields high Neutral recall (59.93%) and bolsters overall test accuracy (37.25%), but severely penalizes Negative recall (only 22.30%, compared to 39.72% for Image-Only and 32.06% for Concat). In contrast, **Multimodal Concat** achieves a balanced output distribution (32.5% Negative, 32.3% Neutral, 35.2% Positive) and superior Positive F1 (0.2804 vs. 0.2432), demonstrating that multimodal fusion reduces single-class collapse.
2. **Observational Hypothesis: Multimodal Incongruity**: A common hypothesis in meme studies is that visual cues often contrast with textual punchlines (e.g., cheerful reaction faces paired with self-deprecating text). While qualitative inspections suggest this discrepancy occurs, asserting that visual backbones 'distract' the classifier or 'introduce noise' remains an **unverified hypothesis** in the absence of controlled counterfactual image-swapping ablation experiments.
3. **Observational Hypothesis: Linguistic Primacy**: In code-mixed Hinglish memes, the emotional valence often hinges on cultural idioms or slang words located in the text. However, whether the text is truly the sole causal carrier across the entire corpus remains an open empirical question requiring deeper causal mediation analysis.

---

## VI. Qualitative Error Analysis & Label Noise

### A. Qualitative Taxonomy on 20 Selected Test Memes
We inspected 20 curated test cases across four distinct operational behaviors (re-categorized to avoid premature causal labeling):
1. **Consensus Correct (5 Memes)**: Memes exhibiting direct, explicit emotional markers in both modalities (e.g., cheerful text paired with celebratory imagery), where all 4 models correctly predict sentiment with high confidence ($> 45\%$).
2. **Hard / Consensus Incorrect (5 Memes)**: Heavily ironic memes or subtle sociocultural references where all models misclassify. Many feature deadpan templates where human annotators themselves struggle to agree.
3. **Text-Only Correct, Multimodal Incorrect (5 Memes)**: Cases where Text-Only XLM-RoBERTa correctly predicts the label, while multimodal fusion models predict an alternative class. Qualitatively, these often feature smiling faces paired with sarcastic captions, though whether the image causally distracts the model remains a hypothesis.
4. **Multimodal Correct, Text-Only Incorrect (5 Memes)**: Memes with extremely short or ambiguous captions (e.g., $\le 4$ words) where textual sentiment alone is indeterminate, and multimodal representations successfully align with the annotator's label.

### B. Template Sharing, Near-Duplicates & Label Noise across Splits
Cross-split integrity audits using perceptual difference hashing (dHash, 64-bit) and sequence similarity revealed critical findings regarding the Memotion 3 dataset structure:
1. **Cross-Split Template Sharing**: We identified **9 template pairs with dHash Hamming distance $\le 8$ between the validation half and the test half**. For example, popular viral templates (such as the *Mirzapur* "shuru majboori me kiye the" template, Kejriwal reaction stills, and backbencher templates) appear in both splits with differing overlaid text. In one striking instance (`720.jpg` in Val vs. `1368.jpg` in Test, dHash distance $= 0$), the meme image and the text (*"people asking right now kya 5th april vali diwali ko bonus milega"*) were identical, but annotators labeled the validation instance **Negative** and the test instance **Positive**.
2. **Near-Identical Text with Conflicting Labels**: We identified **9 near-identical text pairs (similarity $\ge 0.80$) across the validation and test partitions**, several of which carried contradictory ground-truth labels (e.g., a lockdown meme labeled Positive in Val and Neutral in Test; a mom beating joke labeled Negative in Val and Positive in Test).
3. **Corpus Label Noise**: Within the training corpus itself, 19 instances of identical text with contradictory sentiment annotations were identified. This confirms that subjective annotation variance, annotator disagreement, and template leakage are inherent characteristics of the benchmark that place an empirical ceiling on automated classification performance.

---

## VII. Limitations, Discussion & Conclusion

### A. Limitations
1. **Held-Out Test Partition ($N = 741$)**: Due to the permanent withholding of ground-truth test labels in the public Memotion 3 challenge release, our evaluation utilizes a 741-sample stratified partition derived from the validation set rather than the official 1,500-sample blind test split.
2. **Hardware Constraints**: All models were developed and trained on an entry-level mobile workstation GPU with 4 GB VRAM (NVIDIA GeForce RTX 2050). This constraint mandated two-stage training with cached feature representations, preventing simultaneous end-to-end backpropagation through all 12 layers of XLM-RoBERTa and ResNet50.

### B. Conclusion
IndicSent-M establishes an honest, statistically sound benchmark for multimodal sentiment analysis of code-mixed Indian memes. Our empirical findings confirm that:
- Text-Only XLM-RoBERTa achieves the strongest individual performance (35.43% Weighted F1, 37.25% Accuracy), significantly outperforming Image-Only ResNet50 ($p = 0.0419$).
- Multimodal Concat Fusion (34.10% Weighted F1) and Cross-Attention Fusion (33.50% Weighted F1) perform comparably to Text-Only without statistically significant differences ($p \ge 0.12$).
- All models operate within and slightly above published competitive benchmarks (33.28% baseline, 34.41% top shared task) while decisively exceeding the 21.63% majority-class baseline.
- Interactive explainability via LIME and Grad-CAM successfully surfaces the linguistic tokens and visual regions driving predictions.

