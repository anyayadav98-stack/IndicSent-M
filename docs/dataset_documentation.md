# IndicSent-M: Memotion 3 Dataset Documentation

## 1. Dataset Overview & Source
- **Dataset Name**: Memotion 3.0 (AAAI-22 De-Factify 2 Workshop Benchmark)
- **Primary Domain**: Multimodal sentiment analysis and emotion classification for Internet memes in Indian languages (specifically Hindi-English code-mixed Hinglish).
- **Academic Citation Reference**: Mishra et al., *"Memotion 3: Dataset on Sentiment and Emotion Analysis of Codemixed Hindi-English Memes"*, AAAI Workshop on Multimodal Fact-Checking and Hate Speech Analysis (De-Factify 2).
- **Core Challenge**: Memes convey sentiment through complex interplay between visual context, facial expressions, cultural metaphors, and embedded code-mixed text.

---

## 2. Raw vs Cleaned Dataset Sizes
- **Official Training Set**: 7,000 raw memes $\rightarrow$ **6,961 clean memes** (39 rows removed due to empty OCR strings after normalization).
- **Official Validation Set**: 1,500 raw memes $\rightarrow$ **1,481 clean memes** (19 rows removed due to empty OCR strings).
- **Official Blind Test Set**: 1,500 raw memes $\rightarrow$ **1,484 clean memes** (held-out blind evaluation set without labels).

### Processed Split Strategy for Academic Evaluation:
Because the official `test.csv` was released unlabelled for blind leaderboard evaluation, we utilize the official training set strictly for training, and divide the 1,481 verified ground-truth validation set into stratified Validation (740 samples) and Test (741 samples) sets to enable rigorous metric computation (Accuracy, Precision, Recall, Macro F1, Confusion Matrix).

| Split | Raw Count | Cleaned Count | % of Labelled Data | Image Directory | Purpose |
|---|---|---|---|---|---|
| **Train** | 7,000 | **6,961** | 82.46% | `data/memotion3/trainImages/` | Model parameter optimization |
| **Validation** | 750 (from val.csv) | **740** | 8.77% | `data/memotion3/valImages/` | Early stopping & hyperparameter tuning |
| **Test** | 750 (from val.csv) | **741** | 8.77% | `data/memotion3/valImages/` | Unbiased benchmark evaluation |
| **Blind Test** | 1,500 | **1,484** | - | `data/memotion3/testImages/` | Blind inference & deployment demo |

---

## 3. Data Schema & Columns
The processed splits in `data/processed/` adhere to the standardized schema:

| Column Name | Data Type | Description |
|---|---|---|
| `image_path` | string | Relative or absolute path to the local verified JPEG meme image. |
| `text` | string | Cleaned multilingual OCR text extracted from the meme. |
| `sentiment` | string | Coarse 3-class sentiment label: `Positive`, `Neutral`, or `Negative`. |
| `label` | integer | Numeric target index: `0: Negative`, `1: Neutral`, `2: Positive`. |
| `image_url` | string | (Optional) Original web source URL. |

---

## 4. Sentiment Label Mapping
The raw Memotion 3 dataset contains 5 fine-grained emotion/sentiment annotations in column `overall`. For 3-class polarity classification, labels were mapped as follows:

| Raw Label | Mapped Class | Numeric Target | Rationale |
|---|---|---|---|
| `positive` | **Positive** | 2 | Moderate positive sentiment |
| `very_positive` | **Positive** | 2 | Strong positive valence merged into Positive |
| `neutral` | **Neutral** | 1 | Objective, informational, or balanced memes |
| `negative` | **Negative** | 0 | Moderate negative sentiment / disparagement |
| `very_negative` | **Negative** | 0 | Severe negative valence merged into Negative |

---

## 5. Class Distributions

### Train Set ($N = 6,961$):
- **Neutral**: 2,955 samples (42.45%)
- **Positive**: 2,268 samples (32.58%)
- **Negative**: 1,738 samples (24.97%)

### Validation Set ($N = 740$):
- **Neutral**: 286 samples (38.65%)
- **Negative**: 286 samples (38.65%)
- **Positive**: 168 samples (22.70%)

### Test Set ($N = 741$):
- **Neutral**: 287 samples (38.73%)
- **Negative**: 287 samples (38.73%)
- **Positive**: 167 samples (22.54%)

---

## 6. Data Cleaning & Integrity Steps
1. **Image Integrity Verification**: Every image in `trainImages/`, `valImages/`, and `testImages/` was opened and verified via `PIL.Image.verify()`. Zero corrupted or missing image files were found across all 10,000 files.
2. **Text Normalization**: Stripped trailing escape characters, carriage returns (`\r\n`), and normalized irregular spacing while preserving Devanagari Unicode scripts and Hinglish transliterations.
3. **Empty Entry Filtering**: Dropped rows where OCR was blank or composed purely of punctuation.
4. **Duplicate Deduplication**: Validated pairwise uniqueness over `(text, image_path)`.

---

## 7. Known Dataset Limitations & Challenges
1. **Class Imbalance**: Neutral and Negative categories constitute over 75% of the validation/test distributions, with Positive being a minority (~22.6%). Weighted Cross-Entropy Loss is implemented in `backend/train.py` to prevent majority-class bias.
2. **Code-Mixed Hinglish**: Text blends Romanized Hindi ("*bhai ye kya kar diya*"), standard Hindi Devanagari ("*पोस्ट करने वाले*"), and English colloquial slang. Standard English BERT struggles; multilingual **XLM-RoBERTa** (`xlm-roberta-base`) is strictly utilized to address this.
3. **OCR Noise**: Raw OCR strings occasionally contain watermark artifacts (e.g. `RVCJ`, `9GAG`, `memechat`), broken typography, or missing overlay text when text blend into photographic backgrounds.
4. **Sarcasm and Incongruity**: Internet memes frequently exhibit semantic dissonance where positive text accompanies an ironic or satirical image. This emphasizes why multimodal fusion (concatenation and cross-attention) is required over unimodal baselines.

