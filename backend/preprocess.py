"""
preprocess.py - Preprocessing script for Memotion 3 dataset in IndicSent-M
Maps sentiment labels into 3 classes: Positive, Neutral, Negative.
Cleans missing text/labels, verifies image integrity, handles duplicates,
and produces clean stratified train/val/test splits in data/processed/.
"""

import os
import re
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

# Base directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "memotion3")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

os.makedirs(PROCESSED_DIR, exist_ok=True)

# 3-Class Sentiment Mapping:
# Positive: 'positive', 'very_positive'
# Neutral: 'neutral'
# Negative: 'negative', 'very_negative'
LABEL_MAP = {
    "positive": "Positive",
    "very_positive": "Positive",
    "neutral": "Neutral",
    "negative": "Negative",
    "very_negative": "Negative"
}

LABEL_TO_INT = {
    "Negative": 0,
    "Neutral": 1,
    "Positive": 2
}

def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).strip()
    # Normalize whitespaces while keeping multilingual Indic and Latin characters
    text = re.sub(r'[\r\n]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def verify_image(image_path):
    if not os.path.exists(image_path):
        return False
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True
    except Exception:
        return False

def load_and_clean_split(csv_path, img_dir, sep=','):
    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path, sep=sep)
    print(f"Initial count: {len(df)}")
    
    # Identify image id column
    id_col = 'Unnamed: 0' if 'Unnamed: 0' in df.columns else df.columns[0]
    
    # Map labels if overall sentiment column exists
    if 'overall' in df.columns:
        df['raw_label'] = df['overall'].astype(str).str.strip().str.lower()
        df['sentiment'] = df['raw_label'].map(LABEL_MAP)
        # Drop rows with unmapped or missing labels
        df = df.dropna(subset=['sentiment']).copy()
        df['label'] = df['sentiment'].map(LABEL_TO_INT)
    else:
        df['sentiment'] = None
        df['label'] = None
        
    # Clean OCR text
    if 'ocr' in df.columns:
        df['text'] = df['ocr'].apply(clean_text)
        # Drop rows with empty text
        df = df[df['text'].str.len() > 0].copy()
    else:
        df['text'] = ""

    # Build image file paths
    def get_img_path(val):
        s = str(val).strip()
        fname = s if s.endswith('.jpg') else f"{s}.jpg"
        return os.path.join(img_dir, fname)

    df['image_path'] = df[id_col].apply(get_img_path)
    
    # Verify image exists and is not corrupt
    valid_mask = df['image_path'].apply(verify_image)
    df = df[valid_mask].copy()
    
    # Drop duplicates by text and image_path
    before_dedup = len(df)
    df = df.drop_duplicates(subset=['text', 'image_path']).copy()
    print(f"Cleaned count: {len(df)} (Dropped {before_dedup - len(df)} duplicates)")
    
    return df

def run_preprocessing():
    print("=" * 60)
    print("IndicSent-M Data Preprocessing Pipeline")
    print("=" * 60)

    train_csv = os.path.join(RAW_DIR, "train.csv")
    val_csv = os.path.join(RAW_DIR, "val.csv")
    test_csv = os.path.join(RAW_DIR, "test.csv")
    
    train_imgs = os.path.join(RAW_DIR, "trainImages")
    val_imgs = os.path.join(RAW_DIR, "valImages")
    test_imgs = os.path.join(RAW_DIR, "testImages")

    # Load and clean official train (comma-separated)
    df_train = load_and_clean_split(train_csv, train_imgs, sep=',')
    
    # Load and clean official val (tab-separated)
    df_val_raw = load_and_clean_split(val_csv, val_imgs, sep='\t')
    
    # Load official blind test
    df_blind_test = load_and_clean_split(test_csv, test_imgs, sep=',')

    # The official test.csv has no ground truth labels.
    # To evaluate models on a true held-out test set with real labels,
    # we stratify official val into 50% Val (750) and 50% Test (750).
    # Official train (7000) remains purely train.
    print("\nSplitting official validation set into Validation and Test sets with stratification...")
    df_val, df_test = train_test_split(
        df_val_raw,
        test_size=0.50,
        random_state=42,
        stratify=df_val_raw['label']
    )

    cols_to_save = ['image_path', 'text', 'sentiment', 'label']
    if 'image_url' in df_train.columns:
        cols_to_save.append('image_url')

    train_out = os.path.join(PROCESSED_DIR, "train.csv")
    val_out = os.path.join(PROCESSED_DIR, "val.csv")
    test_out = os.path.join(PROCESSED_DIR, "test.csv")
    blind_test_out = os.path.join(PROCESSED_DIR, "official_blind_test.csv")

    df_train[cols_to_save].to_csv(train_out, index=False)
    df_val[cols_to_save].to_csv(val_out, index=False)
    df_test[cols_to_save].to_csv(test_out, index=False)
    df_blind_test[['image_path', 'text']].to_csv(blind_test_out, index=False)

    print("\nProcessed files successfully saved:")
    print(f" - Train: {train_out} ({len(df_train)} rows)")
    print(f" - Val:   {val_out} ({len(df_val)} rows)")
    print(f" - Test:  {test_out} ({len(df_test)} rows)")
    print(f" - Blind Test: {blind_test_out} ({len(df_blind_test)} rows)")

    print("\n" + "=" * 60)
    print("Class Distributions (Positive, Neutral, Negative):")
    print("=" * 60)
    
    for name, df in [("Train", df_train), ("Val", df_val), ("Test", df_test)]:
        print(f"\n--- {name} Set (Total: {len(df)}) ---")
        counts = df['sentiment'].value_counts()
        percents = df['sentiment'].value_counts(normalize=True) * 100
        for cls in ["Positive", "Neutral", "Negative"]:
            c = counts.get(cls, 0)
            p = percents.get(cls, 0.0)
            print(f"  {cls:8s}: {c:5d} ({p:5.2f}%)")

if __name__ == "__main__":
    run_preprocessing()

