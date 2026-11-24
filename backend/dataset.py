"""
dataset.py - PyTorch Dataset & Transforms for IndicSent-M
Handles image preprocessing (224x224, ImageNet normalization) and
XLM-RoBERTa text tokenization (max length 128 / 64) for Memotion 3.
"""

import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
from torchvision import transforms

# ImageNet normalization statistics
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def get_image_transforms(is_train=True):
    """
    Returns torchvision transforms for ResNet50.
    Light augmentations (horizontal flip, subtle color jitter) applied only during training.
    """
    if is_train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])

class MemeDataset(Dataset):
    """
    Multimodal Dataset for Internet Memes.
    Returns:
      - input_ids: XLM-R token IDs
      - attention_mask: XLM-R attention mask
      - image: Normalized 3x224x224 tensor for ResNet50
      - label: Tensor containing class index (0: Negative, 1: Neutral, 2: Positive)
      - text: Raw text string
      - image_path: Path to local image file
    """
    def __init__(self, csv_file, tokenizer=None, max_length=128, is_train=False, subset=None):
        if isinstance(csv_file, str):
            self.df = pd.read_csv(csv_file)
        elif isinstance(csv_file, pd.DataFrame):
            self.df = csv_file.copy()
        else:
            raise ValueError("csv_file must be a filepath string or a pandas DataFrame")

        # Optional subset for fast debugging / testing
        if subset is not None and subset > 0 and subset < len(self.df):
            self.df = self.df.sample(n=subset, random_state=42).reset_index(drop=True)

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_train = is_train
        self.transform = get_image_transforms(is_train=self.is_train)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = str(row['text']) if pd.notna(row['text']) else ""
        img_path = str(row['image_path'])
        
        # Load image safely
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            # Fallback blank canvas if file IO glitch
            image = Image.new('RGB', (224, 224), color=(128, 128, 128))
            
        img_tensor = self.transform(image)

        # Tokenize text if tokenizer provided
        if self.tokenizer is not None:
            encoding = self.tokenizer(
                text,
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            input_ids = encoding['input_ids'].squeeze(0)
            attention_mask = encoding['attention_mask'].squeeze(0)
        else:
            # Dummy tensors if text tokenizer not used (e.g. ImageOnly model)
            input_ids = torch.zeros(self.max_length, dtype=torch.long)
            attention_mask = torch.zeros(self.max_length, dtype=torch.long)

        # Label extraction
        if 'label' in row and pd.notna(row['label']):
            label = torch.tensor(int(row['label']), dtype=torch.long)
        else:
            label = torch.tensor(-1, dtype=torch.long)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'image': img_tensor,
            'label': label,
            'text': text,
            'image_path': img_path
        }

