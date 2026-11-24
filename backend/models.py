"""
models.py - Neural Network Architectures for IndicSent-M
Implements 4 architectures adhering to the approved tech stack:
1. TextOnlyModel: XLM-RoBERTa base
2. ImageOnlyModel: ResNet50 (ImageNet pretrained)
3. MultimodalConcatModel: Early concatenation fusion + MLP
4. MultimodalAttentionModel: Cross-attention fusion between text and image features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from transformers import XLMRobertaModel

class TextOnlyModel(nn.Module):
    """
    Unimodal Text Sentiment Model using multilingual XLM-RoBERTa.
    Projects the 768-dim [CLS] token representation into 3 sentiment classes.
    """
    def __init__(self, pretrained_name="xlm-roberta-base", num_classes=3, dropout=0.3, freeze_layers=8):
        super().__init__()
        self.text_encoder = XLMRobertaModel.from_pretrained(pretrained_name)
        
        # Memory optimization for 4GB VRAM: freeze lower transformer layers if specified
        if freeze_layers > 0:
            for param in self.text_encoder.embeddings.parameters():
                param.requires_grad = False
            for layer in self.text_encoder.encoder.layer[:freeze_layers]:
                for param in layer.parameters():
                    param.requires_grad = False
                    
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.text_encoder.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Use [CLS] token (first token)
        cls_output = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(self.dropout(cls_output))
        return logits


class ImageOnlyModel(nn.Module):
    """
    Unimodal Visual Sentiment Model using ResNet50 pretrained on ImageNet.
    Extracts 2048-dim global average pooled feature vectors and classifies.
    """
    def __init__(self, num_classes=3, dropout=0.3, freeze_base=True):
        super().__init__()
        # Pretrained ResNet50
        weights = ResNet50_Weights.DEFAULT
        base_resnet = resnet50(weights=weights)
        
        if freeze_base:
            # Freeze conv1, bn1, layer1, layer2 to conserve GPU memory
            for name, param in base_resnet.named_parameters():
                if "layer4" not in name and "layer3" not in name and "fc" not in name:
                    param.requires_grad = False

        # Keep everything except the final classification layer
        self.resnet_features = nn.Sequential(*list(base_resnet.children())[:-1])
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(2048, num_classes)

    def forward(self, image):
        # image: [batch_size, 3, 224, 224]
        feats = self.resnet_features(image) # [batch_size, 2048, 1, 1]
        feats = torch.flatten(feats, 1)     # [batch_size, 2048]
        logits = self.classifier(self.dropout(feats))
        return logits


class MultimodalConcatModel(nn.Module):
    """
    Optimized Multimodal Gated Concat Model.
    Uses LayerNorm, balanced projections, Sigmoid gating, and a residual text highway.
    Prevents 2048-dim visual features from dominating 768-dim text representations.
    """
    def __init__(self, pretrained_text="xlm-roberta-base", num_classes=3, hidden_dim=256, dropout=0.35, freeze_text_layers=8, freeze_img_base=True):
        super().__init__()
        # 1. Text Encoder (XLM-RoBERTa)
        self.text_encoder = XLMRobertaModel.from_pretrained(pretrained_text)
        if freeze_text_layers > 0:
            for param in self.text_encoder.embeddings.parameters():
                param.requires_grad = False
            for layer in self.text_encoder.encoder.layer[:freeze_text_layers]:
                for param in layer.parameters():
                    param.requires_grad = False

        # 2. Image Encoder (ResNet50)
        base_resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        if freeze_img_base:
            for name, param in base_resnet.named_parameters():
                if "layer4" not in name and "layer3" not in name and "fc" not in name:
                    param.requires_grad = False
        self.image_encoder = nn.Sequential(*list(base_resnet.children())[:-1])

        # 3. LayerNorm and Projections to balanced representation space (256-dim)
        self.text_norm = nn.LayerNorm(768)
        self.image_norm = nn.LayerNorm(2048)
        self.text_proj = nn.Linear(768, hidden_dim)
        self.image_proj = nn.Linear(2048, hidden_dim)

        # 4. Multimodal Sigmoid Gate
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )

        # 5. Classification MLP Head
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, input_ids, attention_mask, image):
        # Extract text CLS representation
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_feat = text_outputs.last_hidden_state[:, 0, :] # [B, 768]

        # Extract image global feature
        img_raw = self.image_encoder(image)                # [B, 2048, 1, 1]
        img_feat = torch.flatten(img_raw, 1)               # [B, 2048]

        # Normalized gated fusion with residual text highway
        t = F.gelu(self.text_proj(self.text_norm(text_feat)))
        i = F.gelu(self.image_proj(self.image_norm(img_feat)))
        g = self.gate(torch.cat([t, i], dim=1))
        fused = g * t + (1.0 - g) * i + 0.25 * t
        logits = self.classifier(fused)
        return logits


class MultimodalAttentionModel(nn.Module):
    """
    Regularized Multimodal Cross-Attention Model.
    Downsamples 49 spatial patches to 9 key spatial regions to prevent background pixel memorization.
    Uses d_model=256, 4 heads, LayerNorm, and higher dropout for robust generalization.
    """
    def __init__(self, pretrained_text="xlm-roberta-base", num_classes=3, d_model=256, nheads=4, dropout=0.45, freeze_text_layers=8, freeze_img_base=True):
        super().__init__()
        # 1. Text Encoder (XLM-RoBERTa)
        self.text_encoder = XLMRobertaModel.from_pretrained(pretrained_text)
        if freeze_text_layers > 0:
            for param in self.text_encoder.embeddings.parameters():
                param.requires_grad = False
            for layer in self.text_encoder.encoder.layer[:freeze_text_layers]:
                for param in layer.parameters():
                    param.requires_grad = False

        # 2. Image Feature Extractor (ResNet50 up to conv5_x / layer4)
        base_resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        if freeze_img_base:
            for name, param in base_resnet.named_parameters():
                if "layer4" not in name and "layer3" not in name and "fc" not in name:
                    param.requires_grad = False
        # Retain conv layers up to layer4: [B, 2048, 7, 7]
        self.image_encoder = nn.Sequential(*list(base_resnet.children())[:-2])

        # 3. LayerNorm and Projections to d_model (256)
        self.text_norm = nn.LayerNorm(768)
        self.image_norm = nn.LayerNorm(2048)
        self.text_proj = nn.Linear(768, d_model)
        self.image_proj = nn.Linear(2048, d_model)

        # 4. Spatial downsampling (49 patches -> 9 regions) & Multi-Head Cross Attention
        self.spatial_pool = nn.AdaptiveAvgPool1d(9)
        self.cross_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=nheads, dropout=dropout, batch_first=True)
        self.layer_norm = nn.LayerNorm(d_model)

        # 5. Regularized Classifier Head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, input_ids, attention_mask, image):
        # Text sequence states: [B, seq_len, 768]
        text_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_seq = text_out.last_hidden_state              # [B, L, 768]
        text_proj = F.gelu(self.text_proj(self.text_norm(text_seq))) # [B, L, d_model]

        # Image spatial features: [B, 2048, 7, 7] -> 49 spatial patches of 2048-dim
        img_spatial = self.image_encoder(image)           # [B, 2048, 7, 7]
        B, C, H, W = img_spatial.shape
        img_patches = img_spatial.view(B, C, H * W).permute(0, 2, 1) # [B, 49, 2048]
        img_proj = F.gelu(self.image_proj(self.image_norm(img_patches))) # [B, 49, d_model]

        # Adaptive pool 49 patches down to 9 semantic regions: [B, 9, d_model]
        i_regions = self.spatial_pool(img_proj.permute(0, 2, 1)).permute(0, 2, 1)

        # Cross attention: text queries attend to 9 visual regions
        attn_out, attn_weights = self.cross_attention(
            query=text_proj,
            key=i_regions,
            value=i_regions
        ) # [B, L, d_model]

        # Residual connection & Layer Norm
        fused_seq = self.layer_norm(text_proj + attn_out)

        # Pool over sequence using attention mask
        mask_expanded = attention_mask.unsqueeze(-1).expand_as(fused_seq).float()
        fused_sum = torch.sum(fused_seq * mask_expanded, dim=1)
        mask_sum = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        pooled = fused_sum / mask_sum # [B, d_model]

        logits = self.classifier(pooled)
        return logits

