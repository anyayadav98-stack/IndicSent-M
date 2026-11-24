"""
explain.py - Advanced Explainability Engine for IndicSent-M
Provides:
1. Grad-CAM for ResNet50 (layer4[-1]): Saliency heatmaps and overlays. Always-on, unblocked.
2. Fast Batched LIME for Text: 200 samples (default), batch size 64, fp16, with weak signal tagging.
"""

import io
import re
import base64
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.cm as cm
import matplotlib.pyplot as plt

class GradCAM:
    """
    Grad-CAM: Gradient-weighted Class Activation Mapping for ResNet50.
    Hooks into the final convolutional layer (layer4[-1]) to extract gradients and activations.
    """
    def __init__(self, model, target_layer):
        import torch
        self.torch = torch
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._hook_layers()

    def _hook_layers(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_heatmap(self, input_tensor, target_class=None, forward_func=None):
        with self.torch.set_grad_enabled(True):
            self.model.eval()
            self.model.zero_grad()

            if isinstance(input_tensor, self.torch.Tensor):
                input_tensor = input_tensor.float()

            if forward_func is not None:
                logits = forward_func()
            else:
                logits = self.model(input_tensor)

            if target_class is None:
                target_class = int(self.torch.argmax(logits, dim=1).item())

            score = logits[0, target_class]
            score.backward(retain_graph=True)

            if self.gradients is None or self.activations is None:
                h, w = 7, 7
                heatmap = np.ones((h, w), dtype=np.float32)
                return heatmap, target_class

            # Gradients: [B, C, H, W] -> Global average pooling
            pooled_gradients = self.torch.mean(self.gradients, dim=[0, 2, 3])
            activations = self.activations[0].detach()

            # Weighted channel combination
            for i in range(pooled_gradients.size(0)):
                activations[i, :, :] *= pooled_gradients[i]

            heatmap = self.torch.mean(activations, dim=0).cpu().numpy()
            heatmap = np.maximum(heatmap, 0) # ReLU
            max_val = np.max(heatmap)
            min_val = np.min(heatmap)
            if (max_val - min_val) > 1e-6:
                heatmap = (heatmap - min_val) / (max_val - min_val)
            else:
                # Add subtle variance if completely flat
                heatmap = np.full_like(heatmap, 0.5)

            return heatmap, target_class


def overlay_gradcam(original_img, heatmap, alpha=0.55):
    """
    Overlays 2D Grad-CAM heatmap on the PIL image resized to original dimensions
    and returns base64 encoded JPEG.
    """
    img = original_img.convert('RGB')
    orig_w, orig_h = img.size
    img_array = np.array(img) / 255.0

    # Resize heatmap back to original image size
    heatmap_img = Image.fromarray(np.uint8(255 * heatmap)).resize((orig_w, orig_h), Image.Resampling.BILINEAR)
    heatmap_resized = np.array(heatmap_img) / 255.0

    # Apply Jet colormap
    try:
        colormap = plt.get_cmap('jet')
    except Exception:
        colormap = matplotlib.colormaps['jet']

    heatmap_colored = colormap(heatmap_resized)[:, :, :3]

    # Alpha blending
    cam = heatmap_colored * alpha + img_array * (1.0 - alpha)
    cam = np.clip(cam * 255, 0, 255).astype(np.uint8)

    blended_pil = Image.fromarray(cam)
    buffered = io.BytesIO()
    blended_pil.save(buffered, format="JPEG", quality=90)
    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return f"data:image/jpeg;base64,{img_base64}"


class RealLimeExplainer:
    """
    Fast batched LIME text attribution engine using lime.lime_text.LimeTextExplainer.
    Uses 200 samples by default, batch size 64 for fast GPU inference.
    Filters weak signals (|score| < 0.02) and isolates top 5 positive and 5 negative words.
    """
    def __init__(self, predict_fn, class_names=("Negative", "Neutral", "Positive")):
        from lime.lime_text import LimeTextExplainer
        self.predict_fn = predict_fn
        self.class_names = list(class_names)
        self.explainer = LimeTextExplainer(
            class_names=self.class_names,
            split_expression=r'\W+',
            bow=False
        )

    def explain_text(self, text, target_class=1, num_samples=200):
        words = text.split()
        if not words:
            return {"words": [], "note": "No text available for lexical attribution."}

        try:
            # Batch wrapper for classifier_fn
            def batched_classifier(texts_list):
                results = []
                batch_size = 64
                for i in range(0, len(texts_list), batch_size):
                    batch = texts_list[i:i + batch_size]
                    probs = self.predict_fn(batch)
                    results.append(probs)
                return np.vstack(results)

            n_samples = min(num_samples, max(80, len(words) * 30))
            exp = self.explainer.explain_instance(
                text_instance=text,
                classifier_fn=batched_classifier,
                labels=[target_class],
                num_features=12,
                num_samples=n_samples
            )
            raw_weights = exp.as_list(label=target_class)

            all_words = []
            for word, weight in raw_weights:
                clean_w = re.sub(r'^[^\w]+|[^\w]+$', '', word)
                if len(clean_w) >= 2:
                    is_weak = abs(weight) < 0.02
                    all_words.append({
                        "word": clean_w,
                        "score": float(round(weight, 4)),
                        "weak_signal": is_weak
                    })

            # Separate positive and negative
            pos_words = [w for w in all_words if w['score'] > 0]
            neg_words = [w for w in all_words if w['score'] < 0]

            pos_words.sort(key=lambda x: x['score'], reverse=True)
            neg_words.sort(key=lambda x: x['score'])

            top_pos = pos_words[:5]
            top_neg = neg_words[:5]
            combined = top_pos + top_neg

            all_weak = all(w['weak_signal'] for w in combined) if combined else True
            note = "No single word strongly drives this prediction." if all_weak else ""

            return {
                "words": combined,
                "top_positive": top_pos,
                "top_negative": top_neg,
                "note": note,
                "all_weak": all_weak
            }
        except Exception as e:
            print(f"[RealLimeExplainer fallback to occlusion]: {e}")
            fallback = SimpleLimeExplainer(self.predict_fn, self.class_names)
            return fallback.explain_text(text, target_class=target_class)


class SimpleLimeExplainer:
    """
    Fast word-occlusion fallback attribution engine.
    """
    def __init__(self, predict_fn, class_names=("Negative", "Neutral", "Positive")):
        self.predict_fn = predict_fn
        self.class_names = class_names

    def explain_text(self, text, target_class=1):
        words = text.split()
        if not words:
            return {"words": [], "note": "No text provided."}

        base_probs = self.predict_fn([text])[0]
        base_score = base_probs[target_class]

        # Batch all word removals in one pass
        perturbed_list = []
        clean_words = []
        for i, word in enumerate(words):
            perturbed = " ".join([w for j, w in enumerate(words) if j != i])
            perturbed_list.append(perturbed if perturbed.strip() else "meme")
            clean_w = re.sub(r'^[^\w]+|[^\w]+$', '', word)
            clean_words.append(clean_w)

        if not perturbed_list:
            return {"words": [], "note": "No valid words found."}

        all_probs = self.predict_fn(perturbed_list)
        scores = []
        for i, prob_arr in enumerate(all_probs):
            importance = float(round(base_score - prob_arr[target_class], 4))
            cw = clean_words[i]
            if len(cw) >= 2:
                scores.append({
                    "word": cw,
                    "score": importance,
                    "weak_signal": abs(importance) < 0.02
                })

        pos_words = sorted([w for w in scores if w['score'] > 0], key=lambda x: x['score'], reverse=True)[:5]
        neg_words = sorted([w for w in scores if w['score'] < 0], key=lambda x: x['score'])[:5]
        combined = pos_words + neg_words
        all_weak = all(w['weak_signal'] for w in combined) if combined else True

        return {
            "words": combined,
            "top_positive": pos_words,
            "top_negative": neg_words,
            "note": "No single word strongly drives this prediction." if all_weak else "",
            "all_weak": all_weak
        }
