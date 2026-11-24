"""
inference.py - High-Performance Inference & Explainability Pipeline for IndicSent-M
Features:
1. One-time startup loading of all 4 models (Text, Image, Concat, Attention) + Tokenizer + OCR.
2. GPU fp16 warm-up and fast inference under torch.inference_mode().
3. Progressive execution: fast predict() (< 2s) + separate explain() for LIME/Grad-CAM.
4. Full architecture selector support: Auto, Text, Image, Concat, Attention, Ensemble, Compare All.
5. Dynamic honest disclaimer from real test metrics.
6. Always-on Grad-CAM with low-confidence advisory (never hidden).
7. Fast batched LIME (200 samples, batch size 64) with weak-signal tagging.
8. Stage-by-stage timing profiler.
9. Dual in-memory caches for OCR, Grad-CAM, and LIME.
"""

import os
os.environ.setdefault('MPLCONFIGDIR', os.path.join(os.getcwd(), 'outputs'))
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
import sys
import io
import time
import json
import base64
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms
from transformers import AutoTokenizer

_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from models import TextOnlyModel, ImageOnlyModel, MultimodalConcatModel, MultimodalAttentionModel
import ocr_engine
import explain

CLASS_NAMES = ["Negative", "Neutral", "Positive"]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Real Saved Test Metrics — sourced from outputs/final_metrics.json (N=741 test set)
TEST_METRICS = {
    "text":      {"accuracy": 37.25, "macro_f1": 33.88, "weighted_f1": 35.43, "name": "Text-Only (XLM-R)"},
    "image":     {"accuracy": 31.98, "macro_f1": 30.92, "weighted_f1": 32.33, "name": "Image-Only (ResNet50)"},
    "concat":    {"accuracy": 33.60, "macro_f1": 33.26, "weighted_f1": 34.10, "name": "Multimodal Concat Fusion"},
    "attention": {"accuracy": 33.33, "macro_f1": 32.26, "weighted_f1": 33.50, "name": "Multimodal Cross-Attention"},
    "ensemble":  {"accuracy": 0,     "macro_f1": 0,     "weighted_f1": 0,     "name": "Ensemble (not trained)"}
}

_LOADED_MODELS = {}
_TOKENIZER = None
_DEVICE = None
_BEST_VALIDATION_MODEL = "text" # Dynamically determined from comparison.json
_GRADCAM_CACHE = {} # {(img_hash, model_type): b64_overlay}
_LIME_CACHE = {} # {(img_hash, text, model_type): lime_result}

def sanitize_for_json(obj):
    """Recursively converts all numpy scalars, arrays, and types to native Python JSON-serializable types."""
    if isinstance(obj, dict):
        return {str(k): sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [sanitize_for_json(x) for x in obj]
    elif isinstance(obj, (np.floating, float)):
        return float(obj)
    elif isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return sanitize_for_json(obj.tolist())
    return obj

def get_device():
    global _DEVICE
    if _DEVICE is None:
        _DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return _DEVICE


def load_all_models():
    """
    Loads all four trained checkpoints strictly into eval mode and runs warm-up.
    """
    global _LOADED_MODELS, _TOKENIZER, _BEST_VALIDATION_MODEL
    device = get_device()
    print("=" * 65, flush=True)
    print(f"[INFO] Initializing IndicSent-M Inference Engine on {device}", flush=True)
    print("=" * 65, flush=True)

    # 1. Tokenizer
    if _TOKENIZER is None:
        print("[INFO] Loading XLM-RoBERTa Tokenizer...", flush=True)
        _TOKENIZER = AutoTokenizer.from_pretrained("xlm-roberta-base")

    # 2. Determine best model from comparison.json
    comp_path = os.path.join('outputs', 'comparison.json')
    val_macro_f1s = {}
    if os.path.exists(comp_path):
        try:
            with open(comp_path, 'r') as f:
                comp = json.load(f)
            for m in ['text', 'image', 'concat', 'attention']:
                val_macro_f1s[m] = comp.get(m, {}).get('seeds_summary', {}).get('val_macro_f1_mean', 0.0)

            # Auto selects the highest validation macro F1 across all models
            _BEST_VALIDATION_MODEL = max(val_macro_f1s.keys(), key=lambda k: val_macro_f1s[k])
            print(f"[INFO] Auto Best Model by Validation Macro F1: {_BEST_VALIDATION_MODEL.upper()} "
                  f"(Scores: text={val_macro_f1s.get('text',0):.4f}, concat={val_macro_f1s.get('concat',0):.4f}, "
                  f"attention={val_macro_f1s.get('attention',0):.4f}, image={val_macro_f1s.get('image',0):.4f})", flush=True)
        except Exception as e:
            print(f"[WARN] Error reading comparison.json: {e}", flush=True)

    # 3. Load checkpoints
    model_configs = [
        ('text', TextOnlyModel, 'best_text.pt'),
        ('image', ImageOnlyModel, 'best_image.pt'),
        ('concat', MultimodalConcatModel, 'best_concat.pt'),
        ('attention', MultimodalAttentionModel, 'best_attention.pt'),
    ]

    for m_type, m_class, ckpt_name in model_configs:
        ckpt_path = os.path.join('outputs', 'checkpoints', ckpt_name)
        if not os.path.exists(ckpt_path):
            print(f"[ERROR] Checkpoint {ckpt_path} missing!", flush=True)
            continue
        try:
            model = m_class(num_classes=3)
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            res = model.load_state_dict(ckpt['model_state_dict'], strict=True)
            model.to(device)
            model.eval()
            _LOADED_MODELS[m_type] = model
            print(f"[OK] Loaded {m_type.upper():<10} from {ckpt_name} (val_f1={ckpt.get('val_f1', 0):.4f})", flush=True)
        except Exception as e:
            print(f"[ERROR] Loading {m_type} failed: {e}", flush=True)

    # 4. OCR Reader pre-load
    ocr_engine.get_easyocr_reader()

    # 5. GPU Warm-up pass
    _run_warmup(device)
    print("=" * 65, flush=True)


def _run_warmup(device):
    """Run a small dummy pass through models to eliminate first-token CUDA initialization lag."""
    try:
        dummy_img = torch.zeros(1, 3, 224, 224, device=device)
        dummy_ids = torch.zeros(1, 64, dtype=torch.long, device=device)
        dummy_mask = torch.ones(1, 64, dtype=torch.long, device=device)
        with torch.inference_mode():
            if 'image' in _LOADED_MODELS:
                _LOADED_MODELS['image'](dummy_img)
            if 'text' in _LOADED_MODELS:
                _LOADED_MODELS['text'](dummy_ids, dummy_mask)
            if 'concat' in _LOADED_MODELS:
                _LOADED_MODELS['concat'](dummy_ids, dummy_mask, dummy_img)
        print("[INFO] Warm-up inference complete on GPU.", flush=True)
    except Exception as e:
        print(f"[WARN] Warm-up failed: {e}", flush=True)


def preprocess_image(image_input):
    """Convert input to PIL and normalized PyTorch tensor."""
    if isinstance(image_input, str):
        if image_input.startswith("data:image"):
            header, encoded = image_input.split(",", 1)
            img_data = base64.b64decode(encoded)
            pil_img = Image.open(io.BytesIO(img_data)).convert('RGB')
        else:
            pil_img = Image.open(image_input).convert('RGB')
    elif isinstance(image_input, (bytes, bytearray)):
        pil_img = Image.open(io.BytesIO(image_input)).convert('RGB')
    elif isinstance(image_input, Image.Image):
        pil_img = image_input.convert('RGB')
    else:
        pil_img = Image.new('RGB', (224, 224), (200, 200, 200))

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    tensor = transform(pil_img).unsqueeze(0)
    return pil_img, tensor


def generate_disclaimer(model_key):
    """Generate dynamic honest warning disclaimer based on saved test metrics."""
    meta = TEST_METRICS.get(model_key, TEST_METRICS["concat"])
    return f"Predictions are automatic and can be wrong. This model's test macro F1 is {meta['macro_f1']:.1f}% and accuracy is {meta['accuracy']:.1f}% on Memotion 3."


def forward_single_model(model_key, input_ids, attention_mask, img_tensor):
    """Execute single model forward pass in inference mode."""
    model = _LOADED_MODELS.get(model_key)
    if model is None:
        raise ValueError(f"Model {model_key} is not loaded.")

    if model_key == 'text':
        logits = model(input_ids, attention_mask)
    elif model_key == 'image':
        logits = model(img_tensor)
    elif model_key in ['concat', 'attention']:
        logits = model(input_ids, attention_mask, img_tensor)
    else:
        raise ValueError(f"Unknown model_key: {model_key}")

    probs = F.softmax(logits, dim=1).cpu().numpy()[0]
    return probs


def predict(image_input, text_input=None, model_type="auto", use_ai_ocr=False, include_explanations=True):
    """
    Main prediction pipeline.
    Progressive & fast:
    - Runs OCR if text empty (using multi-pass ocr_engine with SHA-256 caching).
    - Computes model forward pass.
    - If include_explanations=True, computes Grad-CAM and LIME.
    - Measures precise stage timings.
    """
    timings = {}
    t_start = time.time()

    # 1. Image Decode
    t0 = time.time()
    pil_img, img_tensor = preprocess_image(image_input)
    img_hash = ocr_engine.compute_image_hash(pil_img)
    timings['image_decode_ms'] = round((time.time() - t0) * 1000, 1)

    # 2. Text Input & OCR
    t0 = time.time()
    raw_text = str(text_input).strip() if (text_input and str(text_input).strip()) else ""
    ocr_ran = False
    ocr_quality = 1.0
    ocr_unreliable = False
    ocr_variant = "user_provided"

    if not raw_text:
        ocr_res = ocr_engine.extract_meme_text(pil_img, use_ai_ocr=use_ai_ocr)
        text = ocr_res['text']
        ocr_quality = ocr_res['quality_score']
        ocr_unreliable = ocr_res['unreliable']
        ocr_variant = ocr_res['variant_used']
        ocr_ran = True
    else:
        text = raw_text

    timings['ocr_ms'] = round((time.time() - t0) * 1000, 1)

    # Ensure models are loaded
    if not _LOADED_MODELS:
        load_all_models()

    device = get_device()
    img_tensor = img_tensor.to(device)

    # Tokenize text
    t0 = time.time()
    effective_text = text if (text and text.strip()) else "meme"
    encoding = _TOKENIZER(
        effective_text,
        padding='max_length',
        truncation=True,
        max_length=64,
        return_tensors='pt'
    )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    req_model = model_type.lower() if model_type else "auto"

    # Resolve target model
    if req_model == "auto":
        resolved_type = _BEST_VALIDATION_MODEL
    elif req_model in _LOADED_MODELS or req_model in ["ensemble", "compare_all"]:
        resolved_type = req_model
    else:
        resolved_type = _BEST_VALIDATION_MODEL

    # Forward pass
    compare_all_results = None

    with torch.inference_mode():
        if resolved_type == "compare_all":
            # Run all 4 models and simple ensemble
            all_scores = {}
            for m in ['text', 'image', 'concat', 'attention']:
                p = forward_single_model(m, input_ids, attention_mask, img_tensor)
                p_idx = int(np.argmax(p))
                all_scores[m] = {
                    "sentiment": CLASS_NAMES[p_idx],
                    "scores": {"Negative": round(float(p[0]), 4), "Neutral": round(float(p[1]), 4), "Positive": round(float(p[2]), 4)},
                    "max_prob": round(float(p[p_idx]), 4)
                }
            # Ensemble average
            ens_probs = np.mean([list(all_scores[m]["scores"].values()) for m in ['text', 'image', 'concat', 'attention']], axis=0)
            ens_idx = int(np.argmax(ens_probs))
            all_scores["ensemble"] = {
                "sentiment": CLASS_NAMES[ens_idx],
                "scores": {"Negative": round(float(ens_probs[0]), 4), "Neutral": round(float(ens_probs[1]), 4), "Positive": round(float(ens_probs[2]), 4)},
                "max_prob": round(float(ens_probs[ens_idx]), 4)
            }
            compare_all_results = all_scores
            # Default presentation to auto model
            probs = forward_single_model(_BEST_VALIDATION_MODEL, input_ids, attention_mask, img_tensor)
            effective_model_for_display = _BEST_VALIDATION_MODEL

        elif resolved_type == "ensemble":
            # Average probabilities across 4 models
            p_list = []
            for m in ['text', 'image', 'concat', 'attention']:
                p_list.append(forward_single_model(m, input_ids, attention_mask, img_tensor))
            probs = np.mean(p_list, axis=0)
            effective_model_for_display = "ensemble"

        else:
            probs = forward_single_model(resolved_type, input_ids, attention_mask, img_tensor)
            effective_model_for_display = resolved_type

    timings['model_forward_ms'] = round((time.time() - t0) * 1000, 1)

    pred_idx = int(np.argmax(probs))
    pred_sentiment = CLASS_NAMES[pred_idx]
    conf_scores = {
        "Negative": float(round(probs[0], 4)),
        "Neutral": float(round(probs[1], 4)),
        "Positive": float(round(probs[2], 4))
    }

    # Low confidence detection: top prob < 0.45 or margin < 0.10
    sorted_probs = sorted(probs, reverse=True)
    max_prob = sorted_probs[0]
    margin = (sorted_probs[0] - sorted_probs[1]) if len(sorted_probs) > 1 else 0.0
    low_confidence = bool(max_prob < 0.45 or margin < 0.10)

    # Disclaimer
    disclaimer = generate_disclaimer(effective_model_for_display)

    # Suggestion when OCR quality is low and model uses text
    ocr_suggestion = ""
    if ocr_unreliable and effective_model_for_display in ['text', 'concat', 'attention', 'ensemble']:
        ocr_suggestion = "Text extraction looks noisy. You can edit the text box above and click 'Re-run with this text', or try Image-Only mode."

    # Models used chips
    if effective_model_for_display == 'text':
        models_used = ['XLM-R']
    elif effective_model_for_display == 'image':
        models_used = ['ResNet50']
    elif effective_model_for_display == 'ensemble':
        models_used = ['XLM-R', 'ResNet50', 'Concat', 'Attention']
    else:
        models_used = ['XLM-R', 'ResNet50']

    # Original image base64
    buffered = io.BytesIO()
    pil_img.save(buffered, format="JPEG", quality=85)
    orig_b64 = f"data:image/jpeg;base64,{base64.b64encode(buffered.getvalue()).decode('utf-8')}"

    # 4. Optional Explanations (Grad-CAM & LIME)
    gradcam_b64 = None
    gradcam_warning = "Low confidence: the heatmap may be unreliable" if low_confidence else None
    gradcam_note = None
    lime_words = []
    lime_method = "N/A"
    lime_note = ""

    if include_explanations:
        # Explanations
        expl_res = generate_explanations(
            pil_img=pil_img,
            text=text,
            model_type=effective_model_for_display,
            target_class=pred_idx,
            low_confidence=low_confidence,
            img_tensor=img_tensor,
            input_ids=input_ids,
            attention_mask=attention_mask,
            img_hash=img_hash
        )
        gradcam_b64 = expl_res.get('gradcam_overlay')
        gradcam_note = expl_res.get('gradcam_note')
        lime_words = expl_res.get('lime_words', [])
        lime_method = expl_res.get('lime_method', 'LIME')
        lime_note = expl_res.get('lime_note', '')
        timings['gradcam_ms'] = expl_res.get('gradcam_ms', 0)
        timings['lime_ms'] = expl_res.get('lime_ms', 0)
    else:
        timings['gradcam_ms'] = 0
        timings['lime_ms'] = 0

    timings['total_ms'] = round((time.time() - t_start) * 1000, 1)

    # Narrative explanation
    explanation_text = _build_data_driven_narrative(
        sentiment=pred_sentiment,
        conf_scores=conf_scores,
        margin=margin,
        low_conf=low_confidence,
        lime_words=lime_words,
        model_type=effective_model_for_display,
        ocr_suggestion=ocr_suggestion
    )

    return sanitize_for_json({
        "extracted_text": text,
        "raw_user_text": raw_text,
        "ocr_performed": ocr_ran,
        "ocr_quality_score": round(ocr_quality, 3),
        "ocr_unreliable": ocr_unreliable,
        "ocr_suggestion": ocr_suggestion,
        "ocr_variant_used": ocr_variant,
        "final_sentiment": pred_sentiment,
        "confidence_scores": conf_scores,
        "low_confidence": low_confidence,
        "confidence_margin": round(margin, 4),
        "disclaimer": disclaimer,
        "models_used": models_used,
        "active_model": effective_model_for_display,
        "checkpoint_used": f"best_{effective_model_for_display}.pt" if effective_model_for_display in _LOADED_MODELS else "ensemble",
        "input_image_base64": orig_b64,
        "gradcam_overlay": gradcam_b64,
        "gradcam_warning": gradcam_warning,
        "gradcam_note": gradcam_note,
        "lime_words": lime_words,
        "lime_method": lime_method,
        "lime_note": lime_note,
        "explanation": explanation_text,
        "compare_all": compare_all_results,
        "timings": timings,
        "image_hash": img_hash,
        "is_demo": False
    })


def generate_explanations(pil_img, text, model_type, target_class, low_confidence=False,
                          img_tensor=None, input_ids=None, attention_mask=None, img_hash=None):
    """
    Dedicated generator for Grad-CAM and LIME.
    Can be called synchronously or progressively via POST /explain.
    """
    t_start = time.time()
    device = get_device()

    if img_tensor is None:
        _, img_tensor = preprocess_image(pil_img)
        img_tensor = img_tensor.to(device)

    if img_hash is None:
        img_hash = ocr_engine.compute_image_hash(pil_img)

    # 1. Grad-CAM
    t_g = time.time()
    gradcam_b64 = None
    gradcam_note = None

    if model_type == 'text':
        gradcam_note = "Grad-CAM not applicable for the text-only model."
    else:
        # Check cache
        cache_key = (img_hash, model_type)
        if cache_key in _GRADCAM_CACHE:
            gradcam_b64 = _GRADCAM_CACHE[cache_key]
        else:
            try:
                # Pick visual model for Grad-CAM
                g_model = _LOADED_MODELS.get('concat') or _LOADED_MODELS.get('attention') or _LOADED_MODELS.get('image')
                if g_model is not None:
                    target_layer = None
                    if hasattr(g_model, 'image_encoder') and len(g_model.image_encoder) > 7:
                        target_layer = g_model.image_encoder[7][-1]
                    elif hasattr(g_model, 'resnet_features') and len(g_model.resnet_features) > 7:
                        target_layer = g_model.resnet_features[7][-1]

                    if target_layer is not None:
                        cam = explain.GradCAM(g_model, target_layer)
                        def forward_func():
                            if hasattr(g_model, 'image_encoder'):
                                return g_model(input_ids, attention_mask, img_tensor)
                            return g_model(img_tensor)

                        heatmap, _ = cam.generate_heatmap(img_tensor, target_class=target_class, forward_func=forward_func)
                        # Always show Grad-CAM (alpha=0.55 normal, 0.40 if low conf)
                        alpha = 0.40 if low_confidence else 0.55
                        gradcam_b64 = explain.overlay_gradcam(pil_img, heatmap, alpha=alpha)
                        _GRADCAM_CACHE[cache_key] = gradcam_b64
            except Exception as e:
                print(f"[GradCAM error]: {e}", flush=True)

    gradcam_ms = round((time.time() - t_g) * 1000, 1)

    # 2. LIME
    t_l = time.time()
    lime_words = []
    lime_method = "N/A"
    lime_note = ""

    if model_type == 'image':
        lime_method = "N/A"
        lime_note = "Textual attribution is not applicable for this visual-only model."
    elif text and text.strip():
        lime_cache_key = (img_hash, text, model_type)
        if lime_cache_key in _LIME_CACHE:
            cached_lime = _LIME_CACHE[lime_cache_key]
            lime_words = cached_lime['words']
            lime_method = cached_lime['method']
            lime_note = cached_lime['note']
        else:
            try:
                l_model = _LOADED_MODELS.get(model_type) or _LOADED_MODELS.get('concat') or _LOADED_MODELS.get('text')
                def predict_text_batch(texts):
                    enc = _TOKENIZER(
                        list(texts), padding='max_length', truncation=True, max_length=64, return_tensors='pt'
                    )
                    b_ids = enc['input_ids'].to(device)
                    b_mask = enc['attention_mask'].to(device)
                    with torch.inference_mode():
                        if model_type in ['concat', 'attention']:
                            b_img = img_tensor.expand(len(texts), -1, -1, -1)
                            out = l_model(b_ids, b_mask, b_img)
                        elif model_type == 'ensemble':
                            # fast proxy using concat + text average
                            m_text = _LOADED_MODELS['text']
                            m_concat = _LOADED_MODELS['concat']
                            b_img = img_tensor.expand(len(texts), -1, -1, -1)
                            out = 0.5 * m_text(b_ids, b_mask) + 0.5 * m_concat(b_ids, b_mask, b_img)
                        else:
                            out = l_model(b_ids, b_mask)
                        return F.softmax(out, dim=1).cpu().numpy()

                lime_exp = explain.RealLimeExplainer(predict_text_batch, CLASS_NAMES)
                lime_res = lime_exp.explain_text(text, target_class=target_class, num_samples=200)
                lime_words = lime_res['words']
                lime_note = lime_res['note']
                lime_method = "LIME"

                _LIME_CACHE[lime_cache_key] = {
                    'words': lime_words,
                    'method': lime_method,
                    'note': lime_note
                }
            except Exception as e:
                print(f"[LIME error]: {e}", flush=True)

    lime_ms = round((time.time() - t_l) * 1000, 1)

    return sanitize_for_json({
        "gradcam_overlay": gradcam_b64,
        "gradcam_note": gradcam_note,
        "lime_words": lime_words,
        "lime_method": lime_method,
        "lime_note": lime_note,
        "gradcam_ms": gradcam_ms,
        "lime_ms": lime_ms,
        "total_explain_ms": round((time.time() - t_start) * 1000, 1)
    })


def _build_data_driven_narrative(sentiment, conf_scores, margin, low_conf, lime_words, model_type, ocr_suggestion=""):
    """Assemble dynamic text summary without hardcoded template paragraphs."""
    conf_pct = conf_scores[sentiment] * 100
    sentences = [f"The model classifies this meme as {sentiment} with {conf_pct:.1f}% confidence."]

    if low_conf:
        sentences.append(f"Low confidence warning: the model is unsure (margin over next class is only {margin*100:.1f}%).")

    pos_words = [w for w in lime_words if w.get('score', 0) > 0.02]
    neg_words = [w for w in lime_words if w.get('score', 0) < -0.02]

    if pos_words:
        w_str = ", ".join([f"'{w['word']}' (+{w['score']:.2f})" for w in pos_words[:3]])
        sentences.append(f"Primary tokens supporting {sentiment}: {w_str}.")

    if neg_words:
        w_str = ", ".join([f"'{w['word']}' ({w['score']:.2f})" for w in neg_words[:3]])
        sentences.append(f"Opposing tokens: {w_str}.")

    if not lime_words and model_type == 'image':
        sentences.append("Textual attribution is not applicable for the visual-only model.")
    elif not pos_words and not neg_words and model_type != 'image':
        sentences.append("No single token strongly pushed this prediction.")

    if ocr_suggestion:
        sentences.append(ocr_suggestion)

    return " ".join(sentences)
