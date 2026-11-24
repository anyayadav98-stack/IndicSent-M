"""
app.py - Production FastAPI Microservice for IndicSent-M
Provides endpoints:
- POST /predict: Fast progressive sentiment inference with timing profiler & OCR quality scoring
- POST /explain: Asynchronous/progressive generator for Grad-CAM overlay & fast batched LIME
- POST /predict-json: JSON payload alternative for base64 requests
- GET /model-info: Official validation benchmarks, test metrics, and active model descriptions
- GET /metrics: Real benchmark evaluation metrics from comparison.json
- GET /dataset-stats: Memotion 3.0 dataset distributions and split details
- GET /history: Live query history
- GET /health: GPU hardware and model status
"""

import os
os.environ.setdefault('MPLCONFIGDIR', os.path.join(os.getcwd(), 'outputs'))
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

import sys
import io
import json
import base64
from typing import Optional
from contextlib import asynccontextmanager

_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image

import inference

@asynccontextmanager
async def lifespan(app: FastAPI):
    inference.load_all_models()
    yield

app = FastAPI(
    title="IndicSent-M API",
    description="Multimodal Meme Sentiment Analysis for Indian Languages (XLM-R, ResNet50, Concat, Cross-Attention)",
    version="2.1.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PREDICTION_HISTORY = []

class PredictJSONRequest(BaseModel):
    image_base64: str
    text: Optional[str] = None
    model_type: Optional[str] = "auto"
    use_ai_ocr: Optional[bool] = False
    include_explanations: Optional[bool] = True

class ExplainRequest(BaseModel):
    image_base64: str
    text: Optional[str] = ""
    model_type: Optional[str] = "auto"
    target_class: Optional[int] = 1
    low_confidence: Optional[bool] = False
    image_hash: Optional[str] = None

@app.get("/health")
def health():
    import torch
    cuda_avail = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_avail else "CPU"
    loaded = list(inference._LOADED_MODELS.keys())
    return {
        "status": "healthy",
        "service": "IndicSent-M Multimodal Sentiment Engine",
        "gpu_available": cuda_avail,
        "device_name": device_name,
        "loaded_models": loaded,
        "default_model": inference._BEST_VALIDATION_MODEL,
        "is_demo_mode": False
    }

@app.get("/model-info")
def model_info():
    """
    Returns empirical validation metrics and descriptions for UI tooltips.
    """
    return {
        "auto_default": inference._BEST_VALIDATION_MODEL,
        "auto_tooltip": f"Selects the architecture with the highest validation Macro F1 score on Memotion 3 — currently {inference._BEST_VALIDATION_MODEL.capitalize()} Fusion.",
        "models": {
            "auto": {
                "label": "Auto (Best by Validation)",
                "description": f"Automatically routes to the highest-performing architecture on validation data ({inference._BEST_VALIDATION_MODEL.upper()})."
            },
            "concat": {
                "label": "Concat Fusion (Early)",
                "description": "Concatenates XLM-R 768d text embedding with ResNet50 2048d visual feature vector into a joint MLP.",
                "val_macro_f1": 34.59,
                "test_accuracy": 33.47
            },
            "attention": {
                "label": "Cross-Attention Fusion",
                "description": "Multi-head attention mechanism where text tokens dynamically attend to 49 spatial visual patches from ResNet50 layer4.",
                "val_macro_f1": 33.01,
                "test_accuracy": 33.33
            },
            "text": {
                "label": "Text-Only (XLM-RoBERTa)",
                "description": "Analyzes code-mixed Hinglish caption tokens using multilingual XLM-R without visual context.",
                "val_macro_f1": 34.08,
                "test_accuracy": 37.38
            },
            "image": {
                "label": "Image-Only (ResNet50)",
                "description": "Analyzes visual expressions, characters, and objects using deep ResNet50 without text.",
                "val_macro_f1": 34.42,
                "test_accuracy": 31.98
            },
            "ensemble": {
                "label": "Multimodal Ensemble",
                "description": "Averages predicted class probability distributions across all 4 trained architectures.",
                "val_macro_f1": 34.80,
                "test_accuracy": 35.80
            },
            "compare_all": {
                "label": "Compare All Architectures",
                "description": "Evaluates all four architectures + ensemble side-by-side with sentiment labels and probabilities."
            }
        }
    }

@app.post("/predict")
async def predict_endpoint(
    file: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None),
    model_type: Optional[str] = Form("auto"),
    use_ai_ocr: Optional[bool] = Form(False),
    include_explanations: Optional[bool] = Form(True)
):
    try:
        if file is not None:
            contents = await file.read()
            pil_img = Image.open(io.BytesIO(contents)).convert('RGB')
        else:
            raise HTTPException(status_code=400, detail="Meme image file is required for prediction.")

        result = inference.predict(
            image_input=pil_img,
            text_input=text,
            model_type=model_type or "auto",
            use_ai_ocr=bool(use_ai_ocr),
            include_explanations=bool(include_explanations)
        )

        # Store in history
        history_entry = {
            "id": f"pred-{len(PREDICTION_HISTORY)+1}",
            "extracted_text": result["extracted_text"],
            "final_sentiment": result["final_sentiment"],
            "confidence_scores": result["confidence_scores"],
            "active_model": result["active_model"],
            "input_image_base64": result.get("input_image_base64", "")
        }
        PREDICTION_HISTORY.insert(0, history_entry)
        if len(PREDICTION_HISTORY) > 10:
            PREDICTION_HISTORY.pop()

        clean_res = inference.sanitize_for_json(result)
        return clean_res
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

@app.post("/explain")
def explain_endpoint(payload: ExplainRequest):
    """
    Progressive explanation generator for Grad-CAM overlay and LIME attributions.
    """
    try:
        header, encoded = payload.image_base64.split(",", 1) if "," in payload.image_base64 else ("", payload.image_base64)
        img_bytes = base64.b64decode(encoded)
        pil_img = Image.open(io.BytesIO(img_bytes)).convert('RGB')

        res = inference.generate_explanations(
            pil_img=pil_img,
            text=payload.text or "",
            model_type=payload.model_type or "auto",
            target_class=payload.target_class or 1,
            low_confidence=bool(payload.low_confidence),
            img_hash=payload.image_hash
        )
        return inference.sanitize_for_json(res)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explanation generation error: {str(e)}")

@app.post("/predict-json")
def predict_json_endpoint(payload: PredictJSONRequest):
    try:
        header, encoded = payload.image_base64.split(",", 1) if "," in payload.image_base64 else ("", payload.image_base64)
        img_bytes = base64.b64decode(encoded)
        pil_img = Image.open(io.BytesIO(img_bytes)).convert('RGB')

        result = inference.predict(
            image_input=pil_img,
            text_input=payload.text,
            model_type=payload.model_type or "auto",
            use_ai_ocr=bool(payload.use_ai_ocr),
            include_explanations=bool(payload.include_explanations)
        )
        clean_res = inference.sanitize_for_json(result)
        return clean_res
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"JSON Inference error: {str(e)}")

@app.get("/metrics")
def get_metrics():
    """
    Returns complete, honest benchmark metrics, 95% CIs, and significance test results.
    """
    final_path = os.path.join('outputs', 'final_metrics.json')
    sig_path = os.path.join('outputs', 'significance_tests.json')

    if os.path.exists(final_path):
        with open(final_path, 'r') as f:
            data = json.load(f)

        sig_data = {}
        if os.path.exists(sig_path):
            with open(sig_path, 'r') as f:
                sig_data = json.load(f)

        # Merge top-level model keys for backward compatibility
        res = {
            "test_samples": data.get("test_samples", 741),
            "official_metric": data.get("official_metric", "Weighted F1"),
            "models": data.get("models", {}),
            "significance_tests": sig_data,
            "literature_reference": data.get("literature_reference", {}),
            "auto_default": inference._BEST_VALIDATION_MODEL,
            "majority_baseline": {
                "accuracy": 0.3873,
                "macro_f1": 0.1861,
                "weighted_f1": 0.2163,
                "majority_class": "Negative"
            }
        }
        for k, v in data.get("models", {}).items():
            res[k] = v
        return res

    comp_path = os.path.join('outputs', 'comparison.json')
    if os.path.exists(comp_path):
        with open(comp_path, 'r') as f:
            data = json.load(f)
        return data

    models = ['text', 'image', 'concat', 'attention']
    combined = {}
    for m in models:
        m_path = os.path.join('outputs', f'metrics_{m}.json')
        if os.path.exists(m_path):
            with open(m_path, 'r') as f:
                combined[m] = json.load(f)
    return combined

@app.get("/error-analysis")
def get_error_analysis():
    """
    Returns the 20-sample detailed error analysis for the Compare page.
    """
    ea_path = os.path.join('outputs', 'error_analysis.json')
    if os.path.exists(ea_path):
        with open(ea_path, 'r') as f:
            return json.load(f)
    return []

@app.get("/dataset-stats")
def dataset_stats():
    import pandas as pd
    processed_dir = os.path.join('data', 'processed')
    train_path = os.path.join(processed_dir, 'train.csv')
    val_path = os.path.join(processed_dir, 'val.csv')
    test_path = os.path.join(processed_dir, 'test.csv')

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    def get_dist(df):
        vc = df['sentiment'].value_counts()
        return {
            "Positive": int(vc.get("Positive", 0)),
            "Neutral": int(vc.get("Neutral", 0)),
            "Negative": int(vc.get("Negative", 0))
        }

    return {
        "dataset_name": "Memotion 3.0 (Hinglish Multimodal Sentiment)",
        "total_labelled": len(train_df) + len(val_df) + len(test_df),
        "splits": {
            "train": {"total": len(train_df), "distribution": get_dist(train_df)},
            "validation": {"total": len(val_df), "distribution": get_dist(val_df)},
            "test": {"total": len(test_df), "distribution": get_dist(test_df)},
            "blind_test": {"total": 1484}
        },
        "text_stats": {
            "avg_word_count": 14.8,
            "max_word_count": 68,
            "languages": ["Hinglish (Roman Hindi)", "Devanagari Hindi", "English"]
        }
    }

@app.get("/history")
def history():
    return PREDICTION_HISTORY

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
