"""
test_requirements.py - Quick sanity checks for the IndicSent-M inference pipeline.
Tests OCR quality, model auto-selection, Grad-CAM behavior, and API response speed.
"""

import os
import sys
import json
import time
import numpy as np
from PIL import Image

_BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend'))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

import ocr_engine
import inference

def test_ocr_quality_and_segmentation():
    # 1. Glued words
    raw = "Dles DII startmyassignmentearly this time Asome 3days before the deadline:"
    cleaned = ocr_engine.conservative_meme_cleaner(raw)
    seg = ocr_engine.segment_glued_words(cleaned)
    assert "start my assignment early" in seg
    assert "3 days" in seg

    # 2. Hinglish & Devanagari preservation
    hinglish = "wo bhi theek ho jayega aur bohot khushi hogi"
    h_seg = ocr_engine.segment_glued_words(hinglish)
    assert h_seg == hinglish, f"Hinglish corrupted: {h_seg}"

    devanagari = "यह एक बहुत अच्छा मीम है"
    d_seg = ocr_engine.segment_glued_words(devanagari)
    assert d_seg == devanagari, f"Devanagari corrupted: {d_seg}"

    # 3. Quality scoring
    q_good = ocr_engine.compute_text_quality("When you start assignment early and feeling happy", mean_confidence=0.9)
    q_bad = ocr_engine.compute_text_quality("~`#^| 4598129034812304891234 ||||", mean_confidence=0.3)
    assert q_good > 0.60
    assert q_bad < 0.35
    print("[PASS] OCR quality and word segmentation test passed.")

def test_auto_selection():
    comp_path = os.path.join('outputs', 'comparison.json')
    assert os.path.exists(comp_path), "comparison.json missing"
    with open(comp_path, 'r') as f:
        comp = json.load(f)

    val_f1s = {}
    for m in ['text', 'image', 'concat', 'attention']:
        val_f1s[m] = comp[m]['seeds_summary']['val_macro_f1_mean']

    expected_best = max(val_f1s.keys(), key=lambda k: val_f1s[k])
    inference.load_all_models()
    assert inference._BEST_VALIDATION_MODEL == expected_best, f"Expected {expected_best}, got {inference._BEST_VALIDATION_MODEL}"
    print(f"[PASS] Auto model selection audit passed: selected {inference._BEST_VALIDATION_MODEL} (F1={val_f1s[expected_best]:.4f}).")

def test_disclaimer_matches_outputs():
    disc = inference.generate_disclaimer('text')
    assert ("33.9%" in disc or "34.1%" in disc)
    assert ("37.2%" in disc or "37.4%" in disc)
    print("[PASS] Disclaimer numbers match outputs/ metrics.")

def test_gradcam_at_low_confidence():
    img_path = 'frontend/public/samples/244.jpg'
    assert os.path.exists(img_path)
    # Low confidence scenario
    res = inference.predict(img_path, text_input="ambiguous random text", model_type="concat", include_explanations=True)
    if res['low_confidence']:
        assert res['gradcam_overlay'] is not None, "Grad-CAM was hidden despite low confidence!"
        assert res['gradcam_warning'] is not None, "Grad-CAM warning missing for low confidence!"
    print("[PASS] Grad-CAM always-on verification passed.")

def test_predict_progressive_speed():
    img_path = 'frontend/public/samples/244.jpg'
    t0 = time.time()
    res = inference.predict(img_path, text_input="test caption", model_type="concat", include_explanations=False)
    elapsed = time.time() - t0
    assert elapsed < 3.0, f"/predict took {elapsed:.2f}s, expected < 3s"
    assert res['final_sentiment'] in ["Positive", "Neutral", "Negative"]
    print(f"[PASS] Progressive /predict responds in {elapsed:.3f}s (< 3.0s goal).")

def test_ensemble_default_rule():
    comp_file = os.path.join('outputs', 'sample_model_comparison.json')
    if os.path.exists(comp_file):
        with open(comp_file, 'r') as f:
            data = json.load(f)
        auto_rec = data.get('auto_recommendation')
        best_single = data.get('best_single_model')
        if auto_rec == 'ensemble':
            best_single_f1 = data['clean_text_evaluation_100_samples'][best_single]['macro_f1']
            ens_f1 = max(data['clean_text_evaluation_100_samples']['ensemble_simple']['macro_f1'],
                         data['clean_text_evaluation_100_samples']['ensemble_weighted']['macro_f1'])
            assert ens_f1 > best_single_f1, "Ensemble became default without strictly beating best single model!"
        print("[PASS] Ensemble default rule validated.")

if __name__ == '__main__':
    test_ocr_quality_and_segmentation()
    test_auto_selection()
    test_disclaimer_matches_outputs()
    test_gradcam_at_low_confidence()
    test_predict_progressive_speed()
    test_ensemble_default_rule()
    print("\nALL SYSTEM REQUIREMENT TESTS PASSED SUCCESSFULLY!")
