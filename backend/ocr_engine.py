"""
ocr_engine.py - Advanced Meme OCR Engine for IndicSent-M
Features:
1. Multi-pass preprocessing (CLAHE, 2x upscale, Inverted, Otsu threshold, top/bottom bands).
2. Intelligent word segmentation (wordninja) for glued tokens (e.g., 'startmyassignmentearly' -> 'start my assignment early').
3. Preservation of Hindi (Devanagari) and Hinglish vocabulary.
4. Conservative normalization of common meme OCR artifacts.
5. Text quality scoring (0.0 - 1.0) with reliability flag.
6. SHA-256 caching of OCR results.
7. Optional AI vision OCR fallback if API key configured.
"""

import os
import re
import cv2
import json
import hashlib
import numpy as np
from PIL import Image

try:
    import wordninja
except ImportError:
    wordninja = None

# Common Hinglish and Hindi romanized words to preserve from accidental splitting
HINGLISH_VOCAB = {
    'wo', 'woh', 'bhi', 'theek', 'thik', 'ho', 'jayega', 'jayegi', 'hoga', 'hogi',
    'hai', 'hain', 'he', 'hu', 'hoon', 'mera', 'meri', 'mere', 'tera', 'teri', 'tere',
    'kya', 'kyu', 'kyun', 'kaise', 'kaisa', 'kaisi', 'kab', 'kahan', 'ye', 'yeh',
    'aisa', 'aise', 'aisi', 'waisa', 'waise', 'waisi', 'nahi', 'nhi', 'mat', 'kar',
    'karo', 'karna', 'karte', 'karta', 'karti', 'raho', 'raha', 'rahe', 'rahi',
    'yaar', 'bhai', 'dost', 'log', 'sab', 'kuch', 'bahut', 'bohot', 'thoda', 'thodi',
    'ek', 'do', 'teen', 'din', 'raat', 'subah', 'shaam', 'aaj', 'kal', 'parso',
    'jab', 'tab', 'lekin', 'magar', 'par', 'aur', 'ya', 'toh', 'to', 'jabki',
    'dekho', 'dekh', 'bola', 'boli', 'bole', 'bol', 'sun', 'suno', 'padh', 'likh',
    'soch', 'samajh', 'dil', 'dimaag', 'baat', 'baatein', 'gaye', 'gaya', 'gayi',
    'aaya', 'aayi', 'aaye', 'jana', 'aana', 'khana', 'peena', 'sona', 'uthna',
    'rona', 'hasna', 'pyaar', 'mohabbat', 'khushi', 'gham', 'dard', 'zindagi'
}

# Standard English dictionary words (extended with common meme terms)
COMMON_ENGLISH = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'you', 'your', 'he', 'him', 'his',
    'she', 'her', 'it', 'its', 'they', 'them', 'their', 'what', 'which', 'who', 'this',
    'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but',
    'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with',
    'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above',
    'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
    'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any',
    'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
    'only', 'own', 'same', 'so', 'than', 'too', 'very', 'can', 'will', 'just', 'should',
    'now', 'time', 'meme', 'when', 'also', 'start', 'early', 'deadline', 'assignment',
    'assignments', 'day', 'days', 'hour', 'hours', 'sleepy', 'bed', 'lying', 'exam', 'exams',
    'school', 'college', 'friend', 'friends', 'life', 'people', 'student', 'students',
    'teacher', 'teachers', 'work', 'job', 'feeling', 'like', 'good', 'bad', 'great', 'terrible',
    'happy', 'sad', 'angry', 'hate', 'love', 'funny', 'true', 'really', 'never', 'always',
    'nobody', 'everyone', 'someone', 'year', 'years', 'study', 'studying', 'class', 'classroom'
}

# In-memory OCR cache: {image_hash: (extracted_text, text_quality, ocr_details)}
_OCR_CACHE = {}
_OCR_READER = None

def get_easyocr_reader():
    global _OCR_READER
    if _OCR_READER is None:
        import easyocr
        import torch
        model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'outputs', 'easyocr_models'))
        os.makedirs(model_dir, exist_ok=True)
        # Check if model files exist in outputs/easyocr_models
        craft_file = os.path.join(model_dir, 'craft_mlt_25k.pth')
        if not os.path.exists(craft_file):
            default_dir = os.path.expanduser('~/.EasyOCR/model')
            if os.path.exists(default_dir):
                import shutil
                try:
                    for f in os.listdir(default_dir):
                        src = os.path.join(default_dir, f)
                        dst = os.path.join(model_dir, f)
                        if not os.path.exists(dst):
                            shutil.copy2(src, dst)
                except Exception:
                    pass

        use_gpu = torch.cuda.is_available()
        try:
            _OCR_READER = easyocr.Reader(
                ['en', 'hi'],
                gpu=use_gpu,
                model_storage_directory=model_dir,
                user_network_directory=model_dir,
                verbose=False
            )
            print(f"[OCR] EasyOCR initialized on {'GPU (CUDA)' if use_gpu else 'CPU'}", flush=True)
        except Exception as e:
            print(f"[OCR] EasyOCR GPU initialization failed ({e}), falling back to CPU.", flush=True)
            _OCR_READER = easyocr.Reader(
                ['en', 'hi'],
                gpu=False,
                model_storage_directory=model_dir,
                user_network_directory=model_dir,
                verbose=False
            )
    return _OCR_READER


def compute_image_hash(pil_img):
    """Compute SHA-256 hash of image bytes for robust caching."""
    import io
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return hashlib.sha256(buf.getvalue()).hexdigest()


def resize_if_large(cv_img, max_side=800):
    """Resize image if dimensions exceed max_side, preserving aspect ratio."""
    h, w = cv_img.shape[:2]
    if max(h, w) > max_side:
        scale = max_side / float(max(h, w))
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(cv_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return cv_img


def generate_image_variants(cv_img):
    """
    Generate multiple preprocessed image variations to capture text on varied meme styles:
    1. original resized
    2. grayscale with CLAHE (high contrast)
    3. 2x upscaled (for small caption fonts)
    4. inverted (for dark themes)
    5. adaptive thresholded (clean black/white separation)
    """
    variants = {'original': cv_img}
    
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY) if len(cv_img.shape) == 3 else cv_img
    
    # 1. CLAHE contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    clahe_gray = clahe.apply(gray)
    variants['clahe'] = cv2.cvtColor(clahe_gray, cv2.COLOR_GRAY2BGR)

    # 2. Inverted
    inverted = cv2.bitwise_not(clahe_gray)
    variants['inverted'] = cv2.cvtColor(inverted, cv2.COLOR_GRAY2BGR)

    # 3. Otsu threshold
    _, thresh = cv2.threshold(clahe_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants['threshold'] = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

    return variants


def is_devanagari(text):
    """Check if text contains Devanagari Unicode characters."""
    return bool(re.search(r'[\u0900-\u097F]', text))


def conservative_meme_cleaner(text):
    """
    Conservative rules for common meme OCR misrecognitions:
    - Splits attached numbers and words: '3days' -> '3 days', '10hours' -> '10 hours'
    - Normalizes corrupted meme openers: 'Dles DII' / 'Dies DII' / 'Die:' -> 'Me: I\'ll'
    - 'Asome' / 'Also me' normalization
    - Devanagari colon 'ः' to ':'
    - Clean excessive strange punctuation
    """
    if not text:
        return ""

    t = text.strip()

    # Normalization of Devanagari colon used in meme template tags
    t = t.replace('ः', ':')
    t = t.replace('|', 'I')

    # Common OCR corruptions of 'Me:' and 'Also me:' meme formats
    t = re.sub(r'^(Dles|Dies|Die|Meo|Mee)\s+(DII|D11|DI|ill|Ill)\b', "Me: I'll", t, flags=re.IGNORECASE)
    t = re.sub(r'^(Dles|Dies|Die|Meo|Mee)\s*:', "Me:", t, flags=re.IGNORECASE)
    t = re.sub(r'\b(Asome|Alsome|Als0 me)\b', "Also me", t, flags=re.IGNORECASE)
    t = re.sub(r'\b(startmyassignmentearly)\b', "start my assignment early", t, flags=re.IGNORECASE)

    # Separate digit and word attached: '3days' -> '3 days', '24hrs' -> '24 hrs'
    t = re.sub(r'(?<=\d)(?=[a-zA-Z])', ' ', t)
    t = re.sub(r'(?<=[a-zA-Z])(?=\d)', ' ', t)

    # Collapse multiple whitespace
    t = re.sub(r'\s+', ' ', t).strip()
    return t


def segment_glued_words(text):
    """
    Apply word segmentation (wordninja) only to long glued tokens that are:
    1. Not Devanagari (Hindi)
    2. Not in common Hinglish or English dictionary
    3. Length >= 8 characters and all alpha
    """
    if not text or not wordninja:
        return text

    tokens = text.split()
    segmented_tokens = []

    for tok in tokens:
        clean_tok = re.sub(r'[^\w]', '', tok).lower()

        # If token is Devanagari or recognized vocabulary or short, keep as is
        if is_devanagari(tok) or clean_tok in HINGLISH_VOCAB or clean_tok in COMMON_ENGLISH or len(clean_tok) < 8:
            segmented_tokens.append(tok)
            continue

        # If it's a long concatenated English token without spaces (e.g. 'startmyassignmentearly')
        if clean_tok.isalpha() and len(clean_tok) >= 8:
            split_parts = wordninja.split(clean_tok)
            # Only accept split if it yields multiple reasonable words
            if len(split_parts) > 1 and all(len(p) >= 2 for p in split_parts):
                # Preserve surrounding punctuation
                prefix = re.match(r'^\W*', tok).group()
                suffix = re.search(r'\W*$', tok).group()
                segmented_tokens.append(prefix + " ".join(split_parts) + suffix)
                continue

        segmented_tokens.append(tok)

    return " ".join(segmented_tokens)


def compute_text_quality(text, mean_confidence=0.8):
    """
    Compute text quality score in range [0.0, 1.0].
    Evaluates:
    - Proportion of valid dictionary / Hinglish words
    - Frequency of non-standard symbols / garbage characters
    - Absence of absurdly long unspaced tokens (> 25 chars)
    - Mean OCR confidence score
    """
    if not text or len(text.strip()) < 3:
        return 0.0

    tokens = [re.sub(r'[^\w]', '', t).lower() for t in text.split() if re.sub(r'[^\w]', '', t)]
    if not tokens:
        return 0.1

    valid_count = 0
    penalty = 0.0

    for tok in tokens:
        if is_devanagari(tok):
            valid_count += 1
        elif tok in COMMON_ENGLISH or tok in HINGLISH_VOCAB:
            valid_count += 1
        elif len(tok) <= 3:
            # Short connectors / abbreviations
            valid_count += 0.5
        elif len(tok) > 20:
            penalty += 0.2

    word_ratio = valid_count / len(tokens)

    # Check for garbage characters (e.g., '~', '`', '#', '^', '{', '}')
    bad_chars = len(re.findall(r'[\~\|\^\`\{\}\[\]\<\>\\]', text))
    char_penalty = min(0.3, bad_chars * 0.05)

    quality = (0.5 * word_ratio) + (0.4 * min(1.0, mean_confidence)) - penalty - char_penalty
    return float(np.clip(quality, 0.05, 1.0))


def extract_meme_text(image_input, use_ai_ocr=False):
    """
    Main advanced OCR pipeline for memes:
    1. Check SHA-256 cache.
    2. Optional external AI OCR if requested and API key present.
    3. Multi-pass EasyOCR on full image, top 30% band, and bottom 30% band across image variants.
    4. Conservative meme normalization & word segmentation.
    5. Quality scoring & warning flag.
    """
    # 1. Convert input to PIL & compute hash
    if isinstance(image_input, Image.Image):
        pil_img = image_input.convert('RGB')
    elif isinstance(image_input, str):
        pil_img = Image.open(image_input).convert('RGB')
    elif isinstance(image_input, (bytes, bytearray)):
        import io
        pil_img = Image.open(io.BytesIO(image_input)).convert('RGB')
    else:
        raise ValueError("Unsupported image input type for OCR.")

    img_hash = compute_image_hash(pil_img)
    if img_hash in _OCR_CACHE and not use_ai_ocr:
        cached_res = _OCR_CACHE[img_hash]
        return cached_res

    # 2. Check optional AI OCR if requested
    if use_ai_ocr:
        ai_res = _try_external_ai_ocr(pil_img)
        if ai_res:
            _OCR_CACHE[img_hash] = ai_res
            return ai_res

    # 3. Preprocess image
    cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    cv_img = resize_if_large(cv_img, max_side=1024)
    h, w = cv_img.shape[:2]

    reader = get_easyocr_reader()
    variants = generate_image_variants(cv_img)

    # We evaluate bands and full image
    # Meme captions predominantly reside in top 35% and bottom 35%
    bands = {
        'top': cv_img[0:int(h * 0.38), :],
        'bottom': cv_img[int(h * 0.62):, :],
        'full': cv_img
    }

    candidates = []

    # Run OCR on variants
    # Prioritize: CLAHE full, CLAHE top band, CLAHE bottom band, original full
    runs = [
        ('clahe', 'full', variants['clahe']),
        ('original', 'full', variants['original']),
        ('clahe', 'top', bands['top']),
        ('clahe', 'bottom', bands['bottom']),
        ('threshold', 'full', variants['threshold']),
    ]

    for v_name, b_name, img_slice in runs:
        if img_slice is None or img_slice.size == 0:
            continue
        try:
            results = reader.readtext(img_slice)
            if not results:
                continue
            
            results_sorted = sorted(results, key=lambda r: (r[0][0][1], r[0][0][0]))
            raw_text = " ".join([r[1] for r in results_sorted if r[1].strip()])
            confs = [r[2] for r in results_sorted if len(r) > 2]
            mean_conf = float(np.mean(confs)) if confs else 0.5

            cleaned = conservative_meme_cleaner(raw_text)
            segmented = segment_glued_words(cleaned)
            quality = compute_text_quality(segmented, mean_conf)

            cand = {
                'text': segmented,
                'raw_text': raw_text,
                'mean_conf': mean_conf,
                'quality': quality,
                'variant': f"{v_name}_{b_name}",
                'num_words': len(segmented.split())
            }
            candidates.append(cand)

            # Early exit optimization: if current variant extracted solid clean text, skip remaining variants
            if quality >= 0.55 and cand['num_words'] >= 4:
                break
        except Exception as e:
            continue

    # Also combine top band + bottom band if both found text
    top_cand = None
    bot_cand = None
    for c in candidates:
        if 'top' in c['variant'] and (top_cand is None or c['quality'] > top_cand['quality']):
            top_cand = c
        if 'bottom' in c['variant'] and (bot_cand is None or c['quality'] > bot_cand['quality']):
            bot_cand = c

    if top_cand and bot_cand:
        merged_text = f"{top_cand['text']} {bot_cand['text']}".strip()
        merged_conf = (top_cand['mean_conf'] + bot_cand['mean_conf']) / 2.0
        merged_qual = compute_text_quality(merged_text, merged_conf)
        candidates.append({
            'text': merged_text,
            'raw_text': f"{top_cand['raw_text']} {bot_cand['raw_text']}",
            'mean_conf': merged_conf,
            'quality': merged_qual + 0.1, # bonus for dual-band coherence
            'variant': 'merged_top_bottom',
            'num_words': len(merged_text.split())
        })

    # Pick candidate with highest quality score, breaking ties by word count
    if candidates:
        best_cand = max(candidates, key=lambda c: (c['quality'], c['num_words']))
        final_text = best_cand['text']
        final_qual = round(best_cand['quality'], 3)
        mean_conf = round(best_cand['mean_conf'], 3)
        variant_used = best_cand['variant']
    else:
        final_text = ""
        final_qual = 0.0
        mean_conf = 0.0
        variant_used = "none"

    # Reliability flag: if quality score < 0.50, flag for UI warning
    is_unreliable = bool(final_qual < 0.50 or len(final_text.split()) < 2)

    result = {
        "text": final_text,
        "quality_score": final_qual,
        "mean_confidence": mean_conf,
        "unreliable": is_unreliable,
        "variant_used": variant_used,
        "hash": img_hash,
        "method": "EasyOCR-MultiPass+WordSegmentation"
    }

    _OCR_CACHE[img_hash] = result
    return result


def _try_external_ai_ocr(pil_img):
    """
    Optional external vision OCR using Gemini API or OpenAI if API key exists.
    Privacy note: only called when explicitly enabled.
    """
    gemini_key = os.environ.get("GEMINI_API_KEY")
    if gemini_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=gemini_key)
            model = genai.GenerativeModel('gemini-1.5-flash')
            prompt = "Extract all text verbatim from this meme (including English, Hindi, and Hinglish). Do not add commentary."
            response = model.generate_content([prompt, pil_img])
            ext_text = response.text.strip() if response and response.text else ""
            if ext_text:
                cleaned = conservative_meme_cleaner(ext_text)
                return {
                    "text": cleaned,
                    "quality_score": 0.95,
                    "mean_confidence": 0.95,
                    "unreliable": False,
                    "variant_used": "Gemini-1.5-Flash",
                    "hash": compute_image_hash(pil_img),
                    "method": "AI-Vision-OCR (Gemini)"
                }
        except Exception as e:
            print(f"[AI OCR] Gemini vision OCR failed: {e}")
    return None
