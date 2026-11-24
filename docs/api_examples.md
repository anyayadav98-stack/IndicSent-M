# IndicSent-M: FastAPI REST API Documentation & Examples

Base URL: `http://localhost:8000`

---

## 1. Health Check
- **Endpoint**: `GET /health`
- **Description**: Returns server status, active model, and GPU acceleration info.
- **Example Request**:
  ```bash
  curl -X GET http://localhost:8000/health
  ```
- **Example Response**:
  ```json
  {
    "status": "healthy",
    "service": "IndicSent-M Multimodal Sentiment Engine",
    "gpu_available": true,
    "device_name": "NVIDIA GeForce RTX 2050",
    "active_model": "concat",
    "is_demo_mode": false
  }
  ```

---

## 2. Multimodal Sentiment Analysis
- **Endpoint**: `POST /predict`
- **Content-Type**: `multipart/form-data`
- **Parameters**:
  - `file`: (binary) Meme image file (JPEG / PNG).
  - `text`: (string, optional) Extracted or user-provided meme caption.
- **Example Request (cURL)**:
  ```bash
  curl -X POST http://localhost:8000/predict \
    -F "file=@data/memotion3/valImages/0.jpg" \
    -F "text=Tum paas aaye yun muskuraye BETA PHIR BHI AAP HUMKO BILKUL PASAND NAHI AAYE!"
  ```
- **Example Response**:
  ```json
  {
    "extracted_text": "Tum paas aaye yun muskuraye BETA PHIR BHI AAP HUMKO BILKUL PASAND NAHI AAYE!",
    "final_sentiment": "Negative",
    "confidence_scores": {
      "Negative": 0.6840,
      "Neutral": 0.2150,
      "Positive": 0.1010
    },
    "lime_words": [
      {"word": "BILKUL", "score": 0.18},
      {"word": "NAHI", "score": 0.15},
      {"word": "PASAND", "score": 0.12},
      {"word": "muskuraye", "score": -0.06}
    ],
    "gradcam_overlay": "data:image/jpeg;base64,...",
    "explanation": "The model predicts a Negative sentiment with 68.4% confidence because the caption expresses dissatisfaction, frustration, or critical commentary (cues: BILKUL, NAHI, PASAND). The Grad-CAM image influence map shows salient visual areas indicate high intensity or contrasting context.",
    "models_used": ["XLM-R", "ResNet50"],
    "unimodal_comparison": {
      "text_only": "Negative",
      "image_only": "Neutral"
    },
    "is_demo": false
  }
  ```

---

## 3. Benchmark Metrics
- **Endpoint**: `GET /metrics`
- **Description**: Returns precision, recall, F1-scores, accuracy, and confusion matrices across Text-Only, Image-Only, Concat Fusion, and Cross-Attention Fusion models.
- **Example Request**:
  ```bash
  curl -X GET http://localhost:8000/metrics
  ```

---

## 4. Dataset Statistics
- **Endpoint**: `GET /dataset-stats`
- **Description**: Retrieves Memotion 3 class distributions and split metrics.
- **Example Request**:
  ```bash
  curl -X GET http://localhost:8000/dataset-stats
  ```

---

## 5. Recent Prediction History
- **Endpoint**: `GET /history`
- **Description**: Returns recent queries analyzed through the dashboard.
- **Example Request**:
  ```bash
  curl -X GET http://localhost:8000/history
  ```

