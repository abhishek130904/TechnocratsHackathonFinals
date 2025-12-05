# System Architecture

This document explains the overall architecture of the Counterfeit Product Detection system.

## Overview

The system is built as a **three-tier architecture**:

1. **Frontend Layer**: React application for user interaction
2. **Backend Layer**: FastAPI REST API for request handling
3. **ML Layer**: Machine learning models for prediction

```
┌─────────────┐
│   Frontend  │  React + TypeScript
│  (React)    │  Port: 3000
└──────┬──────┘
       │ HTTP/REST
       │
┌──────▼──────┐
│   Backend   │  FastAPI
│  (Python)   │  Port: 8000
└──────┬──────┘
       │
       ├──────────┬──────────┬──────────┐
       │          │          │          │
┌──────▼──┐ ┌─────▼───┐ ┌────▼────┐ ┌──▼──────┐
│  Image  │ │  Text   │ │ Fusion  │ │Preprocess│
│  Model  │ │  Model  │ │  Model  │ │  Utils   │
│ (CNN)   │ │(BERT)   │ │ (Dense) │ │          │
└─────────┘ └─────────┘ └─────────┘ └──────────┘
```

## Component Details

### 1. Frontend (React)

**Technology Stack:**
- React 18 with TypeScript
- Vite for build tooling
- Axios for HTTP requests

**Key Components:**
- `App.tsx`: Main application component
- `PredictionForm.tsx`: Form for input collection
- `PredictionResult.tsx`: Results display

**Responsibilities:**
- User interface rendering
- Form validation
- API communication
- Result visualization

### 2. Backend (FastAPI)

**Technology Stack:**
- FastAPI for REST API
- Uvicorn as ASGI server
- Pydantic for data validation

**Key Modules:**

#### `app/main.py`
- FastAPI application initialization
- CORS configuration
- Route registration
- Startup/shutdown events

#### `app/routes/predict.py`
- `/api/predict` endpoint handler
- Request validation
- Response formatting
- Error handling

#### `app/services/inference_service.py`
- Orchestrates ML model calls
- Feature extraction coordination
- Score calculation
- Explanation generation

#### `app/ml/` (Model Interfaces)
- `image_model.py`: CNN model wrapper
- `text_model.py`: Transformer model wrapper
- `fusion_model.py`: Multimodal fusion model
- `preprocessing.py`: Image preprocessing utilities

### 3. Machine Learning Pipeline

#### Image Model (CNN)

**Architecture:**
- Base: EfficientNetB0 (pretrained on ImageNet)
- Transfer learning approach
- Custom classification head:
  - GlobalAveragePooling2D
  - Dropout (0.2)
  - Dense(1, sigmoid) for binary classification

**Input:** 224x224x3 RGB images
**Output:** Authenticity score (0.0 = fake, 1.0 = real)

**Training:**
- Binary classification (fake vs real)
- Data augmentation (rotation, flip, brightness)
- Transfer learning with frozen base initially

#### Text Model (NLP)

**Architecture:**
- Base: DistilBERT (pretrained)
- Feature extraction: [CLS] token embedding
- Embedding dimension: 768

**Input:** Product title + description (max 512 tokens)
**Output:** 768-dimensional feature vector

**Usage:**
- Extract embeddings for fusion
- Optional: Simple keyword-based scoring

#### Fusion Model

**Architecture:**
- Multi-input model:
  - Image features (1280 dims) → Dense(256) → Dropout(0.3)
  - Text features (768 dims) → Dense(256) → Dropout(0.3)
  - Metadata (2 dims) → Dense(32) → Dropout(0.2)
- Concatenation layer
- Final layers:
  - Dense(128) → Dropout(0.4)
  - Dense(64) → Dropout(0.3)
  - Dense(1, sigmoid)

**Input:** Combined features from all modalities
**Output:** Final authenticity score (0.0-1.0)

## Data Flow

### Prediction Request Flow

```
1. User uploads image + enters text
   ↓
2. Frontend sends POST /api/predict
   ↓
3. Backend receives request
   ↓
4. Preprocess image (resize, normalize)
   ↓
5. Extract image features (CNN)
   ↓
6. Extract text features (DistilBERT)
   ↓
7. Prepare metadata features
   ↓
8. Combine features (Fusion Model)
   ↓
9. Generate explanations
   ↓
10. Return response to frontend
```

### Training Flow

```
1. Load dataset (train/val splits)
   ↓
2. Data augmentation (training only)
   ↓
3. Train image model (EfficientNetB0)
   ↓
4. Extract features from trained model
   ↓
5. Train fusion model (multimodal)
   ↓
6. Save models to disk
   ↓
7. Load models in backend at startup
```

## Decision Logic

The system uses **threshold-based decisions**:

```python
if score >= 0.75:
    decision = "approve"  # Likely authentic
elif score >= 0.50:
    decision = "flag"     # Suspicious, needs review
else:
    decision = "reject"   # Likely counterfeit
```

These thresholds are configurable in `backend/app/config.py`.

## Explainability

### Grad-CAM Visualization

The system generates heatmaps showing which image regions the model focuses on:

1. Extract feature maps from last convolutional layer
2. Compute gradients of prediction w.r.t. feature maps
3. Weight feature maps by gradient importance
4. Generate heatmap overlay

### Text Explanations

- Keyword detection (suspicious terms)
- Pattern analysis
- Professional language assessment

### Metadata Explanations

- Seller rating analysis
- Review count assessment
- Historical pattern matching

## Scalability Considerations

### Current Design (Single Server)

- All models loaded in memory
- Synchronous request handling
- Suitable for moderate traffic

### Future Enhancements

1. **Model Serving**: Separate model serving layer (TensorFlow Serving, TorchServe)
2. **Async Processing**: Celery + Redis for background tasks
3. **Caching**: Redis for frequently accessed predictions
4. **Database**: MongoDB/PostgreSQL for prediction history
5. **Load Balancing**: Multiple backend instances

## Security Considerations

1. **File Upload Validation**:
   - File type checking
   - File size limits
   - Malware scanning (future)

2. **API Security**:
   - Rate limiting (future)
   - Authentication (future)
   - Input sanitization

3. **Model Security**:
   - Adversarial attack protection (future)
   - Model versioning
   - A/B testing framework (future)

## Performance Metrics

### Expected Latency

- Image preprocessing: ~50ms
- Image model inference: ~100-200ms
- Text embedding: ~50-100ms
- Fusion inference: ~10-20ms
- **Total**: ~200-400ms per prediction

### Optimization Opportunities

1. Model quantization (TFLite, ONNX)
2. Batch processing
3. GPU acceleration
4. Model pruning

## Deployment Architecture

### Development

```
Frontend (localhost:3000) → Backend (localhost:8000) → Models (local files)
```

### Production (Recommended)

```
┌─────────────┐
│   Nginx     │  Reverse proxy + SSL
└──────┬──────┘
       │
┌──────▼──────┐
│   FastAPI   │  Multiple workers (Gunicorn/Uvicorn)
│   Backend   │
└──────┬──────┘
       │
┌──────▼──────┐
│   Models    │  Model files or serving layer
└─────────────┘
```

## Monitoring & Logging

### Logging

- Application logs: `logs/app.log`
- Request/response logging
- Error tracking
- Model performance metrics

### Metrics to Track

- Prediction latency
- Model accuracy (if feedback available)
- API usage statistics
- Error rates

## Future Enhancements

1. **Active Learning**: Use feedback to improve models
2. **Multi-class Classification**: Distinguish product categories
3. **Real-time Training**: Continuous model updates
4. **Mobile App**: React Native frontend
5. **Batch Processing**: Process multiple products at once

---

For more details, see:
- [API Specification](api_spec.md)
- [Model Training Notes](model_notes.md)

