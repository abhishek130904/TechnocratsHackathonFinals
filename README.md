# 🔍 Multi-Modal AI for Counterfeit Product Detection

A full-stack machine learning system that detects counterfeit products from e-commerce listings using:
- **Image Analysis**: CNN (EfficientNetB0) for visual authenticity detection
- **Text Analysis**: Transformer embeddings (DistilBERT) for description analysis
- **Metadata Analysis**: Seller rating and review count
- **Multimodal Fusion**: Combines all modalities for final prediction

## 📋 Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Running the Application](#running-the-application)
- [Training Models](#training-models)
- [API Documentation](#api-documentation)
- [Architecture](#architecture)
- [Contributing](#contributing)

## ✨ Features

- **Multi-Modal Analysis**: Combines image, text, and metadata for robust detection
- **Real-Time Predictions**: Fast API endpoint for instant authenticity checks
- **Explainable AI**: Provides detailed explanations and Grad-CAM heatmaps
- **Modern UI**: Clean React frontend with intuitive user experience
- **Production Ready**: Well-structured codebase with comprehensive error handling

## 📁 Project Structure

```
project-root/
├── backend/                 # FastAPI backend
│   ├── app/
│   │   ├── main.py         # FastAPI application entry point
│   │   ├── config.py       # Configuration settings
│   │   ├── schemas.py      # Pydantic request/response models
│   │   ├── routes/         # API route handlers
│   │   ├── services/       # Business logic services
│   │   ├── ml/             # ML model interfaces
│   │   └── utils/          # Utility functions
│   └── requirements.txt    # Python dependencies
│
├── ml_pipeline/            # Machine learning training
│   ├── notebooks/          # Jupyter notebooks for exploration
│   ├── scripts/            # Training scripts
│   └── models/             # Trained model files (generated)
│
├── frontend/               # React frontend
│   ├── src/
│   │   ├── components/     # React components
│   │   ├── App.tsx         # Main app component
│   │   └── types.ts        # TypeScript type definitions
│   └── package.json        # Node.js dependencies
│
├── docs/                   # Documentation
│   ├── architecture.md     # System architecture details
│   ├── api_spec.md         # API endpoint documentation
│   └── model_notes.md      # ML model training notes
│
└── README.md              # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Node.js 16+ and npm
- Virtual environment (recommended)

### Installation

1. **Clone the repository** (if applicable)
   ```bash
   cd Techno
   ```

2. **Set up Python backend**
   ```bash
   # Create virtual environment
   python -m venv venv
   
   # Activate virtual environment
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   
   # Install dependencies
   cd backend
   pip install -r requirements.txt
   ```

3. **Set up React frontend**
   ```bash
   cd frontend
   npm install
   ```

## 🏃 Running the Application

### 1. Start the Backend Server

```bash
# From the backend directory
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at:
- API: http://localhost:8000
- API Docs: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### 2. Start the Frontend

```bash
# From the frontend directory
cd frontend
npm run dev
```

The frontend will be available at: http://localhost:3000

### 3. Use the Application

1. Open http://localhost:3000 in your browser
2. Upload a product image
3. Enter product title and description
4. Optionally add seller rating and review count
5. Click "Check Authenticity"
6. View the prediction results with explanations

## 🎓 Training Models

### Dataset Structure

Organize your dataset as follows:

```
dataset/
├── train/
│   ├── real/          # Authentic product images
│   │   ├── img1.jpg
│   │   └── img2.jpg
│   └── fake/          # Counterfeit product images
│       ├── img1.jpg
│       └── img2.jpg
├── val/
│   ├── real/
│   └── fake/
└── test/
    ├── real/
    └── fake/
```

### Step 1: Train Image Model

```bash
python ml_pipeline/scripts/train_image_model.py \
    --dataset_path dataset/ \
    --epochs 20 \
    --batch_size 32
```

This will create `ml_pipeline/models/image_model.h5`

### Step 2: Train Fusion Model

```bash
python ml_pipeline/scripts/train_multimodal.py \
    --dataset_path dataset/ \
    --image_model_path ml_pipeline/models/image_model.h5 \
    --epochs 20
```

This will create `ml_pipeline/models/fusion_model.h5`

### Step 3: Restart Backend

After training, restart the backend server. It will automatically load the new models.

## 📚 API Documentation

### Endpoints

#### `GET /`
Health check endpoint.

**Response:**
```json
{
  "status": "ok",
  "version": "1.0.0",
  "models_loaded": {
    "image_model": true,
    "text_model": true,
    "fusion_model": true
  }
}
```

#### `POST /api/predict`
Main prediction endpoint.

**Request (multipart/form-data):**
- `image`: Image file (JPG, PNG, WEBP)
- `title`: Product title (string)
- `description`: Product description (string)
- `seller_rating`: Optional seller rating (float, 0.0-5.0)
- `num_reviews`: Optional number of reviews (integer)

**Response:**
```json
{
  "authenticity_score": 0.83,
  "decision": "flag",
  "explanations": {
    "image_reason": "Image shows some inconsistencies...",
    "text_reason": "Description contains suspicious terms...",
    "metadata_reason": "Low seller rating (2.1/5.0)...",
    "heatmap": "base64_encoded_image_string_or_null"
  }
}
```

For detailed API documentation, see [docs/api_spec.md](docs/api_spec.md)

## 🏗️ Architecture

### System Overview

1. **Frontend (React)**: User interface for uploading images and viewing results
2. **Backend (FastAPI)**: REST API that handles requests and orchestrates ML models
3. **ML Pipeline**: Training scripts and model interfaces
4. **Models**:
   - Image Model: CNN (EfficientNetB0) for visual analysis
   - Text Model: DistilBERT for text embeddings
   - Fusion Model: Dense network combining all features

For detailed architecture, see [docs/architecture.md](docs/architecture.md)

## 🔧 Configuration

Key configuration options in `backend/app/config.py`:

- `APPROVE_THRESHOLD`: Score threshold for approval (default: 0.75)
- `FLAG_THRESHOLD`: Score threshold for flagging (default: 0.50)
- `IMAGE_SIZE`: Input image size (default: 224x224)
- `MAX_UPLOAD_SIZE`: Maximum file upload size (default: 10 MB)

## 📖 Learning Resources

This project is designed for beginners. Key concepts explained:

- **Transfer Learning**: Using pretrained models (EfficientNetB0, DistilBERT)
- **Multimodal Fusion**: Combining different data types (image + text + metadata)
- **REST APIs**: Building and consuming REST endpoints
- **React Hooks**: State management in React
- **TypeScript**: Type-safe frontend development

## 🐛 Troubleshooting

### Models Not Loading

If models aren't loading:
1. Check that model files exist in `ml_pipeline/models/`
2. Train models using the training scripts
3. Check backend logs for error messages

### CORS Errors

If you see CORS errors:
1. Ensure backend is running on port 8000
2. Check `CORS_ORIGINS` in `backend/app/config.py`
3. Verify frontend is using the correct API URL

### Import Errors

If you see import errors:
1. Ensure virtual environment is activated
2. Install all requirements: `pip install -r backend/requirements.txt`
3. Check Python version (3.8+ required)

## 📝 License

This project is for educational purposes.

## 🤝 Contributing

This is a learning project. Feel free to:
- Add more features
- Improve documentation
- Fix bugs
- Enhance the UI

## 📧 Support

For questions or issues, please check:
- [Architecture Documentation](docs/architecture.md)
- [API Specification](docs/api_spec.md)
- [Model Training Notes](docs/model_notes.md)

---

**Happy Learning! 🎓**
