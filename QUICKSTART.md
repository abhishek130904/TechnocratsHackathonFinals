# Quick Start Guide

This guide will help you get the Counterfeit Product Detection system up and running in minutes.

## Step 1: Set Up Python Backend

### 1.1 Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 1.2 Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

**Note**: This may take a few minutes as it installs TensorFlow and other large packages.

### 1.3 Test Backend (Optional - Models Not Required)

```bash
# From backend directory
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Open http://localhost:8000 in your browser. You should see:
```json
{
  "status": "ok",
  "version": "1.0.0",
  "models_loaded": {
    "image_model": false,
    "text_model": false,
    "fusion_model": false
  }
}
```

Press `Ctrl+C` to stop the server.

## Step 2: Set Up React Frontend

### 2.1 Install Node Dependencies

```bash
cd frontend
npm install
```

### 2.2 Start Frontend (Optional - Can Run Without Backend)

```bash
npm run dev
```

The frontend will open at http://localhost:3000

Press `Ctrl+C` to stop.

## Step 3: Train Models (Optional but Recommended)

### 3.1 Prepare Your Dataset

Organize your images:

```
dataset/
├── train/
│   ├── real/
│   │   ├── img1.jpg
│   │   └── img2.jpg
│   └── fake/
│       ├── img1.jpg
│       └── img2.jpg
├── val/
│   ├── real/
│   └── fake/
└── test/
    ├── real/
    └── fake/
```

**Minimum**: 20 images per class per split (120 total)
**Recommended**: 200+ images per class per split (1200+ total)

### 3.2 Train Image Model

```bash
# From project root
python ml_pipeline/scripts/train_image_model.py \
    --dataset_path dataset/ \
    --epochs 10 \
    --batch_size 32
```

This creates `ml_pipeline/models/image_model.h5`

### 3.3 Train Fusion Model

```bash
python ml_pipeline/scripts/train_multimodal.py \
    --dataset_path dataset/ \
    --image_model_path ml_pipeline/models/image_model.h5 \
    --epochs 10
```

This creates `ml_pipeline/models/fusion_model.h5`

## Step 4: Run the Full Application

### 4.1 Start Backend

```bash
# Terminal 1
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 4.2 Start Frontend

```bash
# Terminal 2
cd frontend
npm run dev
```

### 4.3 Use the Application

1. Open http://localhost:3000
2. Upload a product image
3. Enter product title and description
4. (Optional) Add seller rating and review count
5. Click "Check Authenticity"
6. View results!

## Troubleshooting

### "Models not loaded" Warning

This is normal if you haven't trained models yet. The API will use dummy predictions.

**Solution**: Train models using Step 3, then restart the backend.

### Port Already in Use

If port 8000 or 3000 is already in use:

**Backend**: Change port in command:
```bash
uvicorn app.main:app --reload --port 8001
```

**Frontend**: Edit `frontend/vite.config.ts`:
```typescript
server: {
  port: 3001,  // Change this
}
```

### Import Errors

**Problem**: `ModuleNotFoundError`

**Solution**: 
1. Ensure virtual environment is activated
2. Reinstall dependencies: `pip install -r backend/requirements.txt`

### CORS Errors

**Problem**: Frontend can't connect to backend

**Solution**: 
1. Ensure backend is running on port 8000
2. Check `CORS_ORIGINS` in `backend/app/config.py`

## Next Steps

- Read the [README.md](README.md) for detailed documentation
- Check [docs/architecture.md](docs/architecture.md) for system design
- Review [docs/api_spec.md](docs/api_spec.md) for API details
- See [docs/model_notes.md](docs/model_notes.md) for training tips

## Testing Without Models

You can test the system without training models:

1. The backend will use dummy predictions
2. All endpoints will work
3. Responses will have placeholder scores
4. This is useful for testing the UI and API flow

## Getting Help

- Check the documentation in `docs/` folder
- Review code comments (they're extensive!)
- Check backend logs: `logs/app.log`

---

**Happy coding! 🚀**

