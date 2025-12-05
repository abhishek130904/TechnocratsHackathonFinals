# Models Directory

This directory will store trained model files.

## Expected Model Files

After training, you should have:

- `image_model.h5` - Trained CNN model for image classification
- `text_embedder.pkl` - Text embedding model (or HuggingFace model path)
- `fusion_model.h5` or `fusion_model.pkl` - Multimodal fusion classifier

## Model Loading

Models are loaded by the backend at startup in `backend/app/ml/` modules.

## Training

Run training scripts from `ml_pipeline/scripts/` to generate these models.

