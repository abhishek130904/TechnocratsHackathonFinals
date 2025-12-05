# Model Training Notes

This document contains detailed information about training and using the ML models.

## Overview

The system uses three main models:

1. **Image Model**: CNN for visual authenticity detection
2. **Text Model**: Transformer for text embeddings (pretrained, no training needed)
3. **Fusion Model**: Multimodal classifier combining all features

## Image Model

### Architecture

- **Base Model**: EfficientNetB0 (pretrained on ImageNet)
- **Transfer Learning**: Freeze base, train only classification head
- **Input**: 224x224x3 RGB images
- **Output**: Binary classification (fake=0, real=1)

### Training Process

1. **Data Preparation**:
   ```python
   dataset/
     train/
       real/  # Label = 1
       fake/  # Label = 0
     val/
       real/
       fake/
   ```

2. **Data Augmentation**:
   - Random rotation (±20°)
   - Random shifts (±10%)
   - Random zoom (±10%)
   - Horizontal flip
   - Brightness/contrast adjustment

3. **Training Configuration**:
   - **Optimizer**: Adam (learning_rate=0.001)
   - **Loss**: Binary crossentropy
   - **Metrics**: Accuracy, Precision, Recall
   - **Batch Size**: 32
   - **Epochs**: 20 (with early stopping)

4. **Callbacks**:
   - `ModelCheckpoint`: Save best model based on validation loss
   - `EarlyStopping`: Stop if no improvement for 5 epochs
   - `ReduceLROnPlateau`: Reduce learning rate if loss plateaus

### Training Command

```bash
python ml_pipeline/scripts/train_image_model.py \
    --dataset_path dataset/ \
    --epochs 20 \
    --batch_size 32 \
    --model_save_path ml_pipeline/models/image_model.h5
```

### Expected Performance

- **Training Accuracy**: 85-95% (depends on dataset)
- **Validation Accuracy**: 80-90%
- **Inference Time**: ~100-200ms per image (CPU)

### Fine-Tuning (Optional)

After initial training, you can unfreeze base layers for fine-tuning:

```python
# Unfreeze last few layers
base_model.trainable = True
for layer in base_model.layers[:-10]:
    layer.trainable = False

# Use lower learning rate
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    ...
)
```

## Text Model

### Architecture

- **Base Model**: DistilBERT (pretrained)
- **No Training Required**: Uses pretrained embeddings
- **Input**: Text (max 512 tokens)
- **Output**: 768-dimensional embedding vector

### Usage

The text model is loaded automatically from HuggingFace:

```python
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModel.from_pretrained("distilbert-base-uncased")
```

### Optional: Fine-Tuning for Classification

If you have labeled text data, you can fine-tune DistilBERT:

```python
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2  # fake vs real
)

# Train on your text dataset
# This would require text-only training data
```

## Fusion Model

### Architecture

- **Inputs**:
  - Image features: 1280 dimensions (from EfficientNetB0)
  - Text features: 768 dimensions (from DistilBERT)
  - Metadata: 2 dimensions (rating, reviews)
- **Architecture**:
  ```
  Image Features → Dense(256) → Dropout(0.3)
  Text Features  → Dense(256) → Dropout(0.3)
  Metadata      → Dense(32)  → Dropout(0.2)
  
  Concatenate → Dense(128) → Dropout(0.4) → Dense(64) → Dropout(0.3) → Dense(1, sigmoid)
  ```

### Training Process

1. **Feature Extraction**:
   - Extract features from trained image model
   - Extract embeddings from text model
   - Prepare metadata features

2. **Training Configuration**:
   - **Optimizer**: Adam (learning_rate=0.001)
   - **Loss**: Binary crossentropy
   - **Batch Size**: 32
   - **Epochs**: 20

### Training Command

```bash
python ml_pipeline/scripts/train_multimodal.py \
    --dataset_path dataset/ \
    --image_model_path ml_pipeline/models/image_model.h5 \
    --epochs 20 \
    --batch_size 32
```

### Expected Performance

- **Training Accuracy**: 90-95%
- **Validation Accuracy**: 85-92%
- **Inference Time**: ~10-20ms (very fast)

## Dataset Requirements

### Minimum Dataset Size

- **Training**: At least 100 images per class (200 total)
- **Validation**: At least 20 images per class (40 total)
- **Test**: At least 20 images per class (40 total)

### Recommended Dataset Size

- **Training**: 1000+ images per class
- **Validation**: 200+ images per class
- **Test**: 200+ images per class

### Data Quality Guidelines

1. **Image Quality**:
   - Clear, well-lit images
   - Consistent resolution
   - Remove duplicates

2. **Label Accuracy**:
   - Verify labels are correct
   - Handle edge cases (e.g., "gray market" products)

3. **Class Balance**:
   - Aim for balanced dataset (50/50 fake/real)
   - If imbalanced, use class weights or oversampling

## Hyperparameters

### Image Model

| Parameter | Value | Notes |
|-----------|-------|-------|
| Learning Rate | 0.001 | Standard for Adam |
| Batch Size | 32 | Adjust based on GPU memory |
| Epochs | 20 | With early stopping |
| Dropout | 0.2 | Prevents overfitting |

### Fusion Model

| Parameter | Value | Notes |
|-----------|-------|-------|
| Learning Rate | 0.001 | Can reduce to 0.0001 for fine-tuning |
| Batch Size | 32 | |
| Epochs | 20 | |
| Dropout | 0.3-0.4 | Higher dropout for regularization |

## Decision Thresholds

Configured in `backend/app/config.py`:

```python
APPROVE_THRESHOLD = 0.75  # Score >= 0.75 → approve
FLAG_THRESHOLD = 0.50     # 0.50 <= Score < 0.75 → flag
# Score < 0.50 → reject
```

### Tuning Thresholds

Adjust thresholds based on your use case:

- **High Precision (fewer false positives)**: Increase APPROVE_THRESHOLD to 0.85
- **High Recall (catch more fakes)**: Decrease FLAG_THRESHOLD to 0.40

## Model Evaluation

### Metrics

- **Accuracy**: Overall correctness
- **Precision**: Of predicted "real", how many are actually real?
- **Recall**: Of actual "real", how many did we catch?
- **F1 Score**: Harmonic mean of precision and recall

### Confusion Matrix

```
                Predicted
              Fake    Real
Actual Fake   TN      FP
       Real   FN      TP
```

- **TN (True Negative)**: Correctly identified fake
- **FP (False Positive)**: Incorrectly flagged real as fake
- **FN (False Negative)**: Missed a fake (said it's real)
- **TP (True Positive)**: Correctly identified real

### ROC Curve

Plot True Positive Rate vs False Positive Rate to visualize model performance.

## Model Deployment

### Model Formats

- **SavedModel** (`.h5`): Keras format, easy to load
- **TFLite** (`.tflite`): Optimized for mobile/edge
- **ONNX** (`.onnx`): Cross-platform format

### Conversion to TFLite (Optional)

```python
import tensorflow as tf

# Load model
model = tf.keras.models.load_model('image_model.h5')

# Convert
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save
with open('image_model.tflite', 'wb') as f:
    f.write(tflite_model)
```

## Troubleshooting

### Overfitting

**Symptoms**: High training accuracy, low validation accuracy

**Solutions**:
- Increase dropout
- Add more data augmentation
- Reduce model complexity
- Use early stopping

### Underfitting

**Symptoms**: Low training and validation accuracy

**Solutions**:
- Train for more epochs
- Increase model capacity
- Reduce regularization
- Check data quality

### Slow Training

**Solutions**:
- Use GPU acceleration
- Reduce batch size
- Use mixed precision training
- Optimize data loading

### Model Not Loading

**Check**:
1. Model file exists
2. File path is correct
3. TensorFlow version compatibility
4. Model architecture matches

## Future Improvements

1. **Advanced Architectures**:
   - Vision Transformer (ViT)
   - CLIP for image-text alignment
   - Ensemble methods

2. **Training Enhancements**:
   - Active learning
   - Semi-supervised learning
   - Few-shot learning

3. **Optimization**:
   - Model quantization
   - Pruning
   - Knowledge distillation

---

For more details, see:
- [Architecture Documentation](architecture.md)
- [API Specification](api_spec.md)

