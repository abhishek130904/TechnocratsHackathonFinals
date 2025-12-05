"""
Training script for the multimodal fusion model.

This script:
1. Loads pre-extracted features from image and text models
2. Combines them with metadata
3. Trains a fusion model to make final predictions

The fusion model learns to combine:
- Image features (from CNN)
- Text features (from transformer)
- Metadata (seller rating, review count)

To run:
    python ml_pipeline/scripts/train_multimodal.py --epochs 20
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.layers import Dense, Dropout, Concatenate, Input
    from tensorflow.keras.models import Model
    from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
    from sklearn.model_selection import train_test_split
    TF_AVAILABLE = True
except ImportError:
    print("Error: TensorFlow and scikit-learn are required.")
    sys.exit(1)

from backend.app.ml.image_model import ImageModel
from backend.app.ml.text_model import TextModel
from backend.app.ml.preprocessing import preprocess_image
from ml_pipeline.scripts.data_loader import load_dataset


def extract_features_from_dataset(
    dataset_path: str,
    image_model_path: str,
    image_size: tuple = (224, 224)
):
    """
    Extract features from image and text models for all samples.
    
    This function:
    1. Loads images and labels
    2. Extracts image features using the trained CNN
    3. Extracts text features using the transformer
    4. Returns combined features and labels
    
    Note: In a real scenario, you would have text data (titles, descriptions)
    associated with each image. For now, we'll use placeholder text.
    
    Args:
        dataset_path: Path to dataset
        image_model_path: Path to trained image model
        image_size: Image size for preprocessing
        
    Returns:
        Tuple of (features, labels) where features is a dict with:
        - image_features: (N, image_feat_dim)
        - text_features: (N, text_feat_dim)
        - metadata_features: (N, 2) - [rating, reviews]
    """
    print("Loading dataset...")
    train_images, train_labels = load_dataset(dataset_path, split="train")
    val_images, val_labels = load_dataset(dataset_path, split="val")
    
    # Combine train and val for feature extraction
    all_images = np.concatenate([train_images, val_images], axis=0)
    all_labels = np.concatenate([train_labels, val_labels], axis=0)
    
    print(f"Extracting features from {len(all_images)} images...")
    
    # Load image model
    image_model = ImageModel(model_path=image_model_path)
    if not image_model.is_loaded():
        raise ValueError(f"Could not load image model from {image_model_path}")
    
    # Load text model
    text_model = TextModel()
    if not text_model.is_loaded():
        raise ValueError("Could not load text model")
    
    # Extract features
    image_features_list = []
    text_features_list = []
    metadata_features_list = []
    
    # For each image, extract features
    # In practice, you would have actual text data here
    for i, image in enumerate(all_images):
        if i % 100 == 0:
            print(f"Processing {i}/{len(all_images)}...")
        
        # Image features
        # Add batch dimension
        image_batch = np.expand_dims(image, axis=0)
        img_feat = image_model.get_feature_vector(image_batch)
        image_features_list.append(img_feat[0])  # Remove batch dimension
        
        # Text features (placeholder - in real scenario, use actual text)
        # For now, we'll use a simple text based on the label
        placeholder_text = (
            "Authentic product with original packaging and warranty"
            if all_labels[i] == 1
            else "Product description unavailable"
        )
        text_feat = text_model.get_embeddings(placeholder_text)
        text_features_list.append(text_feat)
        
        # Metadata features (placeholder - in real scenario, use actual metadata)
        # Simulate: real products have higher ratings and more reviews
        if all_labels[i] == 1:
            rating = np.random.uniform(4.0, 5.0)
            reviews = np.random.randint(100, 10000)
        else:
            rating = np.random.uniform(1.0, 3.0)
            reviews = np.random.randint(0, 100)
        
        # Normalize metadata
        rating_norm = rating / 5.0
        reviews_norm = np.log1p(reviews) / np.log1p(10000)
        metadata_features_list.append([rating_norm, reviews_norm])
    
    return {
        'image_features': np.array(image_features_list),
        'text_features': np.array(text_features_list),
        'metadata_features': np.array(metadata_features_list)
    }, all_labels


def create_fusion_model(
    image_feat_dim: int = 1280,  # EfficientNetB0 feature size
    text_feat_dim: int = 768,    # DistilBERT embedding size
    metadata_feat_dim: int = 2
):
    """
    Create the multimodal fusion model.
    
    Architecture:
    1. Separate input layers for each modality
    2. Optional: small processing layers for each
    3. Concatenate all features
    4. Dense layers for final classification
    
    Args:
        image_feat_dim: Dimension of image feature vector
        text_feat_dim: Dimension of text embedding
        metadata_feat_dim: Dimension of metadata (2: rating, reviews)
        
    Returns:
        Compiled Keras model
    """
    print("Creating fusion model...")
    
    # Input layers for each modality
    image_input = Input(shape=(image_feat_dim,), name='image_input')
    text_input = Input(shape=(text_feat_dim,), name='text_input')
    metadata_input = Input(shape=(metadata_feat_dim,), name='metadata_input')
    
    # Optional: process each modality separately
    # This allows the model to learn modality-specific patterns
    image_processed = Dense(256, activation='relu')(image_input)
    image_processed = Dropout(0.3)(image_processed)
    
    text_processed = Dense(256, activation='relu')(text_input)
    text_processed = Dropout(0.3)(text_processed)
    
    metadata_processed = Dense(32, activation='relu')(metadata_input)
    metadata_processed = Dropout(0.2)(metadata_processed)
    
    # Concatenate all features
    # This combines information from all modalities
    concatenated = Concatenate()([image_processed, text_processed, metadata_processed])
    
    # Final classification layers
    x = Dense(128, activation='relu')(concatenated)
    x = Dropout(0.4)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    
    # Output layer: binary classification (fake vs real)
    output = Dense(1, activation='sigmoid', name='output')(x)
    
    # Create model
    model = Model(
        inputs=[image_input, text_input, metadata_input],
        outputs=output
    )
    
    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
    
    print("Fusion model created!")
    model.summary()
    
    return model


def train_fusion_model(
    features: dict,
    labels: np.ndarray,
    epochs: int = 20,
    batch_size: int = 32,
    model_save_path: str = "ml_pipeline/models/fusion_model.h5"
):
    """
    Train the fusion model.
    
    Args:
        features: Dictionary with image_features, text_features, metadata_features
        labels: Ground truth labels (0 = fake, 1 = real)
        epochs: Number of training epochs
        batch_size: Batch size
        model_save_path: Where to save the model
    """
    print("=" * 60)
    print("MULTIMODAL FUSION MODEL TRAINING")
    print("=" * 60)
    
    # Split into train and validation
    # Use 80% for training, 20% for validation
    indices = np.arange(len(labels))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=0.2,
        random_state=42,
        stratify=labels  # Ensure balanced splits
    )
    
    # Split features and labels
    train_features = {
        'image': features['image_features'][train_idx],
        'text': features['text_features'][train_idx],
        'metadata': features['metadata_features'][train_idx]
    }
    train_labels = labels[train_idx]
    
    val_features = {
        'image': features['image_features'][val_idx],
        'text': features['text_features'][val_idx],
        'metadata': features['metadata_features'][val_idx]
    }
    val_labels = labels[val_idx]
    
    print(f"Training samples: {len(train_labels)}")
    print(f"Validation samples: {len(val_labels)}")
    
    # Create model
    model = create_fusion_model(
        image_feat_dim=train_features['image'].shape[1],
        text_feat_dim=train_features['text'].shape[1],
        metadata_feat_dim=train_features['metadata'].shape[1]
    )
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            model_save_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        )
    ]
    
    # Train
    print("\nStarting training...")
    history = model.fit(
        x=[train_features['image'], train_features['text'], train_features['metadata']],
        y=train_labels,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(
            [val_features['image'], val_features['text'], val_features['metadata']],
            val_labels
        ),
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    print("\nEvaluating...")
    val_loss, val_acc, val_prec, val_rec = model.evaluate(
        [val_features['image'], val_features['text'], val_features['metadata']],
        val_labels,
        verbose=1
    )
    
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_acc:.4f}")
    print(f"Validation Precision: {val_prec:.4f}")
    print(f"Validation Recall: {val_rec:.4f}")
    
    # Save
    print(f"\nSaving model to {model_save_path}...")
    Path(model_save_path).parent.mkdir(parents=True, exist_ok=True)
    model.save(model_save_path)
    print("Model saved!")
    
    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train multimodal fusion model")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="dataset",
        help="Path to dataset root"
    )
    parser.add_argument(
        "--image_model_path",
        type=str,
        default="ml_pipeline/models/image_model.h5",
        help="Path to trained image model"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size"
    )
    parser.add_argument(
        "--model_save_path",
        type=str,
        default="ml_pipeline/models/fusion_model.h5",
        help="Path to save fusion model"
    )
    
    args = parser.parse_args()
    
    # Extract features
    print("Extracting features from dataset...")
    features, labels = extract_features_from_dataset(
        args.dataset_path,
        args.image_model_path
    )
    
    # Train
    train_fusion_model(
        features,
        labels,
        epochs=args.epochs,
        batch_size=args.batch_size,
        model_save_path=args.model_save_path
    )

