"""
Training script for the image classification CNN model.

This script:
1. Loads the dataset
2. Creates a CNN model using transfer learning (EfficientNetB0)
3. Trains the model
4. Evaluates on validation set
5. Saves the trained model

To run:
    python ml_pipeline/scripts/train_image_model.py --dataset_path dataset/ --epochs 10
"""

import argparse
import sys
from pathlib import Path
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.applications import EfficientNetB0
    from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
    from tensorflow.keras.models import Model
    from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
    TF_AVAILABLE = True
except ImportError:
    print("Error: TensorFlow is required. Install with: pip install tensorflow")
    sys.exit(1)

from ml_pipeline.scripts.data_loader import load_dataset, create_data_generator


def create_image_model(input_shape=(224, 224, 3), num_classes=1):
    """
    Create a CNN model using transfer learning with EfficientNetB0.
    
    Transfer learning means we:
    1. Use a pretrained model (EfficientNetB0) that was trained on ImageNet
    2. Remove the final classification layers
    3. Add our own layers for binary classification (fake vs real)
    4. Optionally freeze some layers to preserve pretrained features
    
    Args:
        input_shape: Shape of input images (height, width, channels)
        num_classes: Number of output classes (1 for binary classification)
        
    Returns:
        Compiled Keras model
    """
    print("Creating image model with EfficientNetB0...")
    
    # Load pretrained EfficientNetB0
    # include_top=False means we don't want the final classification layers
    # weights='imagenet' means use weights pretrained on ImageNet dataset
    base_model = EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape
    )
    
    # Freeze the base model layers initially
    # This means we won't update these weights during the first training phase
    # We can unfreeze them later for fine-tuning
    base_model.trainable = False
    
    # Add our custom classification head
    # This is the part that learns to classify fake vs real
    inputs = keras.Input(shape=input_shape)
    
    # Pass through base model (feature extraction)
    x = base_model(inputs, training=False)
    
    # Global average pooling: reduces spatial dimensions to a single value per channel
    # This converts (batch, H, W, channels) to (batch, channels)
    x = GlobalAveragePooling2D()(x)
    
    # Dropout: randomly sets some neurons to 0 during training
    # This prevents overfitting (memorizing the training data)
    x = Dropout(0.2)(x)
    
    # Final dense layer for binary classification
    # num_classes=1 with sigmoid activation outputs a probability (0-1)
    # 0 = fake, 1 = real
    outputs = Dense(num_classes, activation='sigmoid')(x)
    
    # Create the complete model
    model = Model(inputs, outputs)
    
    # Compile the model
    # - optimizer: algorithm that updates weights during training
    # - loss: function that measures how wrong predictions are
    # - metrics: additional metrics to track during training
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',  # Standard loss for binary classification
        metrics=['accuracy', 'precision', 'recall']
    )
    
    print("Model created successfully!")
    print(f"Total parameters: {model.count_params():,}")
    print(f"Trainable parameters: {sum([tf.size(w).numpy() for w in model.trainable_weights]):,}")
    
    return model


def train_model(
    dataset_path: str,
    epochs: int = 10,
    batch_size: int = 32,
    model_save_path: str = "ml_pipeline/models/image_model.h5"
):
    """
    Main training function.
    
    Args:
        dataset_path: Path to dataset root directory
        epochs: Number of training epochs (full passes through the data)
        batch_size: Number of images per batch
        model_save_path: Where to save the trained model
    """
    print("=" * 60)
    print("IMAGE MODEL TRAINING")
    print("=" * 60)
    
    # Load datasets
    print("\nLoading training data...")
    train_images, train_labels = load_dataset(dataset_path, split="train")
    
    print("\nLoading validation data...")
    val_images, val_labels = load_dataset(dataset_path, split="val")
    
    # Create data generators
    print("\nCreating data generators...")
    train_gen = create_data_generator(
        train_images, train_labels,
        batch_size=batch_size,
        augment=True,  # Use augmentation for training
        shuffle=True
    )
    
    val_gen = create_data_generator(
        val_images, val_labels,
        batch_size=batch_size,
        augment=False,  # No augmentation for validation
        shuffle=False
    )
    
    # Create model
    print("\nCreating model...")
    model = create_image_model()
    
    # Print model architecture
    print("\nModel architecture:")
    model.summary()
    
    # Set up callbacks
    # Callbacks are functions that run during training
    callbacks = [
        # Save the best model (based on validation loss)
        ModelCheckpoint(
            model_save_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        # Stop training if validation loss doesn't improve
        EarlyStopping(
            monitor='val_loss',
            patience=5,  # Wait 5 epochs without improvement
            restore_best_weights=True,
            verbose=1
        ),
        # Reduce learning rate if loss plateaus
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,  # Reduce LR by half
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # Train the model
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)
    print(f"Training samples: {len(train_images)}")
    print(f"Validation samples: {len(val_images)}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print("=" * 60 + "\n")
    
    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate on validation set
    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)
    val_loss, val_accuracy, val_precision, val_recall = model.evaluate(val_gen, verbose=1)
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    print(f"Validation Precision: {val_precision:.4f}")
    print(f"Validation Recall: {val_recall:.4f}")
    
    # Save final model
    print(f"\nSaving model to {model_save_path}...")
    Path(model_save_path).parent.mkdir(parents=True, exist_ok=True)
    model.save(model_save_path)
    print("Model saved successfully!")
    
    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train image classification model")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="dataset",
        help="Path to dataset root directory"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training"
    )
    parser.add_argument(
        "--model_save_path",
        type=str,
        default="ml_pipeline/models/image_model.h5",
        help="Path to save the trained model"
    )
    
    args = parser.parse_args()
    
    # Run training
    train_model(
        dataset_path=args.dataset_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        model_save_path=args.model_save_path
    )

