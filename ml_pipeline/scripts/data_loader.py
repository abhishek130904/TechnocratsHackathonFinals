"""
Data loading utilities for training.

This module provides functions to:
- Load images from folder structure
- Load text data (titles, descriptions)
- Create train/val/test splits
- Apply data augmentation

Dataset structure expected:
    dataset/
        train/
            real/
                image1.jpg
                image2.jpg
                ...
            fake/
                image1.jpg
                image2.jpg
                ...
        val/
            real/
                ...
            fake/
                ...
        test/
            real/
                ...
            fake/
                ...
"""

import os
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional
from PIL import Image
import pandas as pd

# Try to import TensorFlow
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("Warning: TensorFlow not available. Some functions will not work.")


def load_images_from_folder(
    folder_path: str,
    label: int,
    image_size: Tuple[int, int] = (224, 224)
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load images from a folder and assign labels.
    
    This function:
    1. Scans a folder for image files
    2. Loads and resizes images
    3. Converts to numpy arrays
    4. Assigns labels (0 = fake, 1 = real)
    
    Args:
        folder_path: Path to folder containing images
        label: Label for all images in this folder (0 or 1)
        image_size: Target size for images (width, height)
        
    Returns:
        Tuple of (images, labels) as numpy arrays
        - images: shape (N, H, W, 3) where N is number of images
        - labels: shape (N,) with label values
    """
    images = []
    labels = []
    
    # Supported image extensions
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    
    folder = Path(folder_path)
    if not folder.exists():
        raise ValueError(f"Folder does not exist: {folder_path}")
    
    print(f"Loading images from {folder_path}...")
    
    # Iterate through all files in folder
    for file_path in folder.iterdir():
        if file_path.suffix.lower() in valid_extensions:
            try:
                # Load image
                img = Image.open(file_path)
                
                # Convert to RGB (handles RGBA, grayscale, etc.)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Resize to target size
                img = img.resize(image_size, Image.Resampling.LANCZOS)
                
                # Convert to numpy array and normalize to 0-1
                img_array = np.array(img, dtype=np.float32) / 255.0
                
                images.append(img_array)
                labels.append(label)
                
            except Exception as e:
                print(f"Warning: Could not load {file_path}: {e}")
                continue
    
    if len(images) == 0:
        raise ValueError(f"No valid images found in {folder_path}")
    
    print(f"Loaded {len(images)} images with label {label}")
    
    return np.array(images), np.array(labels)


def load_dataset(
    dataset_root: str,
    split: str = "train",
    image_size: Tuple[int, int] = (224, 224)
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a complete dataset split (train/val/test).
    
    Expected folder structure:
        dataset_root/
            split/
                real/
                    images...
                fake/
                    images...
    
    Args:
        dataset_root: Root directory of the dataset
        split: Which split to load ("train", "val", or "test")
        image_size: Target image size
        
    Returns:
        Tuple of (images, labels)
        - images: shape (N, H, W, 3)
        - labels: shape (N,) where 0 = fake, 1 = real
    """
    dataset_path = Path(dataset_root) / split
    
    # Load real images (label = 1)
    real_path = dataset_path / "real"
    if real_path.exists():
        real_images, real_labels = load_images_from_folder(
            str(real_path),
            label=1,
            image_size=image_size
        )
    else:
        print(f"Warning: {real_path} does not exist")
        real_images, real_labels = np.array([]), np.array([])
    
    # Load fake images (label = 0)
    fake_path = dataset_path / "fake"
    if fake_path.exists():
        fake_images, fake_labels = load_images_from_folder(
            str(fake_path),
            label=0,
            image_size=image_size
        )
    else:
        print(f"Warning: {fake_path} does not exist")
        fake_images, fake_labels = np.array([]), np.array([])
    
    # Combine real and fake
    if len(real_images) > 0 and len(fake_images) > 0:
        all_images = np.concatenate([real_images, fake_images], axis=0)
        all_labels = np.concatenate([real_labels, fake_labels], axis=0)
    elif len(real_images) > 0:
        all_images = real_images
        all_labels = real_labels
    elif len(fake_images) > 0:
        all_images = fake_images
        all_labels = fake_labels
    else:
        raise ValueError(f"No images found in {dataset_path}")
    
    # Shuffle the data
    indices = np.random.permutation(len(all_images))
    all_images = all_images[indices]
    all_labels = all_labels[indices]
    
    print(f"Total loaded: {len(all_images)} images ({np.sum(all_labels==1)} real, {np.sum(all_labels==0)} fake)")
    
    return all_images, all_labels


def create_data_generator(
    images: np.ndarray,
    labels: np.ndarray,
    batch_size: int = 32,
    augment: bool = True,
    shuffle: bool = True
) -> tf.data.Dataset:
    """
    Create a TensorFlow data generator with optional augmentation.
    
    Data augmentation helps the model generalize better by:
    - Randomly flipping images
    - Rotating images slightly
    - Adjusting brightness/contrast
    - Zooming in/out
    
    Args:
        images: Image array (N, H, W, 3)
        labels: Label array (N,)
        batch_size: Number of images per batch
        augment: Whether to apply data augmentation
        shuffle: Whether to shuffle the data
        
    Returns:
        TensorFlow Dataset object
    """
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow is required for data generation")
    
    # Create ImageDataGenerator for augmentation
    if augment:
        datagen = ImageDataGenerator(
            rotation_range=20,        # Rotate up to 20 degrees
            width_shift_range=0.1,   # Shift horizontally by up to 10%
            height_shift_range=0.1,  # Shift vertically by up to 10%
            shear_range=0.1,         # Apply shear transformation
            zoom_range=0.1,          # Zoom in/out by up to 10%
            horizontal_flip=True,    # Randomly flip horizontally
            fill_mode='nearest'       # Fill empty pixels
        )
    else:
        datagen = ImageDataGenerator()
    
    # Create dataset
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(images))
    
    # Apply augmentation if needed
    if augment:
        # Note: ImageDataGenerator is used differently in practice
        # For simplicity, we'll use tf.image augmentation
        def augment_fn(image, label):
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_brightness(image, 0.2)
            image = tf.image.random_contrast(image, 0.8, 1.2)
            return image, label
        
        dataset = dataset.map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Batch the dataset
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

