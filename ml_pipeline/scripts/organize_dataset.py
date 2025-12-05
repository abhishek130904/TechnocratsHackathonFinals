"""
Dataset Organization Script

This script helps you organize your collected images into the proper
folder structure for training.

Usage:
    python ml_pipeline/scripts/organize_dataset.py \
        --source_dir "raw_images" \
        --output_dir "dataset" \
        --train_ratio 0.8 \
        --val_ratio 0.1
"""

import argparse
import shutil
import random
from pathlib import Path
from typing import List


def organize_dataset(
    source_dir: str,
    output_dir: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42
):
    """
    Organize images into train/val/test splits.
    
    This function:
    1. Reads images from source_dir/real and source_dir/fake
    2. Randomly shuffles them
    3. Splits into train/val/test based on ratios
    4. Copies to organized structure
    
    Args:
        source_dir: Directory containing 'real' and 'fake' subdirectories
        output_dir: Where to create organized dataset
        train_ratio: Percentage for training (default 0.8 = 80%)
        val_ratio: Percentage for validation (default 0.1 = 10%)
        seed: Random seed for reproducibility
    """
    source = Path(source_dir)
    output = Path(output_dir)
    
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Validate ratios
    test_ratio = 1.0 - train_ratio - val_ratio
    if test_ratio <= 0:
        raise ValueError("train_ratio + val_ratio must be less than 1.0")
    
    print("=" * 60)
    print("Dataset Organization")
    print("=" * 60)
    print(f"Source directory: {source_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Split ratios: Train={train_ratio:.1%}, Val={val_ratio:.1%}, Test={test_ratio:.1%}")
    print()
    
    total_real = 0
    total_fake = 0
    
    for label in ['real', 'fake']:
        label_dir = source / label
        
        if not label_dir.exists():
            print(f"⚠️  Warning: {label_dir} does not exist. Skipping...")
            continue
        
        # Find all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        images = [
            img for img in label_dir.iterdir()
            if img.suffix.lower() in image_extensions and img.is_file()
        ]
        
        if len(images) == 0:
            print(f"⚠️  Warning: No images found in {label_dir}")
            continue
        
        # Shuffle images
        random.shuffle(images)
        
        # Calculate splits
        n_total = len(images)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        n_test = n_total - n_train - n_val
        
        # Split images
        train_images = images[:n_train]
        val_images = images[n_train:n_train+n_val]
        test_images = images[n_train+n_val:]
        
        print(f"\n{label.upper()} Images:")
        print(f"  Total: {n_total}")
        print(f"  Train: {n_train} ({n_train/n_total:.1%})")
        print(f"  Val:   {n_val} ({n_val/n_total:.1%})")
        print(f"  Test:  {n_test} ({n_test/n_total:.1%})")
        
        # Copy to organized structure
        for split, img_list in [
            ('train', train_images),
            ('val', val_images),
            ('test', test_images)
        ]:
            dest_dir = output / split / label
            dest_dir.mkdir(parents=True, exist_ok=True)
            
            for img in img_list:
                # Copy with original filename
                shutil.copy2(img, dest_dir / img.name)
        
        if label == 'real':
            total_real = n_total
        else:
            total_fake = n_total
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Total Real Images: {total_real}")
    print(f"Total Fake Images: {total_fake}")
    print(f"Total Images: {total_real + total_fake}")
    print(f"\n✅ Dataset organized successfully!")
    print(f"   Output: {output_dir}")
    print(f"\nNext step: Train your model with:")
    print(f"  python ml_pipeline/scripts/train_image_model.py --dataset_path {output_dir}")


def validate_dataset(dataset_dir: str):
    """
    Validate that dataset is properly organized.
    
    Args:
        dataset_dir: Path to organized dataset
    """
    dataset = Path(dataset_dir)
    
    print("\n" + "=" * 60)
    print("Dataset Validation")
    print("=" * 60)
    
    issues = []
    
    for split in ['train', 'val', 'test']:
        split_dir = dataset / split
        if not split_dir.exists():
            issues.append(f"Missing {split} directory")
            continue
        
        for label in ['real', 'fake']:
            label_dir = split_dir / label
            if not label_dir.exists():
                issues.append(f"Missing {split}/{label} directory")
                continue
            
            images = list(label_dir.glob("*.jpg")) + list(label_dir.glob("*.png"))
            print(f"{split}/{label}: {len(images)} images")
            
            if len(images) == 0:
                issues.append(f"No images in {split}/{label}")
    
    if issues:
        print("\n⚠️  Issues found:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\n✅ Dataset structure is valid!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Organize images into train/val/test splits"
    )
    parser.add_argument(
        "--source_dir",
        type=str,
        required=True,
        help="Directory containing 'real' and 'fake' subdirectories"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="dataset",
        help="Output directory for organized dataset"
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Ratio of images for training (default: 0.8)"
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Ratio of images for validation (default: 0.1)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate dataset structure after organizing"
    )
    
    args = parser.parse_args()
    
    # Organize dataset
    organize_dataset(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed
    )
    
    # Validate if requested
    if args.validate:
        validate_dataset(args.output_dir)

