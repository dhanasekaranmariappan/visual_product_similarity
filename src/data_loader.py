"""
data_loader.py — Dataset Loading, Preprocessing & Metadata Generation
======================================================================

This module handles everything related to reading product images from disk,
applying the correct preprocessing transforms, and generating a metadata
catalog (CSV) that maps each image to its category and file path.

Architecture:
    data/raw/
    ├── category_1/
    │   ├── img_001.jpg
    │   ├── img_002.png
    │   └── ...
    ├── category_2/
    │   ├── img_101.jpg
    │   └── ...
    └── ...

    The folder name IS the category label. This is the standard convention
    for image classification/retrieval datasets (ImageFolder structure).

Usage:
    from src.data_loader import ProductImageDataset, create_metadata_csv

    # Generate metadata
    metadata_df = create_metadata_csv()

    # Create dataset for batch processing
    dataset = ProductImageDataset(metadata_df)
    dataloader = DataLoader(dataset, batch_size=64, num_workers=4)
"""

import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
from PIL import Image
from loguru import logger
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Add project root to path for config import
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import Config


# =========================================================================
# IMAGE VALIDATION
# =========================================================================

def is_valid_image(image_path: str) -> bool:
    """
    Check if an image file is valid and can be opened.
    
    Some images in datasets may be corrupted, truncated, or in an
    unsupported format. This function catches those before they crash
    the embedding pipeline.
    
    Args:
        image_path: Absolute path to the image file.
    
    Returns:
        True if the image can be opened and has valid dimensions.
    """
    try:
        with Image.open(image_path) as img:
            img.verify()  # Verify file integrity without loading full pixels
        # Re-open after verify (verify() makes the object unusable)
        with Image.open(image_path) as img:
            width, height = img.size
            if width < 10 or height < 10:
                logger.warning(f"Image too small ({width}x{height}): {image_path}")
                return False
        return True
    except Exception as e:
        logger.warning(f"Invalid image skipped: {image_path} — {e}")
        return False


# =========================================================================
# METADATA GENERATION
# =========================================================================

def scan_image_directory(data_dir: Optional[Path] = None) -> List[Dict]:
    """
    Recursively scan a directory for product images organized in category folders.
    
    Expects the ImageFolder structure:
        data_dir/
        ├── Shoes/
        │   ├── shoe_001.jpg
        │   └── shoe_002.png
        ├── Bags/
        │   ├── bag_001.jpg
        │   └── ...
    
    Args:
        data_dir: Root directory containing category subfolders.
                  Defaults to Config.RAW_DATA_DIR.
    
    Returns:
        List of dicts with keys: image_id, image_path, category, filename
    """
    if data_dir is None:
        data_dir = Config.RAW_DATA_DIR

    data_dir = Path(data_dir)
    if not data_dir.exists():
        logger.error(f"Data directory does not exist: {data_dir}")
        raise FileNotFoundError(
            f"Data directory not found: {data_dir}\n"
            f"Please place your product images in: {data_dir}/\n"
            f"Structure: {data_dir}/<category_name>/<image_file>"
        )

    records = []
    image_id = 0

    # Get all category folders (sorted for reproducibility)
    category_dirs = sorted([
        d for d in data_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ])

    if not category_dirs:
        # Check if images are directly in data_dir (no subfolders)
        direct_images = [
            f for f in data_dir.iterdir()
            if f.is_file() and f.suffix.lower() in Config.SUPPORTED_EXTENSIONS
        ]
        if direct_images:
            logger.info(
                f"Found {len(direct_images)} images directly in {data_dir} "
                f"(no category subfolders). Assigning category='uncategorized'."
            )
            for img_path in sorted(direct_images):
                records.append({
                    "image_id": image_id,
                    "image_path": str(img_path.resolve()),
                    "category": "uncategorized",
                    "filename": img_path.name,
                })
                image_id += 1
            return records

        logger.error(f"No category folders or images found in: {data_dir}")
        raise FileNotFoundError(
            f"No images found in {data_dir}. "
            f"Expected structure: {data_dir}/<category>/<image>.*"
        )

    # Scan each category folder
    for cat_dir in category_dirs:
        category_name = cat_dir.name
        image_files = sorted([
            f for f in cat_dir.iterdir()
            if f.is_file() and f.suffix.lower() in Config.SUPPORTED_EXTENSIONS
        ])

        if not image_files:
            logger.warning(f"No images found in category folder: {cat_dir}")
            continue

        for img_path in image_files:
            records.append({
                "image_id": image_id,
                "image_path": str(img_path.resolve()),
                "category": category_name,
                "filename": img_path.name,
            })
            image_id += 1

    logger.info(
        f"Scanned {len(records)} images across "
        f"{len(category_dirs)} categories from {data_dir}"
    )
    return records


def create_metadata_csv(
    data_dir: Optional[Path] = None,
    output_path: Optional[Path] = None,
    validate_images: bool = True,
) -> pd.DataFrame:
    """
    Generate a metadata CSV cataloging all product images.
    
    This CSV serves as the single source of truth mapping:
    image_id → image_path → category → filename
    
    Args:
        data_dir: Root directory with category subfolders.
        output_path: Where to save the CSV. Defaults to Config.METADATA_CSV.
        validate_images: If True, verify each image is readable (slower but safer).
    
    Returns:
        DataFrame with columns: image_id, image_path, category, filename
    """
    if output_path is None:
        output_path = Config.METADATA_CSV

    # Scan directory
    records = scan_image_directory(data_dir)
    df = pd.DataFrame(records)

    if df.empty:
        logger.error("No images found. Cannot create metadata CSV.")
        raise ValueError("Empty dataset — no images to process.")

    # Validate images (filter out corrupted files)
    if validate_images:
        logger.info("Validating images (checking for corrupted files)...")
        valid_mask = []
        for path in tqdm(df["image_path"], desc="Validating images"):
            valid_mask.append(is_valid_image(path))

        num_invalid = len(valid_mask) - sum(valid_mask)
        if num_invalid > 0:
            logger.warning(f"Filtered out {num_invalid} corrupted/invalid images.")
        df = df[valid_mask].reset_index(drop=True)
        # Re-assign sequential image IDs
        df["image_id"] = range(len(df))

    # Save to CSV
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Metadata CSV saved: {output_path} ({len(df)} images)")

    # Print category distribution
    cat_counts = df["category"].value_counts()
    logger.info(f"Category distribution:\n{cat_counts.to_string()}")

    return df


# =========================================================================
# PYTORCH DATASET
# =========================================================================

def get_preprocessing_transforms() -> transforms.Compose:
    """
    Get the standard ImageNet preprocessing pipeline.
    
    This MUST match what ResNet50 was trained on:
    1. Resize to 256px (shortest edge)
    2. Center crop to 224x224
    3. Convert to tensor (scales pixels from [0,255] to [0.0,1.0])
    4. Normalize with ImageNet channel statistics
    
    Why these exact values?
    - ResNet50 was trained on ImageNet with these exact transforms.
    - Using different normalization would produce meaningless embeddings.
    - CenterCrop ensures consistent framing regardless of input aspect ratio.
    
    Returns:
        torchvision.transforms.Compose pipeline
    """
    return transforms.Compose([
        transforms.Resize(Config.RESIZE_DIM),
        transforms.CenterCrop(Config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=Config.IMAGENET_MEAN,
            std=Config.IMAGENET_STD,
        ),
    ])


class ProductImageDataset(Dataset):
    """
    PyTorch Dataset for loading product images with preprocessing.
    
    This dataset class enables efficient batch loading via DataLoader,
    which parallelizes image reading and preprocessing across multiple
    CPU workers — critical for processing 18K+ images.
    
    Args:
        metadata_df: DataFrame with 'image_path' and 'category' columns.
        transform: Image preprocessing pipeline. Defaults to ImageNet transforms.
    
    Example:
        >>> df = pd.read_csv("data/processed/metadata.csv")
        >>> dataset = ProductImageDataset(df)
        >>> dataloader = DataLoader(dataset, batch_size=64, num_workers=4)
        >>> for images, paths, categories in dataloader:
        ...     embeddings = model(images)  # (batch_size, 2048)
    """

    def __init__(
        self,
        metadata_df: pd.DataFrame,
        transform: Optional[transforms.Compose] = None,
    ):
        self.metadata = metadata_df.reset_index(drop=True)
        self.transform = transform or get_preprocessing_transforms()
        self.image_paths = self.metadata["image_path"].tolist()
        self.categories = self.metadata["category"].tolist()

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str, str]:
        """
        Load and preprocess a single image.
        
        Args:
            idx: Index into the metadata DataFrame.
        
        Returns:
            Tuple of (preprocessed_image_tensor, image_path, category)
        """
        image_path = self.image_paths[idx]
        category = self.categories[idx]

        try:
            # Open image and convert to RGB
            # Some product images may be RGBA (with transparency) or grayscale —
            # convert('RGB') handles all cases uniformly.
            image = Image.open(image_path).convert("RGB")

            if self.transform:
                image = self.transform(image)

            return image, image_path, category

        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            # Return a blank tensor as fallback (will be filtered later)
            blank = torch.zeros(3, Config.IMAGE_SIZE, Config.IMAGE_SIZE)
            return blank, image_path, category


def create_dataloader(
    metadata_df: pd.DataFrame,
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
    shuffle: bool = False,
) -> DataLoader:
    """
    Create a DataLoader for batch processing of product images.
    
    Args:
        metadata_df: DataFrame with image_path and category columns.
        batch_size: Images per batch. Defaults to Config.BATCH_SIZE.
        num_workers: Parallel workers. Defaults to Config.NUM_WORKERS.
        shuffle: Whether to shuffle images (False for embedding extraction).
    
    Returns:
        DataLoader yielding (images_batch, paths_batch, categories_batch)
    """
    dataset = ProductImageDataset(metadata_df)
    return DataLoader(
        dataset,
        batch_size=batch_size or Config.BATCH_SIZE,
        num_workers=num_workers or Config.NUM_WORKERS,
        shuffle=shuffle,
        pin_memory=torch.cuda.is_available(),  # Faster GPU transfer
        drop_last=False,
    )


# =========================================================================
# QUICK STATS
# =========================================================================

def print_dataset_stats(metadata_df: pd.DataFrame) -> None:
    """Print a formatted summary of the dataset."""
    print("\n" + "=" * 60)
    print("  Dataset Statistics")
    print("=" * 60)
    print(f"  Total images:    {len(metadata_df):,}")
    print(f"  Categories:      {metadata_df['category'].nunique()}")
    print(f"  Avg per category: {len(metadata_df) / metadata_df['category'].nunique():.0f}")
    print()
    print("  Category breakdown:")
    for cat, count in metadata_df["category"].value_counts().items():
        bar = "█" * min(int(count / len(metadata_df) * 40), 40)
        print(f"    {cat:<25s} {count:>5,}  {bar}")
    print("=" * 60 + "\n")


# =========================================================================
# MODULE TEST
# =========================================================================

if __name__ == "__main__":
    """Quick test: scan data directory and create metadata CSV."""
    logger.info("Testing data_loader module...")

    try:
        df = create_metadata_csv(validate_images=True)
        print_dataset_stats(df)
        logger.info("data_loader test passed!")
    except FileNotFoundError as e:
        logger.error(f"{e}")
        logger.info(
            "Place your product images in: "
            f"{Config.RAW_DATA_DIR}/<category_name>/<image_files>"
        )
