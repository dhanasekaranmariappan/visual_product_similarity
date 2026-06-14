"""
download_data.py — Dataset Download & Setup
=============================================
Helper script to download a product image dataset for the project.

This script provides multiple options:
    1. Use existing images in data/raw/ (user-provided)
    2. Generate a small demo dataset using torchvision's built-in data

If you already have images, simply place them in:
    data/raw/<category_name>/<image_files>

Usage:
    python scripts/download_data.py
    python scripts/download_data.py --demo    # Create a small demo dataset
"""

import argparse
import shutil
import sys
from pathlib import Path

from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import Config


def check_existing_data() -> bool:
    """Check if product images already exist in data/raw/."""
    raw_dir = Config.RAW_DATA_DIR
    if not raw_dir.exists():
        return False

    image_count = sum(
        1 for f in raw_dir.rglob("*")
        if f.is_file() and f.suffix.lower() in Config.SUPPORTED_EXTENSIONS
    )

    if image_count > 0:
        logger.info(f"Found {image_count} existing images in {raw_dir}")
        return True
    return False


def create_demo_dataset(num_per_category: int = 50):
    """
    Create a small demo dataset using CIFAR-10 images.
    
    This is ONLY for testing the pipeline when no real product images
    are available. For real results, use actual product images.
    """
    import numpy as np
    from PIL import Image
    from torchvision import datasets

    logger.info("Creating demo dataset from CIFAR-10 (for pipeline testing only)...")

    # Selected categories that loosely map to "products"
    cifar_categories = {
        0: "airplane", 1: "automobile", 2: "bird", 3: "cat", 4: "deer",
        5: "dog", 6: "frog", 7: "horse", 8: "ship", 9: "truck"
    }

    # Download CIFAR-10
    dataset = datasets.CIFAR10(
        root=str(Config.DATA_DIR / "cifar_temp"),
        train=True,
        download=True,
    )

    # Save images organized by category
    counts = {cat: 0 for cat in cifar_categories.values()}

    for img_array, label in dataset:
        category = cifar_categories[label]
        if counts[category] >= num_per_category:
            continue

        cat_dir = Config.RAW_DATA_DIR / category
        cat_dir.mkdir(parents=True, exist_ok=True)

        img = img_array  # Already PIL Image from CIFAR
        if not isinstance(img, Image.Image):
            img = Image.fromarray(np.array(img))

        # Upscale from 32x32 to 224x224 for ResNet compatibility
        img = img.resize((224, 224), Image.LANCZOS)

        img_path = cat_dir / f"{category}_{counts[category]:04d}.jpg"
        img.save(img_path, "JPEG", quality=95)
        counts[category] += 1

        if all(c >= num_per_category for c in counts.values()):
            break

    # Cleanup temp download
    temp_dir = Config.DATA_DIR / "cifar_temp"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)

    total = sum(counts.values())
    logger.info(f"Created demo dataset: {total} images across {len(counts)} categories")
    logger.info(f"Location: {Config.RAW_DATA_DIR}")
    logger.info("This is a demo dataset. For real results, use product images.")

    return total


def main():
    """Main entry point for data setup."""
    parser = argparse.ArgumentParser(description="Download or setup product image dataset")
    parser.add_argument("--demo", action="store_true",
                        help="Create a demo dataset using CIFAR-10 for testing")
    parser.add_argument("--num-per-category", type=int, default=50,
                        help="Number of images per category for demo dataset")
    args = parser.parse_args()

    Config.ensure_directories()

    # Check for existing data
    if check_existing_data():
        logger.info("Existing images found! Skipping download.")
        logger.info(f"   Location: {Config.RAW_DATA_DIR}")
        logger.info("   Run 'python scripts/build_index.py' to build the FAISS index.")
        return

    if args.demo:
        create_demo_dataset(num_per_category=args.num_per_category)
        logger.info("\nNext step: python scripts/build_index.py")
        return

    # No data found — provide instructions
    print("\n" + "=" * 60)
    print("  No product images found!")
    print("=" * 60)
    print(f"\n  Please add your product images to:\n  {Config.RAW_DATA_DIR}/")
    print("\n  Expected structure:")
    print("    data/raw/")
    print("    ├── Category_1/")
    print("    │   ├── image_001.jpg")
    print("    │   └── image_002.png")
    print("    ├── Category_2/")
    print("    │   ├── image_101.jpg")
    print("    │   └── ...")
    print("    └── ...")
    print(f"\n  Supported formats: {', '.join(Config.SUPPORTED_EXTENSIONS)}")
    print("\n  Or create a demo dataset for testing:")
    print("    python scripts/download_data.py --demo")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
