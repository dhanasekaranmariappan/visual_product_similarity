"""
build_index.py — End-to-End Pipeline: Images → Embeddings → FAISS Index
=========================================================================
CLI script that executes the full indexing pipeline:
    1. Scan data/raw/ for product images
    2. Generate metadata CSV
    3. Extract ResNet50 embeddings (GPU-accelerated)
    4. Build and save FAISS index
    5. Save image paths and categories for lookup

This only needs to run ONCE (or when new images are added).
After this, the Gradio app can load the pre-built index instantly.

Usage:
    python scripts/build_index.py
    python scripts/build_index.py --batch-size 32  # If GPU OOM
"""

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from loguru import logger

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import Config
from src.data_loader import create_metadata_csv, print_dataset_stats
from src.feature_extractor import ImageFeatureExtractor
from src.index_builder import FAISSIndexBuilder


def main(batch_size: int = None, validate: bool = True):
    """Run the full indexing pipeline."""
    pipeline_start = time.time()

    Config.ensure_directories()
    Config.summary()

    # ── Step 1: Generate Metadata ──────────────────────────────────────
    logger.info("Step 1/4: Scanning images and generating metadata...")
    step_start = time.time()

    metadata_df = create_metadata_csv(validate_images=validate)
    print_dataset_stats(metadata_df)

    image_paths = metadata_df["image_path"].tolist()
    categories = metadata_df["category"].tolist()

    logger.info(f"  → {len(image_paths)} images cataloged in {time.time() - step_start:.1f}s")

    # ── Step 2: Extract Embeddings ─────────────────────────────────────
    logger.info("Step 2/4: Extracting image embeddings with ResNet50...")
    step_start = time.time()

    extractor = ImageFeatureExtractor()
    bs = batch_size or Config.BATCH_SIZE
    embeddings = extractor.extract_batch(image_paths, batch_size=bs)

    logger.info(
        f"  → {embeddings.shape[0]} embeddings extracted "
        f"(shape={embeddings.shape}) in {time.time() - step_start:.1f}s"
    )

    # ── Step 3: Build FAISS Index ──────────────────────────────────────
    logger.info("Step 3/4: Building FAISS index...")
    step_start = time.time()

    builder = FAISSIndexBuilder(dimension=extractor.embedding_dim)
    index = builder.build_index(embeddings)
    builder.save_index(index)

    logger.info(f"  → FAISS index built in {time.time() - step_start:.1f}s")

    # ── Step 4: Save Metadata ──────────────────────────────────────────
    logger.info("Step 4/4: Saving metadata files...")

    # Save embeddings as numpy array (for potential re-indexing)
    np.save(str(Config.EMBEDDINGS_PATH), embeddings)
    logger.info(f"  → Embeddings saved: {Config.EMBEDDINGS_PATH}")

    # Save image paths list (maps FAISS index positions to file paths)
    with open(Config.IMAGE_PATHS_PATH, "wb") as f:
        pickle.dump(image_paths, f)
    logger.info(f"  → Image paths saved: {Config.IMAGE_PATHS_PATH}")

    # Save categories list (maps FAISS index positions to categories)
    with open(Config.CATEGORIES_PATH, "wb") as f:
        pickle.dump(categories, f)
    logger.info(f"  → Categories saved: {Config.CATEGORIES_PATH}")

    # ── Summary ────────────────────────────────────────────────────────
    total_time = time.time() - pipeline_start
    print("\n" + "=" * 60)
    print("  Indexing Pipeline Complete!")
    print("=" * 60)
    print(f"  Images processed:  {len(image_paths):,}")
    print(f"  Categories:        {len(set(categories))}")
    print(f"  Embedding dim:     {embeddings.shape[1]}")
    print(f"  FAISS index size:  {index.ntotal:,} vectors")
    print(f"  Total time:        {total_time:.1f}s")
    print(f"  Index file:        {Config.FAISS_INDEX_PATH}")
    print("=" * 60)
    print("\n  Ready! Launch the app with: python app.py\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build FAISS index from product images")
    parser.add_argument("--batch-size", type=int, default=None,
                        help=f"Batch size for embedding extraction (default: {Config.BATCH_SIZE})")
    parser.add_argument("--skip-validation", action="store_true",
                        help="Skip image validation (faster but may fail on corrupt images)")
    args = parser.parse_args()

    main(batch_size=args.batch_size, validate=not args.skip_validation)
