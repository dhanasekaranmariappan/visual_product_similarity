"""
config.py — Centralized Configuration for Visual Product Similarity System
===========================================================================
All tunable parameters, file paths, and model settings are defined here.
This avoids scattering magic numbers and hardcoded paths across the codebase.

Usage:
    from config import Config
    print(Config.EMBEDDING_DIM)    # 2048
    print(Config.DEVICE)           # cuda or cpu
"""

import os
from pathlib import Path

import torch


class Config:
    """
    Central configuration class for the Visual Product Similarity System.
    
    All settings are class-level constants — no instantiation needed.
    Modify values here to tune the entire pipeline from one place.
    """

    # =====================================================================
    # PROJECT PATHS
    # =====================================================================
    # Root directory of the project (where this file lives)
    PROJECT_ROOT = Path(__file__).resolve().parent

    # Data directories
    DATA_DIR = PROJECT_ROOT / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"           # Original product images
    PROCESSED_DATA_DIR = DATA_DIR / "processed"  # Generated metadata CSVs

    # Output / artifacts directories
    OUTPUT_DIR = PROJECT_ROOT / "output"
    INDEX_DIR = OUTPUT_DIR / "index"           # FAISS index files
    EMBEDDINGS_DIR = OUTPUT_DIR / "embeddings" # Saved embedding arrays
    PLOTS_DIR = OUTPUT_DIR / "plots"           # Evaluation charts

    # Specific file paths
    METADATA_CSV = PROCESSED_DATA_DIR / "metadata.csv"
    FAISS_INDEX_PATH = INDEX_DIR / "product_index.faiss"
    EMBEDDINGS_PATH = EMBEDDINGS_DIR / "embeddings.npy"
    IMAGE_PATHS_PATH = EMBEDDINGS_DIR / "image_paths.pkl"
    CATEGORIES_PATH = EMBEDDINGS_DIR / "categories.pkl"

    # =====================================================================
    # DEVICE CONFIGURATION
    # =====================================================================
    # Automatically detect CUDA GPU (e.g., RTX 4050) for acceleration.
    # Falls back to CPU if no GPU is available.
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # =====================================================================
    # MODEL CONFIGURATION
    # =====================================================================
    # Pretrained CNN model for feature extraction
    MODEL_NAME = "resnet50"

    # Dimension of the embedding vector produced by the model.
    # ResNet50 outputs 2048-d after the average pooling layer.
    # EfficientNet-B0 would output 1280-d.
    EMBEDDING_DIM = 2048

    # =====================================================================
    # IMAGE PREPROCESSING
    # =====================================================================
    # Input image size expected by ResNet50 (trained on 224x224 ImageNet images)
    IMAGE_SIZE = 224

    # Resize dimension before center-cropping (standard practice)
    RESIZE_DIM = 256

    # ImageNet normalization statistics (mean and std per RGB channel).
    # These MUST match what the pretrained model was trained with.
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    # Supported image file extensions
    SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}

    # =====================================================================
    # BATCH PROCESSING
    # =====================================================================
    # Batch size for embedding extraction.
    # With RTX 4050 (6GB VRAM), 64 is safe for ResNet50 at 224x224.
    # Reduce to 32 if you encounter CUDA out-of-memory errors.
    BATCH_SIZE = 64

    # Number of DataLoader workers for parallel image loading.
    # Set to 0 for debugging (sequential loading).
    NUM_WORKERS = 4

    # =====================================================================
    # FAISS / SIMILARITY SEARCH
    # =====================================================================
    # Default number of similar products to retrieve
    DEFAULT_TOP_K = 10

    # Maximum allowed Top-K (UI slider upper bound)
    MAX_TOP_K = 50

    # Minimum similarity score threshold (0.0 to 1.0 for cosine similarity).
    # Results below this threshold are filtered out.
    MIN_SIMILARITY_THRESHOLD = 0.0

    # =====================================================================
    # EVALUATION
    # =====================================================================
    # K values to evaluate Precision@K and Recall@K
    EVAL_K_VALUES = [1, 3, 5, 10, 20]

    # Number of random query images to use for evaluation.
    # Set to None to evaluate on ALL images (slower but comprehensive).
    EVAL_NUM_QUERIES = 500

    # Random seed for reproducibility
    RANDOM_SEED = 42

    # =====================================================================
    # GRADIO UI
    # =====================================================================
    # Server configuration
    GRADIO_SERVER_PORT = 7860
    GRADIO_SHARE = False  # Set True to create a public Gradio link

    # Number of example images to show in the UI
    NUM_EXAMPLES = 8

    # Gallery columns for displaying results
    GALLERY_COLUMNS = 5

    # =====================================================================
    # LOGGING
    # =====================================================================
    LOG_LEVEL = "INFO"
    LOG_FILE = PROJECT_ROOT / "logs" / "pipeline.log"

    @classmethod
    def ensure_directories(cls):
        """Create all necessary output directories if they don't exist."""
        dirs_to_create = [
            cls.RAW_DATA_DIR,
            cls.PROCESSED_DATA_DIR,
            cls.OUTPUT_DIR,
            cls.INDEX_DIR,
            cls.EMBEDDINGS_DIR,
            cls.PLOTS_DIR,
            cls.LOG_FILE.parent,  # logs/
        ]
        for directory in dirs_to_create:
            directory.mkdir(parents=True, exist_ok=True)

    @classmethod
    def summary(cls):
        """Print a summary of the current configuration."""
        print("=" * 60)
        print("  Visual Product Similarity — Configuration")
        print("=" * 60)
        print(f"  Device:          {cls.DEVICE}")
        print(f"  Model:           {cls.MODEL_NAME}")
        print(f"  Embedding Dim:   {cls.EMBEDDING_DIM}")
        print(f"  Image Size:      {cls.IMAGE_SIZE}x{cls.IMAGE_SIZE}")
        print(f"  Batch Size:      {cls.BATCH_SIZE}")
        print(f"  Default Top-K:   {cls.DEFAULT_TOP_K}")
        print(f"  Data Dir:        {cls.RAW_DATA_DIR}")
        print(f"  Index Path:      {cls.FAISS_INDEX_PATH}")
        print(f"  CUDA Available:  {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  GPU Name:        {torch.cuda.get_device_name(0)}")
            props = torch.cuda.get_device_properties(0)
            gpu_mem = getattr(props, 'total_memory', getattr(props, 'total_mem', 0)) / (1024**3)
            print(f"  GPU Memory:      {gpu_mem:.1f} GB")
        print("=" * 60)


# Auto-create directories on import
Config.ensure_directories()
