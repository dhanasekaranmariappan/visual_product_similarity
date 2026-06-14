"""
feature_extractor.py — Deep Learning Image Embedding Extraction
================================================================
Uses pretrained ResNet50 to convert product images into 2048-d vectors.

Usage:
    extractor = ImageFeatureExtractor()
    embedding = extractor.extract_single("path/to/image.jpg")  # (2048,)
    embeddings = extractor.extract_batch(image_paths)           # (N, 2048)
    embedding = extractor.extract_from_pil(pil_image)           # (2048,)
"""

import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from loguru import logger
from tqdm import tqdm
from torchvision import models
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import Config
from src.data_loader import ProductImageDataset, get_preprocessing_transforms


class ImageFeatureExtractor:
    """
    Extract visual embeddings from product images using a pretrained CNN.

    Wraps ResNet50 with classification head removed → outputs L2-normalized
    2048-d feature vectors. Similar-looking products have nearby embeddings.
    """

    def __init__(self, model_name: str = Config.MODEL_NAME, device=None):
        self.device = device or Config.DEVICE
        self.model_name = model_name
        self.transform = get_preprocessing_transforms()

        if model_name == "resnet50":
            self.model = models.resnet50(weights="IMAGENET1K_V2")
            # Replace classification head with identity → pass through 2048-d vector
            self.model.fc = nn.Identity()
            self.embedding_dim = 2048
        elif model_name == "efficientnet_b0":
            self.model = models.efficientnet_b0(weights="IMAGENET1K_V1")
            self.model.classifier = nn.Identity()
            self.embedding_dim = 1280
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        self.model.eval()
        self.model.to(self.device)
        logger.info(f"Loaded {model_name} on {self.device} (dim={self.embedding_dim})")

    def _preprocess_pil(self, image: Image.Image) -> torch.Tensor:
        """Preprocess PIL Image → (1, 3, 224, 224) tensor."""
        image = image.convert("RGB")
        return self.transform(image).unsqueeze(0)

    @staticmethod
    def _l2_normalize(vec: np.ndarray) -> np.ndarray:
        """L2-normalize so cosine similarity = dot product."""
        norm = np.linalg.norm(vec, axis=-1, keepdims=True)
        return vec / np.where(norm > 0, norm, 1.0)

    @torch.no_grad()
    def extract_single(self, image_path: str) -> np.ndarray:
        """Extract L2-normalized embedding for one image file → (2048,)."""
        image = Image.open(image_path).convert("RGB")
        tensor = self._preprocess_pil(image).to(self.device)
        emb = self.model(tensor).cpu().numpy().flatten()
        return self._l2_normalize(emb.reshape(1, -1)).flatten()

    @torch.no_grad()
    def extract_from_pil(self, image: Image.Image) -> np.ndarray:
        """Extract embedding from PIL Image (for Gradio uploads) → (2048,)."""
        tensor = self._preprocess_pil(image).to(self.device)
        emb = self.model(tensor).cpu().numpy().flatten()
        return self._l2_normalize(emb.reshape(1, -1)).flatten()

    @torch.no_grad()
    def extract_batch(self, image_paths: List[str], batch_size=None, show_progress=True) -> np.ndarray:
        """
        Batch-extract embeddings for many images using DataLoader.
        Returns L2-normalized array of shape (N, embedding_dim).
        """
        batch_size = batch_size or Config.BATCH_SIZE
        df = pd.DataFrame({"image_path": image_paths, "category": ["x"] * len(image_paths)})
        dataset = ProductImageDataset(df, transform=self.transform)
        loader = DataLoader(dataset, batch_size=batch_size, num_workers=Config.NUM_WORKERS,
                            shuffle=False, pin_memory=True, drop_last=False)

        all_embs = []
        for imgs, _, _ in tqdm(loader, desc="Extracting embeddings", disable=not show_progress):
            embs = self.model(imgs.to(self.device)).cpu().numpy()
            if embs.ndim > 2:
                embs = embs.reshape(embs.shape[0], -1)
            all_embs.append(embs)

        result = np.vstack(all_embs).astype(np.float32)
        result = self._l2_normalize(result)
        logger.info(f"Extracted {result.shape[0]} embeddings (dim={result.shape[1]})")
        return result


if __name__ == "__main__":
    import glob
    Config.summary()
    extractor = ImageFeatureExtractor()
    test_imgs = [p for p in glob.glob(str(Config.RAW_DATA_DIR / "**/*.*"), recursive=True)
                 if Path(p).suffix.lower() in Config.SUPPORTED_EXTENSIONS]
    if test_imgs:
        emb = extractor.extract_single(test_imgs[0])
        logger.info(f"Shape: {emb.shape}, L2 norm: {np.linalg.norm(emb):.4f}")
        logger.info("feature_extractor test passed!")
    else:
        logger.warning(f"No images in {Config.RAW_DATA_DIR}")
