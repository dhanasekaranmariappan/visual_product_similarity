"""
index_builder.py — FAISS Vector Index Construction & Persistence
=================================================================
Builds a FAISS index from image embeddings for millisecond-level similarity search.

FAISS (Facebook AI Similarity Search) stores vectors in an optimized index
structure and retrieves nearest neighbors orders of magnitude faster than
brute-force numpy operations.

We use IndexFlatIP (Inner Product) with L2-normalized vectors:
    cosine_similarity(a, b) = dot(a, b) / (||a|| * ||b||)
    When ||a|| = ||b|| = 1 (L2-normalized), this simplifies to: dot(a, b)
    FAISS IndexFlatIP computes dot products → equivalent to cosine similarity.

Usage:
    builder = FAISSIndexBuilder()
    index = builder.build_index(embeddings)
    builder.save_index(index, "product_index.faiss")
    index = builder.load_index("product_index.faiss")
"""

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import faiss
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import Config


class FAISSIndexBuilder:
    """
    Build, save, and load FAISS similarity search indexes.

    For datasets <100K images, we use IndexFlatIP (exact search, no approximation).
    For larger datasets, this class can be extended to use IndexIVFFlat (ANN).

    Attributes:
        dimension: Embedding vector dimension (2048 for ResNet50).
    """

    def __init__(self, dimension: int = Config.EMBEDDING_DIM):
        self.dimension = dimension
        logger.info(f"FAISSIndexBuilder initialized (dimension={dimension})")

    def build_index(self, embeddings: np.ndarray) -> faiss.Index:
        """
        Build a FAISS index from a matrix of embeddings.

        Steps:
            1. Validate embeddings shape and dtype.
            2. L2-normalize vectors (ensures cosine similarity via dot product).
            3. Create IndexFlatIP and add all vectors.

        Args:
            embeddings: Array of shape (N, dimension), dtype float32.
                        Should already be L2-normalized, but we normalize
                        again for safety.

        Returns:
            faiss.Index ready for similarity search.
        """
        if embeddings.ndim != 2 or embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Expected shape (N, {self.dimension}), got {embeddings.shape}"
            )

        embeddings = embeddings.astype(np.float32)

        # L2-normalize (idempotent if already normalized)
        faiss.normalize_L2(embeddings)

        # IndexFlatIP = exact inner product search
        # With normalized vectors: inner product = cosine similarity
        index = faiss.IndexFlatIP(self.dimension)
        index.add(embeddings)

        logger.info(
            f"FAISS index built: {index.ntotal} vectors, "
            f"dimension={self.dimension}, type=IndexFlatIP"
        )
        return index

    @staticmethod
    def save_index(index: faiss.Index, path: Optional[str] = None) -> None:
        """Save FAISS index to disk for later reuse."""
        path = Path(path or Config.FAISS_INDEX_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(path))
        logger.info(f"FAISS index saved: {path}")

    @staticmethod
    def load_index(path: Optional[str] = None) -> faiss.Index:
        """Load a previously saved FAISS index from disk."""
        path = Path(path or Config.FAISS_INDEX_PATH)
        if not path.exists():
            raise FileNotFoundError(f"FAISS index not found: {path}")
        index = faiss.read_index(str(path))
        logger.info(f"FAISS index loaded: {index.ntotal} vectors from {path}")
        return index


if __name__ == "__main__":
    # Quick test with random vectors
    logger.info("Testing index_builder module...")
    builder = FAISSIndexBuilder(dimension=2048)
    fake_embeddings = np.random.randn(100, 2048).astype(np.float32)
    index = builder.build_index(fake_embeddings)

    # Test search
    query = np.random.randn(1, 2048).astype(np.float32)
    faiss.normalize_L2(query)
    distances, indices = index.search(query, 5)
    logger.info(f"Top-5 indices: {indices[0]}, scores: {distances[0]}")
    logger.info("index_builder test passed!")
