"""
similarity_search.py — Visual Product Similarity Search Engine
===============================================================
The query engine that ties everything together: given an input image,
find the Top-K most visually similar products from the indexed database.

Pipeline:
    Query Image → ResNet50 → 2048-d embedding → L2 normalize → FAISS search
    → Top-K results (ranked by cosine similarity) → optional category filter

Usage:
    engine = SimilaritySearchEngine()
    results = engine.search(query_image_path, top_k=10)
    results = engine.search_pil(pil_image, top_k=5, category_filter="Shoes")
"""

import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

import faiss
import numpy as np
from loguru import logger
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import Config
from src.feature_extractor import ImageFeatureExtractor
from src.index_builder import FAISSIndexBuilder


class SimilaritySearchEngine:
    """
    End-to-end visual similarity search engine.

    Combines the feature extractor and FAISS index to provide
    a single search() method that accepts an image and returns
    ranked similar products.

    Attributes:
        extractor: ImageFeatureExtractor for converting images to embeddings.
        index: FAISS index containing all product embeddings.
        image_paths: List of file paths corresponding to index positions.
        categories: List of category labels corresponding to index positions.
    """

    def __init__(
        self,
        index_path: Optional[str] = None,
        image_paths_path: Optional[str] = None,
        categories_path: Optional[str] = None,
        extractor: Optional[ImageFeatureExtractor] = None,
    ):
        """
        Initialize the search engine by loading the FAISS index and metadata.

        Args:
            index_path: Path to saved FAISS index file.
            image_paths_path: Path to pickled list of image paths.
            categories_path: Path to pickled list of categories.
            extractor: Pre-initialized feature extractor (avoids reloading model).
        """
        # Load or create the feature extractor
        self.extractor = extractor or ImageFeatureExtractor()

        # Load FAISS index
        index_path = index_path or str(Config.FAISS_INDEX_PATH)
        self.index = FAISSIndexBuilder.load_index(index_path)

        # Load image paths mapping (index position → file path)
        img_paths_file = Path(image_paths_path or Config.IMAGE_PATHS_PATH)
        with open(img_paths_file, "rb") as f:
            self.image_paths = pickle.load(f)

        # Load categories mapping
        cat_file = Path(categories_path or Config.CATEGORIES_PATH)
        with open(cat_file, "rb") as f:
            self.categories = pickle.load(f)

        # Build category index for fast filtering
        self.unique_categories = sorted(set(self.categories))

        logger.info(
            f"Search engine ready: {self.index.ntotal} products, "
            f"{len(self.unique_categories)} categories"
        )

    def search(
        self,
        query_image_path: str,
        top_k: int = Config.DEFAULT_TOP_K,
        category_filter: Optional[str] = None,
    ) -> Dict:
        """
        Search for visually similar products given a query image file path.

        Args:
            query_image_path: Path to the query image.
            top_k: Number of similar products to return.
            category_filter: If set, only return results from this category.

        Returns:
            Dict with keys:
                - results: List of dicts with image_path, similarity, category, rank
                - query_path: The input image path
                - search_time_ms: Search latency in milliseconds
                - total_results: Number of results returned
        """
        query_embedding = self.extractor.extract_single(query_image_path)
        return self._search_with_embedding(query_embedding, top_k, category_filter,
                                           query_info=query_image_path)

    def search_pil(
        self,
        query_image: Image.Image,
        top_k: int = Config.DEFAULT_TOP_K,
        category_filter: Optional[str] = None,
    ) -> Dict:
        """Search using a PIL Image (for Gradio uploads)."""
        query_embedding = self.extractor.extract_from_pil(query_image)
        return self._search_with_embedding(query_embedding, top_k, category_filter,
                                           query_info="uploaded_image")

    def search_by_index(
        self,
        image_index: int,
        top_k: int = Config.DEFAULT_TOP_K,
        category_filter: Optional[str] = None,
    ) -> Dict:
        """
        Find products similar to an existing indexed product.
        This powers the "Similar Items" / "Customers also viewed" feature.
        """
        # Reconstruct the embedding from the FAISS index
        embedding = self.index.reconstruct(image_index).reshape(1, -1)
        faiss.normalize_L2(embedding)
        # +1 because the query itself will appear in results
        results = self._search_with_embedding(
            embedding.flatten(), top_k + 1, category_filter,
            query_info=self.image_paths[image_index]
        )
        # Remove the query image from results
        results["results"] = [
            r for r in results["results"]
            if r["image_path"] != self.image_paths[image_index]
        ][:top_k]
        results["total_results"] = len(results["results"])
        return results

    def _search_with_embedding(
        self,
        query_embedding: np.ndarray,
        top_k: int,
        category_filter: Optional[str],
        query_info: str = "",
    ) -> Dict:
        """Core search logic shared by all public search methods."""
        query_vec = query_embedding.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(query_vec)

        # If filtering by category, request more results to ensure we have
        # enough after filtering
        search_k = top_k * 5 if category_filter else top_k

        start_time = time.perf_counter()
        distances, indices = self.index.search(query_vec, min(search_k, self.index.ntotal))
        search_time_ms = (time.perf_counter() - start_time) * 1000

        # Build results list
        results = []
        for rank, (idx, score) in enumerate(zip(indices[0], distances[0])):
            if idx < 0:  # FAISS returns -1 for padding
                continue

            category = self.categories[idx]

            # Apply category filter
            if category_filter and category != category_filter:
                continue

            # Apply similarity threshold
            if score < Config.MIN_SIMILARITY_THRESHOLD:
                continue

            results.append({
                "image_path": self.image_paths[idx],
                "similarity": float(score),
                "category": category,
                "rank": len(results) + 1,
                "index": int(idx),
            })

            if len(results) >= top_k:
                break

        return {
            "results": results,
            "query_info": query_info,
            "search_time_ms": round(search_time_ms, 2),
            "total_results": len(results),
        }

    def get_categories(self) -> List[str]:
        """Return all available product categories."""
        return self.unique_categories

    def get_sample_images(self, n: int = Config.NUM_EXAMPLES) -> List[str]:
        """Get sample image paths for the Gradio examples gallery."""
        np.random.seed(Config.RANDOM_SEED)
        indices = np.random.choice(len(self.image_paths), size=min(n, len(self.image_paths)),
                                   replace=False)
        return [self.image_paths[i] for i in indices]


if __name__ == "__main__":
    logger.info("Testing similarity_search module...")
    try:
        engine = SimilaritySearchEngine()
        samples = engine.get_sample_images(3)
        for img in samples:
            result = engine.search(img, top_k=5)
            logger.info(
                f"Query: {Path(img).name} → "
                f"{result['total_results']} results in {result['search_time_ms']:.1f}ms"
            )
        logger.info("similarity_search test passed!")
    except FileNotFoundError as e:
        logger.warning(f"Cannot test (index not built yet): {e}")
