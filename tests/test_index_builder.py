"""
test_index_builder.py — Tests for src/index_builder.py
"""
import sys
from pathlib import Path

import numpy as np
import faiss
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.index_builder import FAISSIndexBuilder


class TestFAISSIndexBuilder:
    """Test FAISS index construction, saving, and loading."""

    @pytest.fixture
    def builder(self):
        return FAISSIndexBuilder(dimension=128)

    @pytest.fixture
    def random_embeddings(self):
        """Generate normalized random embeddings for testing."""
        emb = np.random.randn(50, 128).astype(np.float32)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        return emb / norms

    def test_build_index_returns_faiss_index(self, builder, random_embeddings):
        index = builder.build_index(random_embeddings)
        assert isinstance(index, faiss.Index)

    def test_index_contains_all_vectors(self, builder, random_embeddings):
        index = builder.build_index(random_embeddings)
        assert index.ntotal == 50

    def test_index_dimension_matches(self, builder, random_embeddings):
        index = builder.build_index(random_embeddings)
        assert index.d == 128

    def test_search_returns_correct_k(self, builder, random_embeddings):
        index = builder.build_index(random_embeddings)
        query = np.random.randn(1, 128).astype(np.float32)
        faiss.normalize_L2(query)
        distances, indices = index.search(query, 5)
        assert indices.shape == (1, 5)
        assert distances.shape == (1, 5)

    def test_search_self_similarity_is_high(self, builder, random_embeddings):
        """Searching for an indexed vector should return itself with high similarity."""
        index = builder.build_index(random_embeddings)
        query = random_embeddings[0:1].copy()
        faiss.normalize_L2(query)
        distances, indices = index.search(query, 1)
        assert distances[0][0] > 0.99  # Near-perfect self-match

    def test_save_and_load_index(self, builder, random_embeddings, tmp_path):
        index = builder.build_index(random_embeddings)
        save_path = tmp_path / "test_index.faiss"
        builder.save_index(index, str(save_path))

        assert save_path.exists()

        loaded_index = FAISSIndexBuilder.load_index(str(save_path))
        assert loaded_index.ntotal == index.ntotal
        assert loaded_index.d == index.d

    def test_load_nonexistent_index_raises(self):
        with pytest.raises(FileNotFoundError):
            FAISSIndexBuilder.load_index("/nonexistent/path.faiss")

    def test_build_index_wrong_dimension_raises(self, builder):
        wrong_dim = np.random.randn(10, 256).astype(np.float32)
        with pytest.raises(ValueError):
            builder.build_index(wrong_dim)

    def test_build_index_1d_raises(self, builder):
        wrong_shape = np.random.randn(128).astype(np.float32)
        with pytest.raises(ValueError):
            builder.build_index(wrong_shape)
