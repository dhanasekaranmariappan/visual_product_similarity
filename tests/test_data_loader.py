"""
test_data_loader.py — Tests for src/data_loader.py
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import Config
from src.data_loader import (
    ProductImageDataset,
    get_preprocessing_transforms,
    is_valid_image,
)


class TestPreprocessingTransforms:
    """Verify the image preprocessing pipeline."""

    def test_returns_compose_object(self):
        transform = get_preprocessing_transforms()
        assert transform is not None

    def test_transform_produces_correct_shape(self):
        transform = get_preprocessing_transforms()
        img = Image.new("RGB", (300, 400), color="red")
        tensor = transform(img)
        assert tensor.shape == (3, Config.IMAGE_SIZE, Config.IMAGE_SIZE)

    def test_transform_produces_float_tensor(self):
        transform = get_preprocessing_transforms()
        img = Image.new("RGB", (300, 400), color="blue")
        tensor = transform(img)
        assert tensor.dtype == torch.float32

    def test_transform_handles_grayscale_via_convert(self):
        """Grayscale images should be converted to RGB before transform."""
        transform = get_preprocessing_transforms()
        img = Image.new("L", (300, 400), color=128).convert("RGB")
        tensor = transform(img)
        assert tensor.shape == (3, Config.IMAGE_SIZE, Config.IMAGE_SIZE)

    def test_transform_handles_rgba_via_convert(self):
        """RGBA images should be converted to RGB before transform."""
        transform = get_preprocessing_transforms()
        img = Image.new("RGBA", (300, 400), color=(255, 0, 0, 128)).convert("RGB")
        tensor = transform(img)
        assert tensor.shape == (3, Config.IMAGE_SIZE, Config.IMAGE_SIZE)


class TestImageValidation:
    """Test the is_valid_image function."""

    def test_valid_image(self, tmp_path):
        img_path = tmp_path / "valid.jpg"
        Image.new("RGB", (100, 100), color="green").save(img_path)
        assert is_valid_image(str(img_path)) is True

    def test_too_small_image(self, tmp_path):
        img_path = tmp_path / "tiny.jpg"
        Image.new("RGB", (5, 5), color="red").save(img_path)
        assert is_valid_image(str(img_path)) is False

    def test_nonexistent_image(self):
        assert is_valid_image("/nonexistent/fake_image.jpg") is False

    def test_corrupt_file(self, tmp_path):
        corrupt_path = tmp_path / "corrupt.jpg"
        corrupt_path.write_text("this is not an image")
        assert is_valid_image(str(corrupt_path)) is False


class TestProductImageDataset:
    """Test the PyTorch Dataset class."""

    @pytest.fixture
    def sample_dataset(self, tmp_path):
        """Create a small test dataset with real images."""
        import pandas as pd

        records = []
        for i in range(5):
            cat = "shoes" if i < 3 else "bags"
            img_path = tmp_path / f"{cat}_{i}.jpg"
            Image.new("RGB", (224, 224), color=(i * 50, 100, 200)).save(img_path)
            records.append({
                "image_id": i,
                "image_path": str(img_path),
                "category": cat,
                "filename": img_path.name,
            })

        df = pd.DataFrame(records)
        return ProductImageDataset(df)

    def test_dataset_length(self, sample_dataset):
        assert len(sample_dataset) == 5

    def test_getitem_returns_tuple(self, sample_dataset):
        result = sample_dataset[0]
        assert isinstance(result, tuple)
        assert len(result) == 3  # (tensor, path, category)

    def test_getitem_tensor_shape(self, sample_dataset):
        tensor, path, category = sample_dataset[0]
        assert tensor.shape == (3, Config.IMAGE_SIZE, Config.IMAGE_SIZE)

    def test_getitem_returns_correct_category(self, sample_dataset):
        _, _, category = sample_dataset[0]
        assert category == "shoes"
