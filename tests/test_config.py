"""
test_config.py — Tests for config.py
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import Config


class TestConfigPaths:
    """Verify all configured paths are valid Path objects."""

    def test_project_root_exists(self):
        assert Config.PROJECT_ROOT.exists()

    def test_project_root_is_directory(self):
        assert Config.PROJECT_ROOT.is_dir()

    def test_data_dir_is_path(self):
        assert isinstance(Config.DATA_DIR, Path)

    def test_output_dir_is_path(self):
        assert isinstance(Config.OUTPUT_DIR, Path)

    def test_faiss_index_path_has_faiss_extension(self):
        assert Config.FAISS_INDEX_PATH.suffix == ".faiss"

    def test_embeddings_path_has_npy_extension(self):
        assert Config.EMBEDDINGS_PATH.suffix == ".npy"

    def test_metadata_csv_has_csv_extension(self):
        assert Config.METADATA_CSV.suffix == ".csv"


class TestConfigModel:
    """Verify model configuration values are sensible."""

    def test_embedding_dim_positive(self):
        assert Config.EMBEDDING_DIM > 0

    def test_embedding_dim_is_2048_for_resnet50(self):
        if Config.MODEL_NAME == "resnet50":
            assert Config.EMBEDDING_DIM == 2048

    def test_image_size_is_224(self):
        assert Config.IMAGE_SIZE == 224

    def test_resize_dim_greater_than_image_size(self):
        assert Config.RESIZE_DIM >= Config.IMAGE_SIZE

    def test_imagenet_mean_has_3_channels(self):
        assert len(Config.IMAGENET_MEAN) == 3

    def test_imagenet_std_has_3_channels(self):
        assert len(Config.IMAGENET_STD) == 3

    def test_imagenet_mean_values_in_range(self):
        for val in Config.IMAGENET_MEAN:
            assert 0.0 < val < 1.0

    def test_imagenet_std_values_in_range(self):
        for val in Config.IMAGENET_STD:
            assert 0.0 < val < 1.0


class TestConfigSearch:
    """Verify search configuration values."""

    def test_default_top_k_positive(self):
        assert Config.DEFAULT_TOP_K > 0

    def test_max_top_k_gte_default(self):
        assert Config.MAX_TOP_K >= Config.DEFAULT_TOP_K

    def test_similarity_threshold_in_range(self):
        assert 0.0 <= Config.MIN_SIMILARITY_THRESHOLD <= 1.0

    def test_batch_size_positive(self):
        assert Config.BATCH_SIZE > 0

    def test_num_workers_non_negative(self):
        assert Config.NUM_WORKERS >= 0


class TestConfigUI:
    """Verify Gradio UI configuration."""

    def test_server_port_valid(self):
        assert 1024 <= Config.GRADIO_SERVER_PORT <= 65535

    def test_num_examples_positive(self):
        assert Config.NUM_EXAMPLES > 0

    def test_gallery_columns_positive(self):
        assert Config.GALLERY_COLUMNS > 0


class TestConfigMethods:
    """Test Config class methods."""

    def test_ensure_directories_creates_dirs(self):
        Config.ensure_directories()
        assert Config.RAW_DATA_DIR.exists()
        assert Config.PROCESSED_DATA_DIR.exists()
        assert Config.INDEX_DIR.exists()
        assert Config.EMBEDDINGS_DIR.exists()

    def test_supported_extensions_not_empty(self):
        assert len(Config.SUPPORTED_EXTENSIONS) > 0

    def test_supported_extensions_all_start_with_dot(self):
        for ext in Config.SUPPORTED_EXTENSIONS:
            assert ext.startswith(".")

    def test_eval_k_values_sorted(self):
        assert Config.EVAL_K_VALUES == sorted(Config.EVAL_K_VALUES)
