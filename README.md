# Visual Product Similarity & Image-Based Recommendation System

> **Amazon-style** visual product search engine powered by Deep Learning embeddings and FAISS vector search.

Upload any product image → Get visually similar items in milliseconds.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Quick Start](#quick-start)
- [How It Works](#how-it-works)
- [Evaluation](#evaluation)
- [Future Improvements](#future-improvements)

---

## Overview

Online marketplaces host millions of products where textual metadata is often noisy, incomplete, or misleading. Traditional keyword-based search fails when users want **visually similar products** (same style, color, or design).

This system solves that by using **Computer Vision** to:
- Extract rich visual features from product images using a pretrained **ResNet50** CNN
- Store features in a **FAISS** vector index for millisecond-level similarity search
- Present results through a premium **Gradio** web interface

### Business Use Cases
- **Image-based search** — Upload a photo, find matching products
- **"Similar Items"** — Amazon's "Customers who viewed this also viewed"
- **Increase conversion** — Recommend visually relevant alternatives
- **Metadata-free discovery** — Works even without product descriptions

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                    INDEXING PIPELINE (Run Once)                  │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Product Images → ResNet50 → 2048-d Embedding → L2 Normalize     │
│       (N images)    (CNN)      (feature vector)                  │
│                                         ↓                        │
│                                   FAISS Index                    │
│                                  (IndexFlatIP)                   │
│                                                                  │
├──────────────────────────────────────────────────────────────────┤
│                    SEARCH PIPELINE (Real-time)                   │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Query Image → ResNet50 → Query Embedding → FAISS Search         │
│                  (same)     (2048-d)          (cosine sim)       │
│                                                  ↓               │
│                                         Top-K Similar Products   │
│                                                  ↓               │
│                                          Gradio UI (Gallery)     │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Component | Technology | Why? |
|-----------|-----------|------|
| Deep Learning | **PyTorch** + **torchvision** | Load pretrained ResNet50 model |
| Feature Extraction | **ResNet50** (ImageNet weights) | 2048-d embeddings capturing visual features |
| Vector Search | **FAISS** (IndexFlatIP) | Millisecond-level similarity search |
| Distance Metric | **Cosine Similarity** | Measures visual similarity between embeddings |
| Frontend | **Gradio** | Interactive web UI with image upload/gallery |
| Data Processing | **pandas** + **numpy** | Metadata management and array operations |
| Evaluation | **Precision@K, Recall@K, MAP@K** | Standard information retrieval metrics |

---

## Project Structure

```
visual_product_similarity/
├── README.md                       # This file
├── requirements.txt                # Dependencies (annotated with explanations)
├── config.py                       # Centralized configuration
├── app.py                          # Gradio application (main UI)
├── data/
│   ├── raw/                        # Place your product images here
│   │   ├── Category_1/
│   │   │   ├── image_001.jpg
│   │   │   └── ...
│   │   └── Category_2/
│   │       └── ...
│   └── processed/
│       └── metadata.csv            # Auto-generated image catalog
├── src/
│   ├── __init__.py
│   ├── data_loader.py              # Dataset loading & preprocessing
│   ├── feature_extractor.py        # ResNet50 embedding extraction
│   ├── index_builder.py            # FAISS index construction
│   ├── similarity_search.py        # Query engine (search + ranking)
│   └── evaluation.py               # Precision@K, Recall@K metrics
├── scripts/
│   ├── download_data.py            # Dataset setup helper
│   └── build_index.py              # Build FAISS index from images
├── output/
│   ├── index/                      # Saved FAISS index
│   ├── embeddings/                 # Saved embedding vectors
│   └── plots/                      # Evaluation charts
└── logs/
    └── pipeline.log                # Execution logs
```

---

## Setup & Installation

### Prerequisites
- Python 3.9+
- NVIDIA GPU (recommended — RTX 4050 or better)
- CUDA toolkit (for GPU acceleration)

### Step 1: Install Dependencies

```bash
# (Optional) Install PyTorch with CUDA support first
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install all other dependencies
cd visual_product_similarity
pip install -r requirements.txt
```

### Step 2: Add Your Product Images

Place your product images in `data/raw/` organized by category:

```
data/raw/
├── Shoes/
│   ├── shoe_001.jpg
│   ├── shoe_002.png
│   └── ...
├── Bags/
│   ├── bag_001.jpg
│   └── ...
├── Electronics/
│   └── ...
└── ...
```

**Supported formats:** `.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`, `.tiff`

> **No images yet?** Create a demo dataset: `python scripts/download_data.py --demo`

### Step 3: Build the FAISS Index

```bash
python scripts/build_index.py
```

This will:
1. Scan all images and create a metadata catalog
2. Extract 2048-d embeddings using ResNet50 (GPU-accelerated)
3. Build and save the FAISS similarity index

### Step 4: Launch the App

```bash
python app.py
```

Open your browser at `http://localhost:7860`

---

## Quick Start

```bash
# Clone and setup
cd visual_product_similarity
pip install -r requirements.txt

# Add images to data/raw/<category>/<images> OR use demo data
python scripts/download_data.py --demo

# Build the search index
python scripts/build_index.py

# Launch the visual search app
python app.py
```

---

## How It Works

### 1. Feature Extraction

```python
from src.feature_extractor import ImageFeatureExtractor

extractor = ImageFeatureExtractor()  # Loads ResNet50 on GPU

# Extract a 2048-dimensional embedding from any image
embedding = extractor.extract_single("product.jpg")  # → numpy array (2048,)
```

- ResNet50's final classification layer is replaced with `nn.Identity()`
- The model outputs the 2048-d vector from the global average pooling layer
- Vectors are L2-normalized for cosine similarity compatibility

### 2. FAISS Indexing

```python
from src.index_builder import FAISSIndexBuilder

builder = FAISSIndexBuilder(dimension=2048)
index = builder.build_index(all_embeddings)  # Shape: (N, 2048)
builder.save_index(index, "product_index.faiss")
```

- Uses `IndexFlatIP` (Inner Product) with L2-normalized vectors
- Inner product on unit vectors = cosine similarity
- Exact search, optimal for datasets <100K images

### 3. Similarity Search

```python
from src.similarity_search import SimilaritySearchEngine

engine = SimilaritySearchEngine()
results = engine.search("query_image.jpg", top_k=10)

for r in results["results"]:
    print(f"#{r['rank']}: {r['similarity']:.2%} — {r['category']}")
```

---

## Evaluation

The system is evaluated using standard Information Retrieval metrics:

| Metric | Description |
|--------|------------|
| **Precision@K** | % of top-K results that are from the same category |
| **Recall@K** | % of same-category items captured in the top-K |
| **MAP@K** | Mean Average Precision — rewards relevant items ranked higher |

Run evaluation:

```bash
python -m src.evaluation
```

Or use the **Evaluation tab** in the Gradio UI.

---

## Future Improvements

- [ ] **EfficientNet / CLIP** — Swap ResNet50 for more modern architectures
- [ ] **Fine-tuning** — Train on domain-specific product images for better accuracy
- [ ] **Text + Image** — Multimodal search combining text queries with image similarity
- [ ] **IndexIVFFlat** — Approximate nearest neighbor for million-scale datasets
- [ ] **Price/availability filtering** — Integrate product metadata
- [ ] **User feedback loop** — Learn from click-through data to improve rankings

---

## License

This project is for educational and portfolio purposes.

---

*Built with PyTorch, FAISS & Gradio*
