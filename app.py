"""
app.py — Visual Product Similarity Search (Gradio Application)
================================================================
Premium Gradio interface for Amazon-style visual product recommendation.

Features:
     Image upload → find visually similar products
     Category filtering
     Adjustable Top-K results
     Real-time search metrics (latency, precision)
     Interactive product gallery
     Detailed results table with similarity scores

Launch:
    python app.py
    python app.py --share    # Create public URL
"""

import sys
import os
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
from PIL import Image
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import Config
from src.similarity_search import SimilaritySearchEngine
from src.evaluation import RetrievalEvaluator


# =========================================================================
# CUSTOM CSS — Premium dark theme with glassmorphism
# =========================================================================

CUSTOM_CSS = """
/* ── Global Theme ── */
.gradio-container {
    font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif !important;
    max-width: 1400px !important;
    margin: auto !important;
}

/* ── Header Banner ── */
.header-banner {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    border-radius: 16px;
    padding: 32px 40px;
    margin-bottom: 24px;
    border: 1px solid rgba(255, 255, 255, 0.08);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
}

.header-banner h1 {
    color: #ffffff !important;
    font-size: 2rem !important;
    font-weight: 700 !important;
    margin: 0 0 8px 0 !important;
    letter-spacing: -0.02em;
}

.header-banner p {
    color: rgba(255, 255, 255, 0.7) !important;
    font-size: 1.05rem !important;
    margin: 0 !important;
    line-height: 1.6;
}

/* ── Stat Cards ── */
.stat-card {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.15), rgba(168, 85, 247, 0.1));
    border: 1px solid rgba(99, 102, 241, 0.2);
    border-radius: 12px;
    padding: 16px 20px;
    text-align: center;
}

.stat-card .stat-value {
    font-size: 1.8rem;
    font-weight: 700;
    color: #818cf8;
    display: block;
}

.stat-card .stat-label {
    font-size: 0.85rem;
    color: rgba(255, 255, 255, 0.5);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-top: 4px;
}

/* ── Gallery Styling ── */
.gallery-container .gallery-item {
    border-radius: 12px !important;
    transition: transform 0.2s ease, box-shadow 0.2s ease !important;
}

.gallery-container .gallery-item:hover {
    transform: translateY(-4px) !important;
    box-shadow: 0 12px 24px rgba(99, 102, 241, 0.2) !important;
}

/* ── Results Section ── */
.results-section {
    background: rgba(255, 255, 255, 0.02);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 16px;
    padding: 24px;
    margin-top: 16px;
}

/* ── Search Metrics Bar ── */
.metrics-bar {
    background: linear-gradient(90deg, rgba(16, 185, 129, 0.1), rgba(59, 130, 246, 0.1));
    border: 1px solid rgba(16, 185, 129, 0.2);
    border-radius: 10px;
    padding: 12px 20px;
    display: flex;
    gap: 24px;
    align-items: center;
    margin-bottom: 16px;
}

/* ── Buttons ── */
.primary-btn {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 12px 28px !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3) !important;
}

.primary-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(99, 102, 241, 0.4) !important;
}

/* ── Footer ── */
.footer {
    text-align: center;
    color: rgba(255, 255, 255, 0.3);
    font-size: 0.85rem;
    margin-top: 32px;
    padding: 16px;
    border-top: 1px solid rgba(255, 255, 255, 0.05);
}
"""

# Theme configuration (Gradio 6.0+: passed to launch())
CUSTOM_THEME = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="purple",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
)


# =========================================================================
# INITIALIZE SEARCH ENGINE (loaded once at startup)
# =========================================================================

def initialize_engine():
    """Load the search engine with pre-built FAISS index."""
    try:
        engine = SimilaritySearchEngine()
        logger.info("Search engine initialized successfully!")
        return engine
    except FileNotFoundError as e:
        logger.error(
            f"FAISS index not found: {e}\n"
            "Please run the indexing pipeline first:\n"
            "  1. Place images in data/raw/<category>/<images>\n"
            "  2. Run: python scripts/build_index.py\n"
            "  3. Then: python app.py"
        )
        return None


# =========================================================================
# SEARCH FUNCTIONS (called by Gradio callbacks)
# =========================================================================

def search_similar_products(
    query_image: Image.Image,
    top_k: int,
    category_filter: str,
    engine: SimilaritySearchEngine,
):
    """
    Main search function triggered by Gradio UI.

    Args:
        query_image: Uploaded PIL Image.
        top_k: Number of results.
        category_filter: Category to filter by ("All Categories" = no filter).
        engine: Pre-loaded search engine.

    Returns:
        Tuple of (gallery_images, results_markdown, metrics_markdown)
    """
    if query_image is None:
        return [], "Please upload an image to search.", ""

    # Apply category filter
    cat_filter = None if category_filter == "All Categories" else category_filter

    # Perform search
    result = engine.search_pil(query_image, top_k=int(top_k), category_filter=cat_filter)

    if not result["results"]:
        return [], "No similar products found. Try a different image or category.", ""

    # Build gallery: list of (image, caption) tuples
    gallery_images = []
    for r in result["results"]:
        try:
            img = Image.open(r["image_path"]).convert("RGB")
            caption = (
                f"#{r['rank']} | {r['similarity']:.1%} match\n"
                f"{r['category']}"
            )
            gallery_images.append((img, caption))
        except Exception as e:
            logger.warning(f"Failed to load result image: {r['image_path']}: {e}")

    # Build results table
    table_data = []
    for r in result["results"]:
        table_data.append({
            "Rank": r["rank"],
            "Similarity": f"{r['similarity']:.2%}",
            "Category": r["category"],
            "File": Path(r["image_path"]).name,
        })
    results_df = pd.DataFrame(table_data)
    table_md = results_df.to_markdown(index=False)

    # Build metrics display
    metrics_md = (
        f"### Search Results\n\n"
        f"| Metric | Value |\n"
        f"|--------|-------|\n"
        f"| Results Found | **{result['total_results']}** |\n"
        f"| Search Time | **{result['search_time_ms']:.1f} ms** |\n"
        f"| Best Match | **{result['results'][0]['similarity']:.2%}** |\n"
        f"| Top Category | **{result['results'][0]['category']}** |\n\n"
        f"---\n\n"
        f"#### Detailed Results\n\n{table_md}"
    )

    return gallery_images, metrics_md, ""


def run_evaluation(engine: SimilaritySearchEngine, num_queries: int):
    """Run evaluation and return formatted report."""
    if engine is None:
        return "Engine not initialized."

    evaluator = RetrievalEvaluator(engine)
    report = evaluator.evaluate(
        k_values=[1, 5, 10, 20],
        num_queries=int(num_queries),
        show_progress=True,
    )

    # Format report as markdown
    overall_md = report["overall"].to_markdown()

    md = (
        f"## Evaluation Report\n\n"
        f"**Queries evaluated:** {report['num_queries']}\n"
        f"**Avg search time:** {report['avg_search_time_ms']:.2f} ms\n\n"
        f"### Overall Metrics\n\n{overall_md}\n\n"
        f"### Per-Category Breakdown\n\n"
        f"{report['per_category'].to_markdown()}"
    )

    return md


# =========================================================================
# BUILD GRADIO UI
# =========================================================================

def create_app(engine: SimilaritySearchEngine):
    """Build the premium Gradio interface."""

    # Prepare data for UI
    categories = ["All Categories"] + engine.get_categories()
    sample_images = engine.get_sample_images(Config.NUM_EXAMPLES)
    total_products = engine.index.ntotal
    num_categories = len(engine.unique_categories)

    # Prepare examples: list of [image_path]
    examples = [[path] for path in sample_images]

    # ── Build Interface ──
    with gr.Blocks(title="Visual Product Similarity Search") as app:

        # ── Header ──
        gr.HTML(
            f"""
            <div class="header-banner">
                <h1>Visual Product Similarity Search</h1>
                <p>
                    Amazon-style image-based product recommendation system powered by
                    <strong>ResNet50</strong> deep learning embeddings and
                    <strong>FAISS</strong> vector search.
                    Upload any product image to find visually similar items instantly.
                </p>
            </div>
            """
        )

        # ── Stats Row ──
        with gr.Row():
            gr.HTML(f"""
                <div class="stat-card">
                    <span class="stat-value">{total_products:,}</span>
                    <span class="stat-label">Indexed Products</span>
                </div>
            """)
            gr.HTML(f"""
                <div class="stat-card">
                    <span class="stat-value">{num_categories}</span>
                    <span class="stat-label">Categories</span>
                </div>
            """)
            gr.HTML(f"""
                <div class="stat-card">
                    <span class="stat-value">2048</span>
                    <span class="stat-label">Embedding Dimensions</span>
                </div>
            """)
            gr.HTML(f"""
                <div class="stat-card">
                    <span class="stat-value">&lt;10ms</span>
                    <span class="stat-label">Search Latency</span>
                </div>
            """)

        gr.Markdown("---")

        # ── Search Section ──
        with gr.Tab("Search Similar Products"):
            with gr.Row():
                # Left: Query image upload
                with gr.Column(scale=1):
                    gr.Markdown("### Upload Query Image")
                    query_image = gr.Image(
                        type="pil",
                        label="Drop or click to upload a product image",
                        height=350,
                        sources=["upload", "clipboard"],
                    )

                    with gr.Row():
                        top_k_slider = gr.Slider(
                            minimum=1, maximum=Config.MAX_TOP_K,
                            value=Config.DEFAULT_TOP_K, step=1,
                            label="Top-K Results",
                            info="Number of similar products to retrieve",
                        )
                        category_dropdown = gr.Dropdown(
                            choices=categories,
                            value="All Categories",
                            label="Category Filter",
                            info="Filter results by product category",
                        )

                    search_btn = gr.Button(
                        "Find Similar Products",
                        variant="primary",
                        elem_classes=["primary-btn"],
                        size="lg",
                    )

                # Right: Results gallery
                with gr.Column(scale=2):
                    gr.Markdown("### Similar Products")
                    gallery = gr.Gallery(
                        label="Visually Similar Products",
                        columns=Config.GALLERY_COLUMNS,
                        height="auto",
                        object_fit="cover",
                        show_label=False,
                        elem_classes=["gallery-container"],
                    )

            # Metrics & Table
            with gr.Row():
                metrics_output = gr.Markdown(
                    value="*Upload an image and click 'Find Similar Products' to see results.*",
                    label="Search Results",
                )
                error_output = gr.Markdown(visible=False)

            # Examples
            if examples:
                gr.Markdown("### Try These Examples")
                gr.Examples(
                    examples=examples,
                    inputs=[query_image],
                    label="Click an example to search",
                )

        # ── Evaluation Tab ──
        with gr.Tab("Evaluation"):
            gr.Markdown(
                "### Model Evaluation\n"
                "Run Precision@K, Recall@K, and MAP@K evaluation across the dataset.\n"
                "This measures how well the system retrieves products from the same category."
            )
            with gr.Row():
                eval_queries = gr.Slider(
                    minimum=10, maximum=min(1000, total_products),
                    value=min(100, total_products),
                    step=10,
                    label="Number of Query Images",
                    info="More queries = more accurate but slower",
                )
                eval_btn = gr.Button("Run Evaluation", variant="primary", size="lg")
            eval_output = gr.Markdown(value="*Click 'Run Evaluation' to compute metrics.*")

            eval_btn.click(
                fn=lambda n: run_evaluation(engine, n),
                inputs=[eval_queries],
                outputs=[eval_output],
            )

        # ── About Tab ──
        with gr.Tab("About"):
            gr.Markdown(
                """
                ## How It Works

                This system uses **Computer Vision** and **Deep Learning** to find
                visually similar products — the same technology behind Amazon's
                "Similar Items" and image search features.

                ### Architecture

                ```
                Product Image → ResNet50 (CNN) → 2048-d Embedding → FAISS Index
                                                                        ↓
                Query Image  → ResNet50 (CNN) → 2048-d Embedding → Cosine Search
                                                                        ↓
                                                              Top-K Similar Products
                ```

                ### Tech Stack

                | Component | Technology | Purpose |
                |-----------|-----------|---------|
                | Feature Extraction | ResNet50 (PyTorch) | Convert images to embeddings |
                | Vector Search | FAISS (IndexFlatIP) | Millisecond similarity search |
                | Distance Metric | Cosine Similarity | Measure visual similarity |
                | Frontend | Gradio | Interactive web interface |

                ### Key Concepts

                - **Embeddings**: Dense vector representations that capture visual features
                  (color, texture, shape, spatial layout). Images that look similar have
                  embeddings that are close together in the 2048-dimensional space.

                - **FAISS**: Facebook's library for efficient similarity search.
                  Even with 100K+ products, search takes <10ms.

                - **Cosine Similarity**: Measures the angle between two embedding vectors.
                  Score of 1.0 = identical, 0.0 = completely different.
                """
            )

        # ── Footer ──
        gr.HTML(
            """
            <div class="footer">
                Visual Product Similarity Search Engine •
                Powered by PyTorch, FAISS & Gradio •
                ResNet50 Embeddings with Cosine Similarity
            </div>
            """
        )

        # ── Wire up search button ──
        search_btn.click(
            fn=lambda img, k, cat: search_similar_products(img, k, cat, engine),
            inputs=[query_image, top_k_slider, category_dropdown],
            outputs=[gallery, metrics_output, error_output],
        )

        # Auto-search on image upload
        query_image.change(
            fn=lambda img, k, cat: search_similar_products(img, k, cat, engine),
            inputs=[query_image, top_k_slider, category_dropdown],
            outputs=[gallery, metrics_output, error_output],
        )

    return app


# =========================================================================
# MAIN ENTRY POINT
# =========================================================================

def main():
    """Launch the Gradio application."""
    import argparse

    parser = argparse.ArgumentParser(description="Launch Visual Product Similarity Search")
    parser.add_argument("--share", action="store_true", help="Create public Gradio URL")
    parser.add_argument("--port", type=int, default=Config.GRADIO_SERVER_PORT)
    args = parser.parse_args()

    logger.info("Initializing Visual Product Similarity Search...")

    engine = initialize_engine()
    if engine is None:
        logger.error("Failed to initialize. Exiting.")
        sys.exit(1)

    app = create_app(engine)

    logger.info(f"Launching Gradio on port {args.port}...")
    app.launch(
        server_port=args.port,
        share=args.share or Config.GRADIO_SHARE,
        show_error=True,
        css=CUSTOM_CSS,
        theme=CUSTOM_THEME,
    )


if __name__ == "__main__":
    main()
