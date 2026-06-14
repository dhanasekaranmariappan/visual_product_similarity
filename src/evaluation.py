"""
evaluation.py — Retrieval Quality Evaluation Metrics
=====================================================
Evaluates how well the similarity search engine retrieves relevant products.

Ground truth: Two products are "relevant" if they share the same category label.
This is a standard proxy in visual retrieval — if the system retrieves a shoe
when queried with a shoe, it's doing well.

Metrics:
    Precision@K: Of the top-K results, what fraction are relevant?
    Recall@K:    Of ALL relevant items in the database, what fraction appear in top-K?
    MAP@K:       Mean Average Precision — rewards relevant items ranked higher.

Usage:
    evaluator = RetrievalEvaluator(search_engine)
    report = evaluator.evaluate(k_values=[1, 5, 10, 20])
    evaluator.print_report(report)
"""

import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import Config


class RetrievalEvaluator:
    """
    Evaluate the visual similarity search engine using standard IR metrics.

    Relevance definition: a retrieved product is "relevant" if its category
    matches the query product's category.
    """

    def __init__(self, search_engine):
        """
        Args:
            search_engine: An initialized SimilaritySearchEngine instance.
        """
        self.engine = search_engine
        self.image_paths = search_engine.image_paths
        self.categories = search_engine.categories

        # Precompute: for each category, how many images exist?
        self.category_counts = {}
        for cat in self.categories:
            self.category_counts[cat] = self.category_counts.get(cat, 0) + 1

    @staticmethod
    def precision_at_k(retrieved_categories: List[str], query_category: str, k: int) -> float:
        """
        Precision@K = (# relevant in top-K) / K

        Args:
            retrieved_categories: Categories of the top-K retrieved items.
            query_category: Category of the query item.
            k: Cutoff rank.

        Returns:
            Precision score between 0.0 and 1.0.
        """
        top_k = retrieved_categories[:k]
        relevant = sum(1 for cat in top_k if cat == query_category)
        return relevant / k if k > 0 else 0.0

    @staticmethod
    def recall_at_k(
        retrieved_categories: List[str],
        query_category: str,
        k: int,
        total_relevant: int,
    ) -> float:
        """
        Recall@K = (# relevant in top-K) / (total relevant in database)

        Args:
            total_relevant: Total number of same-category images in the database.
                           Subtract 1 if the query itself is in the database.
        """
        top_k = retrieved_categories[:k]
        relevant = sum(1 for cat in top_k if cat == query_category)
        # Subtract 1 from total_relevant because the query itself is relevant
        # but shouldn't count
        effective_total = max(total_relevant - 1, 1)
        return relevant / effective_total

    @staticmethod
    def average_precision_at_k(
        retrieved_categories: List[str], query_category: str, k: int
    ) -> float:
        """
        Average Precision@K — rewards relevant results appearing earlier.

        AP@K = (1/min(K, R)) * Σ(Precision@i * rel(i)) for i=1..K
        where R = total relevant items, rel(i) = 1 if item i is relevant.
        """
        top_k = retrieved_categories[:k]
        hits = 0
        sum_precision = 0.0

        for i, cat in enumerate(top_k):
            if cat == query_category:
                hits += 1
                sum_precision += hits / (i + 1)

        return sum_precision / min(k, max(hits, 1))

    def evaluate(
        self,
        k_values: Optional[List[int]] = None,
        num_queries: Optional[int] = None,
        show_progress: bool = True,
    ) -> Dict:
        """
        Run full evaluation across many query images.

        Args:
            k_values: List of K values to evaluate (e.g., [1, 5, 10, 20]).
            num_queries: Number of random query images. None = use all.
            show_progress: Show tqdm progress bar.

        Returns:
            Dict with:
                - overall: DataFrame of mean metrics per K
                - per_category: DataFrame of metrics per category per K
                - search_times: List of search latencies
                - num_queries: Total queries evaluated
        """
        k_values = k_values or Config.EVAL_K_VALUES
        max_k = max(k_values)

        # Select query images
        np.random.seed(Config.RANDOM_SEED)
        n_total = len(self.image_paths)
        if num_queries and num_queries < n_total:
            query_indices = np.random.choice(n_total, size=num_queries, replace=False)
        else:
            query_indices = np.arange(n_total)
            num_queries = n_total

        logger.info(f"Evaluating {len(query_indices)} queries with K={k_values}")

        # Collect metrics
        all_metrics = []
        search_times = []

        iterator = tqdm(query_indices, desc="Evaluating", disable=not show_progress)
        for idx in iterator:
            query_path = self.image_paths[idx]
            query_cat = self.categories[idx]
            total_relevant = self.category_counts[query_cat]

            # Search
            result = self.engine.search_by_index(int(idx), top_k=max_k)
            search_times.append(result["search_time_ms"])

            # Get retrieved categories
            retrieved_cats = [r["category"] for r in result["results"]]

            # Compute metrics for each K
            for k in k_values:
                p_at_k = self.precision_at_k(retrieved_cats, query_cat, k)
                r_at_k = self.recall_at_k(retrieved_cats, query_cat, k, total_relevant)
                ap_at_k = self.average_precision_at_k(retrieved_cats, query_cat, k)

                all_metrics.append({
                    "query_index": idx,
                    "query_category": query_cat,
                    "k": k,
                    "precision_at_k": p_at_k,
                    "recall_at_k": r_at_k,
                    "ap_at_k": ap_at_k,
                })

        metrics_df = pd.DataFrame(all_metrics)

        # Aggregate: overall means per K
        overall = metrics_df.groupby("k").agg({
            "precision_at_k": "mean",
            "recall_at_k": "mean",
            "ap_at_k": "mean",
        }).round(4)
        overall.columns = ["Mean_Precision@K", "Mean_Recall@K", "MAP@K"]

        # Per-category breakdown
        per_category = metrics_df.groupby(["query_category", "k"]).agg({
            "precision_at_k": "mean",
            "recall_at_k": "mean",
            "ap_at_k": "mean",
        }).round(4)

        avg_search_time = np.mean(search_times)

        return {
            "overall": overall,
            "per_category": per_category,
            "search_times": search_times,
            "avg_search_time_ms": round(avg_search_time, 2),
            "num_queries": len(query_indices),
            "raw_metrics": metrics_df,
        }

    @staticmethod
    def print_report(report: Dict) -> None:
        """Print a formatted evaluation report."""
        print("\n" + "=" * 65)
        print("  Retrieval Evaluation Report")
        print("=" * 65)

        print(f"\n  Queries evaluated:     {report['num_queries']}")
        print(f"  Avg search time:       {report['avg_search_time_ms']:.2f} ms")

        print("\n  ── Overall Metrics ──")
        print(report["overall"].to_string())

        print("\n  ── Per-Category Breakdown ──")
        print(report["per_category"].to_string())

        print("\n" + "=" * 65)


if __name__ == "__main__":
    from src.similarity_search import SimilaritySearchEngine

    logger.info("Testing evaluation module...")
    try:
        engine = SimilaritySearchEngine()
        evaluator = RetrievalEvaluator(engine)
        report = evaluator.evaluate(k_values=[1, 5, 10], num_queries=50)
        evaluator.print_report(report)
        logger.info("evaluation test passed!")
    except FileNotFoundError as e:
        logger.warning(f"Cannot test (index not built yet): {e}")
