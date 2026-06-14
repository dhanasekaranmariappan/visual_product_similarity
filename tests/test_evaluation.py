"""
test_evaluation.py — Tests for src/evaluation.py
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.evaluation import RetrievalEvaluator


class TestPrecisionAtK:
    """Test Precision@K metric calculation."""

    def test_perfect_precision(self):
        retrieved = ["shoes", "shoes", "shoes", "shoes", "shoes"]
        assert RetrievalEvaluator.precision_at_k(retrieved, "shoes", 5) == 1.0

    def test_zero_precision(self):
        retrieved = ["bags", "hats", "belts", "rings", "socks"]
        assert RetrievalEvaluator.precision_at_k(retrieved, "shoes", 5) == 0.0

    def test_partial_precision(self):
        retrieved = ["shoes", "bags", "shoes", "hats", "belts"]
        assert RetrievalEvaluator.precision_at_k(retrieved, "shoes", 5) == 0.4

    def test_precision_at_1(self):
        retrieved = ["shoes", "bags"]
        assert RetrievalEvaluator.precision_at_k(retrieved, "shoes", 1) == 1.0

    def test_precision_at_k_with_k_larger_than_results(self):
        retrieved = ["shoes", "shoes"]
        # k=5 but only 2 results — still divides by k
        result = RetrievalEvaluator.precision_at_k(retrieved, "shoes", 5)
        assert result == 2.0 / 5.0

    def test_precision_at_k_zero(self):
        assert RetrievalEvaluator.precision_at_k([], "shoes", 0) == 0.0


class TestRecallAtK:
    """Test Recall@K metric calculation."""

    def test_perfect_recall_small_category(self):
        # 3 total relevant (minus query = 2 effective), found 2 in top-5
        retrieved = ["shoes", "shoes", "bags", "hats", "belts"]
        recall = RetrievalEvaluator.recall_at_k(retrieved, "shoes", 5, total_relevant=3)
        assert recall == 1.0  # Found 2 out of effective 2

    def test_zero_recall(self):
        retrieved = ["bags", "hats", "belts"]
        recall = RetrievalEvaluator.recall_at_k(retrieved, "shoes", 3, total_relevant=10)
        assert recall == 0.0


class TestAveragePrecisionAtK:
    """Test AP@K metric calculation."""

    def test_perfect_ap(self):
        retrieved = ["shoes", "shoes", "shoes"]
        ap = RetrievalEvaluator.average_precision_at_k(retrieved, "shoes", 3)
        assert ap == 1.0

    def test_zero_ap(self):
        retrieved = ["bags", "hats", "belts"]
        ap = RetrievalEvaluator.average_precision_at_k(retrieved, "shoes", 3)
        assert ap == 0.0

    def test_ap_rewards_early_relevance(self):
        # Relevant at position 1 should score higher than relevant at position 3
        early = ["shoes", "bags", "bags"]
        late = ["bags", "bags", "shoes"]
        ap_early = RetrievalEvaluator.average_precision_at_k(early, "shoes", 3)
        ap_late = RetrievalEvaluator.average_precision_at_k(late, "shoes", 3)
        assert ap_early > ap_late
