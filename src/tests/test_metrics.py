import sys
from pathlib import Path

# Ensure src/ directory is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from metrics import (
    compute_auc,
    compute_bootstrap_ci,
    compute_catalog_coverage,
    compute_intra_list_diversity,
    compute_mrr,
    compute_ndcg_at_k,
    compute_novelty,
    compute_recall_at_k,
)


def test_compute_auc_perfect_and_reversed():
    # Perfect ranking: positive has score 0.9, negative has 0.1
    labels = [1, 0]
    scores = [0.9, 0.1]
    assert compute_auc(labels, scores) == 1.0

    # Reversed ranking: positive has score 0.1, negative has 0.9
    labels = [1, 0]
    scores = [0.1, 0.9]
    assert compute_auc(labels, scores) == 0.0

    # Tied ranking: both have equal score -> 0.5
    labels = [1, 0]
    scores = [0.5, 0.5]
    assert compute_auc(labels, scores) == 0.5


def test_compute_auc_multi_item():
    labels = [1, 0, 1, 0]
    scores = [0.8, 0.7, 0.6, 0.2]
    # Positives are at ranks 4 and 2 (scores 0.8 and 0.6)
    # Pairs (P1, N1) = (0.8 > 0.7) -> 1
    # Pairs (P1, N2) = (0.8 > 0.2) -> 1
    # Pairs (P2, N1) = (0.6 < 0.7) -> 0
    # Pairs (P2, N2) = (0.6 > 0.2) -> 1
    # AUC = (1 + 1 + 0 + 1) / (2 * 2) = 3/4 = 0.75
    assert pytest.approx(compute_auc(labels, scores), 1e-5) == 0.75


def test_compute_auc_edge_cases():
    # Only positives or only negatives
    assert compute_auc([1, 1, 1], [0.5, 0.4, 0.3]) is None
    assert compute_auc([0, 0, 0], [0.5, 0.4, 0.3]) is None


def test_compute_mrr():
    # First positive at rank 1 -> 1.0
    assert compute_mrr([1, 0, 0]) == 1.0
    # First positive at rank 2 -> 0.5
    assert compute_mrr([0, 1, 0]) == 0.5
    # First positive at rank 3 -> 1/3
    assert pytest.approx(compute_mrr([0, 0, 1]), 1e-5) == 1.0 / 3.0
    # No positives -> 0.0
    assert compute_mrr([0, 0, 0]) == 0.0

    # Test with ranked_indices
    labels = [0, 1, 0]  # item 1 is positive
    ranked_indices = [1, 0, 2]  # item 1 placed at position 0 (rank 1)
    assert compute_mrr(labels, ranked_indices) == 1.0


def test_compute_ndcg_at_k():
    # Perfect top-2 ranking
    labels = [1, 1, 0, 0]
    assert pytest.approx(compute_ndcg_at_k(labels, k=5), 1e-5) == 1.0

    # Reversed top-2 ranking: [0, 0, 1, 1]
    labels_rev = [0, 0, 1, 1]
    # DCG@5 = 1/log2(4) + 1/log2(5) = 1/2 + 1/2.321928 = 0.5 + 0.430676 = 0.930676
    # IDCG@5 = 1/log2(2) + 1/log2(3) = 1 + 0.630929 = 1.630929
    # nDCG = 0.930676 / 1.630929 ~= 0.5706
    dcg = (1.0 / np.log2(4)) + (1.0 / np.log2(5))
    idcg = (1.0 / np.log2(2)) + (1.0 / np.log2(3))
    expected = dcg / idcg
    assert pytest.approx(compute_ndcg_at_k(labels_rev, k=5), 1e-4) == expected

    # Edge case: all zeros
    assert compute_ndcg_at_k([0, 0, 0], k=5) == 0.0


def test_compute_recall_at_k():
    retrieved = ["a1", "a2", "a3", "a4", "a5"]
    gt = {"a2", "a5", "a6"}
    # Top 3: ["a1", "a2", "a3"] -> intersection is {"a2"} -> 1 / 3
    assert pytest.approx(compute_recall_at_k(retrieved, gt, k=3), 1e-5) == 1.0 / 3.0
    # Top 5: ["a1", "a2", "a3", "a4", "a5"] -> intersection is {"a2", "a5"} -> 2 / 3
    assert pytest.approx(compute_recall_at_k(retrieved, gt, k=5), 1e-5) == 2.0 / 3.0
    # Empty gt
    assert compute_recall_at_k(retrieved, set(), k=5) == 0.0


def test_compute_intra_list_diversity():
    # 3 orthogonal items -> pairwise cosine dist = 1.0, ILD = 1.0
    embeddings = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    item_to_idx = {"a1": 0, "a2": 1, "a3": 2}
    ranked_ids = ["a1", "a2", "a3"]
    ild = compute_intra_list_diversity(ranked_ids, embeddings, item_to_idx, k=3)
    assert pytest.approx(ild, 1e-5) == 1.0

    # 3 identical items -> pairwise cosine dist = 0.0, ILD = 0.0
    identical_embeddings = np.array([
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
    ])
    ild_identical = compute_intra_list_diversity(ranked_ids, identical_embeddings, item_to_idx, k=3)
    assert pytest.approx(ild_identical, 1e-5) == 0.0


def test_compute_novelty():
    # Popularity prob: a1 is 0.5 (surprise 1 bit), a2 is 0.25 (surprise 2 bits)
    pop = {"a1": 0.5, "a2": 0.25}
    ranked_ids = ["a1", "a2"]
    # Mean novelty = (1 + 2) / 2 = 1.5 bits
    assert pytest.approx(compute_novelty(ranked_ids, pop, k=2), 1e-5) == 1.5


def test_compute_catalog_coverage():
    recs = [
        ["a1", "a2", "a3"],
        ["a2", "a4", "a5"],
    ]
    # Unique top-2 items: {"a1", "a2", "a4"} -> 3 unique items
    # Total catalog size = 10
    # Coverage@2 = 3 / 10 = 0.3
    assert pytest.approx(compute_catalog_coverage(recs, total_catalog_size=10, k=2), 1e-5) == 0.3


def test_compute_bootstrap_ci():
    scores = [0.8, 0.82, 0.79, 0.81, 0.83, 0.78, 0.80]
    mean, lower, upper = compute_bootstrap_ci(scores, n_bootstraps=200, ci=0.95, seed=42)
    assert lower <= mean <= upper
    assert pytest.approx(mean, 1e-4) == np.mean(scores)
