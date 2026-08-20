import sys
from pathlib import Path

# Ensure src/ directory is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest

from eval_harness import OfflineEvaluationHarness


def test_harness_retrieval_evaluation():
    articles_df = pd.DataFrame({
        "article_id": ["a1", "a2", "a3", "a4", "a5", "a6", "a7", "a8", "a9", "a10"],
        "title": [f"Article {i}" for i in range(1, 11)],
    })
    history_df = pd.DataFrame({
        "user_id": ["u1", "u1", "u2"],
        "clicked_article_id": ["a1", "a2", "a1"],
    })

    embeddings = np.eye(10, dtype=np.float32)
    article_id_to_idx = {f"a{i}": i - 1 for i in range(1, 11)}

    harness = OfflineEvaluationHarness.from_data(
        articles_df=articles_df,
        history_df=history_df,
        embeddings=embeddings,
        article_id_to_idx=article_id_to_idx,
        n_bootstraps=50,
    )

    retrieved_lists = [
        ["a1", "a2", "a3", "a4", "a5"],
        ["a6", "a7", "a8", "a9", "a10"],
    ]
    ground_truth_sets = [
        {"a1", "a3"},
        {"a6"},
    ]
    user_history_lengths = [2, 10]
    is_head_flags = [True, False]

    results = harness.evaluate_retrieval(
        retrieved_lists=retrieved_lists,
        ground_truth_sets=ground_truth_sets,
        user_history_lengths=user_history_lengths,
        is_head_flags=is_head_flags,
        k_list=(2, 5),
        beyond_k=3,
    )

    assert results["n_impressions"] == 2
    assert "Recall@2" in results["metrics"]
    assert "Recall@5" in results["metrics"]
    assert "ILD@3" in results["metrics"]
    assert "Novelty@3" in results["metrics"]
    assert "Coverage@3" in results["metrics"]
    assert "user_cohort" in results["slices"]
    assert "item_popularity" in results["slices"]


def test_harness_impression_ranking_evaluation():
    articles_df = pd.DataFrame({"article_id": ["a1", "a2", "a3"]})
    harness = OfflineEvaluationHarness.from_data(articles_df=articles_df, n_bootstraps=50)

    candidate_article_lists = [
        ["a1", "a2", "a3"],
        ["a1", "a2", "a3"],
    ]
    candidate_labels = [
        [1, 0, 0],
        [0, 1, 0],
    ]
    candidate_scores = [
        [0.9, 0.5, 0.1],  # rank order: [0, 1, 2] -> label [1, 0, 0] -> AUC=1.0, MRR=1.0, nDCG=1.0
        [0.1, 0.9, 0.2],  # rank order: [1, 2, 0] -> label [1, 0, 0] -> AUC=1.0, MRR=1.0, nDCG=1.0
    ]

    results = harness.evaluate_impression_ranking(
        candidate_article_lists=candidate_article_lists,
        candidate_labels=candidate_labels,
        candidate_scores=candidate_scores,
        user_history_lengths=[1, 6],
    )

    assert results["n_impressions"] == 2
    assert results["metrics"]["AUC"]["mean"] == 1.0
    assert results["metrics"]["MRR"]["mean"] == 1.0
    assert results["metrics"]["nDCG@5"]["mean"] == 1.0
