import sys
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import polars as pl
import pytest

# Ensure src/ and scripts/experiments/ directories are in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))
sys.path.insert(0, str(ROOT_DIR / "scripts" / "experiments"))

from eval_harness import OfflineEvaluationHarness
from metrics import (
    compute_auc,
    compute_bootstrap_ci,
    compute_catalog_coverage,
    compute_intra_list_diversity,
    compute_mrr,
    compute_ndcg_at_k,
    compute_novelty,
)
from run_q2_extended_eval import compute_user_history_lengths, identify_head_articles


def test_identify_head_articles():
    # 5 articles with different click frequencies
    h_df = pl.DataFrame({
        "clicked_article_id": ["A", "A", "A", "A", "B", "B", "C", "D", "E"]
    })
    # Top 20% of 5 unique articles is 1 article ("A")
    head = identify_head_articles(h_df, head_percentile=0.20)
    assert "A" in head
    assert len(head) == 1


def test_compute_user_history_lengths_leak_free():
    t0 = datetime(2023, 5, 20, 12, 0, 0)
    history_df = pl.DataFrame({
        "user_id": ["u1", "u1", "u1", "u2"],
        "click_time": [
            t0 - timedelta(hours=2),  # Valid past click
            t0 - timedelta(hours=1),  # Valid past click
            t0 + timedelta(hours=1),  # FUTURE click (must be excluded!)
            t0 - timedelta(hours=5),  # u2 past click
        ],
    })
    impressions_df = pl.DataFrame({
        "user_id": ["u1", "u2", "u3"],
        "timestamp": [t0, t0, t0],
    })

    lengths = compute_user_history_lengths(impressions_df, history_df)
    assert lengths[0] == 2  # Only 2 past clicks for u1
    assert lengths[1] == 1  # 1 past click for u2
    assert lengths[2] == 0  # 0 clicks for u3 (cold start)


def test_offline_harness_all_seven_metrics_and_slices():
    catalog_size = 20
    # Dummy embeddings: 20 items, dim=8
    np.random.seed(42)
    embs = np.random.randn(catalog_size, 8).astype(np.float32)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    embs_norm = embs / norms
    id_to_idx = {f"art_{i}": i for i in range(catalog_size)}
    pop_prob = {f"art_{i}": 1.0 / catalog_size for i in range(catalog_size)}

    harness = OfflineEvaluationHarness(
        embeddings=embs_norm,
        article_id_to_idx=id_to_idx,
        item_popularity_prob=pop_prob,
        total_catalog_size=catalog_size,
        n_bootstraps=200,
        ci=0.95,
        cold_start_threshold=1,
    )

    cands = [
        ["art_0", "art_1", "art_2", "art_3", "art_4"],
        ["art_5", "art_6", "art_7", "art_8", "art_9"],
    ]
    labels = [
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
    ]
    scores = [
        [0.9, 0.4, 0.3, 0.2, 0.1],
        [0.1, 0.8, 0.3, 0.2, 0.1],
    ]
    user_hist_lens = [0, 5]  # Imp 0 is cold, Imp 1 is warm
    is_head = [True, False]  # Imp 0 is head, Imp 1 is tail

    results = harness.evaluate_impression_ranking(
        candidate_article_lists=cands,
        candidate_labels=labels,
        candidate_scores=scores,
        user_history_lengths=user_hist_lens,
        is_head_flags=is_head,
    )

    # 1. Assert all 7 metrics exist in results
    expected_metrics = ["AUC", "MRR", "nDCG@5", "nDCG@10", "ILD@10", "Novelty@10", "Coverage@10"]
    for m in expected_metrics:
        assert m in results["metrics"], f"Metric {m} missing from results"
        val = results["metrics"][m]["mean"]
        ci_low = results["metrics"][m]["ci_lower"]
        ci_up = results["metrics"][m]["ci_upper"]
        assert not np.isnan(val)
        assert ci_low <= val <= ci_up or np.isclose(ci_low, val)

    # 2. Assert User Slices exist
    user_slices = results["slices"]["user_cohort"]
    assert user_slices["cold_count"] == 1
    assert user_slices["warm_count"] == 1
    for m in ["AUC", "MRR", "nDCG@5", "nDCG@10", "ILD@10", "Novelty@10", "Coverage@10"]:
        assert f"Cold_{m}" in user_slices
        assert f"Warm_{m}" in user_slices

    # 3. Assert Item Slices exist
    item_slices = results["slices"]["item_popularity"]
    assert item_slices["head_count"] == 1
    assert item_slices["tail_count"] == 1
    for m in ["AUC", "MRR", "nDCG@5", "nDCG@10", "ILD@10", "Novelty@10"]:
        assert f"Head_{m}" in item_slices
        assert f"Tail_{m}" in item_slices
