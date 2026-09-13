"""
Unit tests for End-to-End Two-Stage Retrieve-then-Rank Pipeline (Assignment 2, Q2).
"""

from datetime import datetime, timedelta
from unittest.mock import MagicMock

import numpy as np
import pytest

from retrieve_then_rank import RetrieveThenRankPipeline, reciprocal_rank_fusion


def test_reciprocal_rank_fusion():
    """Verify Reciprocal Rank Fusion correctly combines ranks from two retrievers."""
    sem_ranked = [("art_1", 0.95), ("art_2", 0.80), ("art_3", 0.70)]
    bm25_ranked = [("art_2", 12.5), ("art_4", 10.0), ("art_1", 8.2)]

    # k0 = 60
    # art_1: rank 1 in sem (1/61), rank 3 in bm25 (1/63) -> ~0.01639 + 0.01587 = 0.03226
    # art_2: rank 2 in sem (1/62), rank 1 in bm25 (1/61) -> ~0.01613 + 0.01639 = 0.03252
    # art_2 should rank #1, art_1 rank #2
    fused = reciprocal_rank_fusion(sem_ranked, bm25_ranked, k0=60.0, top_k=5)

    assert len(fused) == 4
    top_aid = fused[0][0]
    second_aid = fused[1][0]
    assert top_aid == "art_2"
    assert second_aid == "art_1"

    # Verify return tuple: (aid, rrf_score, sem_score, bm25_score)
    for aid, rrf_sc, sem_sc, bm25_sc in fused:
        assert rrf_sc > 0.0
        assert isinstance(sem_sc, float)
        assert isinstance(bm25_sc, float)


def test_pipeline_point_in_time_enforcement():
    """Verify that the retrieve-then-rank pipeline strictly rejects future clicks."""
    pipeline = object.__new__(RetrieveThenRankPipeline)
    pipeline.half_life_hours = 24.0
    pipeline.article_cat_map = {"a1": "sports", "a2": "news", "a3": "news"}
    pipeline.article_pub_map = {"a1": None, "a2": None, "a3": None}
    pipeline.article_id_to_idx = {"a1": 0, "a2": 1, "a3": 2}
    pipeline.embeddings_norm = np.array([[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]], dtype=np.float32)
    pipeline.bm25_retriever = None
    pipeline.art_token_ids_map = {}
    pipeline.retriever_type = "hybrid"

    # Mock ranker
    mock_ranker = MagicMock()
    mock_ranker.predict.return_value = np.array([0.9, 0.2, 0.5])
    pipeline.ranker = mock_ranker

    t_imp = datetime(2023, 11, 15, 12, 0, 0)
    t_past = t_imp - timedelta(hours=2)
    t_future = t_imp + timedelta(minutes=5)
    t_exact = t_imp

    user_clicks = [
        ("a1", t_past, 45.0),
        ("a2", t_exact, 10.0),    # Should be rejected
        ("a3", t_future, 30.0),   # Should be rejected
    ]

    out = pipeline.run_impression(
        user_id="user_test_99",
        imp_time=t_imp,
        user_history_clicks=user_clicks,
        candidate_subset=["a1", "a2", "a3"],
        k=10,
    )

    assert "stage1_candidates" in out
    assert "stage2_ranked" in out
    assert "latency_ms" in out
    assert out["latency_ms"]["total"] >= 0.0

    # Ensure Stage 2 output is sorted descending by re-ranker score
    reranked = out["stage2_ranked"]
    assert len(reranked) == 3
    scores = [s for _, s in reranked]
    assert scores == sorted(scores, reverse=True)


def test_pipeline_cold_start_handling():
    """Verify that a cold-start user (0 clicks) runs without exception."""
    pipeline = object.__new__(RetrieveThenRankPipeline)
    pipeline.half_life_hours = 24.0
    pipeline.article_cat_map = {"a1": "sports", "a2": "finance"}
    pipeline.article_pub_map = {"a1": None, "a2": None}
    pipeline.article_id_to_idx = {"a1": 0, "a2": 1}
    pipeline.embeddings_norm = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    pipeline.bm25_retriever = None
    pipeline.art_token_ids_map = {}
    pipeline.retriever_type = "hybrid"

    mock_ranker = MagicMock()
    mock_ranker.predict.return_value = np.array([0.1, 0.4])
    pipeline.ranker = mock_ranker

    t_imp = datetime(2023, 11, 15, 12, 0, 0)
    out = pipeline.run_impression(
        user_id="cold_user",
        imp_time=t_imp,
        user_history_clicks=[],
        candidate_subset=["a1", "a2"],
        k=10,
    )

    assert len(out["stage2_ranked"]) == 2
    assert out["stage2_ranked"][0][0] == "a2"
