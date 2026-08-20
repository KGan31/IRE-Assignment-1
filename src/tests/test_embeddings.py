import sys
from pathlib import Path

# Ensure src/ directory is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest

from embeddings import (
    EmbeddingIndex,
    compute_user_representation,
    normalize_l2,
)
from eval_embeddings import calculate_recall_at_k


def test_normalize_l2():
    vecs = np.array([
        [3.0, 4.0],
        [0.0, 5.0],
        [1.0, 1.0],
    ], dtype=np.float32)
    normed = normalize_l2(vecs)
    
    # Each row should have L2 norm close to 1.0
    row_norms = np.linalg.norm(normed, axis=1)
    np.testing.assert_allclose(row_norms, [1.0, 1.0, 1.0], rtol=1e-5)


def test_compute_user_representation():
    embeddings = np.array([
        [1.0, 0.0, 0.0],  # art_0
        [0.0, 1.0, 0.0],  # art_1
        [0.0, 0.0, 1.0],  # art_2
    ], dtype=np.float32)
    
    id_to_idx = {"art_0": 0, "art_1": 1, "art_2": 2}
    
    # 1. User clicked art_0 and art_1
    u_vec = compute_user_representation(["art_0", "art_1"], id_to_idx, embeddings)
    assert u_vec is not None
    expected = normalize_l2(np.array([0.5, 0.5, 0.0], dtype=np.float32))
    np.testing.assert_allclose(u_vec, expected, rtol=1e-5)
    
    # 2. Cold start user (no valid clicks)
    cold_vec = compute_user_representation([], id_to_idx, embeddings)
    assert cold_vec is None
    
    # 3. Weighted user representation
    weighted_vec = compute_user_representation(
        ["art_0", "art_1"], id_to_idx, embeddings, weights=[0.8, 0.2]
    )
    expected_w = normalize_l2(np.array([0.8, 0.2, 0.0], dtype=np.float32))
    np.testing.assert_allclose(weighted_vec, expected_w, rtol=1e-5)


def test_embedding_index_batch_search():
    embeddings = np.array([
        [1.0, 0.0, 0.0],  # art_0 (sports)
        [0.9, 0.1, 0.0],  # art_1 (sports)
        [0.0, 1.0, 0.0],  # art_2 (politics)
        [0.0, 0.0, 1.0],  # art_3 (tech)
    ], dtype=np.float32)
    article_ids = ["art_0", "art_1", "art_2", "art_3"]

    index = EmbeddingIndex()
    index.build_index(embeddings, article_ids)
    assert index.is_built

    # Query with sports vector [1, 0, 0]
    query_vec = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    results = index.batch_search(query_vec, top_k=2)

    assert len(results) == 1
    top_items = results[0]
    assert len(top_items) == 2
    assert top_items[0][0] == "art_0"
    assert top_items[1][0] == "art_1"


def test_anti_gaming_temporal_boundary_safety():
    """Verify that clicks after impression timestamp are strictly excluded."""
    user_clicks = [
        (pd.Timestamp("2023-01-01 10:00:00"), "art_1"),
        (pd.Timestamp("2023-01-01 12:00:00"), "art_2"),
        (pd.Timestamp("2023-01-01 15:00:00"), "art_3"),  # Future click
    ]
    impression_time = pd.Timestamp("2023-01-01 14:00:00")
    
    prior_clicks = [aid for c_time, aid in user_clicks if c_time < impression_time]
    assert "art_3" not in prior_clicks
    assert prior_clicks == ["art_1", "art_2"]
