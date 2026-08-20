"""
Unit tests for feature_store.py using Polars.
"""

from datetime import datetime, timedelta
from pathlib import Path
import sys
import polars as pl
import pytest

# Ensure src/ directory is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from feature_store import build_user_features


def test_build_user_features_aggregation():
    """Verify recency-decay scoring and click history list aggregation."""
    as_of_ts = datetime(2024, 1, 5, 0, 0, 0)
    history = pl.DataFrame({
        "dataset": ["mind", "mind", "mind"],
        "user_id": ["u1", "u1", "u2"],
        "clicked_article_id": ["a1", "a2", "b1"],
        # a1 clicked 24h ago (weight = 0.5), a2 clicked 48h ago (weight = 0.25)
        "click_time": [
            as_of_ts - timedelta(hours=24),
            as_of_ts - timedelta(hours=48),
            as_of_ts - timedelta(hours=24),
        ],
    })

    feats = build_user_features(history, as_of_ts, half_life_hours=24.0)

    assert len(feats) == 2
    u1_row = feats.filter(pl.col("user_id") == "u1")
    assert u1_row["n_clicks"][0] == 2
    assert set(u1_row["click_history"][0]) == {"a1", "a2"}
    assert pytest.approx(u1_row["recency_score"][0], rel=1e-3) == 0.75

    u2_row = feats.filter(pl.col("user_id") == "u2")
    assert u2_row["n_clicks"][0] == 1
    assert pytest.approx(u2_row["recency_score"][0], rel=1e-3) == 0.5


def test_build_user_features_empty():
    """Verify empty input returns proper schema."""
    as_of_ts = datetime(2024, 1, 5, 0, 0, 0)
    history = pl.DataFrame(schema={
        "dataset": pl.Utf8,
        "user_id": pl.Utf8,
        "clicked_article_id": pl.Utf8,
        "click_time": pl.Datetime,
    })

    feats = build_user_features(history, as_of_ts)
    assert feats.is_empty()
    assert set(feats.columns) == {"user_id", "click_history", "n_clicks", "recency_score"}
