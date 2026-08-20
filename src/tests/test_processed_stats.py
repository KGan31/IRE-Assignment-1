"""
Unit tests for processed_stats.py analytics functions.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pytest

# Ensure root directory is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from processed_stats import (
    get_series_stats,
    compute_temporal_stats,
    analyze_split_impressions,
    safe_to_list,
)


def test_safe_to_list():
    assert safe_to_list(None) == []
    assert safe_to_list([]) == []
    assert safe_to_list(np.array([1, 2, 3])) == [1, 2, 3]
    assert safe_to_list(["a", "b"]) == ["a", "b"]


def test_get_series_stats():
    data = [10, 20, 30, 40, 50]
    stats = get_series_stats(data)
    assert stats["min"] == 10
    assert stats["max"] == 50
    assert stats["mean"] == 30.0
    assert stats["median"] == 30.0
    assert "std" in stats

    empty_stats = get_series_stats([])
    assert empty_stats["mean"] == 0.0


def test_analyze_split_impressions():
    df = pd.DataFrame({
        "impression_id": ["imp1", "imp2"],
        "user_id": ["u1", "u2"],
        "candidate_article_ids": [["a1", "a2", "a3"], ["a4", "a5"]],
        "clicked_article_ids": [["a1"], ["a5"]],
    })
    res = analyze_split_impressions(df)
    assert res["impression_count"] == 2
    assert res["unique_users"] == 2
    assert res["total_candidates"] == 5
    assert res["total_clicks"] == 2
    assert res["ctr_percent"] == 40.0
    assert res["candidates_per_imp"]["mean"] == 2.5
    assert res["clicks_per_imp"]["mean"] == 1.0


def test_compute_temporal_stats():
    train_df = pd.DataFrame({"timestamp": [pd.Timestamp("2023-01-01 00:00:00"), pd.Timestamp("2023-01-05 00:00:00")]})
    val_df = pd.DataFrame({"timestamp": [pd.Timestamp("2023-01-05 01:00:00"), pd.Timestamp("2023-01-06 00:00:00")]})
    test_df = pd.DataFrame({"timestamp": [pd.Timestamp("2023-01-06 01:00:00"), pd.Timestamp("2023-01-07 00:00:00")]})

    imp_dfs = {"train": train_df, "val": val_df, "test": test_df}
    res = compute_temporal_stats(imp_dfs)

    assert res["train"]["duration_days"] == 4.0
    assert res["val"]["duration_days"] == 0.96
    assert res["combined"]["train_to_val_gap_hours"] == 1.0
    assert res["combined"]["val_to_test_gap_hours"] == 1.0
