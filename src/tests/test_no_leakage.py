"""
Anti-leakage tests (Q9): asserts the temporal split has no future-click
leakage, at two levels:

1. Split-level: train impressions all precede val impressions, which all
   precede test impressions (no timestamp overlap between splits).
2. Feature-level: user click-history features attached to a split only
   ever contain clicks that happened strictly before that split's
   earliest impression timestamp.

Run with:
    pytest src/tests/test_no_leakage.py -v
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from split import temporal_split, history_as_of  # noqa: E402


@pytest.fixture
def toy_impressions():
    return pd.DataFrame({
        "impression_id": [f"i{i}" for i in range(10)],
        "user_id": ["u1"] * 10,
        "timestamp": pd.date_range("2026-01-01", periods=10, freq="D"),
        "candidate_article_ids": [["a1", "a2"]] * 10,
        "clicked_article_ids": [["a1"]] * 10,
    })


@pytest.fixture
def toy_history():
    return pd.DataFrame({
        "user_id": ["u1"] * 5,
        "clicked_article_id": [f"a{i}" for i in range(5)],
        "click_time": pd.date_range("2026-01-01", periods=5, freq="D"),
    })


def test_split_has_no_timestamp_overlap(toy_impressions):
    train, val, test = temporal_split(toy_impressions, test_days=2, val_days=2)

    assert len(train) + len(val) + len(test) == len(toy_impressions)

    if len(train) and len(val):
        assert train["timestamp"].max() < val["timestamp"].min()
    if len(val) and len(test):
        assert val["timestamp"].max() < test["timestamp"].min()
    if len(train) and len(test):
        assert train["timestamp"].max() < test["timestamp"].min()


def test_no_duplicate_impressions_across_splits(toy_impressions):
    train, val, test = temporal_split(toy_impressions, test_days=2, val_days=2)
    ids = pd.concat([train["impression_id"], val["impression_id"], test["impression_id"]])
    assert ids.is_unique


def test_history_as_of_excludes_future_clicks(toy_history):
    cutoff = pd.Timestamp("2026-01-03")
    filtered = history_as_of(toy_history, cutoff)

    assert (filtered["click_time"] < cutoff).all()
    # sanity: at least the clicks strictly before cutoff must survive
    assert len(filtered) == (toy_history["click_time"] < cutoff).sum()


def test_user_features_do_not_leak_future_clicks(toy_history):
    """Simulates the feature_store.py logic: user features computed
    as_of a split's earliest impression must not include any click that
    happens at or after that timestamp."""
    as_of_ts = pd.Timestamp("2026-01-03")
    feats = history_as_of(toy_history, as_of_ts)

    leaked = feats[feats["click_time"] >= as_of_ts]
    assert leaked.empty, f"Leakage detected: {len(leaked)} clicks at/after cutoff"


def test_empty_impressions_raises():
    with pytest.raises(ValueError):
        temporal_split(pd.DataFrame(columns=["timestamp"]), test_days=1, val_days=1)
