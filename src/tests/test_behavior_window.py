"""
Unit tests for Behavioural-Window Boundary Enforcement (Q1.4 & Q9 Anti-Gaming).

Asserts that no future clicks leak into features at training or serving time:
1. Exact timestamp boundary enforcement (click_time < impression_time strictly).
2. Training-time sequential impressions boundary (future clicks/impressions never leak into past impressions).
3. Serving-time boundary (events logged after serving_time are strictly excluded).
4. Point-in-time article popularity window (clicks occurring after impression_time never inflate popularity).
5. Target leakage immunity (clicked items in the current impression cannot leak into the impression's own input features).
"""

from datetime import datetime, timedelta
from pathlib import Path
import sys
import polars as pl
import pytest

# Ensure src/ is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from features import (
    filter_history_point_in_time,
    compute_user_behavior_features,
    compute_article_features,
    compute_cross_features,
)


def test_behavior_window_boundary_exact_and_future_exclusion():
    """
    Assert that clicks occurring at exactly cutoff_ts or after cutoff_ts
    are strictly excluded from the user's history window.
    """
    cutoff_ts = datetime(2024, 1, 10, 14, 0, 0)

    history = pl.DataFrame({
        "user_id": ["u1", "u1", "u1", "u1"],
        "clicked_article_id": ["past_1", "past_2", "boundary_exact", "future_1"],
        "click_time": [
            cutoff_ts - timedelta(hours=2),    # Valid past
            cutoff_ts - timedelta(seconds=1),  # Valid past (1 sec before)
            cutoff_ts,                         # EXACT boundary (MUST be excluded)
            cutoff_ts + timedelta(seconds=1),  # Future (MUST be excluded)
        ],
    })

    filtered = filter_history_point_in_time(history, impression_time=cutoff_ts)

    # Assert strictly prior
    assert len(filtered) == 2
    clicked_articles = filtered["clicked_article_id"].to_list()
    assert "past_1" in clicked_articles
    assert "past_2" in clicked_articles
    assert "boundary_exact" not in clicked_articles, "Leakage: click at exact impression time was included!"
    assert "future_1" not in clicked_articles, "Leakage: future click was included!"
    assert (filtered["click_time"] < cutoff_ts).all(), "Leakage: found click_time >= cutoff_ts!"


def test_no_future_clicks_leak_across_training_impressions():
    """
    Simulate sequential impressions for a user over multiple days during training.
    Verify that features for earlier impressions never see clicks that occurred in later sessions.
    """
    t1 = datetime(2024, 1, 1, 10, 0, 0)
    t2 = datetime(2024, 1, 2, 10, 0, 0)
    t3 = datetime(2024, 1, 3, 10, 0, 0)

    history = pl.DataFrame({
        "user_id": ["u1", "u1"],
        "clicked_article_id": ["art_click_day1", "art_click_day2"],
        "click_time": [
            datetime(2024, 1, 1, 10, 30, 0),  # After Imp 1, before Imp 2
            datetime(2024, 1, 2, 11, 0, 0),   # After Imp 2, before Imp 3
        ],
    })

    articles = pl.DataFrame({
        "article_id": ["art_click_day1", "art_click_day2"],
        "category": ["sports", "technology"],
    })

    # 1. Features at Impression 1 (t1 = Jan 1, 10:00)
    feats_t1 = compute_user_behavior_features(history, impression_time=t1, user_id="u1", articles_df=articles)
    assert feats_t1["click_count"] == 0
    assert feats_t1["recency_score"] == 0.0
    assert feats_t1["recent_clicked_ids"] == []
    assert feats_t1["category_affinity"] == {}

    # 2. Features at Impression 2 (t2 = Jan 2, 10:00)
    feats_t2 = compute_user_behavior_features(history, impression_time=t2, user_id="u1", articles_df=articles)
    assert feats_t2["click_count"] == 1
    assert feats_t2["recent_clicked_ids"] == ["art_click_day1"]
    assert "sports" in feats_t2["category_affinity"]
    assert "technology" not in feats_t2["category_affinity"], "Leakage: Day 2 click leaked into Impression 2 features!"

    # 3. Features at Impression 3 (t3 = Jan 3, 10:00)
    feats_t3 = compute_user_behavior_features(history, impression_time=t3, user_id="u1", articles_df=articles)
    assert feats_t3["click_count"] == 2
    assert set(feats_t3["recent_clicked_ids"]) == {"art_click_day1", "art_click_day2"}
    assert "technology" in feats_t3["category_affinity"]


def test_no_future_clicks_at_serving_time():
    """
    Simulate real-time serving request at serving_time = t_serve.
    Even if future click events exist in the database/event stream,
    the serving feature generator must strictly filter to t < t_serve.
    """
    serving_time = datetime(2024, 1, 5, 12, 0, 0)

    # Event stream with historical and future events
    stream_history = pl.DataFrame({
        "user_id": ["u_prod", "u_prod", "u_prod"],
        "clicked_article_id": ["art_prod_past", "art_prod_future1", "art_prod_future2"],
        "click_time": [
            serving_time - timedelta(minutes=15),
            serving_time + timedelta(minutes=5),
            serving_time + timedelta(hours=2),
        ],
    })

    serving_feats = compute_user_behavior_features(
        stream_history, impression_time=serving_time, user_id="u_prod"
    )

    assert serving_feats["click_count"] == 1
    assert serving_feats["recent_clicked_ids"] == ["art_prod_past"]
    assert "art_prod_future1" not in serving_feats["recent_clicked_ids"]
    assert "art_prod_future2" not in serving_feats["recent_clicked_ids"]


def test_article_popularity_no_future_leakage():
    """
    Assert that clicks occurring after impression_time do not leak
    into the candidate article's rolling 24-hour popularity feature.
    """
    imp_time = datetime(2024, 1, 10, 18, 0, 0)
    target_article = "news_breaking_101"

    history = pl.DataFrame({
        "user_id": ["u1", "u2", "u3", "u4"],
        "clicked_article_id": [target_article, target_article, target_article, target_article],
        "click_time": [
            imp_time - timedelta(hours=5),  # Within 24h past window -> VALID
            imp_time - timedelta(hours=1),  # Within 24h past window -> VALID
            imp_time + timedelta(minutes=10), # After impression -> MUST EXCLUDE
            imp_time + timedelta(hours=3),   # After impression -> MUST EXCLUDE
        ],
    })

    articles = pl.DataFrame({
        "article_id": [target_article],
        "category": ["general_news"],
        "published_time": [imp_time - timedelta(hours=6)],
    })

    art_feats = compute_article_features(
        article_id=target_article,
        impression_time=imp_time,
        articles_df=articles,
        history_df=history,
        popularity_window_hours=24.0,
    )

    # Exactly 2 clicks occurred before imp_time
    assert art_feats["popularity_24h"] == 2, f"Leakage: popularity was {art_feats['popularity_24h']}, expected 2!"


def test_target_clicks_do_not_leak_into_current_impression_features():
    """
    Anti-gaming assertion:
    When an impression occurs with clicked_article_ids = ['art_clicked'],
    those target clicks must NOT be used when computing user history features
    for ranking candidates of THAT impression.
    """
    imp_time = datetime(2024, 1, 15, 14, 0, 0)
    current_clicked_article = "art_target"

    # The user has clicked art_prior in the past.
    # At imp_time, the impression log records that art_target was clicked.
    history = pl.DataFrame({
        "user_id": ["user_x"],
        "clicked_article_id": ["art_prior"],
        "click_time": [imp_time - timedelta(hours=1)],
    })

    articles = pl.DataFrame({
        "article_id": ["art_prior", current_clicked_article],
        "category": ["finance", "celebrity"],
    })

    user_feats = compute_user_behavior_features(
        history, impression_time=imp_time, user_id="user_x", articles_df=articles
    )

    cross_feats = compute_cross_features(
        user_feats=user_feats,
        article_feats={"category": "celebrity", "freshness_hours": 1.0, "popularity_24h": 5},
        session_position=0,
        first_stage_score=2.5,
    )

    # Since art_target (celebrity) is clicked IN THIS IMPRESSION,
    # category_affinity_score at the time of ranking must be 0.0 (only finance is in past history).
    assert cross_feats["category_affinity_score"] == 0.0, "Target leakage: current impression click leaked into feature!"
    assert current_clicked_article not in user_feats["recent_clicked_ids"]


def test_cross_features_hybrid_lexical_semantic():
    """
    Verify that both BM25 lexical score and dense semantic similarity score
    are captured simultaneously as distinct features in the re-ranking vector.
    """
    user_feats = {
        "user_id": "u_test",
        "click_count": 5,
        "recency_score": 3.25,
        "category_affinity": {"tech": 0.8},
        "mean_dwell_time": 45.0,
    }
    article_feats = {
        "article_id": "art_hybrid",
        "category": "tech",
        "freshness_hours": 3.5,
        "popularity_24h": 12,
    }

    cross = compute_cross_features(
        user_feats=user_feats,
        article_feats=article_feats,
        session_position=1,
        bm25_score=8.42,
        semantic_score=0.791,
    )

    assert cross["bm25_score"] == 8.42
    assert cross["semantic_score"] == 0.791
    assert cross["first_stage_score"] == 8.42
    assert cross["category_affinity_score"] == 0.8
    assert cross["session_position"] == 1
    assert cross["article_freshness_hours"] == 3.5
    assert cross["article_popularity_24h"] == 12
    assert cross["user_click_count"] == 5

