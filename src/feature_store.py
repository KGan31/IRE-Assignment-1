"""
Builds a small, reusable feature store:
    - article_features.parquet : per-article content features
    - user_features.parquet    : per-user click-history summaries,
                                  computed separately for each split so
                                  that val/test features never see
                                  clicks that happen after that split's
                                  own impressions (see split.history_as_of).

Usage:
    python src/feature_store.py --config configs/pipeline.yaml
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from split import history_as_of

HALF_LIFE_DEFAULT_HOURS = 24


def build_article_features(interim_dir: Path) -> pd.DataFrame:
    article_files = sorted(interim_dir.glob("articles_*.parquet"))
    if not article_files:
        return pd.DataFrame()

    articles = pd.concat([pd.read_parquet(f) for f in article_files], ignore_index=True)
    articles = articles.drop_duplicates(subset="article_id").reset_index(drop=True)
    return articles


def build_user_features(history: pd.DataFrame, as_of_ts: pd.Timestamp,
                         half_life_hours: float) -> pd.DataFrame:
    """One row per user: recent click list + a recency-weighted click count.

    Recency weighting uses exponential decay: weight = 0.5 ** (age_hours / half_life).
    This becomes a simple, reusable behavioural feature for later ranking models.
    """
    hist = history_as_of(history, as_of_ts)
    if hist.empty:
        return pd.DataFrame(columns=["user_id", "click_history", "n_clicks", "recency_score"])

    age_hours = (as_of_ts - hist["click_time"]).dt.total_seconds() / 3600.0
    hist = hist.assign(weight=0.5 ** (age_hours / half_life_hours))

    grouped = hist.groupby("user_id").agg(
        click_history=("clicked_article_id", list),
        n_clicks=("clicked_article_id", "count"),
        recency_score=("weight", "sum"),
    ).reset_index()

    return grouped


def main(config_path: str) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    interim_dir = Path(cfg["paths"]["interim_dir"])
    processed_dir = Path(cfg["paths"]["processed_dir"])
    half_life = cfg.get("features", {}).get(
        "recency_half_life_hours", HALF_LIFE_DEFAULT_HOURS
    )

    for dataset_name in ["mind", "ebnerd"]:
        ds_interim = interim_dir / dataset_name
        ds_processed = processed_dir / dataset_name
        ds_processed.mkdir(parents=True, exist_ok=True)

        # --- article features (shared across splits) ---
        articles = build_article_features(ds_interim)
        if articles.empty:
            print(f"[{dataset_name}] no articles found, skipping")
            continue
        articles.to_parquet(ds_processed / "article_features.parquet")

        # --- user features (computed per split, as-of that split's cutoff) ---
        history_files = sorted(ds_interim.glob("history_*.parquet"))
        if not history_files:
            print(f"[{dataset_name}] no history files found, skipping user features")
            continue
        history = pd.concat([pd.read_parquet(f) for f in history_files], ignore_index=True)

        for split_name in ["train", "val", "test"]:
            impressions_path = ds_processed / f"impressions_{split_name}.parquet"
            if not impressions_path.exists():
                continue
            impressions = pd.read_parquet(impressions_path)
            if impressions.empty:
                continue

            # as_of cutoff = the earliest timestamp in this split, so user
            # features for every impression in the split only use clicks
            # strictly before the split begins. Conservative but leak-free.
            as_of_ts = impressions["timestamp"].min()
            user_feats = build_user_features(history, as_of_ts, half_life)
            user_feats.to_parquet(ds_processed / f"user_features_{split_name}.parquet")

        print(f"[{dataset_name}] article_features={len(articles)} rows, "
              f"user feature tables written per split")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pipeline.yaml")
    args = parser.parse_args()
    main(args.config)
