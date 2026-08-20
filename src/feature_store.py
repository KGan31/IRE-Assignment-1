"""
Builds a reusable feature store using Polars:
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
from typing import Union

import pandas as pd
import polars as pl
import yaml

from split import history_as_of

HALF_LIFE_DEFAULT_HOURS = 24.0


def build_article_features(interim_dir: Path) -> pl.DataFrame:
    article_files = sorted(interim_dir.glob("articles_*.parquet"))
    if not article_files:
        return pl.DataFrame()

    articles_list = [pl.read_parquet(f) for f in article_files]
    articles = pl.concat(articles_list, how="diagonal").unique(subset="article_id")
    return articles


def build_user_features(
    history: Union[pl.DataFrame, pd.DataFrame],
    as_of_ts: object,
    half_life_hours: float = HALF_LIFE_DEFAULT_HOURS,
) -> pl.DataFrame:
    """One row per user: recent click list + a recency-weighted click count.

    Recency weighting uses exponential decay: weight = 0.5 ** (age_hours / half_life).
    This becomes a simple, reusable behavioural feature for ranking models.
    """
    hist = history_as_of(history, as_of_ts)
    if isinstance(hist, pd.DataFrame):
        hist = pl.from_pandas(hist)

    if hist.is_empty():
        return pl.DataFrame(
            schema={
                "user_id": pl.Utf8,
                "click_history": pl.List(pl.Utf8),
                "n_clicks": pl.UInt32,
                "recency_score": pl.Float64,
            }
        )

    hist_with_weights = hist.with_columns(
        ((pl.lit(as_of_ts) - pl.col("click_time")).dt.total_seconds() / 3600.0).alias("age_hours")
    ).with_columns(
        (0.5 ** (pl.col("age_hours") / half_life_hours)).alias("weight")
    )

    grouped = hist_with_weights.group_by("user_id").agg([
        pl.col("clicked_article_id").alias("click_history"),
        pl.len().cast(pl.UInt32).alias("n_clicks"),
        pl.col("weight").sum().alias("recency_score"),
    ])

    return grouped


def main(config_path: str) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    interim_dir = Path(cfg["paths"]["interim_dir"])
    processed_dir = Path(cfg["paths"]["processed_dir"])
    half_life = float(cfg.get("features", {}).get(
        "recency_half_life_hours", HALF_LIFE_DEFAULT_HOURS
    ))

    for dataset_name in ["mind", "ebnerd"]:
        ds_interim = interim_dir / dataset_name
        ds_processed = processed_dir / dataset_name
        ds_processed.mkdir(parents=True, exist_ok=True)

        # --- article features (shared across splits) ---
        articles = build_article_features(ds_interim)
        if articles.is_empty():
            print(f"[{dataset_name}] no articles found, skipping")
            continue
        articles.write_parquet(ds_processed / "article_features.parquet")

        # --- user features (computed per split, as-of that split's cutoff) ---
        history_files = sorted(ds_interim.glob("history_*.parquet"))
        if not history_files:
            print(f"[{dataset_name}] no history files found, skipping user features")
            continue
        history = pl.concat([pl.read_parquet(f) for f in history_files], how="diagonal")

        for split_name in ["train", "val", "test"]:
            impressions_path = ds_processed / f"impressions_{split_name}.parquet"
            if not impressions_path.exists():
                continue
            impressions = pl.read_parquet(impressions_path)
            if impressions.is_empty():
                continue

            # as_of cutoff = the earliest timestamp in this split
            as_of_ts = impressions["timestamp"].min()
            user_feats = build_user_features(history, as_of_ts, half_life)
            user_feats.write_parquet(ds_processed / f"user_features_{split_name}.parquet")

        print(f"[{dataset_name}] article_features={len(articles)} rows, "
              f"user feature tables written per split")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pipeline.yaml")
    args = parser.parse_args()
    main(args.config)
