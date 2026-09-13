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


def build_article_features(interim_dir: Path, processed_dir: Optional[Path] = None) -> pl.DataFrame:
    article_files = sorted(interim_dir.glob("articles_*.parquet"))
    if not article_files:
        if processed_dir and (processed_dir / "articles.parquet").exists():
            return pl.read_parquet(processed_dir / "articles.parquet")
        return pl.DataFrame()

    articles_list = [pl.read_parquet(f) for f in article_files]
    articles = pl.concat(articles_list, how="diagonal").unique(subset="article_id")
    return articles


def build_user_features(
    history: Union[pl.DataFrame, pd.DataFrame, pl.LazyFrame],
    as_of_ts: object,
    half_life_hours: float = HALF_LIFE_DEFAULT_HOURS,
    is_train: bool = False,
) -> pl.DataFrame:
    """One row per user: recent click list + a recency-weighted click count.

    Recency weighting uses exponential decay: weight = 0.5 ** (age_hours / half_life).
    This becomes a simple, reusable behavioural feature for ranking models.
    """
    if isinstance(history, pl.LazyFrame):
        hist_lazy = history
        if is_train:
            hist_filtered = hist_lazy.filter(pl.col("click_time") <= as_of_ts)
        else:
            hist_filtered = hist_lazy.filter(pl.col("click_time") < as_of_ts)

        hist_with_weights = hist_filtered.with_columns(
            ((pl.lit(as_of_ts) - pl.col("click_time")).dt.total_seconds() / 3600.0).alias("age_hours")
        ).with_columns(
            (0.5 ** (pl.col("age_hours") / half_life_hours)).alias("weight")
        )

        grouped = hist_with_weights.group_by("user_id").agg([
            pl.col("clicked_article_id").alias("click_history"),
            pl.len().cast(pl.UInt32).alias("n_clicks"),
            pl.col("weight").sum().alias("recency_score"),
        ]).collect()

        return grouped

    hist = history_as_of(history, as_of_ts) if not is_train else history
    if isinstance(hist, pd.DataFrame):
        hist = pl.from_pandas(hist)

    if is_train and not hist.is_empty():
        hist = hist.filter(pl.col("click_time") <= as_of_ts)

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


def process_feature_store(
    dataset_name: str,
    interim_dir: Path,
    processed_dir: Path,
    half_life: float = HALF_LIFE_DEFAULT_HOURS,
) -> None:
    ds_interim = interim_dir / dataset_name
    ds_processed = processed_dir / dataset_name
    ds_processed.mkdir(parents=True, exist_ok=True)

    print(f"\n[{dataset_name}] Building Feature Store in {ds_processed}...")

    # --- article features (shared across splits) ---
    articles = build_article_features(ds_interim, ds_processed)
    if articles.is_empty():
        print(f"[{dataset_name}] No articles found, skipping")
        return
    articles.write_parquet(ds_processed / "article_features.parquet")
    print(f"[{dataset_name}] Wrote article_features.parquet ({len(articles):,} articles)")

    # --- user features per split ---
    # For large datasets (or large history files), use lazy streaming scans per split
    # to avoid loading hundreds of millions of interaction rows into memory at once.
    if dataset_name == "mind_large":
        # Mind Large specific memory-efficient pipeline:
        hist_train_path = ds_processed / "history_train.parquet"
        hist_dev_path = ds_processed / "history_dev.parquet"
        if not hist_dev_path.exists():
            hist_dev_path = ds_processed / "history_val.parquet"

        for split_name in ["train", "val", "test"]:
            imp_path = ds_processed / f"impressions_{split_name}.parquet"
            if not imp_path.exists() and split_name == "val":
                imp_path = ds_processed / "impressions_dev.parquet"
            if not imp_path.exists():
                print(f"[{dataset_name}] {imp_path.name} not found, skipping {split_name}")
                continue

            imp_df = pl.read_parquet(imp_path)
            if imp_df.is_empty():
                continue

            if split_name == "train":
                # Train user features use the max training timestamp so training users have complete histories
                as_of_ts = imp_df["timestamp"].max()
                hist_lazy = pl.scan_parquet(hist_train_path)
                print(f"[{dataset_name}] Computing user_features_train (cutoff <= {as_of_ts})...")
                user_feats = build_user_features(hist_lazy, as_of_ts, half_life, is_train=True)
            elif split_name in ("val", "dev"):
                # Validation user features strictly use prior history (all train clicks prior to dev)
                as_of_ts = imp_df["timestamp"].min()
                hist_lazy = pl.scan_parquet(hist_train_path)
                print(f"[{dataset_name}] Computing user_features_val (cutoff < {as_of_ts})...")
                user_feats = build_user_features(hist_lazy, as_of_ts, half_life, is_train=False)
            else:
                # Test user features use history from train and dev prior to test
                as_of_ts = imp_df["timestamp"].min()
                hist_inputs = [hist_train_path]
                if hist_dev_path.exists():
                    hist_inputs.append(hist_dev_path)
                hist_lazy = pl.concat([pl.scan_parquet(p) for p in hist_inputs])
                print(f"[{dataset_name}] Computing user_features_test (cutoff < {as_of_ts})...")
                user_feats = build_user_features(hist_lazy, as_of_ts, half_life, is_train=False)

            out_user_path = ds_processed / f"user_features_{split_name}.parquet"
            user_feats.write_parquet(out_user_path)
            print(f"[{dataset_name}] Wrote {out_user_path.name} ({len(user_feats):,} users)")

    else:
        # Standard pipeline for mind (small) and ebnerd
        history_files = sorted(ds_interim.glob("history_*.parquet"))
        if not history_files:
            # Fallback to processed dir if interim files are split
            history_files = sorted(ds_processed.glob("history_*.parquet"))
        if not history_files:
            print(f"[{dataset_name}] No history files found, skipping user features")
            return

        for split_name in ["train", "val", "test"]:
            impressions_path = ds_processed / f"impressions_{split_name}.parquet"
            if not impressions_path.exists() and split_name == "val":
                impressions_path = ds_processed / "impressions_validation.parquet"
            if not impressions_path.exists():
                continue

            impressions = pl.read_parquet(impressions_path)
            if impressions.is_empty():
                continue

            if split_name == "train":
                as_of_ts = impressions["timestamp"].max()
                is_train = True
            else:
                as_of_ts = impressions["timestamp"].min()
                is_train = False

            hist_lazy = pl.concat([pl.scan_parquet(f) for f in history_files])
            user_feats = build_user_features(hist_lazy, as_of_ts, half_life, is_train=is_train)
            user_feats.write_parquet(ds_processed / f"user_features_{split_name}.parquet")
            print(f"[{dataset_name}] Wrote user_features_{split_name}.parquet ({len(user_feats):,} users)")


def main(config_path: str, dataset_arg: str = "all") -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    interim_dir = Path(cfg["paths"]["interim_dir"])
    processed_dir = Path(cfg["paths"]["processed_dir"])
    half_life = float(cfg.get("features", {}).get(
        "recency_half_life_hours", HALF_LIFE_DEFAULT_HOURS
    ))

    if dataset_arg in ("all", "both"):
        datasets = ["mind", "ebnerd"]
    else:
        datasets = [d.strip() for d in dataset_arg.split(",")]

    for dataset_name in datasets:
        process_feature_store(dataset_name, interim_dir, processed_dir, half_life)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build article and per-split user feature stores.")
    parser.add_argument("--config", default="configs/pipeline.yaml")
    parser.add_argument("--dataset", default="all", help="mind, ebnerd, mind_large, ebnerd_large, or all")
    args = parser.parse_args()
    main(args.config, args.dataset)

