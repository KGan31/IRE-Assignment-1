"""
Temporal (never random) train/val/test split for interaction data using Polars.

Pools interim train and dev/validation splits together and cuts them chronologically
by dataset size (80% train, 10% val, 10% test by default) or relative day offsets.

Also produces:
    - Global articles catalog: data/processed/{dataset}/articles.parquet
    - Split-specific history files: history_train, history_val, history_test

Usage:
    python src/split.py --config configs/pipeline.yaml
"""

import argparse
from datetime import timedelta
from pathlib import Path
from typing import Tuple, Union

import pandas as pd
import polars as pl
import yaml


def temporal_split_by_ratio(
    impressions: Union[pl.DataFrame, pd.DataFrame],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> Tuple[Union[pl.DataFrame, pd.DataFrame], Union[pl.DataFrame, pd.DataFrame], Union[pl.DataFrame, pd.DataFrame], object, object]:
    """Split impressions chronologically into train/val/test splits based on impression counts.

    - Total impressions sorted chronologically by timestamp (stable with impression_id if available).
    - Train: first train_ratio (e.g. 80%) of impressions.
    - Val:   next val_ratio (e.g. 10%) of impressions.
    - Test:  remaining impressions (e.g. 10%).

    Returns:
        (train_df, val_df, test_df, train_cutoff_ts, val_cutoff_ts)
    """
    is_pandas = isinstance(impressions, pd.DataFrame)
    df = pl.from_pandas(impressions) if is_pandas else impressions

    if df.is_empty():
        raise ValueError("Cannot split an empty impressions frame")

    total_ratio = train_ratio + val_ratio + test_ratio
    if not (0.99 <= total_ratio <= 1.01):
        raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio}")

    sort_cols = ["timestamp", "impression_id"] if "impression_id" in df.columns else ["timestamp"]
    sorted_df = df.sort(by=sort_cols)

    n = len(sorted_df)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train = sorted_df.slice(0, n_train)
    val = sorted_df.slice(n_train, n_val)
    test = sorted_df.slice(n_train + n_val)

    train_end = val["timestamp"].min() if not val.is_empty() else train["timestamp"].max()
    val_end = test["timestamp"].min() if not test.is_empty() else (val["timestamp"].max() if not val.is_empty() else train_end)

    if is_pandas:
        return train.to_pandas(), val.to_pandas(), test.to_pandas(), pd.Timestamp(train_end), pd.Timestamp(val_end)
    return train, val, test, train_end, val_end


def temporal_split_relative(
    impressions: Union[pl.DataFrame, pd.DataFrame],
    train_days: float,
    val_days: float,
    test_days: float,
) -> Tuple[Union[pl.DataFrame, pd.DataFrame], Union[pl.DataFrame, pd.DataFrame], Union[pl.DataFrame, pd.DataFrame], object, object]:
    """Split impressions into train/val/test using relative forward offsets from minimum timestamp."""
    is_pandas = isinstance(impressions, pd.DataFrame)
    df = pl.from_pandas(impressions) if is_pandas else impressions

    if df.is_empty():
        raise ValueError("Cannot split an empty impressions frame")

    min_ts = df["timestamp"].min()
    train_end = min_ts + timedelta(days=train_days)
    val_end = train_end + timedelta(days=val_days)

    train = df.filter(pl.col("timestamp") < train_end)
    val = df.filter((pl.col("timestamp") >= train_end) & (pl.col("timestamp") < val_end))
    test = df.filter(pl.col("timestamp") >= val_end)

    if is_pandas:
        return train.to_pandas(), val.to_pandas(), test.to_pandas(), pd.Timestamp(train_end), pd.Timestamp(val_end)
    return train, val, test, train_end, val_end


def temporal_split(impressions: Union[pl.DataFrame, pd.DataFrame], test_days: int, val_days: int):
    """Split impressions into train/val/test by counting backward from max_ts."""
    is_pandas = isinstance(impressions, pd.DataFrame)
    df = pl.from_pandas(impressions) if is_pandas else impressions

    if df.is_empty():
        raise ValueError("Cannot split an empty impressions frame")

    max_ts = df["timestamp"].max()
    test_start = max_ts - timedelta(days=test_days)
    val_start = test_start - timedelta(days=val_days)

    train = df.filter(pl.col("timestamp") < val_start)
    val = df.filter((pl.col("timestamp") >= val_start) & (pl.col("timestamp") < test_start))
    test = df.filter(pl.col("timestamp") >= test_start)

    if is_pandas:
        return train.to_pandas(), val.to_pandas(), test.to_pandas()
    return train, val, test


def history_as_of(history: Union[pl.DataFrame, pd.DataFrame], cutoff_ts) -> Union[pl.DataFrame, pd.DataFrame]:
    """Return only click-history rows strictly before cutoff_ts to prevent future leakage."""
    is_pandas = isinstance(history, pd.DataFrame)
    if is_pandas:
        if history.empty:
            return history.copy()
        return history[history["click_time"] < cutoff_ts].copy()

    if history.is_empty():
        return history
    return history.filter(pl.col("click_time") < cutoff_ts)


def split_dataset(dataset_name: str, interim_dir: Path, processed_dir: Path, split_cfg: dict) -> None:
    ds_interim = interim_dir / dataset_name
    ds_processed = processed_dir / dataset_name
    ds_processed.mkdir(parents=True, exist_ok=True)

    if not ds_interim.exists():
        print(f"[{dataset_name}] interim directory {ds_interim} not found, skipping")
        return

    # 1. --- Articles (Global catalog) ---
    article_files = sorted(ds_interim.glob("articles_*.parquet"))
    if not article_files:
        article_files = sorted(ds_interim.glob("articles.parquet"))

    if article_files:
        articles_dfs = [pl.read_parquet(f) for f in article_files]
        articles = pl.concat(articles_dfs, how="diagonal").unique(subset="article_id")
        articles.write_parquet(ds_processed / "articles.parquet")
        print(f"[{dataset_name}] Wrote global articles.parquet ({len(articles)} rows)")
    else:
        print(f"[{dataset_name}] WARNING: No article files found in interim directory")

    # 2. --- Impressions (Merge train + dev/validation and re-split) ---
    impression_files = sorted(ds_interim.glob("impressions_*.parquet"))
    if not impression_files:
        print(f"[{dataset_name}] No impression files found in {ds_interim}, skipping split")
        return

    impressions = pl.concat([pl.read_parquet(f) for f in impression_files], how="diagonal")
    if "impression_id" in impressions.columns:
        impressions = impressions.unique(subset="impression_id")
    impressions = impressions.sort("timestamp")

    if "train_ratio" in split_cfg:
        train_ratio = float(split_cfg.get("train_ratio", 0.8))
        val_ratio = float(split_cfg.get("val_ratio", 0.1))
        test_ratio = float(split_cfg.get("test_ratio", 0.1))
        train_imp, val_imp, test_imp, train_end, val_end = temporal_split_by_ratio(
            impressions, train_ratio, val_ratio, test_ratio
        )
    else:
        train_days = split_cfg.get("train_days")
        val_days = split_cfg.get("val_days")
        test_days = split_cfg.get("test_days")
        train_imp, val_imp, test_imp, train_end, val_end = temporal_split_relative(
            impressions, train_days, val_days, test_days
        )

    train_imp.write_parquet(ds_processed / "impressions_train.parquet")
    val_imp.write_parquet(ds_processed / "impressions_val.parquet")
    test_imp.write_parquet(ds_processed / "impressions_test.parquet")

    min_ts = impressions["timestamp"].min()
    max_ts = impressions["timestamp"].max()
    print(f"[{dataset_name}] Impressions total={len(impressions)} | Min TS: {min_ts} | Max TS: {max_ts}")
    print(f"[{dataset_name}] Split: train={len(train_imp)} ({len(train_imp)/len(impressions):.1%}, < {train_end}), "
          f"val={len(val_imp)} ({len(val_imp)/len(impressions):.1%}, {train_end} to {val_end}), "
          f"test={len(test_imp)} ({len(test_imp)/len(impressions):.1%}, >= {val_end})")

    # 3. --- History (Pool and split chronologically by cutoff timestamps) ---
    history_files = sorted(ds_interim.glob("history_*.parquet"))
    if history_files:
        history = pl.concat([pl.read_parquet(f) for f in history_files], how="diagonal")
        if not history.is_empty() and "clicked_article_id" in history.columns:
            history = history.unique()

        history_train = history_as_of(history, train_end)
        history_val = history_as_of(history, val_end)
        history_test = history.clone()

        history_train.write_parquet(ds_processed / "history_train.parquet")
        history_val.write_parquet(ds_processed / "history_val.parquet")
        history_test.write_parquet(ds_processed / "history_test.parquet")

        print(f"[{dataset_name}] History split: train={len(history_train)}, "
              f"val={len(history_val)}, test={len(history_test)}")


def main(config_path: str) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    interim_dir = Path(cfg["paths"]["interim_dir"])
    processed_dir = Path(cfg["paths"]["processed_dir"])
    split_cfgs = cfg["split"]

    for dataset_name in ["mind", "ebnerd"]:
        if dataset_name in split_cfgs:
            split_dataset(dataset_name, interim_dir, processed_dir, split_cfgs[dataset_name])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pipeline.yaml")
    args = parser.parse_args()
    main(args.config)