"""
Temporal (never random) train/val/test split for interaction data.

Pools interim train and dev/validation splits together and cuts them chronologically
using relative day offsets from the start timestamp:

    - MIND:    5 days train, 1 day val, 1 day test (Total = 7 days)
    - EB-NeRD: 10 days train, 2 days val, 2 days test (Total = 14 days)

Also produces:
    - Global articles catalog: data/processed/{dataset}/articles.parquet
    - Split-specific history files: history_train, history_val, history_test

Usage:
    python src/split.py --config configs/pipeline.yaml
"""

import argparse
from pathlib import Path
from typing import Tuple

import pandas as pd
import yaml


def temporal_split(impressions: pd.DataFrame, test_days: int, val_days: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split impressions into train/val/test by counting backward from max_ts.
    Maintained for backwards compatibility / generic 3-way cuts.
    """
    if impressions.empty:
        raise ValueError("Cannot split an empty impressions frame")

    max_ts = impressions["timestamp"].max()
    test_start = max_ts - pd.Timedelta(days=test_days)
    val_start = test_start - pd.Timedelta(days=val_days)

    train = impressions[impressions["timestamp"] < val_start].copy()
    val = impressions[
        (impressions["timestamp"] >= val_start) & (impressions["timestamp"] < test_start)
    ].copy()
    test = impressions[impressions["timestamp"] >= test_start].copy()

    return train, val, test


def temporal_split_relative(impressions: pd.DataFrame, train_days: float, val_days: float, test_days: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    """Split impressions into train/val/test using relative forward offsets from minimum timestamp.

    - Train: [min_ts, min_ts + train_days)
    - Val:   [min_ts + train_days, min_ts + train_days + val_days)
    - Test:  [min_ts + train_days + val_days, end)
    """
    if impressions.empty:
        raise ValueError("Cannot split an empty impressions frame")

    min_ts = impressions["timestamp"].min()
    train_end = min_ts + pd.Timedelta(days=train_days)
    val_end = train_end + pd.Timedelta(days=val_days)

    train = impressions[impressions["timestamp"] < train_end].copy()
    val = impressions[
        (impressions["timestamp"] >= train_end) & (impressions["timestamp"] < val_end)
    ].copy()
    test = impressions[impressions["timestamp"] >= val_end].copy()

    return train, val, test, train_end, val_end


def history_as_of(history: pd.DataFrame, cutoff_ts: pd.Timestamp) -> pd.DataFrame:
    """Return only the click-history rows that happened strictly before
    cutoff_ts to avoid data leakage.
    """
    if history.empty:
        return history.copy()
    return history[history["click_time"] < cutoff_ts].copy()


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
        articles_dfs = [pd.read_parquet(f) for f in article_files]
        articles = pd.concat(articles_dfs, ignore_index=True).drop_duplicates(subset="article_id").reset_index(drop=True)
        articles.to_parquet(ds_processed / "articles.parquet")
        print(f"[{dataset_name}] Wrote global articles.parquet ({len(articles)} rows)")
    else:
        print(f"[{dataset_name}] WARNING: No article files found in interim directory")

    # 2. --- Impressions (Merge train + dev/validation and re-split) ---
    impression_files = sorted(ds_interim.glob("impressions_*.parquet"))
    if not impression_files:
        print(f"[{dataset_name}] No impression files found in {ds_interim}, skipping split")
        return

    impressions = pd.concat([pd.read_parquet(f) for f in impression_files], ignore_index=True)
    if "impression_id" in impressions.columns:
        impressions = impressions.drop_duplicates(subset="impression_id")
    impressions = impressions.sort_values("timestamp").reset_index(drop=True)

    train_days = split_cfg.get("train_days")
    val_days = split_cfg.get("val_days")
    test_days = split_cfg.get("test_days")

    train_imp, val_imp, test_imp, train_end, val_end = temporal_split_relative(
        impressions, train_days, val_days, test_days
    )

    train_imp.to_parquet(ds_processed / "impressions_train.parquet")
    val_imp.to_parquet(ds_processed / "impressions_val.parquet")
    test_imp.to_parquet(ds_processed / "impressions_test.parquet")

    min_ts = impressions["timestamp"].min()
    max_ts = impressions["timestamp"].max()
    print(f"[{dataset_name}] Impressions total={len(impressions)} | Min TS: {min_ts} | Max TS: {max_ts}")
    print(f"[{dataset_name}] Split: train={len(train_imp)} (< {train_end}), "
          f"val={len(val_imp)} ({train_end} to {val_end}), "
          f"test={len(test_imp)} (>= {val_end})")

    # 3. --- History (Pool and split chronologically by cutoff timestamps) ---
    history_files = sorted(ds_interim.glob("history_*.parquet"))
    if history_files:
        history = pd.concat([pd.read_parquet(f) for f in history_files], ignore_index=True)
        if not history.empty and "clicked_article_id" in history.columns:
            history = history.drop_duplicates().reset_index(drop=True)

        history_train = history_as_of(history, train_end)
        history_val = history_as_of(history, val_end)
        history_test = history.copy()

        history_train.to_parquet(ds_processed / "history_train.parquet")
        history_val.to_parquet(ds_processed / "history_val.parquet")
        history_test.to_parquet(ds_processed / "history_test.parquet")

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