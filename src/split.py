"""
Temporal (never random) train/val/test split for interaction data.

Splits by impression timestamp: the most recent `test_days` become the
test set, the `val_days` before that become validation, everything
earlier is train. This is applied per-dataset (MIND and EB-NeRD have
different date ranges) and per-dataset splits are concatenated at the end.

Usage:
    python src/split.py --config configs/pipeline.yaml
"""

import argparse
from pathlib import Path

import pandas as pd
import yaml


def temporal_split(impressions: pd.DataFrame, test_days: int, val_days: int):
    """Split impressions into train/val/test by timestamp.

    Returns three DataFrames. Boundaries are inclusive on the lower edge,
    exclusive on the upper edge of each bucket — i.e. no impression
    appears in two splits.
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


def history_as_of(history: pd.DataFrame, cutoff_ts: pd.Timestamp) -> pd.DataFrame:
    """Return only the click-history rows that happened strictly before
    cutoff_ts. Use this when building features for a given split — using
    the FULL history (including future clicks) is the classic leakage bug.
    """
    return history[history["click_time"] < cutoff_ts].copy()


def split_dataset(dataset_name: str, interim_dir: Path, processed_dir: Path,
                   test_days: int, val_days: int) -> None:
    impression_files = sorted(interim_dir.glob("impressions_*.parquet"))
    if not impression_files:
        print(f"[{dataset_name}] no impression files found in {interim_dir}, skipping")
        return

    impressions = pd.concat([pd.read_parquet(f) for f in impression_files], ignore_index=True)
    impressions = impressions.sort_values("timestamp").reset_index(drop=True)

    train, val, test = temporal_split(impressions, test_days, val_days)

    out_dir = processed_dir / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    train.to_parquet(out_dir / "impressions_train.parquet")
    val.to_parquet(out_dir / "impressions_val.parquet")
    test.to_parquet(out_dir / "impressions_test.parquet")

    print(f"[{dataset_name}] train={len(train)} val={len(val)} test={len(test)} "
          f"(cut points: val_start={train['timestamp'].max() if len(train) else 'NA'}, "
          f"test_start={val['timestamp'].max() if len(val) else 'NA'})")


def main(config_path: str) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    interim_dir = Path(cfg["paths"]["interim_dir"])
    processed_dir = Path(cfg["paths"]["processed_dir"])
    test_days = cfg["split"]["test_days"]
    val_days = cfg["split"]["val_days"]

    for dataset_name in ["mind", "ebnerd"]:
        split_dataset(dataset_name, interim_dir / dataset_name, processed_dir,
                       test_days, val_days)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pipeline.yaml")
    args = parser.parse_args()
    main(args.config)
