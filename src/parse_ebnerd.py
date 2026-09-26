"""
Parses raw EB-NeRD parquet files into the unified schema (see schema.py) using Polars.

Supports demo, small, and large datasets.
For large dataset, train, validation, and test splits are processed directly as-is.

EB-NeRD files:
articles.parquet:
    article_id | title | subtitle | body | category_str | published_time | ...

train/behaviors.parquet & validation/behaviors.parquet:
    impression_id | user_id | impression_time | article_ids_inview |
    article_ids_clicked | ...

test/behaviors.parquet:
    impression_id | user_id | impression_time | article_ids_inview | ... (no article_ids_clicked)

train/history.parquet, validation/history.parquet, test/history.parquet:
    user_id | impression_time_fixed | article_id_fixed | scroll_percentage_fixed | ...

Usage:
    python src/parse_ebnerd.py --split train
    python src/parse_ebnerd.py --split validation
    python src/parse_ebnerd.py --split test
    python src/parse_ebnerd.py --split all --dataset_type large
"""

import argparse
from pathlib import Path
from typing import List, Optional

import polars as pl

from schema import validate_articles, validate_impressions

DATASET = "ebnerd"


def parse_articles(articles_parquet: Path) -> pl.DataFrame:
    df = pl.read_parquet(articles_parquet)

    def col_expr(name: str, default_val: str = ""):
        if name in df.columns:
            return pl.col(name).cast(pl.Utf8).fill_null(default_val)
        return pl.lit(default_val)

    pub_time = (
        pl.col("published_time").cast(pl.Datetime)
        if "published_time" in df.columns
        else pl.lit(None).cast(pl.Datetime)
    )

    out = df.select([
        (pl.lit("ebnerd_") + pl.col("article_id").cast(pl.Utf8)).alias("article_id"),
        pl.lit(DATASET).alias("dataset"),
        col_expr("title").alias("title"),
        col_expr("subtitle").alias("abstract"),
        col_expr("body").alias("body"),
        col_expr("category_str").alias("category"),
        pub_time.alias("published_time"),
        pl.lit([]).cast(pl.List(pl.Utf8)).alias("entities"),
        pl.lit(None).alias("embedding"),
    ])
    validate_articles(out)
    return out


def parse_impressions(behaviors_parquet: Path) -> pl.DataFrame:
    # Inspect available columns first to load only required columns for low memory usage
    schema = pl.scan_parquet(behaviors_parquet).collect_schema()
    available_cols = schema.names()

    desired_cols = ["impression_id", "user_id", "impression_time"]
    if "article_ids_inview" in available_cols:
        desired_cols.append("article_ids_inview")
    if "article_ids_clicked" in available_cols:
        desired_cols.append("article_ids_clicked")

    df = pl.read_parquet(behaviors_parquet, columns=desired_cols)

    cands = (
        pl.col("article_ids_inview").list.eval(pl.lit("ebnerd_") + pl.element().cast(pl.Utf8))
        if "article_ids_inview" in df.columns
        else pl.lit([]).cast(pl.List(pl.Utf8))
    )
    clicked = (
        pl.col("article_ids_clicked").list.eval(pl.lit("ebnerd_") + pl.element().cast(pl.Utf8))
        if "article_ids_clicked" in df.columns
        else pl.lit([]).cast(pl.List(pl.Utf8))
    )

    out = df.select([
        (pl.lit("ebnerd_") + pl.col("impression_id").cast(pl.Utf8)).alias("impression_id"),
        pl.lit(DATASET).alias("dataset"),
        (pl.lit("ebnerd_") + pl.col("user_id").cast(pl.Utf8)).alias("user_id"),
        pl.col("impression_time").cast(pl.Datetime).alias("timestamp"),
        cands.alias("candidate_article_ids"),
        clicked.alias("clicked_article_ids"),
        pl.lit(None).alias("session_context"),
    ])
    validate_impressions(out)
    return out


def parse_history(history_parquet: Path) -> pl.DataFrame:
    schema = pl.scan_parquet(history_parquet).collect_schema()
    available_cols = schema.names()

    desired_cols = ["user_id"]
    if "article_id_fixed" in available_cols:
        desired_cols.append("article_id_fixed")
    if "impression_time_fixed" in available_cols:
        desired_cols.append("impression_time_fixed")
    if "read_time_fixed" in available_cols:
        desired_cols.append("read_time_fixed")
    if "scroll_percentage_fixed" in available_cols:
        desired_cols.append("scroll_percentage_fixed")

    df = pl.read_parquet(history_parquet, columns=desired_cols)
    if df.is_empty() or "article_id_fixed" not in df.columns:
        return pl.DataFrame(
            schema={
                "dataset": pl.Utf8,
                "user_id": pl.Utf8,
                "clicked_article_id": pl.Utf8,
                "click_time": pl.Datetime,
                "dwell_time": pl.Float32,
                "scroll_percentage": pl.Float32,
            }
        )

    has_imp_time = "impression_time_fixed" in df.columns
    has_read_time = "read_time_fixed" in df.columns
    has_scroll = "scroll_percentage_fixed" in df.columns

    select_exprs = [
        pl.lit(DATASET).alias("dataset"),
        (pl.lit("ebnerd_") + pl.col("user_id").cast(pl.Utf8)).alias("user_id"),
        pl.col("article_id_fixed"),
    ]
    explode_cols = ["article_id_fixed"]

    if has_imp_time:
        select_exprs.append(pl.col("impression_time_fixed"))
        explode_cols.append("impression_time_fixed")
    if has_read_time:
        select_exprs.append(pl.col("read_time_fixed"))
        explode_cols.append("read_time_fixed")
    if has_scroll:
        select_exprs.append(pl.col("scroll_percentage_fixed"))
        explode_cols.append("scroll_percentage_fixed")

    exploded = (
        df.select(select_exprs)
        .explode(explode_cols, empty_as_null=True)
        .filter(pl.col("article_id_fixed").is_not_null())
    )

    final_select = [
        pl.col("dataset"),
        pl.col("user_id"),
        (pl.lit("ebnerd_") + pl.col("article_id_fixed").cast(pl.Utf8)).alias("clicked_article_id"),
        pl.col("impression_time_fixed").cast(pl.Datetime).alias("click_time") if has_imp_time else pl.lit(None).cast(pl.Datetime).alias("click_time"),
    ]
    if has_read_time:
        final_select.append(pl.col("read_time_fixed").cast(pl.Float32).alias("dwell_time"))
    if has_scroll:
        final_select.append(pl.col("scroll_percentage_fixed").cast(pl.Float32).alias("scroll_percentage"))

    out = exploded.select(final_select)
    return out


def find_split_dir(raw_base: Path, split: str) -> Optional[Path]:
    """Find directory containing behaviors.parquet for split."""
    candidates = [
        raw_base / split,
        raw_base / "demo" / split,
        raw_base / "ebnerd_testset" / split,
        raw_base / "ebnerd_testset" if split == "test" else None,
    ]
    for c in candidates:
        if c and c.exists() and (c / "behaviors.parquet").exists():
            return c
    return None



def find_articles_files(raw_base: Path) -> List[Path]:
    """Find all articles parquet files in raw_base directory."""
    files = []
    direct = raw_base / "articles.parquet"
    if direct.exists():
        files.append(direct)
    for p in raw_base.glob("**/articles*.parquet"):
        if p not in files:
            files.append(p)
    return files


def process_split(raw_base: Path, out_dir: Path, split: str) -> bool:
    split_dir = find_split_dir(raw_base, split)
    if not split_dir:
        print(f"[ebnerd/{split}] Directory for split '{split}' not found in {raw_base}, skipping.")
        return False

    behaviors_path = split_dir / "behaviors.parquet"
    history_path = split_dir / "history.parquet"

    print(f"Parsing EB-NeRD split '{split}' from {split_dir}...")
    impressions = parse_impressions(behaviors_path)
    
    history = pl.DataFrame()
    if history_path.exists():
        history = parse_history(history_path)

    out_dir.mkdir(parents=True, exist_ok=True)
    impressions.write_parquet(out_dir / f"impressions_{split}.parquet")
    if not history.is_empty():
        history.write_parquet(out_dir / f"history_{split}.parquet")

    print(f"[ebnerd/{split}] impressions={len(impressions)} history_rows={len(history)}")
    return True


def main(split: str, dataset_type: str = "demo") -> None:
    raw_dir_name = "ebnerd_large" if dataset_type == "large" else "ebnerd"
    raw_base = Path("data/raw") / raw_dir_name
    if dataset_type == "demo" and not (raw_base / "demo").exists() and not (raw_base / "train").exists():
        raw_base = Path("data/raw/ebnerd/demo")

    interim_dir = Path("data/interim") / raw_dir_name
    processed_dir = Path("data/processed") / raw_dir_name
    interim_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    # 1. Parse articles catalog
    article_files = find_articles_files(raw_base)
    if not article_files:
        # Fallback to demo articles if large not downloaded yet
        if Path("data/raw/ebnerd/demo/articles.parquet").exists():
            article_files = [Path("data/raw/ebnerd/demo/articles.parquet")]

    if article_files:
        print(f"Parsing and consolidating articles from: {[str(p) for p in article_files]}")
        all_articles = [parse_articles(p) for p in article_files]
        articles = pl.concat(all_articles).unique(subset=["article_id"])
        articles.write_parquet(processed_dir / "articles.parquet")
        articles.write_parquet(interim_dir / "articles.parquet")
        print(f"Total unique articles: {len(articles)} saved to {processed_dir / 'articles.parquet'}")

    # 2. Process requested splits
    splits = ["train", "validation", "test"] if split == "all" else [split]
    for s in splits:
        success = process_split(raw_base, interim_dir, s)
        if success:
            # Copy/link to processed_dir
            imp = pl.read_parquet(interim_dir / f"impressions_{s}.parquet")
            imp.write_parquet(processed_dir / f"impressions_{s}.parquet")
            if s == "validation":
                imp.write_parquet(processed_dir / "impressions_val.parquet")

            if (interim_dir / f"history_{s}.parquet").exists():
                hist = pl.read_parquet(interim_dir / f"history_{s}.parquet")
                hist.write_parquet(processed_dir / f"history_{s}.parquet")
                if s == "validation":
                    hist.write_parquet(processed_dir / "history_val.parquet")

    print(f"\nDone! Processed EB-NeRD data in: {processed_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "validation", "val", "test", "all"], default="all")
    parser.add_argument("--dataset_type", choices=["demo", "small", "large"], default="demo")
    args = parser.parse_args()
    split_arg = "validation" if args.split == "val" else args.split
    main(split_arg, args.dataset_type)

