"""
Parses raw EB-NeRD parquet files into the unified schema (see schema.py) using Polars.

EB-NeRD demo bundle ships:
articles.parquet:
    article_id | title | subtitle | body | category_str | published_time | ...

train/behaviors.parquet & validation/behaviors.parquet:
    impression_id | user_id | impression_time | article_ids_inview |
    article_ids_clicked | ...

train/history.parquet & validation/history.parquet:
    user_id | impression_time_fixed | article_id_fixed | scroll_percentage_fixed | ...

Usage:
    python src/parse_ebnerd.py --split train
    python src/parse_ebnerd.py --split validation
"""

import argparse
from pathlib import Path

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
    df = pl.read_parquet(behaviors_parquet)

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
    df = pl.read_parquet(history_parquet)
    if df.is_empty():
        return pl.DataFrame(
            schema={
                "dataset": pl.Utf8,
                "user_id": pl.Utf8,
                "clicked_article_id": pl.Utf8,
                "click_time": pl.Datetime,
            }
        )

    out = (
        df.select([
            pl.lit(DATASET).alias("dataset"),
            (pl.lit("ebnerd_") + pl.col("user_id").cast(pl.Utf8)).alias("user_id"),
            pl.col("article_id_fixed"),
            pl.col("impression_time_fixed"),
        ])
        .explode(["article_id_fixed", "impression_time_fixed"])
        .filter(pl.col("article_id_fixed").is_not_null())
        .select([
            pl.col("dataset"),
            pl.col("user_id"),
            (pl.lit("ebnerd_") + pl.col("article_id_fixed").cast(pl.Utf8)).alias("clicked_article_id"),
            pl.col("impression_time_fixed").cast(pl.Datetime).alias("click_time"),
        ])
    )
    return out


def main(split: str) -> None:
    raw_dir = Path("data/raw/ebnerd/demo") / split
    interim_dir = Path("data/interim/ebnerd")
    interim_dir.mkdir(parents=True, exist_ok=True)

    articles_path = Path("data/raw/ebnerd/demo/articles.parquet")
    if not articles_path.exists():
        articles_path = raw_dir.parent / "articles.parquet"

    articles = parse_articles(articles_path)
    impressions = parse_impressions(raw_dir / "behaviors.parquet")
    history = parse_history(raw_dir / "history.parquet")

    articles.write_parquet(interim_dir / f"articles_{split}.parquet")
    impressions.write_parquet(interim_dir / f"impressions_{split}.parquet")
    history.write_parquet(interim_dir / f"history_{split}.parquet")

    print(f"[ebnerd/{split}] articles={len(articles)} impressions={len(impressions)} "
          f"history_rows={len(history)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "validation"], required=True)
    args = parser.parse_args()
    main(args.split)
