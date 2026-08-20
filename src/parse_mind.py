"""
Parses raw MIND TSV files into the unified schema (see schema.py) using Polars.

MIND raw format (per official docs):

news.tsv (tab-separated, no header):
    News ID | Category | SubCategory | Title | Abstract | URL |
    Title Entities (JSON) | Abstract Entities (JSON)

behaviors.tsv (tab-separated, no header):
    Impression ID | User ID | Time | History (space-sep News IDs) |
    Impressions (space-sep "NewsID-label", label in {0,1})

Usage:
    python src/parse_mind.py --split train
    python src/parse_mind.py --split dev
"""

import argparse
import json
from pathlib import Path

import polars as pl

from schema import validate_articles, validate_impressions

DATASET = "mind"

NEWS_COLS = [
    "news_id", "category", "subcategory", "title", "abstract",
    "url", "title_entities", "abstract_entities",
]
BEHAVIOR_COLS = ["impression_id", "user_id", "time", "history", "impressions"]


def parse_entities(raw: str) -> list:
    """MIND stores entities as a JSON list of dicts with a 'Label' field."""
    if not isinstance(raw, str) or not raw.strip():
        return []
    try:
        parsed = json.loads(raw)
        return [e.get("Label", "") for e in parsed if e.get("Label")]
    except (json.JSONDecodeError, TypeError):
        return []


def parse_articles(news_tsv: Path) -> pl.DataFrame:
    df = pl.read_csv(
        news_tsv,
        separator="\t",
        has_header=False,
        new_columns=NEWS_COLS,
        quote_char=None,
        truncate_ragged_lines=True,
    )

    title_ents = pl.col("title_entities").map_elements(
        parse_entities, return_dtype=pl.List(pl.String)
    )
    abstract_ents = pl.col("abstract_entities").map_elements(
        parse_entities, return_dtype=pl.List(pl.String)
    )
    entities = title_ents.list.concat(abstract_ents)

    out = df.select([
        (pl.lit("mind_") + pl.col("news_id")).alias("article_id"),
        pl.lit(DATASET).alias("dataset"),
        pl.col("title").fill_null("").alias("title"),
        pl.col("abstract").fill_null("").alias("abstract"),
        pl.lit("").alias("body"),
        (pl.col("category").fill_null("") + "/" + pl.col("subcategory").fill_null("")).alias("category"),
        pl.lit(None).cast(pl.Datetime).alias("published_time"),
        entities.alias("entities"),
        pl.lit(None).alias("embedding"),
    ])
    validate_articles(out)
    return out


def parse_impressions_and_history(behaviors_tsv: Path):
    df = pl.read_csv(
        behaviors_tsv,
        separator="\t",
        has_header=False,
        new_columns=BEHAVIOR_COLS,
        quote_char=None,
        truncate_ragged_lines=True,
    )
    df = df.with_columns(
        pl.col("time").str.to_datetime(format="%m/%d/%Y %I:%M:%S %p").alias("timestamp")
    )

    candidates = (
        pl.col("impressions")
        .str.split(" ")
        .list.eval(
            pl.lit("mind_") + pl.element().str.replace(r"-[01]$", "")
        )
    )
    clicked = (
        pl.col("impressions")
        .str.split(" ")
        .list.eval(
            pl.element()
            .filter(pl.element().str.ends_with("-1"))
            .str.replace(r"-[01]$", "")
        )
        .list.eval(pl.lit("mind_") + pl.element())
    )

    impressions = df.select([
        (pl.lit("mind_") + pl.col("impression_id").cast(pl.Utf8)).alias("impression_id"),
        pl.lit(DATASET).alias("dataset"),
        (pl.lit("mind_") + pl.col("user_id")).alias("user_id"),
        pl.col("timestamp"),
        candidates.alias("candidate_article_ids"),
        clicked.alias("clicked_article_ids"),
        pl.lit(None).alias("session_context"),
    ])

    history = (
        df.filter(pl.col("history").is_not_null() & (pl.col("history").str.strip_chars() != ""))
        .select([
            pl.lit(DATASET).alias("dataset"),
            (pl.lit("mind_") + pl.col("user_id")).alias("user_id"),
            pl.col("history").str.split(" ").alias("clicked_article_id"),
            pl.col("timestamp").alias("click_time"),
        ])
        .explode("clicked_article_id")
        .with_columns((pl.lit("mind_") + pl.col("clicked_article_id")).alias("clicked_article_id"))
    )

    validate_impressions(impressions)
    return impressions, history


def find_file(base_dir: Path, filename: str) -> Path:
    direct = base_dir / filename
    if direct.exists():
        return direct
    matches = list(base_dir.glob(f"**/{filename}"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Could not find {filename} in {base_dir}")


def process_split(raw_base: Path, out_dir: Path, split: str) -> pl.DataFrame:
    split_dir = raw_base / split
    news_file = find_file(split_dir, "news.tsv")
    behaviors_file = find_file(split_dir, "behaviors.tsv")

    print(f"Parsing {split} from {split_dir}...")
    articles = parse_articles(news_file)
    impressions, history = parse_impressions_and_history(behaviors_file)

    out_dir.mkdir(parents=True, exist_ok=True)
    articles.write_parquet(out_dir / f"articles_{split}.parquet")
    impressions.write_parquet(out_dir / f"impressions_{split}.parquet")
    history.write_parquet(out_dir / f"history_{split}.parquet")

    print(f"[mind/{split}] articles={len(articles)} impressions={len(impressions)} history_rows={len(history)}")
    return articles


def main(split: str, dataset_type: str = "small") -> None:
    raw_dir_name = "mind_large" if dataset_type == "large" else "mind"
    raw_base = Path("data/raw") / raw_dir_name
    interim_dir = Path("data/interim") / raw_dir_name
    processed_dir = Path("data/processed") / raw_dir_name

    splits = ["train", "dev", "test"] if split == "all" else [split]
    
    all_articles = []
    for s in splits:
        if (raw_base / s).exists():
            art = process_split(raw_base, interim_dir, s)
            all_articles.append(art)

    if all_articles:
        print("Consolidating unique articles catalog...")
        combined_articles = pl.concat(all_articles).unique(subset=["article_id"])
        processed_dir.mkdir(parents=True, exist_ok=True)
        combined_articles.write_parquet(processed_dir / "articles.parquet")
        print(f"Total unique articles: {len(combined_articles)} saved to {processed_dir / 'articles.parquet'}")

        # Copy/link split files to processed_dir for direct consumption
        for s in splits:
            if (interim_dir / f"impressions_{s}.parquet").exists():
                imp = pl.read_parquet(interim_dir / f"impressions_{s}.parquet")
                imp.write_parquet(processed_dir / f"impressions_{s}.parquet")
                # Also alias dev as val if needed for standard evaluation harness
                if s == "dev":
                    imp.write_parquet(processed_dir / "impressions_val.parquet")
            if (interim_dir / f"history_{s}.parquet").exists():
                hist = pl.read_parquet(interim_dir / f"history_{s}.parquet")
                hist.write_parquet(processed_dir / f"history_{s}.parquet")
                if s == "dev":
                    hist.write_parquet(processed_dir / "history_val.parquet")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "dev", "test", "all"], default="all")
    parser.add_argument("--dataset_type", choices=["small", "large"], default="small")
    args = parser.parse_args()
    main(args.split, args.dataset_type)
