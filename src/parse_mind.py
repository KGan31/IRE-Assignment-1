"""
Parses raw MIND TSV files into the unified schema (see schema.py).

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

import pandas as pd

from schema import make_article_id, make_user_id, validate_articles, validate_impressions

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


def parse_articles(news_tsv: Path) -> pd.DataFrame:
    df = pd.read_csv(news_tsv, sep="\t", header=None, names=NEWS_COLS, quoting=3)

    out = pd.DataFrame({
        "article_id": df["news_id"].map(lambda x: make_article_id(DATASET, x)),
        "dataset": DATASET,
        "title": df["title"].fillna(""),
        "abstract": df["abstract"].fillna(""),
        "body": "",  # MIND does not ship full body text
        "category": df["category"].fillna("") + "/" + df["subcategory"].fillna(""),
        "published_time": pd.NaT,  # MIND does not provide per-article publish time
        "entities": (
            df["title_entities"].map(parse_entities)
            + df["abstract_entities"].map(parse_entities)
        ),
        "embedding": None,
    })
    validate_articles(out)
    return out


def parse_impressions_and_history(behaviors_tsv: Path):
    df = pd.read_csv(behaviors_tsv, sep="\t", header=None, names=BEHAVIOR_COLS)
    df["timestamp"] = pd.to_datetime(df["time"], format="%m/%d/%Y %I:%M:%S %p")

    impression_rows = []
    history_rows = []

    for row in df.itertuples(index=False):
        user_id = make_user_id(DATASET, row.user_id)

        # --- click history (articles clicked before this impression) ---
        if isinstance(row.history, str) and row.history.strip():
            for news_id in row.history.split():
                history_rows.append({
                    "dataset": DATASET,
                    "user_id": user_id,
                    "clicked_article_id": make_article_id(DATASET, news_id),
                    # MIND does not timestamp individual history clicks;
                    # approximate with the impression time as an upper bound.
                    "click_time": row.timestamp,
                })

        # --- impression candidates + labels ---
        candidates, clicked = [], []
        if isinstance(row.impressions, str):
            for token in row.impressions.split():
                news_id, label = token.rsplit("-", 1)
                aid = make_article_id(DATASET, news_id)
                candidates.append(aid)
                if label == "1":
                    clicked.append(aid)

        impression_rows.append({
            "impression_id": make_article_id(DATASET, str(row.impression_id)),
            "dataset": DATASET,
            "user_id": user_id,
            "timestamp": row.timestamp,
            "candidate_article_ids": candidates,
            "clicked_article_ids": clicked,
            "session_context": {},
        })

    impressions = pd.DataFrame(impression_rows)
    history = pd.DataFrame(history_rows)

    validate_impressions(impressions)
    return impressions, history


def main(split: str) -> None:
    raw_dir = Path("data/raw/mind") / split
    dataset_root = raw_dir / f"MINDsmall_{split}"  # train -> MINDsmall_train


    interim_dir = Path("data/interim/mind")
    interim_dir.mkdir(parents=True, exist_ok=True)

    articles = parse_articles(dataset_root / "news.tsv")
    impressions, history = parse_impressions_and_history(dataset_root / "behaviors.tsv")

    articles.to_parquet(interim_dir / f"articles_{split}.parquet")
    impressions.to_parquet(interim_dir / f"impressions_{split}.parquet")
    history.to_parquet(interim_dir / f"history_{split}.parquet")

    print(f"[mind/{split}] articles={len(articles)} impressions={len(impressions)} "
          f"history_rows={len(history)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "dev"], required=True)
    args = parser.parse_args()
    main(args.split)
