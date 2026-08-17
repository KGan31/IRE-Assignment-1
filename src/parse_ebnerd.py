"""
Parses raw EB-NeRD parquet files into the unified schema (see schema.py).

EB-NeRD demo bundle ships (approximate schema — verify against your
downloaded version, field names have shifted slightly across releases):

articles.parquet:
    article_id | title | subtitle | body | category_str | published_time | ...

train/behaviors.parquet & validation/behaviors.parquet:
    impression_id | user_id | impression_time | article_ids_inview |
    article_ids_clicked | ...

train/history.parquet & validation/history.parquet:
    user_id | impression_time_fixed | article_id_fixed | scroll_percentage_fixed | ...
    (the *_fixed columns are parallel lists: article_id_fixed[i] was
    clicked at impression_time_fixed[i])

NOTE: column names occasionally differ between demo/small/large bundles.
If pd.read_parquet + the column selects below throw a KeyError, run
`df.columns.tolist()` first and adjust COLUMN maps below — the parsing
logic itself does not need to change.

Usage:
    python src/parse_ebnerd.py --split train
    python src/parse_ebnerd.py --split validation
"""

import argparse
from pathlib import Path

import pandas as pd

from schema import make_article_id, make_user_id, validate_articles, validate_impressions

DATASET = "ebnerd"


def parse_articles(articles_parquet: Path) -> pd.DataFrame:
    df = pd.read_parquet(articles_parquet)

    def col(name, default=""):
        return df[name] if name in df.columns else pd.Series([default] * len(df))

    out = pd.DataFrame({
        "article_id": df["article_id"].map(lambda x: make_article_id(DATASET, x)),
        "dataset": DATASET,
        "title": col("title").fillna(""),
        "abstract": col("subtitle").fillna(""),
        "body": col("body").fillna(""),
        "category": col("category_str").fillna(""),
        "published_time": pd.to_datetime(col("published_time"), errors="coerce"),
        "entities": [[] for _ in range(len(df))],  # EB-NeRD has no entity annotations natively
        "embedding": None,  # filled in separately from the word2vec/BERT artifact files, if used
    })
    validate_articles(out)
    return out


def parse_impressions(behaviors_parquet: Path) -> pd.DataFrame:
    df = pd.read_parquet(behaviors_parquet)

    rows = []
    for row in df.itertuples(index=False):
        inview = list(getattr(row, "article_ids_inview", []) or [])
        clicked = list(getattr(row, "article_ids_clicked", []) or [])

        rows.append({
            "impression_id": make_article_id(DATASET, str(row.impression_id)),
            "dataset": DATASET,
            "user_id": make_user_id(DATASET, row.user_id),
            "timestamp": row.impression_time,
            "candidate_article_ids": [make_article_id(DATASET, a) for a in inview],
            "clicked_article_ids": [make_article_id(DATASET, a) for a in clicked],
            "session_context": {},
        })

    out = pd.DataFrame(rows)
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    validate_impressions(out)
    return out


def parse_history(history_parquet: Path) -> pd.DataFrame:
    df = pd.read_parquet(history_parquet)

    rows = []
    for row in df.itertuples(index=False):
        article_ids = list(getattr(row, "article_id_fixed", []) or [])
        times = list(getattr(row, "impression_time_fixed", []) or [])
        user_id = make_user_id(DATASET, row.user_id)

        for aid, t in zip(article_ids, times):
            rows.append({
                "dataset": DATASET,
                "user_id": user_id,
                "clicked_article_id": make_article_id(DATASET, aid),
                "click_time": t,
            })

    out = pd.DataFrame(rows)
    if not out.empty:
        out["click_time"] = pd.to_datetime(out["click_time"])
    return out


def main(split: str) -> None:
    raw_dir = Path("data/raw/ebnerd/demo") / split
    interim_dir = Path("data/interim/ebnerd")
    interim_dir.mkdir(parents=True, exist_ok=True)

    articles_path = Path("data/raw/ebnerd/demo/articles.parquet")
    if not articles_path.exists():
        # some bundles place articles.parquet at the top level, shared across splits
        articles_path = raw_dir.parent / "articles.parquet"

    articles = parse_articles(articles_path)
    impressions = parse_impressions(raw_dir / "behaviors.parquet")
    history = parse_history(raw_dir / "history.parquet")

    articles.to_parquet(interim_dir / f"articles_{split}.parquet")
    impressions.to_parquet(interim_dir / f"impressions_{split}.parquet")
    history.to_parquet(interim_dir / f"history_{split}.parquet")

    print(f"[ebnerd/{split}] articles={len(articles)} impressions={len(impressions)} "
          f"history_rows={len(history)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "validation"], required=True)
    args = parser.parse_args()
    main(args.split)
