"""
Unified schema for MIND + EB-NeRD news recommendation datasets.

Both parse_mind.py and parse_ebnerd.py must produce DataFrames that
conform to these column contracts. Downstream code (split, feature_store,
retrieval, eval) only ever talks to THIS schema — it never touches
dataset-specific fields directly.
"""

from dataclasses import dataclass, field
from typing import Optional
import pandas as pd


# ---------------------------------------------------------------------------
# Column contracts (used for validation, not enforced at runtime by pandas)
# ---------------------------------------------------------------------------

ARTICLE_COLUMNS = {
    "article_id": "str",        # prefixed e.g. 'mind_N12345' / 'ebnerd_9876543'
    "dataset": "str",           # 'mind' | 'ebnerd'
    "title": "str",
    "abstract": "str",
    "body": "str",              # may be empty string if unavailable
    "category": "str",
    "published_time": "datetime64[ns]",
    "entities": "object",       # list[str], empty list if unavailable
    "embedding": "object",      # np.ndarray or None
}

IMPRESSION_COLUMNS = {
    "impression_id": "str",
    "dataset": "str",
    "user_id": "str",
    "timestamp": "datetime64[ns]",
    "candidate_article_ids": "object",   # list[str]
    "clicked_article_ids": "object",     # list[str], subset of candidates
    "session_context": "object",         # dict
}

USER_HISTORY_COLUMNS = {
    "dataset": "str",
    "user_id": "str",
    "clicked_article_id": "str",
    "click_time": "datetime64[ns]",
}


def make_article_id(dataset: str, raw_id: str) -> str:
    """Namespace article ids so MIND and EB-NeRD never collide if combined."""
    return f"{dataset}_{raw_id}"


def make_user_id(dataset: str, raw_id: str) -> str:
    return f"{dataset}_{raw_id}"


from typing import Optional, Union
import pandas as pd
import polars as pl


def validate_articles(df: Union[pd.DataFrame, pl.DataFrame]) -> None:
    _validate_columns(df, ARTICLE_COLUMNS, "articles")


def validate_impressions(df: Union[pd.DataFrame, pl.DataFrame]) -> None:
    _validate_columns(df, IMPRESSION_COLUMNS, "impressions")


def validate_user_history(df: Union[pd.DataFrame, pl.DataFrame]) -> None:
    _validate_columns(df, USER_HISTORY_COLUMNS, "user_history")


def _validate_columns(df: Union[pd.DataFrame, pl.DataFrame], expected: dict, name: str) -> None:
    cols = set(df.columns)
    missing = set(expected.keys()) - cols
    if missing:
        raise ValueError(f"[{name}] missing required columns: {missing}")
    is_empty = df.is_empty() if hasattr(df, "is_empty") else df.empty
    if is_empty:
        raise ValueError(f"[{name}] dataframe is empty")


# ---------------------------------------------------------------------------
# Lightweight in-memory record types (handy for tests / small utilities)
# ---------------------------------------------------------------------------

@dataclass
class Article:
    article_id: str
    dataset: str
    title: str
    abstract: str = ""
    body: str = ""
    category: str = ""
    published_time: Optional[pd.Timestamp] = None
    entities: list = field(default_factory=list)
    embedding: Optional[object] = None


@dataclass
class Impression:
    impression_id: str
    dataset: str
    user_id: str
    timestamp: pd.Timestamp
    candidate_article_ids: list
    clicked_article_ids: list
    session_context: dict = field(default_factory=dict)
