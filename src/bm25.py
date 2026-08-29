"""
BM25 Candidate Retrieval Engine powered by bm25s (with rank_bm25 fallback).

Implements Okapi BM25 candidate generation using the high-performance
bm25s library for batch tokenization and retrieval over article titles and abstracts.
"""

import math
import re
from typing import List, Tuple, Dict, Optional, Set
import numpy as np
import pandas as pd

try:
    import bm25s
    HAS_BM25S = True
except ImportError:
    bm25s = None
    HAS_BM25S = False

try:
    from rank_bm25 import BM25Okapi
    HAS_RANK_BM25 = True
except ImportError:
    BM25Okapi = None
    HAS_RANK_BM25 = False


def default_tokenize(text: str) -> List[str]:
    """Lowercase and extract alphanumeric word tokens."""
    if not text or pd.isna(text):
        return []
    return re.findall(r"\w+", str(text).lower())


def create_article_text_map(
    articles_df: pd.DataFrame, fields: str = "title"
) -> Dict[str, str]:
    """
    Build a mapping from article_id to query text representation based on selected fields.

    Args:
        articles_df: DataFrame containing at least 'article_id' and 'title', and optionally 'abstract'.
        fields: 'title' for title-only, or 'title_abstract' / 'title+abstract' / 'both' for title + abstract.

    Returns:
        Dict mapping article_id (as str) to text string.
    """
    if articles_df.empty or "article_id" not in articles_df.columns:
        return {}

    article_ids = articles_df["article_id"].astype(str).tolist()
    titles = articles_df["title"].fillna("").astype(str).tolist() if "title" in articles_df.columns else [""] * len(article_ids)

    norm_fields = str(fields).lower().replace("+", "_").replace(" ", "_")
    if norm_fields in ("title_abstract", "title_and_abstract", "both", "abstract_title"):
        abstracts = (
            articles_df["abstract"].fillna("").astype(str).tolist()
            if "abstract" in articles_df.columns
            else (
                articles_df["subtitle"].fillna("").astype(str).tolist()
                if "subtitle" in articles_df.columns
                else [""] * len(article_ids)
            )
        )
        return {
            aid: f"{t} {a}".strip()
            for aid, t, a in zip(article_ids, titles, abstracts)
        }
    else:
        return {aid: t.strip() for aid, t in zip(article_ids, titles)}


class BM25InvertedIndex:
    """High-performance BM25 Index wrapper using bm25s or rank-bm25."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.article_ids: List[str] = []
        self.article_id_to_idx: Dict[str, int] = {}
        self.retriever: Optional[object] = None
        self.is_built: bool = False

    @property
    def vocab(self) -> Dict[str, int]:
        """Return vocabulary mapping."""
        if HAS_BM25S and self.retriever is not None and hasattr(self.retriever, "vocab_dict"):
            return self.retriever.vocab_dict
        return {}

    def build_index(self, articles_df: pd.DataFrame, include_body: bool = False) -> None:
        """Build BM25 index over article text.

        Args:
            articles_df: DataFrame with at least 'article_id' and 'title' columns.
            include_body: If True and a 'body' column exists, append body text to
                the indexed document string (title + abstract + body).  Useful for
                the EB-NeRD body-text ablation.  Has no effect when the column is
                absent.
        """
        if articles_df.empty:
            raise ValueError("Cannot build BM25 index on empty DataFrame")

        self.article_ids = articles_df["article_id"].tolist()
        self.article_id_to_idx = {aid: idx for idx, aid in enumerate(self.article_ids)}
        num_docs = len(self.article_ids)

        titles = articles_df["title"].fillna("").astype(str).tolist()
        # EB-NeRD uses 'subtitle' as its abstract-equivalent field.
        abstracts = (
            articles_df["abstract"].fillna("").astype(str).tolist()
            if "abstract" in articles_df.columns
            else (
                articles_df["subtitle"].fillna("").astype(str).tolist()
                if "subtitle" in articles_df.columns
                else [""] * num_docs
            )
        )

        if include_body and "body" in articles_df.columns:
            bodies = articles_df["body"].fillna("").astype(str).tolist()
            full_texts = [f"{t} {a} {b}".strip() for t, a, b in zip(titles, abstracts, bodies)]
        else:
            full_texts = [f"{t} {a}".strip() for t, a in zip(titles, abstracts)]

        if HAS_BM25S:
            corpus_tokens = bm25s.tokenize(full_texts, show_progress=False)
            retriever = bm25s.BM25(k1=self.k1, b=self.b)
            retriever.index(corpus_tokens, show_progress=False)
            self.retriever = retriever
        elif HAS_RANK_BM25:
            tokenized_corpus = [default_tokenize(txt) for txt in full_texts]
            self.retriever = BM25Okapi(tokenized_corpus, k1=self.k1, b=self.b)
        else:
            raise ImportError("Please install bm25s or rank-bm25: pip install bm25s rank-bm25")

        self.is_built = True

    def batch_search(
        self,
        query_texts: List[str],
        top_k: int = 200,
    ) -> List[List[Tuple[str, float]]]:
        """Batch retrieve top_k (article_id, score) for a list of query strings."""
        if not self.is_built or not self.article_ids or not query_texts:
            return [[] for _ in query_texts]

        if HAS_BM25S and self.retriever is not None:
            query_tokens = bm25s.tokenize(query_texts, show_progress=False)
            k = min(top_k, len(self.article_ids))
            results_idx, scores = self.retriever.retrieve(
                query_tokens, k=k, show_progress=False
            )

            batch_results: List[List[Tuple[str, float]]] = []
            for doc_indices, score_row in zip(results_idx, scores):
                row_res: List[Tuple[str, float]] = []
                for doc_idx, score in zip(doc_indices, score_row):
                    if score > 0.0:
                        row_res.append((self.article_ids[doc_idx], float(score)))
                batch_results.append(row_res)
            return batch_results

        elif HAS_RANK_BM25 and self.retriever is not None:
            batch_results = []
            num_docs = len(self.article_ids)
            k = min(top_k, num_docs)
            for q_text in query_texts:
                tokens = default_tokenize(q_text)
                if not tokens:
                    batch_results.append([])
                    continue
                scores = self.retriever.get_scores(tokens)
                top_indices = np.argsort(-scores)[:k]
                row_res = [
                    (self.article_ids[idx], float(scores[idx]))
                    for idx in top_indices
                    if scores[idx] > 0.0
                ]
                batch_results.append(row_res)
            return batch_results

        return [[] for _ in query_texts]

    def search(
        self,
        query_text: str,
        top_k: int = 200,
        candidate_ids: Optional[List[str]] = None,
    ) -> List[Tuple[str, float]]:
        """Single query search helper with optional candidate filtering."""
        batch_res = self.batch_search([query_text], top_k=top_k)
        results = batch_res[0] if batch_res else []

        if candidate_ids is not None:
            cand_set = set(candidate_ids)
            results = [(aid, score) for aid, score in results if aid in cand_set]
            return results[:top_k]

        return results
