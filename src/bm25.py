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
        fallback_popular_ids: Optional[List[str]] = None,
    ) -> List[List[Tuple[str, float]]]:
        """
        Batch retrieve top_k (article_id, score) for a list of query strings.
        If a query is empty (cold-start user with no clicks), falls back to top popular articles.
        """
        if not self.is_built or not self.article_ids or not query_texts:
            return [[] for _ in query_texts]

        num_docs = len(self.article_ids)
        k = min(top_k, num_docs)

        # Precompute fallback popular pairs if provided
        popular_pairs: List[Tuple[str, float]] = []
        if fallback_popular_ids:
            for rank_idx, pop_id in enumerate(fallback_popular_ids[:k]):
                # Assign decaying score based on popularity rank
                popular_pairs.append((pop_id, 1.0 / (rank_idx + 1.0)))

        if HAS_BM25S and self.retriever is not None:
            query_tokens = bm25s.tokenize(query_texts, show_progress=False)
            results_idx, scores = self.retriever.retrieve(
                query_tokens, k=k, show_progress=False
            )

            batch_results: List[List[Tuple[str, float]]] = []
            for q_idx, (doc_indices, score_row) in enumerate(zip(results_idx, scores)):
                row_res: List[Tuple[str, float]] = []
                for doc_idx, score in zip(doc_indices, score_row):
                    if score > 0.0:
                        row_res.append((self.article_ids[doc_idx], float(score)))

                # If query is empty / no positive BM25 scores (cold start), use popular articles fallback
                if not row_res and popular_pairs:
                    row_res = list(popular_pairs)

                batch_results.append(row_res)
            return batch_results

        elif HAS_RANK_BM25 and self.retriever is not None:
            batch_results = []
            for q_text in query_texts:
                tokens = default_tokenize(q_text)
                if not tokens:
                    batch_results.append(list(popular_pairs) if popular_pairs else [])
                    continue
                scores = self.retriever.get_scores(tokens)
                top_indices = np.argsort(-scores)[:k]
                row_res = [
                    (self.article_ids[idx], float(scores[idx]))
                    for idx in top_indices
                    if scores[idx] > 0.0
                ]
                if not row_res and popular_pairs:
                    row_res = list(popular_pairs)
                batch_results.append(row_res)
            return batch_results

        return [[] for _ in query_texts]

    def search(
        self,
        query_text: str,
        top_k: int = 200,
        candidate_ids: Optional[List[str]] = None,
        fallback_popular_ids: Optional[List[str]] = None,
        popularity_map: Optional[Dict[str, float]] = None,
    ) -> List[Tuple[str, float]]:
        """Single query search helper with optional candidate filtering and popularity fallback."""
        batch_res = self.batch_search(
            [query_text], top_k=top_k, fallback_popular_ids=fallback_popular_ids
        )
        results = batch_res[0] if batch_res else []

        if candidate_ids is not None:
            cand_set = set(candidate_ids)
            filtered = [(aid, score) for aid, score in results if aid in cand_set]

            # If cold start or no candidates matched, rank candidate_ids by popularity
            if not filtered and candidate_ids:
                if popularity_map:
                    # Sort candidate_ids by popularity descending
                    sorted_cands = sorted(
                        candidate_ids,
                        key=lambda aid: popularity_map.get(aid, 0.0),
                        reverse=True,
                    )
                elif fallback_popular_ids:
                    pop_rank = {aid: idx for idx, aid in enumerate(fallback_popular_ids)}
                    sorted_cands = sorted(
                        candidate_ids,
                        key=lambda aid: pop_rank.get(aid, 999999),
                    )
                else:
                    sorted_cands = list(candidate_ids)

                filtered = [
                    (aid, float(popularity_map.get(aid, 0.0) if popularity_map else (1.0 / (idx + 1.0))))
                    for idx, aid in enumerate(sorted_cands)
                ]

            return filtered[:top_k]

        return results

