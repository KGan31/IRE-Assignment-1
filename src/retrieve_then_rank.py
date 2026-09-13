"""
End-to-End Two-Stage Retrieve-then-Rank Pipeline (Assignment 2, Q2).

Stage 1: Candidate Retrieval (Top-K, K ~ 100-200) from full catalog using:
         - Semantic FAISS search (dense inner product over normalized embeddings)
         - BM25 lexical inverted index
         - Hybrid (Reciprocal Rank Fusion / RRF)
Stage 2: Re-Ranking with LightGBM over point-in-time engineered features.

Evaluates AUC, MRR, nDCG@5, and nDCG@10 before (Stage 1) and after (Stage 2).
"""

import argparse
from datetime import datetime
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import polars as pl
from tqdm import tqdm

try:
    import bm25s
    HAS_BM25S = True
except ImportError:
    HAS_BM25S = False

from bm25 import BM25InvertedIndex, create_article_text_map
from embeddings import EmbeddingIndex, compute_user_representation, normalize_l2
from features import HALF_LIFE_DEFAULT_HOURS
from metrics import compute_auc, compute_mrr, compute_ndcg_at_k


def reciprocal_rank_fusion(
    sem_ranked_pairs: List[Tuple[str, float]],
    bm25_ranked_pairs: List[Tuple[str, float]],
    k0: float = 60.0,
    top_k: int = 100,
) -> List[Tuple[str, float, float, float]]:
    """
    Fuses Semantic and BM25 candidate ranks using Reciprocal Rank Fusion (RRF).
    Returns list of tuples: (article_id, rrf_score, sem_score, bm25_score) sorted descending by rrf_score.
    """
    rrf_scores: Dict[str, float] = {}
    sem_scores: Dict[str, float] = {}
    bm25_scores: Dict[str, float] = {}

    for rank, (aid, score) in enumerate(sem_ranked_pairs, start=1):
        rrf_scores[aid] = rrf_scores.get(aid, 0.0) + (1.0 / (k0 + rank))
        sem_scores[aid] = float(score)

    for rank, (aid, score) in enumerate(bm25_ranked_pairs, start=1):
        rrf_scores[aid] = rrf_scores.get(aid, 0.0) + (1.0 / (k0 + rank))
        bm25_scores[aid] = float(score)

    sorted_aids = sorted(rrf_scores.keys(), key=lambda a: rrf_scores[a], reverse=True)[:top_k]
    return [
        (aid, rrf_scores[aid], sem_scores.get(aid, 0.0), bm25_scores.get(aid, 0.0))
        for aid in sorted_aids
    ]


class RetrieveThenRankPipeline:
    """
    Two-stage retrieval and re-ranking system.
    """

    def __init__(
        self,
        dataset: str,
        processed_dir: Path = Path("data/processed"),
        model_path: Optional[Path] = None,
        retriever_type: str = "hybrid",
        top_k: int = 100,
        half_life_hours: float = HALF_LIFE_DEFAULT_HOURS,
    ):
        self.dataset = dataset.lower()
        self.data_dir = processed_dir / self.dataset
        self.retriever_type = retriever_type.lower()
        self.top_k = top_k
        self.half_life_hours = half_life_hours

        # 1. Load articles catalog
        articles_path = self.data_dir / "articles.parquet"
        print(f"[{self.dataset.upper()}] Loading articles catalog from {articles_path}...")
        self.articles_pl = pl.read_parquet(articles_path)
        self.articles_pd = self.articles_pl.to_pandas()

        self.article_ids = self.articles_pl["article_id"].to_list()
        self.article_id_to_idx = {aid: idx for idx, aid in enumerate(self.article_ids)}
        self.article_cat_map = dict(zip(self.article_ids, self.articles_pl["category"].fill_null("").to_list()))
        
        pub_times = self.articles_pl["published_time"].to_list() if "published_time" in self.articles_pl.columns else [None] * len(self.article_ids)
        self.article_pub_map = dict(zip(self.article_ids, pub_times))

        # 2. Load dense embeddings & build FAISS index
        emb_path = self.data_dir / "article_embeddings.npy"
        if emb_path.exists():
            print(f"[{self.dataset.upper()}] Loading embeddings and building FAISS index from {emb_path}...")
            raw_embs = np.load(emb_path)
            self.embeddings_norm = normalize_l2(raw_embs)
            self.sem_index = EmbeddingIndex(use_approximate=False)
            self.sem_index.build_index(self.embeddings_norm, self.article_ids)
        else:
            self.embeddings_norm = None
            self.sem_index = None

        # 3. Build BM25 Index & token mapping for fast scoring
        print(f"[{self.dataset.upper()}] Indexing BM25 corpus over {len(self.articles_pd):,} articles...")
        self.bm25_retriever = None
        self.art_token_ids_map = {}
        if HAS_BM25S:
            titles = self.articles_pl["title"].fill_null("").to_list() if "title" in self.articles_pl.columns else [""] * len(self.article_ids)
            if "abstract" in self.articles_pl.columns:
                abstracts = self.articles_pl["abstract"].fill_null("").to_list()
            elif "subtitle" in self.articles_pl.columns:
                abstracts = self.articles_pl["subtitle"].fill_null("").to_list()
            else:
                abstracts = [""] * len(self.article_ids)
            full_texts = [f"{t} {a}".strip() for t, a in zip(titles, abstracts)]
            tokens = bm25s.tokenize(full_texts, show_progress=False)
            self.bm25_retriever = bm25s.BM25(k1=1.5, b=0.75)
            self.bm25_retriever.index(tokens, show_progress=False)
            self.art_token_ids_map = {aid: set(tokens.ids[i]) for i, aid in enumerate(self.article_ids)}

        self.bm25_index = BM25InvertedIndex(k1=1.5, b=0.75)
        self.bm25_index.build_index(self.articles_pd)
        self.article_text_map = create_article_text_map(self.articles_pd, fields="title_abstract")

        # 4. Load trained LightGBM Re-Ranker
        m_path = model_path if model_path else Path(f"models/{self.dataset}_lgbm_ranker.pkl")
        if not m_path.exists():
            raise FileNotFoundError(f"Trained re-ranker checkpoint not found at {m_path}. Train it first via src/reranker.py.")
        print(f"[{self.dataset.upper()}] Loading trained LightGBM Re-Ranker from {m_path}...")
        self.ranker = joblib.load(m_path)

        self.feature_cols = [
            "user_click_count",
            "user_recency_score",
            "user_mean_dwell_time",
            "article_freshness_hours",
            "article_popularity_24h",
            "category_affinity_score",
            "session_position",
            "bm25_score",
            "semantic_score",
            "first_stage_score",
        ]

    def stage1_retrieve(
        self,
        history_aids: List[str],
        history_weights: List[float],
        k: int = 100,
    ) -> List[Tuple[str, float, float, float, int]]:
        """
        Stage 1: Generates top-K candidates from the full catalog.
        Returns: [(article_id, stage1_prior_score, sem_score, bm25_score, session_pos), ...]
        """
        sem_pairs: List[Tuple[str, float]] = []
        if self.sem_index is not None and history_aids:
            user_vec = compute_user_representation(
                clicked_article_ids=history_aids,
                article_id_to_idx=self.article_id_to_idx,
                embeddings=self.embeddings_norm,
                weights=history_weights,
            )
            if user_vec is not None:
                sem_res = self.sem_index.batch_search(np.array([user_vec]), top_k=k)
                if sem_res:
                    sem_pairs = sem_res[0]

        bm25_pairs: List[Tuple[str, float]] = []
        if history_aids:
            q_texts = [self.article_text_map.get(aid, "") for aid in history_aids[-10:] if aid in self.article_text_map]
            query_str = " ".join(t for t in q_texts if t).strip()
            if query_str:
                bm25_res = self.bm25_index.batch_search([query_str], top_k=k)
                if bm25_res:
                    bm25_pairs = bm25_res[0]

        if self.retriever_type == "semantic":
            return [
                (aid, float(score), float(score), 0.0, rank_idx)
                for rank_idx, (aid, score) in enumerate(sem_pairs[:k])
            ]
        elif self.retriever_type == "bm25":
            return [
                (aid, float(score), 0.0, float(score), rank_idx)
                for rank_idx, (aid, score) in enumerate(bm25_pairs[:k])
            ]
        else:
            # Hybrid: RRF fusion
            fused = reciprocal_rank_fusion(sem_pairs, bm25_pairs, k0=60.0, top_k=k)
            return [
                (aid, rrf, sem, bm25, rank_idx)
                for rank_idx, (aid, rrf, sem, bm25) in enumerate(fused)
            ]

    def stage2_rerank(
        self,
        candidate_items: List[Tuple[str, float, float, float, int]],
        user_feats: Dict[str, Any],
        imp_time: Any,
    ) -> List[Tuple[str, float]]:
        """
        Stage 2: Extracts features for candidate articles and scores them using LightGBM.
        Returns: [(article_id, lgbm_score), ...] sorted descending.
        """
        if not candidate_items:
            return []

        feature_matrix = []
        c_ids = []

        for (cand_id, stage1_prior, sem_score, bm25_score, pos_idx) in candidate_items:
            cat = self.article_cat_map.get(cand_id, "")
            pub = self.article_pub_map.get(cand_id)
            freshness = 0.0
            if pub is not None and isinstance(pub, (datetime, pd.Timestamp)):
                try:
                    freshness = max(0.0, (imp_time - pub).total_seconds() / 3600.0)
                except Exception:
                    freshness = 0.0

            cat_aff = user_feats.get("category_affinity", {}).get(cat, 0.0)
            first_stage = max(bm25_score, sem_score)

            row = [
                user_feats.get("click_count", 0),
                user_feats.get("recency_score", 0.0),
                user_feats.get("mean_dwell_time", 0.0),
                float(freshness),
                0,  # popularity_24h
                float(cat_aff),
                pos_idx,
                float(bm25_score),
                float(sem_score),
                float(first_stage),
            ]
            feature_matrix.append(row)
            c_ids.append(cand_id)

        X = np.array(feature_matrix, dtype=np.float32)
        scores = self.ranker.predict(X)

        reranked = sorted(zip(c_ids, scores), key=lambda x: x[1], reverse=True)
        return reranked

    def run_impression(
        self,
        user_id: str,
        imp_time: Any,
        user_history_clicks: List[Tuple[str, Any, float]],
        candidate_subset: Optional[List[str]] = None,
        k: int = 100,
    ) -> Dict[str, Any]:
        """
        Executes Stage 1 Retrieval + Stage 2 Re-Ranking for a single impression request.
        """
        # Strict point-in-time filter: t_click < t_imp
        past_clicks = [c for c in user_history_clicks if c[1] < imp_time]
        n_clicks = len(past_clicks)
        history_aids = []
        history_weights = []
        recency_score = 0.0
        dwell_sum = 0.0
        cat_counts: Dict[str, int] = {}

        for aid, c_time, dwell in past_clicks:
            history_aids.append(aid)
            diff_hours = (imp_time - c_time).total_seconds() / 3600.0 if hasattr(imp_time, "__sub__") else 0.0
            w = 0.5 ** (diff_hours / self.half_life_hours)
            history_weights.append(w)
            recency_score += w
            dwell_sum += dwell
            cat = self.article_cat_map.get(aid, "")
            if cat:
                cat_counts[cat] = cat_counts.get(cat, 0) + 1

        user_feats = {
            "user_id": user_id,
            "click_count": n_clicks,
            "recency_score": recency_score,
            "mean_dwell_time": dwell_sum / max(1, n_clicks),
            "category_affinity": {k: v / max(1, n_clicks) for k, v in cat_counts.items()},
        }

        # Stage 1
        t0 = time.perf_counter()
        if candidate_subset is not None:
            # Score specific candidate list (inview session candidates)
            bm25_all_scores = None
            if self.bm25_retriever is not None and history_aids:
                user_q_tokens = set()
                for aid in history_aids[-20:]:
                    user_q_tokens.update(self.art_token_ids_map.get(aid, set()))
                if user_q_tokens:
                    bm25_all_scores = self.bm25_retriever.get_scores(list(user_q_tokens))

            user_vec = None
            if self.embeddings_norm is not None and history_aids:
                user_vec = compute_user_representation(
                    clicked_article_ids=history_aids,
                    article_id_to_idx=self.article_id_to_idx,
                    embeddings=self.embeddings_norm,
                    weights=history_weights,
                )

            stage1_candidates = []
            for orig_pos, cand_id in enumerate(candidate_subset):
                c_idx = self.article_id_to_idx.get(cand_id)
                sem_score = 0.0
                if user_vec is not None and c_idx is not None and c_idx < len(self.embeddings_norm):
                    sem_score = float(np.dot(user_vec, self.embeddings_norm[c_idx]))

                bm25_score = 0.0
                if bm25_all_scores is not None and c_idx is not None and c_idx < len(bm25_all_scores):
                    bm25_score = float(bm25_all_scores[c_idx])

                if self.retriever_type == "semantic":
                    prior = sem_score
                elif self.retriever_type == "bm25":
                    prior = bm25_score
                else:
                    prior = max(bm25_score, sem_score)

                stage1_candidates.append((cand_id, prior, sem_score, bm25_score, orig_pos))
        else:
            stage1_candidates = self.stage1_retrieve(history_aids, history_weights, k=k)

        t_stage1 = (time.perf_counter() - t0) * 1000.0

        # Stage 2
        t1 = time.perf_counter()
        stage2_ranked = self.stage2_rerank(stage1_candidates, user_feats, imp_time)
        t_stage2 = (time.perf_counter() - t1) * 1000.0

        return {
            "stage1_candidates": stage1_candidates,
            "stage2_ranked": stage2_ranked,
            "latency_ms": {
                "stage1_retrieval": t_stage1,
                "stage2_reranking": t_stage2,
                "total": t_stage1 + t_stage2,
            },
        }

    def retrieve_and_rerank(
        self,
        user_id: str,
        imp_time: Any,
        user_history_clicks: List[Tuple[str, Any, float]],
        top_k: int = 100,
        rerank_top_m: int = 10,
    ) -> List[Tuple[str, float]]:
        """
        Global catalog serving interface:
        1. Retrieves top-K candidates from the entire catalog via Stage 1.
        2. Re-ranks those candidates with LightGBM.
        3. Returns top-M final article recommendations.
        """
        out = self.run_impression(
            user_id=user_id,
            imp_time=imp_time,
            user_history_clicks=user_history_clicks,
            candidate_subset=None,
            k=top_k,
        )
        return out["stage2_ranked"][:rerank_top_m]


def evaluate_two_stage_pipeline(
    dataset: str,
    split: str = "val",
    sample_size: int = 2000,
    retriever_type: str = "hybrid",
    top_k: int = 100,
) -> Dict[str, Any]:
    """
    Evaluates the two-stage pipeline on validation impressions, measuring Stage 1 vs Stage 2 metrics.
    """
    pipeline = RetrieveThenRankPipeline(
        dataset=dataset,
        retriever_type=retriever_type,
        top_k=top_k,
    )

    data_dir = Path(f"data/processed/{dataset}")
    imp_file = data_dir / f"impressions_{split}.parquet"
    if not imp_file.exists() and split == "val":
        imp_file = data_dir / "impressions_validation.parquet"

    print(f"\nLoading validation impressions from {imp_file}...")
    impressions_df = pl.read_parquet(imp_file)
    if sample_size is not None and len(impressions_df) > sample_size:
        impressions_df = impressions_df.slice(0, sample_size)

    hist_file = data_dir / f"history_{split}.parquet"
    if not hist_file.exists() and split == "val":
        hist_file = data_dir / "history_validation.parquet"

    user_hist_map: Dict[str, List[Tuple[str, Any, float]]] = {}
    if hist_file.exists():
        print(f"Loading history from {hist_file}...")
        h_df = pl.read_parquet(hist_file)
        has_dwell = "dwell_time" in h_df.columns
        for row in h_df.select([
            "user_id",
            "clicked_article_id",
            "click_time",
            "dwell_time" if has_dwell else pl.lit(0.0).alias("dwell_time"),
        ]).iter_rows():
            uid, aid, c_time, dwell = row[0], row[1], row[2], row[3]
            if uid not in user_hist_map:
                user_hist_map[uid] = []
            user_hist_map[uid].append((aid, c_time, dwell or 0.0))

    # Evaluation accumulators
    stage1_auc_list = []
    stage1_mrr_list = []
    stage1_ndcg5_list = []
    stage1_ndcg10_list = []

    stage2_auc_list = []
    stage2_mrr_list = []
    stage2_ndcg5_list = []
    stage2_ndcg10_list = []

    latencies_ms = []
    latencies_stage1 = []
    latencies_stage2 = []

    print(f"\nRunning Two-Stage Retrieve-Then-Rank on {len(impressions_df):,} impressions (Retriever: {retriever_type.upper()})...")
    for imp in tqdm(impressions_df.iter_rows(named=True), total=len(impressions_df), desc="Pipeline Inference"):
        uid = imp["user_id"]
        ts = imp["timestamp"]
        cands = imp.get("candidate_article_ids") or []
        clicked = set(imp.get("clicked_article_ids") or [])
        if not cands or not clicked:
            continue

        u_clicks = user_hist_map.get(uid, [])
        out = pipeline.run_impression(
            user_id=uid,
            imp_time=ts,
            user_history_clicks=u_clicks,
            candidate_subset=cands,
            k=top_k,
        )
        latencies_ms.append(out["latency_ms"]["total"])
        latencies_stage1.append(out["latency_ms"]["stage1_retrieval"])
        latencies_stage2.append(out["latency_ms"]["stage2_reranking"])

        # Ground truth labels
        labels = [1 if c in clicked else 0 for c in cands]
        if not any(lbl == 1 for lbl in labels):
            continue

        # Stage 1 scores
        stage1_map = {item[0]: item[1] for item in out["stage1_candidates"]}
        stage1_scores = [stage1_map.get(c, 0.0) for c in cands]
        stage1_order = np.argsort(stage1_scores)[::-1]

        # Stage 2 scores
        stage2_map = dict(out["stage2_ranked"])
        stage2_scores = [stage2_map.get(c, -999.0) for c in cands]
        stage2_order = np.argsort(stage2_scores)[::-1]

        # Metrics
        auc1 = compute_auc(labels, stage1_scores)
        auc2 = compute_auc(labels, stage2_scores)
        if auc1 is not None and auc2 is not None:
            stage1_auc_list.append(auc1)
            stage2_auc_list.append(auc2)

        stage1_mrr_list.append(compute_mrr(labels, ranked_indices=stage1_order))
        stage2_mrr_list.append(compute_mrr(labels, ranked_indices=stage2_order))

        stage1_ndcg5_list.append(compute_ndcg_at_k(labels, k=5, ranked_indices=stage1_order))
        stage2_ndcg5_list.append(compute_ndcg_at_k(labels, k=5, ranked_indices=stage2_order))

        stage1_ndcg10_list.append(compute_ndcg_at_k(labels, k=10, ranked_indices=stage1_order))
        stage2_ndcg10_list.append(compute_ndcg_at_k(labels, k=10, ranked_indices=stage2_order))

    summary = {
        "dataset": dataset,
        "retriever_type": retriever_type,
        "sample_impressions": len(latencies_ms),
        "metrics": {
            "Stage 1 (Before Re-Ranking)": {
                "AUC": float(np.mean(stage1_auc_list)),
                "MRR": float(np.mean(stage1_mrr_list)),
                "nDCG@5": float(np.mean(stage1_ndcg5_list)),
                "nDCG@10": float(np.mean(stage1_ndcg10_list)),
            },
            "Stage 2 (After LightGBM)": {
                "AUC": float(np.mean(stage2_auc_list)),
                "MRR": float(np.mean(stage2_mrr_list)),
                "nDCG@5": float(np.mean(stage2_ndcg5_list)),
                "nDCG@10": float(np.mean(stage2_ndcg10_list)),
            },
            "Delta": {
                "AUC": float(np.mean(stage2_auc_list) - np.mean(stage1_auc_list)),
                "MRR": float(np.mean(stage2_mrr_list) - np.mean(stage1_mrr_list)),
                "nDCG@5": float(np.mean(stage2_ndcg5_list) - np.mean(stage1_ndcg5_list)),
                "nDCG@10": float(np.mean(stage2_ndcg10_list) - np.mean(stage1_ndcg10_list)),
            },
        },
        "latency_ms": {
            "p50": float(np.percentile(latencies_ms, 50)),
            "p95": float(np.percentile(latencies_ms, 95)),
            "p99": float(np.percentile(latencies_ms, 99)),
            "mean": float(np.mean(latencies_ms)),
            "stage1_mean": float(np.mean(latencies_stage1)),
            "stage2_mean": float(np.mean(latencies_stage2)),
        },
    }

    print("\n" + "=" * 70)
    print(f"TWO-STAGE PIPELINE EVALUATION: {dataset.upper()} ({retriever_type.upper()})")
    print("=" * 70)
    print(f"{'Metric':<12s} | {'Stage 1 (Prior)':<18s} | {'Stage 2 (Re-Ranker)':<18s} | {'Delta':<10s}")
    print("-" * 70)
    for m in ["AUC", "MRR", "nDCG@5", "nDCG@10"]:
        b = summary["metrics"]["Stage 1 (Before Re-Ranking)"][m]
        a = summary["metrics"]["Stage 2 (After LightGBM)"][m]
        d = summary["metrics"]["Delta"][m]
        print(f"{m:<12s} | {b:18.4f} | {a:18.4f} | {d:+10.4f}")
    print("-" * 70)
    print(f"Latency (Total):   p50={summary['latency_ms']['p50']:.2f}ms | p95={summary['latency_ms']['p95']:.2f}ms | p99={summary['latency_ms']['p99']:.2f}ms")
    print(f"Latency Breakdown: Stage 1={summary['latency_ms']['stage1_mean']:.2f}ms | Stage 2={summary['latency_ms']['stage2_mean']:.2f}ms")
    print("=" * 70)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Run Two-Stage Retrieve-then-Rank Pipeline.")
    parser.add_argument("--dataset", type=str, default="ebnerd", choices=["ebnerd", "mind"])
    parser.add_argument("--retriever", type=str, default="hybrid", choices=["hybrid", "semantic", "bm25"])
    parser.add_argument("--sample_size", type=int, default=2000)
    parser.add_argument("--top_k", type=int, default=100)
    parser.add_argument("--output_json", type=str, default=None)
    args = parser.parse_args()

    results = evaluate_two_stage_pipeline(
        dataset=args.dataset,
        retriever_type=args.retriever,
        sample_size=args.sample_size,
        top_k=args.top_k,
    )

    out_file = args.output_json if args.output_json else f"models/{args.dataset}_two_stage_pipeline_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Pipeline results saved to: {out_file}")


if __name__ == "__main__":
    main()
