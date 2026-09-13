"""
Evaluate EB-NeRD Large Validation Set using the exact same logical function
and feature extraction pipeline as the test re-ranker submission generator.

Computes official ranking metrics:
- Impression-level AUC
- MRR (Mean Reciprocal Rank)
- nDCG@5
- nDCG@10
Compares First-Stage baseline against LightGBM Re-Ranker.
"""

import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["POLARS_MAX_THREADS"] = "4"

import argparse
from datetime import datetime
import gc
import json
from pathlib import Path
import sys
import time
import warnings
from typing import Any, Dict, List, Optional, Set, Tuple

import joblib
import numpy as np
import pandas as pd
import polars as pl
import scipy.sparse as sp
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")

# Ensure src is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from embeddings import compute_user_representation, normalize_l2
from metrics import compute_auc, compute_mrr, compute_ndcg_at_k
from reranker import FEATURE_COLS


class UserProfile:
    """Compact user profile representation (~400 bytes) avoiding millions of Python objects."""
    __slots__ = ("n_clicks", "mean_dwell", "cat_affinity", "user_vec", "q_tokens")

    def __init__(
        self,
        n_clicks: int,
        mean_dwell: float,
        cat_affinity: Dict[str, float],
        user_vec: Optional[np.ndarray],
        q_tokens: List[int],
    ):
        self.n_clicks = n_clicks
        self.mean_dwell = mean_dwell
        self.cat_affinity = cat_affinity
        self.user_vec = user_vec
        self.q_tokens = q_tokens


def build_ebnerd_user_profiles(
    raw_hist_path: Path,
    processed_hist_path: Path,
    user_ids: List[str],
    embeddings_norm: Optional[np.ndarray],
    article_id_to_idx: Dict[str, int],
    art_token_ids_map: Dict[str, Set[int]],
    art_cat_map: Dict[str, str],
    max_history_len: int = 30,
) -> Dict[str, UserProfile]:
    """
    Builds compact user profiles for active EB-NeRD users by reading history with fast
    list-column extraction from raw parquet (or fallback to processed history)
    and capping per-user history to max_history_len.
    """
    print(f"Precomputing user profiles for {len(user_ids):,} active users...")
    start_t = time.time()
    user_set = set(user_ids)

    user_aids_map: Dict[str, List[str]] = {}
    user_dwells_map: Dict[str, List[float]] = {}

    if raw_hist_path.exists():
        print(f"  - Reading fast list-column user history from {raw_hist_path}...")
        try:
            raw_hist = pl.read_parquet(raw_hist_path, columns=["user_id", "article_id_fixed", "read_time_fixed"])
            # Filter users directly
            raw_hist = raw_hist.with_columns(
                (pl.lit("ebnerd_") + pl.col("user_id").cast(pl.Utf8)).alias("uid_str")
            ).filter(pl.col("uid_str").is_in(user_set))

            capped = raw_hist.select([
                pl.col("uid_str").alias("user_id"),
                pl.col("article_id_fixed").list.tail(max_history_len).list.eval(pl.lit("ebnerd_") + pl.element().cast(pl.Utf8)).alias("recent_clicks"),
                pl.col("read_time_fixed").list.tail(max_history_len).alias("recent_dwells")
            ])
            del raw_hist
            gc.collect()

            for uid, aids, dwells in zip(capped["user_id"].to_list(), capped["recent_clicks"].to_list(), capped["recent_dwells"].to_list()):
                user_aids_map[uid] = aids or []
                user_dwells_map[uid] = dwells or []
            del capped
            gc.collect()
            print(f"  - Extracted history for {len(user_aids_map):,} users from raw parquet.")
        except Exception as e:
            print(f"  - [warning] Raw history load failed ({e}), falling back to processed history.")
            user_aids_map.clear()
            user_dwells_map.clear()

    if not user_aids_map and processed_hist_path.exists():
        print(f"  - Scanning processed history from {processed_hist_path}...")
        uids_df = pl.DataFrame({"user_id": list(user_set)})
        capped = (
            pl.scan_parquet(processed_hist_path)
            .join(uids_df.lazy(), on="user_id", how="semi")
            .collect()
            .sort("click_time")
            .group_by("user_id")
            .tail(max_history_len)
        )
        has_dwell = "dwell_time" in capped.columns
        agg_exprs = [pl.col("clicked_article_id").alias("aids")]
        if has_dwell:
            agg_exprs.append(pl.col("dwell_time").fill_null(0.0).alias("dwells"))
        user_agg = capped.group_by("user_id").agg(agg_exprs)
        del capped
        gc.collect()

        uids = user_agg["user_id"].to_list()
        aids_list = user_agg["aids"].to_list()
        dwells_list = user_agg["dwells"].to_list() if has_dwell else [[0.0] * len(a) for a in aids_list]
        del user_agg
        gc.collect()

        for uid, aids, dwells in zip(uids, aids_list, dwells_list):
            user_aids_map[uid] = aids or []
            user_dwells_map[uid] = dwells or []

    # Construct compact UserProfile objects
    user_profiles: Dict[str, UserProfile] = {}
    for uid in user_ids:
        aids = user_aids_map.get(uid, [])
        dwells = user_dwells_map.get(uid, [])
        n = len(aids)
        md = sum(dwells) / n if n and dwells else 0.0

        # Query tokens for BM25 from the latest 20 articles
        recent = aids[-20:]
        q_toks = set()
        for a in recent:
            q_toks.update(art_token_ids_map.get(a, set()))

        # Category distribution
        cat_counts: Dict[str, int] = {}
        for a in aids:
            c = art_cat_map.get(a, "")
            if c:
                cat_counts[c] = cat_counts.get(c, 0) + 1
        cat_aff = {k: v / n for k, v in cat_counts.items()} if n else {}

        # Semantic user vector
        u_vec = None
        if embeddings_norm is not None and recent:
            u_vec = compute_user_representation(
                clicked_article_ids=recent,
                article_id_to_idx=article_id_to_idx,
                embeddings=embeddings_norm,
            )

        user_profiles[uid] = UserProfile(
            n_clicks=n,
            mean_dwell=float(md),
            cat_affinity=cat_aff,
            user_vec=u_vec,
            q_tokens=list(q_toks),
        )

    print(f"Successfully precomputed {len(user_profiles):,} user profiles in {time.time() - start_t:.2f}s.")
    return user_profiles


def evaluate_ebnerd_validation(
    model: Any,
    val_impressions_path: Path,
    articles_df: pl.DataFrame,
    raw_val_hist_path: Path,
    processed_val_hist_path: Path,
    embeddings: Optional[np.ndarray],
    article_id_to_idx: Optional[Dict[str, int]],
    max_impressions: Optional[int] = 50000,
    batch_predict_size: int = 5000,
) -> Dict[str, Any]:
    """
    Evaluates validation impressions using the exact test ranking logic and scores against ground truth.
    """
    print(f"\n{'='*65}")
    print("  EB-NeRD LARGE VALIDATION EVALUATION (EXACT TEST LOGIC)")
    print(f"{'='*65}")
    print(f"Loading validation impressions from {val_impressions_path}...")

    if max_impressions is not None:
        val_df = pl.read_parquet(val_impressions_path, n_rows=max_impressions)
        total_val = len(val_df)
    else:
        val_df = pl.read_parquet(val_impressions_path)
        total_val = len(val_df)
    print(f"Evaluating on {total_val:,} validation impressions...")

    # 1. Pre-normalize embeddings
    embeddings_norm = None
    if embeddings is not None:
        embeddings_norm = np.ascontiguousarray(normalize_l2(embeddings), dtype=np.float32)

    # 2. Pre-build article metadata lookups
    art_ids = articles_df["article_id"].to_list()
    if article_id_to_idx is None:
        article_id_to_idx = {aid: idx for idx, aid in enumerate(art_ids)}

    categories = articles_df["category"].fill_null("").to_list() if "category" in articles_df.columns else [""] * len(art_ids)
    art_cat_map = dict(zip(art_ids, categories))
    pub_times = articles_df["published_time"].to_list() if "published_time" in articles_df.columns else [None] * len(art_ids)
    art_pub_map = dict(zip(art_ids, pub_times))

    # 3. Pre-build BM25 index and candidate-targeted doc-token score matrix
    doc_token_scores = None
    art_token_ids_map: Dict[str, Set[int]] = {}
    try:
        import bm25s
        titles = articles_df["title"].fill_null("").to_list() if "title" in articles_df.columns else [""] * len(art_ids)
        if "abstract" in articles_df.columns:
            abstracts = articles_df["abstract"].fill_null("").to_list()
        elif "subtitle" in articles_df.columns:
            abstracts = articles_df["subtitle"].fill_null("").to_list()
        else:
            abstracts = [""] * len(art_ids)
        full_texts = [f"{t} {a}".strip() for t, a in zip(titles, abstracts)]
        tokens = bm25s.tokenize(full_texts, show_progress=False)
        bm25_retriever = bm25s.BM25()
        bm25_retriever.index(tokens, show_progress=False)
        art_token_ids_map = {aid: set(tokens.ids[i]) for i, aid in enumerate(art_ids)}

        # Transpose token x doc CSR to doc x token CSR for fast candidate-targeted scoring
        csr_token_doc = sp.csr_matrix(
            (bm25_retriever.scores["data"], bm25_retriever.scores["indices"], bm25_retriever.scores["indptr"]),
            shape=(len(bm25_retriever.scores["indptr"]) - 1, bm25_retriever.scores["num_docs"])
        )
        doc_token_scores = csr_token_doc.T.tocsr()
        print("Candidate-targeted BM25 sparse matrix successfully initialized.")
    except Exception as e:
        print(f"BM25 initialization warning: {e}")
        doc_token_scores = None

    # 4. Find unique users in validation sample and build compact UserProfiles
    val_uids = val_df["user_id"].unique().to_list()
    user_profiles = build_ebnerd_user_profiles(
        raw_hist_path=raw_val_hist_path,
        processed_hist_path=processed_val_hist_path,
        user_ids=val_uids,
        embeddings_norm=embeddings_norm,
        article_id_to_idx=article_id_to_idx,
        art_token_ids_map=art_token_ids_map,
        art_cat_map=art_cat_map,
    )

    # 5. Score impressions and accumulate predictions & ground truth
    impr_ids = val_df["impression_id"].to_list()
    user_ids = val_df["user_id"].to_list()
    timestamps = val_df["timestamp"].to_list()
    candidates_list = val_df["candidate_article_ids"].to_list()
    clicked_list = val_df["clicked_article_ids"].to_list()
    del val_df
    gc.collect()

    base_auc_list, ranker_auc_list = [], []
    base_mrr_list, ranker_mrr_list = [], []
    base_ndcg5_list, ranker_ndcg5_list = [], []
    base_ndcg10_list, ranker_ndcg10_list = [], []

    batch_X: List[np.ndarray] = []
    batch_meta: List[Tuple[np.ndarray, np.ndarray, int]] = []  # (base_scores, labels, n_cand)

    def evaluate_batch():
        if not batch_X:
            return
        X_mat = np.vstack(batch_X)
        ranker_scores = model.predict(X_mat)
        offset = 0
        for base_sc, labs, n_c in batch_meta:
            r_sc = ranker_scores[offset : offset + n_c]
            offset += n_c

            if np.sum(labs) == 0 or np.sum(labs) == len(labs):
                continue

            auc_b = compute_auc(labs, base_sc)
            auc_r = compute_auc(labs, r_sc)
            if auc_b is not None:
                base_auc_list.append(auc_b)
            if auc_r is not None:
                ranker_auc_list.append(auc_r)

            order_base = np.argsort(-base_sc)
            order_ranker = np.argsort(-r_sc)

            base_mrr_list.append(compute_mrr(labs, order_base))
            ranker_mrr_list.append(compute_mrr(labs, order_ranker))

            base_ndcg5_list.append(compute_ndcg_at_k(labs, k=5, ranked_indices=order_base))
            ranker_ndcg5_list.append(compute_ndcg_at_k(labs, k=5, ranked_indices=order_ranker))

            base_ndcg10_list.append(compute_ndcg_at_k(labs, k=10, ranked_indices=order_base))
            ranker_ndcg10_list.append(compute_ndcg_at_k(labs, k=10, ranked_indices=order_ranker))

        batch_X.clear()
        batch_meta.clear()

    start_eval = time.time()
    for i in tqdm(range(total_val), desc="Evaluating Re-Ranker on Validation Sample"):
        cands = candidates_list[i]
        clicks = set(clicked_list[i] if clicked_list[i] is not None else [])
        if not cands or not clicks:
            continue

        num_cands = len(cands)
        labels = np.array([1 if c in clicks else 0 for c in cands], dtype=np.int32)
        if np.sum(labels) == 0 or np.sum(labels) == num_cands:
            continue

        uid = user_ids[i]
        imp_time = timestamps[i]

        prof = user_profiles.get(uid)
        if prof is not None:
            n_clicks = prof.n_clicks
            recency_score = 1.0 if n_clicks > 0 else 0.0
            mean_dwell = prof.mean_dwell
            cat_affinity = prof.cat_affinity
            user_vec = prof.user_vec
            user_q_tokens = prof.q_tokens
        else:
            n_clicks = 0
            recency_score = 0.0
            mean_dwell = 0.0
            cat_affinity = {}
            user_vec = None
            user_q_tokens = []

        X_cand = np.zeros((num_cands, len(FEATURE_COLS)), dtype=np.float32)
        c_indices = [article_id_to_idx.get(cid, -1) for cid in cands]

        # 1. Semantic scoring via BLAS
        sem_scores = np.zeros(num_cands, dtype=np.float32)
        if user_vec is not None and embeddings_norm is not None:
            valid_mask = [0 <= idx < len(embeddings_norm) for idx in c_indices]
            if any(valid_mask):
                valid_idxs = [idx for idx, v in zip(c_indices, valid_mask) if v]
                cand_embs = embeddings_norm[valid_idxs]
                dots = np.dot(cand_embs, user_vec)
                dot_pos = 0
                for pos_idx, v in enumerate(valid_mask):
                    if v:
                        sem_scores[pos_idx] = dots[dot_pos]
                        dot_pos += 1

        # 2. Candidate-targeted BM25 scoring
        bm25_scores = np.zeros(num_cands, dtype=np.float32)
        if doc_token_scores is not None and user_q_tokens:
            valid_cand_mask = [0 <= idx < doc_token_scores.shape[0] for idx in c_indices]
            if any(valid_cand_mask):
                valid_c_idxs = [idx for idx, v in zip(c_indices, valid_cand_mask) if v]
                sub = doc_token_scores[valid_c_idxs, :]
                scored = np.asarray(sub[:, user_q_tokens].sum(axis=1)).ravel()
                scored_pos = 0
                for pos_idx, v in enumerate(valid_cand_mask):
                    if v:
                        bm25_scores[pos_idx] = scored[scored_pos]
                        scored_pos += 1

        first_stage_scores = np.maximum(bm25_scores, sem_scores)

        X_cand[:, 0] = n_clicks
        X_cand[:, 1] = float(recency_score)
        X_cand[:, 2] = float(mean_dwell)
        X_cand[:, 4] = 0.0
        X_cand[:, 6] = np.arange(num_cands, dtype=np.float32)
        X_cand[:, 7] = bm25_scores
        X_cand[:, 8] = sem_scores
        X_cand[:, 9] = first_stage_scores

        for pos_idx, cand_id in enumerate(cands):
            cat = art_cat_map.get(cand_id, "")
            pub = art_pub_map.get(cand_id)
            freshness = 0.0
            if pub is not None and isinstance(pub, (datetime, pd.Timestamp)):
                freshness = max(0.0, (imp_time - pub).total_seconds() / 3600.0)
            X_cand[pos_idx, 3] = float(freshness)
            X_cand[pos_idx, 5] = float(cat_affinity.get(cat, 0.0))

        batch_X.append(X_cand)
        batch_meta.append((first_stage_scores, labels, num_cands))

        if len(batch_meta) >= batch_predict_size:
            evaluate_batch()

    evaluate_batch()
    eval_elapsed = time.time() - start_eval

    n_eval = len(ranker_auc_list)
    results = {
        "dataset": "EB-NeRD Large (Validation Sample)",
        "evaluated_impressions": n_eval,
        "elapsed_seconds": round(eval_elapsed, 2),
        "throughput_impr_per_sec": round(n_eval / max(0.1, eval_elapsed), 1),
        "stage1_baseline": {
            "AUC": float(np.nanmean(base_auc_list)),
            "MRR": float(np.nanmean(base_mrr_list)),
            "nDCG@5": float(np.nanmean(base_ndcg5_list)),
            "nDCG@10": float(np.nanmean(base_ndcg10_list)),
        },
        "stage2_reranker": {
            "AUC": float(np.nanmean(ranker_auc_list)),
            "MRR": float(np.nanmean(ranker_mrr_list)),
            "nDCG@5": float(np.nanmean(ranker_ndcg5_list)),
            "nDCG@10": float(np.nanmean(ranker_ndcg10_list)),
        },
        "delta": {
            "AUC": float(np.nanmean(ranker_auc_list) - np.nanmean(base_auc_list)),
            "MRR": float(np.nanmean(ranker_mrr_list) - np.nanmean(base_mrr_list)),
            "nDCG@5": float(np.nanmean(ranker_ndcg5_list) - np.nanmean(base_ndcg5_list)),
            "nDCG@10": float(np.nanmean(ranker_ndcg10_list) - np.nanmean(base_ndcg10_list)),
        },
    }

    print("\n" + "=" * 65)
    print(f"  VALIDATION RESULTS SUMMARY ({n_eval:,} valid impressions)")
    print("=" * 65)
    print(f"{'Metric':<12} | {'Stage 1 (Baseline)':<18} | {'Stage 2 (LightGBM)':<18} | {'Delta':<10}")
    print("-" * 65)
    for m in ["AUC", "MRR", "nDCG@5", "nDCG@10"]:
        v1 = results["stage1_baseline"][m]
        v2 = results["stage2_reranker"][m]
        d = results["delta"][m]
        sign = "+" if d >= 0 else ""
        print(f"{m:<12} | {v1:<18.4f} | {v2:<18.4f} | {sign}{d:<.4f}")
    print("=" * 65)
    print(f"Evaluation completed in {eval_elapsed:.2f}s ({n_eval / max(0.1, eval_elapsed):.1f} impr/s)\n")

    out_metrics_path = Path("models/ebnerd_large_val_metrics_exact.json")
    with open(out_metrics_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved evaluation metrics to: {out_metrics_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate EB-NeRD Large Validation Set.")
    parser.add_argument("--model_path", default="models/ebnerd_lgbm_ranker.pkl")
    parser.add_argument("--max_impressions", type=int, default=50000, help="Max impressions to evaluate (-1 for all)")
    args = parser.parse_args()

    ds_dir = Path("data/processed/ebnerd_large")
    raw_dir = Path("data/raw/ebnerd_large")
    articles_path = ds_dir / "articles.parquet"
    val_impr_path = ds_dir / "impressions_val.parquet"
    if not val_impr_path.exists():
        val_impr_path = ds_dir / "impressions_validation.parquet"

    raw_val_hist = raw_dir / "validation" / "history.parquet"
    proc_val_hist = ds_dir / "history_val.parquet"

    emb_path = ds_dir / "article_embeddings.npy"
    ids_path = ds_dir / "article_ids.json"

    if not Path(args.model_path).exists():
        raise FileNotFoundError(f"Model not found at {args.model_path}. Train re-ranker first.")

    model = joblib.load(args.model_path)
    print(f"Loaded EB-NeRD LightGBM Re-Ranker model from {args.model_path}")

    art_cols = ["article_id", "title", "category", "published_time"]
    if "abstract" in pl.scan_parquet(articles_path).collect_schema().names():
        art_cols.append("abstract")
    elif "subtitle" in pl.scan_parquet(articles_path).collect_schema().names():
        art_cols.append("subtitle")
    articles_df = pl.read_parquet(articles_path, columns=art_cols)
    print(f"Loaded {len(articles_df):,} articles (metadata only).")

    embeddings = None
    article_id_to_idx = None
    if emb_path.exists():
        embeddings = np.load(emb_path)
        if ids_path.exists():
            with open(ids_path, "r", encoding="utf-8") as f:
                article_ids = json.load(f)
            article_id_to_idx = {aid: idx for idx, aid in enumerate(article_ids)}
        print(f"Loaded dense embeddings: shape={embeddings.shape}")

    max_imps = None if args.max_impressions < 0 else args.max_impressions
    evaluate_ebnerd_validation(
        model=model,
        val_impressions_path=val_impr_path,
        articles_df=articles_df,
        raw_val_hist_path=raw_val_hist,
        processed_val_hist_path=proc_val_hist,
        embeddings=embeddings,
        article_id_to_idx=article_id_to_idx,
        max_impressions=max_imps,
    )


if __name__ == "__main__":
    main()
