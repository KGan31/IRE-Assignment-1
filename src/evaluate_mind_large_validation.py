"""
Evaluate MIND Large Validation Set using the exact same updated logical function
and feature extraction pipeline as the test re-ranker submission generator.

Includes:
1. Full history representation from prior train + dev splits.
2. Global article popularity fallback for cold-start users.
3. Position bias neutralization (session_position = 0.0).
4. Fine-grained dense semantic cosine blending (+ 0.3 * semantic_score).

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
from generate_mind_reranker_submission import (
    UserProfile,
    build_user_profiles,
    compute_article_popularity,
)


def evaluate_validation(
    model: Any,
    val_impressions_path: Path,
    articles_df: pl.DataFrame,
    hist_train_path: Path,
    hist_dev_path: Path,
    embeddings: Optional[np.ndarray],
    article_id_to_idx: Optional[Dict[str, int]],
    popularity_map: Optional[Dict[str, float]] = None,
    max_impressions: Optional[int] = 50000,
    batch_predict_size: int = 5000,
) -> Dict[str, Any]:
    """
    Evaluates validation impressions using the exact test ranking logic and scores against ground truth.
    """
    print(f"\n{'='*65}")
    print("  MIND-LARGE VALIDATION EVALUATION (EXACT UPDATED LOGIC)")
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
        abstracts = articles_df["abstract"].fill_null("").to_list() if "abstract" in articles_df.columns else [""] * len(art_ids)
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

    # 4. Find all unique validation users in sample and build compact UserProfiles
    val_uids = val_df["user_id"].unique()
    user_profiles = build_user_profiles(
        hist_train_path=hist_train_path,
        hist_dev_path=hist_dev_path,
        test_uids=val_uids,
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

    print("Scoring candidate slates and computing ranking metrics...")
    start_time = time.time()

    auc_base_list, auc_rerank_list = [], []
    mrr_base_list, mrr_rerank_list = [], []
    ndcg5_base_list, ndcg5_rerank_list = [], []
    ndcg10_base_list, ndcg10_rerank_list = [], []

    batch_X: List[np.ndarray] = []
    # (y_true, base_scores, n_cand, is_cold, cand_pop_scores, cand_sem_scores)
    batch_metadata: List[Tuple[np.ndarray, np.ndarray, int, bool, np.ndarray, np.ndarray]] = []

    def evaluate_batch():
        if not batch_X:
            return
        X_batch = np.vstack(batch_X)
        scores_batch = model.predict(X_batch)
        offset = 0
        for y_true, base_scores, n_cand, is_cold, cand_pop, cand_sem in batch_metadata:
            if is_cold:
                rerank_scores = cand_pop
            else:
                r_sc = scores_batch[offset : offset + n_cand]
                # Updated logic: blend tree ranker with dense semantic cosine similarity
                rerank_scores = r_sc + 0.3 * cand_sem + 1e-4 * cand_pop
            offset += n_cand

            # Skip impressions with all positive or all negative labels for AUC
            n_pos = int(np.sum(y_true))
            if 0 < n_pos < n_cand:
                auc_b = compute_auc(y_true, base_scores)
                auc_r = compute_auc(y_true, rerank_scores)
                if auc_b is not None:
                    auc_base_list.append(auc_b)
                if auc_r is not None:
                    auc_rerank_list.append(auc_r)

            # MRR & nDCG (ranked indices by descending score)
            order_base = np.argsort(-base_scores)
            order_rerank = np.argsort(-rerank_scores)

            mrr_base_list.append(compute_mrr(y_true, order_base))
            mrr_rerank_list.append(compute_mrr(y_true, order_rerank))

            ndcg5_base_list.append(compute_ndcg_at_k(y_true, k=5, ranked_indices=order_base))
            ndcg5_rerank_list.append(compute_ndcg_at_k(y_true, k=5, ranked_indices=order_rerank))

            ndcg10_base_list.append(compute_ndcg_at_k(y_true, k=10, ranked_indices=order_base))
            ndcg10_rerank_list.append(compute_ndcg_at_k(y_true, k=10, ranked_indices=order_rerank))

        batch_X.clear()
        batch_metadata.clear()

    for i in tqdm(range(total_val), desc="Evaluating Validation Impressions", unit="impr"):
        cands = candidates_list[i]
        if not cands:
            continue
        cands = list(cands)
        num_cands = len(cands)
        clicked_set = set(clicked_list[i] if clicked_list[i] is not None else [])
        y_true = np.array([1 if c in clicked_set else 0 for c in cands], dtype=np.int32)

        uid = user_ids[i]
        imp_time = timestamps[i]

        prof = user_profiles.get(uid)
        is_cold = (prof is None or prof.n_clicks == 0 or prof.user_vec is None)

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

        num_cands = len(cands)
        X_cand = np.zeros((num_cands, len(FEATURE_COLS)), dtype=np.float32)
        c_indices = [article_id_to_idx.get(cid, -1) for cid in cands]

        cand_pop_scores = np.array(
            [popularity_map.get(cid, 0.0) if popularity_map else 0.0 for cid in cands],
            dtype=np.float32,
        )

        # 1. Semantic score
        sem_scores = np.zeros(num_cands, dtype=np.float32)
        if user_vec is not None and embeddings_norm is not None:
            valid_mask = [0 <= idx < len(embeddings_norm) for idx in c_indices]
            if any(valid_mask):
                valid_idxs = [idx for idx, v in zip(c_indices, valid_mask) if v]
                cand_embs = embeddings_norm[valid_idxs]
                dots = np.dot(cand_embs, user_vec)
                p = 0
                for pi, v in enumerate(valid_mask):
                    if v:
                        sem_scores[pi] = dots[p]
                        p += 1

        # 2. BM25 score
        bm25_scores = np.zeros(num_cands, dtype=np.float32)
        if doc_token_scores is not None and user_q_tokens:
            valid_cand_mask = [0 <= idx < doc_token_scores.shape[0] for idx in c_indices]
            if any(valid_cand_mask):
                valid_c_idxs = [idx for idx, v in zip(c_indices, valid_cand_mask) if v]
                sub = doc_token_scores[valid_c_idxs, :]
                scored = np.asarray(sub[:, user_q_tokens].sum(axis=1)).ravel()
                p = 0
                for pi, v in enumerate(valid_cand_mask):
                    if v:
                        bm25_scores[pi] = scored[p]
                        p += 1

        first_stage_scores = np.maximum(bm25_scores, sem_scores)

        X_cand[:, 0] = n_clicks
        X_cand[:, 1] = float(recency_score)
        X_cand[:, 2] = float(mean_dwell)
        X_cand[:, 4] = cand_pop_scores  # populate popularity feature
        X_cand[:, 6] = 0.0              # neutralize position bias
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
        batch_metadata.append((y_true, first_stage_scores, num_cands, is_cold, cand_pop_scores, sem_scores))

        if len(batch_metadata) >= batch_predict_size:
            evaluate_batch()

    evaluate_batch()
    elapsed = time.time() - start_time

    n_eval = len(auc_rerank_list)
    results = {
        "dataset": "MIND-Large (Validation Exact Updated Logic)",
        "evaluated_impressions": n_eval,
        "elapsed_seconds": round(elapsed, 2),
        "throughput_impr_per_sec": round(total_val / max(0.1, elapsed), 1),
        "stage1_baseline": {
            "AUC": float(np.mean(auc_base_list)),
            "MRR": float(np.mean(mrr_base_list)),
            "nDCG@5": float(np.mean(ndcg5_base_list)),
            "nDCG@10": float(np.mean(ndcg10_base_list)),
        },
        "stage2_reranker": {
            "AUC": float(np.mean(auc_rerank_list)),
            "MRR": float(np.mean(mrr_rerank_list)),
            "nDCG@5": float(np.mean(ndcg5_rerank_list)),
            "nDCG@10": float(np.mean(ndcg10_rerank_list)),
        },
        "delta": {
            "AUC": float(np.mean(auc_rerank_list) - np.mean(auc_base_list)),
            "MRR": float(np.mean(mrr_rerank_list) - np.mean(mrr_base_list)),
            "nDCG@5": float(np.mean(ndcg5_rerank_list) - np.mean(ndcg5_base_list)),
            "nDCG@10": float(np.mean(ndcg10_rerank_list) - np.mean(ndcg10_base_list)),
        },
    }

    print("\n" + "=" * 65)
    print(f"  VALIDATION RESULTS SUMMARY ({n_eval:,} evaluated impressions)")
    print("=" * 65)
    print(f"{'Metric':<12} | {'Stage 1 (Baseline)':<18} | {'Stage 2 (Updated Ranker)':<24} | {'Delta':<10}")
    print("-" * 65)
    for m in ["AUC", "MRR", "nDCG@5", "nDCG@10"]:
        v1 = results["stage1_baseline"][m]
        v2 = results["stage2_reranker"][m]
        d = results["delta"][m]
        sign = "+" if d >= 0 else ""
        print(f"{m:<12} | {v1:<18.4f} | {v2:<24.4f} | {sign}{d:<.4f}")
    print("=" * 65)
    print(f"Evaluation completed in {elapsed:.2f}s ({total_val / max(0.1, elapsed):.1f} impr/s)\n")

    out_metrics_path = Path("models/mind_large_val_metrics_exact.json")
    with open(out_metrics_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved exact evaluation metrics to: {out_metrics_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate MIND Large Validation Set with updated logic.")
    parser.add_argument("--model_path", default="models/mind_lgbm_ranker.pkl")
    parser.add_argument("--max_impressions", type=int, default=50000)
    args = parser.parse_args()

    ds_dir = Path("data/processed/mind_large")
    articles_path = ds_dir / "articles.parquet"
    val_impr_path = ds_dir / "impressions_val.parquet"
    emb_path = ds_dir / "article_embeddings.npy"
    ids_path = ds_dir / "article_ids.json"

    hist_train_path = ds_dir / "history_train.parquet"
    hist_dev_path = ds_dir / "history_dev.parquet"
    if not hist_dev_path.exists():
        hist_dev_path = ds_dir / "history_val.parquet"

    if not Path(args.model_path).exists():
        raise FileNotFoundError(f"Model not found at {args.model_path}. Train re-ranker first.")

    model = joblib.load(args.model_path)
    print(f"Loaded MIND Re-Ranker model from {args.model_path}")

    art_cols = ["article_id", "title", "abstract", "category", "published_time"]
    articles_df = pl.read_parquet(articles_path, columns=art_cols)
    print(f"Loaded {len(articles_df):,} articles.")

    embeddings = np.load(emb_path) if emb_path.exists() else None
    article_id_to_idx = None
    if ids_path.exists():
        with open(ids_path, "r", encoding="utf-8") as f:
            ids = json.load(f)
        article_id_to_idx = {aid: idx for idx, aid in enumerate(ids)}
    print(f"Loaded dense embeddings: shape={embeddings.shape}")

    # Compute popularity from history files
    print("Computing article popularity from history files...")
    pop_hist_paths = [hist_train_path, hist_dev_path]
    popularity_map = compute_article_popularity(pop_hist_paths)

    evaluate_validation(
        model=model,
        val_impressions_path=val_impr_path,
        articles_df=articles_df,
        hist_train_path=hist_train_path,
        hist_dev_path=hist_dev_path,
        embeddings=embeddings,
        article_id_to_idx=article_id_to_idx,
        popularity_map=popularity_map,
        max_impressions=args.max_impressions,
    )


if __name__ == "__main__":
    main()
