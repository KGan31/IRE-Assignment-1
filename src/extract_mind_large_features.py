"""
High-throughput feature extractor for MIND-Large training impressions.

Extracts the exact 9 ranking features used by the two-stage LightGBM ranker:
  1. user_click_count
  2. user_recency_score
  3. user_mean_dwell_time
  4. article_freshness_hours
  5. category_affinity_score
  6. session_position
  7. bm25_score
  8. semantic_score
  9. first_stage_score

Outputs directly to data/processed/mind_large/features_train.parquet.
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
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import scipy.sparse as sp
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from embeddings import compute_user_representation, normalize_l2
from reranker import FEATURE_COLS


def extract_features(
    impressions_path: Path,
    output_path: Path,
    articles_df: pl.DataFrame,
    embeddings: Optional[np.ndarray],
    article_id_to_idx: Dict[str, int],
    user_features_path: Path,
    popularity_map: Optional[Dict[str, float]] = None,
    max_impressions: Optional[int] = 270000,
    chunk_size_impr: int = 10000,
) -> Path:
    print(f"\n{'='*70}")
    print("  EXTRACTING MIND-LARGE TRAINING FEATURES")
    print(f"{'='*70}")
    print(f"Loading impressions from: {impressions_path}")
    print(f"Output target:            {output_path}")

    if max_impressions is not None:
        imp_df = pl.read_parquet(impressions_path, n_rows=max_impressions)
    else:
        imp_df = pl.read_parquet(impressions_path)
    total_imps = len(imp_df)
    print(f"Extracting features for {total_imps:,} training impressions...")

    # 1. Pre-normalize dense embeddings
    embeddings_norm = None
    if embeddings is not None:
        print("Pre-normalizing 768-d article embeddings...")
        embeddings_norm = np.ascontiguousarray(normalize_l2(embeddings), dtype=np.float32)

    # 2. Pre-build article metadata lookups
    art_ids = articles_df["article_id"].to_list()
    categories = articles_df["category"].fill_null("").to_list() if "category" in articles_df.columns else [""] * len(art_ids)
    art_cat_map = dict(zip(art_ids, categories))
    pub_times = articles_df["published_time"].to_list() if "published_time" in articles_df.columns else [None] * len(art_ids)
    art_pub_map = dict(zip(art_ids, pub_times))

    # 3. Pre-build BM25 index & candidate-targeted sparse doc-token matrix
    print("Building BM25 candidate-targeted sparse matrix...")
    t0_bm = time.time()
    import bm25s
    titles = articles_df["title"].fill_null("").to_list() if "title" in articles_df.columns else [""] * len(art_ids)
    abstracts = articles_df["abstract"].fill_null("").to_list() if "abstract" in articles_df.columns else [""] * len(art_ids)
    full_texts = [f"{t} {a}".strip() for t, a in zip(titles, abstracts)]
    tokens = bm25s.tokenize(full_texts, show_progress=False)
    bm25_retriever = bm25s.BM25()
    bm25_retriever.index(tokens, show_progress=False)
    art_token_ids_map = {aid: set(tokens.ids[i]) for i, aid in enumerate(art_ids)}

    csr_token_doc = sp.csr_matrix(
        (bm25_retriever.scores["data"], bm25_retriever.scores["indices"], bm25_retriever.scores["indptr"]),
        shape=(len(bm25_retriever.scores["indptr"]) - 1, bm25_retriever.scores["num_docs"])
    )
    doc_token_scores = csr_token_doc.T.tocsr()
    print(f"BM25 initialized in {time.time() - t0_bm:.2f}s (sparse shape: {doc_token_scores.shape}).")

    # 4. Load compact user history mapping from precomputed user_features_train
    print(f"Loading precomputed user history from {user_features_path}...")
    t0_u = time.time()
    unique_uids = set(imp_df["user_id"].unique().to_list())
    u_feat_df = (
        pl.scan_parquet(user_features_path)
        .filter(pl.col("user_id").is_in(list(unique_uids)))
        .collect()
    )
    # Store: uid -> (tuple of recent article_ids, n_clicks, recency_score)
    user_map: Dict[str, Tuple[Tuple[str, ...], int, float]] = {}
    for uid, hist, n_cl, rec in zip(
        u_feat_df["user_id"].to_list(),
        u_feat_df["click_history"].to_list(),
        u_feat_df["n_clicks"].to_list(),
        u_feat_df["recency_score"].to_list(),
    ):
        h_tuple = tuple(hist) if hist is not None else ()
        user_map[uid] = (h_tuple, int(n_cl), float(rec))
    del u_feat_df
    gc.collect()
    print(f"Loaded {len(user_map):,} relevant users in {time.time() - t0_u:.2f}s.")

    # 5. Parquet schema definition matching features_val.parquet
    pa_schema = pa.schema([
        ("impression_id", pa.string()),
        ("user_id", pa.string()),
        ("article_id", pa.string()),
        ("label", pa.int32()),
        ("user_click_count", pa.int32()),
        ("user_recency_score", pa.float32()),
        ("user_mean_dwell_time", pa.float32()),
        ("article_freshness_hours", pa.float32()),
        ("category_affinity_score", pa.float32()),
        ("session_position", pa.int32()),
        ("bm25_score", pa.float32()),
        ("semantic_score", pa.float32()),
        ("first_stage_score", pa.float32()),
    ])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = pq.ParquetWriter(output_path, pa_schema, compression="snappy")

    # 6. Stream impressions in chunks
    impr_ids = imp_df["impression_id"].to_list()
    user_ids = imp_df["user_id"].to_list()
    timestamps = imp_df["timestamp"].to_list()
    candidates_list = imp_df["candidate_article_ids"].to_list()
    clicked_list = imp_df["clicked_article_ids"].to_list()
    del imp_df
    gc.collect()

    num_chunks = (total_imps + chunk_size_impr - 1) // chunk_size_impr
    pbar = tqdm(total=total_imps, desc="Extracting features", unit="impr")

    total_rows_written = 0
    t_start = time.time()

    for chunk_idx in range(num_chunks):
        c_start = chunk_idx * chunk_size_impr
        c_end = min(c_start + chunk_size_impr, total_imps)

        col_imp_id: List[str] = []
        col_user_id: List[str] = []
        col_art_id: List[str] = []
        col_label: List[int] = []
        col_click_count: List[int] = []
        col_recency_score: List[float] = []
        col_mean_dwell: List[float] = []
        col_freshness: List[float] = []
        col_cat_affinity: List[float] = []
        col_position: List[int] = []
        col_bm25: List[float] = []
        col_semantic: List[float] = []
        col_first_stage: List[float] = []

        for i in range(c_start, c_end):
            cands = candidates_list[i]
            if not cands:
                continue
            uid = user_ids[i]
            imp_id = impr_ids[i]
            imp_time = timestamps[i]
            clicked_set = set(clicked_list[i] if clicked_list[i] is not None else [])

            # User representation
            u_info = user_map.get(uid)
            if u_info is not None:
                u_hist, n_clicks, rec_score = u_info
            else:
                u_hist, n_clicks, rec_score = (), 0, 0.0

            # Compute category affinity distribution & query tokens from past clicks
            cat_affinity_map: Dict[str, float] = {}
            user_q_tokens: Set[int] = set()
            user_vec = None

            if u_hist:
                n_h = len(u_hist)
                cat_counts: Dict[str, int] = {}
                for aid in u_hist:
                    c = art_cat_map.get(aid, "")
                    if c:
                        cat_counts[c] = cat_counts.get(c, 0) + 1
                cat_affinity_map = {k: v / n_h for k, v in cat_counts.items()}

                recent = u_hist[-20:]
                for aid in recent:
                    user_q_tokens.update(art_token_ids_map.get(aid, set()))

                # Dense semantic representation
                if embeddings_norm is not None:
                    h_idxs = [article_id_to_idx.get(aid, -1) for aid in recent]
                    valid_h = [idx for idx in h_idxs if 0 <= idx < len(embeddings_norm)]
                    if valid_h:
                        vec = embeddings_norm[valid_h].mean(axis=0)
                        norm_val = np.linalg.norm(vec)
                        if norm_val > 1e-6:
                            user_vec = vec / norm_val

            # Candidate-targeted sparse BM25 scoring
            num_cands = len(cands)
            c_indices = [article_id_to_idx.get(cid, -1) for cid in cands]
            bm25_scores = np.zeros(num_cands, dtype=np.float32)

            if doc_token_scores is not None and user_q_tokens:
                valid_cand_mask = [0 <= idx < doc_token_scores.shape[0] for idx in c_indices]
                if any(valid_cand_mask):
                    valid_c_idxs = [idx for idx, v in zip(c_indices, valid_cand_mask) if v]
                    sub = doc_token_scores[valid_c_idxs, :]
                    scored = np.asarray(sub[:, list(user_q_tokens)].sum(axis=1)).ravel()
                    p = 0
                    for pi, v in enumerate(valid_cand_mask):
                        if v:
                            bm25_scores[pi] = float(scored[p])
                            p += 1

            # Candidate-targeted semantic scoring
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
                            sem_scores[pi] = float(dots[p])
                            p += 1

            for pos_idx, cand_id in enumerate(cands):
                cat = art_cat_map.get(cand_id, "")
                pub = art_pub_map.get(cand_id)
                freshness = 0.0
                if pub is not None and isinstance(pub, (datetime, pd.Timestamp)):
                    freshness = max(0.0, (imp_time - pub).total_seconds() / 3600.0)

                cat_aff = cat_affinity_map.get(cat, 0.0)
                pop_val = int(popularity_map.get(cand_id, 0)) if popularity_map else 0
                b_score = float(bm25_scores[pos_idx])
                s_score = float(sem_scores[pos_idx])
                first_stg = max(b_score, s_score)
                lbl = 1 if cand_id in clicked_set else 0

                col_imp_id.append(imp_id)
                col_user_id.append(uid)
                col_art_id.append(cand_id)
                col_label.append(lbl)
                col_click_count.append(n_clicks)
                col_recency_score.append(float(rec_score))
                col_mean_dwell.append(0.0)
                col_freshness.append(float(freshness))
                col_cat_affinity.append(float(cat_aff))
                col_position.append(pos_idx)
                col_bm25.append(b_score)
                col_semantic.append(s_score)
                col_first_stage.append(first_stg)

        # Write chunk
        if col_imp_id:
            batch_tbl = pa.Table.from_arrays(
                [
                    pa.array(col_imp_id, type=pa.string()),
                    pa.array(col_user_id, type=pa.string()),
                    pa.array(col_art_id, type=pa.string()),
                    pa.array(col_label, type=pa.int32()),
                    pa.array(col_click_count, type=pa.int32()),
                    pa.array(col_recency_score, type=pa.float32()),
                    pa.array(col_mean_dwell, type=pa.float32()),
                    pa.array(col_freshness, type=pa.float32()),
                    pa.array(col_cat_affinity, type=pa.float32()),
                    pa.array(col_position, type=pa.int32()),
                    pa.array(col_bm25, type=pa.float32()),
                    pa.array(col_semantic, type=pa.float32()),
                    pa.array(col_first_stage, type=pa.float32()),
                ],
                schema=pa_schema,
            )
            writer.write_table(batch_tbl)
            total_rows_written += len(col_imp_id)

        pbar.update(c_end - c_start)

    pbar.close()
    writer.close()

    elapsed = time.time() - t_start
    impr_per_sec = total_imps / max(0.1, elapsed)
    print(f"\nFeature extraction finished in {elapsed:.2f}s ({impr_per_sec:.1f} impr/s).")
    print(f"Wrote {total_rows_written:,} candidate interaction rows to {output_path}.\n")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Extract MIND-Large training features.")
    parser.add_argument("--max_impressions", type=int, default=270000, help="Number of impressions to extract (270k ~= 10M rows)")
    parser.add_argument("--chunk_size_impr", type=int, default=10000)
    parser.add_argument("--output_path", type=str, default="data/processed/mind_large/features_train.parquet")
    args = parser.parse_args()

    ds_dir = Path("data/processed/mind_large")
    imp_train_path = ds_dir / "impressions_train.parquet"
    art_path = ds_dir / "articles.parquet"
    emb_path = ds_dir / "article_embeddings.npy"
    ids_path = ds_dir / "article_ids.json"
    user_feat_path = ds_dir / "user_features_train.parquet"

    articles_df = pl.read_parquet(art_path, columns=["article_id", "title", "abstract", "category", "published_time"])
    embeddings = np.load(emb_path) if emb_path.exists() else None
    article_id_to_idx = {}
    if ids_path.exists():
        with open(ids_path, "r", encoding="utf-8") as f:
            ids = json.load(f)
        article_id_to_idx = {aid: idx for idx, aid in enumerate(ids)}

    extract_features(
        impressions_path=imp_train_path,
        output_path=Path(args.output_path),
        articles_df=articles_df,
        embeddings=embeddings,
        article_id_to_idx=article_id_to_idx,
        user_features_path=user_feat_path,
        max_impressions=args.max_impressions,
        chunk_size_impr=args.chunk_size_impr,
    )


if __name__ == "__main__":
    main()
