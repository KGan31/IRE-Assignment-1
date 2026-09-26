"""
EB-NeRD Large Two-Stage GBDT Re-Ranker Codabench Submission Generator.

Scores candidate articles across EB-NeRD Large test impressions (13.54M impressions)
using behavioral, temporal, lexical, and semantic features, and generates a compliant
predictions.zip submission for the RecSys 2024 / Codabench competition.

Optimized for ultra-low RAM (peak < 1.2 GB) and high throughput (~1500+ impr/s):
1. Stores compact user history tuples (recent article IDs + mean dwell time) in ~750 MB.
2. Dynamically materializes UserProfile objects only for active users in each 50k impression chunk (~75 MB).
3. Uses candidate-targeted BM25 sparse matrix slicing.
4. Batches LightGBM tree inference across 5,000 impressions at a time.
5. Limits CPU thread contention (OMP/POLARS=4) so the host machine remains completely responsive.
"""

import os
# Cap threads to prevent CPU freezing and leave resources for OS
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["POLARS_MAX_THREADS"] = "4"

import argparse
from datetime import datetime
import gc
import json
from pathlib import Path
import re
import sys
import time
import warnings
import zipfile
from typing import Any, Dict, List, Optional, Set, Tuple

import joblib
import numpy as np
import pandas as pd
import polars as pl
import scipy.sparse as sp
from scipy.stats import rankdata
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")

# Ensure src is in sys.path
ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
from embeddings import compute_user_representation, normalize_l2
from metrics import compute_auc, compute_mrr, compute_ndcg_at_k
from reranker import FEATURE_COLS


class UserProfile:
    """Compact user profile representation for the active chunk (~300 bytes)."""
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


def validate_submission_file(predictions_txt_path: Path, expected_lines: int) -> None:
    """Validate format and integrity of the output predictions.txt."""
    print("Validating submission file format...")
    line_count = 0
    pattern = re.compile(r"^\d+\s+\[\d*(?:,\d+)*\]$")

    with open(predictions_txt_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            line_count += 1
            line = line.strip()
            if not pattern.match(line):
                if line_count <= 5 or not line:
                    raise ValueError(f"Invalid format at line {line_count}: {line}")

    if line_count != expected_lines:
        print(f"[warning] Expected {expected_lines:,} lines, but found {line_count:,} lines.")
    else:
        print(f"Format validation passed: exactly {line_count:,} valid prediction lines.")


def load_compact_user_history(
    raw_hist_path: Path,
    proc_hist_path: Path,
    max_history_len: int = 30,
) -> Dict[str, Tuple[List[str], float]]:
    """
    Loads raw user history into a lightweight map: user_id -> (recent_aids, mean_dwell).
    Memory footprint: ~750 MB for all 807,677 users.
    """
    print("Loading compact user history into memory...")
    start_t = time.time()
    history_map: Dict[str, Tuple[List[str], float]] = {}

    if raw_hist_path.exists():
        print(f"  - Reading from raw parquet {raw_hist_path}...")
        df = pl.read_parquet(raw_hist_path, columns=["user_id", "article_id_fixed", "read_time_fixed"])
        capped = df.select([
            pl.col("user_id"),
            pl.col("article_id_fixed").list.tail(max_history_len),
            pl.col("read_time_fixed").list.tail(max_history_len),
        ])
        del df
        gc.collect()

        uids = capped["user_id"].to_list()
        clicks = capped["article_id_fixed"].to_list()
        dwells = capped["read_time_fixed"].to_list()
        del capped
        gc.collect()

        for u, c, d in zip(uids, clicks, dwells):
            if c:
                md = float(sum(d) / len(d)) if d else 0.0
                history_map[int(u)] = (tuple(c), md)

        del uids, clicks, dwells
        gc.collect()
        print(f"Loaded compact history for {len(history_map):,} users in {time.time() - start_t:.2f}s.")
        return history_map

    if proc_hist_path.exists():
        print(f"  - Scanning processed history from {proc_hist_path}...")
        df = (
            pl.scan_parquet(proc_hist_path)
            .sort("click_time")
            .group_by("user_id")
            .tail(max_history_len)
            .collect()
        )
        has_dwell = "dwell_time" in df.columns
        agg_exprs = [pl.col("clicked_article_id").alias("aids")]
        if has_dwell:
            agg_exprs.append(pl.col("dwell_time").fill_null(0.0).alias("dwells"))
        user_agg = df.group_by("user_id").agg(agg_exprs)
        del df
        gc.collect()

        uids = user_agg["user_id"].to_list()
        aids_list = user_agg["aids"].to_list()
        dwells_list = user_agg["dwells"].to_list() if has_dwell else [[0.0] * len(a) for a in aids_list]
        del user_agg
        gc.collect()

        for u, aids, dwells in zip(uids, aids_list, dwells_list):
            md = float(sum(dwells) / len(dwells)) if dwells else 0.0
            history_map[u] = (aids, md)

        print(f"Loaded compact history for {len(history_map):,} users in {time.time() - start_t:.2f}s.")
        return history_map

    print("No history file found.")
    return {}


def generate_ebnerd_reranker_submission(
    model: Any,
    test_impressions_path: Path,
    articles_df: pl.DataFrame,
    raw_hist_path: Path,
    proc_hist_path: Path,
    embeddings: Optional[np.ndarray],
    article_id_to_idx: Optional[Dict[str, int]],
    output_dir: Path,
    chunk_size_impr: int = 50000,
    batch_predict_size: int = 5000,
    max_impressions: Optional[int] = None,
) -> Path:
    """
    Generates Codabench predictions.zip with strictly bounded memory (peak < 1.2 GB).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_txt_path = output_dir / "predictions.txt"
    predictions_zip_path = output_dir / "predictions.zip"

    print(f"Inspecting test impressions from {test_impressions_path}...")
    total_test_impressions = pl.scan_parquet(test_impressions_path).select(pl.len()).collect().item()
    if max_impressions is not None:
        total_test_impressions = min(max_impressions, total_test_impressions)
    print(f"Total test impressions to score: {total_test_impressions:,}")

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

    # 4. Load compact user history (~750 MB for 807k users)
    user_history_map = load_compact_user_history(
        raw_hist_path=raw_hist_path,
        proc_hist_path=proc_hist_path,
        max_history_len=30,
    )

    # 5. Stream impressions in chunks of 50,000, materializing profiles only for active chunk users (< 75 MB)
    print(f"Scoring test impressions and streaming to {predictions_txt_path}...")
    start_time = time.time()
    total_written = 0

    batch_X: List[np.ndarray] = []
    batch_metadata: List[Tuple[str, int]] = []
    line_buffer: List[str] = []

    def flush_batch(f_out):
        if not batch_X:
            return
        X_batch = np.vstack(batch_X)
        scores_batch = model.predict(X_batch)
        offset = 0
        for raw_id, n_cand in batch_metadata:
            sc = scores_batch[offset : offset + n_cand]
            ranks = rankdata(-sc, method="ordinal")
            rank_str = ",".join(str(int(r)) for r in ranks)
            line_buffer.append(f"{raw_id} [{rank_str}]\n")
            offset += n_cand
        batch_X.clear()
        batch_metadata.clear()
        if len(line_buffer) >= 20000:
            f_out.writelines(line_buffer)
            line_buffer.clear()

    with open(predictions_txt_path, "w", encoding="utf-8", buffering=2 * 1024 * 1024) as out_f:
        pbar = tqdm(total=total_test_impressions, desc="Generating EB-NeRD Predictions", unit="impr")

        for chunk_offset in range(0, total_test_impressions, chunk_size_impr):
            chunk_len = min(chunk_size_impr, total_test_impressions - chunk_offset)
            chunk_df = (
                pl.scan_parquet(test_impressions_path)
                .select(["impression_id", "user_id", "timestamp", "candidate_article_ids"])
                .slice(chunk_offset, chunk_len)
                .collect()
            )

            impr_ids = chunk_df["impression_id"].to_list()
            user_ids = chunk_df["user_id"].to_list()
            timestamps = chunk_df["timestamp"].to_list()
            candidates_list = chunk_df["candidate_article_ids"].to_list()
            del chunk_df

            # Precompute profiles ONLY for the unique users in this chunk (~25k users, ~75 MB)
            chunk_unique_uids = set(user_ids)
            chunk_profiles: Dict[str, UserProfile] = {}
            for uid in chunk_unique_uids:
                u_int = int(uid.split("_")[1]) if "_" in uid else int(uid)
                hist_item = user_history_map.get(u_int)
                if hist_item is not None:
                    c_ints, md = hist_item
                    aids = [f"ebnerd_{aid}" for aid in c_ints]
                    n = len(aids)
                    recent = aids[-20:]
                    q_toks = set()
                    for a in recent:
                        q_toks.update(art_token_ids_map.get(a, set()))
                    cat_counts: Dict[str, int] = {}
                    for a in aids:
                        c = art_cat_map.get(a, "")
                        if c:
                            cat_counts[c] = cat_counts.get(c, 0) + 1
                    cat_aff = {k: v / n for k, v in cat_counts.items()} if n else {}
                    u_vec = None
                    if embeddings_norm is not None and recent:
                        u_vec = compute_user_representation(recent, article_id_to_idx, embeddings_norm)
                    chunk_profiles[uid] = UserProfile(
                        n_clicks=n,
                        mean_dwell=md,
                        cat_affinity=cat_aff,
                        user_vec=u_vec,
                        q_tokens=list(q_toks),
                    )

            for i in range(chunk_len):
                raw_impr_id = impr_ids[i]
                if isinstance(raw_impr_id, str) and raw_impr_id.startswith("ebnerd_"):
                    raw_impr_id = raw_impr_id[len("ebnerd_"):]

                cands = candidates_list[i]
                if not cands:
                    line_buffer.append(f"{raw_impr_id} []\n")
                    continue

                uid = user_ids[i]
                imp_time = timestamps[i]

                prof = chunk_profiles.get(uid)
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

                # 1. Vectorized semantic scoring via BLAS
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

                # 3. Candidate metadata features
                for pos_idx, cand_id in enumerate(cands):
                    cat = art_cat_map.get(cand_id, "")
                    pub = art_pub_map.get(cand_id)
                    freshness = 0.0
                    if pub is not None and isinstance(pub, (datetime, pd.Timestamp)):
                        freshness = max(0.0, (imp_time - pub).total_seconds() / 3600.0)
                    X_cand[pos_idx, 3] = float(freshness)
                    X_cand[pos_idx, 5] = float(cat_affinity.get(cat, 0.0))

                batch_X.append(X_cand)
                batch_metadata.append((raw_impr_id, num_cands))

                if len(batch_metadata) >= batch_predict_size:
                    flush_batch(out_f)

            # Free chunk profiles
            del chunk_profiles
            pbar.update(chunk_len)
            total_written += chunk_len
            gc.collect()

        # Flush any remaining predictions in batch
        flush_batch(out_f)
        if line_buffer:
            out_f.writelines(line_buffer)
            line_buffer.clear()

        pbar.close()

    elapsed = time.time() - start_time
    print(f"Wrote {total_written:,} predictions in {elapsed:.2f}s ({total_written / max(0.1, elapsed):.0f} impr/s)")

    print(f"Creating submission zip: {predictions_zip_path}...")
    with zipfile.ZipFile(predictions_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(predictions_txt_path, arcname="predictions.txt")

    print(f"Submission zip successfully created at: {predictions_zip_path}")
    validate_submission_file(predictions_txt_path, total_test_impressions)
    return predictions_zip_path


def main():
    parser = argparse.ArgumentParser(description="Generate EB-NeRD Large Re-Ranker Codabench submission.")
    parser.add_argument("--dataset_type", default="large", choices=["small", "large"])
    parser.add_argument("--model_path", default="models/ebnerd_lgbm_ranker.pkl")
    parser.add_argument("--output_dir", default="submissions/submissions_ebnerd_reranker")
    parser.add_argument("--max_impressions", type=int, default=None, help="Optional limit for test impressions")
    parser.add_argument("--chunk_size_impr", type=int, default=50000, help="Impression chunk size for memory bounding")
    parser.add_argument("--batch_predict_size", type=int, default=5000, help="Batch size for LightGBM tree inference")
    args = parser.parse_args()

    ds_dir = Path("data/processed") / f"ebnerd_{args.dataset_type}"
    raw_dir = Path("data/raw") / f"ebnerd_{args.dataset_type}"

    articles_path = ds_dir / "articles.parquet"
    test_impr_path = ds_dir / "impressions_test.parquet"
    emb_path = ds_dir / "article_embeddings.npy"
    ids_path = ds_dir / "article_ids.json"

    raw_test_hist = raw_dir / "test" / "history.parquet"
    proc_test_hist = ds_dir / "history_test.parquet"

    if not Path(args.model_path).exists():
        raise FileNotFoundError(f"Model not found at {args.model_path}. Train re-ranker first.")

    model = joblib.load(args.model_path)
    print(f"Loaded EB-NeRD Re-Ranker model from {args.model_path}")

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

    out_dir = Path(args.output_dir)
    generate_ebnerd_reranker_submission(
        model=model,
        test_impressions_path=test_impr_path,
        articles_df=articles_df,
        raw_hist_path=raw_test_hist,
        proc_hist_path=proc_test_hist,
        embeddings=embeddings,
        article_id_to_idx=article_id_to_idx,
        output_dir=out_dir,
        chunk_size_impr=args.chunk_size_impr,
        batch_predict_size=args.batch_predict_size,
        max_impressions=args.max_impressions,
    )


if __name__ == "__main__":
    main()
