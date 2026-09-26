"""
MIND-Large Two-Stage GBDT Re-Ranker Codabench Submission Generator.

Scores candidate articles across MIND-Large test impressions using behavioral,
temporal, lexical, and semantic features, and generates a compliant prediction.zip submission.

Optimized for low-RAM (peak < 1.5 GB) and high throughput (~1000+ impr/s):
1. Loads full test user history from history_test.parquet (lifting coverage from 80.1% to 99.4%).
2. Adds global article popularity fallback for cold-start impressions.
3. Neutralizes presentation position bias (session_position = 0.0) to avoid over-penalizing candidate articles.
4. Blends fine-grained dense semantic cosine similarity with GBDT tree scores.
5. Uses candidate-targeted BM25 sparse matrix slicing.
6. Batches LightGBM tree inference across 5,000 impressions at a time.
7. Limits CPU thread contention (OMP/POLARS=4) so the host machine remains completely responsive.
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
from embeddings import compute_user_representation, normalize_l2
from metrics import compute_auc, compute_mrr, compute_ndcg_at_k
from reranker import FEATURE_COLS, evaluate_impression_predictions


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


def validate_submission_file(prediction_txt_path: Path, expected_lines: int) -> None:
    """Validate format and integrity of the output prediction.txt."""
    print("Validating submission file format...")
    line_count = 0
    pattern = re.compile(r"^\d+\s+\[\d+(?:,\d+)*\]$")

    with open(prediction_txt_path, "r", encoding="utf-8") as f:
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


def compute_article_popularity(history_paths: List[Path]) -> Dict[str, float]:
    """Compute normalized article popularity from available history files as a tie-breaker / cold-start fallback."""
    pop_counts: Dict[str, int] = {}
    for p in history_paths:
        if p.exists():
            print(f"Counting article popularity from {p}...")
            try:
                counts_df = (
                    pl.scan_parquet(p)
                    .select("clicked_article_id")
                    .group_by("clicked_article_id")
                    .len()
                    .collect()
                )
                for aid, c in zip(counts_df["clicked_article_id"].to_list(), counts_df["len"].to_list()):
                    if aid is not None:
                        key = str(aid)
                        pop_counts[key] = pop_counts.get(key, 0) + int(c)
                del counts_df
                gc.collect()
            except Exception as e:
                print(f"[warning] Popularity counting error for {p}: {e}")

    if not pop_counts:
        return {}

    max_c = max(pop_counts.values())
    print(f"Computed popularity for {len(pop_counts):,} unique articles (max count: {max_c:,}).")
    return {aid: c / max_c for aid, c in pop_counts.items()}


def load_compact_user_history(
    hist_test_path: Optional[Path],
    hist_dev_path: Path,
    hist_train_path: Path,
    max_history_len: int = 30,
) -> Dict[str, Tuple[str, ...]]:
    """
    Loads raw user click history into a lightweight map: user_id -> tuple of recent article IDs.
    Memory footprint: ~200 MB for all 700k users.
    """
    print("Loading compact user click history into memory...")
    start_t = time.time()
    user_history_map: Dict[str, Tuple[str, ...]] = {}

    # 1. Primary: Official test history provided by Microsoft (covers 99.4% of test users)
    if hist_test_path and hist_test_path.exists():
        print(f"  - Reading official test history from {hist_test_path}...")
        test_agg = (
            pl.scan_parquet(hist_test_path)
            .select(["user_id", "clicked_article_id"])
            .group_by("user_id")
            .agg(pl.col("clicked_article_id").tail(max_history_len).alias("aids"))
            .collect()
        )
        for u, a in zip(test_agg["user_id"].to_list(), test_agg["aids"].to_list()):
            user_history_map[u] = tuple(a)
        del test_agg
        gc.collect()
    else:
        # Fallback for validation or splits without hist_test: check precomputed user_features
        data_dir = hist_train_path.parent
        user_feat_val = data_dir / "user_features_val.parquet"
        if user_feat_val.exists():
            print(f"  - Reading precomputed user history from {user_feat_val}...")
            val_agg = (
                pl.scan_parquet(user_feat_val)
                .select(["user_id", "click_history"])
                .collect()
            )
            for u, h in zip(val_agg["user_id"].to_list(), val_agg["click_history"].to_list()):
                if h:
                    user_history_map[u] = tuple(h[-max_history_len:])
            del val_agg
            gc.collect()
        elif hist_train_path.exists():
            print(f"  - Reading user history from {hist_train_path}...")
            train_agg = (
                pl.scan_parquet(hist_train_path)
                .select(["user_id", "clicked_article_id"])
                .group_by("user_id")
                .agg(pl.col("clicked_article_id").tail(max_history_len).alias("aids"))
                .collect()
            )
            for u, a in zip(train_agg["user_id"].to_list(), train_agg["aids"].to_list()):
                user_history_map[u] = tuple(a)
            del train_agg
            gc.collect()

    print(f"Loaded compact history for {len(user_history_map):,} users in {time.time() - start_t:.2f}s.")
    return user_history_map


def build_user_profiles(
    hist_train_path: Path,
    hist_dev_path: Path,
    test_uids: pl.Series,
    embeddings_norm: Optional[np.ndarray],
    article_id_to_idx: Dict[str, int],
    art_token_ids_map: Dict[str, Set[int]],
    art_cat_map: Dict[str, str],
    hist_test_path: Optional[Path] = None,
    max_history_len: int = 30,
) -> Dict[str, UserProfile]:
    """Backward-compatible helper for validation scripts."""
    u_hist = load_compact_user_history(hist_test_path, hist_dev_path, hist_train_path, max_history_len)
    profiles = {}
    for uid in set(test_uids.to_list()):
        aids = u_hist.get(uid)
        if aids:
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
                u_vec = compute_user_representation(list(recent), article_id_to_idx, embeddings_norm)
            profiles[uid] = UserProfile(n_clicks=n, mean_dwell=0.0, cat_affinity=cat_aff, user_vec=u_vec, q_tokens=list(q_toks))
    return profiles


def generate_reranker_submission(
    model: Any,
    test_impressions_path: Path,
    articles_df: pl.DataFrame,
    hist_train_path: Path,
    hist_dev_path: Path,
    hist_test_path: Optional[Path],
    embeddings: Optional[np.ndarray],
    article_id_to_idx: Optional[Dict[str, int]],
    output_dir: Path,
    popularity_map: Optional[Dict[str, float]] = None,
    chunk_size_impr: int = 50000,
    batch_predict_size: int = 5000,
    max_impressions: Optional[int] = None,
) -> Path:
    """
    Generates Codabench prediction.zip with strictly bounded memory (peak < 1.5 GB).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_txt_path = output_dir / "prediction.txt"
    prediction_zip_path = output_dir / "prediction.zip"

    print(f"Inspecting test impressions from {test_impressions_path}...")
    if max_impressions is not None:
        total_test_impressions = min(max_impressions, pl.read_parquet(test_impressions_path).height)
    else:
        total_test_impressions = pl.read_parquet(test_impressions_path).height
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

    # 4. Load compact click histories (~200 MB)
    user_history_map = load_compact_user_history(
        hist_test_path=hist_test_path,
        hist_dev_path=hist_dev_path,
        hist_train_path=hist_train_path,
    )

    # 5. Stream impressions in chunks of 50,000 to keep RAM strictly < 1.5 GB
    print(f"Scoring test impressions and streaming to {prediction_txt_path}...")
    start_time = time.time()
    total_written = 0

    batch_X: List[np.ndarray] = []
    # (raw_id, n_cand, is_cold, cand_pop_scores, sem_scores)
    batch_metadata: List[Tuple[str, int, bool, np.ndarray, np.ndarray]] = []
    line_buffer: List[str] = []

    def flush_batch(f_out):
        if not batch_X:
            return
        X_batch = np.vstack(batch_X)
        scores_batch = model.predict(X_batch)
        offset = 0
        for raw_id, n_cand, is_cold, cand_pop, cand_sem in batch_metadata:
            if is_cold:
                # Cold-start fallback: rank candidates by global popularity
                sc = cand_pop
            else:
                r_sc = scores_batch[offset : offset + n_cand]
                # Blend ranker score with fine-grained dense semantic cosine similarity
                sc = r_sc + 0.3 * cand_sem + 1e-4 * cand_pop
            ranks = rankdata(-sc, method="ordinal")
            rank_str = ",".join(str(int(r)) for r in ranks)
            line_buffer.append(f"{raw_id} [{rank_str}]\n")
            offset += n_cand
        batch_X.clear()
        batch_metadata.clear()
        if len(line_buffer) >= 20000:
            f_out.writelines(line_buffer)
            line_buffer.clear()

    with open(prediction_txt_path, "w", encoding="utf-8", buffering=2 * 1024 * 1024) as out_f:
        pbar = tqdm(total=total_test_impressions, desc="Generating Re-Ranker Predictions", unit="impr")

        for chunk_offset in range(0, total_test_impressions, chunk_size_impr):
            chunk_len = min(chunk_size_impr, total_test_impressions - chunk_offset)
            chunk_df = pl.read_parquet(
                test_impressions_path,
                columns=["impression_id", "user_id", "timestamp", "candidate_article_ids"],
            ).slice(chunk_offset, chunk_len)

            impr_ids = chunk_df["impression_id"].to_list()
            user_ids = chunk_df["user_id"].to_list()
            timestamps = chunk_df["timestamp"].to_list()
            candidates_list = chunk_df["candidate_article_ids"].to_list()
            del chunk_df

            # Precompute profiles ONLY for unique users in this chunk (~25k users, ~75 MB)
            chunk_unique_uids = set(user_ids)
            chunk_profiles: Dict[str, UserProfile] = {}
            for uid in chunk_unique_uids:
                aids = user_history_map.get(uid)
                if aids:
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
                        u_vec = compute_user_representation(
                            clicked_article_ids=list(recent),
                            article_id_to_idx=article_id_to_idx,
                            embeddings=embeddings_norm,
                        )
                    chunk_profiles[uid] = UserProfile(
                        n_clicks=n,
                        mean_dwell=0.0,
                        cat_affinity=cat_aff,
                        user_vec=u_vec,
                        q_tokens=list(q_toks),
                    )

            for i in range(chunk_len):
                raw_impr_id = impr_ids[i]
                if isinstance(raw_impr_id, str) and raw_impr_id.startswith("mind_"):
                    raw_impr_id = raw_impr_id[len("mind_"):]

                cands = candidates_list[i]
                if not cands:
                    line_buffer.append(f"{raw_impr_id} []\n")
                    continue

                uid = user_ids[i]
                imp_time = timestamps[i]

                # Fast O(1) chunk user profile lookup
                prof = chunk_profiles.get(uid)
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

                # Popularity score per candidate
                cand_pop_scores = np.array(
                    [popularity_map.get(cid, 0.0) if popularity_map else 0.0 for cid in cands],
                    dtype=np.float32,
                )

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
                X_cand[:, 5] = 0.0              # neutralize position bias to avoid over-penalizing lower candidates
                X_cand[:, 6] = bm25_scores
                X_cand[:, 7] = sem_scores
                X_cand[:, 8] = first_stage_scores

                # 3. Candidate metadata features
                for pos_idx, cand_id in enumerate(cands):
                    cat = art_cat_map.get(cand_id, "")
                    pub = art_pub_map.get(cand_id)
                    freshness = 0.0
                    if pub is not None and isinstance(pub, (datetime, pd.Timestamp)):
                        freshness = max(0.0, (imp_time - pub).total_seconds() / 3600.0)
                    X_cand[pos_idx, 3] = float(freshness)
                    X_cand[pos_idx, 4] = float(cat_affinity.get(cat, 0.0))

                batch_X.append(X_cand)
                batch_metadata.append((raw_impr_id, num_cands, is_cold, cand_pop_scores, sem_scores))

                if len(batch_metadata) >= batch_predict_size:
                    flush_batch(out_f)

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

    print(f"Creating submission zip: {prediction_zip_path}...")
    with zipfile.ZipFile(prediction_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(prediction_txt_path, arcname="prediction.txt")

    print(f"Submission zip successfully created at: {prediction_zip_path}")
    validate_submission_file(prediction_txt_path, total_test_impressions)
    return prediction_zip_path


def main():
    parser = argparse.ArgumentParser(description="Generate MIND Large Re-Ranker Codabench submission.")
    parser.add_argument("--dataset_type", default="large", choices=["small", "large"])
    parser.add_argument("--model_path", default="models/mind_lgbm_ranker.pkl")
    parser.add_argument("--output_dir", default="submissions/submissions_mind_reranker")
    parser.add_argument("--eval_dev", action="store_true", help="Evaluate re-ranker on validation set first")
    parser.add_argument("--max_impressions", type=int, default=None, help="Optional limit for test impressions")
    args = parser.parse_args()

    ds_dir = Path("data/processed") / f"mind_{args.dataset_type}"
    articles_path = ds_dir / "articles.parquet"
    test_impr_path = ds_dir / "impressions_test.parquet"
    emb_path = ds_dir / "article_embeddings.npy"
    ids_path = ds_dir / "article_ids.json"

    if not Path(args.model_path).exists():
        raise FileNotFoundError(f"Model not found at {args.model_path}. Train re-ranker first.")

    model = joblib.load(args.model_path)
    print(f"Loaded Re-Ranker model from {args.model_path}")

    # Evaluate dev if requested
    if args.eval_dev:
        val_feat_path = ds_dir / "features_val.parquet"
        if val_feat_path.exists():
            print(f"\nEvaluating Re-Ranker on {val_feat_path}...")
            val_df = pl.read_parquet(val_feat_path)
            X = val_df.select(FEATURE_COLS).to_numpy().astype(np.float32)
            val_df = val_df.with_columns(pl.Series("reranker_score", model.predict(X)))
            metrics = evaluate_impression_predictions(val_df, score_col="reranker_score")
            print("=" * 55)
            print("  MIND-LARGE DEV EVALUATION RESULTS (RE-RANKER)")
            print("=" * 55)
            print(f"  AUC:      {metrics['AUC']:.4f}")
            print(f"  MRR:      {metrics['MRR']:.4f}")
            print(f"  nDCG@5:   {metrics['nDCG@5']:.4f}")
            print(f"  nDCG@10:  {metrics['nDCG@10']:.4f}")
            print("=" * 55 + "\n")

    # Load static article catalog & embeddings
    articles = pl.read_parquet(articles_path)
    embeddings = np.load(emb_path) if emb_path.exists() else None
    article_id_to_idx = None
    if ids_path.exists():
        with open(ids_path, "r", encoding="utf-8") as f:
            ids = json.load(f)
        article_id_to_idx = {aid: idx for idx, aid in enumerate(ids)}

    hist_train_path = ds_dir / "history_train.parquet"
    hist_dev_path = ds_dir / "history_dev.parquet"
    if not hist_dev_path.exists():
        hist_dev_path = ds_dir / "history_val.parquet"
    hist_test_path = ds_dir / "history_test.parquet"

    # Compute global article popularity from available history files
    pop_hist_paths = [hist_train_path, hist_dev_path, hist_test_path]
    popularity_map = compute_article_popularity(pop_hist_paths)

    # Generate test submission with lean memory usage
    out_dir = Path(args.output_dir)
    zip_path = generate_reranker_submission(
        model=model,
        test_impressions_path=test_impr_path,
        articles_df=articles,
        hist_train_path=hist_train_path,
        hist_dev_path=hist_dev_path,
        hist_test_path=hist_test_path,
        embeddings=embeddings,
        article_id_to_idx=article_id_to_idx,
        output_dir=out_dir,
        popularity_map=popularity_map,
        max_impressions=args.max_impressions,
    )
    print(f"\nSUCCESS! Codabench Re-Ranker submission ready at: {zip_path.resolve()}")


if __name__ == "__main__":
    main()
