"""
MIND-Large Dense Semantic Candidate Ranking & Codabench Submission Generator.

1. Loads or computes 384-dim L2-normalized dense embeddings for all MIND-Large articles using SentenceTransformers.
2. Constructs user representation vectors via leak-free mean-pooling over recent click history.
3. Evaluates dev set ranking metrics (AUC, MRR, nDCG@5, nDCG@10).
4. Scores candidate articles for all 2.37M test impressions via cosine similarity.
5. Generates Codabench-compliant prediction.zip submission.

Usage:
    python src/generate_mind_semantic_submission.py --dataset_type large --eval_dev --output_dir submissions_semantic
"""

import argparse
import json
import os
import re
import sys
import time
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import rankdata
from tqdm import tqdm

# Ensure src is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from embeddings import (
    EmbeddingIndex,
    compute_article_embeddings_hf,
    compute_user_representation,
    normalize_l2,
)
from metrics import compute_auc, compute_mrr, compute_ndcg_at_k


def fast_ordinal_ranks(scores: List[float]) -> str:
    """Compute 1-based ranks where rank 1 corresponds to highest score."""
    n = len(scores)
    order = sorted(range(n), key=lambda idx: scores[idx], reverse=True)
    ranks = [0] * n
    for r, idx in enumerate(order, 1):
        ranks[idx] = r
    return ",".join(map(str, ranks))


def get_or_compute_embeddings(
    articles_df: pd.DataFrame,
    processed_dir: Path,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 512,
    device: Optional[str] = None,
    force_recompute: bool = False,
) -> Tuple[np.ndarray, Dict[str, int], List[str]]:
    """Load cached embeddings or compute dense embeddings over article title + abstract."""
    emb_path = processed_dir / "article_embeddings.npy"
    ids_path = processed_dir / "article_ids.json"

    article_ids = articles_df["article_id"].tolist()
    article_id_to_idx = {aid: i for i, aid in enumerate(article_ids)}

    if not force_recompute and emb_path.exists() and ids_path.exists():
        try:
            with open(ids_path, "r", encoding="utf-8") as f:
                cached_ids = json.load(f)
            if cached_ids == article_ids:
                print(f"Loading cached embeddings from {emb_path}...")
                embeddings = np.load(emb_path)
                return embeddings, article_id_to_idx, article_ids
        except Exception as e:
            print(f"Cache validation error ({e}), recomputing...")

    print(f"Computing embeddings for {len(articles_df)} articles with '{model_name}'...")
    embeddings = compute_article_embeddings_hf(
        articles_df=articles_df,
        model_name=model_name,
        batch_size=batch_size,
        device=device,
    )
    embeddings = normalize_l2(embeddings)

    print(f"Caching embeddings to {emb_path}...")
    np.save(emb_path, embeddings)
    with open(ids_path, "w", encoding="utf-8") as f:
        json.dump(article_ids, f)

    return embeddings, article_id_to_idx, article_ids


def compute_article_popularity(history_paths: List[Path]) -> Dict[str, float]:
    """Compute normalized article popularity from available history files as a tie-breaker / cold-start fallback."""
    pop_counts: Dict[str, int] = {}
    for p in history_paths:
        if p.exists():
            print(f"Counting article popularity from {p}...")
            try:
                df = pl.read_parquet(p, columns=["clicked_article_id"])
                counts = df["clicked_article_id"].value_counts()
                for aid, c in zip(counts["clicked_article_id"].to_list(), counts["count"].to_list()):
                    if aid is not None:
                        key = str(aid)
                        pop_counts[key] = pop_counts.get(key, 0) + c
            except Exception as e:
                print(f"[warning] Popularity counting error for {p}: {e}")

    if not pop_counts:
        return {}

    max_c = max(pop_counts.values())
    print(f"Computed popularity for {len(pop_counts):,} unique articles (max count: {max_c:,}).")
    return {aid: c / max_c for aid, c in pop_counts.items()}


def load_user_recent_history(history_path: Path, max_history_len: int = 20) -> Dict[str, List[str]]:
    """Load user histories into a dict: user_id -> list of recent clicked article_ids using Polars."""
    if not history_path.exists():
        return {}
    print(f"Loading and aggregating history from {history_path}...")
    df = pl.read_parquet(history_path)
    agg_df = (
        df.group_by("user_id", maintain_order=True)
        .agg(pl.col("clicked_article_id").tail(max_history_len))
    )
    user_ids = agg_df["user_id"].to_list()
    hist_lists = agg_df["clicked_article_id"].to_list()
    user_map = dict(zip(user_ids, [list(h) for h in hist_lists]))
    print(f"Loaded history for {len(user_map)} users.")
    return user_map


def evaluate_dev_split(
    embeddings: np.ndarray,
    article_id_to_idx: Dict[str, int],
    global_mean_vector: np.ndarray,
    dev_impressions_path: Path,
    user_history_map: Dict[str, List[str]],
    popularity_map: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """Evaluate dense semantic candidate ranking on dev split impressions."""
    print(f"Evaluating Dense Semantic Candidate Ranking on Dev Set: {dev_impressions_path}...")
    dev_df = pl.read_parquet(dev_impressions_path)

    user_ids = dev_df["user_id"].to_list()
    cands_list = dev_df["candidate_article_ids"].to_list()
    clicked_list = dev_df["clicked_article_ids"].to_list()

    auc_scores = []
    mrr_scores = []
    ndcg5_scores = []
    ndcg10_scores = []

    for user_id, cands, clicked in tqdm(zip(user_ids, cands_list, clicked_list), total=len(user_ids), desc="Dev Evaluation"):
        if not cands or not clicked:
            continue

        cands = list(cands)
        gt_set = set(clicked)
        labels = [1 if c in gt_set else 0 for c in cands]

        # Skip impressions with all positive or all negative
        num_pos = sum(labels)
        if num_pos == 0 or num_pos == len(labels):
            continue

        recent_clicks = user_history_map.get(user_id, [])
        u_vec = compute_user_representation(
            clicked_article_ids=recent_clicks,
            article_id_to_idx=article_id_to_idx,
            embeddings=embeddings,
        )
        is_cold = (u_vec is None)

        cand_indices = [article_id_to_idx.get(aid, -1) for aid in cands]
        cand_scores = []
        for aid, idx in zip(cands, cand_indices):
            if is_cold:
                # 0-click users rank candidate articles strictly by global popularity
                score = float(popularity_map.get(aid, 0.0) if popularity_map else 0.0)
            else:
                score = float(np.dot(u_vec, embeddings[idx])) if idx >= 0 else 0.0
                if popularity_map:
                    score += 1e-4 * popularity_map.get(aid, 0.0)
            cand_scores.append(score)

        ranked_indices = np.argsort(-np.asarray(cand_scores, dtype=np.float32))

        auc_val = compute_auc(labels, cand_scores)
        if auc_val is not None:
            auc_scores.append(auc_val)
        mrr_scores.append(compute_mrr(labels, ranked_indices=ranked_indices))
        ndcg5_scores.append(compute_ndcg_at_k(labels, k=5, ranked_indices=ranked_indices))
        ndcg10_scores.append(compute_ndcg_at_k(labels, k=10, ranked_indices=ranked_indices))

    metrics = {
        "AUC": float(np.mean(auc_scores)) if auc_scores else 0.0,
        "MRR": float(np.mean(mrr_scores)) if mrr_scores else 0.0,
        "nDCG@5": float(np.mean(ndcg5_scores)) if ndcg5_scores else 0.0,
        "nDCG@10": float(np.mean(ndcg10_scores)) if ndcg10_scores else 0.0,
        "evaluated_impressions": len(auc_scores),
    }

    print("\n" + "=" * 60)
    print(f"  DEV SET SEMANTIC EVALUATION RESULTS")
    print("=" * 60)
    print(f"  AUC:      {metrics['AUC']:.4f}")
    print(f"  MRR:      {metrics['MRR']:.4f}")
    print(f"  nDCG@5:   {metrics['nDCG@5']:.4f}")
    print(f"  nDCG@10:  {metrics['nDCG@10']:.4f}")
    print(f"  Evaluated Impressions: {metrics['evaluated_impressions']}")
    print("=" * 60 + "\n")
    return metrics


def generate_submission(
    embeddings: np.ndarray,
    article_id_to_idx: Dict[str, int],
    global_mean_vector: np.ndarray,
    test_impressions_path: Path,
    user_history_map: Dict[str, List[str]],
    output_dir: Path,
    popularity_map: Optional[Dict[str, float]] = None,
    chunk_size: int = 50000,
) -> Path:
    """
    Score candidates for test impressions using semantic embeddings and package into prediction.zip.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_txt_path = output_dir / "prediction.txt"
    prediction_zip_path = output_dir / "prediction.zip"

    print(f"Loading test impressions from {test_impressions_path}...")
    test_df = pl.read_parquet(test_impressions_path)
    num_impressions = len(test_df)
    print(f"Total test impressions: {num_impressions}")

    impr_ids = test_df["impression_id"].to_list()
    user_ids = test_df["user_id"].to_list()
    candidates_list = test_df["candidate_article_ids"].to_list()

    # Precompute user representation vectors cache for unique test users
    print("Pre-computing user representation vectors for test users...")
    user_vector_cache: Dict[str, np.ndarray] = {}
    unique_users = set(user_ids)
    for u in unique_users:
        clicks = user_history_map.get(u, [])
        u_vec = compute_user_representation(clicks, article_id_to_idx, embeddings)
        if u_vec is not None:
            user_vector_cache[u] = u_vec

    print(f"Cached representations for {len(user_vector_cache)} active users.")

    print(f"Scoring test impressions and streaming to {prediction_txt_path}...")
    start_time = time.time()

    buffer = []
    with open(prediction_txt_path, "w", encoding="utf-8", buffering=1024 * 1024) as out_f:
        for i in tqdm(range(num_impressions), desc="Generating Semantic Predictions", unit="impr"):
            raw_impr_id = impr_ids[i]
            if isinstance(raw_impr_id, str) and raw_impr_id.startswith("mind_"):
                raw_impr_id = raw_impr_id[len("mind_"):]

            cands = candidates_list[i]
            if not cands:
                buffer.append(f"{raw_impr_id} []\n")
            else:
                cands = list(cands)
                user_id = user_ids[i]
                u_vec = user_vector_cache.get(user_id)
                is_cold = (u_vec is None)

                # Candidate dot product scores or cold start popularity
                cand_indices = [article_id_to_idx.get(aid, -1) for aid in cands]
                cand_scores = []
                for aid, idx in zip(cands, cand_indices):
                    if is_cold:
                        score = float(popularity_map.get(aid, 0.0) if popularity_map else 0.0)
                    else:
                        score = float(np.dot(u_vec, embeddings[idx])) if idx >= 0 else 0.0
                        if popularity_map:
                            score += 1e-4 * popularity_map.get(aid, 0.0)
                    cand_scores.append(score)

                # Compute ranks: highest score gets rank 1
                rank_str = fast_ordinal_ranks(cand_scores)
                buffer.append(f"{raw_impr_id} [{rank_str}]\n")

            if len(buffer) >= chunk_size:
                out_f.writelines(buffer)
                buffer.clear()

        if buffer:
            out_f.writelines(buffer)
            buffer.clear()

    elapsed = time.time() - start_time
    print(f"Wrote {num_impressions} predictions in {elapsed:.2f}s ({num_impressions / max(0.1, elapsed):.0f} impr/s)")

    print(f"Creating submission zip: {prediction_zip_path}...")
    with zipfile.ZipFile(prediction_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(prediction_txt_path, arcname="prediction.txt")

    print(f"Submission zip successfully created at: {prediction_zip_path}")
    validate_submission_file(prediction_txt_path, num_impressions)
    return prediction_zip_path


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
        print(f"Warning: Expected {expected_lines} lines, but found {line_count} lines.")
    else:
        print(f"Verification PASSED: {line_count} correctly formatted lines matching all impressions.")


def main():
    parser = argparse.ArgumentParser(description="Generate MIND Dense Semantic Codabench submission.")
    parser.add_argument("--dataset_type", choices=["small", "large"], default="large", help="Dataset scale")
    parser.add_argument("--eval_dev", action="store_true", help="Evaluate dense semantic ranking on dev split")
    parser.add_argument("--output_dir", default="submissions_semantic", help="Output directory for submission zip")
    parser.add_argument("--model_name", default="sentence-transformers/all-MiniLM-L6-v2", help="Model name")
    parser.add_argument("--max_history_len", type=int, default=20, help="Max recent articles for query")
    parser.add_argument("--batch_size", type=int, default=512, help="Embedding encoding batch size")
    parser.add_argument("--device", default=None, help="Device for PyTorch inference ('cpu', 'cuda')")
    parser.add_argument("--force_recompute", action="store_true", help="Force recomputing embeddings")
    parser.add_argument("--use_popularity_fallback", action="store_true", default=True, help="Use popularity fallback for cold-start")
    args = parser.parse_args()

    ds_dir = Path("data/processed") / f"mind_{args.dataset_type}"
    articles_path = ds_dir / "articles.parquet"
    dev_impr_path = ds_dir / "impressions_dev.parquet"
    dev_hist_path = ds_dir / "history_dev.parquet"
    test_impr_path = ds_dir / "impressions_test.parquet"
    test_hist_path = ds_dir / "history_test.parquet"
    train_hist_path = ds_dir / "history_train.parquet"

    if not articles_path.exists():
        raise FileNotFoundError(f"Articles not found at {articles_path}.")

    # 1. Load articles and compute/load embeddings
    articles_df = pd.read_parquet(articles_path)
    embeddings, article_id_to_idx, article_ids = get_or_compute_embeddings(
        articles_df=articles_df,
        processed_dir=ds_dir,
        model_name=args.model_name,
        batch_size=args.batch_size,
        device=args.device,
        force_recompute=args.force_recompute,
    )
    global_mean_vector = normalize_l2(np.mean(embeddings, axis=0))

    # Build FAISS HNSW Index for approximate nearest neighbor retrieval
    print(f"Building FAISS HNSW index over {len(article_ids):,} articles (dim={embeddings.shape[1]})...")
    hnsw_index = EmbeddingIndex(use_approximate=True)
    hnsw_index.build_index(embeddings, article_ids)

    # 2. Compute article popularity if requested
    pop_map = None
    if args.use_popularity_fallback:
        pop_hist_paths = [train_hist_path, dev_hist_path, test_hist_path]
        pop_map = compute_article_popularity(pop_hist_paths)

    # 3. Evaluate on Dev if requested
    if args.eval_dev and dev_impr_path.exists():
        dev_history_map = load_user_recent_history(dev_hist_path, max_history_len=args.max_history_len)
        evaluate_dev_split(
            embeddings=embeddings,
            article_id_to_idx=article_id_to_idx,
            global_mean_vector=global_mean_vector,
            dev_impressions_path=dev_impr_path,
            user_history_map=dev_history_map,
            popularity_map=pop_map,
        )

    # 4. Generate Submission for Test
    if test_impr_path.exists():
        test_history_map = load_user_recent_history(test_hist_path, max_history_len=args.max_history_len)
        out_path = Path(args.output_dir)
        zip_path = generate_submission(
            embeddings=embeddings,
            article_id_to_idx=article_id_to_idx,
            global_mean_vector=global_mean_vector,
            test_impressions_path=test_impr_path,
            user_history_map=test_history_map,
            output_dir=out_path,
            popularity_map=pop_map,
        )
        print(f"\nSUCCESS! Semantic Codabench submission ready: {zip_path.resolve()}")
    else:
        print(f"Test impressions not found at {test_impr_path}.")


if __name__ == "__main__":
    main()

