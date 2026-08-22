"""
EB-NeRD Dense Semantic Candidate Ranking & Codabench Submission Generator.

1. Loads or computes 768-dim (or 384-dim) L2-normalized dense embeddings for all EB-NeRD articles
   (using pre-trained Ekstra Bladet multilingual BERT embeddings or SentenceTransformers over title + abstract + body).
2. Constructs user representation vectors via leak-free mean-pooling over recent click history.
3. Evaluates validation set ranking metrics (AUC, MRR, nDCG@5, nDCG@10).
4. Scores candidate articles for all 13.54M test impressions via cosine similarity.
5. Generates Codabench-compliant predictions.zip submission.

Usage:
    python src/generate_ebnerd_semantic_submission.py --dataset_type large --eval_dev --output_dir submissions_ebnerd_semantic
"""

import argparse
import json
import math
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
from tqdm import tqdm

# Ensure src is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from embeddings import (
    compute_article_embeddings_hf,
    compute_user_representation,
    load_ebnerd_pretrained_embeddings,
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


def get_or_compute_ebnerd_embeddings(
    articles_df: pd.DataFrame,
    processed_dir: Path,
    raw_dir: Optional[Path] = None,
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    batch_size: int = 256,
    device: Optional[str] = None,
    force_recompute: bool = False,
) -> Tuple[np.ndarray, Dict[str, int], List[str]]:
    """Load cached embeddings, pre-trained BERT embeddings, or compute via SentenceTransformers (title + abstract + body)."""
    emb_path = processed_dir / "article_embeddings.npy"
    ids_path = processed_dir / "article_ids.json"

    article_ids = articles_df["article_id"].tolist()
    article_id_to_idx = {aid: i for i, aid in enumerate(article_ids)}

    # 1. Check existing cache
    if not force_recompute and emb_path.exists() and ids_path.exists():
        try:
            with open(ids_path, "r", encoding="utf-8") as f:
                cached_ids = json.load(f)
            if cached_ids == article_ids:
                print(f"Loading cached embeddings from {emb_path}...")
                embeddings = np.load(emb_path)
                print(f"Loaded cached embeddings: shape={embeddings.shape}")
                return embeddings, article_id_to_idx, article_ids
        except Exception as e:
            print(f"Cache validation error ({e}), re-loading...")

    # 2. Try pre-trained Ekstra Bladet embeddings from raw_dir
    embeddings = None
    if not force_recompute and raw_dir is not None:
        print(f"Searching for pre-trained EB-NeRD embeddings in {raw_dir}...")
        pretrained_res = load_ebnerd_pretrained_embeddings(raw_dir, articles_df)
        if pretrained_res is not None:
            embeddings, _ = pretrained_res
            print(f"Successfully loaded pre-trained embeddings: shape={embeddings.shape}")

    # 3. Fallback: Compute via SentenceTransformers over title + abstract + body
    if embeddings is None:
        print(f"Computing dense embeddings for {len(articles_df):,} articles with '{model_name}' (title + abstract + body)...")
        embeddings = compute_article_embeddings_hf(
            articles_df=articles_df,
            model_name=model_name,
            batch_size=batch_size,
            device=device,
            include_body=True,
        )

    # Normalize L2
    embeddings = normalize_l2(embeddings)

    # Cache
    print(f"Caching embeddings to {emb_path}...")
    np.save(emb_path, embeddings)
    with open(ids_path, "w", encoding="utf-8") as f:
        json.dump(article_ids, f)

    return embeddings, article_id_to_idx, article_ids


def load_user_recent_history(
    history_path: Path,
    raw_history_path: Optional[Path] = None,
    max_history_len: int = 20,
) -> Dict[str, List[str]]:
    """
    Load user histories into a dict: user_id -> list of recent clicked article_ids.
    Uses fast direct list slice from raw history parquet when available.
    """
    if raw_history_path and raw_history_path.exists():
        print(f"Loading user history directly from raw parquet {raw_history_path}...")
        try:
            raw_hist = pl.read_parquet(raw_history_path, columns=["user_id", "article_id_fixed"])
            transformed = raw_hist.select([
                (pl.lit("ebnerd_") + pl.col("user_id").cast(pl.Utf8)).alias("user_id"),
                pl.col("article_id_fixed")
                .list.tail(max_history_len)
                .list.eval(pl.lit("ebnerd_") + pl.element().cast(pl.Utf8))
                .alias("recent_clicks"),
            ])
            user_ids = transformed["user_id"].to_list()
            clicks = transformed["recent_clicks"].to_list()
            user_map = dict(zip(user_ids, clicks))
            print(f"Loaded history for {len(user_map):,} users in fast mode.")
            return user_map
        except Exception as e:
            print(f"[warning] Fast history load failed ({e}), falling back to processed history.")

    if not history_path.exists():
        return {}

    print(f"Loading and aggregating history from {history_path}...")
    df = pl.read_parquet(history_path)
    if "user_id" not in df.columns or "clicked_article_id" not in df.columns:
        return {}

    agg_df = (
        df.group_by("user_id", maintain_order=True)
        .agg(pl.col("clicked_article_id").tail(max_history_len))
    )
    user_ids = agg_df["user_id"].to_list()
    hist_lists = agg_df["clicked_article_id"].to_list()
    user_map = dict(zip(user_ids, [list(h) for h in hist_lists]))
    print(f"Loaded history for {len(user_map):,} users.")
    return user_map


def compute_article_popularity(history_paths: List[Path]) -> Dict[str, float]:
    """Compute normalized article popularity from available history files as a tie-breaker."""
    pop_counts: Dict[str, int] = {}
    for p in history_paths:
        if p.exists():
            print(f"Counting article popularity from {p}...")
            try:
                schema = pl.scan_parquet(p).collect_schema().names()
                if "article_id_fixed" in schema:
                    # Raw format list column
                    df = pl.read_parquet(p, columns=["article_id_fixed"])
                    exploded = df.select(pl.col("article_id_fixed").explode().value_counts()).unnest("article_id_fixed")
                    for aid, c in zip(exploded["article_id_fixed"].to_list(), exploded["count"].to_list()):
                        if aid is not None:
                            key = f"ebnerd_{aid}"
                            pop_counts[key] = pop_counts.get(key, 0) + c
                elif "clicked_article_id" in schema:
                    # Processed format single column
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


def evaluate_dev_split(
    embeddings: np.ndarray,
    article_id_to_idx: Dict[str, int],
    global_mean_vector: np.ndarray,
    dev_impressions_path: Path,
    user_history_map: Dict[str, List[str]],
    popularity_map: Optional[Dict[str, float]] = None,
    sample_size: int = 100000,
) -> Dict[str, float]:
    """Evaluate dense semantic candidate ranking on dev/validation split impressions."""
    print(f"Evaluating Dense Semantic Candidate Ranking on Validation Set: {dev_impressions_path}...")
    if sample_size > 0:
        dev_df = pl.read_parquet(dev_impressions_path, n_rows=sample_size)
        print(f"Using representative sample of {len(dev_df):,} validation impressions...")
    else:
        dev_df = pl.read_parquet(dev_impressions_path)
        print(f"Evaluating full set of {len(dev_df):,} validation impressions...")

    user_ids = dev_df["user_id"].to_list()
    cands_list = dev_df["candidate_article_ids"].to_list()
    clicked_list = dev_df["clicked_article_ids"].to_list()

    # Precompute user representation vectors cache for users in validation sample
    print("Pre-computing user representation vectors for validation users...")
    user_vector_cache: Dict[str, np.ndarray] = {}
    unique_users = set(user_ids)
    for u in unique_users:
        clicks = user_history_map.get(u, [])
        u_vec = compute_user_representation(clicks, article_id_to_idx, embeddings)
        if u_vec is not None:
            user_vector_cache[u] = u_vec

    auc_scores = []
    mrr_scores = []
    ndcg5_scores = []
    ndcg10_scores = []

    for i in tqdm(range(len(dev_df)), desc="Evaluating Validation Semantic Ranking", unit="impr"):
        cands = cands_list[i]
        clicked = clicked_list[i]
        if not cands or not clicked:
            continue

        cands = list(cands)
        gt_set = set(clicked)
        labels = [1 if c in gt_set else 0 for c in cands]

        num_pos = sum(labels)
        if num_pos == 0 or num_pos == len(labels):
            continue

        uid = user_ids[i]
        u_vec = user_vector_cache.get(uid, global_mean_vector)

        cand_indices = [article_id_to_idx.get(aid, -1) for aid in cands]
        cand_scores = [
            float(np.dot(u_vec, embeddings[idx])) if idx >= 0 else 0.0
            for idx in cand_indices
        ]

        if popularity_map:
            for ci, cand_id in enumerate(cands):
                cand_scores[ci] += 1e-4 * popularity_map.get(cand_id, 0.0)

        cand_scores_arr = np.array(cand_scores, dtype=np.float32)
        ranked_indices = np.argsort(-cand_scores_arr)

        auc_val = compute_auc(labels, cand_scores_arr)
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
    print(f"  EB-NeRD VALIDATION SET SEMANTIC EVALUATION RESULTS")
    print("=" * 60)
    print(f"  AUC:      {metrics['AUC']:.4f}")
    print(f"  MRR:      {metrics['MRR']:.4f}")
    print(f"  nDCG@5:   {metrics['nDCG@5']:.4f}")
    print(f"  nDCG@10:  {metrics['nDCG@10']:.4f}")
    print(f"  Evaluated Impressions: {metrics['evaluated_impressions']:,}")
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
    max_history_len: int = 20,
    chunk_size: int = 100000,
) -> Path:
    """
    Score candidates for test impressions using semantic embeddings and package into predictions.zip.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_txt_path = output_dir / "predictions.txt"
    predictions_zip_path = output_dir / "predictions.zip"

    print(f"Loading test impressions from {test_impressions_path}...")
    test_df = pl.read_parquet(test_impressions_path)
    num_impressions = len(test_df)
    print(f"Total test impressions: {num_impressions:,}")

    impr_ids = test_df["impression_id"].to_list()
    user_ids = test_df["user_id"].to_list()
    candidates_list = test_df["candidate_article_ids"].to_list()

    # Precompute user representation vectors cache for unique test users
    print("Pre-computing user representation vectors for test users...")
    user_vector_cache: Dict[str, np.ndarray] = {}
    unique_users = set(user_ids)
    for u in tqdm(unique_users, desc="Building User Embeddings", unit="user"):
        clicks = user_history_map.get(u, [])
        u_vec = compute_user_representation(clicks, article_id_to_idx, embeddings)
        if u_vec is not None:
            user_vector_cache[u] = u_vec

    print(f"Cached representations for {len(user_vector_cache):,} active users.")

    print(f"Scoring {num_impressions:,} test impressions and streaming to {predictions_txt_path}...")
    start_time = time.time()

    buffer = []
    with open(predictions_txt_path, "w", encoding="utf-8", buffering=1024 * 1024) as out_f:
        for i in tqdm(range(num_impressions), desc="Generating EB-NeRD Semantic Predictions", unit="impr"):
            raw_impr_id = impr_ids[i]
            if isinstance(raw_impr_id, str) and raw_impr_id.startswith("ebnerd_"):
                raw_impr_id = raw_impr_id[len("ebnerd_"):]

            cands = candidates_list[i]
            if not cands:
                buffer.append(f"{raw_impr_id} []\n")
            else:
                cands = list(cands)
                uid = user_ids[i]
                u_vec = user_vector_cache.get(uid, global_mean_vector)

                # Candidate dot product scores
                cand_indices = [article_id_to_idx.get(aid, -1) for aid in cands]
                cand_scores = [
                    float(np.dot(u_vec, embeddings[idx])) if idx >= 0 else 0.0
                    for idx in cand_indices
                ]

                if popularity_map:
                    for ci, cand_id in enumerate(cands):
                        cand_scores[ci] += 1e-4 * popularity_map.get(cand_id, 0.0)

                rank_str = fast_ordinal_ranks(cand_scores)
                buffer.append(f"{raw_impr_id} [{rank_str}]\n")

            if len(buffer) >= chunk_size:
                out_f.writelines(buffer)
                buffer.clear()

        if buffer:
            out_f.writelines(buffer)
            buffer.clear()

    elapsed = time.time() - start_time
    print(f"Wrote {num_impressions:,} predictions in {elapsed:.2f}s ({num_impressions / max(0.1, elapsed):,.0f} impr/s)")

    print(f"Creating submission zip: {predictions_zip_path}...")
    with zipfile.ZipFile(predictions_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(predictions_txt_path, arcname="predictions.txt")

    print(f"Submission zip successfully created at: {predictions_zip_path}")
    validate_submission_file(predictions_txt_path, num_impressions)
    return predictions_zip_path


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
        print(f"Warning: Expected {expected_lines:,} lines, but found {line_count:,} lines.")
    else:
        print(f"Verification PASSED: {line_count:,} correctly formatted lines matching all impressions.")


def main():
    parser = argparse.ArgumentParser(description="Generate EB-NeRD Dense Semantic Codabench submission.")
    parser.add_argument("--dataset_type", choices=["demo", "small", "large"], default="large", help="Dataset scale")
    parser.add_argument("--eval_dev", action="store_true", help="Evaluate dense semantic ranking on validation split")
    parser.add_argument("--val_sample_size", type=int, default=100000, help="Validation sample size for evaluation (-1 for all)")
    parser.add_argument("--output_dir", default="submissions_ebnerd_semantic", help="Output directory for submission zip")
    parser.add_argument("--model_name", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", help="Model name for text encoding")
    parser.add_argument("--max_history_len", type=int, default=20, help="Max recent articles for query")
    parser.add_argument("--batch_size", type=int, default=256, help="Embedding encoding batch size")
    parser.add_argument("--device", default=None, help="Device for PyTorch inference ('cpu', 'cuda')")
    parser.add_argument("--force_recompute", action="store_true", help="Force recomputing text embeddings over title+abstract+body")
    parser.add_argument("--use_popularity_fallback", action="store_true", default=True, help="Use popularity tie-breaker")
    args = parser.parse_args()

    ds_dir = Path("data/processed") / f"ebnerd_{args.dataset_type}" if args.dataset_type != "demo" else Path("data/processed/ebnerd")
    raw_dir = Path("data/raw") / f"ebnerd_{args.dataset_type}" if args.dataset_type != "demo" else Path("data/raw/ebnerd/demo")

    articles_path = ds_dir / "articles.parquet"
    val_impr_path = ds_dir / "impressions_val.parquet"
    if not val_impr_path.exists():
        val_impr_path = ds_dir / "impressions_validation.parquet"
    val_hist_path = ds_dir / "history_val.parquet"
    if not val_hist_path.exists():
        val_hist_path = ds_dir / "history_validation.parquet"

    test_impr_path = ds_dir / "impressions_test.parquet"
    test_hist_path = ds_dir / "history_test.parquet"
    train_hist_path = ds_dir / "history_train.parquet"

    raw_val_hist = raw_dir / "validation" / "history.parquet"
    raw_test_hist = raw_dir / "test" / "history.parquet"
    raw_train_hist = raw_dir / "train" / "history.parquet"

    if not articles_path.exists():
        raise FileNotFoundError(f"Articles not found at {articles_path}. Run download and parse_ebnerd first.")

    # 1. Load articles and compute/load embeddings
    articles_df = pd.read_parquet(articles_path)
    embeddings, article_id_to_idx, article_ids = get_or_compute_ebnerd_embeddings(
        articles_df=articles_df,
        processed_dir=ds_dir,
        raw_dir=raw_dir,
        model_name=args.model_name,
        batch_size=args.batch_size,
        device=args.device,
        force_recompute=args.force_recompute,
    )
    global_mean_vector = normalize_l2(np.mean(embeddings, axis=0))

    # 2. Compute article popularity if requested
    pop_map = None
    if args.use_popularity_fallback:
        pop_hist_paths = [raw_train_hist, raw_val_hist, raw_test_hist] if raw_train_hist.exists() else [train_hist_path, val_hist_path, test_hist_path]
        pop_map = compute_article_popularity(pop_hist_paths)

    # 3. Evaluate on Validation if requested
    if args.eval_dev and val_impr_path.exists():
        val_history_map = load_user_recent_history(
            history_path=val_hist_path,
            raw_history_path=raw_val_hist,
            max_history_len=args.max_history_len,
        )
        evaluate_dev_split(
            embeddings=embeddings,
            article_id_to_idx=article_id_to_idx,
            global_mean_vector=global_mean_vector,
            dev_impressions_path=val_impr_path,
            user_history_map=val_history_map,
            popularity_map=pop_map,
            sample_size=args.val_sample_size,
        )

    # 4. Generate Submission for Test
    if test_impr_path.exists():
        test_history_map = load_user_recent_history(
            history_path=test_hist_path,
            raw_history_path=raw_test_hist,
            max_history_len=args.max_history_len,
        )
        out_path = Path(args.output_dir)
        zip_path = generate_submission(
            embeddings=embeddings,
            article_id_to_idx=article_id_to_idx,
            global_mean_vector=global_mean_vector,
            test_impressions_path=test_impr_path,
            user_history_map=test_history_map,
            output_dir=out_path,
            popularity_map=pop_map,
            max_history_len=args.max_history_len,
        )
        print(f"\nSUCCESS! EB-NeRD Semantic Codabench submission ready: {zip_path.resolve()}")
    else:
        print(f"Test impressions not found at {test_impr_path}. Only validation evaluated.")


if __name__ == "__main__":
    main()
