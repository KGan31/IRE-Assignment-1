"""
Offline Evaluation Runner for Embedding-based Semantic Candidate Generation and Ranking.

Evaluates dense semantic retrieval:
1. Loads/computes article embeddings (MIND via sentence-transformers, EB-NeRD via BERT/W2V/paraphrase).
2. Builds FAISS dense vector index.
3. Constructs user representation vectors via leak-free mean-pooling over recent clicks.
4. Evaluates:
   - Candidate Retrieval (Global Catalog): Recall@50, Recall@100, Recall@200, ILD@10, Novelty@10, Coverage@10
   - Impression Ranking: AUC, MRR, nDCG@5, nDCG@10, ILD@10, Novelty@10
   - Slicing: Cold-Start vs. Warm users, Head vs. Tail articles
   - Statistical Rigor: Bootstrap 95% Confidence Intervals for all metrics

Usage:
    python src/eval_embeddings.py --dataset mind --split val
    python src/eval_embeddings.py --dataset ebnerd --split val
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Ensure src directory is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from embeddings import (
    EmbeddingIndex,
    compute_user_representation,
    get_or_compute_article_embeddings,
    normalize_l2,
)
from eval_harness import OfflineEvaluationHarness, print_evaluation_summary
from metrics import compute_recall_at_k

# Backwards compatibility alias
calculate_recall_at_k = compute_recall_at_k


def get_popular_articles(history_df: pd.DataFrame, top_n: int = 200) -> List[str]:
    """Fallback candidate list of most frequent clicked articles in history."""
    if history_df.empty or "clicked_article_id" not in history_df.columns:
        return []
    top_articles = (
        history_df["clicked_article_id"].value_counts().head(top_n).index.tolist()
    )
    return top_articles


def evaluate_embeddings(
    dataset_name: str,
    split_name: str,
    processed_dir: Path,
    raw_dir: Optional[Path] = None,
    model_name: Optional[str] = None,
    max_history_len: int = 20,
    eval_mode: str = "all",
    k_list: List[int] = [50, 100, 200],
    batch_size: int = 2000,
    device: Optional[str] = None,
    force_recompute: bool = False,
    n_bootstraps: int = 1000,
) -> Dict[str, Any]:
    """Run Embedding-based candidate retrieval and ranking evaluation."""
    ds_processed = processed_dir / dataset_name
    articles_path = ds_processed / "articles.parquet"
    impressions_path = ds_processed / f"impressions_{split_name}.parquet"
    history_path = ds_processed / f"history_{split_name}.parquet"

    if not articles_path.exists():
        raise FileNotFoundError(f"Articles file not found: {articles_path}")
    if not impressions_path.exists():
        raise FileNotFoundError(f"Impressions file not found: {impressions_path}")

    # 1. Load articles and compute/load embeddings
    print(f"[{dataset_name} | {split_name}] Loading articles from {articles_path}...")
    articles_df = pd.read_parquet(articles_path)

    print(f"[{dataset_name} | {split_name}] Preparing embeddings for {len(articles_df)} articles...")
    embeddings, article_id_to_idx, article_ids = get_or_compute_article_embeddings(
        dataset_name=dataset_name,
        articles_df=articles_df,
        processed_dir=processed_dir,
        raw_dir=raw_dir,
        model_name=model_name,
        device=device,
        force_recompute=force_recompute,
    )

    # Ensure embeddings are L2 normalized
    embeddings_norm = normalize_l2(embeddings)

    # 2. Build FAISS Vector Index
    print(f"[{dataset_name} | {split_name}] Building FAISS vector index (dim={embeddings.shape[1]})...")
    vector_index = EmbeddingIndex()
    vector_index.build_index(embeddings_norm, article_ids)

    # Compute global mean vector for cold-start users
    global_mean_vector = normalize_l2(np.mean(embeddings_norm, axis=0))

    # 3. Load history & impressions
    print(f"[{dataset_name} | {split_name}] Loading impressions from {impressions_path}...")
    impressions_df = pd.read_parquet(impressions_path)

    history_df = pd.DataFrame()
    if history_path.exists():
        history_df = pd.read_parquet(history_path)

    # Initialize evaluation harness
    harness = OfflineEvaluationHarness.from_data(
        articles_df=articles_df,
        history_df=history_df,
        embeddings=embeddings_norm,
        article_id_to_idx=article_id_to_idx,
        n_bootstraps=n_bootstraps,
        cold_start_threshold=5,
    )

    # Popular fallback
    popular_fallback = get_popular_articles(history_df, top_n=max(k_list))
    if not popular_fallback and not articles_df.empty:
        popular_fallback = articles_df["article_id"].head(max(k_list)).tolist()

    # Identify head articles (top 20% most clicked articles in history)
    head_articles: Set[str] = set()
    if not history_df.empty and "clicked_article_id" in history_df.columns:
        counts = history_df["clicked_article_id"].value_counts()
        n_head = max(1, int(len(counts) * 0.2))
        head_articles = set(counts.head(n_head).index)

    # Build user click history map: user_id -> sorted list of (click_time, article_id)
    user_history_map: Dict[str, List[Tuple[pd.Timestamp, str]]] = {}
    if not history_df.empty and "user_id" in history_df.columns:
        grouped = history_df.groupby("user_id")
        for user_id, group in grouped:
            clicks = list(zip(group["click_time"], group["clicked_article_id"]))
            clicks.sort(key=lambda x: x[0])
            user_history_map[str(user_id)] = clicks

    max_k = max(k_list)

    # 4. Construct user vectors for impressions (leakage-free)
    print(f"[{dataset_name} | {split_name}] Encoding user representation vectors for {len(impressions_df)} impressions...")
    user_vectors_list: List[np.ndarray] = []
    ground_truths: List[Set[str]] = []
    history_lengths: List[int] = []
    is_head_gt: List[bool] = []
    impression_cands_list: List[List[str]] = []
    impression_labels_list: List[List[int]] = []

    user_ids = impressions_df["user_id"].astype(str).tolist()
    timestamps = impressions_df["timestamp"].tolist()
    clicked_lists = impressions_df["clicked_article_ids"].tolist()
    candidate_lists = (
        impressions_df["candidate_article_ids"].tolist()
        if "candidate_article_ids" in impressions_df.columns
        else [[] for _ in range(len(impressions_df))]
    )

    for user_id, ts, gt, cand in zip(user_ids, timestamps, clicked_lists, candidate_lists):
        if isinstance(gt, np.ndarray):
            gt = gt.tolist()
        gt_set = set(gt)

        if not gt_set:
            continue

        ground_truths.append(gt_set)

        if isinstance(cand, np.ndarray):
            cand = cand.tolist()
        cand_list = list(cand) if cand else []
        impression_cands_list.append(cand_list)
        impression_labels_list.append([1 if aid in gt_set else 0 for aid in cand_list])

        # Check if GT contains head articles
        has_head = any(aid in head_articles for aid in gt_set)
        is_head_gt.append(has_head)

        # Get leakage-free prior clicks
        user_clicks = user_history_map.get(user_id, [])
        prior_clicks = [aid for c_time, aid in user_clicks if c_time < ts]
        recent_clicks = prior_clicks[-max_history_len:] if prior_clicks else []
        history_lengths.append(len(recent_clicks))

        u_vec = compute_user_representation(
            clicked_article_ids=recent_clicks,
            article_id_to_idx=article_id_to_idx,
            embeddings=embeddings_norm,
        )

        if u_vec is None:
            user_vectors_list.append(global_mean_vector)
        else:
            user_vectors_list.append(u_vec)

    query_matrix = np.vstack(user_vectors_list).astype(np.float32)
    num_queries = len(query_matrix)

    all_results: Dict[str, Any] = {}

    # 5. Impression Candidate Ranking (AUC, MRR, nDCG@5, nDCG@10)
    if eval_mode in ["all", "impression"] and any(impression_cands_list):
        print(f"[{dataset_name} | {split_name}] Computing Impression Candidate Ranking scores (AUC, MRR, nDCG@K)...")
        all_candidate_scores: List[List[float]] = []

        for i in range(num_queries):
            u_vec = query_matrix[i]
            cands = impression_cands_list[i]
            cand_scores = []
            for aid in cands:
                if aid in article_id_to_idx:
                    doc_vec = embeddings_norm[article_id_to_idx[aid]]
                    score = float(np.dot(u_vec, doc_vec))
                else:
                    score = 0.0
                cand_scores.append(score)
            all_candidate_scores.append(cand_scores)

        ranking_results = harness.evaluate_impression_ranking(
            candidate_article_lists=impression_cands_list,
            candidate_labels=impression_labels_list,
            candidate_scores=all_candidate_scores,
            user_history_lengths=history_lengths,
        )

        print_evaluation_summary(
            f"Dense Semantic Impression Ranking: {dataset_name.upper()} ({split_name})",
            ranking_results,
        )
        all_results["ranking"] = ranking_results

    # 6. Global Candidate Retrieval (Recall@K, ILD, Novelty, Coverage)
    if eval_mode in ["all", "global"]:
        print(f"[{dataset_name} | {split_name}] Running fast batch vector search for {num_queries} queries...")
        all_retrieved_ids: List[List[str]] = []

        for start_idx in tqdm(range(0, num_queries, batch_size), desc="Dense Retrieval"):
            end_idx = min(start_idx + batch_size, num_queries)
            batch_q = query_matrix[start_idx:end_idx]

            batch_retrieved_pairs = vector_index.batch_search(batch_q, top_k=max_k)

            for i, pairs in enumerate(batch_retrieved_pairs):
                retrieved_ids = [aid for aid, score in pairs]

                # Fallback padding
                if len(retrieved_ids) < max_k:
                    existing = set(retrieved_ids)
                    for pop_id in popular_fallback:
                        if pop_id not in existing:
                            retrieved_ids.append(pop_id)
                            existing.add(pop_id)
                            if len(retrieved_ids) >= max_k:
                                break

                all_retrieved_ids.append(retrieved_ids)

        retrieval_results = harness.evaluate_retrieval(
            retrieved_lists=all_retrieved_ids,
            ground_truth_sets=ground_truths,
            user_history_lengths=history_lengths,
            is_head_flags=is_head_gt,
            k_list=tuple(k_list),
            beyond_k=10,
        )

        print_evaluation_summary(
            f"Dense Candidate Retrieval (Global Search): {dataset_name.upper()} ({split_name})",
            retrieval_results,
        )
        all_results["retrieval"] = retrieval_results

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Run Embedding-based candidate retrieval and ranking evaluation.")
    parser.add_argument("--config", default="configs/pipeline.yaml", help="Path to pipeline YAML config")
    parser.add_argument("--dataset", choices=["mind", "ebnerd", "all"], default="mind", help="Dataset to evaluate")
    parser.add_argument("--split", choices=["train", "val", "test"], default="val", help="Split to evaluate")
    parser.add_argument("--model_name", default=None, help="HuggingFace model name override")
    parser.add_argument("--max_history_len", type=int, default=20, help="Max recent clicked articles for user query")
    parser.add_argument("--eval_mode", choices=["all", "global", "impression"], default="all", help="Evaluation mode (all, global, impression)")
    parser.add_argument("--batch_size", type=int, default=2000, help="Batch size for vector search")
    parser.add_argument("--device", default=None, help="Device to use for embedding inference ('cpu', 'cuda')")
    parser.add_argument("--force_recompute", action="store_true", help="Force recomputing embeddings")
    parser.add_argument("--n_bootstraps", type=int, default=1000, help="Number of bootstrap resamples for 95%% CI")

    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    processed_dir = Path(cfg["paths"]["processed_dir"])
    raw_dir = Path(cfg["paths"]["raw_dir"]) if "raw_dir" in cfg["paths"] else None

    datasets = ["mind", "ebnerd"] if args.dataset == "all" else [args.dataset]

    results_all = {}
    for ds in datasets:
        print(f"\n--- Starting Embedding Evaluation for {ds.upper()} ---")
        try:
            res = evaluate_embeddings(
                dataset_name=ds,
                split_name=args.split,
                processed_dir=processed_dir,
                raw_dir=raw_dir,
                model_name=args.model_name,
                max_history_len=args.max_history_len,
                eval_mode=args.eval_mode,
                batch_size=args.batch_size,
                device=args.device,
                force_recompute=args.force_recompute,
                n_bootstraps=args.n_bootstraps,
            )
            results_all[ds] = res
        except Exception as e:
            print(f"Error evaluating dataset {ds}: {e}")
            import traceback
            traceback.print_exc()

    return results_all


if __name__ == "__main__":
    main()
