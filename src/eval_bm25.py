"""
Offline Evaluation Runner for BM25 Candidate Generation using bm25s.

Evaluates BM25 candidate retrieval over article titles and abstracts:
1. Builds BM25 index over article text catalog (title + abstract).
2. Constructs search queries for each impression from recent clicked article titles (leakage-free).
3. Retrieves top-K candidate articles using BM25 batch scoring.
4. Reports Recall@K for K in {50, 100, 200}.

Usage:
    python src/eval_bm25.py --dataset mind --split val --max_history_len 20 --eval_mode global
    python src/eval_bm25.py --dataset ebnerd --split val --max_history_len 20 --eval_mode global
"""

from typing import Optional
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

# Ensure src directory is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from bm25 import BM25InvertedIndex


def calculate_recall_at_k(retrieved_ids: List[str], ground_truth_set: Set[str], k: int) -> float:
    """Calculate Recall@K: |Retrieved[:K] cap GroundTruth| / |GroundTruth|"""
    if not ground_truth_set:
        return 0.0
    top_k_retrieved = set(retrieved_ids[:k])
    hits = len(top_k_retrieved.intersection(ground_truth_set))
    return hits / len(ground_truth_set)


def get_popular_articles(history_df: pd.DataFrame, top_n: int = 200) -> List[str]:
    """Fallback candidate list of most frequent clicked articles in history."""
    if history_df.empty or "clicked_article_id" not in history_df.columns:
        return []
    top_articles = (
        history_df["clicked_article_id"].value_counts().head(top_n).index.tolist()
    )
    return top_articles


def evaluate_bm25(
    dataset_name: str,
    split_name: str,
    processed_dir: Path,
    max_history_len: int = 20,
    eval_mode: str = "global",
    k_list: List[int] = [50, 100, 200],
    batch_size: int = 2000,
) -> Dict[str, float]:
    """Run BM25 candidate retrieval evaluation on dataset split using fast batching."""
    ds_processed = processed_dir / dataset_name
    articles_path = ds_processed / "articles.parquet"
    impressions_path = ds_processed / f"impressions_{split_name}.parquet"
    history_path = ds_processed / f"history_{split_name}.parquet"

    if not articles_path.exists():
        raise FileNotFoundError(f"Articles file not found: {articles_path}")
    if not impressions_path.exists():
        raise FileNotFoundError(f"Impressions file not found: {impressions_path}")

    # 1. Load articles and build BM25 index
    print(f"[{dataset_name} | {split_name}] Loading articles from {articles_path}...")
    articles_df = pd.read_parquet(articles_path)
    title_map: Dict[str, str] = dict(zip(articles_df["article_id"], articles_df["title"].fillna("")))

    print(f"[{dataset_name} | {split_name}] Building BM25 index ({len(articles_df)} articles)...")
    bm25_index = BM25InvertedIndex()
    bm25_index.build_index(articles_df)

    # 2. Load history & impressions
    print(f"[{dataset_name} | {split_name}] Loading impressions from {impressions_path}...")
    impressions_df = pd.read_parquet(impressions_path)

    history_df = pd.DataFrame()
    if history_path.exists():
        history_df = pd.read_parquet(history_path)

    # Compute popular fallback articles
    popular_fallback = get_popular_articles(history_df, top_n=max(k_list))
    if not popular_fallback and not articles_df.empty:
        popular_fallback = articles_df["article_id"].head(max(k_list)).tolist()

    # Build user click history map: user_id -> sorted list of (click_time, article_id)
    user_history_map: Dict[str, List[Tuple[pd.Timestamp, str]]] = {}
    if not history_df.empty and "user_id" in history_df.columns:
        grouped = history_df.groupby("user_id")
        for user_id, group in grouped:
            clicks = list(zip(group["click_time"], group["clicked_article_id"]))
            clicks.sort(key=lambda x: x[0])
            user_history_map[str(user_id)] = clicks

    max_k = max(k_list)

    # 3. Construct queries for all impressions
    print(f"[{dataset_name} | {split_name}] Preparing queries for {len(impressions_df)} impressions...")
    queries: List[str] = []
    ground_truths: List[Set[str]] = []
    candidate_subsets: List[Optional[List[str]]] = []

    user_ids = impressions_df["user_id"].astype(str).tolist()
    timestamps = impressions_df["timestamp"].tolist()
    clicked_lists = impressions_df["clicked_article_ids"].tolist()
    candidate_lists = (
        impressions_df["candidate_article_ids"].tolist()
        if (eval_mode == "impression" and "candidate_article_ids" in impressions_df.columns)
        else [None] * len(impressions_df)
    )

    for user_id, ts, gt, cand in zip(user_ids, timestamps, clicked_lists, candidate_lists):
        if isinstance(gt, np.ndarray):
            gt = gt.tolist()
        gt_set = set(gt)

        if not gt_set:
            continue

        ground_truths.append(gt_set)

        # Get leakage-free prior clicks
        user_clicks = user_history_map.get(user_id, [])
        prior_clicks = [aid for c_time, aid in user_clicks if c_time < ts]
        recent_clicks = prior_clicks[-max_history_len:] if prior_clicks else []

        query_titles = [title_map.get(aid, "") for aid in recent_clicks if aid in title_map]
        q_text = " ".join([t for t in query_titles if t]).strip()
        queries.append(q_text)

        if eval_mode == "impression" and cand is not None:
            if isinstance(cand, np.ndarray):
                cand = cand.tolist()
            candidate_subsets.append(cand)
        else:
            candidate_subsets.append(None)

    # 4. Fast Batch BM25 Retrieval
    print(f"[{dataset_name} | {split_name}] Running fast batch BM25 search for {len(queries)} queries...")
    recall_scores: Dict[int, List[float]] = {k: [] for k in k_list}

    num_queries = len(queries)
    for start_idx in tqdm(range(0, num_queries, batch_size), desc="Batch Retrieval"):
        end_idx = min(start_idx + batch_size, num_queries)
        batch_q = queries[start_idx:end_idx]

        batch_retrieved_pairs = bm25_index.batch_search(batch_q, top_k=max_k)

        for i, pairs in enumerate(batch_retrieved_pairs):
            q_idx = start_idx + i
            gt_set = ground_truths[q_idx]
            cand_subset = candidate_subsets[q_idx]

            retrieved_ids = [aid for aid, score in pairs]

            if cand_subset is not None:
                cand_set = set(cand_subset)
                retrieved_ids = [aid for aid in retrieved_ids if aid in cand_set]

            # Fallback padding if retrieved_ids < max_k
            if len(retrieved_ids) < max_k:
                existing = set(retrieved_ids)
                for pop_id in popular_fallback:
                    if pop_id not in existing:
                        if cand_subset is None or pop_id in cand_subset:
                            retrieved_ids.append(pop_id)
                            existing.add(pop_id)
                        if len(retrieved_ids) >= max_k:
                            break

            for k in k_list:
                recall = calculate_recall_at_k(retrieved_ids, gt_set, k)
                recall_scores[k].append(recall)

    # 5. Report Results
    mean_results = {}
    print(f"\n=======================================================")
    print(f" BM25 Candidate Retrieval Results: {dataset_name.upper()} ({split_name})")
    print(f" Mode: {eval_mode} | Max History: {max_history_len}")
    print(f"=======================================================")
    for k in k_list:
        mean_recall = float(np.mean(recall_scores[k])) if recall_scores[k] else 0.0
        mean_results[f"recall@{k}"] = mean_recall
        print(f" Recall@{k:3d} : {mean_recall:.4f}")
    print(f"=======================================================\n")

    return mean_results


def main():
    parser = argparse.ArgumentParser(description="Run BM25 candidate retrieval evaluation using bm25s.")
    parser.add_argument("--config", default="configs/pipeline.yaml", help="Path to pipeline YAML config")
    parser.add_argument("--dataset", choices=["mind", "ebnerd", "all"], default="mind", help="Dataset to evaluate")
    parser.add_argument("--split", choices=["train", "val", "test"], default="val", help="Split to evaluate")
    parser.add_argument("--max_history_len", type=int, default=20, help="Max recent clicked articles for user query")
    parser.add_argument("--eval_mode", choices=["global", "impression"], default="global", help="Retrieval evaluation mode")

    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    processed_dir = Path(cfg["paths"]["processed_dir"])

    datasets = ["mind", "ebnerd"] if args.dataset == "all" else [args.dataset]

    results_all = {}
    for ds in datasets:
        print(f"\n--- Starting BM25 Evaluation for {ds.upper()} ---")
        try:
            res = evaluate_bm25(
                dataset_name=ds,
                split_name=args.split,
                processed_dir=processed_dir,
                max_history_len=args.max_history_len,
                eval_mode=args.eval_mode,
            )
            results_all[ds] = res
        except Exception as e:
            print(f"Error evaluating dataset {ds}: {e}")

    return results_all


if __name__ == "__main__":
    main()
