"""
Offline Evaluation Runner for BM25 Candidate Generation and Ranking using bm25s.

Evaluates BM25 retrieval and ranking:
1. Builds BM25 index over article text catalog (title + abstract).
2. Constructs search queries for each impression from recent clicked article titles (leakage-free).
3. Evaluates:
   - Candidate Retrieval (Global Catalog): Recall@50, Recall@100, Recall@200, ILD@10, Novelty@10, Coverage@10
   - Impression Ranking: AUC, MRR, nDCG@5, nDCG@10, ILD@10, Novelty@10
   - Slicing: Cold-Start vs. Warm users, Head vs. Tail articles
   - Statistical Rigor: Bootstrap 95% Confidence Intervals for all metrics

Usage:
    python src/eval_bm25.py --dataset mind --split val
    python src/eval_bm25.py --dataset ebnerd --split val
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

from bm25 import BM25InvertedIndex, create_article_text_map
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


def evaluate_bm25(
    dataset_name: str,
    split_name: str,
    processed_dir: Path,
    max_history_len: int = 20,
    eval_mode: str = "all",
    k_list: List[int] = [50, 100, 200],
    batch_size: int = 2000,
    n_bootstraps: int = 1000,
    k1: float = 1.5,
    b: float = 0.75,
    cold_start_percentile: Optional[float] = None,
    history_fields: str = "title_abstract",
    include_body: bool = False,
) -> Dict[str, Any]:
    """Run BM25 candidate retrieval and ranking evaluation.

    Args:
        k1: BM25 term-frequency saturation (default 1.5).
            Higher values slow down TF saturation, favouring documents with
            many repeated query terms.
        b: BM25 document-length normalisation (default 0.75).
            0 = no normalisation; 1 = full normalisation.
        cold_start_percentile: If given, cold-start users are defined as those
            whose per-impression history length falls at or below this percentile
            of the observed distribution (e.g. 5.0 = bottom 5%%).  When None,
            falls back to the fixed absolute threshold of 5 clicks.
        history_fields: Fields from user history articles to include in query
            ('title' or 'title_abstract' / 'title+abstract').
        include_body: If True, include body text in the BM25 document INDEX
            (title + abstract/subtitle + body).  Only meaningful for EB-NeRD
            which supplies a 'body' column; silently ignored on MIND.
    """
    ds_processed = processed_dir / dataset_name
    articles_path = ds_processed / "articles.parquet"
    impressions_path = ds_processed / f"impressions_{split_name}.parquet"
    history_path = ds_processed / f"history_{split_name}.parquet"
    embeddings_path = ds_processed / "article_embeddings.npy"
    if not embeddings_path.exists():
        embeddings_path = ds_processed / "embeddings.npy"

    if not articles_path.exists():
        raise FileNotFoundError(f"Articles file not found: {articles_path}")
    if not impressions_path.exists():
        raise FileNotFoundError(f"Impressions file not found: {impressions_path}")

    # 1. Load articles and build BM25 index
    print(f"[{dataset_name} | {split_name}] Loading articles from {articles_path}...")
    articles_df = pd.read_parquet(articles_path)
    article_text_map: Dict[str, str] = create_article_text_map(articles_df, fields=history_fields)

    print(f"[{dataset_name} | {split_name}] Building BM25 index ({len(articles_df)} articles, k1={k1}, b={b}, body={include_body})...")
    bm25_index = BM25InvertedIndex(k1=k1, b=b)
    bm25_index.build_index(articles_df, include_body=include_body)

    # 2. Load history & impressions
    print(f"[{dataset_name} | {split_name}] Loading impressions from {impressions_path}...")
    impressions_df = pd.read_parquet(impressions_path)

    history_df = pd.DataFrame()
    if history_path.exists():
        history_df = pd.read_parquet(history_path)

    # Load precomputed embeddings if available for ILD calculation
    embeddings = None
    article_id_to_idx = {aid: idx for idx, aid in enumerate(articles_df["article_id"].astype(str))}
    if embeddings_path.exists():
        try:
            embeddings = np.load(embeddings_path)
        except Exception:
            pass

    # NOTE: harness is initialised after the impression loop so that the
    # empirical percentile of history_lengths can be computed first.

    # Compute popular fallback articles and normalized popularity map
    popular_fallback = get_popular_articles(history_df, top_n=max(k_list))
    if not popular_fallback and not articles_df.empty:
        popular_fallback = articles_df["article_id"].head(max(k_list)).tolist()

    popularity_map: Dict[str, float] = {}
    if not history_df.empty and "clicked_article_id" in history_df.columns:
        counts = history_df["clicked_article_id"].value_counts()
        max_c = max(1, counts.max())
        popularity_map = {str(aid): float(c) / float(max_c) for aid, c in counts.items()}

    # Identify head articles (top 20% most clicked articles in history)
    head_articles: Set[str] = set()
    if not history_df.empty and "clicked_article_id" in history_df.columns:
        counts = history_df["clicked_article_id"].value_counts()
        n_head = max(1, int(len(counts) * 0.2))
        head_articles = set(counts.head(n_head).index)

    # Build user click history map: user_id -> sorted list of (click_time, article_id)
    user_history_map: Dict[str, List[Tuple[pd.Timestamp, str]]] = {}
    if not history_df.empty and "user_id" in history_df.columns:
        val_uids = set(impressions_df["user_id"].astype(str))
        hist_filtered = history_df[history_df["user_id"].astype(str).isin(val_uids)].sort_values(["user_id", "click_time"])
        grouped = hist_filtered.groupby("user_id")
        for user_id, group in grouped:
            clicks = list(zip(group["click_time"], group["clicked_article_id"]))
            user_history_map[str(user_id)] = clicks

    max_k = max(k_list)

    # 3. Construct queries for all impressions
    print(f"[{dataset_name} | {split_name}] Preparing queries (using history fields: {history_fields}) for {len(impressions_df)} impressions...")
    queries: List[str] = []
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

        query_texts = [article_text_map.get(aid, "") for aid in recent_clicks if aid in article_text_map]
        q_text = " ".join([t for t in query_texts if t]).strip()
        queries.append(q_text)

    # Compute cold-start threshold: percentile-based or fixed absolute.
    if cold_start_percentile is not None and history_lengths:
        cold_start_threshold = int(np.percentile(history_lengths, cold_start_percentile))
        print(
            f"[{dataset_name} | {split_name}] Cold-start threshold "
            f"(p{cold_start_percentile}): <= {cold_start_threshold} clicks "
            f"(covers {sum(h <= cold_start_threshold for h in history_lengths)} "
            f"/ {len(history_lengths)} impressions)"
        )
    else:
        cold_start_threshold = 5

    # Initialize evaluation harness (after loop so threshold is known).
    harness = OfflineEvaluationHarness.from_data(
        articles_df=articles_df,
        history_df=history_df,
        embeddings=embeddings,
        article_id_to_idx=article_id_to_idx,
        n_bootstraps=n_bootstraps,
        cold_start_threshold=cold_start_threshold,
    )

    num_queries = len(queries)
    all_results: Dict[str, Any] = {}

    # 4. Fast Batch BM25 Retrieval
    print(f"[{dataset_name} | {split_name}] Running fast batch BM25 search for {num_queries} queries...")
    all_retrieved_pairs: List[List[Tuple[str, float]]] = []

    for start_idx in tqdm(range(0, num_queries, batch_size), desc="BM25 Search"):
        end_idx = min(start_idx + batch_size, num_queries)
        batch_q = queries[start_idx:end_idx]
        batch_pairs = bm25_index.batch_search(batch_q, top_k=max_k, fallback_popular_ids=popular_fallback)
        all_retrieved_pairs.extend(batch_pairs)

    # 5. Impression Candidate Ranking (AUC, MRR, nDCG@5, nDCG@10)
    if eval_mode in ["all", "impression"] and any(impression_cands_list):
        print(f"[{dataset_name} | {split_name}] Computing Impression Candidate Ranking scores (AUC, MRR, nDCG@K)...")
        all_candidate_scores: List[List[float]] = []

        for i in range(num_queries):
            cands = impression_cands_list[i]
            retrieved_map = dict(all_retrieved_pairs[i])
            # Assign BM25 score if retrieved in top_k, and add popularity tie-breaker.
            # For cold-start users (0 clicks), scores strictly order candidates by popularity.
            cand_scores = [
                retrieved_map.get(aid, 0.0) + (1e-4 * popularity_map.get(str(aid), 0.0))
                for aid in cands
            ]
            all_candidate_scores.append(cand_scores)

        ranking_results = harness.evaluate_impression_ranking(
            candidate_article_lists=impression_cands_list,
            candidate_labels=impression_labels_list,
            candidate_scores=all_candidate_scores,
            user_history_lengths=history_lengths,
            is_head_flags=is_head_gt,
        )

        print_evaluation_summary(
            f"BM25 Impression Candidate Ranking: {dataset_name.upper()} ({split_name})",
            ranking_results,
        )
        all_results["ranking"] = ranking_results

    # 6. Global Candidate Retrieval (Recall@K, ILD, Novelty, Coverage)
    if eval_mode in ["all", "global"]:
        all_retrieved_ids: List[List[str]] = []
        for i, pairs in enumerate(all_retrieved_pairs):
            retrieved_ids = [aid for aid, score in pairs]

            # Fallback padding if retrieved_ids < max_k
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
            f"BM25 Candidate Retrieval (Global Search): {dataset_name.upper()} ({split_name})",
            retrieval_results,
        )
        all_results["retrieval"] = retrieval_results

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Run BM25 candidate retrieval and ranking evaluation using bm25s.")
    parser.add_argument("--config", default="configs/pipeline.yaml", help="Path to pipeline YAML config")
    parser.add_argument("--dataset", choices=["mind", "ebnerd", "all"], default="mind", help="Dataset to evaluate")
    parser.add_argument("--split", choices=["train", "val", "test"], default="val", help="Split to evaluate")
    parser.add_argument("--max_history_len", type=int, default=20, help="Max recent clicked articles for user query")
    parser.add_argument("--eval_mode", choices=["all", "global", "impression"], default="all", help="Evaluation mode (all, global, impression)")
    parser.add_argument("--n_bootstraps", type=int, default=1000, help="Number of bootstrap resamples for 95%% CI")
    parser.add_argument("--k1", type=float, default=1.5,
        help="BM25 k1 parameter (term-frequency saturation). Suggested sweep: 0.5, 1.0, 1.5, 2.0, 3.0")
    parser.add_argument("--b", type=float, default=0.75,
        help="BM25 b parameter (document-length normalisation). Suggested sweep: 0.0, 0.25, 0.5, 0.75, 1.0")
    parser.add_argument("--history_fields",
        choices=["title", "title_abstract", "title+abstract", "both"],
        default="title_abstract",
        help="Fields from history articles to include in query: 'title' or 'title_abstract' / 'title+abstract' (default: title_abstract)",
    )
    parser.add_argument("--cold_start_percentile", type=float, default=None,
        help=(
            "Define cold-start users as those at or below this percentile of the "
            "per-impression history-length distribution. "
            "Suggested values: 1.0, 2.0, 5.0, 10.0. "
            "If not set, uses a fixed absolute threshold of 5 clicks."
        ))
    parser.add_argument("--include_body", action="store_true",
        help="Include article body text in the BM25 INDEX (title+abstract+body). "
             "Only effective for EB-NeRD which has a 'body' column.")

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
                n_bootstraps=args.n_bootstraps,
                k1=args.k1,
                b=args.b,
                cold_start_percentile=args.cold_start_percentile,
                history_fields=args.history_fields,
                include_body=args.include_body,
            )
            results_all[ds] = res
        except Exception as e:
            print(f"Error evaluating dataset {ds}: {e}")
            import traceback
            traceback.print_exc()

    return results_all


if __name__ == "__main__":
    main()
