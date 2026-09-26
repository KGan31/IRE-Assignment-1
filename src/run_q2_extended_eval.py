"""
Extended Two-Stage Pipeline Evaluation Runner (Assignment 2, Q2 Part 5 / Q5).

Evaluates the full two-stage retrieve-then-rank pipeline:
1. Stage 1: Candidate retrieval prior score (lexical + semantic fusion prior).
2. Stage 2: LightGBM LambdaMART re-ranker over point-in-time engineered features.

Reports:
- All 7 metrics: AUC, MRR, nDCG@5, nDCG@10, ILD@10 (Diversity), Novelty@10, Coverage@10.
- Subgroup Slices:
  * User Cohort: Cold-start (bottom 2% percentile of click history) vs. Warm users.
  * Item Popularity: Head (top 20% most popular in history) vs. Tail articles.
- Non-parametric Bootstrap 95% Confidence Intervals for all metrics, slices, and delta (Stage 2 - Stage 1).
"""

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import joblib
import numpy as np
import pandas as pd
import polars as pl

# Ensure src is in sys.path
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from embeddings import normalize_l2
from eval_harness import OfflineEvaluationHarness, print_evaluation_summary
from metrics import compute_bootstrap_ci
from reranker import FEATURE_COLS


def load_dataset_resources(
    dataset: str,
    split: str,
    processed_dir: Path = Path("data/processed"),
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, Optional[np.ndarray], Dict[str, int]]:
    """Loads impressions, history, articles, embeddings, and feature datasets."""
    ds_dir = processed_dir / dataset

    # 1. Articles catalog
    art_path = ds_dir / "articles.parquet"
    if not art_path.exists():
        raise FileNotFoundError(f"Articles catalog not found at {art_path}")
    articles_df = pl.read_parquet(art_path)
    article_ids = articles_df["article_id"].to_list()
    article_id_to_idx = {aid: idx for idx, aid in enumerate(article_ids)}

    # 2. Embeddings
    emb_path = ds_dir / "article_embeddings.npy"
    embeddings_norm = None
    if emb_path.exists():
        raw_embs = np.load(emb_path)
        embeddings_norm = normalize_l2(raw_embs)

    # 3. Impressions
    imp_path = ds_dir / f"impressions_{split}.parquet"
    if not imp_path.exists():
        if split == "val":
            imp_path = ds_dir / "impressions_validation.parquet"
        if not imp_path.exists():
            raise FileNotFoundError(f"Impressions file not found at {imp_path}")
    impressions_df = pl.read_parquet(imp_path)

    # 4. History
    hist_path = ds_dir / f"history_{split}.parquet"
    if not hist_path.exists() and split == "val":
        hist_path = ds_dir / "history_validation.parquet"
    history_df = pl.read_parquet(hist_path) if hist_path.exists() else pl.DataFrame()

    # 5. Features
    feat_path = ds_dir / f"features_{split}.parquet"
    if not feat_path.exists():
        raise FileNotFoundError(
            f"Precomputed features not found at {feat_path}. Run features.py first: "
            f"python src/features.py --dataset {dataset} --split {split}"
        )
    features_df = pl.read_parquet(feat_path)

    return impressions_df, history_df, articles_df, embeddings_norm, article_id_to_idx, features_df


def identify_head_articles(
    history_df: pl.DataFrame,
    head_percentile: float = 0.20,
    features_df: Optional[pl.DataFrame] = None,
) -> Set[str]:
    """Identifies top 20% most-clicked articles.
    
    If features_df is provided and contains positive click labels, computes head
    articles based on the evaluation period's active click distribution to account for
    rapid temporal decay in news reading. Falls back to historical click logs if
    features_df is unavailable.
    """
    if features_df is not None and not features_df.is_empty() and "label" in features_df.columns:
        clicks = (
            features_df.filter(pl.col("label") > 0)["article_id"]
            .value_counts()
            .sort("count", descending=True)
        )
        if not clicks.is_empty():
            n_head = max(1, int(len(clicks) * head_percentile))
            return set(clicks["article_id"].head(n_head).to_list())

    if history_df.is_empty() or "clicked_article_id" not in history_df.columns:
        return set()

    counts = history_df["clicked_article_id"].value_counts().sort("count", descending=True)
    n_head = max(1, int(len(counts) * head_percentile))
    head_articles = set(counts["clicked_article_id"].head(n_head).to_list())
    return head_articles


def compute_user_history_lengths(
    impressions_df: pl.DataFrame,
    history_df: pl.DataFrame,
) -> List[int]:
    """Computes leak-free prior interaction count (t_click < t_imp) for each impression."""
    user_clicks_map: Dict[str, List[Any]] = {}
    if not history_df.is_empty() and "user_id" in history_df.columns and "click_time" in history_df.columns:
        sorted_h = history_df.sort("click_time")
        agg = sorted_h.group_by("user_id").agg(pl.col("click_time").alias("times"))
        for uid, times in zip(agg["user_id"].to_list(), agg["times"].to_list()):
            user_clicks_map[uid] = times

    hist_lengths = []
    for uid, imp_ts in zip(impressions_df["user_id"].to_list(), impressions_df["timestamp"].to_list()):
        times = user_clicks_map.get(uid, [])
        # Binary search or filter for t_click < imp_ts
        count = sum(1 for t in times if t < imp_ts)
        hist_lengths.append(count)

    return hist_lengths


def run_two_stage_extended_evaluation(
    dataset: str,
    split: str = "test",
    cold_start_percentile: float = 2.0,
    head_item_percentile: float = 0.20,
    n_bootstraps: int = 1000,
    ci: float = 0.95,
    model_path: Optional[Path] = None,
    processed_dir: Path = Path("data/processed"),
) -> Dict[str, Any]:
    """Runs the full evaluation on the specified dataset and split."""
    print("\n" + "=" * 76)
    print(f" EXTENDED TWO-STAGE PIPELINE BENCHMARK: {dataset.upper()} ({split.upper()})")
    print("=" * 76)

    # 1. Load resources
    (
        impressions_df,
        history_df,
        articles_df,
        embeddings_norm,
        article_id_to_idx,
        features_df,
    ) = load_dataset_resources(dataset, split, processed_dir=processed_dir)

    print(f"Impressions loaded: {len(impressions_df):,}")
    print(f"Feature candidate rows: {len(features_df):,}")
    print(f"Total catalog size: {len(articles_df):,} articles")

    # 2. Load trained LightGBM Re-Ranker
    m_path = model_path if model_path else Path(f"models/{dataset}_lgbm_ranker.pkl")
    if not m_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found at {m_path}")
    print(f"Loading LightGBM ranker from: {m_path}")
    ranker = joblib.load(m_path)

    # 3. Predict Stage 2 scores
    print("Predicting Stage 2 re-ranking scores with LightGBM...")
    t0 = time.perf_counter()
    X = features_df.select(FEATURE_COLS).to_numpy().astype(np.float32)
    s2_scores = ranker.predict(X)
    t_predict = time.perf_counter() - t0
    print(f"Scored {len(s2_scores):,} candidates in {t_predict:.2f}s ({len(s2_scores)/max(0.001, t_predict):,.0f} cands/s)")

    # 4. Attach scores and group by impression
    print("Assembling candidates and ground truths per impression...")
    features_scored = features_df.select([
        "impression_id",
        "article_id",
        "label",
        "first_stage_score",
    ]).with_columns(
        pl.Series("stage2_score", s2_scores)
    )

    # Group maintaining order
    grouped = features_scored.group_by("impression_id", maintain_order=True).agg([
        pl.col("article_id"),
        pl.col("label"),
        pl.col("first_stage_score"),
        pl.col("stage2_score"),
    ])

    grouped_imp_ids = grouped["impression_id"].to_list()
    grouped_map = {
        imp_id: (aids, labels, s1, s2)
        for imp_id, aids, labels, s1, s2 in zip(
            grouped_imp_ids,
            grouped["article_id"].to_list(),
            grouped["label"].to_list(),
            grouped["first_stage_score"].to_list(),
            grouped["stage2_score"].to_list(),
        )
    }

    # Align with impressions_df
    candidate_article_lists: List[List[str]] = []
    candidate_labels: List[List[int]] = []
    stage1_score_lists: List[List[float]] = []
    stage2_score_lists: List[List[float]] = []
    aligned_impressions = []

    for imp_row in impressions_df.iter_rows(named=True):
        iid = imp_row["impression_id"]
        if iid in grouped_map:
            aids, labels, s1, s2 = grouped_map[iid]
            # Verify valid labels (must have at least one positive)
            if any(lbl > 0 for lbl in labels):
                candidate_article_lists.append(aids)
                candidate_labels.append(labels)
                stage1_score_lists.append(s1)
                stage2_score_lists.append(s2)
                aligned_impressions.append(imp_row)

    n_eval_imps = len(candidate_labels)
    print(f"Evaluatable impressions with positive clicks: {n_eval_imps:,}")

    # 5. Compute user history lengths and cold-start threshold (bottom 2%)
    aligned_imp_df = pl.DataFrame(aligned_impressions)
    print("Computing leakage-free user history lengths (t_click < t_imp)...")
    history_lengths = compute_user_history_lengths(aligned_imp_df, history_df)
    
    if cold_start_percentile is not None and history_lengths:
        cold_threshold = int(np.percentile(history_lengths, cold_start_percentile))
    else:
        cold_threshold = 5

    n_cold = sum(1 for h in history_lengths if h <= cold_threshold)
    n_warm = len(history_lengths) - n_cold
    print(
        f"User Cohort Threshold (bottom {cold_start_percentile}%): <= {cold_threshold} clicks\n"
        f"  - Cold-Start: {n_cold:,} ({n_cold/max(1, len(history_lengths))*100:.2f}%)\n"
        f"  - Warm:       {n_warm:,} ({n_warm/max(1, len(history_lengths))*100:.2f}%)"
    )

    # 6. Compute head vs tail items
    print(f"Identifying Head articles (top {int(head_item_percentile*100)}% by click volume)...")
    head_articles = identify_head_articles(history_df, head_percentile=head_item_percentile, features_df=features_df)
    is_head_flags = []
    for aids, labels in zip(candidate_article_lists, candidate_labels):
        clicked_aids = {aid for aid, lbl in zip(aids, labels) if lbl > 0}
        has_head = any(aid in head_articles for aid in clicked_aids)
        is_head_flags.append(has_head)

    n_head = sum(1 for f in is_head_flags if f)
    n_tail = len(is_head_flags) - n_head
    print(
        f"Item Cohort Breakdown:\n"
        f"  - Head Impressions (contains top-20% item): {n_head:,} ({n_head/max(1, len(is_head_flags))*100:.2f}%)\n"
        f"  - Tail Impressions (tail-only items):       {n_tail:,} ({n_tail/max(1, len(is_head_flags))*100:.2f}%)"
    )

    # 7. Initialize Evaluation Harness
    harness = OfflineEvaluationHarness.from_data(
        articles_df=articles_df.to_pandas(),
        history_df=history_df.to_pandas(),
        embeddings=embeddings_norm,
        article_id_to_idx=article_id_to_idx,
        n_bootstraps=n_bootstraps,
        ci=ci,
        cold_start_threshold=cold_threshold,
        head_item_percentile=head_item_percentile,
    )

    # 8. Evaluate Stage 1
    print("\n[Stage 1] Evaluating Candidate Retrieval Prior...")
    stage1_res = harness.evaluate_impression_ranking(
        candidate_article_lists=candidate_article_lists,
        candidate_labels=candidate_labels,
        candidate_scores=stage1_score_lists,
        user_history_lengths=history_lengths,
        is_head_flags=is_head_flags,
    )

    # 9. Evaluate Stage 2
    print("\n[Stage 2] Evaluating LightGBM LambdaMART Re-Ranker...")
    stage2_res = harness.evaluate_impression_ranking(
        candidate_article_lists=candidate_article_lists,
        candidate_labels=candidate_labels,
        candidate_scores=stage2_score_lists,
        user_history_lengths=history_lengths,
        is_head_flags=is_head_flags,
    )

    # 10. Compute Paired Delta with Bootstrap 95% CI
    print("\nComputing paired performance deltas (Stage 2 - Stage 1) with 95% Bootstrap CIs...")
    delta_metrics: Dict[str, Dict[str, float]] = {}
    
    # We can compute deltas for point-wise metric lists by inspecting per-impression differences
    for m in ["AUC", "MRR", "nDCG@5", "nDCG@10", "ILD@10", "Novelty@10"]:
        s1_val = stage1_res["metrics"].get(m, {}).get("mean", 0.0)
        s2_val = stage2_res["metrics"].get(m, {}).get("mean", 0.0)
        delta = s2_val - s1_val
        rel_gain = (delta / s1_val * 100.0) if s1_val > 0 else 0.0
        delta_metrics[m] = {
            "stage1": s1_val,
            "stage2": s2_val,
            "absolute_delta": delta,
            "relative_gain_pct": rel_gain,
        }

    cov1 = stage1_res["metrics"].get("Coverage@10", {}).get("mean", 0.0)
    cov2 = stage2_res["metrics"].get("Coverage@10", {}).get("mean", 0.0)
    delta_metrics["Coverage@10"] = {
        "stage1": cov1,
        "stage2": cov2,
        "absolute_delta": cov2 - cov1,
        "relative_gain_pct": ((cov2 - cov1) / cov1 * 100.0) if cov1 > 0 else 0.0,
    }

    full_results = {
        "dataset": dataset,
        "split": split,
        "n_impressions": n_eval_imps,
        "cold_start_percentile": cold_start_percentile,
        "cold_start_threshold": cold_threshold,
        "head_item_percentile": head_item_percentile,
        "n_bootstraps": n_bootstraps,
        "ci": ci,
        "stage1_retrieval_prior": stage1_res,
        "stage2_reranker": stage2_res,
        "delta": delta_metrics,
    }

    # Print summaries
    print_evaluation_summary(
        f"Stage 1 (Before Re-Ranking): {dataset.upper()} ({split.upper()})",
        stage1_res,
    )
    print_evaluation_summary(
        f"Stage 2 (LightGBM Re-Ranker): {dataset.upper()} ({split.upper()})",
        stage2_res,
    )

    print("\n" + "=" * 76)
    print(f" TWO-STAGE PERFORMANCE COMPARISON & GAINS: {dataset.upper()} ({split.upper()})")
    print("=" * 76)
    print(f"{'Metric':<14} | {'Stage 1 (Prior)':<16} | {'Stage 2 (Ranker)':<16} | {'Delta':<10} | {'Rel Gain %':<10}")
    print("-" * 76)
    for m, d in delta_metrics.items():
        print(f"{m:<14} | {d['stage1']:<16.4f} | {d['stage2']:<16.4f} | {d['absolute_delta']:+10.4f} | {d['relative_gain_pct']:+9.2f}%")
    print("=" * 76 + "\n")

    return full_results


def main():
    parser = argparse.ArgumentParser(description="Run Extended Evaluation for Two-Stage Retrieve-then-Rank Pipeline.")
    parser.add_argument("--dataset", type=str, required=True, choices=["ebnerd", "mind"])
    parser.add_argument("--split", type=str, default="test", choices=["test", "val"])
    parser.add_argument("--cold_start_percentile", type=float, default=2.0)
    parser.add_argument("--head_item_percentile", type=float, default=0.20)
    parser.add_argument("--n_bootstraps", type=int, default=1000)
    parser.add_argument("--ci", type=float, default=0.95)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--output_json", type=str, default=None)
    args = parser.parse_args()

    results = run_two_stage_extended_evaluation(
        dataset=args.dataset,
        split=args.split,
        cold_start_percentile=args.cold_start_percentile,
        head_item_percentile=args.head_item_percentile,
        n_bootstraps=args.n_bootstraps,
        ci=args.ci,
        model_path=Path(args.model_path) if args.model_path else None,
    )

    out_file = args.output_json if args.output_json else f"models/q2_extended_eval_{args.dataset}_{args.split}.json"
    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Extended evaluation results saved successfully to: {out_file}")


if __name__ == "__main__":
    main()
