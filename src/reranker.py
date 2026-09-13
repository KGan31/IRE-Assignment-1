"""
Two-Stage Retrieve-then-Rank Model using LightGBM (Assignment 2, Q2).

Trains a gradient-boosted decision tree ranker (LGBMRanker with LambdaMART)
over engineered behavioral, temporal, and semantic features, and evaluates
AUC, MRR, nDCG@5, and nDCG@10 before and after re-ranking.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import lightgbm as lgb
import numpy as np
import polars as pl

from metrics import compute_auc, compute_mrr, compute_ndcg_at_k

FEATURE_COLS = [
    "user_click_count",
    "user_recency_score",
    "user_mean_dwell_time",
    "article_freshness_hours",
    "article_popularity_24h",
    "category_affinity_score",
    "session_position",
    "bm25_score",
    "semantic_score",
    "first_stage_score",
]


def prepare_features_and_groups(
    df: pl.DataFrame,
    feature_cols: List[str] = FEATURE_COLS,
    max_impressions: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Extracts feature matrix X, target labels y, query groups, and impression/article metadata.
    """
    if max_impressions is not None:
        unique_imps = df["impression_id"].unique(maintain_order=True).head(max_impressions)
        df = df.filter(pl.col("impression_id").is_in(unique_imps))

    # Ensure grouped by impression_id and preserving order
    groups = df.group_by("impression_id", maintain_order=True).len()["len"].to_numpy()
    X = df.select(feature_cols).to_numpy().astype(np.float32)
    y = df["label"].to_numpy().astype(np.int32)
    imp_ids = df["impression_id"].to_list()
    art_ids = df["article_id"].to_list()

    return X, y, groups, imp_ids, art_ids


def evaluate_impression_predictions(
    df: pl.DataFrame,
    score_col: str,
    k_list: Tuple[int, ...] = (5, 10),
) -> Dict[str, float]:
    """
    Computes impression-level mean AUC, MRR, nDCG@5, and nDCG@10.
    """
    # Group by impression_id
    grouped = df.group_by("impression_id", maintain_order=True).agg([
        pl.col("label"),
        pl.col(score_col),
    ])

    auc_scores: List[float] = []
    mrr_scores: List[float] = []
    ndcg_scores: Dict[int, List[float]] = {k: [] for k in k_list}

    for row in grouped.iter_rows():
        labels = list(row[1])
        scores = list(row[2])

        # Skip impressions with only 0s or only 1s for AUC
        auc = compute_auc(labels, scores)
        if auc is not None:
            auc_scores.append(auc)

        # MRR and nDCG
        if any(lbl > 0 for lbl in labels):
            ranked_order = np.argsort(scores)[::-1]
            mrr = compute_mrr(labels, ranked_indices=ranked_order)
            mrr_scores.append(mrr)
            for k in k_list:
                ndcg = compute_ndcg_at_k(labels, k=k, ranked_indices=ranked_order)
                ndcg_scores[k].append(ndcg)


    return {
        "AUC": float(np.mean(auc_scores)) if auc_scores else 0.0,
        "MRR": float(np.mean(mrr_scores)) if mrr_scores else 0.0,
        "nDCG@5": float(np.mean(ndcg_scores[5])) if ndcg_scores.get(5) else 0.0,
        "nDCG@10": float(np.mean(ndcg_scores[10])) if ndcg_scores.get(10) else 0.0,
    }


def train_reranker(
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
    feature_cols: List[str] = FEATURE_COLS,
    max_train_impressions: Optional[int] = None,
    learning_rate: float = 0.05,
    num_leaves: int = 31,
    n_estimators: int = 300,
    early_stopping_rounds: int = 30,
    random_state: int = 42,
) -> Tuple[lgb.LGBMRanker, Dict[str, Any]]:
    """
    Trains LGBMRanker with LambdaRank on training features and validates on validation features.
    """
    print(f"Preparing training data (impressions limit: {max_train_impressions})...")
    X_train, y_train, train_groups, _, _ = prepare_features_and_groups(
        train_df, feature_cols=feature_cols, max_impressions=max_train_impressions
    )
    print(f"Train samples: {len(X_train):,} rows across {len(train_groups):,} impressions.")

    print("Preparing validation data...")
    X_val, y_val, val_groups, _, _ = prepare_features_and_groups(
        val_df, feature_cols=feature_cols
    )
    print(f"Val samples: {len(X_val):,} rows across {len(val_groups):,} impressions.")

    ranker = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        eval_at=[5, 10],
        boosting_type="gbdt",
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
    )

    print(f"\nTraining LGBMRanker ({n_estimators} max estimators, lr={learning_rate}, leaves={num_leaves})...")
    ranker.fit(
        X_train,
        y_train,
        group=train_groups,
        eval_set=[(X_val, y_val)],
        eval_group=[val_groups],
        callbacks=[
            lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=True),
            lgb.log_evaluation(period=25),
        ],
    )

    # Feature Importances
    importances = dict(zip(feature_cols, [float(v) for v in ranker.feature_importances_]))
    sorted_importances = sorted(importances.items(), key=lambda x: x[1], reverse=True)

    print("\nFeature Importances (split counts):")
    print("-" * 50)
    for feat, imp in sorted_importances:
        print(f"  {feat:<26s}: {imp:8.0f}")
    print("-" * 50)

    # Evaluate Before vs. After on Validation
    print("\nEvaluating Before (Stage 1) vs After (Stage 2 Re-Ranker)...")
    val_df_eval = val_df.clone()
    val_preds = ranker.predict(X_val)
    val_df_eval = val_df_eval.with_columns(pl.Series("reranked_score", val_preds))

    metrics_before = evaluate_impression_predictions(val_df_eval, score_col="first_stage_score")
    metrics_after = evaluate_impression_predictions(val_df_eval, score_col="reranked_score")

    eval_summary = {
        "before_reranking": metrics_before,
        "after_reranking": metrics_after,
        "delta": {m: metrics_after[m] - metrics_before[m] for m in metrics_before},
        "feature_importances": importances,
    }

    print("\n" + "=" * 65)
    print(f"{'Metric':<12s} | {'Stage 1 (Before)':<18s} | {'Stage 2 (LightGBM)':<18s} | {'Delta':<10s}")
    print("=" * 65)
    for m in ["AUC", "MRR", "nDCG@5", "nDCG@10"]:
        b = metrics_before.get(m, 0.0)
        a = metrics_after.get(m, 0.0)
        diff = a - b
        print(f"{m:<12s} | {b:18.4f} | {a:18.4f} | {diff:+10.4f}")
    print("=" * 65)

    return ranker, eval_summary


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate LightGBM Re-Ranker.")
    parser.add_argument("--dataset", type=str, default="mind", choices=["mind", "ebnerd"])
    parser.add_argument("--train_path", type=str, default=None)
    parser.add_argument("--val_path", type=str, default=None)
    parser.add_argument("--max_train_impressions", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=0.05)
    parser.add_argument("--num_leaves", type=int, default=31)
    parser.add_argument("--n_estimators", type=int, default=300)
    parser.add_argument("--output_model", type=str, default=None)
    args = parser.parse_args()

    train_path = Path(args.train_path) if args.train_path else Path(f"data/processed/{args.dataset}/features_train.parquet")
    val_path = Path(args.val_path) if args.val_path else Path(f"data/processed/{args.dataset}/features_val.parquet")

    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(f"Feature files not found: {train_path} or {val_path}. Run features.py first.")

    print(f"\n{'='*70}")
    print(f"  TRAINING LIGHTGBM RE-RANKER: {args.dataset.upper()}")
    print(f"{'='*70}")
    print(f"Loading features from:\n  Train: {train_path}\n  Val:   {val_path}")

    train_df = pl.read_parquet(train_path)
    val_df = pl.read_parquet(val_path)

    ranker, eval_summary = train_reranker(
        train_df=train_df,
        val_df=val_df,
        max_train_impressions=args.max_train_impressions,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        n_estimators=args.n_estimators,
    )

    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    out_model_path = Path(args.output_model) if args.output_model else models_dir / f"{args.dataset}_lgbm_ranker.pkl"
    joblib.dump(ranker, out_model_path)
    print(f"\nTrained model checkpoint saved to: {out_model_path}")

    metrics_out_path = models_dir / f"{args.dataset}_reranker_metrics.json"
    with open(metrics_out_path, "w", encoding="utf-8") as f:
        json.dump(eval_summary, f, indent=2)
    print(f"Evaluation metrics report saved to: {metrics_out_path}")


if __name__ == "__main__":
    main()
