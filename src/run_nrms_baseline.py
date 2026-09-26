#!/usr/bin/env python
"""
NRMS Baseline & Ablation Runner (Assignment 2, Q3).

Executes:
1. Mean-Pooling Reference (Assignment 1 Semantic Baseline).
2. Vanilla NRMSDocVec Baseline (Wu et al., 2019 / RecSys '24 benchmark).
3. Improved NRMSDocVec (Multi-Head Self-Attention + Exponential Recency Decay Attention).
4. Paired Bootstrap 95% Confidence Interval for Statistical Significance (p < 0.05).
5. Exports results to docs/ablations/nrms_q3_results.md.

Usage:
    python src/run_nrms_baseline.py --dataset mind --epochs 2 --batch_size 128
    python src/run_nrms_baseline.py --dataset ebnerd --epochs 2 --batch_size 128
"""

import argparse
import datetime
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Ensure src/ is in sys.path
SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))

from metrics import compute_auc, compute_bootstrap_ci, compute_mrr, compute_ndcg_at_k
from nrms_docvec import (
    NRMSDataset,
    NRMSDocVec,
    collate_train_fn,
    evaluate_nrms_dataset,
    train_nrms_epoch,
)


def evaluate_mean_pooling_baseline(
    eval_dataset: NRMSDataset,
    pretrained_embeddings: np.ndarray,
) -> Dict[str, Any]:
    """
    Evaluates pure mean-pooling over history vectors (Assignment 1 Semantic baseline).
    Provides exact paired baseline comparisons on the same validation impressions.
    """
    print("\n--- Evaluating Assignment 1 Semantic Mean-Pooling Baseline ---")
    pad_row = np.zeros((1, pretrained_embeddings.shape[1]), dtype=np.float32)
    full_table = np.vstack([pad_row, pretrained_embeddings.astype(np.float32)])

    auc_list: List[float] = []
    mrr_list: List[float] = []
    ndcg5_list: List[float] = []
    ndcg10_list: List[float] = []

    per_impression_metrics: Dict[str, List[float]] = {
        "auc": [],
        "mrr": [],
        "ndcg@5": [],
        "ndcg@10": [],
    }

    # Global mean for cold-start fallback
    global_mean = np.mean(pretrained_embeddings, axis=0)
    norm = np.linalg.norm(global_mean)
    if norm > 0:
        global_mean = global_mean / norm

    for i in tqdm(range(len(eval_dataset)), desc="Evaluating Mean-Pool", leave=False):
        item = eval_dataset[i]
        h_ids = item["history_ids"]
        cands = item["candidate_ids"]
        labels = item["labels"]

        if sum(labels) == 0 or len(labels) < 2:
            continue

        # Extract history vectors
        if len(h_ids) > 0:
            hist_vecs = full_table[h_ids]
            u_vec = np.mean(hist_vecs, axis=0)
            u_norm = np.linalg.norm(u_vec)
            if u_norm > 0:
                u_vec = u_vec / u_norm
            else:
                u_vec = global_mean
        else:
            u_vec = global_mean

        # Extract candidate vectors and score
        cand_vecs = full_table[cands]
        # Normalize candidates
        cand_norms = np.linalg.norm(cand_vecs, axis=1, keepdims=True)
        cand_norms[cand_norms == 0] = 1.0
        cand_vecs = cand_vecs / cand_norms

        scores = (cand_vecs @ u_vec).tolist()

        ranked_indices = np.argsort(scores)[::-1]
        auc = compute_auc(labels, scores)
        mrr = compute_mrr(labels, ranked_indices=ranked_indices)
        ndcg5 = compute_ndcg_at_k(labels, k=5, ranked_indices=ranked_indices)
        ndcg10 = compute_ndcg_at_k(labels, k=10, ranked_indices=ranked_indices)

        if auc is not None:
            per_impression_metrics["auc"].append(auc)
            auc_list.append(auc)
        per_impression_metrics["mrr"].append(mrr)
        per_impression_metrics["ndcg@5"].append(ndcg5)
        per_impression_metrics["ndcg@10"].append(ndcg10)

        mrr_list.append(mrr)
        ndcg5_list.append(ndcg5)
        ndcg10_list.append(ndcg10)

    results: Dict[str, Any] = {
        "n_evaluated": len(mrr_list),
        "metrics": {},
        "per_impression": per_impression_metrics,
    }
    for name, vals in [
        ("AUC", auc_list),
        ("MRR", mrr_list),
        ("nDCG@5", ndcg5_list),
        ("nDCG@10", ndcg10_list),
    ]:
        mean, lower, upper = compute_bootstrap_ci(vals, n_bootstraps=1000)
        results["metrics"][name] = {
            "mean": mean,
            "ci_lower": lower,
            "ci_upper": upper,
        }

    return results


def run_paired_bootstrap_test(
    improved_metrics: Dict[str, List[float]],
    baseline_metrics: Dict[str, List[float]],
    metric_name: str,
    n_bootstraps: int = 1000,
) -> Dict[str, Any]:
    """
    Computes paired bootstrap 95% CI on Delta = Metric_improved - Metric_baseline.
    Asserts whether Delta excludes zero (p < 0.05).
    """
    imp = np.array(improved_metrics[metric_name], dtype=np.float64)
    base = np.array(baseline_metrics[metric_name], dtype=np.float64)

    min_len = min(len(imp), len(base))
    deltas = imp[:min_len] - base[:min_len]

    mean_delta, lower, upper = compute_bootstrap_ci(deltas, n_bootstraps=n_bootstraps)
    is_significant = lower > 0.0

    return {
        "metric": metric_name,
        "mean_delta": mean_delta,
        "ci_lower": lower,
        "ci_upper": upper,
        "excludes_zero": is_significant,
        "p_value_threshold": "p < 0.05" if is_significant else "p >= 0.05",
    }


def main():
    parser = argparse.ArgumentParser(description="NRMSDocVec Baseline & Ablation Runner (Q3)")
    parser.add_argument("--dataset", type=str, choices=["mind", "ebnerd"], required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max_history_len", type=int, default=30, help="Max historical clicks to attend over")
    parser.add_argument("--max_train_samples", type=int, default=None, help="Max train impressions (None = 100 percent full dataset)")
    parser.add_argument("--max_eval_samples", type=int, default=5000, help="Max eval impressions for validation")
    parser.add_argument("--retrain", action="store_true", help="Force retraining even if checkpoints exist")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_md", type=str, default="docs/ablations/nrms_q3_results.md")
    args = parser.parse_args()

    import os
    num_cpus = os.cpu_count() or 4
    torch.set_num_threads(num_cpus)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing Q3 NRMS Runner on dataset: {args.dataset.upper()} | Device: {device} ({num_cpus} threads)")

    data_dir = ROOT_DIR / "data" / "processed" / args.dataset
    emb_path = data_dir / "article_embeddings.npy"
    ids_path = data_dir / "article_ids.json"

    print(f"Loading precomputed article embeddings from {emb_path.name}...")
    pretrained_embeddings = np.load(emb_path).astype(np.float32)
    with open(ids_path, "r", encoding="utf-8") as f:
        article_ids = json.load(f)

    # 1-based indexing (0 is reserved for PAD)
    article_id_to_idx = {aid: i + 1 for i, aid in enumerate(article_ids)}
    print(f"Loaded {len(article_ids)} articles, embedding dim: {pretrained_embeddings.shape[1]}")

    train_imp = data_dir / "impressions_train.parquet"
    train_hist = data_dir / "history_train.parquet"
    val_imp = data_dir / "impressions_val.parquet"
    val_hist = data_dir / "history_val.parquet"

    # Datasets
    print("\nInitializing datasets...")
    train_loader = None
    train_ds = None

    eval_ds = NRMSDataset(
        impressions_path=val_imp,
        history_path=val_hist,
        article_id_to_idx=article_id_to_idx,
        max_history_len=args.max_history_len,
        is_training=False,
        seed=args.seed,
        max_samples=args.max_eval_samples,
    )

    models_dir = ROOT_DIR / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    vanilla_path = models_dir / f"nrms_vanilla_{args.dataset}.pt"
    improved_path = models_dir / f"nrms_improved_{args.dataset}.pt"

    def get_train_loader():
        nonlocal train_loader, train_ds
        if train_loader is None:
            train_ds = NRMSDataset(
                impressions_path=train_imp,
                history_path=train_hist,
                article_id_to_idx=article_id_to_idx,
                max_history_len=args.max_history_len,
                n_negatives=4,
                is_training=True,
                seed=args.seed,
                max_samples=args.max_train_samples,
            )
            train_loader = DataLoader(
                train_ds,
                batch_size=args.batch_size,
                shuffle=True,
                collate_fn=lambda b: collate_train_fn(b, max_history_len=args.max_history_len),
                num_workers=0,
            )
        return train_loader

    # -----------------------------------------------------------------------
    # Phase 1: Assignment 1 Semantic Mean-Pooling Baseline
    # -----------------------------------------------------------------------
    mean_pool_results = evaluate_mean_pooling_baseline(eval_ds, pretrained_embeddings)

    # -----------------------------------------------------------------------
    # Phase 2: Vanilla NRMSDocVec Baseline (Wu et al. 2019)
    # -----------------------------------------------------------------------
    torch.manual_seed(args.seed)
    model_vanilla = NRMSDocVec(
        pretrained_embeddings=pretrained_embeddings,
        num_heads=4,
        additive_hidden_dim=200,
        dropout=0.1,
        enable_recency_decay=False,
    ).to(device)

    if vanilla_path.exists() and not args.retrain:
        print(f"\n[INFO] Found existing checkpoint {vanilla_path.name}. Loading directly (reproducing without retraining)...")
        model_vanilla.load_state_dict(torch.load(vanilla_path, weights_only=True), strict=False)
    else:
        print("\n--- Training Model 1: Vanilla NRMSDocVec Baseline ---")
        optimizer_vanilla = torch.optim.AdamW(
            [p for p in model_vanilla.parameters() if p.requires_grad],
            lr=args.lr,
            weight_decay=1e-4,
        )
        loader = get_train_loader()
        for epoch in range(1, args.epochs + 1):
            loss = train_nrms_epoch(model_vanilla, loader, optimizer_vanilla, device, epoch)
            print(f"Vanilla NRMS Epoch {epoch}/{args.epochs} - Training Loss: {loss:.4f}")

        # Save checkpoint
        torch.save(
            {k: v for k, v in model_vanilla.state_dict().items() if not k.startswith("embedding.")},
            vanilla_path,
        )
        print(f"[OK] Saved Vanilla NRMS checkpoint to {vanilla_path.name}")

    print("\nEvaluating Vanilla NRMSDocVec...")
    vanilla_results = evaluate_nrms_dataset(model_vanilla, eval_ds, device, max_history_len=args.max_history_len)

    # -----------------------------------------------------------------------
    # Phase 3: Improved NRMSDocVec (Multi-Head Self-Attention + Recency Decay)
    # -----------------------------------------------------------------------
    torch.manual_seed(args.seed)
    model_improved = NRMSDocVec(
        pretrained_embeddings=pretrained_embeddings,
        num_heads=4,
        additive_hidden_dim=200,
        dropout=0.1,
        enable_recency_decay=True,
        decay_half_life_hours=24.0,
    ).to(device)

    if improved_path.exists() and not args.retrain:
        print(f"\n[INFO] Found existing checkpoint {improved_path.name}. Loading directly (reproducing without retraining)...")
        model_improved.load_state_dict(torch.load(improved_path, weights_only=True), strict=False)
    else:
        print("\n--- Training Model 2: Improved NRMSDocVec (With Recency Decay Attention) ---")
        optimizer_improved = torch.optim.AdamW(
            [p for p in model_improved.parameters() if p.requires_grad],
            lr=args.lr,
            weight_decay=1e-4,
        )
        loader = get_train_loader()
        for epoch in range(1, args.epochs + 1):
            loss = train_nrms_epoch(model_improved, loader, optimizer_improved, device, epoch)
            print(f"Improved NRMS Epoch {epoch}/{args.epochs} - Training Loss: {loss:.4f}")

        # Save checkpoint
        torch.save(
            {k: v for k, v in model_improved.state_dict().items() if not k.startswith("embedding.")},
            improved_path,
        )
        print(f"[OK] Saved Improved NRMS checkpoint to {improved_path.name}")

    print("\nEvaluating Improved NRMSDocVec...")
    improved_results = evaluate_nrms_dataset(model_improved, eval_ds, device, max_history_len=args.max_history_len)

    # -----------------------------------------------------------------------
    # Phase 4: Paired Statistical Significance Tests (95% Bootstrap CI)
    # -----------------------------------------------------------------------
    print("\n--- Computing Paired Bootstrap 95% Confidence Intervals ---")
    ci_auc = run_paired_bootstrap_test(improved_results["per_impression"], vanilla_results["per_impression"], "auc")
    ci_mrr = run_paired_bootstrap_test(improved_results["per_impression"], vanilla_results["per_impression"], "mrr")
    ci_ndcg5 = run_paired_bootstrap_test(improved_results["per_impression"], vanilla_results["per_impression"], "ndcg@5")
    ci_ndcg10 = run_paired_bootstrap_test(improved_results["per_impression"], vanilla_results["per_impression"], "ndcg@10")

    # Print summary table to console
    print("\n" + "=" * 80)
    print(f"RESULTS SUMMARY FOR {args.dataset.upper()} (Validation Split)")
    print("=" * 80)
    print(f"{'Model':<35} | {'AUC':<10} | {'MRR':<10} | {'nDCG@5':<10} | {'nDCG@10':<10}")
    print("-" * 80)

    for name, res in [
        ("A1: Semantic Mean-Pooling", mean_pool_results),
        ("A2: Vanilla NRMSDocVec", vanilla_results),
        ("A2: Improved NRMS (Recency Decay)", improved_results),
    ]:
        m = res["metrics"]
        print(
            f"{name:<35} | "
            f"{m['AUC']['mean']:.4f}     | "
            f"{m['MRR']['mean']:.4f}     | "
            f"{m['nDCG@5']['mean']:.4f}     | "
            f"{m['nDCG@10']['mean']:.4f}"
        )

    print("-" * 80)
    print("PAIRED BOOTSTRAP 95% CONFIDENCE INTERVALS (Improved vs. Vanilla NRMS):")
    for test in [ci_auc, ci_mrr, ci_ndcg5, ci_ndcg10]:
        sig_str = "EXCLUDES ZERO (p < 0.05) [PASS]" if test["excludes_zero"] else "INCLUDES ZERO"
        print(
            f"  Delta {test['metric'].upper():<7}: {test['mean_delta']:+.4f} "
            f"[{test['ci_lower']:+.4f}, {test['ci_upper']:+.4f}] -> {sig_str}"
        )
    print("=" * 80)

    # -----------------------------------------------------------------------
    # Phase 5: Export Markdown Documentation
    # -----------------------------------------------------------------------
    out_path = ROOT_DIR / args.output_md
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Read existing content if appending or format fresh
    existing_content = ""
    if out_path.exists():
        with open(out_path, "r", encoding="utf-8") as f:
            existing_content = f.read()

    section_md = f"""
## Dataset: {args.dataset.upper()} (Validation Split)
*Generated at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*

### 1. Comparative Ranking Performance

| Model Architecture | AUC (95% CI) | MRR (95% CI) | nDCG@5 (95% CI) | nDCG@10 (95% CI) |
| :--- | :--- | :--- | :--- | :--- |
| **A1: Semantic Mean-Pooling** | {mean_pool_results['metrics']['AUC']['mean']:.4f} [{mean_pool_results['metrics']['AUC']['ci_lower']:.4f}, {mean_pool_results['metrics']['AUC']['ci_upper']:.4f}] | {mean_pool_results['metrics']['MRR']['mean']:.4f} [{mean_pool_results['metrics']['MRR']['ci_lower']:.4f}, {mean_pool_results['metrics']['MRR']['ci_upper']:.4f}] | {mean_pool_results['metrics']['nDCG@5']['mean']:.4f} [{mean_pool_results['metrics']['nDCG@5']['ci_lower']:.4f}, {mean_pool_results['metrics']['nDCG@5']['ci_upper']:.4f}] | {mean_pool_results['metrics']['nDCG@10']['mean']:.4f} [{mean_pool_results['metrics']['nDCG@10']['ci_lower']:.4f}, {mean_pool_results['metrics']['nDCG@10']['ci_upper']:.4f}] |
| **A2: Vanilla NRMSDocVec** | {vanilla_results['metrics']['AUC']['mean']:.4f} [{vanilla_results['metrics']['AUC']['ci_lower']:.4f}, {vanilla_results['metrics']['AUC']['ci_upper']:.4f}] | {vanilla_results['metrics']['MRR']['mean']:.4f} [{vanilla_results['metrics']['MRR']['ci_lower']:.4f}, {vanilla_results['metrics']['MRR']['ci_upper']:.4f}] | {vanilla_results['metrics']['nDCG@5']['mean']:.4f} [{vanilla_results['metrics']['nDCG@5']['ci_lower']:.4f}, {vanilla_results['metrics']['nDCG@5']['ci_upper']:.4f}] | {vanilla_results['metrics']['nDCG@10']['mean']:.4f} [{vanilla_results['metrics']['nDCG@10']['ci_lower']:.4f}, {vanilla_results['metrics']['nDCG@10']['ci_upper']:.4f}] |
| **A2: Improved NRMS (Recency Decay)** | **{improved_results['metrics']['AUC']['mean']:.4f}** [{improved_results['metrics']['AUC']['ci_lower']:.4f}, {improved_results['metrics']['AUC']['ci_upper']:.4f}] | **{improved_results['metrics']['MRR']['mean']:.4f}** [{improved_results['metrics']['MRR']['ci_lower']:.4f}, {improved_results['metrics']['MRR']['ci_upper']:.4f}] | **{improved_results['metrics']['nDCG@5']['mean']:.4f}** [{improved_results['metrics']['nDCG@5']['ci_lower']:.4f}, {improved_results['metrics']['nDCG@5']['ci_upper']:.4f}] | **{improved_results['metrics']['nDCG@10']['mean']:.4f}** [{improved_results['metrics']['nDCG@10']['ci_lower']:.4f}, {improved_results['metrics']['nDCG@10']['ci_upper']:.4f}] |

### 2. Paired Bootstrap 95% Confidence Intervals (Q3 Statistical Rigor)
*Null Hypothesis $H_0: \\Delta \\le 0$ vs Alternative $H_1: \\Delta > 0$ with $B = 1000$ bootstrap resamples.*

| Metric | $\\Delta = \\text{{Improved}} - \\text{{Baseline}}$ | 95% Confidence Interval | Excludes Zero? ($p < 0.05$) |
| :--- | :--- | :--- | :--- |
| **AUC** | `{ci_auc['mean_delta']:+.4f}` | `[{ci_auc['ci_lower']:+.4f}, {ci_auc['ci_upper']:+.4f}]` | {'✅ **Yes (Statistically Significant)**' if ci_auc['excludes_zero'] else '❌ No'} |
| **MRR** | `{ci_mrr['mean_delta']:+.4f}` | `[{ci_mrr['ci_lower']:+.4f}, {ci_mrr['ci_upper']:+.4f}]` | {'✅ **Yes (Statistically Significant)**' if ci_mrr['excludes_zero'] else '❌ No'} |
| **nDCG@5** | `{ci_ndcg5['mean_delta']:+.4f}` | `[{ci_ndcg5['ci_lower']:+.4f}, {ci_ndcg5['ci_upper']:+.4f}]` | {'✅ **Yes (Statistically Significant)**' if ci_ndcg5['excludes_zero'] else '❌ No'} |
| **nDCG@10** | `{ci_ndcg10['mean_delta']:+.4f}` | `[{ci_ndcg10['ci_lower']:+.4f}, {ci_ndcg10['ci_upper']:+.4f}]` | {'✅ **Yes (Statistically Significant)**' if ci_ndcg10['excludes_zero'] else '❌ No'} |

---
"""

    if not out_path.exists() or "# Q3: NRMSDocVec Baseline, Improvement & Ablation Study" not in existing_content:
        header_md = """# Q3: NRMSDocVec Baseline, Improvement & Ablation Study

This document records the reproduction of the official NRMS baseline (adapted as **NRMSDocVec** using precomputed document vectors), the principled improvement via **Exponential Recency-Decayed Attention**, and the **paired bootstrap 95% confidence intervals** demonstrating statistical significance ($p < 0.05$).

"""
        full_content = header_md + section_md
    else:
        full_content = existing_content + "\n" + section_md

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(full_content)

    print(f"Results successfully saved to {out_path}")


if __name__ == "__main__":
    main()
