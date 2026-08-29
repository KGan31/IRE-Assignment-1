"""
Unified Offline Evaluation Harness for News Recommendation and Candidate Generation.

Features:
1. Core Ranking & Accuracy: AUC, MRR, nDCG@5, nDCG@10, Recall@50, Recall@100, Recall@200.
2. Beyond-Accuracy: Intra-List Diversity (ILD@10), Novelty@10, Catalog Coverage@10.
3. Subgroup Slicing:
   - User cohorts: Cold-start users (<= 5 clicks) vs. Warm users (> 5 clicks).
   - Item popularity: Head articles (top 20% most clicked) vs. Tail articles (bottom 80%).
4. Statistical Rigor: Bootstrap 95% Confidence Interval for each metric.
5. Formatted tabular output and JSON export.
"""

from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
from pathlib import Path
import numpy as np
import pandas as pd

from metrics import (
    compute_auc,
    compute_bootstrap_ci,
    compute_catalog_coverage,
    compute_intra_list_diversity,
    compute_mrr,
    compute_ndcg_at_k,
    compute_novelty,
    compute_recall_at_k,
)


class OfflineEvaluationHarness:
    """
    Offline Evaluation Harness for evaluating news recommendation and retrieval models.
    """

    def __init__(
        self,
        embeddings: Optional[np.ndarray] = None,
        article_id_to_idx: Optional[Dict[str, int]] = None,
        item_popularity_prob: Optional[Dict[str, float]] = None,
        total_catalog_size: Optional[int] = None,
        n_bootstraps: int = 1000,
        ci: float = 0.95,
        cold_start_threshold: int = 5,
        head_item_percentile: float = 0.20,
    ):
        self.embeddings = embeddings
        self.article_id_to_idx = article_id_to_idx or {}
        self.item_popularity_prob = item_popularity_prob or {}
        self.total_catalog_size = total_catalog_size or (len(article_id_to_idx) if article_id_to_idx else 0)
        self.n_bootstraps = n_bootstraps
        self.ci = ci
        self.cold_start_threshold = cold_start_threshold
        self.head_item_percentile = head_item_percentile

    @classmethod
    def from_data(
        cls,
        articles_df: pd.DataFrame,
        history_df: Optional[pd.DataFrame] = None,
        embeddings: Optional[np.ndarray] = None,
        article_id_to_idx: Optional[Dict[str, int]] = None,
        n_bootstraps: int = 1000,
        ci: float = 0.95,
        cold_start_threshold: int = 5,
        head_item_percentile: float = 0.20,
    ) -> "OfflineEvaluationHarness":
        """Factory method to initialize the harness from raw DataFrames."""
        total_catalog_size = len(articles_df)
        if article_id_to_idx is None:
            article_id_to_idx = {aid: idx for idx, aid in enumerate(articles_df["article_id"].astype(str))}

        item_popularity_prob: Dict[str, float] = {}
        if history_df is not None and not history_df.empty and "clicked_article_id" in history_df.columns:
            counts = history_df["clicked_article_id"].astype(str).value_counts()
            total_clicks = counts.sum()
            if total_clicks > 0:
                item_popularity_prob = (counts / total_clicks).to_dict()

        return cls(
            embeddings=embeddings,
            article_id_to_idx=article_id_to_idx,
            item_popularity_prob=item_popularity_prob,
            total_catalog_size=total_catalog_size,
            n_bootstraps=n_bootstraps,
            ci=ci,
            cold_start_threshold=cold_start_threshold,
            head_item_percentile=head_item_percentile,
        )

    def evaluate_retrieval(
        self,
        retrieved_lists: List[List[str]],
        ground_truth_sets: List[Set[str]],
        user_history_lengths: Optional[List[int]] = None,
        is_head_flags: Optional[List[bool]] = None,
        k_list: Tuple[int, ...] = (50, 100, 200),
        beyond_k: int = 10,
    ) -> Dict[str, Any]:
        """
        Evaluate candidate retrieval over test impressions.
        
        Computes:
        - Recall@K for each K in k_list with 95% Bootstrap CI.
        - Intra-List Diversity (ILD@beyond_k), Novelty@beyond_k, Catalog Coverage@beyond_k.
        - Slices: Cold vs. Warm users, Head vs. Tail ground truth items.
        """
        n_impressions = len(retrieved_lists)
        if n_impressions == 0:
            return {}

        results: Dict[str, Any] = {
            "n_impressions": n_impressions,
            "metrics": {},
            "slices": {
                "user_cohort": {},
                "item_popularity": {},
            },
        }

        # 1. Candidate Retrieval Recalls
        recall_scores: Dict[int, List[float]] = {k: [] for k in k_list}
        cold_recalls: Dict[int, List[float]] = {k: [] for k in k_list}
        warm_recalls: Dict[int, List[float]] = {k: [] for k in k_list}
        head_recalls: Dict[int, List[float]] = {k: [] for k in k_list}
        tail_recalls: Dict[int, List[float]] = {k: [] for k in k_list}

        # Beyond-accuracy lists
        ild_scores: List[float] = []
        novelty_scores: List[float] = []

        cold_ild: List[float] = []
        warm_ild: List[float] = []
        head_ild: List[float] = []
        tail_ild: List[float] = []

        cold_novelty: List[float] = []
        warm_novelty: List[float] = []
        head_novelty: List[float] = []
        tail_novelty: List[float] = []

        cold_recs: List[List[str]] = []
        warm_recs: List[List[str]] = []
        head_recs: List[List[str]] = []
        tail_recs: List[List[str]] = []

        for i in range(n_impressions):
            retrieved = retrieved_lists[i]
            gt = ground_truth_sets[i]

            is_cold = False
            if user_history_lengths is not None:
                is_cold = (user_history_lengths[i] <= self.cold_start_threshold)
                if is_cold:
                    cold_recs.append(retrieved)
                else:
                    warm_recs.append(retrieved)

            is_head = False
            if is_head_flags is not None:
                is_head = bool(is_head_flags[i])
                if is_head:
                    head_recs.append(retrieved)
                else:
                    tail_recs.append(retrieved)

            # Recalls
            for k in k_list:
                rec = compute_recall_at_k(retrieved, gt, k)
                recall_scores[k].append(rec)

                if user_history_lengths is not None:
                    if is_cold:
                        cold_recalls[k].append(rec)
                    else:
                        warm_recalls[k].append(rec)

                if is_head_flags is not None:
                    if is_head:
                        head_recalls[k].append(rec)
                    else:
                        tail_recalls[k].append(rec)

            # Beyond accuracy
            if self.embeddings is not None and self.article_id_to_idx:
                ild = compute_intra_list_diversity(
                    retrieved, self.embeddings, self.article_id_to_idx, k=beyond_k
                )
                ild_scores.append(ild)
                if user_history_lengths is not None:
                    (cold_ild if is_cold else warm_ild).append(ild)
                if is_head_flags is not None:
                    (head_ild if is_head else tail_ild).append(ild)

            if self.item_popularity_prob:
                nov = compute_novelty(
                    retrieved, self.item_popularity_prob, k=beyond_k
                )
                novelty_scores.append(nov)
                if user_history_lengths is not None:
                    (cold_novelty if is_cold else warm_novelty).append(nov)
                if is_head_flags is not None:
                    (head_novelty if is_head else tail_novelty).append(nov)

        # Compute Point Estimates & Bootstrap 95% CIs
        for k in k_list:
            mean, lower, upper = compute_bootstrap_ci(
                recall_scores[k], n_bootstraps=self.n_bootstraps, ci=self.ci
            )
            results["metrics"][f"Recall@{k}"] = {
                "mean": mean,
                "ci_lower": lower,
                "ci_upper": upper,
            }

        # Beyond Accuracy point estimates & CIs
        if ild_scores:
            mean, lower, upper = compute_bootstrap_ci(
                ild_scores, n_bootstraps=self.n_bootstraps, ci=self.ci
            )
            results["metrics"][f"ILD@{beyond_k}"] = {
                "mean": mean,
                "ci_lower": lower,
                "ci_upper": upper,
            }

        if novelty_scores:
            mean, lower, upper = compute_bootstrap_ci(
                novelty_scores, n_bootstraps=self.n_bootstraps, ci=self.ci
            )
            results["metrics"][f"Novelty@{beyond_k}"] = {
                "mean": mean,
                "ci_lower": lower,
                "ci_upper": upper,
            }

        cov = compute_catalog_coverage(
            retrieved_lists, self.total_catalog_size, k=beyond_k
        )
        results["metrics"][f"Coverage@{beyond_k}"] = {
            "mean": cov,
            "ci_lower": cov,
            "ci_upper": cov,
        }

        # Slices
        if user_history_lengths is not None:
            results["slices"]["user_cohort"]["cold_threshold"] = self.cold_start_threshold
            results["slices"]["user_cohort"]["cold_count"] = len(cold_recalls[k_list[0]])
            results["slices"]["user_cohort"]["warm_count"] = len(warm_recalls[k_list[0]])
            for k in k_list:
                c_mean, c_low, c_up = compute_bootstrap_ci(cold_recalls[k], n_bootstraps=self.n_bootstraps, ci=self.ci)
                w_mean, w_low, w_up = compute_bootstrap_ci(warm_recalls[k], n_bootstraps=self.n_bootstraps, ci=self.ci)
                results["slices"]["user_cohort"][f"Cold_Recall@{k}"] = {"mean": c_mean, "ci_lower": c_low, "ci_upper": c_up}
                results["slices"]["user_cohort"][f"Warm_Recall@{k}"] = {"mean": w_mean, "ci_lower": w_low, "ci_upper": w_up}
            if cold_ild:
                c_mean, c_low, c_up = compute_bootstrap_ci(cold_ild, n_bootstraps=self.n_bootstraps, ci=self.ci)
                w_mean, w_low, w_up = compute_bootstrap_ci(warm_ild, n_bootstraps=self.n_bootstraps, ci=self.ci)
                results["slices"]["user_cohort"][f"Cold_ILD@{beyond_k}"] = {"mean": c_mean, "ci_lower": c_low, "ci_upper": c_up}
                results["slices"]["user_cohort"][f"Warm_ILD@{beyond_k}"] = {"mean": w_mean, "ci_lower": w_low, "ci_upper": w_up}
            if cold_novelty:
                c_mean, c_low, c_up = compute_bootstrap_ci(cold_novelty, n_bootstraps=self.n_bootstraps, ci=self.ci)
                w_mean, w_low, w_up = compute_bootstrap_ci(warm_novelty, n_bootstraps=self.n_bootstraps, ci=self.ci)
                results["slices"]["user_cohort"][f"Cold_Novelty@{beyond_k}"] = {"mean": c_mean, "ci_lower": c_low, "ci_upper": c_up}
                results["slices"]["user_cohort"][f"Warm_Novelty@{beyond_k}"] = {"mean": w_mean, "ci_lower": w_low, "ci_upper": w_up}
            if cold_recs:
                cov_cold = compute_catalog_coverage(cold_recs, self.total_catalog_size, k=beyond_k)
                cov_warm = compute_catalog_coverage(warm_recs, self.total_catalog_size, k=beyond_k)
                results["slices"]["user_cohort"][f"Cold_Coverage@{beyond_k}"] = {"mean": cov_cold, "ci_lower": cov_cold, "ci_upper": cov_cold}
                results["slices"]["user_cohort"][f"Warm_Coverage@{beyond_k}"] = {"mean": cov_warm, "ci_lower": cov_warm, "ci_upper": cov_warm}

        if is_head_flags is not None:
            results["slices"]["item_popularity"]["head_count"] = len(head_recalls[k_list[0]])
            results["slices"]["item_popularity"]["tail_count"] = len(tail_recalls[k_list[0]])
            for k in k_list:
                h_mean, h_low, h_up = compute_bootstrap_ci(head_recalls[k], n_bootstraps=self.n_bootstraps, ci=self.ci)
                t_mean, t_low, t_up = compute_bootstrap_ci(tail_recalls[k], n_bootstraps=self.n_bootstraps, ci=self.ci)
                results["slices"]["item_popularity"][f"Head_Recall@{k}"] = {"mean": h_mean, "ci_lower": h_low, "ci_upper": h_up}
                results["slices"]["item_popularity"][f"Tail_Recall@{k}"] = {"mean": t_mean, "ci_lower": t_low, "ci_upper": t_up}
            if head_ild:
                h_mean, h_low, h_up = compute_bootstrap_ci(head_ild, n_bootstraps=self.n_bootstraps, ci=self.ci)
                t_mean, t_low, t_up = compute_bootstrap_ci(tail_ild, n_bootstraps=self.n_bootstraps, ci=self.ci)
                results["slices"]["item_popularity"][f"Head_ILD@{beyond_k}"] = {"mean": h_mean, "ci_lower": h_low, "ci_upper": h_up}
                results["slices"]["item_popularity"][f"Tail_ILD@{beyond_k}"] = {"mean": t_mean, "ci_lower": t_low, "ci_upper": t_up}
            if head_novelty:
                h_mean, h_low, h_up = compute_bootstrap_ci(head_novelty, n_bootstraps=self.n_bootstraps, ci=self.ci)
                t_mean, t_low, t_up = compute_bootstrap_ci(tail_novelty, n_bootstraps=self.n_bootstraps, ci=self.ci)
                results["slices"]["item_popularity"][f"Head_Novelty@{beyond_k}"] = {"mean": h_mean, "ci_lower": h_low, "ci_upper": h_up}
                results["slices"]["item_popularity"][f"Tail_Novelty@{beyond_k}"] = {"mean": t_mean, "ci_lower": t_low, "ci_upper": t_up}

        return results

    def evaluate_impression_ranking(
        self,
        candidate_article_lists: List[List[str]],
        candidate_labels: List[List[int]],
        candidate_scores: List[List[float]],
        user_history_lengths: Optional[List[int]] = None,
        is_head_flags: Optional[List[bool]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate impression candidate reranking (AUC, MRR, nDCG@5, nDCG@10, ILD@10, Novelty@10, Coverage@10).
        Computes overall metrics and slices across User Cohort and Item Popularity with Bootstrap 95% CIs.
        """
        n_impressions = len(candidate_labels)
        if n_impressions == 0:
            return {}

        auc_list: List[float] = []
        mrr_list: List[float] = []
        ndcg5_list: List[float] = []
        ndcg10_list: List[float] = []
        ild_list: List[float] = []
        novelty_list: List[float] = []

        cold_auc: List[float] = []
        warm_auc: List[float] = []
        head_auc: List[float] = []
        tail_auc: List[float] = []

        cold_mrr: List[float] = []
        warm_mrr: List[float] = []
        head_mrr: List[float] = []
        tail_mrr: List[float] = []

        cold_ndcg5: List[float] = []
        warm_ndcg5: List[float] = []
        head_ndcg5: List[float] = []
        tail_ndcg5: List[float] = []

        cold_ndcg10: List[float] = []
        warm_ndcg10: List[float] = []
        head_ndcg10: List[float] = []
        tail_ndcg10: List[float] = []

        cold_ild: List[float] = []
        warm_ild: List[float] = []
        head_ild: List[float] = []
        tail_ild: List[float] = []

        cold_novelty: List[float] = []
        warm_novelty: List[float] = []
        head_novelty: List[float] = []
        tail_novelty: List[float] = []

        cold_recs: List[List[str]] = []
        warm_recs: List[List[str]] = []
        head_recs: List[List[str]] = []
        tail_recs: List[List[str]] = []

        all_ranked_recs: List[List[str]] = []

        for i in range(n_impressions):
            cands = candidate_article_lists[i]
            y_true = candidate_labels[i]
            scores = candidate_scores[i]

            ranked_order = np.argsort(scores)[::-1]
            ranked_cands = [cands[idx] for idx in ranked_order]
            all_ranked_recs.append(ranked_cands)

            is_cold = False
            if user_history_lengths is not None:
                is_cold = (user_history_lengths[i] <= self.cold_start_threshold)
                if is_cold:
                    cold_recs.append(ranked_cands)
                else:
                    warm_recs.append(ranked_cands)

            is_head = False
            if is_head_flags is not None:
                is_head = bool(is_head_flags[i])
                if is_head:
                    head_recs.append(ranked_cands)
                else:
                    tail_recs.append(ranked_cands)

            # AUC
            auc = compute_auc(y_true, scores)
            if auc is not None:
                auc_list.append(auc)
                if user_history_lengths is not None:
                    (cold_auc if is_cold else warm_auc).append(auc)
                if is_head_flags is not None:
                    (head_auc if is_head else tail_auc).append(auc)

            # MRR
            mrr = compute_mrr(y_true, ranked_order)
            mrr_list.append(mrr)
            if user_history_lengths is not None:
                (cold_mrr if is_cold else warm_mrr).append(mrr)
            if is_head_flags is not None:
                (head_mrr if is_head else tail_mrr).append(mrr)

            # nDCG@5 and nDCG@10
            ndcg5 = compute_ndcg_at_k(y_true, k=5, ranked_indices=ranked_order)
            ndcg10 = compute_ndcg_at_k(y_true, k=10, ranked_indices=ranked_order)
            ndcg5_list.append(ndcg5)
            ndcg10_list.append(ndcg10)
            if user_history_lengths is not None:
                (cold_ndcg5 if is_cold else warm_ndcg5).append(ndcg5)
                (cold_ndcg10 if is_cold else warm_ndcg10).append(ndcg10)
            if is_head_flags is not None:
                (head_ndcg5 if is_head else tail_ndcg5).append(ndcg5)
                (head_ndcg10 if is_head else tail_ndcg10).append(ndcg10)

            # Beyond Accuracy on Top-10
            if self.embeddings is not None and self.article_id_to_idx:
                ild = compute_intra_list_diversity(
                    ranked_cands, self.embeddings, self.article_id_to_idx, k=10
                )
                ild_list.append(ild)
                if user_history_lengths is not None:
                    (cold_ild if is_cold else warm_ild).append(ild)
                if is_head_flags is not None:
                    (head_ild if is_head else tail_ild).append(ild)

            if self.item_popularity_prob:
                nov = compute_novelty(
                    ranked_cands, self.item_popularity_prob, k=10
                )
                novelty_list.append(nov)
                if user_history_lengths is not None:
                    (cold_novelty if is_cold else warm_novelty).append(nov)
                if is_head_flags is not None:
                    (head_novelty if is_head else tail_novelty).append(nov)

        results: Dict[str, Any] = {
            "n_impressions": n_impressions,
            "metrics": {},
            "slices": {
                "user_cohort": {},
                "item_popularity": {},
            },
        }

        for name, scores in [
            ("AUC", auc_list),
            ("MRR", mrr_list),
            ("nDCG@5", ndcg5_list),
            ("nDCG@10", ndcg10_list),
            ("ILD@10", ild_list),
            ("Novelty@10", novelty_list),
        ]:
            if scores:
                mean, low, up = compute_bootstrap_ci(scores, n_bootstraps=self.n_bootstraps, ci=self.ci)
                results["metrics"][name] = {"mean": mean, "ci_lower": low, "ci_upper": up}

        cov = compute_catalog_coverage(all_ranked_recs, self.total_catalog_size, k=10)
        results["metrics"]["Coverage@10"] = {"mean": cov, "ci_lower": cov, "ci_upper": cov}

        # User cohort slices
        if user_history_lengths is not None:
            results["slices"]["user_cohort"]["cold_threshold"] = self.cold_start_threshold
            results["slices"]["user_cohort"]["cold_count"] = len(cold_mrr)
            results["slices"]["user_cohort"]["warm_count"] = len(warm_mrr)
            for m_name, c_scores, w_scores in [
                ("AUC", cold_auc, warm_auc),
                ("MRR", cold_mrr, warm_mrr),
                ("nDCG@5", cold_ndcg5, warm_ndcg5),
                ("nDCG@10", cold_ndcg10, warm_ndcg10),
                ("ILD@10", cold_ild, warm_ild),
                ("Novelty@10", cold_novelty, warm_novelty),
            ]:
                if c_scores:
                    c_mean, c_low, c_up = compute_bootstrap_ci(c_scores, n_bootstraps=self.n_bootstraps, ci=self.ci)
                    results["slices"]["user_cohort"][f"Cold_{m_name}"] = {"mean": c_mean, "ci_lower": c_low, "ci_upper": c_up}
                if w_scores:
                    w_mean, w_low, w_up = compute_bootstrap_ci(w_scores, n_bootstraps=self.n_bootstraps, ci=self.ci)
                    results["slices"]["user_cohort"][f"Warm_{m_name}"] = {"mean": w_mean, "ci_lower": w_low, "ci_upper": w_up}
            if cold_recs:
                cov_c = compute_catalog_coverage(cold_recs, self.total_catalog_size, k=10)
                cov_w = compute_catalog_coverage(warm_recs, self.total_catalog_size, k=10)
                results["slices"]["user_cohort"]["Cold_Coverage@10"] = {"mean": cov_c, "ci_lower": cov_c, "ci_upper": cov_c}
                results["slices"]["user_cohort"]["Warm_Coverage@10"] = {"mean": cov_w, "ci_lower": cov_w, "ci_upper": cov_w}

        # Item popularity slices
        if is_head_flags is not None:
            results["slices"]["item_popularity"]["head_count"] = len(head_mrr)
            results["slices"]["item_popularity"]["tail_count"] = len(tail_mrr)
            for m_name, h_scores, t_scores in [
                ("AUC", head_auc, tail_auc),
                ("MRR", head_mrr, tail_mrr),
                ("nDCG@5", head_ndcg5, tail_ndcg5),
                ("nDCG@10", head_ndcg10, tail_ndcg10),
                ("ILD@10", head_ild, tail_ild),
                ("Novelty@10", head_novelty, tail_novelty),
            ]:
                if h_scores:
                    h_mean, h_low, h_up = compute_bootstrap_ci(h_scores, n_bootstraps=self.n_bootstraps, ci=self.ci)
                    results["slices"]["item_popularity"][f"Head_{m_name}"] = {"mean": h_mean, "ci_lower": h_low, "ci_upper": h_up}
                if t_scores:
                    t_mean, t_low, t_up = compute_bootstrap_ci(t_scores, n_bootstraps=self.n_bootstraps, ci=self.ci)
                    results["slices"]["item_popularity"][f"Tail_{m_name}"] = {"mean": t_mean, "ci_lower": t_low, "ci_upper": t_up}

        return results


def print_evaluation_summary(
    title: str,
    results: Dict[str, Any],
) -> None:
    """Format and print benchmark results with 95% confidence intervals and slices."""
    print("\n" + "=" * 70)
    print(f" {title.upper()}")
    print(f" Total Impressions Evaluated: {results.get('n_impressions', 0):,}")
    print("=" * 70)

    print("\n[1] Overall Performance (Mean & 95% Bootstrap Confidence Interval):")
    print(f"{'Metric':<18} | {'Mean':<10} | {'95% CI Lower':<14} | {'95% CI Upper':<14}")
    print("-" * 64)
    for m_name, m_data in results.get("metrics", {}).items():
        print(
            f"{m_name:<18} | {m_data['mean']:<10.4f} | {m_data['ci_lower']:<14.4f} | {m_data['ci_upper']:<14.4f}"
        )

    slices = results.get("slices", {})
    user_slices = slices.get("user_cohort", {})
    if user_slices:
        thresh = user_slices.get("cold_threshold", "N/A")
        print(f"\n[2] User Slicing: Cold-Start (<= {thresh} clicks) vs. Warm (> {thresh} clicks):")
        print(f"    Cold Cohort N: {user_slices.get('cold_count', 0):,} | Warm Cohort N: {user_slices.get('warm_count', 0):,}")
        print(f"{'Slice Metric':<22} | {'Mean':<10} | {'95% CI Lower':<14} | {'95% CI Upper':<14}")
        print("-" * 68)
        for s_name, s_data in user_slices.items():
            if isinstance(s_data, dict):
                print(
                    f"{s_name:<22} | {s_data['mean']:<10.4f} | {s_data['ci_lower']:<14.4f} | {s_data['ci_upper']:<14.4f}"
                )

    item_slices = slices.get("item_popularity", {})
    if item_slices:
        print("\n[3] Item Slicing: Head Articles (Top 20%) vs. Tail Articles (Bottom 80%):")
        print(f"    Head Impressions N: {item_slices.get('head_count', 0):,} | Tail Impressions N: {item_slices.get('tail_count', 0):,}")
        print(f"{'Slice Metric':<22} | {'Mean':<10} | {'95% CI Lower':<14} | {'95% CI Upper':<14}")
        print("-" * 68)
        for s_name, s_data in item_slices.items():
            if isinstance(s_data, dict):
                print(
                    f"{s_name:<22} | {s_data['mean']:<10.4f} | {s_data['ci_lower']:<14.4f} | {s_data['ci_upper']:<14.4f}"
                )

    print("=" * 70 + "\n")
