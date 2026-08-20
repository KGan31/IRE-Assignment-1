"""
Evaluation Metrics Core for News Recommendation and Candidate Retrieval.

Implements:
1. Ranking & Accuracy Metrics:
   - Impression-level AUC (Area Under ROC Curve)
   - MRR (Mean Reciprocal Rank)
   - nDCG@K (Normalized Discounted Cumulative Gain at K)
   - Recall@K (Candidate Retrieval Recall)
2. Beyond-Accuracy Metrics:
   - Intra-List Diversity (ILD@K) via pairwise embedding cosine distance
   - Novelty@K (Self-Information Surprise based on item popularity)
   - Catalog Coverage@K (Proportion of catalog exposed)
3. Statistical Rigor:
   - Non-parametric Bootstrap 95% Confidence Intervals
"""

from typing import Dict, List, Optional, Sequence, Set, Tuple, Union
import numpy as np


def compute_auc(labels: Sequence[int], scores: Sequence[float]) -> Optional[float]:
    """
    Compute Area Under ROC Curve (AUC) for a single impression / ranked candidate list.
    
    Uses the Wilcoxon-Mann-Whitney statistic:
    AUC = (sum(rank_pos) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    
    Returns:
        float AUC in [0.0, 1.0], or None if the impression contains only positives or only negatives.
    """
    y_true = np.asarray(labels, dtype=np.int32)
    y_score = np.asarray(scores, dtype=np.float64)

    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))

    if n_pos == 0 or n_neg == 0:
        return None

    # Handle ties with average ranking (1-based ranks)
    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(y_score) + 1)

    # Handle duplicate score tie adjustments
    sorted_scores = y_score[order]
    unique_scores, inverse_indices, counts = np.unique(sorted_scores, return_inverse=True, return_counts=True)
    if len(unique_scores) < len(sorted_scores):
        tie_ranks = np.zeros_like(unique_scores, dtype=np.float64)
        np.add.at(tie_ranks, inverse_indices, ranks[order])
        tie_ranks /= counts
        ranks = tie_ranks[inverse_indices][np.argsort(order)]

    pos_ranks = ranks[y_true == 1]
    rank_sum = np.sum(pos_ranks)

    auc = (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(np.clip(auc, 0.0, 1.0))


def compute_mrr(labels: Sequence[int], ranked_indices: Optional[Sequence[int]] = None) -> float:
    """
    Compute Reciprocal Rank (RR) for a single ranked list.
    
    RR = 1 / rank_first_positive (1-based rank), or 0.0 if no positive exists.
    """
    y_true = np.asarray(labels, dtype=np.int32)
    if ranked_indices is not None:
        y_true = y_true[np.asarray(ranked_indices, dtype=np.int64)]

    pos_positions = np.where(y_true == 1)[0]
    if len(pos_positions) == 0:
        return 0.0
    return float(1.0 / (pos_positions[0] + 1))


def compute_ndcg_at_k(
    labels: Sequence[int],
    k: int = 10,
    ranked_indices: Optional[Sequence[int]] = None,
) -> float:
    """
    Compute Normalized Discounted Cumulative Gain at rank K (nDCG@K) for binary labels.
    
    DCG@K = sum_{i=1}^K y_i / log2(i + 1)
    IDCG@K = sum_{i=1}^{min(K, n_pos)} 1 / log2(i + 1)
    nDCG@K = DCG@K / IDCG@K
    """
    y_true = np.asarray(labels, dtype=np.int32)
    if ranked_indices is not None:
        y_true = y_true[np.asarray(ranked_indices, dtype=np.int64)]

    y_k = y_true[:k]
    n_pos = int(np.sum(y_true == 1))
    if n_pos == 0:
        return 0.0

    # Discount factors 1 / log2(i + 1) for i = 1, ..., len(y_k)
    discounts = 1.0 / np.log2(np.arange(2, len(y_k) + 2))
    dcg = np.sum(y_k * discounts)

    # Ideal DCG
    ideal_hits = min(k, n_pos)
    ideal_discounts = 1.0 / np.log2(np.arange(2, ideal_hits + 2))
    idcg = np.sum(ideal_discounts)

    if idcg <= 0.0:
        return 0.0
    return float(np.clip(dcg / idcg, 0.0, 1.0))


def compute_recall_at_k(retrieved_ids: Sequence[str], ground_truth_set: Set[str], k: int) -> float:
    """
    Compute Recall@K for candidate retrieval:
    Recall@K = |Retrieved[:K] cap GroundTruth| / |GroundTruth|
    """
    if not ground_truth_set:
        return 0.0
    top_k_set = set(retrieved_ids[:k])
    hits = len(top_k_set.intersection(ground_truth_set))
    return float(hits / len(ground_truth_set))


def compute_intra_list_diversity(
    ranked_item_ids: Sequence[str],
    embeddings: np.ndarray,
    item_to_idx: Dict[str, int],
    k: int = 10,
) -> float:
    """
    Compute Intra-List Diversity (ILD@K) using average pairwise cosine distance:
    ILD@K = 2 / (K * (K - 1)) * sum_{i < j} (1 - cosine_similarity(e_i, e_j))
    
    If fewer than 2 valid items exist in embeddings, returns 0.0.
    """
    top_k_ids = ranked_item_ids[:k]
    valid_indices = [item_to_idx[aid] for aid in top_k_ids if aid in item_to_idx]

    n_valid = len(valid_indices)
    if n_valid < 2:
        return 0.0

    top_vecs = embeddings[valid_indices]  # Shape: (n_valid, dim)
    # Ensure vectors are L2-normalized
    norms = np.linalg.norm(top_vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    norm_vecs = top_vecs / norms

    # Pairwise cosine similarities
    sim_matrix = np.dot(norm_vecs, norm_vecs.T)
    # Extract strictly upper triangle
    upper_tri_indices = np.triu_indices(n_valid, k=1)
    pairwise_similarities = sim_matrix[upper_tri_indices]
    pairwise_distances = 1.0 - pairwise_similarities

    return float(np.clip(np.mean(pairwise_distances), 0.0, 2.0))


def compute_novelty(
    ranked_item_ids: Sequence[str],
    item_popularity_prob: Dict[str, float],
    k: int = 10,
    default_min_prob: float = 1e-6,
) -> float:
    """
    Compute Novelty@K (Self-Information Surprise) of a recommended list:
    Novelty@K = 1 / K * sum_{i=1}^K -log2(p(a_i))
    
    where p(a_i) is the prior click probability of article a_i in the training catalog.
    Unseen/new articles get assigned `default_min_prob`.
    """
    top_k_ids = ranked_item_ids[:k]
    if not top_k_ids:
        return 0.0

    surprises = []
    for aid in top_k_ids:
        p = item_popularity_prob.get(aid, default_min_prob)
        p = max(p, default_min_prob)
        surprises.append(-np.log2(p))

    return float(np.mean(surprises))


def compute_catalog_coverage(
    all_recommended_lists: Sequence[Sequence[str]],
    total_catalog_size: int,
    k: int = 10,
) -> float:
    """
    Compute Catalog Coverage@K:
    Coverage@K = |Unique items recommended across all users in top-K| / Total Catalog Size
    """
    if total_catalog_size <= 0:
        return 0.0

    unique_items: Set[str] = set()
    for rec_list in all_recommended_lists:
        unique_items.update(rec_list[:k])

    return float(len(unique_items) / total_catalog_size)


def compute_bootstrap_ci(
    scores: Sequence[float],
    n_bootstraps: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Compute non-parametric Bootstrap Confidence Interval for a list of sample metric scores.
    
    Returns:
        (mean_score, ci_lower, ci_upper)
    """
    arr = np.asarray(scores, dtype=np.float64)
    if len(arr) == 0:
        return 0.0, 0.0, 0.0

    mean_score = float(np.mean(arr))
    if len(arr) == 1:
        return mean_score, mean_score, mean_score

    rng = np.random.default_rng(seed)
    n = len(arr)
    # Resample with replacement in chunks to be memory efficient
    boot_means = np.empty(n_bootstraps, dtype=np.float64)
    batch_size = min(n_bootstraps, 500)

    for i in range(0, n_bootstraps, batch_size):
        chunk_size = min(batch_size, n_bootstraps - i)
        indices = rng.integers(0, n, size=(chunk_size, n))
        boot_means[i : i + chunk_size] = np.mean(arr[indices], axis=1)

    alpha = (1.0 - ci) / 2.0
    ci_lower = float(np.percentile(boot_means, alpha * 100))
    ci_upper = float(np.percentile(boot_means, (1.0 - alpha) * 100))

    return mean_score, ci_lower, ci_upper
