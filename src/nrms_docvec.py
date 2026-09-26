"""
NRMSDocVec: Neural News Recommendation with Multi-Head Self-Attention over Document Vectors.

Implements:
1. Additive Attention layer with optional exponential recency-decay bias.
2. NRMSDocVec dual-encoder architecture:
   - News Encoder: Precomputed dense document embeddings (e.g. MiniLM or BERT).
   - User Encoder: Multi-Head Self-Attention across history + Additive Attention pooling.
   - Click Predictor: Dot-product scoring with negative-sampling Cross-Entropy Loss.
3. Principled Improvement:
   - Recency-decay penalty in attention logits: older clicks are downweighted exponentially.
4. Evaluation integration with OfflineEvaluationHarness and paired bootstrap testing.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union
import bisect
import json
import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from metrics import compute_auc, compute_bootstrap_ci, compute_mrr, compute_ndcg_at_k


# ---------------------------------------------------------------------------
# Attention & Model Modules
# ---------------------------------------------------------------------------

class AdditiveAttention(nn.Module):
    """
    Additive Attention layer (Query-based pooling).
    Learns a context query vector to compute attention weights over sequence representations.
    Supports injecting recency decay or engagement penalties into the attention logits.
    """

    def __init__(self, embed_dim: int, hidden_dim: int = 200):
        super().__init__()
        self.proj = nn.Linear(embed_dim, hidden_dim)
        self.query = nn.Linear(hidden_dim, 1, bias=False)

    def forward(
        self,
        h: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        decay_penalty: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h: Input representations of shape (batch_size, seq_len, embed_dim).
            mask: Boolean tensor of shape (batch_size, seq_len) where True indicates padding.
            decay_penalty: Optional tensor of shape (batch_size, seq_len) subtracted from logits.

        Returns:
            user_vec: Pooled representation of shape (batch_size, embed_dim).
            attn_weights: Softmax attention distribution of shape (batch_size, seq_len).
        """
        # Linear projection + Tanh activation
        v = torch.tanh(self.proj(h))  # (B, H, hidden_dim)
        scores = self.query(v).squeeze(-1)  # (B, H)

        if decay_penalty is not None:
            # Downweight older clicks by subtracting decay penalty
            scores = scores - decay_penalty

        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        attn_weights = torch.softmax(scores, dim=-1)  # (B, H)
        # Numerical guard for all-masked sequences
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

        # Weighted combination: sum_i (alpha_i * h_i)
        user_vec = torch.sum(attn_weights.unsqueeze(-1) * h, dim=1)  # (B, D)
        return user_vec, attn_weights


def safe_l2_norm(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Safely normalizes vectors to unit L2 norm without NaN gradients at zero."""
    norm = torch.norm(x, p=2, dim=-1, keepdim=True)
    mask = (norm > eps).float()
    return mask * (x / (norm + 1e-12))


class NRMSDocVec(nn.Module):
    """
    NRMSDocVec Model.
    Utilizes precomputed article embeddings as frozen document representations,
    and trains a Multi-Head Self-Attention + Additive Attention User Encoder.
    """

    def __init__(
        self,
        pretrained_embeddings: np.ndarray,
        num_heads: int = 4,
        additive_hidden_dim: int = 200,
        dropout: float = 0.1,
        enable_recency_decay: bool = False,
        decay_half_life_hours: float = 24.0,
    ):
        super().__init__()
        # Pretrained embeddings table: row 0 is reserved for PAD / UNK
        # Shape: (num_articles + 1, embed_dim)
        embed_dim = pretrained_embeddings.shape[1]
        self.embed_dim = embed_dim
        self.enable_recency_decay = enable_recency_decay
        self.decay_half_life_hours = decay_half_life_hours

        # Construct PyTorch embedding with index 0 as padding (all zeros)
        pad_row = np.zeros((1, embed_dim), dtype=np.float32)
        full_table = np.vstack([pad_row, pretrained_embeddings.astype(np.float32)])
        self.embedding = nn.Embedding.from_pretrained(
            torch.from_numpy(full_table),
            freeze=True,
            padding_idx=0,
        )

        # Global average fallback for cold-start users
        global_avg = np.mean(pretrained_embeddings, axis=0)
        norm = np.linalg.norm(global_avg)
        if norm > 0:
            global_avg = global_avg / norm
        self.global_user_vec = nn.Parameter(torch.tensor(global_avg, dtype=torch.float32), requires_grad=True)

        # Multi-Head Self-Attention over history
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

        # Additive Attention pooling
        self.additive_attn = AdditiveAttention(
            embed_dim=embed_dim,
            hidden_dim=additive_hidden_dim,
        )

        # Learnable decay scale parameter (if recency decay enabled)
        if enable_recency_decay:
            self.decay_scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        else:
            self.register_parameter("decay_scale", None)

    def encode_user(
        self,
        history_ids: torch.Tensor,
        history_deltas: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode user history sequence into a single user representation vector.
        """
        # Identify padding positions: (B, H)
        padding_mask = history_ids == 0
        all_padded = padding_mask.all(dim=-1)

        # Guard against MHA NaN when an entire sequence is padding
        safe_mask = padding_mask.clone()
        if all_padded.any():
            safe_mask[all_padded, 0] = False

        # Lookup embeddings: (B, H, D)
        h = self.embedding(history_ids)

        # Multi-Head Self-Attention:
        attn_out, _ = self.multihead_attn(h, h, h, key_padding_mask=safe_mask)
        attn_out = self.dropout(attn_out)
        if all_padded.any():
            attn_out[all_padded] = 0.0
        h = self.layer_norm(h + attn_out)

        # Compute recency decay penalty if enabled
        decay_penalty = None
        if self.enable_recency_decay and history_deltas is not None:
            scale = F.softplus(self.decay_scale)
            decay_penalty = scale * (history_deltas / self.decay_half_life_hours)

        # Additive attention pooling:
        user_vec, _ = self.additive_attn(
            h=h,
            mask=padding_mask,
            decay_penalty=decay_penalty,
        )

        # Fallback to global_user_vec for users with no prior history
        if all_padded.any():
            user_vec = torch.where(all_padded.unsqueeze(-1), self.global_user_vec.unsqueeze(0), user_vec)

        # Normalize user vector safely
        user_vec = safe_l2_norm(user_vec)
        return user_vec

    def forward(
        self,
        history_ids: torch.Tensor,
        candidate_ids: torch.Tensor,
        history_deltas: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass for training or batched inference."""
        user_vec = self.encode_user(history_ids, history_deltas)
        cand_embeds = safe_l2_norm(self.embedding(candidate_ids))
        scores = torch.sum(cand_embeds * user_vec.unsqueeze(1), dim=-1)
        return scores


# ---------------------------------------------------------------------------
# Dataset & Data Handling
# ---------------------------------------------------------------------------

class NRMSDataset(Dataset):
    """
    PyTorch Dataset for NRMS impressions.
    Supports negative sampling for training, and full candidate list passing for evaluation.
    """

    def __init__(
        self,
        impressions_path: Path,
        history_path: Path,
        article_id_to_idx: Dict[str, int],
        max_history_len: int = 20,
        n_negatives: int = 4,
        is_training: bool = True,
        seed: int = 42,
        max_samples: Optional[int] = None,
    ):
        super().__init__()
        self.article_id_to_idx = article_id_to_idx
        self.max_history_len = max_history_len
        self.n_negatives = n_negatives
        self.is_training = is_training
        self.rng = np.random.default_rng(seed)

        # 1. Load impressions first
        print(f"Loading impressions from {impressions_path.name}...")
        imp_df = pl.read_parquet(impressions_path)
        if max_samples is not None:
            imp_df = imp_df.head(max_samples)

        active_uids = set(imp_df["user_id"].unique().to_list())

        # 2. Load history filtered to active impression users
        print(f"Loading history from {history_path.name}...")
        hist_df = pl.read_parquet(history_path)
        if len(active_uids) < len(hist_df):
            hist_df = hist_df.filter(pl.col("user_id").is_in(active_uids))

        hist_sorted = hist_df.sort(["user_id", "click_time"])
        grouped_hist = hist_sorted.group_by("user_id", maintain_order=True).agg([
            pl.col("clicked_article_id"),
            pl.col("click_time"),
        ])

        self.user_to_articles: Dict[str, List[str]] = dict(
            zip(grouped_hist["user_id"].to_list(), grouped_hist["clicked_article_id"].to_list())
        )
        self.user_to_times: Dict[str, List[Any]] = dict(
            zip(grouped_hist["user_id"].to_list(), grouped_hist["click_time"].to_list())
        )

        self.user_ids: List[str] = imp_df["user_id"].to_list()
        self.timestamps: List[Any] = imp_df["timestamp"].to_list()
        self.candidates: List[List[str]] = imp_df["candidate_article_ids"].to_list()
        self.clicked: List[List[str]] = imp_df["clicked_article_ids"].to_list()

        # Filter valid impressions
        self.valid_indices: List[int] = []
        for i in range(len(self.user_ids)):
            cands = self.candidates[i]
            clicks = self.clicked[i]
            if clicks and cands:
                if self.is_training:
                    # Must have at least 1 positive and 1 negative candidate
                    pos_set = set(clicks)
                    neg_list = [aid for aid in cands if aid not in pos_set]
                    if len(neg_list) >= 1:
                        self.valid_indices.append(i)
                else:
                    self.valid_indices.append(i)

        print(f"Loaded {len(self.valid_indices)} valid impressions for {'training' if is_training else 'eval'}.")

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        imp_idx = self.valid_indices[idx]
        user_id = self.user_ids[imp_idx]
        imp_time = self.timestamps[imp_idx]
        cands = self.candidates[imp_idx]
        clicks = self.clicked[imp_idx]
        clicks_set = set(clicks)

        # 1. Leakage-free history lookup using bisect
        all_articles = self.user_to_articles.get(user_id, [])
        all_times = self.user_to_times.get(user_id, [])

        if all_articles:
            split_idx = bisect.bisect_left(all_times, imp_time)
            prior_articles = all_articles[:split_idx][-self.max_history_len:]
            prior_times = all_times[:split_idx][-self.max_history_len:]
        else:
            prior_articles = []
            prior_times = []

        # Convert history articles to 1-based indices (0 = PAD)
        hist_indices = [self.article_id_to_idx.get(aid, 0) for aid in prior_articles]
        hist_deltas = []
        for c_time in prior_times:
            # Hours elapsed between click and impression
            delta_sec = (imp_time - c_time).total_seconds() if hasattr(imp_time - c_time, "total_seconds") else float(imp_time - c_time) / 1e6
            delta_hours = max(0.0, float(delta_sec) / 3600.0)
            hist_deltas.append(delta_hours)

        if self.is_training:
            # Positive sample: 1 randomly selected clicked item
            pos_aid = self.rng.choice(clicks)
            pos_idx = self.article_id_to_idx.get(pos_aid, 0)

            # Negative samples: n_negatives selected from unclicked candidates
            neg_pool = [aid for aid in cands if aid not in clicks_set]
            if len(neg_pool) >= self.n_negatives:
                sampled_negs = self.rng.choice(neg_pool, size=self.n_negatives, replace=False).tolist()
            else:
                # Sample with replacement if candidate pool is smaller than n_negatives
                sampled_negs = self.rng.choice(neg_pool, size=self.n_negatives, replace=True).tolist()

            cand_indices = [pos_idx] + [self.article_id_to_idx.get(na, 0) for na in sampled_negs]
            # Ground truth label index is 0 (positive item is placed first)
            return {
                "history_ids": hist_indices,
                "history_deltas": hist_deltas,
                "candidate_ids": cand_indices,
                "label": 0,
            }
        else:
            # Evaluation mode: all candidate articles with binary ground truth labels
            cand_indices = [self.article_id_to_idx.get(aid, 0) for aid in cands]
            labels = [1 if aid in clicks_set else 0 for aid in cands]
            return {
                "history_ids": hist_indices,
                "history_deltas": hist_deltas,
                "candidate_ids": cand_indices,
                "candidate_aids": cands,
                "labels": labels,
                "user_history_len": len(hist_indices),
            }


def collate_train_fn(batch: List[Dict[str, Any]], max_history_len: int = 20) -> Dict[str, torch.Tensor]:
    """
    Collate function for training batches.
    Pads history_ids and history_deltas to max_history_len.
    """
    batch_size = len(batch)
    hist_tensor = torch.zeros((batch_size, max_history_len), dtype=torch.long)
    deltas_tensor = torch.zeros((batch_size, max_history_len), dtype=torch.float32)
    cand_tensor = torch.empty((batch_size, len(batch[0]["candidate_ids"])), dtype=torch.long)
    labels_tensor = torch.zeros(batch_size, dtype=torch.long)

    for i, item in enumerate(batch):
        h_ids = item["history_ids"]
        h_deltas = item["history_deltas"]
        h_len = len(h_ids)
        if h_len > 0:
            hist_tensor[i, :h_len] = torch.tensor(h_ids, dtype=torch.long)
            deltas_tensor[i, :h_len] = torch.tensor(h_deltas, dtype=torch.float32)
        cand_tensor[i] = torch.tensor(item["candidate_ids"], dtype=torch.long)

    return {
        "history_ids": hist_tensor,
        "history_deltas": deltas_tensor,
        "candidate_ids": cand_tensor,
        "labels": labels_tensor,
    }


# ---------------------------------------------------------------------------
# Training & Evaluation Loops
# ---------------------------------------------------------------------------

def train_nrms_epoch(
    model: NRMSDocVec,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> float:
    """Trains NRMSDocVec for one epoch using Cross-Entropy Loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    criterion = nn.CrossEntropyLoss()

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} Training", leave=False)
    for batch in pbar:
        hist_ids = batch["history_ids"].to(device)
        hist_deltas = batch["history_deltas"].to(device)
        cand_ids = batch["candidate_ids"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        logits = model(
            history_ids=hist_ids,
            candidate_ids=cand_ids,
            history_deltas=hist_deltas,
        )

        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1
        pbar.set_postfix(loss=f"{total_loss / n_batches:.4f}")

    return total_loss / max(1, n_batches)


@torch.no_grad()
def evaluate_nrms_dataset(
    model: NRMSDocVec,
    eval_dataset: NRMSDataset,
    device: torch.device,
    batch_size: int = 256,
    max_history_len: int = 20,
) -> Dict[str, Any]:
    """
    Evaluates NRMSDocVec impression ranking on a validation dataset.
    Computes impression-level AUC, MRR, nDCG@5, and nDCG@10.
    """
    model.eval()

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

    # Evaluate impression by impression or micro-batches
    for i in tqdm(range(len(eval_dataset)), desc="Evaluating Impressions", leave=False):
        item = eval_dataset[i]
        h_ids = item["history_ids"][:max_history_len]
        h_deltas = item["history_deltas"][:max_history_len]
        cands = item["candidate_ids"]
        labels = item["labels"]

        if sum(labels) == 0 or len(labels) < 2:
            continue

        # Prepare tensors
        hist_tensor = torch.zeros((1, max_history_len), dtype=torch.long, device=device)
        deltas_tensor = torch.zeros((1, max_history_len), dtype=torch.float32, device=device)
        if len(h_ids) > 0:
            hist_tensor[0, :len(h_ids)] = torch.tensor(h_ids, dtype=torch.long, device=device)
            deltas_tensor[0, :len(h_deltas)] = torch.tensor(h_deltas, dtype=torch.float32, device=device)

        cand_tensor = torch.tensor([cands], dtype=torch.long, device=device)

        scores = model(
            history_ids=hist_tensor,
            candidate_ids=cand_tensor,
            history_deltas=deltas_tensor,
        )[0].cpu().numpy().tolist()

        # Compute impression metrics
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

    # Compute bootstrap 95% CIs
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
