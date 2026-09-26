import sys
from pathlib import Path
import numpy as np
import pytest
import torch
import torch.nn as nn

# Ensure src/ directory is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nrms_docvec import AdditiveAttention, NRMSDocVec, collate_train_fn


def test_additive_attention_shapes_and_masking():
    torch.manual_seed(42)
    batch_size = 4
    seq_len = 5
    embed_dim = 16

    h = torch.randn(batch_size, seq_len, embed_dim)
    # Mask the last 2 items for each sequence
    mask = torch.zeros((batch_size, seq_len), dtype=torch.bool)
    mask[:, -2:] = True

    layer = AdditiveAttention(embed_dim=embed_dim, hidden_dim=32)
    user_vec, attn_weights = layer(h, mask=mask)

    assert user_vec.shape == (batch_size, embed_dim)
    assert attn_weights.shape == (batch_size, seq_len)

    # Padded positions must have ~0 attention weight
    np.testing.assert_allclose(
        attn_weights[:, -2:].detach().numpy(),
        np.zeros((batch_size, 2)),
        atol=1e-6,
    )
    # Unmasked weights must sum to 1
    sums = attn_weights.sum(dim=-1).detach().numpy()
    np.testing.assert_allclose(sums, np.ones(batch_size), atol=1e-5)


def test_nrms_docvec_forward_and_backward():
    torch.manual_seed(42)
    num_articles = 50
    embed_dim = 32
    pretrained = np.random.randn(num_articles, embed_dim).astype(np.float32)

    model = NRMSDocVec(
        pretrained_embeddings=pretrained,
        num_heads=2,
        additive_hidden_dim=16,
        dropout=0.0,
        enable_recency_decay=False,
    )

    batch_size = 4
    history_len = 10
    cand_count = 5

    # 1-based indices in [1, num_articles]
    history_ids = torch.randint(1, num_articles + 1, (batch_size, history_len))
    candidate_ids = torch.randint(1, num_articles + 1, (batch_size, cand_count))

    scores = model(history_ids, candidate_ids)
    assert scores.shape == (batch_size, cand_count)

    # Test backward loss
    targets = torch.zeros(batch_size, dtype=torch.long)
    loss = nn.CrossEntropyLoss()(scores, targets)
    loss.backward()

    # Verify query and projection have gradients
    assert model.additive_attn.proj.weight.grad is not None
    assert torch.norm(model.additive_attn.proj.weight.grad) > 0


def test_recency_decay_downweights_older_clicks():
    torch.manual_seed(42)
    num_articles = 10
    embed_dim = 16
    # Same embedding for all articles to isolate the recency effect
    pretrained = np.ones((num_articles, embed_dim), dtype=np.float32)

    model = NRMSDocVec(
        pretrained_embeddings=pretrained,
        num_heads=2,
        additive_hidden_dim=16,
        dropout=0.0,
        enable_recency_decay=True,
        decay_half_life_hours=24.0,
    )
    model.eval()

    # Create a history of 2 identical articles:
    # Article 1 is clicked 1 hour ago
    # Article 2 is clicked 100 hours ago
    history_ids = torch.tensor([[1, 2]], dtype=torch.long)
    history_deltas = torch.tensor([[1.0, 100.0]], dtype=torch.float32)

    with torch.no_grad():
        h = model.embedding(history_ids)
        # Compute additive attention with decay
        scale = torch.nn.functional.softplus(model.decay_scale)
        penalty = scale * (history_deltas / model.decay_half_life_hours)
        _, weights = model.additive_attn(h, mask=None, decay_penalty=penalty)

    w1 = weights[0, 0].item()
    w2 = weights[0, 1].item()

    # Fresh click (1 hour ago) must receive significantly higher attention weight than stale click (100 hours ago)
    assert w1 > w2, f"Expected w1 ({w1}) > w2 ({w2})"
    assert w1 > 0.8, f"Expected strong preference for recent article, got w1={w1}"


def test_collate_train_fn():
    batch = [
        {
            "history_ids": [1, 2, 3],
            "history_deltas": [0.5, 1.2, 5.0],
            "candidate_ids": [10, 20, 30, 40, 50],
        },
        {
            "history_ids": [4],
            "history_deltas": [2.0],
            "candidate_ids": [11, 21, 31, 41, 51],
        },
    ]

    max_history_len = 5
    collated = collate_train_fn(batch, max_history_len=max_history_len)

    assert collated["history_ids"].shape == (2, max_history_len)
    assert collated["history_deltas"].shape == (2, max_history_len)
    assert collated["candidate_ids"].shape == (2, 5)
    assert collated["labels"].shape == (2,)

    # Check padding on row 1
    assert collated["history_ids"][1, 0] == 4
    assert collated["history_ids"][1, 1] == 0
    assert collated["history_ids"][1, 4] == 0
