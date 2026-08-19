import sys
from pathlib import Path

# Ensure src/ directory is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest

from bm25 import BM25InvertedIndex, default_tokenize
from eval_bm25 import calculate_recall_at_k


def test_tokenize():
    text = "Hello World! This is BM25-based Retrieval System (2026)."
    tokens = default_tokenize(text)
    assert tokens == ["hello", "world", "this", "is", "bm25", "based", "retrieval", "system", "2026"]


def test_bm25_build_and_search():
    articles_data = [
        {"article_id": "art_1", "title": "Python Machine Learning", "abstract": "Introduction to AI and ML"},
        {"article_id": "art_2", "title": "Deep Learning with PyTorch", "abstract": "Neural networks and gradient descent"},
        {"article_id": "art_3", "title": "Danish News Today", "abstract": "Breaking politics news in Copenhagen"},
    ]
    df = pd.DataFrame(articles_data)

    index = BM25InvertedIndex()
    index.build_index(df)

    assert index.is_built
    assert len(index.article_ids) == 3
    assert "python" in index.vocab

    # Search query "Machine Learning"
    results = index.search("Machine Learning", top_k=2)
    assert len(results) >= 1
    top_art_id, top_score = results[0]
    assert top_art_id == "art_1"
    assert top_score > 0.0


def test_bm25_candidate_subset_filtering():
    articles_data = [
        {"article_id": "art_1", "title": "Global Economy News", "abstract": "Finance and stock market"},
        {"article_id": "art_2", "title": "World Economy Update", "abstract": "Global inflation and trade"},
        {"article_id": "art_3", "title": "Sports Highlights", "abstract": "Football and tennis summary"},
    ]
    df = pd.DataFrame(articles_data)

    index = BM25InvertedIndex()
    index.build_index(df)

    # Restrict search to only art_2 and art_3
    results = index.search("Economy", top_k=2, candidate_ids=["art_2", "art_3"])
    retrieved_ids = [aid for aid, score in results]
    assert "art_1" not in retrieved_ids
    assert "art_2" in retrieved_ids


def test_calculate_recall_at_k():
    retrieved = ["art_1", "art_2", "art_3", "art_4", "art_5"]
    ground_truth = {"art_2", "art_5", "art_9"}

    # Recall@3: top 3 retrieved = {"art_1", "art_2", "art_3"}. Intersection with GT = {"art_2"} (1 hit out of 3 GT)
    recall_3 = calculate_recall_at_k(retrieved, ground_truth, k=3)
    assert pytest.approx(recall_3, 0.01) == 1 / 3

    # Recall@5: top 5 retrieved = {"art_1", "art_2", "art_3", "art_4", "art_5"}. Intersection = {"art_2", "art_5"} (2 hits out of 3 GT)
    recall_5 = calculate_recall_at_k(retrieved, ground_truth, k=5)
    assert pytest.approx(recall_5, 0.01) == 2 / 3
