import sys
from pathlib import Path

# Ensure src/ directory is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest

from bm25 import BM25InvertedIndex, create_article_text_map, default_tokenize
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


def test_create_article_text_map():
    articles_data = [
        {"article_id": "art_1", "title": "Quantum Computing", "abstract": "Introduction to qubits and superposition"},
        {"article_id": "art_2", "title": "Climate Change", "abstract": "Global warming and carbon emissions"},
    ]
    df = pd.DataFrame(articles_data)

    # 1. Title only
    title_map = create_article_text_map(df, fields="title")
    assert title_map["art_1"] == "Quantum Computing"
    assert title_map["art_2"] == "Climate Change"

    # 2. Title + Abstract
    both_map = create_article_text_map(df, fields="title_abstract")
    assert both_map["art_1"] == "Quantum Computing Introduction to qubits and superposition"
    assert both_map["art_2"] == "Climate Change Global warming and carbon emissions"

    # 3. Alias title+abstract
    plus_map = create_article_text_map(df, fields="title+abstract")
    assert plus_map["art_1"] == "Quantum Computing Introduction to qubits and superposition"
    assert plus_map["art_2"] == "Climate Change Global warming and carbon emissions"


def test_bm25_history_query_title_vs_abstract():
    articles_data = [
        {"article_id": "art_1", "title": "Space Exploration", "abstract": "Mars rover and NASA missions"},
        {"article_id": "art_2", "title": "Deep Ocean", "abstract": "Marine life in Mariana Trench"},
        {"article_id": "art_3", "title": "Red Planet Journey", "abstract": "A journey to Mars"},
    ]
    df = pd.DataFrame(articles_data)
    index = BM25InvertedIndex()
    index.build_index(df)

    history = ["art_1"]
    
    # Query using title only: "Space Exploration"
    title_map = create_article_text_map(df, fields="title")
    q_title = " ".join([title_map[aid] for aid in history])
    res_title = index.search(q_title, top_k=3)
    retrieved_title_ids = [aid for aid, _ in res_title]
    assert "art_1" in retrieved_title_ids

    # Query using title + abstract: contains "Mars rover NASA"
    both_map = create_article_text_map(df, fields="title_abstract")
    q_both = " ".join([both_map[aid] for aid in history])
    assert "mars" in q_both.lower()
    res_both = index.search(q_both, top_k=3)
    retrieved_both_ids = [aid for aid, _ in res_both]
    # art_3 has "Mars" in abstract, so it gets retrieved with high score when history query includes abstract
    assert "art_3" in retrieved_both_ids


def test_bm25_cold_start_popularity_fallback_batch():
    articles_data = [
        {"article_id": "art_1", "title": "Article One", "abstract": "Content one"},
        {"article_id": "art_2", "title": "Article Two", "abstract": "Content two"},
        {"article_id": "art_3", "title": "Article Three", "abstract": "Content three"},
    ]
    df = pd.DataFrame(articles_data)
    index = BM25InvertedIndex()
    index.build_index(df)

    popular_ids = ["art_2", "art_3", "art_1"]

    # Cold start query: empty string "" (user has 0 historical clicks)
    res = index.batch_search([""], top_k=3, fallback_popular_ids=popular_ids)
    assert len(res) == 1
    retrieved = [aid for aid, score in res[0]]
    # Must return popular articles ordered by popularity rank
    assert retrieved == ["art_2", "art_3", "art_1"]


def test_bm25_cold_start_candidate_ordering_by_popularity():
    articles_data = [
        {"article_id": "art_1", "title": "Quantum Physics", "abstract": "Physics concepts"},
        {"article_id": "art_2", "title": "World Cup Soccer", "abstract": "Football tournament"},
        {"article_id": "art_3", "title": "Cooking Recipes", "abstract": "Italian pasta"},
    ]
    df = pd.DataFrame(articles_data)
    index = BM25InvertedIndex()
    index.build_index(df)

    # Candidate set of 3 articles to rank for an impression
    candidates = ["art_1", "art_2", "art_3"]
    popularity_map = {"art_1": 10.0, "art_2": 100.0, "art_3": 50.0}

    # Cold-start user with no query (empty string)
    results = index.search(
        "",
        top_k=3,
        candidate_ids=candidates,
        popularity_map=popularity_map,
    )
    ranked_ids = [aid for aid, score in results]
    # art_2 (popularity 100) > art_3 (popularity 50) > art_1 (popularity 10)
    assert ranked_ids == ["art_2", "art_3", "art_1"]

