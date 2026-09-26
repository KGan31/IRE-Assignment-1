#!/usr/bin/env python
"""
ANN (FAISS HNSW) vs. Exact Cosine Similarity (FAISS FlatIP) Ablation Study.
Runs evaluation on MIND-Large and EB-NeRD-Large validation sets,
measuring speed/latency (QPS, ms/query, build time) and retrieval tradeoffs (Recall@K, ILD, Novelty, Coverage).
"""

import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple

import numpy as np
import pandas as pd
import polars as pl
from tqdm import tqdm

# Ensure src is in sys.path
ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(ROOT_DIR))

from embeddings import (
    EmbeddingIndex,
    compute_user_representation,
    normalize_l2,
)
from eval_harness import OfflineEvaluationHarness
from metrics import (
    compute_recall_at_k,
    compute_intra_list_diversity,
    compute_novelty,
    compute_catalog_coverage,
    compute_auc,
    compute_mrr,
    compute_ndcg_at_k,
)


def run_benchmark(
    dataset_name: str,
    ds_dir: Path,
    impressions_path: Path,
    history_path: Path,
    sample_size: int = 10000,
    k_list: Tuple[int, ...] = (50, 100, 200),
    max_history_len: int = 20,
) -> Dict[str, Any]:
    print(f"\n{'='*70}")
    print(f"  BENCHMARKING: {dataset_name.upper()} (Large Validation Split)")
    print(f"{'='*70}")

    articles_path = ds_dir / "articles.parquet"
    emb_path = ds_dir / "article_embeddings.npy"
    ids_path = ds_dir / "article_ids.json"

    # 1. Load articles & embeddings
    print(f"Loading articles and embeddings from {ds_dir}...")
    articles_df = pd.read_parquet(articles_path)
    embeddings = np.load(emb_path)
    with open(ids_path, "r", encoding="utf-8") as f:
        article_ids = json.load(f)
    article_id_to_idx = {aid: i for i, aid in enumerate(article_ids)}
    embeddings_norm = normalize_l2(embeddings)
    global_mean_vector = normalize_l2(np.mean(embeddings_norm, axis=0))

    catalog_size = len(article_ids)
    dim = embeddings_norm.shape[1]
    print(f"Catalog size: {catalog_size:,} articles | Embedding Dimension: {dim}")

    # 2. Build Exact Index (FAISS FlatIP)
    print("\n--- Building Exact Index (FAISS FlatIP / Cosine Similarity) ---")
    t0 = time.perf_counter()
    exact_index = EmbeddingIndex(use_approximate=False, force_numpy=False)
    exact_index.build_index(embeddings_norm, article_ids)
    exact_build_time = time.perf_counter() - t0
    print(f"Exact Index Build Time: {exact_build_time:.4f}s")

    # 3. Build ANN Index (FAISS IndexHNSWFlat)
    print("\n--- Building ANN Index (FAISS IndexHNSWFlat, M=32, efSearch=64) ---")
    t0 = time.perf_counter()
    ann_index = EmbeddingIndex(use_approximate=True, force_numpy=False)
    ann_index.build_index(embeddings_norm, article_ids)
    ann_build_time = time.perf_counter() - t0
    print(f"ANN Index Build Time: {ann_build_time:.4f}s")

    # 4. Load validation impressions and build user queries
    print(f"\nLoading validation impressions from {impressions_path}...")
    impr_df = pl.read_parquet(impressions_path)
    if sample_size > 0 and len(impr_df) > sample_size:
        impr_df = impr_df.head(sample_size)
    print(f"Evaluating {len(impr_df):,} validation impressions...")

    user_ids = impr_df["user_id"].to_list()
    clicked_lists = impr_df["clicked_article_ids"].to_list()
    cands_lists = impr_df["candidate_article_ids"].to_list() if "candidate_article_ids" in impr_df.columns else [[]] * len(impr_df)

    # Load history
    print(f"Loading history from {history_path}...")
    hist_df = pl.read_parquet(history_path)
    if "user_id" in hist_df.columns and "clicked_article_id" in hist_df.columns:
        agg_hist = (
            hist_df.group_by("user_id", maintain_order=True)
            .agg(pl.col("clicked_article_id").tail(max_history_len))
        )
        u_map = dict(zip(agg_hist["user_id"].to_list(), [list(h) for h in agg_hist["clicked_article_id"].to_list()]))
    else:
        u_map = {}

    # Build query vectors
    query_vectors = []
    ground_truth_sets = []
    valid_cands = []
    for uid, clist, cands in zip(user_ids, clicked_lists, cands_lists):
        gt = set(clist) if clist else set()
        if not gt:
            continue
        clicks = u_map.get(uid, [])
        u_vec = compute_user_representation(clicks, article_id_to_idx, embeddings_norm)
        if u_vec is None:
            u_vec = global_mean_vector
        query_vectors.append(u_vec)
        ground_truth_sets.append(gt)
        valid_cands.append(list(cands) if cands else [])

    query_matrix = np.vstack(query_vectors).astype(np.float32)
    num_queries = len(query_matrix)
    print(f"Generated {num_queries:,} user representation query vectors.")

    # 5. Measure Exact Search Latency & Retrieval
    print("\n[1/2] Running Exact (Flat) Global Catalog Retrieval...")
    batch_size = 500
    exact_latencies = []
    exact_retrieved = []

    for i in tqdm(range(0, num_queries, batch_size), desc="Exact Retrieval"):
        batch_q = query_matrix[i:i + batch_size]
        t0 = time.perf_counter()
        batch_res = exact_index.batch_search(batch_q, top_k=max(k_list))
        elapsed = time.perf_counter() - t0
        exact_latencies.append((elapsed / len(batch_q)) * 1000.0)  # ms per query
        for res in batch_res:
            exact_retrieved.append([aid for aid, sc in res])

    exact_mean_ms = float(np.mean(exact_latencies))
    exact_qps = float(num_queries / (sum(exact_latencies) * batch_size / 1000.0 / len(exact_latencies)))

    # 6. Measure ANN Search Latency & Retrieval
    print("\n[2/2] Running ANN (HNSW) Global Catalog Retrieval...")
    ann_latencies = []
    ann_retrieved = []

    for i in tqdm(range(0, num_queries, batch_size), desc="ANN Retrieval"):
        batch_q = query_matrix[i:i + batch_size]
        t0 = time.perf_counter()
        batch_res = ann_index.batch_search(batch_q, top_k=max(k_list))
        elapsed = time.perf_counter() - t0
        ann_latencies.append((elapsed / len(batch_q)) * 1000.0)  # ms per query
        for res in batch_res:
            ann_retrieved.append([aid for aid, sc in res])

    ann_mean_ms = float(np.mean(ann_latencies))
    ann_qps = float(num_queries / (sum(ann_latencies) * batch_size / 1000.0 / len(ann_latencies)))

    # 7. Compute Retrieval Quality Metrics
    print("\nComputing Retrieval & Beyond-Accuracy Metrics...")
    
    # Item popularity for Novelty
    pop_probs = {}
    if "clicked_article_id" in hist_df.columns:
        vc = hist_df["clicked_article_id"].value_counts()
        total_clicks = len(hist_df)
        if total_clicks > 0:
            aids = vc["clicked_article_id"].to_list()
            cnts = vc["count"].to_list()
            pop_probs = {aid: c / total_clicks for aid, c in zip(aids, cnts)}

    def eval_retrieval_metrics(retrieved_lists):
        r50 = [compute_recall_at_k(ret, gt, k=50) for gt, ret in zip(ground_truth_sets, retrieved_lists)]
        r100 = [compute_recall_at_k(ret, gt, k=100) for gt, ret in zip(ground_truth_sets, retrieved_lists)]
        r200 = [compute_recall_at_k(ret, gt, k=200) for gt, ret in zip(ground_truth_sets, retrieved_lists)]
        
        # Beyond-accuracy on top-10
        top10_lists = [r[:10] for r in retrieved_lists]
        ild10 = [compute_intra_list_diversity(r, embeddings_norm, article_id_to_idx, k=10) for r in top10_lists]
        nov10 = [compute_novelty(r, pop_probs, k=10) for r in top10_lists] if pop_probs else [0.0]
        cov10 = compute_catalog_coverage(top10_lists, catalog_size)
        
        return {
            "Recall@50": float(np.mean(r50)),
            "Recall@100": float(np.mean(r100)),
            "Recall@200": float(np.mean(r200)),
            "ILD@10": float(np.mean(ild10)),
            "Novelty@10": float(np.mean(nov10)),
            "Coverage@10": float(cov10),
        }

    exact_metrics = eval_retrieval_metrics(exact_retrieved)
    ann_metrics = eval_retrieval_metrics(ann_retrieved)

    speedup = exact_mean_ms / max(1e-6, ann_mean_ms)

    results = {
        "dataset": dataset_name,
        "catalog_size": catalog_size,
        "num_queries": num_queries,
        "embedding_dim": dim,
        "exact": {
            "build_time_s": exact_build_time,
            "latency_ms_per_query": exact_mean_ms,
            "qps": exact_qps,
            **exact_metrics,
        },
        "ann": {
            "build_time_s": ann_build_time,
            "latency_ms_per_query": ann_mean_ms,
            "qps": ann_qps,
            **ann_metrics,
        },
        "speedup": speedup,
        "recall_preservation_r200": (ann_metrics["Recall@200"] / max(1e-8, exact_metrics["Recall@200"])) * 100.0,
    }

    return results


def main():
    root = Path(__file__).resolve().parent
    data_dir = root / "data" / "processed"

    results_all = {}

    # MIND-Large Dev Benchmark
    mind_dir = data_dir / "mind_large"
    mind_res = run_benchmark(
        dataset_name="MIND-Large",
        ds_dir=mind_dir,
        impressions_path=mind_dir / "impressions_dev.parquet",
        history_path=mind_dir / "history_dev.parquet",
        sample_size=10000,
    )
    results_all["mind_large"] = mind_res

    # EB-NeRD-Large Val Benchmark
    ebnerd_dir = data_dir / "ebnerd_large"
    ebnerd_res = run_benchmark(
        dataset_name="EB-NeRD-Large",
        ds_dir=ebnerd_dir,
        impressions_path=ebnerd_dir / "impressions_val.parquet",
        history_path=ebnerd_dir / "history_val.parquet",
        sample_size=10000,
    )
    results_all["ebnerd_large"] = ebnerd_res

    # Output formatted JSON and Markdown
    out_json = root / "docs" / "ablations" / "ann_ablation_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results_all, f, indent=2)

    print("\n" + "=" * 80)
    print("                      ANN vs. EXACT COSINE ABLATION SUMMARY")
    print("=" * 80)
    for ds_key, res in results_all.items():
        print(f"\nDataset: {res['dataset']} (Catalog: {res['catalog_size']:,} articles, Dim: {res['embedding_dim']})")
        print("-" * 75)
        print(f"{'Metric / Property':<25} | {'Exact (Cosine / Flat)':<20} | {'ANN (FAISS HNSW)':<20} | {'Delta / Speedup'}")
        print("-" * 75)
        print(f"{'Index Build Time':<25} | {res['exact']['build_time_s']:<18.4f}s | {res['ann']['build_time_s']:<18.4f}s | {res['ann']['build_time_s']/max(1e-4, res['exact']['build_time_s']):.1f}x build cost")
        print(f"{'Query Latency (mean)':<25} | {res['exact']['latency_ms_per_query']:<18.4f}ms| {res['ann']['latency_ms_per_query']:<18.4f}ms| {res['speedup']:.2f}x faster")
        print(f"{'Throughput (QPS)':<25} | {res['exact']['qps']:<18.1f} | {res['ann']['qps']:<18.1f} | +{res['ann']['qps']-res['exact']['qps']:.1f} QPS")
        print(f"{'Recall@50':<25} | {res['exact']['Recall@50']:<20.4f} | {res['ann']['Recall@50']:<20.4f} | {res['ann']['Recall@50']-res['exact']['Recall@50']:+.4f}")
        print(f"{'Recall@100':<25} | {res['exact']['Recall@100']:<20.4f} | {res['ann']['Recall@100']:<20.4f} | {res['ann']['Recall@100']-res['exact']['Recall@100']:+.4f}")
        print(f"{'Recall@200':<25} | {res['exact']['Recall@200']:<20.4f} | {res['ann']['Recall@200']:<20.4f} | {res['ann']['Recall@200']-res['exact']['Recall@200']:+.4f}")
        print(f"{'Recall Preservation @200':<25} | {'100.0%':<20} | {res['recall_preservation_r200']:<19.2f}% | Tradeoff: -{100.0-res['recall_preservation_r200']:.2f}%")
        print(f"{'Intra-List Diversity @10':<25} | {res['exact']['ILD@10']:<20.4f} | {res['ann']['ILD@10']:<20.4f} | {res['ann']['ILD@10']-res['exact']['ILD@10']:+.4f}")
        print(f"{'Novelty @10 (Surprise)':<25} | {res['exact']['Novelty@10']:<20.4f} | {res['ann']['Novelty@10']:<20.4f} | {res['ann']['Novelty@10']-res['exact']['Novelty@10']:+.4f}")
        print(f"{'Catalog Coverage @10':<25} | {res['exact']['Coverage@10']*100:<19.2f}% | {res['ann']['Coverage@10']*100:<19.2f}% | {res['ann']['Coverage@10']*100-res['exact']['Coverage@10']*100:+.2f}% pts")
        print("-" * 75)


if __name__ == "__main__":
    main()
