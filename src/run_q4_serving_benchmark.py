#!/usr/bin/env python
"""
Question 4 Serving & Scale Analysis Benchmark Runner (Assignment 2, Q4).
=======================================================================
1. Measures index memory footprint for FAISS HNSW (ANN) vs FAISS Flat (Exact)
   and feature store lookup structures.
2. Benchmarks single-request latency distribution (p50, p95, p99, mean) for
   Stage 1 Candidate Generation (HNSW ANN + BM25) and Stage 2 LightGBM Re-ranking.
3. Evaluates ranking quality (AUC, MRR, nDCG@5, nDCG@10) across the entire validation split.
4. Models Cost / QPS projections under production SLA (p99 < 100ms).
5. Compares ANN vs. Exact trade-offs and generates data for Q4.md.
"""

import argparse
import gc
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import faiss
import numpy as np
import pandas as pd
import polars as pl
from tqdm import tqdm

# Ensure src is on path
SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))

from embeddings import EmbeddingIndex, compute_user_representation, normalize_l2
from metrics import compute_auc, compute_mrr, compute_ndcg_at_k
from retrieve_then_rank import RetrieveThenRankPipeline


def measure_index_and_feature_memory(
    dataset: str,
    processed_dir: Path = Path("data/processed"),
) -> Dict[str, Any]:
    """
    Measures empirical RAM usage for FAISS Flat vs HNSW indexes,
    along with article and user feature store footprints.
    """
    ds_dir = processed_dir / dataset
    emb_path = ds_dir / "article_embeddings.npy"
    art_path = ds_dir / "articles.parquet"
    user_feat_path = ds_dir / "user_features_val.parquet"
    if not user_feat_path.exists():
        user_feat_path = ds_dir / "user_features.parquet"

    raw_embs = np.load(emb_path).astype(np.float32)
    n_docs, dim = raw_embs.shape

    # 1. Flat Index Memory
    flat_idx = faiss.IndexFlatIP(dim)
    flat_idx.add(raw_embs)
    flat_bytes = len(faiss.serialize_index(flat_idx))

    # 2. HNSW Index Memory (M=32)
    hnsw_idx = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
    hnsw_idx.hnsw.efSearch = 64
    hnsw_idx.add(raw_embs)
    hnsw_bytes = len(faiss.serialize_index(hnsw_idx))

    del flat_idx, hnsw_idx
    gc.collect()

    # 3. Article Feature Store Memory
    art_df = pl.read_parquet(art_path)
    art_memory_bytes = art_df.estimated_size()

    # 4. User Feature Store Memory (if exists)
    user_memory_bytes = 0
    if user_feat_path.exists():
        u_df = pl.read_parquet(user_feat_path)
        user_memory_bytes = u_df.estimated_size()
    else:
        # Check history validation
        hist_val_path = ds_dir / "history_val.parquet"
        if not hist_val_path.exists():
            hist_val_path = ds_dir / "history_validation.parquet"
        if hist_val_path.exists():
            h_df = pl.read_parquet(hist_val_path)
            user_memory_bytes = h_df.estimated_size()

    return {
        "catalog_size": n_docs,
        "embedding_dim": dim,
        "raw_vectors_mb": float(n_docs * dim * 4 / (1024 ** 2)),
        "flat_index_bytes": flat_bytes,
        "flat_index_mb": float(flat_bytes / (1024 ** 2)),
        "hnsw_index_bytes": hnsw_bytes,
        "hnsw_index_mb": float(hnsw_bytes / (1024 ** 2)),
        "graph_overhead_mb": float((hnsw_bytes - flat_bytes) / (1024 ** 2)),
        "graph_overhead_pct": float((hnsw_bytes - flat_bytes) / flat_bytes * 100.0),
        "article_store_mb": float(art_memory_bytes / (1024 ** 2)),
        "user_store_mb": float(user_memory_bytes / (1024 ** 2)),
        "total_serving_ram_mb": float((hnsw_bytes + art_memory_bytes + user_memory_bytes) / (1024 ** 2)),
    }


def benchmark_full_validation(
    dataset: str,
    top_k: int = 100,
    use_approximate: bool = True,
    sample_size: int = 0,
    processed_dir: Path = Path("data/processed"),
) -> Dict[str, Any]:
    """
    Runs live two-stage pipeline evaluation over the entire validation split with HNSW ANN.
    Measures:
    - Stage 1 and Stage 2 ranking quality (AUC, MRR, nDCG@5, nDCG@10)
    - Per-request latency distribution (p50, p95, p99, mean, min, max)
    """
    print(f"\n{'='*75}\n >>> EVALUATING TWO-STAGE PIPELINE: {dataset.upper()} (ANN={use_approximate}, K={top_k})\n{'='*75}")

    pipeline = RetrieveThenRankPipeline(
        dataset=dataset,
        processed_dir=processed_dir,
        retriever_type="hybrid",
        top_k=top_k,
        use_approximate=use_approximate,
    )

    data_dir = processed_dir / dataset
    imp_file = data_dir / "impressions_val.parquet"
    if not imp_file.exists():
        imp_file = data_dir / "impressions_validation.parquet"

    print(f"Loading validation impressions from {imp_file}...")
    impressions_df = pl.read_parquet(imp_file)
    total_val_impressions = len(impressions_df)
    if sample_size > 0 and len(impressions_df) > sample_size:
        impressions_df = impressions_df.slice(0, sample_size)
    print(f"Evaluating {len(impressions_df):,} of {total_val_impressions:,} validation impressions...")

    hist_file = data_dir / "history_val.parquet"
    if not hist_file.exists():
        hist_file = data_dir / "history_validation.parquet"

    user_hist_map: Dict[str, List[Tuple[str, Any, float]]] = {}
    if hist_file.exists():
        print(f"Loading history from {hist_file}...")
        h_df = pl.read_parquet(hist_file)
        has_dwell = "dwell_time" in h_df.columns
        for row in h_df.select([
            "user_id",
            "clicked_article_id",
            "click_time",
            "dwell_time" if has_dwell else pl.lit(0.0).alias("dwell_time"),
        ]).iter_rows():
            uid, aid, c_time, dwell = row[0], row[1], row[2], row[3]
            if uid not in user_hist_map:
                user_hist_map[uid] = []
            user_hist_map[uid].append((aid, c_time, dwell or 0.0))

    # Evaluation accumulators
    stage1_auc_list = []
    stage1_mrr_list = []
    stage1_ndcg5_list = []
    stage1_ndcg10_list = []

    stage2_auc_list = []
    stage2_mrr_list = []
    stage2_ndcg5_list = []
    stage2_ndcg10_list = []

    latencies_total = []
    latencies_stage1 = []
    latencies_stage2 = []

    # Dedicated full-catalog retrieval latency probe (measures end-to-end full catalog retrieval + re-ranking)
    catalog_latencies_total = []
    catalog_latencies_stage1 = []
    catalog_latencies_stage2 = []

    start_bench_time = time.perf_counter()

    for idx, imp in enumerate(tqdm(impressions_df.iter_rows(named=True), total=len(impressions_df), desc=f"Pipeline Inference ({dataset.upper()})")):
        uid = imp["user_id"]
        ts = imp["timestamp"]
        cands = imp.get("candidate_article_ids") or []
        clicked = set(imp.get("clicked_article_ids") or [])
        if not cands or not clicked:
            continue

        u_clicks = user_hist_map.get(uid, [])

        # 1. Evaluate impression candidate ranking
        out = pipeline.run_impression(
            user_id=uid,
            imp_time=ts,
            user_history_clicks=u_clicks,
            candidate_subset=cands,
            k=top_k,
        )

        latencies_total.append(out["latency_ms"]["total"])
        latencies_stage1.append(out["latency_ms"]["stage1_retrieval"])
        latencies_stage2.append(out["latency_ms"]["stage2_reranking"])

        # Ground truth labels
        labels = [1 if c in clicked else 0 for c in cands]
        if not any(lbl == 1 for lbl in labels):
            continue

        # Stage 1 scores
        stage1_map = {item[0]: item[1] for item in out["stage1_candidates"]}
        stage1_scores = [stage1_map.get(c, 0.0) for c in cands]
        stage1_order = np.argsort(stage1_scores)[::-1]

        # Stage 2 scores
        stage2_map = dict(out["stage2_ranked"])
        stage2_scores = [stage2_map.get(c, -999.0) for c in cands]
        stage2_order = np.argsort(stage2_scores)[::-1]

        # Quality metrics
        auc1 = compute_auc(labels, stage1_scores)
        auc2 = compute_auc(labels, stage2_scores)
        if auc1 is not None and auc2 is not None:
            stage1_auc_list.append(auc1)
            stage2_auc_list.append(auc2)

        stage1_mrr_list.append(compute_mrr(labels, ranked_indices=stage1_order))
        stage2_mrr_list.append(compute_mrr(labels, ranked_indices=stage2_order))

        stage1_ndcg5_list.append(compute_ndcg_at_k(labels, k=5, ranked_indices=stage1_order))
        stage2_ndcg5_list.append(compute_ndcg_at_k(labels, k=5, ranked_indices=stage2_order))

        stage1_ndcg10_list.append(compute_ndcg_at_k(labels, k=10, ranked_indices=stage1_order))
        stage2_ndcg10_list.append(compute_ndcg_at_k(labels, k=10, ranked_indices=stage2_order))

        # 2. Probe true full-catalog retrieval latency every 10 impressions
        if idx % 10 == 0:
            t_cat0 = time.perf_counter()
            cat_out = pipeline.run_impression(
                user_id=uid,
                imp_time=ts,
                user_history_clicks=u_clicks,
                candidate_subset=None,
                k=top_k,
            )
            t_cat_tot = (time.perf_counter() - t_cat0) * 1000.0
            catalog_latencies_total.append(t_cat_tot)
            catalog_latencies_stage1.append(cat_out["latency_ms"]["stage1_retrieval"])
            catalog_latencies_stage2.append(cat_out["latency_ms"]["stage2_reranking"])

    total_bench_duration = time.perf_counter() - start_bench_time

    # Latency percentiles helper
    def get_lat_stats(lats: List[float]) -> Dict[str, float]:
        if not lats:
            return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "min": 0.0, "max": 0.0}
        return {
            "mean": float(np.mean(lats)),
            "p50": float(np.percentile(lats, 50)),
            "p95": float(np.percentile(lats, 95)),
            "p99": float(np.percentile(lats, 99)),
            "min": float(np.min(lats)),
            "max": float(np.max(lats)),
        }

    summary = {
        "dataset": dataset,
        "use_approximate": use_approximate,
        "index_type": "FAISS IndexHNSWFlat (M=32, efSearch=64)" if use_approximate else "FAISS IndexFlatIP (Exact)",
        "catalog_size": len(pipeline.article_ids),
        "total_validation_impressions": total_val_impressions,
        "evaluated_impressions": len(latencies_total),
        "benchmark_wall_time_s": float(total_bench_duration),
        "overall_qps": float(len(latencies_total) / total_bench_duration),
        "metrics": {
            "Stage 1 (Before Re-Ranking)": {
                "AUC": float(np.mean(stage1_auc_list)),
                "MRR": float(np.mean(stage1_mrr_list)),
                "nDCG@5": float(np.mean(stage1_ndcg5_list)),
                "nDCG@10": float(np.mean(stage1_ndcg10_list)),
            },
            "Stage 2 (LightGBM Re-Ranker)": {
                "AUC": float(np.mean(stage2_auc_list)),
                "MRR": float(np.mean(stage2_mrr_list)),
                "nDCG@5": float(np.mean(stage2_ndcg5_list)),
                "nDCG@10": float(np.mean(stage2_ndcg10_list)),
            },
            "Delta": {
                "AUC": float(np.mean(stage2_auc_list) - np.mean(stage1_auc_list)),
                "MRR": float(np.mean(stage2_mrr_list) - np.mean(stage1_mrr_list)),
                "nDCG@5": float(np.mean(stage2_ndcg5_list) - np.mean(stage1_ndcg5_list)),
                "nDCG@10": float(np.mean(stage2_ndcg10_list) - np.mean(stage1_ndcg10_list)),
            },
        },
        "impression_scoring_latency_ms": {
            "total": get_lat_stats(latencies_total),
            "stage1_retrieval": get_lat_stats(latencies_stage1),
            "stage2_reranking": get_lat_stats(latencies_stage2),
        },
        "full_catalog_retrieval_latency_ms": {
            "total": get_lat_stats(catalog_latencies_total),
            "stage1_retrieval": get_lat_stats(catalog_latencies_stage1),
            "stage2_reranking": get_lat_stats(catalog_latencies_stage2),
        },
        "sla_p99_ms": float(np.percentile(latencies_total, 99)),
        "sla_target_ms": 100.0,
        "sla_pass": bool(np.percentile(latencies_total, 99) < 100.0),
    }

    print("\n" + "=" * 75)
    print(f"EVALUATION SUMMARY: {dataset.upper()} (use_approximate={use_approximate})")
    print("=" * 75)
    print(f"{'Metric':<12s} | {'Stage 1 (Prior)':<18s} | {'Stage 2 (Re-Ranker)':<18s} | {'Delta':<10s}")
    print("-" * 75)
    for m in ["AUC", "MRR", "nDCG@5", "nDCG@10"]:
        b = summary["metrics"]["Stage 1 (Before Re-Ranking)"][m]
        a = summary["metrics"]["Stage 2 (LightGBM Re-Ranker)"][m]
        d = summary["metrics"]["Delta"][m]
        print(f"{m:<12s} | {b:18.4f} | {a:18.4f} | {d:+10.4f}")
    print("-" * 75)
    p50 = summary["impression_scoring_latency_ms"]["total"]["p50"]
    p95 = summary["impression_scoring_latency_ms"]["total"]["p95"]
    p99 = summary["impression_scoring_latency_ms"]["total"]["p99"]
    s1_mean = summary["impression_scoring_latency_ms"]["stage1_retrieval"]["mean"]
    s2_mean = summary["impression_scoring_latency_ms"]["stage2_reranking"]["mean"]
    print(f"Latency (Impression): p50={p50:.2f}ms | p95={p95:.2f}ms | p99={p99:.2f}ms | SLA (<100ms): {'PASS' if p99 < 100 else 'FAIL'}")
    print(f"Breakdown: Stage 1 = {s1_mean:.2f}ms | Stage 2 = {s2_mean:.2f}ms")
    if catalog_latencies_total:
        cp99 = summary["full_catalog_retrieval_latency_ms"]["total"]["p99"]
        c_mean = summary["full_catalog_retrieval_latency_ms"]["total"]["mean"]
        print(f"Full-Catalog Serving Latency: mean={c_mean:.2f}ms | p99={cp99:.2f}ms | SLA (<100ms): {'PASS' if cp99 < 100 else 'FAIL'}")
    print("=" * 75)

    return summary


def compute_cost_qps_projections(
    benchmark_results: Dict[str, Any],
    instance_type: str = "AWS c6i.2xlarge (8 vCPUs, 16 GB RAM)",
    hourly_rate_usd: float = 0.34,
) -> Dict[str, Any]:
    """
    Back-of-envelope cost and QPS derivation under target SLA (p99 < 100ms).
    """
    projections = {}
    for ds in ["mind", "ebnerd"]:
        data = benchmark_results[ds]
        p99 = data["impression_scoring_latency_ms"]["total"]["p99"]
        mean_lat_ms = data["impression_scoring_latency_ms"]["total"]["mean"]
        cat_p99 = data["full_catalog_retrieval_latency_ms"]["total"]["p99"]
        cat_mean_ms = data["full_catalog_retrieval_latency_ms"]["total"]["mean"]

        # Single core QPS based on mean latency
        single_core_qps = 1000.0 / max(0.1, mean_lat_ms)
        # 8 vCPUs with 70% parallel serving efficiency (concurrency factor)
        instance_qps = single_core_qps * 8.0 * 0.70

        # Cost per query
        queries_per_hour = instance_qps * 3600.0
        cost_per_1000_queries = (hourly_rate_usd / queries_per_hour) * 1000.0
        cost_per_million_queries = cost_per_1000_queries * 1000.0

        projections[ds] = {
            "instance_type": instance_type,
            "hourly_cost_usd": hourly_rate_usd,
            "mean_latency_ms": mean_lat_ms,
            "p99_latency_ms": p99,
            "catalog_mean_latency_ms": cat_mean_ms,
            "catalog_p99_latency_ms": cat_p99,
            "single_core_qps": float(single_core_qps),
            "instance_qps_8cores": float(instance_qps),
            "cost_per_1k_queries_usd": float(cost_per_1000_queries),
            "cost_per_1m_queries_usd": float(cost_per_million_queries),
            "queries_per_dollar": float(1.0 / (cost_per_1000_queries / 1000.0)),
        }

    return projections


def main():
    parser = argparse.ArgumentParser(description="Assignment 2 Question 4 Serving & Scale Benchmark.")
    parser.add_argument("--top_k", type=int, default=100)
    parser.add_argument("--sample_size", type=int, default=0, help="0 to evaluate entire validation split")
    parser.add_argument("--out_json", type=str, default="models/q4_serving_benchmark_results.json")
    args = parser.parse_args()

    results: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "configuration": {
            "top_k": args.top_k,
            "sample_size": "Entire Validation Split" if args.sample_size <= 0 else args.sample_size,
            "use_approximate": True,
            "ann_algorithm": "FAISS IndexHNSWFlat",
            "hnsw_m": 32,
            "hnsw_efSearch": 64,
        },
        "memory_footprint": {},
        "pipeline_evaluations": {},
        "cost_and_qps_projections": {},
    }

    # 1. Memory footprint measurement
    print("\n" + "=" * 75)
    print(" >>> STEP 1: MEASURING INDEX AND FEATURE STORE MEMORY FOOTPRINT")
    print("=" * 75)
    for ds in ["mind", "ebnerd"]:
        mem_info = measure_index_and_feature_memory(ds)
        results["memory_footprint"][ds] = mem_info
        print(f"[{ds.upper()}] Catalog: {mem_info['catalog_size']:,} articles ({mem_info['embedding_dim']}d)")
        print(f"  - Flat Index: {mem_info['flat_index_mb']:.2f} MB")
        print(f"  - HNSW Index: {mem_info['hnsw_index_mb']:.2f} MB (+{mem_info['graph_overhead_mb']:.2f} MB graph overhead)")
        print(f"  - Article Store: {mem_info['article_store_mb']:.2f} MB")
        print(f"  - User History Store: {mem_info['user_store_mb']:.2f} MB")
        print(f"  - Total Serving RAM: {mem_info['total_serving_ram_mb']:.2f} MB")

    # 2. Pipeline evaluations on entire validation split with HNSW ANN
    for ds in ["mind", "ebnerd"]:
        ds_res = benchmark_full_validation(
            dataset=ds,
            top_k=args.top_k,
            use_approximate=True,
            sample_size=args.sample_size,
        )
        results["pipeline_evaluations"][ds] = ds_res

    # 3. Cost and QPS economic projections
    cost_proj = compute_cost_qps_projections(results["pipeline_evaluations"])
    results["cost_and_qps_projections"] = cost_proj

    # Save output JSON
    out_p = Path(args.out_json)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nBenchmark results successfully saved to: {out_p}")


if __name__ == "__main__":
    main()
