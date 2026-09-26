"""
Processed Dataset Statistics Analyzer for MIND and EB-NeRD
===========================================================
This script reads the processed parquet files (articles.parquet, impressions_*.parquet,
history_*.parquet, user_features_*.parquet) for both MIND and EB-NeRD datasets and computes
comprehensive statistics across Train, Validation, and Test splits.

Metrics computed:
  1. Temporal statistics (min/max timestamps, duration, split boundaries, temporal gaps)
  2. Impression & click statistics (counts, CTR, candidates/clicks per impression distributions)
  3. User history & cold-start statistics (history size, cold start %, cross-split user overlap)
  4. Article catalog & candidate coverage (catalog size, categories, word lengths, candidate overlap)
  5. Cross-dataset comparative summary

Usage:
    python processed_stats.py
    python processed_stats.py --dataset all --output_json data/processed_stats.json --output_md data/processed_stats_report.md
    python processed_stats.py --dataset mind
    python processed_stats.py --dataset ebnerd
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Set, Tuple, Optional
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute comprehensive statistics for processed Train, Validation, and Test splits."
    )
    parser.add_argument(
        "--processed_dir",
        type=str,
        default="data/processed",
        help="Root directory containing processed dataset folders (default: data/processed)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["all", "mind", "ebnerd"],
        default="all",
        help="Dataset to analyze: 'all', 'mind', or 'ebnerd' (default: all)"
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="data/processed_stats.json",
        help="Path to save JSON statistics output (default: data/processed_stats.json)"
    )
    parser.add_argument(
        "--output_md",
        type=str,
        default="data/processed_stats_report.md",
        help="Path to save Markdown report output (default: data/processed_stats_report.md)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress console output"
    )
    return parser.parse_args()


def safe_to_list(val) -> list:
    """Safely convert array/list/None to Python list."""
    if val is None:
        return []
    try:
        if len(val) == 0:
            return []
    except TypeError:
        return []
    if isinstance(val, np.ndarray):
        return val.tolist()
    return list(val)


def get_series_stats(data: List[float | int]) -> Dict[str, float]:
    """Compute summary statistics for a numeric list."""
    if not data or len(data) == 0:
        return {"min": 0, "max": 0, "mean": 0.0, "median": 0.0, "std": 0.0}
    arr = np.array(data)
    return {
        "min": int(np.min(arr)) if np.issubdtype(arr.dtype, np.integer) else round(float(np.min(arr)), 2),
        "max": int(np.max(arr)) if np.issubdtype(arr.dtype, np.integer) else round(float(np.max(arr)), 2),
        "mean": round(float(np.mean(arr)), 2),
        "median": round(float(np.median(arr)), 2),
        "std": round(float(np.std(arr)), 2),
    }


def compute_temporal_stats(
    imp_dfs: Dict[str, pd.DataFrame]
) -> Dict[str, Any]:
    """Calculate temporal metrics per split and cross-split gaps."""
    stats = {}
    timestamps = {}

    for split in ["train", "val", "test"]:
        df = imp_dfs.get(split)
        if df is not None and not df.empty and "timestamp" in df.columns:
            ts = pd.to_datetime(df["timestamp"])
            min_ts, max_ts = ts.min(), ts.max()
            duration_days = round((max_ts - min_ts).total_seconds() / 86400.0, 2)
            duration_hours = round((max_ts - min_ts).total_seconds() / 3600.0, 2)
            timestamps[split] = (min_ts, max_ts)
            stats[split] = {
                "min_time": str(min_ts),
                "max_time": str(max_ts),
                "duration_days": duration_days,
                "duration_hours": duration_hours,
            }
        else:
            timestamps[split] = (None, None)
            stats[split] = None

    # Overall Combined
    all_min = min((ts[0] for ts in timestamps.values() if ts[0] is not None), default=None)
    all_max = max((ts[1] for ts in timestamps.values() if ts[1] is not None), default=None)
    
    combined_days = round((all_max - all_min).total_seconds() / 86400.0, 2) if all_min and all_max else 0.0
    combined_hours = round((all_max - all_min).total_seconds() / 3600.0, 2) if all_min and all_max else 0.0

    # Gaps between splits
    train_to_val_gap = None
    if timestamps["train"][1] is not None and timestamps["val"][0] is not None:
        train_to_val_gap = round((timestamps["val"][0] - timestamps["train"][1]).total_seconds() / 3600.0, 4)

    val_to_test_gap = None
    if timestamps["val"][1] is not None and timestamps["test"][0] is not None:
        val_to_test_gap = round((timestamps["test"][0] - timestamps["val"][1]).total_seconds() / 3600.0, 4)

    stats["combined"] = {
        "min_time": str(all_min) if all_min else None,
        "max_time": str(all_max) if all_max else None,
        "duration_days": combined_days,
        "duration_hours": combined_hours,
        "train_to_val_gap_hours": train_to_val_gap,
        "val_to_test_gap_hours": val_to_test_gap,
    }
    return stats


def analyze_split_impressions(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze impression DataFrame for candidate and click statistics."""
    if df is None or df.empty:
        return {
            "impression_count": 0,
            "unique_users": 0,
            "total_candidates": 0,
            "total_clicks": 0,
            "ctr_percent": 0.0,
            "candidates_per_imp": get_series_stats([]),
            "clicks_per_imp": get_series_stats([]),
            "impressions_per_user": get_series_stats([]),
        }

    n_impressions = len(df)
    unique_users = df["user_id"].nunique() if "user_id" in df.columns else 0

    candidate_lens = []
    click_lens = []
    
    # Extract lengths safely
    for cand, clk in zip(df["candidate_article_ids"], df["clicked_article_ids"]):
        c_list = safe_to_list(cand)
        k_list = safe_to_list(clk)
        candidate_lens.append(len(c_list))
        click_lens.append(len(k_list))

    total_candidates = sum(candidate_lens)
    total_clicks = sum(click_lens)
    ctr = round((total_clicks / total_candidates * 100.0), 2) if total_candidates > 0 else 0.0

    imp_per_user = []
    if "user_id" in df.columns:
        user_counts = df["user_id"].value_counts().tolist()
        imp_per_user = user_counts

    return {
        "impression_count": n_impressions,
        "unique_users": unique_users,
        "total_candidates": total_candidates,
        "total_clicks": total_clicks,
        "ctr_percent": ctr,
        "candidates_per_imp": get_series_stats(candidate_lens),
        "clicks_per_imp": get_series_stats(click_lens),
        "impressions_per_user": get_series_stats(imp_per_user),
    }


def analyze_dataset_history_and_users(
    imp_dfs: Dict[str, pd.DataFrame],
    hist_dfs: Dict[str, pd.DataFrame],
) -> Dict[str, Any]:
    """Compute history lengths, cold-start rates, and cross-split user overlap."""
    history_stats = {}
    user_sets = {}

    for split in ["train", "val", "test"]:
        imp_df = imp_dfs.get(split)
        hist_df = hist_dfs.get(split)

        imp_users: Set[str] = set(imp_df["user_id"].dropna().unique()) if imp_df is not None and not imp_df.empty else set()
        user_sets[split] = imp_users

        hist_rows = len(hist_df) if hist_df is not None else 0
        
        if hist_df is not None and not hist_df.empty and "user_id" in hist_df.columns:
            user_click_counts = hist_df["user_id"].value_counts()
            user_hist_lens = user_click_counts.tolist()
            hist_users = set(user_click_counts.index)
        else:
            user_hist_lens = []
            hist_users = set()

        # Check cold-start users in impressions (impression users with 0 history clicks)
        users_with_hist_in_imp = imp_users.intersection(hist_users)
        cold_start_imp_users_count = len(imp_users) - len(users_with_hist_in_imp)
        cold_start_pct = round(cold_start_imp_users_count / len(imp_users) * 100.0, 2) if imp_users else 0.0

        history_stats[split] = {
            "history_rows": hist_rows,
            "unique_history_users": len(hist_users),
            "history_length_per_user": get_series_stats(user_hist_lens),
            "cold_start_users_count": cold_start_imp_users_count,
            "cold_start_users_percent": cold_start_pct,
        }

    # Cross-split user overlap
    train_users = user_sets["train"]
    val_users = user_sets["val"]
    test_users = user_sets["test"]

    all_users = train_users.union(val_users).union(test_users)
    train_val_overlap = train_users.intersection(val_users)
    train_test_overlap = train_users.intersection(test_users)
    val_test_overlap = val_users.intersection(test_users)
    all_three_overlap = train_users.intersection(val_users).intersection(test_users)

    val_cold_start = val_users - train_users
    test_cold_start = test_users - (train_users.union(val_users))

    overlap_stats = {
        "total_unique_impression_users": len(all_users),
        "train_unique_users": len(train_users),
        "val_unique_users": len(val_users),
        "test_unique_users": len(test_users),
        "train_val_overlap_count": len(train_val_overlap),
        "train_val_overlap_pct_of_val": round(len(train_val_overlap) / len(val_users) * 100.0, 2) if val_users else 0.0,
        "val_new_users_count": len(val_cold_start),
        "val_new_users_pct_of_val": round(len(val_cold_start) / len(val_users) * 100.0, 2) if val_users else 0.0,
        "train_test_overlap_count": len(train_test_overlap),
        "train_test_overlap_pct_of_test": round(len(train_test_overlap) / len(test_users) * 100.0, 2) if test_users else 0.0,
        "val_test_overlap_count": len(val_test_overlap),
        "val_test_overlap_pct_of_test": round(len(val_test_overlap) / len(test_users) * 100.0, 2) if test_users else 0.0,
        "test_new_users_count": len(test_cold_start),
        "test_new_users_pct_of_test": round(len(test_cold_start) / len(test_users) * 100.0, 2) if test_users else 0.0,
        "users_in_all_splits_count": len(all_three_overlap),
        "users_in_all_splits_pct": round(len(all_three_overlap) / len(all_users) * 100.0, 2) if all_users else 0.0,
    }

    return {
        "per_split": history_stats,
        "cross_split_user_overlap": overlap_stats,
    }


def analyze_article_catalog_and_coverage(
    articles_df: pd.DataFrame,
    imp_dfs: Dict[str, pd.DataFrame]
) -> Dict[str, Any]:
    """Compute catalog content statistics, category distribution, and split-level coverage."""
    if articles_df is None or articles_df.empty:
        return {"catalog_size": 0}

    total_articles = len(articles_df)
    
    # Category statistics
    cat_counts = {}
    if "category" in articles_df.columns:
        cat_counts = articles_df["category"].fillna("unknown").value_counts().to_dict()
    top_categories = dict(list(cat_counts.items())[:10])

    # Text word counts
    title_words = []
    abstract_words = []
    body_words = []

    for _, row in articles_df.iterrows():
        t = str(row.get("title") or "").strip()
        a = str(row.get("abstract") or "").strip()
        b = str(row.get("body") or "").strip()
        if t:
            title_words.append(len(t.split()))
        else:
            title_words.append(0)
        if a:
            abstract_words.append(len(a.split()))
        else:
            abstract_words.append(0)
        if b:
            body_words.append(len(b.split()))
        else:
            body_words.append(0)

    # Entities and Embeddings coverage
    entities_count = 0
    if "entities" in articles_df.columns:
        for ent in articles_df["entities"]:
            e_list = safe_to_list(ent)
            if len(e_list) > 0:
                entities_count += 1
    entities_pct = round(entities_count / total_articles * 100.0, 2) if total_articles > 0 else 0.0

    emb_count = 0
    emb_dim = None
    if "embedding" in articles_df.columns:
        for emb in articles_df["embedding"]:
            if emb is not None:
                try:
                    if hasattr(emb, "shape"):
                        emb_count += 1
                        if emb_dim is None and len(emb.shape) > 0:
                            emb_dim = int(emb.shape[-1])
                    elif isinstance(emb, (list, tuple)) and len(emb) > 0:
                        emb_count += 1
                        if emb_dim is None:
                            emb_dim = len(emb)
                except Exception:
                    pass
    emb_pct = round(emb_count / total_articles * 100.0, 2) if total_articles > 0 else 0.0

    # Candidate and Clicked articles coverage across splits
    all_catalog_ids = set(articles_df["article_id"].dropna().unique())
    split_candidate_sets = {}
    split_clicked_sets = {}

    for split in ["train", "val", "test"]:
        df = imp_dfs.get(split)
        cand_set = set()
        clk_set = set()
        if df is not None and not df.empty:
            for cand in df["candidate_article_ids"]:
                cand_set.update(safe_to_list(cand))
            for clk in df["clicked_article_ids"]:
                clk_set.update(safe_to_list(clk))
        split_candidate_sets[split] = cand_set
        split_clicked_sets[split] = clk_set

    train_cands = split_candidate_sets["train"]
    val_cands = split_candidate_sets["val"]
    test_cands = split_candidate_sets["test"]
    all_cands = train_cands.union(val_cands).union(test_cands)

    train_clks = split_clicked_sets["train"]
    val_clks = split_clicked_sets["val"]
    test_clks = split_clicked_sets["test"]
    all_clks = train_clks.union(val_clks).union(test_clks)

    val_new_cands = val_cands - train_cands
    test_new_cands = test_cands - (train_cands.union(val_cands))

    coverage_stats = {
        "catalog_total_articles": total_articles,
        "unique_categories_count": len(cat_counts),
        "top_10_categories": top_categories,
        "title_word_count": get_series_stats(title_words),
        "abstract_word_count": get_series_stats(abstract_words),
        "body_word_count": get_series_stats(body_words),
        "articles_with_entities_pct": entities_pct,
        "articles_with_embeddings_pct": emb_pct,
        "embedding_dimension": emb_dim,
        "split_candidate_coverage": {
            "train": {
                "unique_candidate_articles": len(train_cands),
                "unique_clicked_articles": len(train_clks),
                "catalog_coverage_pct": round(len(train_cands) / total_articles * 100.0, 2) if total_articles else 0.0
            },
            "val": {
                "unique_candidate_articles": len(val_cands),
                "unique_clicked_articles": len(val_clks),
                "catalog_coverage_pct": round(len(val_cands) / total_articles * 100.0, 2) if total_articles else 0.0,
                "unseen_candidates_vs_train_count": len(val_new_cands),
                "unseen_candidates_pct_of_val": round(len(val_new_cands) / len(val_cands) * 100.0, 2) if val_cands else 0.0
            },
            "test": {
                "unique_candidate_articles": len(test_cands),
                "unique_clicked_articles": len(test_clks),
                "catalog_coverage_pct": round(len(test_cands) / total_articles * 100.0, 2) if total_articles else 0.0,
                "unseen_candidates_vs_prior_count": len(test_new_cands),
                "unseen_candidates_pct_of_test": round(len(test_new_cands) / len(test_cands) * 100.0, 2) if test_cands else 0.0
            },
            "combined": {
                "total_unique_candidates": len(all_cands),
                "total_unique_clicked": len(all_clks),
                "total_catalog_coverage_pct": round(len(all_cands) / total_articles * 100.0, 2) if total_articles else 0.0
            }
        }
    }
    return coverage_stats


def analyze_dataset(ds_name: str, processed_dir: Path) -> Dict[str, Any]:
    """Run full analytics on a single dataset folder."""
    ds_path = processed_dir / ds_name
    if not ds_path.exists():
        raise FileNotFoundError(f"Processed dataset directory not found: {ds_path}")

    # 1. Load Parquet Files
    imp_dfs = {}
    hist_dfs = {}

    for split in ["train", "val", "test"]:
        imp_p = ds_path / f"impressions_{split}.parquet"
        hist_p = ds_path / f"history_{split}.parquet"

        imp_dfs[split] = pd.read_parquet(imp_p) if imp_p.exists() else None
        hist_dfs[split] = pd.read_parquet(hist_p) if hist_p.exists() else None

    articles_p = ds_path / "articles.parquet"
    articles_df = pd.read_parquet(articles_p) if articles_p.exists() else None

    # 2. Compute Temporal Stats
    temporal_stats = compute_temporal_stats(imp_dfs)

    # 3. Compute Impression & Click Stats per split
    imp_stats = {}
    combined_imp_list = [df for df in imp_dfs.values() if df is not None]
    combined_imp_df = pd.concat(combined_imp_list, ignore_index=True) if combined_imp_list else pd.DataFrame()

    for split in ["train", "val", "test"]:
        imp_stats[split] = analyze_split_impressions(imp_dfs.get(split))
    imp_stats["combined"] = analyze_split_impressions(combined_imp_df)

    # 4. Compute User History and Overlap Stats
    user_stats = analyze_dataset_history_and_users(imp_dfs, hist_dfs)

    # 5. Compute Article Catalog & Candidate Coverage
    article_stats = analyze_article_catalog_and_coverage(articles_df, imp_dfs)

    return {
        "dataset": ds_name.upper(),
        "temporal": temporal_stats,
        "impressions": imp_stats,
        "users_and_history": user_stats,
        "articles_and_catalog": article_stats,
    }


def format_markdown_report(stats_all: Dict[str, Any]) -> str:
    """Generate a clean, structured Markdown report comparing both datasets."""
    lines = []
    lines.append("# Processed Dataset Statistics Report (Train, Val, Test)")
    lines.append("")
    lines.append("This report summarizes dataset metrics across **Train**, **Validation**, and **Test** temporal splits for both **MIND** and **EB-NeRD** datasets from `data/processed`.")
    lines.append("")

    for ds_key, data in stats_all.items():
        ds_title = data["dataset"]
        t = data["temporal"]
        imp = data["impressions"]
        uh = data["users_and_history"]
        art = data["articles_and_catalog"]
        cov = art.get("split_candidate_coverage", {})
        u_overlap = uh.get("cross_split_user_overlap", {})

        lines.append(f"## {ds_title} Dataset Statistics")
        lines.append("")

        # Temporal Table
        lines.append("### 1. Temporal Splits & Boundaries")
        lines.append("")
        lines.append("| Metric | Train Split | Validation Split | Test Split | Combined |")
        lines.append("| :--- | :--- | :--- | :--- | :--- |")
        lines.append(f"| **Min Timestamp** | `{t['train']['min_time']}` | `{t['val']['min_time']}` | `{t['test']['min_time']}` | `{t['combined']['min_time']}` |")
        lines.append(f"| **Max Timestamp** | `{t['train']['max_time']}` | `{t['val']['max_time']}` | `{t['test']['max_time']}` | `{t['combined']['max_time']}` |")
        lines.append(f"| **Duration (Days)** | {t['train']['duration_days']} days | {t['val']['duration_days']} days | {t['test']['duration_days']} days | {t['combined']['duration_days']} days |")
        lines.append(f"| **Duration (Hours)** | {t['train']['duration_hours']}h | {t['val']['duration_hours']}h | {t['test']['duration_hours']}h | {t['combined']['duration_hours']}h |")
        lines.append("")
        lines.append(f"* **Train to Val Gap**: `{t['combined']['train_to_val_gap_hours']} hours`")
        lines.append(f"* **Val to Test Gap**: `{t['combined']['val_to_test_gap_hours']} hours`")
        lines.append("")

        # Impression Table
        lines.append("### 2. Impression & Click Dynamics")
        lines.append("")
        lines.append("| Metric | Train | Validation | Test | Total / Combined |")
        lines.append("| :--- | :--- | :--- | :--- | :--- |")
        lines.append(f"| **Impression Count** | {imp['train']['impression_count']:,} | {imp['val']['impression_count']:,} | {imp['test']['impression_count']:,} | {imp['combined']['impression_count']:,} |")
        lines.append(f"| **Unique Users (Imp)** | {imp['train']['unique_users']:,} | {imp['val']['unique_users']:,} | {imp['test']['unique_users']:,} | {u_overlap.get('total_unique_impression_users', 0):,} |")
        lines.append(f"| **Total Candidates** | {imp['train']['total_candidates']:,} | {imp['val']['total_candidates']:,} | {imp['test']['total_candidates']:,} | {imp['combined']['total_candidates']:,} |")
        lines.append(f"| **Total Clicks** | {imp['train']['total_clicks']:,} | {imp['val']['total_clicks']:,} | {imp['test']['total_clicks']:,} | {imp['combined']['total_clicks']:,} |")
        lines.append(f"| **Click-Through Rate (CTR)** | {imp['train']['ctr_percent']}% | {imp['val']['ctr_percent']}% | {imp['test']['ctr_percent']}% | {imp['combined']['ctr_percent']}% |")
        lines.append(f"| **Candidates / Imp (Mean +/- Std)** | {imp['train']['candidates_per_imp']['mean']} +/- {imp['train']['candidates_per_imp']['std']} | {imp['val']['candidates_per_imp']['mean']} +/- {imp['val']['candidates_per_imp']['std']} | {imp['test']['candidates_per_imp']['mean']} +/- {imp['test']['candidates_per_imp']['std']} | {imp['combined']['candidates_per_imp']['mean']} +/- {imp['combined']['candidates_per_imp']['std']} |")
        lines.append(f"| **Candidates / Imp (Median [Min-Max])** | {imp['train']['candidates_per_imp']['median']} [{imp['train']['candidates_per_imp']['min']}-{imp['train']['candidates_per_imp']['max']}] | {imp['val']['candidates_per_imp']['median']} [{imp['val']['candidates_per_imp']['min']}-{imp['val']['candidates_per_imp']['max']}] | {imp['test']['candidates_per_imp']['median']} [{imp['test']['candidates_per_imp']['min']}-{imp['test']['candidates_per_imp']['max']}] | {imp['combined']['candidates_per_imp']['median']} [{imp['combined']['candidates_per_imp']['min']}-{imp['combined']['candidates_per_imp']['max']}] |")
        lines.append(f"| **Clicks / Imp (Mean / Median)** | {imp['train']['clicks_per_imp']['mean']} / {imp['train']['clicks_per_imp']['median']} | {imp['val']['clicks_per_imp']['mean']} / {imp['val']['clicks_per_imp']['median']} | {imp['test']['clicks_per_imp']['mean']} / {imp['test']['clicks_per_imp']['median']} | {imp['combined']['clicks_per_imp']['mean']} / {imp['combined']['clicks_per_imp']['median']} |")
        lines.append(f"| **Imps / User (Mean / Median)** | {imp['train']['impressions_per_user']['mean']} / {imp['train']['impressions_per_user']['median']} | {imp['val']['impressions_per_user']['mean']} / {imp['val']['impressions_per_user']['median']} | {imp['test']['impressions_per_user']['mean']} / {imp['test']['impressions_per_user']['median']} | {imp['combined']['impressions_per_user']['mean']} / {imp['combined']['impressions_per_user']['median']} |")
        lines.append("")

        # User History & Overlap
        ps_uh = uh.get("per_split", {})
        lines.append("### 3. User History & Cross-Split Overlap")
        lines.append("")
        lines.append("| Metric | Train | Validation | Test |")
        lines.append("| :--- | :--- | :--- | :--- |")
        lines.append(f"| **History Click Events** | {ps_uh['train']['history_rows']:,} | {ps_uh['val']['history_rows']:,} | {ps_uh['test']['history_rows']:,} |")
        lines.append(f"| **Unique History Users** | {ps_uh['train']['unique_history_users']:,} | {ps_uh['val']['unique_history_users']:,} | {ps_uh['test']['unique_history_users']:,} |")
        lines.append(f"| **History Len / User (Mean / Median)** | {ps_uh['train']['history_length_per_user']['mean']} / {ps_uh['train']['history_length_per_user']['median']} | {ps_uh['val']['history_length_per_user']['mean']} / {ps_uh['val']['history_length_per_user']['median']} | {ps_uh['test']['history_length_per_user']['mean']} / {ps_uh['test']['history_length_per_user']['median']} |")
        lines.append(f"| **Cold-Start Users in Split (% of Users)** | {ps_uh['train']['cold_start_users_count']:,} ({ps_uh['train']['cold_start_users_percent']}%) | {ps_uh['val']['cold_start_users_count']:,} ({ps_uh['val']['cold_start_users_percent']}%) | {ps_uh['test']['cold_start_users_count']:,} ({ps_uh['test']['cold_start_users_percent']}%) |")
        lines.append("")
        lines.append("#### User Overlap across Splits:")
        lines.append(f"- **Train and Val Overlap**: {u_overlap.get('train_val_overlap_count', 0):,} users ({u_overlap.get('train_val_overlap_pct_of_val', 0)}% of Val)")
        lines.append(f"- **Val New (Cold) Users**: {u_overlap.get('val_new_users_count', 0):,} users ({u_overlap.get('val_new_users_pct_of_val', 0)}% of Val)")
        lines.append(f"- **Train and Test Overlap**: {u_overlap.get('train_test_overlap_count', 0):,} users ({u_overlap.get('train_test_overlap_pct_of_test', 0)}% of Test)")
        lines.append(f"- **Val and Test Overlap**: {u_overlap.get('val_test_overlap_count', 0):,} users ({u_overlap.get('val_test_overlap_pct_of_test', 0)}% of Test)")
        lines.append(f"- **Test New (Cold) Users**: {u_overlap.get('test_new_users_count', 0):,} users ({u_overlap.get('test_new_users_pct_of_test', 0)}% of Test)")
        lines.append(f"- **Users active across all 3 splits**: {u_overlap.get('users_in_all_splits_count', 0):,} users ({u_overlap.get('users_in_all_splits_pct', 0)}% of Total)")
        lines.append("")

        # Article Catalog & Candidates
        lines.append("### 4. Article Catalog & Candidate Coverage")
        lines.append("")
        lines.append(f"* **Total Catalog Articles**: `{art.get('catalog_total_articles', 0):,}`")
        lines.append(f"* **Unique Categories**: `{art.get('unique_categories_count', 0)}`")
        lines.append(f"* **Articles with Precomputed Embeddings**: `{art.get('articles_with_embeddings_pct', 0)}%` (dim={art.get('embedding_dimension')})")
        lines.append(f"* **Title Word Count (Mean / Median)**: `{art['title_word_count']['mean']} / {art['title_word_count']['median']}`")
        lines.append(f"* **Abstract Word Count (Mean / Median)**: `{art['abstract_word_count']['mean']} / {art['abstract_word_count']['median']}`")
        lines.append(f"* **Body Word Count (Mean / Median)**: `{art['body_word_count']['mean']} / {art['body_word_count']['median']}`")
        lines.append("")
        lines.append("| Split | Unique Candidates | Unique Clicked | Catalog Coverage % | Unseen Candidates vs Prior |")
        lines.append("| :--- | :--- | :--- | :--- | :--- |")
        lines.append(f"| **Train** | {cov['train']['unique_candidate_articles']:,} | {cov['train']['unique_clicked_articles']:,} | {cov['train']['catalog_coverage_pct']}% | - |")
        lines.append(f"| **Validation** | {cov['val']['unique_candidate_articles']:,} | {cov['val']['unique_clicked_articles']:,} | {cov['val']['catalog_coverage_pct']}% | {cov['val'].get('unseen_candidates_vs_train_count', 0):,} ({cov['val'].get('unseen_candidates_pct_of_val', 0)}%) |")
        lines.append(f"| **Test** | {cov['test']['unique_candidate_articles']:,} | {cov['test']['unique_clicked_articles']:,} | {cov['test']['catalog_coverage_pct']}% | {cov['test'].get('unseen_candidates_vs_prior_count', 0):,} ({cov['test'].get('unseen_candidates_pct_of_test', 0)}%) |")
        lines.append(f"| **Combined** | {cov['combined']['total_unique_candidates']:,} | {cov['combined']['total_unique_clicked']:,} | {cov['combined']['total_catalog_coverage_pct']}% | - |")
        lines.append("")
        lines.append("---")
        lines.append("")

    # Cross-Dataset Comparison Summary Table
    if len(stats_all) > 1:
        lines.append("## Cross-Dataset High-Level Comparison")
        lines.append("")
        lines.append("| Feature / Metric | EB-NeRD | MIND |")
        lines.append("| :--- | :--- | :--- |")
        eb = stats_all.get("ebnerd", {})
        md = stats_all.get("mind", {})

        lines.append(f"| **Language** | Danish | English |")
        lines.append(f"| **Catalog Articles** | {eb.get('articles_and_catalog', {}).get('catalog_total_articles', 0):,} | {md.get('articles_and_catalog', {}).get('catalog_total_articles', 0):,} |")
        lines.append(f"| **Total Impressions** | {eb.get('impressions', {}).get('combined', {}).get('impression_count', 0):,} | {md.get('impressions', {}).get('combined', {}).get('impression_count', 0):,} |")
        lines.append(f"| **Train / Val / Test Impressions** | {eb.get('impressions', {}).get('train', {}).get('impression_count', 0):,} / {eb.get('impressions', {}).get('val', {}).get('impression_count', 0):,} / {eb.get('impressions', {}).get('test', {}).get('impression_count', 0):,} | {md.get('impressions', {}).get('train', {}).get('impression_count', 0):,} / {md.get('impressions', {}).get('val', {}).get('impression_count', 0):,} / {md.get('impressions', {}).get('test', {}).get('impression_count', 0):,} |")
        lines.append(f"| **Total Unique Users** | {eb.get('users_and_history', {}).get('cross_split_user_overlap', {}).get('total_unique_impression_users', 0):,} | {md.get('users_and_history', {}).get('cross_split_user_overlap', {}).get('total_unique_impression_users', 0):,} |")
        lines.append(f"| **Overall CTR** | {eb.get('impressions', {}).get('combined', {}).get('ctr_percent', 0)}% | {md.get('impressions', {}).get('combined', {}).get('ctr_percent', 0)}% |")
        lines.append(f"| **Candidates / Imp (Mean)** | {eb.get('impressions', {}).get('combined', {}).get('candidates_per_imp', {}).get('mean', 0)} | {md.get('impressions', {}).get('combined', {}).get('candidates_per_imp', {}).get('mean', 0)} |")
        lines.append(f"| **Total Time Span (Days)** | {eb.get('temporal', {}).get('combined', {}).get('duration_days', 0)} days | {md.get('temporal', {}).get('combined', {}).get('duration_days', 0)} days |")
        lines.append(f"| **Title Word Count (Mean)** | {eb.get('articles_and_catalog', {}).get('title_word_count', {}).get('mean', 0)} words | {md.get('articles_and_catalog', {}).get('title_word_count', {}).get('mean', 0)} words |")
        lines.append(f"| **Abstract Word Count (Mean)** | {eb.get('articles_and_catalog', {}).get('abstract_word_count', {}).get('mean', 0)} words | {md.get('articles_and_catalog', {}).get('abstract_word_count', {}).get('mean', 0)} words |")
        lines.append(f"| **Body Word Count (Mean)** | {eb.get('articles_and_catalog', {}).get('body_word_count', {}).get('mean', 0)} words | {md.get('articles_and_catalog', {}).get('body_word_count', {}).get('mean', 0)} words |")
        lines.append("")

    return "\n".join(lines)


def print_cli_summary(stats_all: Dict[str, Any]) -> None:
    """Print readable ASCII tables to stdout."""
    print("=" * 80)
    print("        PROCESSED DATASET STATISTICS SUMMARY (TRAIN / VAL / TEST)")
    print("=" * 80)

    for ds_key, data in stats_all.items():
        ds_title = data["dataset"]
        t = data["temporal"]
        imp = data["impressions"]
        uh = data["users_and_history"]
        art = data["articles_and_catalog"]
        cov = art.get("split_candidate_coverage", {})
        u_overlap = uh.get("cross_split_user_overlap", {})

        print(f"\n{'#' * 30} [ {ds_title} DATASET ] {'#' * 30}")
        print(f"Total Catalog Articles: {art.get('catalog_total_articles', 0):,} | Total Unique Users: {u_overlap.get('total_unique_impression_users', 0):,}")
        print(f"Total Impressions:      {imp['combined']['impression_count']:,} | Overall CTR: {imp['combined']['ctr_percent']}%")
        print(f"Total Temporal Span:    {t['combined']['min_time']} -> {t['combined']['max_time']} ({t['combined']['duration_days']} days)")
        
        print("\n--- 1. TEMPORAL BOUNDARIES ---")
        print(f"{'Split':<12} | {'Min Timestamp':<20} | {'Max Timestamp':<20} | {'Duration (Days)':<15}")
        print("-" * 75)
        for sp in ["train", "val", "test"]:
            print(f"{sp.capitalize():<12} | {t[sp]['min_time']:<20} | {t[sp]['max_time']:<20} | {t[sp]['duration_days']:<15}")
        print(f"Train->Val Gap: {t['combined']['train_to_val_gap_hours']}h | Val->Test Gap: {t['combined']['val_to_test_gap_hours']}h")

        print("\n--- 2. IMPRESSIONS & CLICKS ---")
        print(f"{'Split':<10} | {'Impressions':<12} | {'Users':<10} | {'Candidates':<12} | {'Clicks':<10} | {'CTR (%)':<8} | {'Cands/Imp (Mean/Med)':<20}")
        print("-" * 92)
        for sp in ["train", "val", "test", "combined"]:
            s_name = sp.capitalize()
            c_mean = imp[sp]['candidates_per_imp']['mean']
            c_med = imp[sp]['candidates_per_imp']['median']
            cand_str = f"{c_mean} / {c_med}"
            print(f"{s_name:<10} | {imp[sp]['impression_count']:<12,d} | {imp[sp]['unique_users']:<10,d} | {imp[sp]['total_candidates']:<12,d} | {imp[sp]['total_clicks']:<10,d} | {imp[sp]['ctr_percent']:<8.2f} | {cand_str:<20}")

        print("\n--- 3. USER HISTORY & OVERLAPS ---")
        ps_uh = uh.get("per_split", {})
        print(f"{'Split':<10} | {'Hist Events':<14} | {'Hist Users':<12} | {'Hist Len (Mean/Med)':<20} | {'Cold-Start Users (%)':<20}")
        print("-" * 84)
        for sp in ["train", "val", "test"]:
            h_len_str = f"{ps_uh[sp]['history_length_per_user']['mean']} / {ps_uh[sp]['history_length_per_user']['median']}"
            cold_str = f"{ps_uh[sp]['cold_start_users_count']:,} ({ps_uh[sp]['cold_start_users_percent']}%)"
            print(f"{sp.capitalize():<10} | {ps_uh[sp]['history_rows']:<14,d} | {ps_uh[sp]['unique_history_users']:<12,d} | {h_len_str:<20} | {cold_str:<20}")
        print(f"Overlap: Train & Val = {u_overlap.get('train_val_overlap_count', 0):,} ({u_overlap.get('train_val_overlap_pct_of_val', 0)}% of Val) | Val New = {u_overlap.get('val_new_users_count', 0):,} ({u_overlap.get('val_new_users_pct_of_val', 0)}%)")
        print(f"Overlap: Train & Test = {u_overlap.get('train_test_overlap_count', 0):,} ({u_overlap.get('train_test_overlap_pct_of_test', 0)}% of Test) | Test New = {u_overlap.get('test_new_users_count', 0):,} ({u_overlap.get('test_new_users_pct_of_test', 0)}%)")
        print(f"Active in all 3 splits: {u_overlap.get('users_in_all_splits_count', 0):,} ({u_overlap.get('users_in_all_splits_pct', 0)}% of total)")

        print("\n--- 4. ARTICLE CATALOG & CANDIDATE COVERAGE ---")
        print(f"{'Split':<10} | {'Unique Cands':<14} | {'Unique Clicked':<14} | {'Catalog Coverage (%)':<22} | {'Unseen Candidates':<20}")
        print("-" * 88)
        for sp in ["train", "val", "test"]:
            unseen = cov[sp].get('unseen_candidates_vs_train_count', cov[sp].get('unseen_candidates_vs_prior_count', '-'))
            if unseen != '-':
                unseen = f"{unseen:,} ({cov[sp].get('unseen_candidates_pct_of_val', cov[sp].get('unseen_candidates_pct_of_test', 0))}%)"
            print(f"{sp.capitalize():<10} | {cov[sp]['unique_candidate_articles']:<14,d} | {cov[sp]['unique_clicked_articles']:<14,d} | {cov[sp]['catalog_coverage_pct']:<22.2f} | {unseen:<20}")
        print(f"Combined   | {cov['combined']['total_unique_candidates']:<14,d} | {cov['combined']['total_unique_clicked']:<14,d} | {cov['combined']['total_catalog_coverage_pct']:<22.2f} | -")

    print("\n" + "=" * 80 + "\n")


def main():
    args = parse_args()
    processed_dir = Path(args.processed_dir)

    datasets_to_run = []
    if args.dataset in ["all", "mind"]:
        datasets_to_run.append("mind")
    if args.dataset in ["all", "ebnerd"]:
        datasets_to_run.append("ebnerd")

    all_stats = {}
    for ds in datasets_to_run:
        if (processed_dir / ds).exists():
            if not args.quiet:
                print(f"Analyzing {ds} processed dataset...")
            all_stats[ds] = analyze_dataset(ds, processed_dir)
        else:
            print(f"Warning: {ds} directory not found at {processed_dir / ds}, skipping.")

    if not all_stats:
        print("No processed datasets found to analyze.")
        return

    # Print to console
    if not args.quiet:
        print_cli_summary(all_stats)

    # Save JSON
    if args.output_json:
        out_json_path = Path(args.output_json)
        out_json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(all_stats, f, indent=2)
        if not args.quiet:
            print(f"[OK] Saved statistics JSON to: {out_json_path}")

    # Save Markdown Report
    if args.output_md:
        out_md_path = Path(args.output_md)
        out_md_path.parent.mkdir(parents=True, exist_ok=True)
        md_content = format_markdown_report(all_stats)
        with open(out_md_path, "w", encoding="utf-8") as f:
            f.write(md_content)
        if not args.quiet:
            print(f"[OK] Saved statistics Markdown report to: {out_md_path}")


if __name__ == "__main__":
    main()
