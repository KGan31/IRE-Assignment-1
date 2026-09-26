"""
EB-NeRD Dataset Statistics Analyzer
===================================
This script reads the EB-NeRD (Ekstra Bladet News Recommendation Dataset) train and validation files
(behaviors.parquet, history.parquet, and articles.parquet) and computes key dataset statistics across time,
users, impressions, and articles.

Usage:
    python ebnerd_stats.py
    python ebnerd_stats.py --train_dir data/raw/ebnerd/demo/train --val_dir data/raw/ebnerd/demo/validation --articles_file data/raw/ebnerd/demo/articles.parquet
    python ebnerd_stats.py --output_md ebnerd_stats_report.md --output_json ebnerd_stats.json
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Set, Tuple
import pandas as pd
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Compute statistics for EB-NeRD Train and Validation datasets.")
    parser.add_argument(
        "--train_dir",
        type=str,
        default="data/raw/ebnerd/demo/train",
        help="Path to EB-NeRD train folder containing behaviors.parquet and history.parquet"
    )
    parser.add_argument(
        "--val_dir",
        type=str,
        default="data/raw/ebnerd/demo/validation",
        help="Path to EB-NeRD validation folder containing behaviors.parquet and history.parquet"
    )
    parser.add_argument(
        "--articles_file",
        type=str,
        default="data/raw/ebnerd/demo/articles.parquet",
        help="Path to EB-NeRD articles.parquet file"
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Optional path to save JSON statistics output"
    )
    parser.add_argument(
        "--output_md",
        type=str,
        default=None,
        help="Optional path to save Markdown report output"
    )
    return parser.parse_args()


def load_behaviors(parquet_path: Path) -> pd.DataFrame:
    """Load behaviors parquet file."""
    if not parquet_path.exists():
        raise FileNotFoundError(f"Behaviors file not found at: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    df["timestamp"] = pd.to_datetime(df["impression_time"])
    return df


def load_history(parquet_path: Path) -> pd.DataFrame:
    """Load history parquet file."""
    if not parquet_path.exists():
        raise FileNotFoundError(f"History file not found at: {parquet_path}")
    return pd.read_parquet(parquet_path)


def load_articles(parquet_path: Path) -> pd.DataFrame:
    """Load articles parquet file."""
    if not parquet_path.exists():
        raise FileNotFoundError(f"Articles file not found at: {parquet_path}")
    return pd.read_parquet(parquet_path)


def safe_to_list(val) -> list:
    """Safely convert array/list/None to Python list."""
    if val is None:
        return []
    try:
        if len(val) == 0:
            return []
    except TypeError:
        return []
    return list(val)


def get_series_stats(data: List[int]) -> Dict[str, float]:
    """Compute summary statistics for a numeric list."""
    if not data:
        return {"min": 0, "max": 0, "mean": 0.0, "median": 0.0, "std": 0.0}
    arr = np.array(data)
    return {
        "min": int(np.min(arr)),
        "max": int(np.max(arr)),
        "mean": round(float(np.mean(arr)), 2),
        "median": round(float(np.median(arr)), 2),
        "std": round(float(np.std(arr)), 2)
    }


def analyze_time_stats(train_b: pd.DataFrame, val_b: pd.DataFrame) -> Dict[str, Any]:
    """Calculate temporal statistics across train and validation splits."""
    train_min, train_max = train_b["timestamp"].min(), train_b["timestamp"].max()
    val_min, val_max = val_b["timestamp"].min(), val_b["timestamp"].max()
    
    combined_min = min(train_min, val_min)
    combined_max = max(train_max, val_max)
    
    return {
        "train": {
            "min_time": str(train_min),
            "max_time": str(train_max),
            "duration_days": round((train_max - train_min).total_seconds() / 86400, 2)
        },
        "validation": {
            "min_time": str(val_min),
            "max_time": str(val_max),
            "duration_days": round((val_max - val_min).total_seconds() / 86400, 2)
        },
        "combined": {
            "min_time": str(combined_min),
            "max_time": str(combined_max),
            "duration_days": round((combined_max - combined_min).total_seconds() / 86400, 2),
            "gap_hours": round((val_min - train_max).total_seconds() / 3600, 4)
        }
    }


def process_ebnerd_impressions(df: pd.DataFrame) -> Tuple[List[int], List[int]]:
    """Extract candidate (inview) and clicked article counts from EB-NeRD behaviors."""
    candidate_counts = []
    click_counts = []
    
    for row in df.itertuples(index=False):
        inview = safe_to_list(getattr(row, "article_ids_inview", None))
        clicked = safe_to_list(getattr(row, "article_ids_clicked", None))
        
        candidate_counts.append(len(inview))
        click_counts.append(len(clicked))
        
    return candidate_counts, click_counts


def process_ebnerd_history(df: pd.DataFrame) -> Tuple[Dict[str, int], List[int]]:
    """Extract click history lengths per user from history dataframe."""
    user_hist_map = {}
    history_lengths = []
    
    for row in df.itertuples(index=False):
        user_id = str(getattr(row, "user_id"))
        fixed_articles = safe_to_list(getattr(row, "article_id_fixed", None))
        length = len(fixed_articles)
        user_hist_map[user_id] = length
        history_lengths.append(length)
        
    return user_hist_map, history_lengths


def analyze_behaviors_stats(
    train_b: pd.DataFrame, val_b: pd.DataFrame,
    train_h: pd.DataFrame, val_h: pd.DataFrame
) -> Dict[str, Any]:
    """Calculate detailed impression and user behavior statistics for EB-NeRD."""
    train_imp_cnt = len(train_b)
    val_imp_cnt = len(val_b)
    
    # Candidates and clicks
    train_cands, train_clicks = process_ebnerd_impressions(train_b)
    val_cands, val_clicks = process_ebnerd_impressions(val_b)
    
    # History
    _, train_hist_lens = process_ebnerd_history(train_h)
    _, val_hist_lens = process_ebnerd_history(val_h)
    
    # Users
    train_users = set(train_b["user_id"])
    val_users = set(val_b["user_id"])
    combined_users = train_users.union(val_users)
    user_overlap = train_users.intersection(val_users)
    cold_start_val_users = val_users - train_users

    # User impression frequency
    train_user_freq = list(train_b["user_id"].value_counts().values)
    val_user_freq = list(val_b["user_id"].value_counts().values)
    
    # Device distribution
    combined_b = pd.concat([train_b, val_b])
    device_dist = combined_b["device_type"].value_counts().to_dict() if "device_type" in combined_b.columns else {}
    
    # Subscriber distribution
    sub_dist = combined_b["is_subscriber"].value_counts().to_dict() if "is_subscriber" in combined_b.columns else {}
    
    return {
        "impressions_count": {
            "train": train_imp_cnt,
            "validation": val_imp_cnt,
            "total": train_imp_cnt + val_imp_cnt
        },
        "users": {
            "train_unique": len(train_users),
            "val_unique": len(val_users),
            "total_unique": len(combined_users),
            "overlap_count": len(user_overlap),
            "overlap_pct_of_val": round(len(user_overlap) / len(val_users) * 100, 2) if val_users else 0,
            "cold_start_val_users": len(cold_start_val_users),
            "cold_start_pct_of_val": round(len(cold_start_val_users) / len(val_users) * 100, 2) if val_users else 0,
        },
        "candidates_per_impression": {
            "train": get_series_stats(train_cands),
            "validation": get_series_stats(val_cands),
            "combined": get_series_stats(train_cands + val_cands)
        },
        "clicks_per_impression": {
            "train": get_series_stats(train_clicks),
            "validation": get_series_stats(val_clicks),
            "combined": get_series_stats(train_clicks + val_clicks)
        },
        "click_through_rate": {
            "train_ctr": round(sum(train_clicks) / sum(train_cands) * 100, 2) if sum(train_cands) > 0 else 0,
            "val_ctr": round(sum(val_clicks) / sum(val_cands) * 100, 2) if sum(val_cands) > 0 else 0,
            "overall_ctr": round((sum(train_clicks) + sum(val_clicks)) / (sum(train_cands) + sum(val_cands)) * 100, 2) if (sum(train_cands) + sum(val_cands)) > 0 else 0
        },
        "history_length": {
            "train": get_series_stats(train_hist_lens),
            "validation": get_series_stats(val_hist_lens),
            "combined": get_series_stats(train_hist_lens + val_hist_lens),
            "train_zero_history_pct": round(sum(1 for x in train_hist_lens if x == 0) / len(train_hist_lens) * 100, 2) if train_hist_lens else 0,
            "val_zero_history_pct": round(sum(1 for x in val_hist_lens if x == 0) / len(val_hist_lens) * 100, 2) if val_hist_lens else 0
        },
        "user_impression_frequency": {
            "train": get_series_stats(train_user_freq),
            "validation": get_series_stats(val_user_freq)
        },
        "device_types": {str(k): int(v) for k, v in device_dist.items()},
        "is_subscriber": {str(k): int(v) for k, v in sub_dist.items()}
    }


def analyze_articles_stats(articles: pd.DataFrame) -> Dict[str, Any]:
    """Calculate EB-NeRD article catalog statistics."""
    total_articles = len(articles)
    
    # Categories
    category_col = "category_str" if "category_str" in articles.columns else "category"
    categories = articles[category_col].dropna() if category_col in articles.columns else pd.Series()
    top_categories = categories.value_counts().head(5).to_dict() if not categories.empty else {}
    
    # Text word counts
    title_lens = [len(str(t).split()) for t in articles["title"].dropna()] if "title" in articles.columns else []
    subtitle_lens = [len(str(s).split()) for s in articles["subtitle"].dropna()] if "subtitle" in articles.columns else []
    body_lens = [len(str(b).split()) for b in articles["body"].dropna()] if "body" in articles.columns else []
    
    # Published time
    pub_times = pd.to_datetime(articles["published_time"], errors="coerce").dropna()
    pub_min = str(pub_times.min()) if not pub_times.empty else "N/A"
    pub_max = str(pub_times.max()) if not pub_times.empty else "N/A"
    
    # Premium articles
    premium_cnt = int(articles["premium"].sum()) if "premium" in articles.columns else 0
    
    # Sentiment distribution
    sentiment_dist = articles["sentiment_label"].value_counts().to_dict() if "sentiment_label" in articles.columns else {}
    
    return {
        "article_count": total_articles,
        "published_time_range": {
            "min_published": pub_min,
            "max_published": pub_max
        },
        "categories": {
            "unique_categories": int(categories.nunique()) if not categories.empty else 0,
            "top_categories": {str(k): int(v) for k, v in top_categories.items()}
        },
        "title_word_count": get_series_stats(title_lens),
        "subtitle_word_count": get_series_stats(subtitle_lens),
        "body_word_count": get_series_stats(body_lens),
        "premium_articles": {
            "count": premium_cnt,
            "percentage": round(premium_cnt / total_articles * 100, 2) if total_articles > 0 else 0
        },
        "sentiment_distribution": {str(k): int(v) for k, v in sentiment_dist.items()}
    }


def print_summary(stats: Dict[str, Any]):
    """Print formatted statistics summary to console."""
    print("=" * 75)
    print("                    EB-NeRD DATASET STATISTICS REPORT                   ")
    print("=" * 75)
    
    # Time Stats
    time_s = stats["time"]
    print("\n1. TEMPORAL STATISTICS (IMPRESSION TIMESTAMPS)")
    print("-" * 75)
    print(f" Train Split Min Time : {time_s['train']['min_time']}")
    print(f" Train Split Max Time : {time_s['train']['max_time']}")
    print(f" Train Span           : {time_s['train']['duration_days']} days")
    print()
    print(f" Val Split Min Time   : {time_s['validation']['min_time']}")
    print(f" Val Split Max Time   : {time_s['validation']['max_time']}")
    print(f" Val Span             : {time_s['validation']['duration_days']} days")
    print()
    print(f" Combined Min Time    : {time_s['combined']['min_time']}")
    print(f" Combined Max Time    : {time_s['combined']['max_time']}")
    print(f" Total Span           : {time_s['combined']['duration_days']} days")
    print(f" Train -> Val Gap     : {time_s['combined']['gap_hours']} hours")
    
    # Impression & User Stats
    beh_s = stats["behaviors"]
    print("\n2. IMPRESSION & USER BEHAVIOR STATISTICS")
    print("-" * 75)
    print(f" Total Impressions    : Train = {beh_s['impressions_count']['train']:,} | Val = {beh_s['impressions_count']['validation']:,} | Total = {beh_s['impressions_count']['total']:,}")
    print(f" Unique Users         : Train = {beh_s['users']['train_unique']:,} | Val = {beh_s['users']['val_unique']:,} | Total = {beh_s['users']['total_unique']:,}")
    print(f" User Overlap         : {beh_s['users']['overlap_count']:,} users ({beh_s['users']['overlap_pct_of_val']}% of Val users)")
    print(f" Cold-Start Val Users : {beh_s['users']['cold_start_val_users']:,} users ({beh_s['users']['cold_start_pct_of_val']}% of Val users)")
    print(f" Overall CTR          : Train = {beh_s['click_through_rate']['train_ctr']}% | Val = {beh_s['click_through_rate']['val_ctr']}% | Overall = {beh_s['click_through_rate']['overall_ctr']}%")
    
    # Candidates per impression
    cands = beh_s["candidates_per_impression"]["combined"]
    print(f" Candidates/Impression: Min={cands['min']}, Max={cands['max']}, Mean={cands['mean']}, Median={cands['median']}, Std={cands['std']}")
    
    # Clicks per impression
    clicks = beh_s["clicks_per_impression"]["combined"]
    print(f" Clicks/Impression     : Min={clicks['min']}, Max={clicks['max']}, Mean={clicks['mean']}, Median={clicks['median']}, Std={clicks['std']}")
    
    # History length
    hist = beh_s["history_length"]["combined"]
    print(f" User History Length  : Min={hist['min']}, Max={hist['max']}, Mean={hist['mean']}, Median={hist['median']}, Std={hist['std']}")
    print(f" Empty History %      : Train = {beh_s['history_length']['train_zero_history_pct']}% | Val = {beh_s['history_length']['val_zero_history_pct']}%")
    print(f" Device Distribution  : {beh_s['device_types']}")
    print(f" Subscriber Status    : {beh_s['is_subscriber']}")
    
    # Article Stats
    art_s = stats["articles"]
    print("\n3. NEWS ARTICLE CATALOG STATISTICS")
    print("-" * 75)
    print(f" Total Articles       : {art_s['article_count']:,}")
    print(f" Published Time Range : Min = {art_s['published_time_range']['min_published']} | Max = {art_s['published_time_range']['max_published']}")
    print(f" Unique Categories    : {art_s['categories']['unique_categories']}")
    print(f" Top 5 Categories     : {art_s['categories']['top_categories']}")
    print(f" Premium Articles     : {art_s['premium_articles']['count']:,} ({art_s['premium_articles']['percentage']}%)")
    print(f" Sentiment Breakdown  : {art_s['sentiment_distribution']}")
    
    title_w = art_s["title_word_count"]
    print(f" Title Word Count     : Min={title_w['min']}, Max={title_w['max']}, Mean={title_w['mean']}, Median={title_w['median']}")
    
    sub_w = art_s["subtitle_word_count"]
    print(f" Subtitle Word Count  : Min={sub_w['min']}, Max={sub_w['max']}, Mean={sub_w['mean']}, Median={sub_w['median']}")
    
    body_w = art_s["body_word_count"]
    print(f" Body Word Count      : Min={body_w['min']}, Max={body_w['max']}, Mean={body_w['mean']}, Median={body_w['median']}")
    print("=" * 75)


def generate_markdown_report(stats: Dict[str, Any]) -> str:
    """Generate Markdown formatted string of EB-NeRD dataset statistics."""
    time_s = stats["time"]
    beh_s = stats["behaviors"]
    art_s = stats["articles"]
    
    md = f"""# EB-NeRD Dataset Statistics Report

## 1. Temporal Statistics (Timestamps)

| Metric | Train Split | Validation Split | Combined |
| :--- | :--- | :--- | :--- |
| **Minimum Timestamp** | `{time_s['train']['min_time']}` | `{time_s['validation']['min_time']}` | `{time_s['combined']['min_time']}` |
| **Maximum Timestamp** | `{time_s['train']['max_time']}` | `{time_s['validation']['max_time']}` | `{time_s['combined']['max_time']}` |
| **Duration (Days)** | {time_s['train']['duration_days']} days | {time_s['validation']['duration_days']} days | {time_s['combined']['duration_days']} days |

* **Train to Validation Gap**: {time_s['combined']['gap_hours']} hours

## 2. Impression & User Behavior Statistics

| Metric | Train | Validation | Total / Combined |
| :--- | :--- | :--- | :--- |
| **Impression Count** | {beh_s['impressions_count']['train']:,} | {beh_s['impressions_count']['validation']:,} | {beh_s['impressions_count']['total']:,} |
| **Unique Users** | {beh_s['users']['train_unique']:,} | {beh_s['users']['val_unique']:,} | {beh_s['users']['total_unique']:,} |
| **Click-Through-Rate (CTR)** | {beh_s['click_through_rate']['train_ctr']}% | {beh_s['click_through_rate']['val_ctr']}% | {beh_s['click_through_rate']['overall_ctr']}% |
| **Empty History %** | {beh_s['history_length']['train_zero_history_pct']}% | {beh_s['history_length']['val_zero_history_pct']}% | - |

* **User Overlap**: {beh_s['users']['overlap_count']:,} users ({beh_s['users']['overlap_pct_of_val']}% of Validation users)
* **Cold-Start Validation Users**: {beh_s['users']['cold_start_val_users']:,} users ({beh_s['users']['cold_start_pct_of_val']}% of Validation users)

### Interaction Summary Statistics

| Statistic | Candidates / Imp | Clicks / Imp | User History Length | User Imp Frequency |
| :--- | :--- | :--- | :--- | :--- |
| **Min** | {beh_s['candidates_per_impression']['combined']['min']} | {beh_s['clicks_per_impression']['combined']['min']} | {beh_s['history_length']['combined']['min']} | {beh_s['user_impression_frequency']['train']['min']} |
| **Max** | {beh_s['candidates_per_impression']['combined']['max']} | {beh_s['clicks_per_impression']['combined']['max']} | {beh_s['history_length']['combined']['max']} | {beh_s['user_impression_frequency']['train']['max']} |
| **Mean** | {beh_s['candidates_per_impression']['combined']['mean']} | {beh_s['clicks_per_impression']['combined']['mean']} | {beh_s['history_length']['combined']['mean']} | {beh_s['user_impression_frequency']['train']['mean']} |
| **Median** | {beh_s['candidates_per_impression']['combined']['median']} | {beh_s['clicks_per_impression']['combined']['median']} | {beh_s['history_length']['combined']['median']} | {beh_s['user_impression_frequency']['train']['median']} |

* **Device Types Breakdown**: `{beh_s['device_types']}`
* **Subscriber Breakdown**: `{beh_s['is_subscriber']}`

## 3. News Article Catalog Statistics

| Metric | Value |
| :--- | :--- |
| **Total Articles** | {art_s['article_count']:,} |
| **Published Time Min** | `{art_s['published_time_range']['min_published']}` |
| **Published Time Max** | `{art_s['published_time_range']['max_published']}` |
| **Unique Categories** | {art_s['categories']['unique_categories']} |
| **Premium Articles** | {art_s['premium_articles']['count']:,} ({art_s['premium_articles']['percentage']}%) |

* **Top 5 Categories**: `{art_s['categories']['top_categories']}`
* **Title Word Count (Mean / Median)**: {art_s['title_word_count']['mean']} / {art_s['title_word_count']['median']}
* **Subtitle Word Count (Mean / Median)**: {art_s['subtitle_word_count']['mean']} / {art_s['subtitle_word_count']['median']}
* **Body Word Count (Mean / Median)**: {art_s['body_word_count']['mean']} / {art_s['body_word_count']['median']}
* **Sentiment Breakdown**: `{art_s['sentiment_distribution']}`
"""
    return md


def main():
    args = parse_args()
    train_dir = Path(args.train_dir)
    val_dir = Path(args.val_dir)
    articles_file = Path(args.articles_file)
    
    print(f"Loading EB-NeRD train dataset from: {train_dir}")
    train_b = load_behaviors(train_dir / "behaviors.parquet")
    train_h = load_history(train_dir / "history.parquet")
    
    print(f"Loading EB-NeRD validation dataset from: {val_dir}")
    val_b = load_behaviors(val_dir / "behaviors.parquet")
    val_h = load_history(val_dir / "history.parquet")
    
    print(f"Loading EB-NeRD articles catalog from: {articles_file}")
    articles = load_articles(articles_file)
    
    print("Computing dataset statistics...")
    time_stats = analyze_time_stats(train_b, val_b)
    behaviors_stats = analyze_behaviors_stats(train_b, val_b, train_h, val_h)
    articles_stats = analyze_articles_stats(articles)
    
    stats = {
        "time": time_stats,
        "behaviors": behaviors_stats,
        "articles": articles_stats
    }
    
    print_summary(stats)
    
    if args.output_json:
        out_json_path = Path(args.output_json)
        out_json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        print(f"\nSaved JSON statistics to: {out_json_path}")
        
    if args.output_md:
        out_md_path = Path(args.output_md)
        out_md_path.parent.mkdir(parents=True, exist_ok=True)
        md_content = generate_markdown_report(stats)
        with open(out_md_path, "w", encoding="utf-8") as f:
            f.write(md_content)
        print(f"Saved Markdown report to: {out_md_path}")


if __name__ == "__main__":
    main()
