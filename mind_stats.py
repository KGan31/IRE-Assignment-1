"""
MIND Dataset Statistics Analyzer
================================
This script reads the MIND (Microsoft News Dataset) train and dev files (behaviors.tsv and news.tsv)
and computes key dataset statistics across time, users, impressions, and articles.

Usage:
    python mind_stats.py
    python mind_stats.py --train_dir data/raw/mind/train/MINDsmall_train --dev_dir data/raw/mind/dev/MINDsmall_dev
    python mind_stats.py --output_md mind_stats_report.md --output_json mind_stats.json
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Set, Tuple
import pandas as pd
import numpy as np


NEWS_COLS = [
    "news_id", "category", "subcategory", "title", "abstract",
    "url", "title_entities", "abstract_entities"
]
BEHAVIOR_COLS = ["impression_id", "user_id", "time", "history", "impressions"]


def parse_args():
    parser = argparse.ArgumentParser(description="Compute statistics for MIND Train and Dev datasets.")
    parser.add_argument(
        "--train_dir",
        type=str,
        default="data/raw/mind/train/MINDsmall_train",
        help="Path to MIND train folder containing behaviors.tsv and news.tsv"
    )
    parser.add_argument(
        "--dev_dir",
        type=str,
        default="data/raw/mind/dev/MINDsmall_dev",
        help="Path to MIND dev folder containing behaviors.tsv and news.tsv"
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


def load_behaviors(tsv_path: Path) -> pd.DataFrame:
    """Load and preprocess behaviors.tsv"""
    if not tsv_path.exists():
        raise FileNotFoundError(f"Behaviors file not found at: {tsv_path}")
    
    df = pd.read_csv(
        tsv_path,
        sep="\t",
        header=None,
        names=BEHAVIOR_COLS,
        quoting=3
    )
    df["timestamp"] = pd.to_datetime(df["time"], format="%m/%d/%Y %I:%M:%S %p")
    return df


def load_news(tsv_path: Path) -> pd.DataFrame:
    """Load and preprocess news.tsv"""
    if not tsv_path.exists():
        raise FileNotFoundError(f"News file not found at: {tsv_path}")
    
    df = pd.read_csv(
        tsv_path,
        sep="\t",
        header=None,
        names=NEWS_COLS,
        quoting=3
    )
    return df


def analyze_time_stats(train_b: pd.DataFrame, dev_b: pd.DataFrame) -> Dict[str, Any]:
    """Calculate temporal statistics across splits."""
    train_min, train_max = train_b["timestamp"].min(), train_b["timestamp"].max()
    dev_min, dev_max = dev_b["timestamp"].min(), dev_b["timestamp"].max()
    
    combined_min = min(train_min, dev_min)
    combined_max = max(train_max, dev_max)
    
    return {
        "train": {
            "min_time": str(train_min),
            "max_time": str(train_max),
            "duration_days": round((train_max - train_min).total_seconds() / 86400, 2)
        },
        "dev": {
            "min_time": str(dev_min),
            "max_time": str(dev_max),
            "duration_days": round((dev_max - dev_min).total_seconds() / 86400, 2)
        },
        "combined": {
            "min_time": str(combined_min),
            "max_time": str(combined_max),
            "duration_days": round((combined_max - combined_min).total_seconds() / 86400, 2),
            "gap_hours": round((dev_min - train_max).total_seconds() / 3600, 2)
        }
    }


def process_impressions_column(impressions_series: pd.Series) -> Tuple[List[int], List[int]]:
    """Parse impression strings to count candidates and clicked articles."""
    candidate_counts = []
    click_counts = []
    
    for imp_str in impressions_series.dropna():
        tokens = imp_str.split()
        candidate_counts.append(len(tokens))
        clicks = sum(1 for tok in tokens if tok.endswith("-1"))
        click_counts.append(clicks)
        
    return candidate_counts, click_counts


def process_history_column(history_series: pd.Series) -> List[int]:
    """Parse user history strings to count articles in click history."""
    history_lengths = []
    for hist_str in history_series:
        if isinstance(hist_str, str) and hist_str.strip():
            history_lengths.append(len(hist_str.split()))
        else:
            history_lengths.append(0)
    return history_lengths


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


def analyze_behaviors_stats(train_b: pd.DataFrame, dev_b: pd.DataFrame) -> Dict[str, Any]:
    """Calculate detailed impression and user behavior statistics."""
    # Impression counts
    train_imp_cnt = len(train_b)
    dev_imp_cnt = len(dev_b)
    
    # Candidates and clicks
    train_cands, train_clicks = process_impressions_column(train_b["impressions"])
    dev_cands, dev_clicks = process_impressions_column(dev_b["impressions"])
    
    # History lengths
    train_hist_lens = process_history_column(train_b["history"])
    dev_hist_lens = process_history_column(dev_b["history"])
    
    # Users
    train_users = set(train_b["user_id"])
    dev_users = set(dev_b["user_id"])
    combined_users = train_users.union(dev_users)
    user_overlap = train_users.intersection(dev_users)
    cold_start_dev_users = dev_users - train_users

    # User impression frequency
    train_user_freq = list(train_b["user_id"].value_counts().values)
    dev_user_freq = list(dev_b["user_id"].value_counts().values)
    
    return {
        "impressions_count": {
            "train": train_imp_cnt,
            "dev": dev_imp_cnt,
            "total": train_imp_cnt + dev_imp_cnt
        },
        "users": {
            "train_unique": len(train_users),
            "dev_unique": len(dev_users),
            "total_unique": len(combined_users),
            "overlap_count": len(user_overlap),
            "overlap_pct_of_dev": round(len(user_overlap) / len(dev_users) * 100, 2) if dev_users else 0,
            "cold_start_dev_users": len(cold_start_dev_users),
            "cold_start_pct_of_dev": round(len(cold_start_dev_users) / len(dev_users) * 100, 2) if dev_users else 0,
        },
        "candidates_per_impression": {
            "train": get_series_stats(train_cands),
            "dev": get_series_stats(dev_cands),
            "combined": get_series_stats(train_cands + dev_cands)
        },
        "clicks_per_impression": {
            "train": get_series_stats(train_clicks),
            "dev": get_series_stats(dev_clicks),
            "combined": get_series_stats(train_clicks + dev_clicks)
        },
        "click_through_rate": {
            "train_ctr": round(sum(train_clicks) / sum(train_cands) * 100, 2) if sum(train_cands) > 0 else 0,
            "dev_ctr": round(sum(dev_clicks) / sum(dev_cands) * 100, 2) if sum(dev_cands) > 0 else 0,
            "overall_ctr": round((sum(train_clicks) + sum(dev_clicks)) / (sum(train_cands) + sum(dev_cands)) * 100, 2) if (sum(train_cands) + sum(dev_cands)) > 0 else 0
        },
        "history_length": {
            "train": get_series_stats(train_hist_lens),
            "dev": get_series_stats(dev_hist_lens),
            "combined": get_series_stats(train_hist_lens + dev_hist_lens),
            "train_zero_history_pct": round(sum(1 for x in train_hist_lens if x == 0) / len(train_hist_lens) * 100, 2),
            "dev_zero_history_pct": round(sum(1 for x in dev_hist_lens if x == 0) / len(dev_hist_lens) * 100, 2)
        },
        "user_impression_frequency": {
            "train": get_series_stats(train_user_freq),
            "dev": get_series_stats(dev_user_freq)
        }
    }


def analyze_news_stats(train_n: pd.DataFrame, dev_n: pd.DataFrame) -> Dict[str, Any]:
    """Calculate news article catalog statistics."""
    train_news_ids = set(train_n["news_id"])
    dev_news_ids = set(dev_n["news_id"])
    combined_news_ids = train_news_ids.union(dev_news_ids)
    overlap_news_ids = train_news_ids.intersection(dev_news_ids)
    
    # Categories
    all_categories = pd.concat([train_n["category"], dev_n["category"]]).dropna()
    all_subcategories = pd.concat([train_n["subcategory"], dev_n["subcategory"]]).dropna()
    top_categories = all_categories.value_counts().head(5).to_dict()
    
    # Text length stats
    train_title_lens = [len(str(t).split()) for t in train_n["title"]]
    dev_title_lens = [len(str(t).split()) for t in dev_n["title"]]
    
    train_abs_lens = [len(str(a).split()) for a in train_n["abstract"] if pd.notna(a)]
    dev_abs_lens = [len(str(a).split()) for a in dev_n["abstract"] if pd.notna(a)]
    
    return {
        "article_counts": {
            "train_unique": len(train_news_ids),
            "dev_unique": len(dev_news_ids),
            "total_unique": len(combined_news_ids),
            "overlap_count": len(overlap_news_ids),
            "overlap_pct_of_dev": round(len(overlap_news_ids) / len(dev_news_ids) * 100, 2) if dev_news_ids else 0
        },
        "categories": {
            "unique_categories": int(all_categories.nunique()),
            "unique_subcategories": int(all_subcategories.nunique()),
            "top_categories": top_categories
        },
        "title_word_count": {
            "train": get_series_stats(train_title_lens),
            "dev": get_series_stats(dev_title_lens),
            "combined": get_series_stats(train_title_lens + dev_title_lens)
        },
        "abstract_word_count": {
            "train": get_series_stats(train_abs_lens),
            "dev": get_series_stats(dev_abs_lens),
            "combined": get_series_stats(train_abs_lens + dev_abs_lens)
        }
    }


def print_summary(stats: Dict[str, Any]):
    """Print formatted statistics summary to console."""
    print("=" * 75)
    print("                      MIND DATASET STATISTICS REPORT                     ")
    print("=" * 75)
    
    # Time Stats
    time_s = stats["time"]
    print("\n1. TEMPORAL STATISTICS (IMPRESSION TIMESTAMPS)")
    print("-" * 75)
    print(f" Train Split Min Time : {time_s['train']['min_time']}")
    print(f" Train Split Max Time : {time_s['train']['max_time']}")
    print(f" Train Span           : {time_s['train']['duration_days']} days")
    print()
    print(f" Dev Split Min Time   : {time_s['dev']['min_time']}")
    print(f" Dev Split Max Time   : {time_s['dev']['max_time']}")
    print(f" Dev Span             : {time_s['dev']['duration_days']} days")
    print()
    print(f" Combined Min Time    : {time_s['combined']['min_time']}")
    print(f" Combined Max Time    : {time_s['combined']['max_time']}")
    print(f" Total Span           : {time_s['combined']['duration_days']} days")
    print(f" Train -> Dev Gap     : {time_s['combined']['gap_hours']} hours")
    
    # Impression & User Stats
    beh_s = stats["behaviors"]
    print("\n2. IMPRESSION & USER BEHAVIOR STATISTICS")
    print("-" * 75)
    print(f" Total Impressions    : Train = {beh_s['impressions_count']['train']:,} | Dev = {beh_s['impressions_count']['dev']:,} | Total = {beh_s['impressions_count']['total']:,}")
    print(f" Unique Users         : Train = {beh_s['users']['train_unique']:,} | Dev = {beh_s['users']['dev_unique']:,} | Total = {beh_s['users']['total_unique']:,}")
    print(f" User Overlap         : {beh_s['users']['overlap_count']:,} users ({beh_s['users']['overlap_pct_of_dev']}% of Dev users)")
    print(f" Cold-Start Dev Users : {beh_s['users']['cold_start_dev_users']:,} users ({beh_s['users']['cold_start_pct_of_dev']}% of Dev users)")
    print(f" Overall CTR          : Train = {beh_s['click_through_rate']['train_ctr']}% | Dev = {beh_s['click_through_rate']['dev_ctr']}% | Overall = {beh_s['click_through_rate']['overall_ctr']}%")
    
    # Candidates per impression
    cands = beh_s["candidates_per_impression"]["combined"]
    print(f" Candidates/Impression: Min={cands['min']}, Max={cands['max']}, Mean={cands['mean']}, Median={cands['median']}, Std={cands['std']}")
    
    # Clicks per impression
    clicks = beh_s["clicks_per_impression"]["combined"]
    print(f" Clicks/Impression     : Min={clicks['min']}, Max={clicks['max']}, Mean={clicks['mean']}, Median={clicks['median']}, Std={clicks['std']}")
    
    # History length
    hist = beh_s["history_length"]["combined"]
    print(f" User History Length  : Min={hist['min']}, Max={hist['max']}, Mean={hist['mean']}, Median={hist['median']}, Std={hist['std']}")
    print(f" Empty History %      : Train = {beh_s['history_length']['train_zero_history_pct']}% | Dev = {beh_s['history_length']['dev_zero_history_pct']}%")
    
    # Article Stats
    news_s = stats["news"]
    print("\n3. NEWS ARTICLE STATISTICS")
    print("-" * 75)
    print(f" Unique Articles      : Train = {news_s['article_counts']['train_unique']:,} | Dev = {news_s['article_counts']['dev_unique']:,} | Total = {news_s['article_counts']['total_unique']:,}")
    print(f" Article Overlap      : {news_s['article_counts']['overlap_count']:,} articles ({news_s['article_counts']['overlap_pct_of_dev']}% of Dev catalog)")
    print(f" Categories Count     : {news_s['categories']['unique_categories']} categories, {news_s['categories']['unique_subcategories']} subcategories")
    print(f" Top 5 Categories     : {news_s['categories']['top_categories']}")
    
    title_w = news_s["title_word_count"]["combined"]
    print(f" Title Word Count     : Min={title_w['min']}, Max={title_w['max']}, Mean={title_w['mean']}, Median={title_w['median']}")
    
    abs_w = news_s["abstract_word_count"]["combined"]
    print(f" Abstract Word Count  : Min={abs_w['min']}, Max={abs_w['max']}, Mean={abs_w['mean']}, Median={abs_w['median']}")
    print("=" * 75)


def generate_markdown_report(stats: Dict[str, Any]) -> str:
    """Generate Markdown formatted string of dataset statistics."""
    time_s = stats["time"]
    beh_s = stats["behaviors"]
    news_s = stats["news"]
    
    md = f"""# MIND Dataset Statistics Report

## 1. Temporal Statistics (Timestamps)

| Metric | Train Split | Dev Split | Combined |
| :--- | :--- | :--- | :--- |
| **Minimum Timestamp** | `{time_s['train']['min_time']}` | `{time_s['dev']['min_time']}` | `{time_s['combined']['min_time']}` |
| **Maximum Timestamp** | `{time_s['train']['max_time']}` | `{time_s['dev']['max_time']}` | `{time_s['combined']['max_time']}` |
| **Duration (Days)** | {time_s['train']['duration_days']} days | {time_s['dev']['duration_days']} days | {time_s['combined']['duration_days']} days |

* **Train to Dev Gap**: {time_s['combined']['gap_hours']} hours

## 2. Impression & User Behavior Statistics

| Metric | Train | Dev | Total / Combined |
| :--- | :--- | :--- | :--- |
| **Impression Count** | {beh_s['impressions_count']['train']:,} | {beh_s['impressions_count']['dev']:,} | {beh_s['impressions_count']['total']:,} |
| **Unique Users** | {beh_s['users']['train_unique']:,} | {beh_s['users']['dev_unique']:,} | {beh_s['users']['total_unique']:,} |
| **Click-Through-Rate (CTR)** | {beh_s['click_through_rate']['train_ctr']}% | {beh_s['click_through_rate']['dev_ctr']}% | {beh_s['click_through_rate']['overall_ctr']}% |
| **Empty History %** | {beh_s['history_length']['train_zero_history_pct']}% | {beh_s['history_length']['dev_zero_history_pct']}% | - |

* **User Overlap**: {beh_s['users']['overlap_count']:,} users ({beh_s['users']['overlap_pct_of_dev']}% of Dev users)
* **Cold-Start Dev Users**: {beh_s['users']['cold_start_dev_users']:,} users ({beh_s['users']['cold_start_pct_of_dev']}% of Dev users)

### Interaction Summary Statistics

| Statistic | Candidates / Imp | Clicks / Imp | User History Length | User Imp Frequency |
| :--- | :--- | :--- | :--- | :--- |
| **Min** | {beh_s['candidates_per_impression']['combined']['min']} | {beh_s['clicks_per_impression']['combined']['min']} | {beh_s['history_length']['combined']['min']} | {beh_s['user_impression_frequency']['train']['min']} |
| **Max** | {beh_s['candidates_per_impression']['combined']['max']} | {beh_s['clicks_per_impression']['combined']['max']} | {beh_s['history_length']['combined']['max']} | {beh_s['user_impression_frequency']['train']['max']} |
| **Mean** | {beh_s['candidates_per_impression']['combined']['mean']} | {beh_s['clicks_per_impression']['combined']['mean']} | {beh_s['history_length']['combined']['mean']} | {beh_s['user_impression_frequency']['train']['mean']} |
| **Median** | {beh_s['candidates_per_impression']['combined']['median']} | {beh_s['clicks_per_impression']['combined']['median']} | {beh_s['history_length']['combined']['median']} | {beh_s['user_impression_frequency']['train']['median']} |

## 3. News Article Statistics

| Metric | Train | Dev | Total / Combined |
| :--- | :--- | :--- | :--- |
| **Unique Articles** | {news_s['article_counts']['train_unique']:,} | {news_s['article_counts']['dev_unique']:,} | {news_s['article_counts']['total_unique']:,} |
| **Article Overlap** | - | - | {news_s['article_counts']['overlap_count']:,} ({news_s['article_counts']['overlap_pct_of_dev']}% of Dev) |

* **Categories Count**: {news_s['categories']['unique_categories']} categories, {news_s['categories']['unique_subcategories']} subcategories
* **Top 5 Categories**: {news_s['categories']['top_categories']}
* **Title Word Count (Mean/Median)**: {news_s['title_word_count']['combined']['mean']} / {news_s['title_word_count']['combined']['median']}
* **Abstract Word Count (Mean/Median)**: {news_s['abstract_word_count']['combined']['mean']} / {news_s['abstract_word_count']['combined']['median']}
"""
    return md


def main():
    args = parse_args()
    train_dir = Path(args.train_dir)
    dev_dir = Path(args.dev_dir)
    
    print(f"Loading MIND train dataset from: {train_dir}")
    train_b = load_behaviors(train_dir / "behaviors.tsv")
    train_n = load_news(train_dir / "news.tsv")
    
    print(f"Loading MIND dev dataset from: {dev_dir}")
    dev_b = load_behaviors(dev_dir / "behaviors.tsv")
    dev_n = load_news(dev_dir / "news.tsv")
    
    print("Computing dataset statistics...")
    time_stats = analyze_time_stats(train_b, dev_b)
    behaviors_stats = analyze_behaviors_stats(train_b, dev_b)
    news_stats = analyze_news_stats(train_n, dev_n)
    
    stats = {
        "time": time_stats,
        "behaviors": behaviors_stats,
        "news": news_stats
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
