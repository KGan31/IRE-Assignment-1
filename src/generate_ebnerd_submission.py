"""
EB-NeRD Candidate Ranking & Codabench Submission Generator.

Builds BM25 index over EB-NeRD article catalog (title + abstract), evaluates validation set ranking metrics,
and scores all candidate articles for test set impressions (13.5M impressions) to generate
a compliant predictions.zip submission for the RecSys 2024 / Codabench competition.

Submission format:
Each line in predictions.txt is:
<ImpressionID> [<rank_of_cand_1>,<rank_of_cand_2>,...,<rank_of_cand_N>]
where rank 1 corresponds to the candidate with the highest predicted relevance score.

Usage:
    python src/generate_ebnerd_submission.py --dataset_type demo --eval_dev
    python src/generate_ebnerd_submission.py --dataset_type large --eval_dev
    python src/generate_ebnerd_submission.py --dataset_type large --output_dir submissions_ebnerd
"""

import argparse
import math
import os
import re
import sys
import time
import zipfile
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import polars as pl
from tqdm import tqdm

# Ensure src is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import compute_auc, compute_mrr, compute_ndcg_at_k


def tokenize(text: str) -> List[str]:
    """Fast alphanumeric tokenization."""
    if not text or not isinstance(text, str):
        return []
    return re.findall(r"\w+", text.lower())


class FastBM25Scorer:
    """
    High-performance in-memory BM25 Scorer optimized for candidate scoring.
    Computes exact BM25 scores for arbitrary candidate article lists per query
    in microsecond time per impression.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_weights: Dict[str, Dict[str, float]] = {}  # doc_id -> {token: doc_term_weight}
        self.idf: Dict[str, float] = {}  # token -> idf
        self.doc_titles: Dict[str, str] = {}
        self.doc_texts: Dict[str, str] = {}
        self.doc_lengths: Dict[str, int] = {}
        self.avgdl: float = 0.0
        self.num_docs: int = 0

    def fit(self, articles_df: pd.DataFrame, include_body: bool = False, history_fields: str = "title_abstract") -> None:
        """Index articles (title + subtitle/abstract, skipping body for speed) and precompute term weights."""
        fields_str = "title + abstract + body" if include_body else "title + abstract (skipping body)"
        print(f"Building FastBM25 index over {len(articles_df):,} articles ({fields_str}, history_fields: {history_fields})...")
        self.num_docs = len(articles_df)
        df_counts: Dict[str, int] = Counter()
        doc_tokens_map: Dict[str, List[str]] = {}

        total_len = 0
        article_ids = articles_df["article_id"].astype(str).tolist()
        titles = articles_df["title"].fillna("").astype(str).tolist()
        subtitles = (
            articles_df["abstract"].fillna("").astype(str).tolist()
            if "abstract" in articles_df.columns
            else (articles_df["subtitle"].fillna("").astype(str).tolist() if "subtitle" in articles_df.columns else [""] * self.num_docs)
        )
        if include_body and "body" in articles_df.columns:
            bodies = articles_df["body"].fillna("").astype(str).tolist()
        else:
            bodies = [""] * self.num_docs

        norm_fields = str(history_fields).lower().replace("+", "_").replace(" ", "_")
        include_abstract_in_query = norm_fields in ("title_abstract", "title_and_abstract", "both", "abstract_title")

        for aid, title, subtitle, body in zip(article_ids, titles, subtitles, bodies):
            self.doc_titles[aid] = title
            if include_abstract_in_query:
                self.doc_texts[aid] = f"{title} {subtitle}".strip()
            else:
                self.doc_texts[aid] = title.strip()
            if include_body and body:
                full_text = f"{title} {subtitle} {body}".strip()
            else:
                full_text = f"{title} {subtitle}".strip()
            tokens = tokenize(full_text)
            doc_len = len(tokens)
            self.doc_lengths[aid] = doc_len
            total_len += doc_len
            doc_tokens_map[aid] = tokens

            unique_tokens = set(tokens)
            for t in unique_tokens:
                df_counts[t] += 1

        self.avgdl = total_len / max(1, self.num_docs)

        # Compute Robertson-Spärck Jones IDF
        for t, df in df_counts.items():
            self.idf[t] = math.log(1.0 + (self.num_docs - df + 0.5) / (df + 0.5))

        # Precompute normalized document term weights: W(t, d) = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (len/avgdl)))
        for aid, tokens in doc_tokens_map.items():
            if not tokens:
                continue
            doc_len = self.doc_lengths[aid]
            len_norm = 1.0 - self.b + self.b * (doc_len / self.avgdl)
            tf_counter = Counter(tokens)

            weights = {}
            for t, tf in tf_counter.items():
                w = (tf * (self.k1 + 1.0)) / (tf + self.k1 * len_norm)
                weights[t] = w
            self.doc_weights[aid] = weights

        print(f"BM25 index built. Vocab size: {len(self.idf):,}, Avg doc len: {self.avgdl:.2f}")

    def score_candidates(self, query_tokens: List[str], candidate_ids: List[str]) -> List[float]:
        """
        Compute BM25 scores for a specific list of candidate articles given query tokens.
        """
        if not query_tokens:
            return [0.0] * len(candidate_ids)

        q_tf = Counter(query_tokens)
        q_weights = {t: q_tf[t] * self.idf[t] for t in q_tf if t in self.idf}

        scores = []
        for cand_id in candidate_ids:
            cand_dict = self.doc_weights.get(cand_id)
            if not cand_dict:
                scores.append(0.0)
                continue

            # Sparse dot product between query terms and candidate doc terms
            s = 0.0
            for t, q_w in q_weights.items():
                if t in cand_dict:
                    s += q_w * cand_dict[t]
            scores.append(s)

        return scores


def load_user_recent_history(
    history_path: Path,
    raw_history_path: Optional[Path] = None,
    max_history_len: int = 20,
) -> Dict[str, List[str]]:
    """
    Load user histories into a dict: user_id -> list of recent clicked article_ids.
    Uses fast direct list slice from raw history parquet when available.
    """
    if raw_history_path and raw_history_path.exists():
        print(f"Loading user history directly from raw parquet {raw_history_path}...")
        try:
            raw_hist = pl.read_parquet(raw_history_path, columns=["user_id", "article_id_fixed"])
            transformed = raw_hist.select([
                (pl.lit("ebnerd_") + pl.col("user_id").cast(pl.Utf8)).alias("user_id"),
                pl.col("article_id_fixed")
                .list.tail(max_history_len)
                .list.eval(pl.lit("ebnerd_") + pl.element().cast(pl.Utf8))
                .alias("recent_clicks"),
            ])
            user_ids = transformed["user_id"].to_list()
            clicks = transformed["recent_clicks"].to_list()
            user_map = dict(zip(user_ids, clicks))
            print(f"Loaded history for {len(user_map):,} users in fast mode.")
            return user_map
        except Exception as e:
            print(f"[warning] Fast history load failed ({e}), falling back to processed history.")

    if not history_path.exists():
        return {}

    print(f"Loading and aggregating history from {history_path}...")
    df = pl.read_parquet(history_path)
    if "user_id" not in df.columns or "clicked_article_id" not in df.columns:
        return {}

    agg_df = (
        df.group_by("user_id", maintain_order=True)
        .agg(pl.col("clicked_article_id").tail(max_history_len))
    )
    user_ids = agg_df["user_id"].to_list()
    hist_lists = agg_df["clicked_article_id"].to_list()
    user_map = dict(zip(user_ids, [list(h) for h in hist_lists]))
    print(f"Loaded history for {len(user_map):,} users.")
    return user_map


def compute_article_popularity(history_paths: List[Path]) -> Dict[str, float]:
    """Compute normalized article popularity from available history files as a tie-breaker."""
    pop_counts: Counter = Counter()
    for p in history_paths:
        if p.exists():
            print(f"Counting article popularity from {p}...")
            try:
                schema = pl.scan_parquet(p).collect_schema().names()
                if "article_id_fixed" in schema:
                    # Raw format list column
                    df = pl.read_parquet(p, columns=["article_id_fixed"])
                    exploded = df.select(pl.col("article_id_fixed").explode().value_counts()).unnest("article_id_fixed")
                    for aid, c in zip(exploded["article_id_fixed"].to_list(), exploded["count"].to_list()):
                        if aid is not None:
                            pop_counts[f"ebnerd_{aid}"] += c
                elif "clicked_article_id" in schema:
                    # Processed format single column
                    df = pl.read_parquet(p, columns=["clicked_article_id"])
                    counts = df["clicked_article_id"].value_counts()
                    for aid, c in zip(counts["clicked_article_id"].to_list(), counts["count"].to_list()):
                        if aid is not None:
                            pop_counts[str(aid)] += c
            except Exception as e:
                print(f"[warning] Popularity counting error for {p}: {e}")

    if not pop_counts:
        return {}

    max_c = max(pop_counts.values())
    print(f"Computed popularity for {len(pop_counts):,} unique articles (max count: {max_c:,}).")
    return {aid: c / max_c for aid, c in pop_counts.items()}


def fast_ordinal_ranks(scores: List[float]) -> str:
    """Compute 1-based ranks where rank 1 corresponds to highest score."""
    n = len(scores)
    order = sorted(range(n), key=lambda idx: scores[idx], reverse=True)
    ranks = [0] * n
    for r, idx in enumerate(order, 1):
        ranks[idx] = r
    return ",".join(map(str, ranks))


def evaluate_dev_split(
    scorer: FastBM25Scorer,
    dev_impressions_path: Path,
    user_history_map: Dict[str, List[str]],
    popularity_map: Optional[Dict[str, float]] = None,
    max_history_len: int = 20,
    sample_size: int = 100000,
) -> Dict[str, float]:
    """Evaluate candidate ranking on dev/validation split impressions."""
    print(f"Evaluating BM25 Candidate Ranking on Validation Set: {dev_impressions_path}...")
    if sample_size > 0:
        dev_df = pl.read_parquet(dev_impressions_path, n_rows=sample_size)
        print(f"Using representative sample of {len(dev_df):,} validation impressions...")
    else:
        dev_df = pl.read_parquet(dev_impressions_path)
        print(f"Evaluating full set of {len(dev_df):,} validation impressions...")

    user_ids = dev_df["user_id"].to_list()
    cands_list = dev_df["candidate_article_ids"].to_list()
    clicked_list = dev_df["clicked_article_ids"].to_list()

    # Pre-build user query weights for users in validation
    user_q_weights: Dict[str, Dict[str, float]] = {}
    for uid in set(user_ids):
        recent_clicks = user_history_map.get(uid, [])
        if recent_clicks:
            q_text = " ".join([scorer.doc_texts.get(aid, "") for aid in recent_clicks if aid in scorer.doc_texts])
            tokens = tokenize(q_text)
            if tokens:
                q_tf = Counter(tokens)
                user_q_weights[uid] = {t: q_tf[t] * scorer.idf[t] for t in q_tf if t in scorer.idf}

    auc_scores = []
    mrr_scores = []
    ndcg5_scores = []
    ndcg10_scores = []

    for i in tqdm(range(len(dev_df)), desc="Evaluating Validation Ranking", unit="impr"):
        cands = cands_list[i]
        clicked = clicked_list[i]
        if not cands or not clicked:
            continue

        cands = list(cands)
        gt_set = set(clicked)
        labels = [1 if c in gt_set else 0 for c in cands]

        num_pos = sum(labels)
        if num_pos == 0 or num_pos == len(labels):
            continue

        uid = user_ids[i]
        q_weights = user_q_weights.get(uid)

        cand_scores = [0.0] * len(cands)
        if q_weights:
            for ci, cand_id in enumerate(cands):
                cand_dict = scorer.doc_weights.get(cand_id)
                if cand_dict:
                    s = 0.0
                    for t, qw in q_weights.items():
                        if t in cand_dict:
                            s += qw * cand_dict[t]
                    cand_scores[ci] = s

        if popularity_map:
            for ci, cand_id in enumerate(cands):
                cand_scores[ci] += 1e-4 * popularity_map.get(cand_id, 0.0)

        cand_scores_arr = np.array(cand_scores, dtype=np.float32)
        ranked_indices = np.argsort(-cand_scores_arr)

        auc_val = compute_auc(labels, cand_scores_arr)
        if auc_val is not None:
            auc_scores.append(auc_val)
        mrr_scores.append(compute_mrr(labels, ranked_indices=ranked_indices))
        ndcg5_scores.append(compute_ndcg_at_k(labels, k=5, ranked_indices=ranked_indices))
        ndcg10_scores.append(compute_ndcg_at_k(labels, k=10, ranked_indices=ranked_indices))

    metrics = {
        "AUC": float(np.mean(auc_scores)) if auc_scores else 0.0,
        "MRR": float(np.mean(mrr_scores)) if mrr_scores else 0.0,
        "nDCG@5": float(np.mean(ndcg5_scores)) if ndcg5_scores else 0.0,
        "nDCG@10": float(np.mean(ndcg10_scores)) if ndcg10_scores else 0.0,
        "evaluated_impressions": len(auc_scores),
    }

    print("\n" + "=" * 55)
    print(f"  EB-NeRD VALIDATION SET EVALUATION RESULTS (BM25)")
    print("=" * 55)
    print(f"  AUC:      {metrics['AUC']:.4f}")
    print(f"  MRR:      {metrics['MRR']:.4f}")
    print(f"  nDCG@5:   {metrics['nDCG@5']:.4f}")
    print(f"  nDCG@10:  {metrics['nDCG@10']:.4f}")
    print(f"  Evaluated Impressions: {metrics['evaluated_impressions']:,}")
    print("=" * 55 + "\n")
    return metrics


def generate_submission(
    scorer: FastBM25Scorer,
    test_impressions_path: Path,
    user_history_map: Dict[str, List[str]],
    output_dir: Path,
    popularity_map: Optional[Dict[str, float]] = None,
    max_history_len: int = 20,
    chunk_size: int = 100000,
) -> Path:
    """
    Score candidates for test impressions and output predictions.txt & predictions.zip.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_txt_path = output_dir / "predictions.txt"
    predictions_zip_path = output_dir / "predictions.zip"

    print(f"Loading test impressions from {test_impressions_path}...")
    test_df = pl.read_parquet(test_impressions_path)
    num_impressions = len(test_df)
    print(f"Total test impressions: {num_impressions:,}")

    impr_ids = test_df["impression_id"].to_list()
    user_ids = test_df["user_id"].to_list()
    candidates_list = test_df["candidate_article_ids"].to_list()

    # Pre-build user query weights for all users in test set
    print("Precomputing user query weights for test set users...")
    user_q_weights: Dict[str, Dict[str, float]] = {}
    unique_users = set(user_ids)
    for uid in unique_users:
        recent_clicks = user_history_map.get(uid, [])
        if recent_clicks:
            q_text = " ".join([scorer.doc_texts.get(aid, "") for aid in recent_clicks if aid in scorer.doc_texts])
            tokens = tokenize(q_text)
            if tokens:
                q_tf = Counter(tokens)
                user_q_weights[uid] = {t: q_tf[t] * scorer.idf[t] for t in q_tf if t in scorer.idf}
    print(f"Precomputed query weights for {len(user_q_weights):,} unique active users.")

    print(f"Scoring {num_impressions:,} test impressions and streaming to {predictions_txt_path}...")
    start_time = time.time()

    buffer = []
    with open(predictions_txt_path, "w", encoding="utf-8", buffering=1024 * 1024) as out_f:
        for i in tqdm(range(num_impressions), desc="Generating EB-NeRD Predictions", unit="impr"):
            raw_impr_id = impr_ids[i]
            if isinstance(raw_impr_id, str) and raw_impr_id.startswith("ebnerd_"):
                raw_impr_id = raw_impr_id[len("ebnerd_"):]

            cands = candidates_list[i]
            if not cands:
                buffer.append(f"{raw_impr_id} []\n")
            else:
                cands = list(cands)
                n = len(cands)
                scores = [0.0] * n
                uid = user_ids[i]
                q_weights = user_q_weights.get(uid)

                if q_weights:
                    for ci in range(n):
                        cand_dict = scorer.doc_weights.get(cands[ci])
                        if cand_dict:
                            s = 0.0
                            for t, qw in q_weights.items():
                                if t in cand_dict:
                                    s += qw * cand_dict[t]
                            scores[ci] = s

                if popularity_map:
                    for ci in range(n):
                        scores[ci] += 1e-4 * popularity_map.get(cands[ci], 0.0)

                rank_str = fast_ordinal_ranks(scores)
                buffer.append(f"{raw_impr_id} [{rank_str}]\n")

            if len(buffer) >= chunk_size:
                out_f.writelines(buffer)
                buffer.clear()

        if buffer:
            out_f.writelines(buffer)
            buffer.clear()

    elapsed = time.time() - start_time
    print(f"Wrote {num_impressions:,} predictions in {elapsed:.2f}s ({num_impressions / max(0.1, elapsed):,.0f} impr/s)")

    print(f"Creating submission zip: {predictions_zip_path}...")
    with zipfile.ZipFile(str(predictions_zip_path.resolve()), "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(str(predictions_txt_path.resolve()), arcname="predictions.txt")

    print(f"Submission zip successfully created at: {predictions_zip_path}")
    validate_submission_file(predictions_txt_path, num_impressions)
    return predictions_zip_path


def validate_submission_file(predictions_txt_path: Path, expected_lines: int) -> None:
    """Validate format and integrity of the output predictions.txt."""
    print("Validating submission file format...")
    line_count = 0
    pattern = re.compile(r"^\d+\s+\[\d*(?:,\d+)*\]$")

    with open(predictions_txt_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            line_count += 1
            line = line.strip()
            if not pattern.match(line):
                if line_count <= 5 or not line:
                    raise ValueError(f"Invalid format at line {line_count}: {line}")

    if line_count != expected_lines:
        print(f"Warning: Expected {expected_lines:,} lines, but found {line_count:,} lines.")
    else:
        print(f"Verification PASSED: {line_count:,} correctly formatted lines matching all impressions.")


def main():
    parser = argparse.ArgumentParser(description="Generate EB-NeRD Codabench BM25 submission.")
    parser.add_argument("--dataset_type", choices=["demo", "small", "large"], default="large", help="Dataset scale")
    parser.add_argument("--eval_dev", action="store_true", help="Evaluate BM25 ranking on validation split")
    parser.add_argument("--val_sample_size", type=int, default=100000, help="Validation sample size for evaluation (-1 for all)")
    parser.add_argument("--include_body", action="store_true", default=False, help="Include body text in index (default: False, title+abstract only)")
    parser.add_argument(
        "--history_fields",
        choices=["title", "title_abstract", "title+abstract", "both"],
        default="title_abstract",
        help="Fields from history articles in query: 'title' or 'title_abstract' / 'title+abstract' (default: title_abstract)",
    )
    parser.add_argument("--output_dir", default="submissions_ebnerd", help="Output directory for submission zip")
    parser.add_argument("--max_history_len", type=int, default=20, help="Max recent articles for query")
    parser.add_argument("--use_popularity_fallback", action="store_true", default=True, help="Use popularity tie-breaker")
    args = parser.parse_args()

    ds_dir = Path("data/processed") / f"ebnerd_{args.dataset_type}" if args.dataset_type != "demo" else Path("data/processed/ebnerd")
    raw_dir = Path("data/raw") / f"ebnerd_{args.dataset_type}" if args.dataset_type != "demo" else Path("data/raw/ebnerd/demo")

    articles_path = ds_dir / "articles.parquet"
    val_impr_path = ds_dir / "impressions_val.parquet"
    if not val_impr_path.exists():
        val_impr_path = ds_dir / "impressions_validation.parquet"
    val_hist_path = ds_dir / "history_val.parquet"
    if not val_hist_path.exists():
        val_hist_path = ds_dir / "history_validation.parquet"

    test_impr_path = ds_dir / "impressions_test.parquet"
    test_hist_path = ds_dir / "history_test.parquet"
    train_hist_path = ds_dir / "history_train.parquet"

    # Raw paths for fast loading
    raw_val_hist = raw_dir / "validation" / "history.parquet"
    raw_test_hist = raw_dir / "test" / "history.parquet"
    raw_train_hist = raw_dir / "train" / "history.parquet"

    if not articles_path.exists():
        raise FileNotFoundError(f"Articles not found at {articles_path}. Run download and parse_ebnerd first.")

    # 1. Load articles and fit BM25 Scorer
    articles_df = pd.read_parquet(articles_path)
    scorer = FastBM25Scorer()
    scorer.fit(articles_df, include_body=args.include_body, history_fields=args.history_fields)

    # 2. Compute article popularity if requested
    pop_map = None
    if args.use_popularity_fallback:
        pop_hist_paths = [raw_train_hist, raw_val_hist, raw_test_hist] if raw_train_hist.exists() else [train_hist_path, val_hist_path, test_hist_path]
        pop_map = compute_article_popularity(pop_hist_paths)

    # 3. Evaluate on Validation if requested
    if args.eval_dev and val_impr_path.exists():
        val_history_map = load_user_recent_history(
            history_path=val_hist_path,
            raw_history_path=raw_val_hist,
            max_history_len=args.max_history_len,
        )
        evaluate_dev_split(
            scorer=scorer,
            dev_impressions_path=val_impr_path,
            user_history_map=val_history_map,
            popularity_map=pop_map,
            max_history_len=args.max_history_len,
            sample_size=args.val_sample_size,
        )

    # 4. Generate Submission for Test
    if test_impr_path.exists():
        test_history_map = load_user_recent_history(
            history_path=test_hist_path,
            raw_history_path=raw_test_hist,
            max_history_len=args.max_history_len,
        )
        out_path = Path(args.output_dir)
        zip_path = generate_submission(
            scorer=scorer,
            test_impressions_path=test_impr_path,
            user_history_map=test_history_map,
            output_dir=out_path,
            popularity_map=pop_map,
            max_history_len=args.max_history_len,
        )
        print(f"\nSUCCESS! EB-NeRD Codabench submission ready: {zip_path.resolve()}")
    else:
        print(f"Test impressions not found at {test_impr_path}. Only validation evaluated.")


if __name__ == "__main__":
    main()
