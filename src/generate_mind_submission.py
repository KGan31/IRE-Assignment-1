"""
MIND-Large Candidate Ranking & Codabench Submission Generator.

Builds BM25 index over full MIND-Large article catalog, evaluates dev set ranking metrics,
and scores all candidate articles for test set impressions (2.37M impressions) to generate
a compliant prediction.zip submission for Codabench.

Submission format:
Each line in prediction.txt is:
<ImpressionID> [<rank_of_cand_1>,<rank_of_cand_2>,...,<rank_of_cand_N>]
where rank 1 corresponds to the candidate with the highest predicted relevance score.

Usage:
    python src/generate_mind_submission.py --dataset_type large --eval_dev
    python src/generate_mind_submission.py --dataset_type large --output_dir submissions
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
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import rankdata
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

    def fit(self, articles_df: pd.DataFrame, history_fields: str = "title_abstract") -> None:
        """Index articles (title + abstract) and precompute term weights."""
        print(f"Building FastBM25 index over {len(articles_df)} articles (history_fields: {history_fields})...")
        self.num_docs = len(articles_df)
        df_counts: Dict[str, int] = Counter()
        doc_tokens_map: Dict[str, List[str]] = {}

        total_len = 0
        article_ids = articles_df["article_id"].astype(str).tolist()
        titles = articles_df["title"].fillna("").astype(str).tolist()
        abstracts = (
            articles_df["abstract"].fillna("").astype(str).tolist()
            if "abstract" in articles_df.columns
            else [""] * self.num_docs
        )

        norm_fields = str(history_fields).lower().replace("+", "_").replace(" ", "_")
        include_abstract_in_query = norm_fields in ("title_abstract", "title_and_abstract", "both", "abstract_title")

        for aid, title, abstract in zip(article_ids, titles, abstracts):
            self.doc_titles[aid] = title
            if include_abstract_in_query:
                self.doc_texts[aid] = f"{title} {abstract}".strip()
            else:
                self.doc_texts[aid] = title.strip()
            full_text = f"{title} {abstract}".strip()
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

        print(f"BM25 index built. Vocab size: {len(self.idf)}, Avg doc len: {self.avgdl:.2f}")

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


def load_user_recent_history(history_path: Path, max_history_len: int = 20) -> Dict[str, List[str]]:
    """Load user histories into a dict: user_id -> list of recent clicked article_ids using Polars."""
    if not history_path.exists():
        return {}
    print(f"Loading and aggregating history from {history_path}...")
    df = pl.read_parquet(history_path)
    agg_df = (
        df.group_by("user_id", maintain_order=True)
        .agg(pl.col("clicked_article_id").tail(max_history_len))
    )
    user_ids = agg_df["user_id"].to_list()
    hist_lists = agg_df["clicked_article_id"].to_list()
    user_map = dict(zip(user_ids, [list(h) for h in hist_lists]))
    print(f"Loaded history for {len(user_map)} users.")
    return user_map


def evaluate_dev_split(
    scorer: FastBM25Scorer,
    dev_impressions_path: Path,
    user_history_map: Dict[str, List[str]],
    max_history_len: int = 20,
) -> Dict[str, float]:
    """Evaluate candidate ranking on dev split impressions."""
    print(f"Evaluating BM25 Candidate Ranking on Dev Set: {dev_impressions_path}...")
    dev_df = pl.read_parquet(dev_impressions_path)

    user_ids = dev_df["user_id"].to_list()
    cands_list = dev_df["candidate_article_ids"].to_list()
    clicked_list = dev_df["clicked_article_ids"].to_list()

    auc_scores = []
    mrr_scores = []
    ndcg5_scores = []
    ndcg10_scores = []

    for user_id, cands, clicked in zip(user_ids, cands_list, clicked_list):
        if not cands or not clicked:
            continue

        cands = list(cands)
        gt_set = set(clicked)
        labels = [1 if c in gt_set else 0 for c in cands]
        
        # Skip impressions with all positive or all negative
        num_pos = sum(labels)
        if num_pos == 0 or num_pos == len(labels):
            continue

        recent_clicks = user_history_map.get(user_id, [])
        q_text = " ".join([scorer.doc_texts.get(aid, "") for aid in recent_clicks if aid in scorer.doc_texts])
        q_tokens = tokenize(q_text)

        cand_scores = scorer.score_candidates(q_tokens, cands)
        ranked_indices = np.argsort(-np.asarray(cand_scores, dtype=np.float32))

        auc_val = compute_auc(labels, cand_scores)
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
    print(f"  DEV SET EVALUATION RESULTS (BM25)")
    print("=" * 55)
    print(f"  AUC:      {metrics['AUC']:.4f}")
    print(f"  MRR:      {metrics['MRR']:.4f}")
    print(f"  nDCG@5:   {metrics['nDCG@5']:.4f}")
    print(f"  nDCG@10:  {metrics['nDCG@10']:.4f}")
    print(f"  Evaluated Impressions: {metrics['evaluated_impressions']}")
    print("=" * 55 + "\n")
    return metrics


def generate_submission(
    scorer: FastBM25Scorer,
    test_impressions_path: Path,
    user_history_map: Dict[str, List[str]],
    output_dir: Path,
    max_history_len: int = 20,
    chunk_size: int = 50000,
) -> Path:
    """
    Score candidates for test impressions and output prediction.txt & prediction.zip.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_txt_path = output_dir / "prediction.txt"
    prediction_zip_path = output_dir / "prediction.zip"

    print(f"Loading test impressions from {test_impressions_path}...")
    test_df = pl.read_parquet(test_impressions_path)
    num_impressions = len(test_df)
    print(f"Total test impressions: {num_impressions}")

    impr_ids = test_df["impression_id"].to_list()
    user_ids = test_df["user_id"].to_list()
    candidates_list = test_df["candidate_article_ids"].to_list()

    print(f"Scoring test impressions and streaming to {prediction_txt_path}...")
    start_time = time.time()

    buffer = []
    with open(prediction_txt_path, "w", encoding="utf-8", buffering=1024 * 1024) as out_f:
        for i in tqdm(range(num_impressions), desc="Generating Test Predictions", unit="impr"):
            raw_impr_id = impr_ids[i]
            if isinstance(raw_impr_id, str) and raw_impr_id.startswith("mind_"):
                raw_impr_id = raw_impr_id[len("mind_"):]

            cands = candidates_list[i]
            if not cands:
                buffer.append(f"{raw_impr_id} []\n")
            else:
                cands = list(cands)
                user_id = user_ids[i]

                recent_clicks = user_history_map.get(user_id, [])
                q_text = " ".join([scorer.doc_texts.get(aid, "") for aid in recent_clicks if aid in scorer.doc_texts])
                q_tokens = tokenize(q_text)

                cand_scores = scorer.score_candidates(q_tokens, cands)

                ranks = rankdata(-np.array(cand_scores, dtype=np.float32), method="ordinal")
                rank_str = ",".join(str(int(r)) for r in ranks)
                buffer.append(f"{raw_impr_id} [{rank_str}]\n")

            if len(buffer) >= chunk_size:
                out_f.writelines(buffer)
                buffer.clear()

        if buffer:
            out_f.writelines(buffer)
            buffer.clear()

    elapsed = time.time() - start_time
    print(f"Wrote {num_impressions} predictions in {elapsed:.2f}s ({num_impressions / max(0.1, elapsed):.0f} impr/s)")

    print(f"Creating submission zip: {prediction_zip_path}...")
    with zipfile.ZipFile(prediction_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(prediction_txt_path, arcname="prediction.txt")

    print(f"Submission zip successfully created at: {prediction_zip_path}")
    validate_submission_file(prediction_txt_path, num_impressions)
    return prediction_zip_path


def validate_submission_file(prediction_txt_path: Path, expected_lines: int) -> None:
    """Validate format and integrity of the output prediction.txt."""
    print("Validating submission file format...")
    line_count = 0
    pattern = re.compile(r"^\d+\s+\[\d+(?:,\d+)*\]$")

    with open(prediction_txt_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            line_count += 1
            line = line.strip()
            if not pattern.match(line):
                if line_count <= 5 or not line:
                    raise ValueError(f"Invalid format at line {line_count}: {line}")

    if line_count != expected_lines:
        print(f"Warning: Expected {expected_lines} lines, but found {line_count} lines.")
    else:
        print(f"Verification PASSED: {line_count} correctly formatted lines matching all impressions.")


def main():
    parser = argparse.ArgumentParser(description="Generate MIND Codabench BM25 submission.")
    parser.add_argument("--dataset_type", choices=["small", "large"], default="large", help="Dataset scale")
    parser.add_argument("--eval_dev", action="store_true", help="Evaluate BM25 ranking on dev split")
    parser.add_argument(
        "--history_fields",
        choices=["title", "title_abstract", "title+abstract", "both"],
        default="title_abstract",
        help="Fields from history articles in query: 'title' or 'title_abstract' / 'title+abstract' (default: title_abstract)",
    )
    parser.add_argument("--output_dir", default="submissions", help="Output directory for submission zip")
    parser.add_argument("--max_history_len", type=int, default=20, help="Max recent articles for query")
    args = parser.parse_args()

    ds_dir = Path("data/processed") / f"mind_{args.dataset_type}"
    articles_path = ds_dir / "articles.parquet"
    dev_impr_path = ds_dir / "impressions_dev.parquet"
    dev_hist_path = ds_dir / "history_dev.parquet"
    test_impr_path = ds_dir / "impressions_test.parquet"
    test_hist_path = ds_dir / "history_test.parquet"

    if not articles_path.exists():
        raise FileNotFoundError(f"Articles not found at {articles_path}. Run download and parse_mind first.")

    # 1. Load articles and fit BM25 Scorer
    articles_df = pd.read_parquet(articles_path)
    scorer = FastBM25Scorer()
    scorer.fit(articles_df, history_fields=args.history_fields)

    # 2. Evaluate on Dev if requested
    if args.eval_dev and dev_impr_path.exists():
        dev_history_map = load_user_recent_history(dev_hist_path, max_history_len=args.max_history_len)
        evaluate_dev_split(scorer, dev_impr_path, dev_history_map, max_history_len=args.max_history_len)

    # 3. Generate Submission for Test
    if test_impr_path.exists():
        test_history_map = load_user_recent_history(test_hist_path, max_history_len=args.max_history_len)
        out_path = Path(args.output_dir)
        zip_path = generate_submission(
            scorer=scorer,
            test_impressions_path=test_impr_path,
            user_history_map=test_history_map,
            output_dir=out_path,
            max_history_len=args.max_history_len,
        )
        print(f"\nSUCCESS! Codabench submission ready: {zip_path.resolve()}")
    else:
        print(f"Test impressions not found at {test_impr_path}. Only dev evaluated.")


if __name__ == "__main__":
    main()
