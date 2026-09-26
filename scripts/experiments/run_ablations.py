#!/usr/bin/env python
"""
Ablation Study Runner
=====================
Runs a structured set of ablations across MIND (small/val) and
EB-NeRD (demo/val) and writes results to docs/ablations/ablations_1000.md.

Ablation Phases
---------------
Phase 1  BM25 Input Fields
    MIND    : title  vs  title+abstract
    EB-NeRD : title  vs  title+abstract  vs  title+abstract+body (index)

Phase 2  History Length (BM25, title+abstract)
    Both datasets: max_history_len in [10, 20, 30]
    -> best_history_len selected by highest mean nDCG@10 across datasets

Phase 3  Cold-Start Threshold (BM25, title+abstract, best_history_len)
    Both datasets: percentiles [1, 2, 5, 10]

Phase 4  Semantic Index Type (Embedding, best_history_len)
    Both datasets: index_type in [flat, ann]

All phases use eval_mode="all" (ranking + retrieval).
Run from the repo root:
    python run_ablations.py
"""

import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Path setup so we can import from src/
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(ROOT))

from eval_bm25 import evaluate_bm25
from eval_embeddings import evaluate_embeddings

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CONFIG_PATH = ROOT / "configs" / "pipeline.yaml"
with open(CONFIG_PATH) as _f:
    _cfg = yaml.safe_load(_f)

PROCESSED_DIR = ROOT / Path(_cfg["paths"]["processed_dir"])
RAW_DIR = ROOT / Path(_cfg["paths"]["raw_dir"])
OUTPUT_MD = ROOT / "docs" / "ablations" / "ablations_1000.md"

DATASETS = ["mind", "ebnerd"]
SPLIT = "val"
# Reduce bootstrap samples for speed; increase to 1000 for publication runs.
N_BOOTSTRAPS = 1000

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def safe_run(fn, label: str, **kwargs) -> Optional[Dict[str, Any]]:
    """Call fn(**kwargs), catch exceptions, return result dict or None."""
    print(f"\n{'='*60}")
    print(f"  RUNNING: {label}")
    print(f"{'='*60}")
    try:
        return fn(**kwargs)
    except Exception as e:
        print(f"  [ERROR] {label}: {e}")
        traceback.print_exc()
        return None


def get_metric(result: Optional[Dict], section: str, metric: str) -> Optional[float]:
    """Safely extract result[section]['metrics'][metric]['mean']."""
    if result is None:
        return None
    m = result.get(section, {}).get("metrics", {}).get(metric, {})
    return m.get("mean") if isinstance(m, dict) else None


def fmt(v: Optional[float]) -> str:
    return f"{v:.4f}" if v is not None else "N/A"


# ---------------------------------------------------------------------------
# Markdown table builders
# ---------------------------------------------------------------------------

RANKING_COLS = ["AUC", "MRR", "nDCG@5", "nDCG@10"]
RETRIEVAL_COLS = ["Recall@50", "Recall@100", "Recall@200"]


def ranking_row(label: str, result: Optional[Dict], ds: str) -> str:
    cells = [f"`{label}`", ds.upper()]
    cells += [fmt(get_metric(result, "ranking", m)) for m in RANKING_COLS]
    return "| " + " | ".join(cells) + " |"


def retrieval_row(label: str, result: Optional[Dict], ds: str) -> str:
    cells = [f"`{label}`", ds.upper()]
    cells += [fmt(get_metric(result, "retrieval", m)) for m in RETRIEVAL_COLS]
    return "| " + " | ".join(cells) + " |"


def ranking_table(rows: List[str]) -> str:
    hdr = "| Config | Dataset | AUC | MRR | nDCG@5 | nDCG@10 |"
    sep = "|--------|---------|-----|-----|--------|---------|"
    return "\n".join([hdr, sep] + rows)


def retrieval_table(rows: List[str]) -> str:
    hdr = "| Config | Dataset | Recall@50 | Recall@100 | Recall@200 |"
    sep = "|--------|---------|-----------|------------|------------|"
    return "\n".join([hdr, sep] + rows)


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------

def run_bm25(label: str, ds: str, **kwargs) -> Optional[Dict]:
    return safe_run(
        evaluate_bm25,
        label=f"BM25 | {ds.upper()} | {label}",
        dataset_name=ds,
        split_name=SPLIT,
        processed_dir=PROCESSED_DIR,
        eval_mode="all",
        n_bootstraps=N_BOOTSTRAPS,
        **kwargs,
    )


def run_semantic(label: str, ds: str, **kwargs) -> Optional[Dict]:
    return safe_run(
        evaluate_embeddings,
        label=f"Semantic | {ds.upper()} | {label}",
        dataset_name=ds,
        split_name=SPLIT,
        processed_dir=PROCESSED_DIR,
        raw_dir=RAW_DIR,
        eval_mode="all",
        n_bootstraps=N_BOOTSTRAPS,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Phase 1 — Input Fields
# ---------------------------------------------------------------------------

def phase1() -> Tuple[List[str], List[str]]:
    """BM25 query field ablation: title vs title+abstract (+ body index for EB-NeRD)."""
    rank_rows: List[str] = []
    retr_rows: List[str] = []

    configs = {
        "title": dict(history_fields="title"),
        "title+abstract": dict(history_fields="title_abstract"),
    }

    for ds in DATASETS:
        for label, kwargs in configs.items():
            res = run_bm25(label, ds, max_history_len=20, **kwargs)
            rank_rows.append(ranking_row(label, res, ds))
            retr_rows.append(retrieval_row(label, res, ds))

        # EB-NeRD only: also index body text in the document store
        if ds == "ebnerd":
            label = "title+abstract+body"
            res = run_bm25(label, ds, max_history_len=20,
                           history_fields="title_abstract", include_body=True)
            rank_rows.append(ranking_row(label, res, ds))
            retr_rows.append(retrieval_row(label, res, ds))

    return rank_rows, retr_rows


# ---------------------------------------------------------------------------
# Phase 2 — History Length  (returns best length)
# ---------------------------------------------------------------------------

def phase2() -> Tuple[List[str], List[str], int]:
    """BM25 + Semantic history length ablation.

    Runs both models for each history length on both datasets.
    best_len is selected by highest mean nDCG@10 pooled across
    both models and both datasets.
    """
    rank_rows: List[str] = []
    retr_rows: List[str] = []
    lengths = [10, 20, 30]
    ndcg_by_len: Dict[int, List[float]] = {l: [] for l in lengths}

    for ds in DATASETS:
        for hl in lengths:
            # --- BM25 ---
            label = f"bm25-history={hl}"
            res = run_bm25(label, ds, max_history_len=hl,
                           history_fields="title_abstract")
            rank_rows.append(ranking_row(label, res, ds))
            retr_rows.append(retrieval_row(label, res, ds))
            v = get_metric(res, "ranking", "nDCG@10")
            if v is not None:
                ndcg_by_len[hl].append(v)

            # --- Semantic ---
            label = f"semantic-history={hl}"
            res = run_semantic(label, ds, max_history_len=hl, index_type="flat")
            rank_rows.append(ranking_row(label, res, ds))
            retr_rows.append(retrieval_row(label, res, ds))
            v = get_metric(res, "ranking", "nDCG@10")
            if v is not None:
                ndcg_by_len[hl].append(v)

    avg_ndcg = {hl: (float(np.mean(vals)) if vals else -1.0)
                for hl, vals in ndcg_by_len.items()}
    best_len = max(avg_ndcg, key=avg_ndcg.get)
    print(f"\n>> Phase 2 result: best history length = {best_len} "
          f"(avg nDCG@10 across BM25 + Semantic = {avg_ndcg[best_len]:.4f})")
    return rank_rows, retr_rows, best_len


# ---------------------------------------------------------------------------
# Phase 3 — Cold-Start Threshold
# ---------------------------------------------------------------------------

def phase3(best_history_len: int) -> Tuple[List[str], List[str]]:
    """BM25 + Semantic cold-start percentile ablation at 1, 2, 5, 10%."""
    rank_rows: List[str] = []
    retr_rows: List[str] = []
    percentiles = [1.0, 2.0, 5.0, 10.0]

    for ds in DATASETS:
        # --- BM25 ---
        label = "bm25-cold=default"
        res = run_bm25(label, ds, max_history_len=best_history_len,
                       history_fields="title_abstract")
        rank_rows.append(ranking_row(label, res, ds))
        retr_rows.append(retrieval_row(label, res, ds))

        for p in percentiles:
            label = f"bm25-cold=p{int(p)}"
            res = run_bm25(label, ds, max_history_len=best_history_len,
                           history_fields="title_abstract",
                           cold_start_percentile=p)
            rank_rows.append(ranking_row(label, res, ds))
            retr_rows.append(retrieval_row(label, res, ds))

        # --- Semantic ---
        label = "semantic-cold=default"
        res = run_semantic(label, ds, max_history_len=best_history_len,
                           index_type="flat")
        rank_rows.append(ranking_row(label, res, ds))
        retr_rows.append(retrieval_row(label, res, ds))

        for p in percentiles:
            label = f"semantic-cold=p{int(p)}"
            res = run_semantic(label, ds, max_history_len=best_history_len,
                               index_type="flat", cold_start_percentile=p)
            rank_rows.append(ranking_row(label, res, ds))
            retr_rows.append(retrieval_row(label, res, ds))

    return rank_rows, retr_rows


# ---------------------------------------------------------------------------
# Phase 4 — Semantic Index Type
# ---------------------------------------------------------------------------

def phase4(best_history_len: int) -> Tuple[List[str], List[str]]:
    """Semantic: FAISS exact (flat) vs FAISS HNSW (ann)."""
    rank_rows: List[str] = []
    retr_rows: List[str] = []

    for ds in DATASETS:
        for idx_type in ["flat", "ann"]:
            label = f"semantic-{idx_type}"
            res = run_semantic(label, ds, max_history_len=best_history_len,
                               index_type=idx_type)
            rank_rows.append(ranking_row(label, res, ds))
            retr_rows.append(retrieval_row(label, res, ds))

    return rank_rows, retr_rows


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

def write_report(
    p1_rank: List[str], p1_retr: List[str],
    p2_rank: List[str], p2_retr: List[str], best_len: int,
    p3_rank: List[str], p3_retr: List[str],
    p4_rank: List[str], p4_retr: List[str],
) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# Ablation Study Results",
        "",
        f"> Generated: {ts}",
        f"> Datasets: MIND-small / val, EB-NeRD-demo / val",
        f"> Bootstrap samples: {N_BOOTSTRAPS}",
        f"> Best history length (from Phase 2): **{best_len}**",
        "",
        "---",
        "",
        "## Phase 1 — BM25 Query Input Fields",
        "",
        "Ablates what text from the user's clicked history articles is used to",
        "build the BM25 query. For EB-NeRD, an additional variant also indexes",
        "body text in the document store (`include_body=True`).",
        "",
        "### Ranking Metrics",
        "",
        ranking_table(p1_rank),
        "",
        "### Retrieval Metrics",
        "",
        retrieval_table(p1_retr),
        "",
        "---",
        "",
        f"## Phase 2 — History Length  (BM25 + Semantic, title+abstract)",
        "",
        "Ablates how many of the user's most recent clicked articles form the query,",
        "comparing both BM25 and Semantic (FAISS flat) at each length.",
        f"Best length chosen by highest mean nDCG@10 pooled across both models and both datasets: **{best_len}**.",
        "",
        "### Ranking Metrics",
        "",
        ranking_table(p2_rank),
        "",
        "### Retrieval Metrics",
        "",
        retrieval_table(p2_retr),
        "",
        "---",
        "",
        f"## Phase 3 — Cold-Start Threshold  (BM25 + Semantic, history={best_len})",
        "",
        "Redefines cold-start users by percentile of the per-impression",
        "history-length distribution rather than a fixed click count.",
        "Run for both BM25 and Semantic to see whether the threshold sensitivity",
        "differs between retrieval paradigms.",
        "The overall metrics are computed over **all** users; the cold/warm",
        "slice breakdown appears in the printed evaluation summary.",
        "",
        "### Ranking Metrics",
        "",
        ranking_table(p3_rank),
        "",
        "### Retrieval Metrics",
        "",
        retrieval_table(p3_retr),
        "",
        "---",
        "",
        f"## Phase 4 — Semantic Index Type  (history={best_len})",
        "",
        "Compares FAISS exact brute-force (`flat`) against FAISS HNSW",
        "approximate nearest-neighbour (`ann`) on the dense retrieval path.",
        "Both use L2-normalised embeddings with cosine similarity.",
        "",
        "### Ranking Metrics",
        "",
        ranking_table(p4_rank),
        "",
        "### Retrieval Metrics",
        "",
        retrieval_table(p4_retr),
        "",
        "---",
        "",
        "## Notes",
        "",
        "- All reported metrics are **mean** values from non-parametric bootstrap resampling.",
        "- Semantic phase uses cached embeddings; pass `force_recompute=True` if you",
        "  switch models.",
        "- EB-NeRD `title+abstract+body` ablation indexes subtitle (EB-NeRD abstract",
        "  equivalent) + body text; MIND has no body column so the flag is a no-op there.",
        "",
    ]
    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n>> Ablation report written to: {OUTPUT_MD}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"Ablation runner started  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Processed dir : {PROCESSED_DIR}")
    print(f"Output file   : {OUTPUT_MD}")

    print("\n" + "#"*60)
    print("# PHASE 1: BM25 Input Fields")
    print("#"*60)
    p1_rank, p1_retr = phase1()

    print("\n" + "#"*60)
    print("# PHASE 2: History Length")
    print("#"*60)
    p2_rank, p2_retr, best_len = phase2()

    print("\n" + "#"*60)
    print(f"# PHASE 3: Cold-Start Threshold  (best_len={best_len})")
    print("#"*60)
    p3_rank, p3_retr = phase3(best_len)

    print("\n" + "#"*60)
    print(f"# PHASE 4: Semantic Index Type  (best_len={best_len})")
    print("#"*60)
    p4_rank, p4_retr = phase4(best_len)

    write_report(
        p1_rank, p1_retr,
        p2_rank, p2_retr, best_len,
        p3_rank, p3_retr,
        p4_rank, p4_retr,
    )
    print(f"\nAblation runner done  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
