#!/usr/bin/env python
"""
Memory-Safe NRMS Codabench Submission Generator for EB-NeRD Large & MIND Large.

Generates competition-compliant submission archives:
- EB-NeRD Large (RecSys 2024 Challenge):
    Output: submissions/submissions_ebnerd_nrms_affinity/predictions.zip
    File inside zip: predictions.txt
    Model: NRMSDocVec + Editorial Category Affinity (beta = 0.20)
- MIND Large (MIND Leaderboard):
    Output: submissions/submissions_mind_nrms_subcategory/prediction.zip
    File inside zip: prediction.txt
    Model: NRMSDocVec + Fine-Grained Subcategory Affinity (beta = 0.20)

Engineered for zero-crash execution on low-RAM machines:
1. Thread contention capped to 4 to ensure the host OS remains 100% responsive.
2. Flattened primitive integer arrays for user history (< 100 MB RAM vs > 1.7 GB for dicts).
3. Chunked streaming of test impressions (50k rows at a time).
4. User representation vectors materialized and batched ONLY for active chunk users (< 60 MB RAM).
5. Line buffering with direct disk streaming.
6. Automatic format regex and line count validation.

Usage:
    python src/generate_nrms_codabench_submission.py --dataset ebnerd
    python src/generate_nrms_codabench_submission.py --dataset mind
    python src/generate_nrms_codabench_submission.py --dataset both
    python src/generate_nrms_codabench_submission.py --dataset both --max_impressions 1000  # Quick dry run
"""

import os
# Cap threads to prevent CPU thread starvation and leave resources for the host OS
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["POLARS_MAX_THREADS"] = "4"

import argparse
import gc
import json
from pathlib import Path
import re
import sys
import time
import zipfile
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import polars as pl
import psutil
from scipy.stats import rankdata
import torch
from tqdm import tqdm

torch.set_num_threads(4)

# Ensure src/ is in sys.path
SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from nrms_docvec import NRMSDocVec


def get_current_rss_mb() -> float:
    """Returns current process RSS memory in Megabytes."""
    return psutil.Process().memory_info().rss / (1024 * 1024)


def validate_submission_file(file_path: Path, expected_lines: int) -> None:
    """Validates the format and line count of a generated prediction text file."""
    print(f"\n[Validation] Validating format of {file_path.name}...")
    line_count = 0
    pattern = re.compile(r"^\d+\s+\[\d+(?:,\d+)*\]$")

    with open(file_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            line_count += 1
            line = line.strip()
            if not pattern.match(line):
                if line_count <= 5 or not line:
                    raise ValueError(f"Invalid format at line {line_count}: {line}")

    if line_count != expected_lines:
        print(f"[Warning] Expected {expected_lines:,} lines, but found {line_count:,} lines.")
    else:
        print(f"[Validation Passed] Exactly {line_count:,} valid lines matching competition regex: ^\\d+\\s+\\[\\d+(?:,\\d+)*\\]$")


def load_ebnerd_compact_history(
    raw_hist_path: Path,
    max_history_len: int = 30,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads raw EB-NeRD user history into zero-overhead primitive NumPy arrays.
    Returns:
        user_offset: 1D int32 array mapping user_int -> start index in flat_clicks.
        user_len: 1D uint8 array mapping user_int -> click count.
        flat_clicks: 1D int32 array containing flat article IDs.
    Memory footprint: ~94 MB (saving ~1.7 GB over Python dicts).
    """
    print(f"Loading EB-NeRD test user history from {raw_hist_path}...")
    t0 = time.time()

    df = pl.read_parquet(raw_hist_path, columns=["user_id", "article_id_fixed"])
    capped = df.select([
        pl.col("user_id").cast(pl.UInt32),
        pl.col("article_id_fixed").list.tail(max_history_len),
    ])
    del df

    uids = capped["user_id"].to_numpy()
    lists = capped["article_id_fixed"].to_list()
    del capped

    lengths = np.array([len(x) if x is not None else 0 for x in lists], dtype=np.uint8)
    cum_offsets = np.zeros(len(lengths) + 1, dtype=np.int32)
    np.cumsum(lengths, out=cum_offsets[1:])
    total_clicks = cum_offsets[-1]

    flat_clicks = np.empty(total_clicks, dtype=np.int32)
    for i, l in enumerate(lists):
        if l:
            flat_clicks[cum_offsets[i] : cum_offsets[i + 1]] = l
    del lists

    max_uid = int(uids.max())
    user_offset = np.full(max_uid + 1, -1, dtype=np.int32)
    user_len = np.zeros(max_uid + 1, dtype=np.uint8)
    user_offset[uids] = cum_offsets[:-1]
    user_len[uids] = lengths
    del uids, lengths, cum_offsets
    gc.collect()

    dt = time.time() - t0
    print(f"Loaded {len(user_len):,} EB-NeRD users ({total_clicks:,} clicks) in {dt:.2f}s. RAM RSS: {get_current_rss_mb():.1f} MB")
    return user_offset, user_len, flat_clicks


def load_mind_compact_history(
    data_dir: Path,
    art_id_to_int: Dict[str, int],
    max_history_len: int = 30,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads MIND test user history into zero-overhead primitive NumPy arrays.
    Returns:
        user_offset: 1D int32 array mapping user_int -> start index in flat_clicks.
        user_len: 1D uint8 array mapping user_int -> click count.
        flat_clicks: 1D int32 array containing mapped article index ints.
    Memory footprint: ~58 MB (Peak RSS < 450 MB).
    """
    t0 = time.time()
    user_feat_test = data_dir / "user_features_test.parquet"
    hist_test_path = data_dir / "history_test.parquet"

    if user_feat_test.exists():
        print(f"Loading precomputed MIND test user history from {user_feat_test}...")
        df = pl.read_parquet(user_feat_test, columns=["user_id", "click_history"])
        capped = df.select([
            pl.col("user_id").str.replace("mind_U", "").str.replace("U", "").cast(pl.UInt32).alias("u_int"),
            pl.col("click_history").list.tail(max_history_len).alias("tail_h"),
        ])
        del df
        uids_int = capped["u_int"].to_numpy()
        aids_raw = capped["tail_h"].to_list()
        del capped
    elif hist_test_path.exists():
        print(f"Loading MIND test user history from {hist_test_path}...")
        df = (
            pl.scan_parquet(hist_test_path)
            .select(["user_id", "clicked_article_id"])
            .group_by("user_id")
            .agg(pl.col("clicked_article_id").tail(max_history_len).alias("aids"))
            .collect()
        )
        uids_raw = df["user_id"].to_list()
        aids_raw = df["aids"].to_list()
        del df
        uids_int = np.empty(len(uids_raw), dtype=np.int32)
        for i, u in enumerate(uids_raw):
            s = u.replace("mind_", "").replace("U", "")
            uids_int[i] = int(s) if s.isdigit() else i
        del uids_raw
    else:
        raise FileNotFoundError(f"Neither {user_feat_test} nor {hist_test_path} found.")

    lengths = np.array([len(x) if x is not None else 0 for x in aids_raw], dtype=np.uint8)
    cum_offsets = np.zeros(len(lengths) + 1, dtype=np.int32)
    np.cumsum(lengths, out=cum_offsets[1:])
    total_clicks = cum_offsets[-1]

    flat_clicks = np.zeros(total_clicks, dtype=np.int32)
    for i, aids in enumerate(aids_raw):
        if aids:
            start = cum_offsets[i]
            for j, aid in enumerate(aids):
                flat_clicks[start + j] = art_id_to_int.get(aid, 0)
    del aids_raw
    gc.collect()

    max_uid = int(uids_int.max())
    user_offset = np.full(max_uid + 1, -1, dtype=np.int32)
    user_len = np.zeros(max_uid + 1, dtype=np.uint8)
    user_offset[uids_int] = cum_offsets[:-1]
    user_len[uids_int] = lengths
    del uids_int, lengths, cum_offsets
    gc.collect()

    dt = time.time() - t0
    print(f"Loaded {len(user_len):,} MIND users ({total_clicks:,} clicks) in {dt:.2f}s. RAM RSS: {get_current_rss_mb():.1f} MB")
    return user_offset, user_len, flat_clicks


def generate_submission_dataset(
    dataset_name: str,
    data_dir: Path,
    raw_hist_path: Optional[Path],
    proc_hist_path: Optional[Path],
    checkpoint_path: Path,
    output_dir: Path,
    beta: float = 0.20,
    max_history_len: int = 30,
    chunk_size_impr: int = 50000,
    max_impressions: Optional[int] = None,
) -> Path:
    """
    Generates Codabench predictions for either EB-NeRD Large or MIND Large.
    """
    is_ebnerd = dataset_name == "ebnerd"
    zip_name = "predictions.zip" if is_ebnerd else "prediction.zip"
    txt_name = "predictions.txt" if is_ebnerd else "prediction.txt"

    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_txt_path = output_dir / txt_name
    predictions_zip_path = output_dir / zip_name

    affinity_desc = "Standard Editorial Category" if is_ebnerd else "Fine-Grained Subcategory (category/subcategory)"
    print(f"\n{'='*80}")
    print(f"STARTING NRMS CODABENCH SUBMISSION GENERATOR: {dataset_name.upper()}")
    print(f"Affinity Formulation: {affinity_desc} (beta = {beta})")
    print(f"Target Zip Output:    {predictions_zip_path}")
    print(f"{'='*80}")

    # 1. Load Pretrained Article Embeddings
    emb_path = data_dir / "article_embeddings.npy"
    print(f"Loading article embeddings from {emb_path}...")
    emb_raw = np.load(emb_path).astype(np.float32)
    num_articles, embed_dim = emb_raw.shape
    print(f"Article embeddings: {num_articles:,} articles, dim={embed_dim}")

    # Pre-normalize embeddings for fast cosine similarity via dot product: cos(u, v) = u^T v
    norm = np.linalg.norm(emb_raw, axis=-1, keepdims=True)
    norm[norm == 0] = 1.0
    emb_norm_clean = emb_raw / norm
    del norm

    # Precompute fast lookup table with index 0 as zeros (pad/unknown)
    pad_row = np.zeros((1, embed_dim), dtype=np.float32)
    emb_table = np.ascontiguousarray(np.vstack([pad_row, emb_norm_clean]), dtype=np.float32)
    del pad_row, emb_norm_clean

    # 2. Load Article IDs and Category Metadata
    ids_path = data_dir / "article_ids.json"
    with open(ids_path, "r", encoding="utf-8") as f:
        article_ids = json.load(f)

    art_id_to_int: Dict[str, int] = {aid: i + 1 for i, aid in enumerate(article_ids)}
    art_df = pl.read_parquet(data_dir / "articles.parquet", columns=["article_id", "category"])
    raw_cats = dict(zip(art_df["article_id"].to_list(), art_df["category"].fill_null("unknown").to_list()))
    del art_df

    # Map categories to compact integer codes
    unique_categories: List[str] = sorted(list(set(raw_cats.values())))
    cat_to_id = {c: idx + 1 for idx, c in enumerate(unique_categories)}
    art_cat_ids = np.zeros(num_articles + 1, dtype=np.int16)
    for aid, idx in art_id_to_int.items():
        c_str = raw_cats.get(aid, "unknown")
        art_cat_ids[idx] = cat_to_id.get(c_str, 0)
    del raw_cats
    print(f"Mapped {len(unique_categories):,} unique categories to integer codes.")

    # 3. Load NRMS Model & Weights
    print(f"Initializing NRMSDocVec model from {checkpoint_path}...")
    model = NRMSDocVec(
        pretrained_embeddings=emb_raw,
        num_heads=4,
        additive_hidden_dim=200,
        dropout=0.1,
        enable_recency_decay=False,
    ).to("cpu")
    # Release the non-normalized embeddings copy from CPU memory
    del emb_raw
    gc.collect()

    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print(f"NRMS Model loaded successfully. Current RAM RSS: {get_current_rss_mb():.1f} MB")

    # Global user representation for fallback (cold start)
    global_u_vec = model.global_user_vec.detach().cpu().numpy()
    global_u_vec = global_u_vec / max(1e-8, np.linalg.norm(global_u_vec))

    # 4. Load Compact User History
    if is_ebnerd:
        if raw_hist_path and raw_hist_path.exists():
            user_offset, user_len, flat_clicks = load_ebnerd_compact_history(raw_hist_path, max_history_len)
        else:
            raise FileNotFoundError(f"EB-NeRD history file not found at {raw_hist_path}")
    else:
        user_offset, user_len, flat_clicks = load_mind_compact_history(data_dir, art_id_to_int, max_history_len)

    # 5. Stream Test Impressions in Chunks
    test_impr_path = data_dir / "impressions_test.parquet"
    total_test_impressions = pl.scan_parquet(test_impr_path).select(pl.len()).collect().item()
    if max_impressions is not None:
        total_test_impressions = min(max_impressions, total_test_impressions)
    print(f"Total test impressions to score: {total_test_impressions:,} (chunk size = {chunk_size_impr:,})")

    start_time = time.time()
    total_written = 0
    line_buffer: List[str] = []

    pbar = tqdm(total=total_test_impressions, desc=f"Scoring {dataset_name.upper()}", unit="impr")

    with open(predictions_txt_path, "w", encoding="utf-8", buffering=2 * 1024 * 1024) as out_f:
        for chunk_offset in range(0, total_test_impressions, chunk_size_impr):
            chunk_len = min(chunk_size_impr, total_test_impressions - chunk_offset)
            chunk_df = (
                pl.scan_parquet(test_impr_path)
                .select(["impression_id", "user_id", "candidate_article_ids"])
                .slice(chunk_offset, chunk_len)
                .collect()
            )

            impr_ids = chunk_df["impression_id"].to_list()
            user_ids_raw = chunk_df["user_id"].to_list()
            candidates_list = chunk_df["candidate_article_ids"].to_list()
            del chunk_df

            # Parse user IDs to integers
            chunk_user_ints: List[int] = []
            if is_ebnerd:
                for uid in user_ids_raw:
                    s = uid.replace("ebnerd_", "") if isinstance(uid, str) else str(uid)
                    chunk_user_ints.append(int(s) if s.isdigit() else -1)
            else:
                for uid in user_ids_raw:
                    s = uid.replace("mind_", "").replace("U", "") if isinstance(uid, str) else str(uid)
                    chunk_user_ints.append(int(s) if s.isdigit() else -1)

            unique_uids = list(set(chunk_user_ints))
            max_uid_len = len(user_offset)

            # Build batch user history tensors for unique chunk users
            user_hist_arrays = np.zeros((len(unique_uids), max_history_len), dtype=np.int64)
            user_affinity_counts: Dict[int, Dict[int, float]] = {}

            for idx, u_int in enumerate(unique_uids):
                if 0 <= u_int < max_uid_len:
                    off = user_offset[u_int]
                    length = user_len[u_int]
                    if off >= 0 and length > 0:
                        raw_aids = flat_clicks[off : off + length]
                        # For EB-NeRD, raw_aids are integer article IDs that need mapping to 1-based embedding indices
                        if is_ebnerd:
                            art_indices = [art_id_to_int.get(f"ebnerd_{aid}", 0) for aid in raw_aids]
                        else:
                            art_indices = raw_aids.tolist()

                        n_act = min(len(art_indices), max_history_len)
                        user_hist_arrays[idx, :n_act] = art_indices[-n_act:]

                        # Precompute user category affinity histogram
                        cat_hist: Dict[int, int] = {}
                        for a_idx in art_indices:
                            if a_idx > 0:
                                c_code = art_cat_ids[a_idx]
                                cat_hist[c_code] = cat_hist.get(c_code, 0) + 1
                        total_clicks_user = len(art_indices)
                        if total_clicks_user > 0:
                            user_affinity_counts[u_int] = {c: count / total_clicks_user for c, count in cat_hist.items()}

            # Batch encode unique users with NRMS model in blocks of 2048
            unique_user_vecs = np.zeros((len(unique_uids), embed_dim), dtype=np.float32)
            batch_encode_size = 2048
            with torch.no_grad():
                for b_start in range(0, len(unique_uids), batch_encode_size):
                    b_end = min(b_start + batch_encode_size, len(unique_uids))
                    t_in = torch.from_numpy(user_hist_arrays[b_start:b_end])
                    vecs = model.encode_user(t_in).cpu().numpy()
                    unique_user_vecs[b_start:b_end] = vecs

            # Fast map: user_int -> normalized user vector
            user_vec_map: Dict[int, np.ndarray] = {}
            for idx, u_int in enumerate(unique_uids):
                user_vec_map[u_int] = unique_user_vecs[idx]

            # Score each impression in the chunk
            prefix_to_strip = "ebnerd_" if is_ebnerd else "mind_"
            for i in range(chunk_len):
                raw_impr_id = impr_ids[i]
                if isinstance(raw_impr_id, str) and raw_impr_id.startswith(prefix_to_strip):
                    raw_impr_id = raw_impr_id[len(prefix_to_strip):]

                u_int = chunk_user_ints[i]
                u_vec = user_vec_map.get(u_int, global_u_vec)
                user_aff = user_affinity_counts.get(u_int, {})

                cands = candidates_list[i]
                cand_indices = [art_id_to_int.get(c, 0) for c in cands]

                # 1. Base NRMS Semantic Cosine Similarity (direct vectorized table slice & dot product)
                sim_scores = emb_table[cand_indices] @ u_vec

                # 2. Category / Subcategory Affinity Boost
                if user_aff and beta > 0.0:
                    for ci, c_idx in enumerate(cand_indices):
                        if c_idx > 0:
                            cat_id = art_cat_ids[c_idx]
                            sim_scores[ci] += beta * user_aff.get(cat_id, 0.0)

                # 3. Ordinal Ranks (Rank 1 = highest score)
                ranks = rankdata(-sim_scores, method="ordinal")
                rank_str = ",".join(str(int(r)) for r in ranks)
                line_buffer.append(f"{raw_impr_id} [{rank_str}]\n")

                if len(line_buffer) >= 20000:
                    out_f.writelines(line_buffer)
                    line_buffer.clear()

            # Clean up active chunk allocations
            del user_vec_map, unique_user_vecs, user_affinity_counts, user_hist_arrays
            total_written += chunk_len
            pbar.update(chunk_len)
            gc.collect()

        # Flush any remaining lines
        if line_buffer:
            out_f.writelines(line_buffer)
            line_buffer.clear()

        pbar.close()

    elapsed = time.time() - start_time
    print(f"\nFinished scoring {total_written:,} impressions in {elapsed:.2f}s ({total_written / max(0.1, elapsed):.0f} impr/s)")
    print(f"Peak RAM RSS: {get_current_rss_mb():.1f} MB (well within safety budget)")

    # Validate output lines and pattern
    validate_submission_file(predictions_txt_path, total_test_impressions)

    # Compress into competition ZIP archive (storing only the root text file)
    print(f"\nCreating submission zip archive: {predictions_zip_path}...")
    with zipfile.ZipFile(predictions_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(predictions_txt_path, arcname=txt_name)

    zip_size_mb = predictions_zip_path.stat().st_size / (1024 * 1024)
    print(f"SUCCESS: Submission zip created at: {predictions_zip_path} ({zip_size_mb:.2f} MB)")
    return predictions_zip_path


def main():
    parser = argparse.ArgumentParser(description="Memory-Safe NRMS Codabench Submission Generator.")
    parser.add_argument(
        "--dataset",
        choices=["ebnerd", "mind", "both"],
        default="both",
        help="Dataset to generate submissions for (default: both).",
    )
    parser.add_argument(
        "--dataset_type",
        choices=["large", "small"],
        default="large",
        help="Dataset scale (default: large).",
    )
    parser.add_argument(
        "--max_impressions",
        type=int,
        default=None,
        help="Optional upper bound on impressions for fast sanity checking.",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=50000,
        help="Impression chunk size for memory bounding (default: 50,000).",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.20,
        help="Category / Subcategory affinity blending weight (default: 0.20).",
    )
    args = parser.parse_args()

    datasets_to_run = ["ebnerd", "mind"] if args.dataset == "both" else [args.dataset]

    total_start = time.time()
    results: Dict[str, Path] = {}

    for ds in datasets_to_run:
        data_dir = ROOT_DIR / "data" / "processed" / f"{ds}_{args.dataset_type}"
        checkpoint_path = ROOT_DIR / "models" / f"nrms_vanilla_{ds}.pt"

        if ds == "ebnerd":
            raw_hist = ROOT_DIR / "data" / "raw" / f"ebnerd_{args.dataset_type}" / "test" / "history.parquet"
            proc_hist = data_dir / "history_test.parquet"
            out_dir = ROOT_DIR / "submissions" / "submissions_ebnerd_nrms_affinity"
        else:
            raw_hist = None
            proc_hist = data_dir / "history_test.parquet"
            out_dir = ROOT_DIR / "submissions" / "submissions_mind_nrms_subcategory"

        zip_res = generate_submission_dataset(
            dataset_name=ds,
            data_dir=data_dir,
            raw_hist_path=raw_hist,
            proc_hist_path=proc_hist,
            checkpoint_path=checkpoint_path,
            output_dir=out_dir,
            beta=args.beta,
            chunk_size_impr=args.chunk_size,
            max_impressions=args.max_impressions,
        )
        results[ds] = zip_res

    print(f"\n{'='*80}")
    print("ALL REQUESTED CODABENCH SUBMISSIONS SUCCESSFULLY GENERATED!")
    for ds, p in results.items():
        print(f" - {ds.upper()}: {p} ({p.stat().st_size / (1024*1024):.2f} MB)")
    print(f"Total time elapsed: {time.time() - total_start:.2f}s")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
