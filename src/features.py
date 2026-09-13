"""
Behavioral and Session Feature Engineering for News Recommendation (Assignment 2).

Includes strict point-in-time boundary enforcement to ensure no future clicks
leak into features at training or serving time (Q1.4, Q9).
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Union
import numpy as np
import pandas as pd
import polars as pl

from split import history_as_of

HALF_LIFE_DEFAULT_HOURS = 24.0


def filter_history_point_in_time(
    history: Union[pl.DataFrame, pd.DataFrame],
    impression_time: Union[datetime, pd.Timestamp],
    user_id: Optional[str] = None,
) -> Union[pl.DataFrame, pd.DataFrame]:
    """
    Filter user history strictly before impression_time (click_time < impression_time).
    Enforces the behavioural-window boundary (Q1.4, Q9) preventing any future-click leakage.
    """
    is_pandas = isinstance(history, pd.DataFrame)
    df = pl.from_pandas(history) if is_pandas else history

    if df.is_empty():
        return history.copy() if is_pandas else df

    pred = pl.col("click_time") < impression_time
    if user_id is not None:
        pred = pred & (pl.col("user_id") == user_id)

    filtered = df.filter(pred)
    return filtered.to_pandas() if is_pandas else filtered


def compute_user_behavior_features(
    history: Union[pl.DataFrame, pd.DataFrame],
    impression_time: Union[datetime, pd.Timestamp],
    user_id: str,
    articles_df: Optional[Union[pl.DataFrame, pd.DataFrame]] = None,
    half_life_hours: float = HALF_LIFE_DEFAULT_HOURS,
) -> Dict[str, Any]:
    """
    Extract point-in-time behavioural features for a user at the exact moment of an impression:
    1. click_count: Total clicks prior to impression_time.
    2. recency_score: Exponential decay-weighted history: sum(0.5 ** (age_hours / half_life)).
    3. recent_clicked_ids: List of clicked article IDs (most recent first).
    4. category_affinity: Normalized category frequency distribution from historical clicks.
    5. mean_dwell_time: Average dwell time (seconds) if available, else 0.0.
    """
    is_pandas = isinstance(history, pd.DataFrame)
    df = pl.from_pandas(history) if is_pandas else history

    # Strict point-in-time filter
    hist = df.filter((pl.col("user_id") == user_id) & (pl.col("click_time") < impression_time))

    if hist.is_empty():
        return {
            "user_id": user_id,
            "click_count": 0,
            "recency_score": 0.0,
            "recent_clicked_ids": [],
            "category_affinity": {},
            "mean_dwell_time": 0.0,
        }

    # Sort descending by click_time (most recent first)
    hist = hist.sort("click_time", descending=True)
    clicked_ids = hist["clicked_article_id"].to_list()

    # Recency weighting: weight = 0.5 ** (age_hours / half_life_hours)
    # age_hours = (impression_time - click_time) in hours
    hist_weights = hist.with_columns(
        ((pl.lit(impression_time) - pl.col("click_time")).dt.total_seconds() / 3600.0).alias("age_hours")
    ).with_columns(
        (0.5 ** (pl.col("age_hours") / half_life_hours)).alias("decay_weight")
    )
    recency_score = float(hist_weights["decay_weight"].sum())

    # Dwell time if available in history (e.g. EB-NeRD)
    mean_dwell = 0.0
    if "dwell_time" in hist.columns:
        dwell_vals = hist["dwell_time"].drop_nulls()
        if not dwell_vals.is_empty():
            mean_dwell = float(dwell_vals.mean())

    # Category affinity distribution
    category_affinity: Dict[str, float] = {}
    if articles_df is not None:
        art_df = pl.from_pandas(articles_df) if isinstance(articles_df, pd.DataFrame) else articles_df
        if not art_df.is_empty() and "category" in art_df.columns:
            joined = hist.join(art_df.select(["article_id", "category"]), left_on="clicked_article_id", right_on="article_id", how="inner")
            if not joined.is_empty():
                cat_counts = joined["category"].value_counts()
                total_cats = len(joined)
                for row in cat_counts.iter_rows():
                    cat_name, cnt = row[0], row[1]
                    if cat_name:
                        category_affinity[cat_name] = cnt / total_cats

    return {
        "user_id": user_id,
        "click_count": len(clicked_ids),
        "recency_score": recency_score,
        "recent_clicked_ids": clicked_ids,
        "category_affinity": category_affinity,
        "mean_dwell_time": mean_dwell,
    }


def compute_article_features(
    article_id: str,
    impression_time: Union[datetime, pd.Timestamp],
    articles_df: Union[pl.DataFrame, pd.DataFrame],
    history_df: Optional[Union[pl.DataFrame, pd.DataFrame]] = None,
    popularity_window_hours: float = 24.0,
) -> Dict[str, Any]:
    """
    Extract article features:
    1. freshness_hours: Time since publication (impression_time - published_time).
    2. historical_popularity: Total clicks received by this article in the preceding window (e.g. 24h),
       strictly prior to impression_time (preventing future-leakage).
    3. category: Article topic category.
    """
    art_df = pl.from_pandas(articles_df) if isinstance(articles_df, pd.DataFrame) else articles_df
    row = art_df.filter(pl.col("article_id") == article_id)

    category = ""
    freshness_hours = 0.0
    if not row.is_empty():
        category = str(row["category"][0]) if row["category"][0] is not None else ""
        pub_time = row["published_time"][0]
        if pub_time is not None:
            if isinstance(pub_time, (datetime, pd.Timestamp)):
                diff_sec = (impression_time - pub_time).total_seconds()
                freshness_hours = max(0.0, diff_sec / 3600.0)

    # Point-in-time popularity: clicks strictly in [impression_time - window, impression_time)
    popularity = 0
    if history_df is not None:
        h_df = pl.from_pandas(history_df) if isinstance(history_df, pd.DataFrame) else history_df
        if not h_df.is_empty() and "click_time" in h_df.columns:
            window_start = impression_time - timedelta(hours=popularity_window_hours)
            pop_hits = h_df.filter(
                (pl.col("clicked_article_id") == article_id)
                & (pl.col("click_time") >= window_start)
                & (pl.col("click_time") < impression_time)
            )
            popularity = len(pop_hits)

    return {
        "article_id": article_id,
        "category": category,
        "freshness_hours": freshness_hours,
        "popularity_24h": popularity,
    }


def compute_cross_features(
    user_feats: Dict[str, Any],
    article_feats: Dict[str, Any],
    session_position: int = 0,
    bm25_score: float = 0.0,
    semantic_score: float = 0.0,
    first_stage_score: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Combines user behavioral signals with candidate article metadata:
    - category_affinity_score: Overlap probability between user reading history and article category.
    - session_position: 0-indexed candidate position (captures presentation bias).
    - bm25_score: Lexical retrieval score.
    - semantic_score: Dense embedding retrieval / cosine similarity score.
    - first_stage_score: Retained for backward compatibility (defaults to max(bm25, semantic)).
    """
    art_cat = article_feats.get("category", "")
    affinity_map = user_feats.get("category_affinity", {})
    cat_affinity_score = affinity_map.get(art_cat, 0.0)

    resolved_first_stage = (
        first_stage_score if first_stage_score is not None else max(bm25_score, semantic_score)
    )

    return {
        "user_click_count": user_feats.get("click_count", 0),
        "user_recency_score": user_feats.get("recency_score", 0.0),
        "user_mean_dwell_time": user_feats.get("mean_dwell_time", 0.0),
        "article_freshness_hours": article_feats.get("freshness_hours", 0.0),
        "article_popularity_24h": article_feats.get("popularity_24h", 0),
        "category_affinity_score": cat_affinity_score,
        "session_position": session_position,
        "bm25_score": bm25_score,
        "semantic_score": semantic_score,
        "first_stage_score": resolved_first_stage,
    }


def build_feature_dataset(
    impressions_df: Union[pl.DataFrame, pd.DataFrame],
    articles_df: Union[pl.DataFrame, pd.DataFrame],
    history_df: Optional[Union[pl.DataFrame, pd.DataFrame]] = None,
    embeddings: Optional[np.ndarray] = None,
    article_id_to_idx: Optional[Dict[str, int]] = None,
    max_impressions: Optional[int] = None,
    negative_sampling_ratio: Optional[int] = None,
    half_life_hours: float = HALF_LIFE_DEFAULT_HOURS,
    show_progress: bool = True,
    bm25_retriever: Optional[Any] = None,
    art_token_ids_map: Optional[Dict[str, Set[int]]] = None,
    art_cat_map: Optional[Dict[str, str]] = None,
    art_pub_map: Optional[Dict[str, Any]] = None,
    user_history_map: Optional[Dict[str, List[tuple]]] = None,
) -> pl.DataFrame:
    """
    Construct a tabular feature dataset for GBDT training or offline re-ranking.
    Enforces point-in-time boundary with zero leakage (Q1.4, Q9).
    High-performance vectorised column accumulation.
    """
    from embeddings import compute_user_representation, normalize_l2

    imp_df = pl.from_pandas(impressions_df) if isinstance(impressions_df, pd.DataFrame) else impressions_df
    art_df = pl.from_pandas(articles_df) if isinstance(articles_df, pd.DataFrame) else articles_df

    if max_impressions is not None and len(imp_df) > max_impressions:
        imp_df = imp_df.slice(0, max_impressions)

    # 1. Pre-normalize embeddings ONCE upfront for ultra-fast dot products
    embeddings_norm = None
    if embeddings is not None:
        embeddings_norm = np.ascontiguousarray(normalize_l2(embeddings), dtype=np.float32)

    # 2. Build fast O(1) article metadata lookups
    art_ids = art_df["article_id"].to_list()
    if article_id_to_idx is None:
        article_id_to_idx = {aid: idx for idx, aid in enumerate(art_ids)}

    if art_cat_map is None:
        categories = art_df["category"].fill_null("").to_list() if "category" in art_df.columns else [""] * len(art_ids)
        art_cat_map = dict(zip(art_ids, categories))

    if art_pub_map is None:
        pub_times = art_df["published_time"].to_list() if "published_time" in art_df.columns else [None] * len(art_ids)
        art_pub_map = dict(zip(art_ids, pub_times))

    # 3. Build BM25 index over article title + abstract for fast exact lexical scoring if not provided
    if bm25_retriever is None or art_token_ids_map is None:
        try:
            import bm25s
            titles = art_df["title"].fill_null("").to_list() if "title" in art_df.columns else [""] * len(art_ids)
            if "abstract" in art_df.columns:
                abstracts = art_df["abstract"].fill_null("").to_list()
            elif "subtitle" in art_df.columns:
                abstracts = art_df["subtitle"].fill_null("").to_list()
            else:
                abstracts = [""] * len(art_ids)
            full_texts = [f"{t} {a}".strip() for t, a in zip(titles, abstracts)]
            tokens = bm25s.tokenize(full_texts, show_progress=False)
            bm25_retriever = bm25s.BM25()
            bm25_retriever.index(tokens, show_progress=False)
            art_token_ids_map = {aid: set(tokens.ids[i]) for i, aid in enumerate(art_ids)}
        except Exception as e:
            bm25_retriever = None
            art_token_ids_map = {}

    # 4. Build fast point-in-time user history map if not provided
    if user_history_map is None:
        user_history_map = {}
        if history_df is not None:
            h_df = pl.from_pandas(history_df) if isinstance(history_df, pd.DataFrame) else history_df
            if not h_df.is_empty() and "click_time" in h_df.columns:
                has_dwell = "dwell_time" in h_df.columns
                agg_exprs = [
                    pl.col("clicked_article_id").alias("aids"),
                    pl.col("click_time").alias("times"),
                ]
                if has_dwell:
                    agg_exprs.append(pl.col("dwell_time").fill_null(0.0).alias("dwells"))
                user_agg = h_df.sort("click_time").group_by("user_id").agg(agg_exprs)

                uids = user_agg["user_id"].to_list()
                aids_list = user_agg["aids"].to_list()
                times_list = user_agg["times"].to_list()
                if has_dwell:
                    dwells_list = user_agg["dwells"].to_list()
                    for uid, aids, times, dwells in zip(uids, aids_list, times_list, dwells_list):
                        user_history_map[uid] = list(zip(aids, times, dwells))
                else:
                    for uid, aids, times in zip(uids, aids_list, times_list):
                        user_history_map[uid] = [(aid, t, 0.0) for aid, t in zip(aids, times)]

    # 5. Column lists for fast Polars construction
    col_imp_id = []
    col_user_id = []
    col_art_id = []
    col_label = []
    col_click_count = []
    col_recency_score = []
    col_mean_dwell = []
    col_freshness = []
    col_popularity = []
    col_cat_affinity = []
    col_position = []
    col_bm25 = []
    col_semantic = []
    col_first_stage = []

    has_labels = "clicked_article_ids" in imp_df.columns

    try:
        from tqdm import tqdm
        iterator = tqdm(imp_df.iter_rows(named=True), total=len(imp_df), desc="Extracting features") if show_progress else imp_df.iter_rows(named=True)
    except ImportError:
        iterator = imp_df.iter_rows(named=True)

    for imp in iterator:
        imp_id = imp["impression_id"]
        uid = imp["user_id"]
        imp_time = imp["timestamp"]
        cands = imp["candidate_article_ids"] or []
        if not cands:
            continue

        clicked_set = set(imp.get("clicked_article_ids") or []) if has_labels else set()

        # Strict point-in-time history filter: click_time < imp_time
        user_clicks = user_history_map.get(uid, [])
        past_clicks = [c for c in user_clicks if c[1] < imp_time]

        n_clicks = len(past_clicks)
        recency_score = 0.0
        mean_dwell = 0.0
        cat_counts: Dict[str, int] = {}
        history_aids = []
        history_weights = []

        if n_clicks > 0:
            dwell_sum = 0.0
            for aid, c_time, dwell in past_clicks:
                history_aids.append(aid)
                diff_hours = (imp_time - c_time).total_seconds() / 3600.0
                weight = 0.5 ** (diff_hours / half_life_hours)
                recency_score += weight
                history_weights.append(weight)
                dwell_sum += dwell
                cat = art_cat_map.get(aid, "")
                if cat:
                    cat_counts[cat] = cat_counts.get(cat, 0) + 1
            mean_dwell = dwell_sum / n_clicks

        cat_affinity = {k: v / n_clicks for k, v in cat_counts.items()} if n_clicks > 0 else {}

        # Compute user representation vector for semantic similarity
        user_vec = None
        if embeddings_norm is not None and article_id_to_idx is not None and history_aids:
            try:
                user_vec = compute_user_representation(
                    clicked_article_ids=history_aids,
                    article_id_to_idx=article_id_to_idx,
                    embeddings=embeddings_norm,
                    weights=history_weights,
                )
            except Exception:
                user_vec = None

        # Compute point-in-time BM25 scores for candidate articles
        bm25_all_scores = None
        if bm25_retriever is not None and history_aids:
            user_q_tokens = set()
            for aid in history_aids[-20:]:
                user_q_tokens.update(art_token_ids_map.get(aid, set()))
            if user_q_tokens:
                bm25_all_scores = bm25_retriever.get_scores(list(user_q_tokens))

        # Negative sampling if requested
        if negative_sampling_ratio is not None and has_labels and len(clicked_set) > 0:
            pos_cands = [c for c in cands if c in clicked_set]
            neg_cands = [c for c in cands if c not in clicked_set]
            n_neg = min(len(neg_cands), len(pos_cands) * negative_sampling_ratio)
            selected_cands = pos_cands + neg_cands[:n_neg]
        else:
            selected_cands = cands

        for pos_idx, cand_id in enumerate(selected_cands):
            cat = art_cat_map.get(cand_id, "")
            pub = art_pub_map.get(cand_id)
            freshness = 0.0
            if pub is not None and isinstance(pub, (datetime, pd.Timestamp)):
                freshness = max(0.0, (imp_time - pub).total_seconds() / 3600.0)

            cat_aff = cat_affinity.get(cat, 0.0)

            # Fast single dot product
            sem_score = 0.0
            c_idx = article_id_to_idx.get(cand_id)
            if user_vec is not None and embeddings_norm is not None and c_idx is not None:
                if c_idx < len(embeddings_norm):
                    sem_score = float(np.dot(user_vec, embeddings_norm[c_idx]))

            # Exact BM25 lexical score
            bm25_score = 0.0
            if bm25_all_scores is not None and c_idx is not None:
                if c_idx < len(bm25_all_scores):
                    bm25_score = float(bm25_all_scores[c_idx])

            first_stage = max(bm25_score, sem_score)
            label = 1 if cand_id in clicked_set else 0

            col_imp_id.append(imp_id)
            col_user_id.append(uid)
            col_art_id.append(cand_id)
            col_label.append(label)
            col_click_count.append(n_clicks)
            col_recency_score.append(float(recency_score))
            col_mean_dwell.append(float(mean_dwell))
            col_freshness.append(float(freshness))
            col_popularity.append(0)
            col_cat_affinity.append(float(cat_aff))
            col_position.append(pos_idx)
            col_bm25.append(float(bm25_score))
            col_semantic.append(float(sem_score))
            col_first_stage.append(float(first_stage))


    if not col_imp_id:
        return pl.DataFrame(schema={
            "impression_id": pl.Utf8,
            "user_id": pl.Utf8,
            "article_id": pl.Utf8,
            "label": pl.Int32,
            "user_click_count": pl.Int32,
            "user_recency_score": pl.Float32,
            "user_mean_dwell_time": pl.Float32,
            "article_freshness_hours": pl.Float32,
            "article_popularity_24h": pl.Int32,
            "category_affinity_score": pl.Float32,
            "session_position": pl.Int32,
            "bm25_score": pl.Float32,
            "semantic_score": pl.Float32,
            "first_stage_score": pl.Float32,
        })

    return pl.DataFrame({
        "impression_id": col_imp_id,
        "user_id": col_user_id,
        "article_id": col_art_id,
        "label": pl.Series(col_label, dtype=pl.Int32),
        "user_click_count": pl.Series(col_click_count, dtype=pl.Int32),
        "user_recency_score": pl.Series(col_recency_score, dtype=pl.Float32),
        "user_mean_dwell_time": pl.Series(col_mean_dwell, dtype=pl.Float32),
        "article_freshness_hours": pl.Series(col_freshness, dtype=pl.Float32),
        "article_popularity_24h": pl.Series(col_popularity, dtype=pl.Int32),
        "category_affinity_score": pl.Series(col_cat_affinity, dtype=pl.Float32),
        "session_position": pl.Series(col_position, dtype=pl.Int32),
        "bm25_score": pl.Series(col_bm25, dtype=pl.Float32),
        "semantic_score": pl.Series(col_semantic, dtype=pl.Float32),
        "first_stage_score": pl.Series(col_first_stage, dtype=pl.Float32),
    })


def batch_feature_generator(
    impressions_df: Union[pl.DataFrame, pd.DataFrame],
    articles_df: Union[pl.DataFrame, pd.DataFrame],
    history_df: Optional[Union[pl.DataFrame, pd.DataFrame]] = None,
    embeddings: Optional[np.ndarray] = None,
    article_id_to_idx: Optional[Dict[str, int]] = None,
    batch_size: int = 10000,
):
    """
    Generator yielding tabular feature batches to avoid Out-Of-Memory on large test sets.
    """
    imp_df = pl.from_pandas(impressions_df) if isinstance(impressions_df, pd.DataFrame) else impressions_df
    total = len(imp_df)
    for start_idx in range(0, total, batch_size):
        chunk = imp_df.slice(start_idx, batch_size)
        yield build_feature_dataset(
            impressions_df=chunk,
            articles_df=articles_df,
            history_df=history_df,
            embeddings=embeddings,
            article_id_to_idx=article_id_to_idx,
            show_progress=False,
        )


def process_dataset_split(
    dataset: str,
    split: str,
    max_impressions: Optional[int] = None,
    output_dir: Optional[str] = None,
) -> Path:
    import json
    from pathlib import Path

    data_dir = Path(f"data/processed/{dataset}")
    out_dir = Path(output_dir) if output_dir else data_dir
    out_path = out_dir / f"features_{split}.parquet"

    imp_file = data_dir / f"impressions_{split}.parquet"
    if not imp_file.exists():
        # Fallback to validation if val is named validation
        if split == "val":
            fallback = data_dir / "impressions_validation.parquet"
            if fallback.exists():
                imp_file = fallback

    if not imp_file.exists():
        print(f"[{dataset.upper()}] Warning: {imp_file} does not exist, skipping.")
        return out_path

    print(f"\n{'='*70}")
    print(f"[{dataset.upper()}] Building features for split={split}...")
    print(f"{'='*70}")
    impressions = pl.read_parquet(imp_file)
    articles = pl.read_parquet(data_dir / "articles.parquet")

    if max_impressions is not None and len(impressions) > max_impressions:
        impressions = impressions.slice(0, max_impressions)

    hist_file = data_dir / f"history_{split}.parquet"
    if not hist_file.exists() and split == "val":
        fallback_hist = data_dir / "history_validation.parquet"
        if fallback_hist.exists():
            hist_file = fallback_hist

    if dataset == "mind_large" and split in ("val", "dev"):
        # For MIND-Large validation, load train history for the val users to ensure full prior reading history
        val_uids = impressions["user_id"].unique()
        hist_train_path = data_dir / "history_train.parquet"
        hist_parts = []
        if hist_train_path.exists():
            print(f"[{dataset.upper()}] Loading prior train history for {len(val_uids):,} validation users...")
            hist_parts.append(
                pl.scan_parquet(hist_train_path).filter(pl.col("user_id").is_in(val_uids.implode())).collect()
            )
        if hist_file and hist_file.exists():
            hist_parts.append(pl.read_parquet(hist_file).filter(pl.col("user_id").is_in(val_uids.implode())))
        history = pl.concat(hist_parts, how="diagonal") if hist_parts else None
    elif dataset == "mind_large" and split == "test":
        # For MIND-Large test, load prior history from train + dev for test users
        test_uids = impressions["user_id"].unique()
        hist_train_path = data_dir / "history_train.parquet"
        hist_dev_path = data_dir / "history_dev.parquet"
        if not hist_dev_path.exists():
            hist_dev_path = data_dir / "history_val.parquet"
        hist_parts = []
        if hist_train_path.exists():
            print(f"[{dataset.upper()}] Loading prior train history for {len(test_uids):,} test users...")
            hist_parts.append(
                pl.scan_parquet(hist_train_path).filter(pl.col("user_id").is_in(test_uids.implode())).collect()
            )
        if hist_dev_path.exists():
            print(f"[{dataset.upper()}] Loading prior dev history for {len(test_uids):,} test users...")
            hist_parts.append(
                pl.scan_parquet(hist_dev_path).filter(pl.col("user_id").is_in(test_uids.implode())).collect()
            )
        history = pl.concat(hist_parts, how="diagonal") if hist_parts else None
    else:
        history = pl.read_parquet(hist_file) if hist_file and hist_file.exists() else None

    emb_path = data_dir / "article_embeddings.npy"
    ids_path = data_dir / "article_ids.json"
    embeddings = np.load(emb_path) if emb_path.exists() else None
    article_id_to_idx = None
    if ids_path.exists():
        with open(ids_path, "r", encoding="utf-8") as f:
            ids = json.load(f)
        article_id_to_idx = {aid: idx for idx, aid in enumerate(ids)}

    total_imps = len(impressions) if max_impressions is None else min(len(impressions), max_impressions)
    print(f"Processing {total_imps:,} impressions (total catalog: {len(articles):,} articles)...")

    batch_size = 10000
    if total_imps > batch_size:
        import pyarrow.parquet as pq
        out_path.parent.mkdir(parents=True, exist_ok=True)
        writer = None
        total_rows_written = 0
        from embeddings import normalize_l2
        embeddings_norm = np.ascontiguousarray(normalize_l2(embeddings), dtype=np.float32) if embeddings is not None else None

        art_ids = articles["article_id"].to_list()
        if article_id_to_idx is None:
            article_id_to_idx = {aid: idx for idx, aid in enumerate(art_ids)}
        categories = articles["category"].fill_null("").to_list() if "category" in articles.columns else [""] * len(art_ids)
        art_cat_map = dict(zip(art_ids, categories))
        pub_times = articles["published_time"].to_list() if "published_time" in articles.columns else [None] * len(art_ids)
        art_pub_map = dict(zip(art_ids, pub_times))

        bm25_retriever = None
        art_token_ids_map = {}
        try:
            import bm25s
            titles = articles["title"].fill_null("").to_list() if "title" in articles.columns else [""] * len(art_ids)
            if "abstract" in articles.columns:
                abstracts = articles["abstract"].fill_null("").to_list()
            elif "subtitle" in articles.columns:
                abstracts = articles["subtitle"].fill_null("").to_list()
            else:
                abstracts = [""] * len(art_ids)
            full_texts = [f"{t} {a}".strip() for t, a in zip(titles, abstracts)]
            tokens = bm25s.tokenize(full_texts, show_progress=False)
            bm25_retriever = bm25s.BM25()
            bm25_retriever.index(tokens, show_progress=False)
            art_token_ids_map = {aid: set(tokens.ids[i]) for i, aid in enumerate(art_ids)}
        except Exception:
            bm25_retriever = None

        user_history_map = {}
        if history is not None:
            h_df = pl.from_pandas(history) if isinstance(history, pd.DataFrame) else history
            if not h_df.is_empty() and "click_time" in h_df.columns:
                has_dwell = "dwell_time" in h_df.columns
                agg_exprs = [
                    pl.col("clicked_article_id").alias("aids"),
                    pl.col("click_time").alias("times"),
                ]
                if has_dwell:
                    agg_exprs.append(pl.col("dwell_time").fill_null(0.0).alias("dwells"))
                user_agg = h_df.sort("click_time").group_by("user_id").agg(agg_exprs)

                uids = user_agg["user_id"].to_list()
                aids_list = user_agg["aids"].to_list()
                times_list = user_agg["times"].to_list()
                if has_dwell:
                    dwells_list = user_agg["dwells"].to_list()
                    for uid, aids, times, dwells in zip(uids, aids_list, times_list, dwells_list):
                        user_history_map[uid] = list(zip(aids, times, dwells))
                else:
                    for uid, aids, times in zip(uids, aids_list, times_list):
                        user_history_map[uid] = [(aid, t, 0.0) for aid, t in zip(aids, times)]

        from tqdm import tqdm
        num_chunks = (total_imps + batch_size - 1) // batch_size
        pbar = tqdm(total=total_imps, desc=f"[{dataset.upper()}] Extracting features ({split})")
        for chunk_idx in range(num_chunks):
            start = chunk_idx * batch_size
            end = min(start + batch_size, total_imps)
            chunk_imp = impressions.slice(start, end - start)

            chunk_df = build_feature_dataset(
                impressions_df=chunk_imp,
                articles_df=articles,
                history_df=history,
                embeddings=embeddings,
                article_id_to_idx=article_id_to_idx,
                show_progress=False,
                bm25_retriever=bm25_retriever,
                art_token_ids_map=art_token_ids_map,
                art_cat_map=art_cat_map,
                art_pub_map=art_pub_map,
                user_history_map=user_history_map,
            )

            arrow_tbl = chunk_df.to_arrow()
            if writer is None:
                writer = pq.ParquetWriter(out_path, arrow_tbl.schema)
            writer.write_table(arrow_tbl)
            total_rows_written += len(chunk_df)
            pbar.update(len(chunk_imp))

        pbar.close()
        if writer is not None:
            writer.close()
        print(f"Successfully saved {total_rows_written:,} feature rows to {out_path}")
        return out_path
    else:
        features_df = build_feature_dataset(
            impressions_df=impressions,
            articles_df=articles,
            history_df=history,
            embeddings=embeddings,
            article_id_to_idx=article_id_to_idx,
            max_impressions=max_impressions,
        )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        features_df.write_parquet(out_path)
        print(f"Successfully saved {len(features_df):,} feature rows to {out_path}")
        return out_path


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Precompute feature datasets for re-ranking.")
    parser.add_argument("--dataset", type=str, default="all", help="mind, ebnerd, or all")
    parser.add_argument("--split", type=str, default="train,val", help="Comma-separated splits or 'all'")
    parser.add_argument("--max_impressions", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    datasets = ["ebnerd", "mind"] if args.dataset in ("all", "both") else [d.strip() for d in args.dataset.split(",")]
    if args.split == "all":
        splits = ["train", "val", "test"]
    else:
        splits = [s.strip() for s in args.split.split(",")]

    for ds in datasets:
        for sp in splits:
            process_dataset_split(
                dataset=ds,
                split=sp,
                max_impressions=args.max_impressions,
                output_dir=args.output_dir,
            )


if __name__ == "__main__":
    main()


