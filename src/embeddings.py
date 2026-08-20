"""
Dense Semantic Candidate Retrieval Engine using Sentence-Transformers and FAISS.

Supports:
1. Loading or computing dense embeddings for articles (MIND & EB-NeRD):
   - MIND: sentence-transformers (e.g. all-MiniLM-L6-v2 or paraphrase-multilingual-MiniLM-L12-v2)
   - EB-NeRD: Pre-computed Ekstra Bladet BERT/Word2Vec embeddings or multilingual sentence-transformers
2. Building fast vector indices with FAISS (Inner Product / Cosine Similarity) with pure numpy fallback
3. Computing leak-free user representations via mean-pooled article embeddings
4. Batch nearest-neighbor candidate retrieval
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

# Optional FAISS import with fallback
try:
    import faiss
    HAS_FAISS = True
except ImportError:
    faiss = None
    HAS_FAISS = False

# Optional SentenceTransformers import
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    SentenceTransformer = None
    HAS_SENTENCE_TRANSFORMERS = False


def normalize_l2(vectors: np.ndarray) -> np.ndarray:
    """L2 normalize a 2D numpy array of vectors along axis 1."""
    if vectors.ndim == 1:
        norm = np.linalg.norm(vectors)
        return vectors / (norm + 1e-12)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-12, norms)
    return vectors / norms


def load_ebnerd_pretrained_embeddings(
    raw_dir: Path,
    articles_df: pd.DataFrame,
) -> Optional[Tuple[np.ndarray, List[str]]]:
    """Attempt to load Ekstra Bladet pre-trained embeddings (BERT or Word2Vec)."""
    possible_paths = [
        raw_dir / "ebnerd" / "bert",
        raw_dir / "ebnerd" / "word2vec",
        raw_dir / "ebnerd" / "demo" / "bert",
        raw_dir / "ebnerd" / "demo" / "word2vec",
    ]

    target_files = []
    for p in possible_paths:
        if p.exists():
            target_files.extend(list(p.glob("*.parquet")) + list(p.glob("*.npy")))

    if not target_files:
        return None

    for tf in target_files:
        try:
            if tf.suffix == ".parquet":
                emb_df = pd.read_parquet(tf)
                id_col = None
                vec_col = None
                for col in emb_df.columns:
                    if "id" in col.lower():
                        id_col = col
                    if "vector" in col.lower() or "emb" in col.lower() or "bert" in col.lower():
                        vec_col = col

                if id_col and vec_col:
                    print(f"[EB-NeRD] Loading pre-trained embeddings from {tf} (cols: {id_col}, {vec_col})...")
                    raw_ids = emb_df[id_col].astype(str).tolist()
                    prefixed_ids = [
                        f"ebnerd_{rid}" if not str(rid).startswith("ebnerd_") else str(rid)
                        for rid in raw_ids
                    ]
                    emb_matrix = np.array(emb_df[vec_col].tolist(), dtype=np.float32)
                    id_to_vec = dict(zip(prefixed_ids, emb_matrix))

                    # Align with articles_df
                    article_ids = articles_df["article_id"].tolist()
                    dim = emb_matrix.shape[1]
                    aligned_matrix = np.zeros((len(article_ids), dim), dtype=np.float32)
                    for i, aid in enumerate(article_ids):
                        if aid in id_to_vec:
                            aligned_matrix[i] = id_to_vec[aid]

                    return aligned_matrix, article_ids
        except Exception as e:
            print(f"Warning: Failed to load pre-trained embeddings from {tf}: {e}")

    return None


def compute_article_embeddings_hf(
    articles_df: pd.DataFrame,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 256,
    device: Optional[str] = None,
) -> np.ndarray:
    """Compute dense text embeddings using SentenceTransformer over title + abstract."""
    if not HAS_SENTENCE_TRANSFORMERS:
        raise ImportError(
            "sentence-transformers is required to compute text embeddings. "
            "Install with: pip install sentence-transformers"
        )

    titles = articles_df["title"].fillna("").astype(str).tolist()
    abstracts = (
        articles_df["abstract"].fillna("").astype(str).tolist()
        if "abstract" in articles_df.columns
        else [""] * len(titles)
    )

    texts = [f"{t} {a}".strip() for t, a in zip(titles, abstracts)]
    # Fallback non-empty text for blanks
    texts = [txt if txt else "news article" for txt in texts]

    print(f"Encoding {len(texts)} articles using '{model_name}' on device '{device or 'auto'}'...")
    model = SentenceTransformer(model_name, device=device)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    return np.ascontiguousarray(embeddings, dtype=np.float32)


def get_or_compute_article_embeddings(
    dataset_name: str,
    articles_df: pd.DataFrame,
    processed_dir: Path,
    raw_dir: Optional[Path] = None,
    model_name: Optional[str] = None,
    batch_size: int = 256,
    device: Optional[str] = None,
    force_recompute: bool = False,
) -> Tuple[np.ndarray, Dict[str, int], List[str]]:
    """
    Get article embeddings from cache or compute them and cache.
    
    Returns:
        embeddings: (N, D) float32 numpy array, L2 normalized
        article_id_to_idx: map from article_id to row index
        article_ids: list of article_ids in row order
    """
    ds_processed = processed_dir / dataset_name
    ds_processed.mkdir(parents=True, exist_ok=True)
    cache_emb_path = ds_processed / "article_embeddings.npy"
    cache_ids_path = ds_processed / "article_ids.json"

    article_ids = articles_df["article_id"].tolist()
    article_id_to_idx = {aid: i for i, aid in enumerate(article_ids)}

    # Check cache
    if not force_recompute and cache_emb_path.exists() and cache_ids_path.exists():
        try:
            with open(cache_ids_path, "r", encoding="utf-8") as f:
                cached_ids = json.load(f)
            if cached_ids == article_ids:
                print(f"[{dataset_name}] Loading cached embeddings from {cache_emb_path}...")
                embeddings = np.load(cache_emb_path)
                return embeddings, article_id_to_idx, article_ids
        except Exception as e:
            print(f"[{dataset_name}] Cache validation failed ({e}), recomputing...")

    # For EB-NeRD, try pre-trained first if raw_dir provided
    embeddings = None
    if dataset_name.lower() == "ebnerd" and raw_dir is not None:
        pretrained_res = load_ebnerd_pretrained_embeddings(raw_dir, articles_df)
        if pretrained_res is not None:
            embeddings, _ = pretrained_res

    # Otherwise compute via SentenceTransformers
    if embeddings is None:
        if model_name is None:
            if dataset_name.lower() == "mind":
                model_name = "sentence-transformers/all-MiniLM-L6-v2"
            else:
                model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

        embeddings = compute_article_embeddings_hf(
            articles_df=articles_df,
            model_name=model_name,
            batch_size=batch_size,
            device=device,
        )

    # Normalize L2
    embeddings = normalize_l2(embeddings)

    # Save to cache
    print(f"[{dataset_name}] Caching embeddings of shape {embeddings.shape} to {cache_emb_path}...")
    np.save(cache_emb_path, embeddings)
    with open(cache_ids_path, "w", encoding="utf-8") as f:
        json.dump(article_ids, f)

    return embeddings, article_id_to_idx, article_ids


class EmbeddingIndex:
    """FAISS-powered dense vector index with pure numpy fallback."""

    def __init__(self, use_approximate: bool = False):
        self.use_approximate = use_approximate
        self.article_ids: List[str] = []
        self.article_id_to_idx: Dict[str, int] = {}
        self.embeddings: Optional[np.ndarray] = None
        self.index: Optional[object] = None
        self.is_built: bool = False

    def build_index(self, embeddings: np.ndarray, article_ids: List[str]) -> None:
        """Index dense L2-normalized article embeddings."""
        if len(embeddings) != len(article_ids):
            raise ValueError(f"Embeddings count ({len(embeddings)}) != article IDs count ({len(article_ids)})")

        self.embeddings = np.ascontiguousarray(normalize_l2(embeddings), dtype=np.float32)
        self.article_ids = list(article_ids)
        self.article_id_to_idx = {aid: i for i, aid in enumerate(self.article_ids)}
        num_docs, dim = self.embeddings.shape

        if HAS_FAISS:
            if self.use_approximate and num_docs > 10000:
                # HNSW flat index with M=32
                index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
                index.hnsw.efSearch = 64
                index.add(self.embeddings)
                self.index = index
            else:
                index = faiss.IndexFlatIP(dim)
                index.add(self.embeddings)
                self.index = index
        else:
            self.index = None

        self.is_built = True

    def batch_search(
        self,
        query_vectors: np.ndarray,
        top_k: int = 200,
    ) -> List[List[Tuple[str, float]]]:
        """
        Batch retrieve top-K (article_id, similarity_score) for a matrix of query vectors.
        query_vectors shape: (Q, D), float32
        """
        if not self.is_built or len(self.article_ids) == 0 or len(query_vectors) == 0:
            return [[] for _ in range(len(query_vectors))]

        q_vecs = np.ascontiguousarray(normalize_l2(query_vectors), dtype=np.float32)
        num_docs = len(self.article_ids)
        k = min(top_k, num_docs)

        if HAS_FAISS and self.index is not None:
            scores, indices = self.index.search(q_vecs, k)
            results: List[List[Tuple[str, float]]] = []
            for row_indices, row_scores in zip(indices, scores):
                row_res = []
                for idx, score in zip(row_indices, row_scores):
                    if idx >= 0:
                        row_res.append((self.article_ids[idx], float(score)))
                results.append(row_res)
            return results

        # Numpy fallback (matrix multiplication for Inner Product / Cosine Similarity)
        # q_vecs: (Q, D), self.embeddings.T: (D, N) -> sim_matrix: (Q, N)
        sim_matrix = np.dot(q_vecs, self.embeddings.T)
        results = []
        for i in range(len(query_vectors)):
            row_sims = sim_matrix[i]
            top_indices = np.argpartition(-row_sims, k - 1)[:k]
            top_indices = top_indices[np.argsort(-row_sims[top_indices])]
            row_res = [(self.article_ids[idx], float(row_sims[idx])) for idx in top_indices]
            results.append(row_res)

        return results


def compute_user_representation(
    clicked_article_ids: List[str],
    article_id_to_idx: Dict[str, int],
    embeddings: np.ndarray,
    weights: Optional[List[float]] = None,
) -> Optional[np.ndarray]:
    """
    Compute mean-pooled (or weighted) representation vector for a user from their click history.
    """
    valid_vectors = []
    valid_weights = []

    for i, aid in enumerate(clicked_article_ids):
        if aid in article_id_to_idx:
            idx = article_id_to_idx[aid]
            valid_vectors.append(embeddings[idx])
            w = weights[i] if (weights is not None and i < len(weights)) else 1.0
            valid_weights.append(w)

    if not valid_vectors:
        return None

    vec_array = np.array(valid_vectors, dtype=np.float32)
    weight_array = np.array(valid_weights, dtype=np.float32).reshape(-1, 1)

    # Weighted mean
    user_vec = np.sum(vec_array * weight_array, axis=0) / (np.sum(weight_array) + 1e-12)
    return normalize_l2(user_vec)
