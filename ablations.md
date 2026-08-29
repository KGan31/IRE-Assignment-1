# Ablation Study Results

> Generated: 2026-08-26 19:07:33
> Datasets: MIND-small / val, EB-NeRD-demo / val
> Bootstrap samples: 500
> Best history length (from Phase 2): **20**

---

## Phase 1 — BM25 Query Input Fields

Ablates what text from the user's clicked history articles is used to
build the BM25 query. For EB-NeRD, an additional variant also indexes
body text in the document store (`include_body=True`).

### Ranking Metrics

| Config | Dataset | AUC | MRR | nDCG@5 | nDCG@10 |
|--------|---------|-----|-----|--------|---------|
| `title` | MIND | 0.5039 | 0.2421 | 0.2146 | 0.2783 |
| `title+abstract` | MIND | 0.5032 | 0.2406 | 0.2133 | 0.2772 |
| `title` | EBNERD | 0.5014 | 0.3135 | 0.3455 | 0.4295 |
| `title+abstract` | EBNERD | 0.5000 | 0.3122 | 0.3439 | 0.4282 |
| `title+abstract+body` | EBNERD | 0.5007 | 0.3127 | 0.3440 | 0.4289 |

### Retrieval Metrics

| Config | Dataset | Recall@50 | Recall@100 | Recall@200 |
|--------|---------|-----------|------------|------------|
| `title` | MIND | 0.0026 | 0.0068 | 0.0109 |
| `title+abstract` | MIND | 0.0021 | 0.0051 | 0.0088 |
| `title` | EBNERD | 0.0076 | 0.0148 | 0.0264 |
| `title+abstract` | EBNERD | 0.0073 | 0.0139 | 0.0248 |
| `title+abstract+body` | EBNERD | 0.0107 | 0.0186 | 0.0305 |

---

## Phase 2 — History Length  (BM25 + Semantic, title+abstract)

Ablates how many of the user's most recent clicked articles form the query,
comparing both BM25 and Semantic (FAISS flat) at each length.
Best length chosen by highest mean nDCG@10 pooled across both models and both datasets: **20**.

### Ranking Metrics

| Config | Dataset | AUC | MRR | nDCG@5 | nDCG@10 |
|--------|---------|-----|-----|--------|---------|
| `bm25-history=10` | MIND | 0.5031 | 0.2401 | 0.2128 | 0.2766 |
| `semantic-history=10` | MIND | 0.5906 | 0.3011 | 0.2812 | 0.3436 |
| `bm25-history=20` | MIND | 0.5032 | 0.2406 | 0.2133 | 0.2772 |
| `semantic-history=20` | MIND | 0.5950 | 0.3036 | 0.2846 | 0.3460 |
| `bm25-history=30` | MIND | 0.5030 | 0.2398 | 0.2128 | 0.2766 |
| `semantic-history=30` | MIND | 0.5964 | 0.3043 | 0.2857 | 0.3467 |
| `bm25-history=10` | EBNERD | 0.5007 | 0.3123 | 0.3440 | 0.4284 |
| `semantic-history=10` | EBNERD | 0.4958 | 0.3209 | 0.3493 | 0.4342 |
| `bm25-history=20` | EBNERD | 0.5000 | 0.3122 | 0.3439 | 0.4282 |
| `semantic-history=20` | EBNERD | 0.4962 | 0.3214 | 0.3498 | 0.4353 |
| `bm25-history=30` | EBNERD | 0.4994 | 0.3106 | 0.3424 | 0.4268 |
| `semantic-history=30` | EBNERD | 0.4966 | 0.3214 | 0.3499 | 0.4351 |

### Retrieval Metrics

| Config | Dataset | Recall@50 | Recall@100 | Recall@200 |
|--------|---------|-----------|------------|------------|
| `bm25-history=10` | MIND | 0.0023 | 0.0049 | 0.0087 |
| `semantic-history=10` | MIND | 0.0032 | 0.0063 | 0.0103 |
| `bm25-history=20` | MIND | 0.0021 | 0.0051 | 0.0088 |
| `semantic-history=20` | MIND | 0.0034 | 0.0063 | 0.0114 |
| `bm25-history=30` | MIND | 0.0022 | 0.0047 | 0.0082 |
| `semantic-history=30` | MIND | 0.0036 | 0.0068 | 0.0113 |
| `bm25-history=10` | EBNERD | 0.0084 | 0.0143 | 0.0254 |
| `semantic-history=10` | EBNERD | 0.0082 | 0.0155 | 0.0303 |
| `bm25-history=20` | EBNERD | 0.0073 | 0.0139 | 0.0248 |
| `semantic-history=20` | EBNERD | 0.0079 | 0.0164 | 0.0311 |
| `bm25-history=30` | EBNERD | 0.0057 | 0.0121 | 0.0238 |
| `semantic-history=30` | EBNERD | 0.0077 | 0.0160 | 0.0305 |

---

## Phase 3 — Cold-Start Threshold  (BM25 + Semantic, history=20)

Redefines cold-start users by percentile of the per-impression
history-length distribution rather than a fixed click count.
Run for both BM25 and Semantic to see whether the threshold sensitivity
differs between retrieval paradigms.
The overall metrics are computed over **all** users; the cold/warm
slice breakdown appears in the printed evaluation summary.

### Ranking Metrics

| Config | Dataset | AUC | MRR | nDCG@5 | nDCG@10 |
|--------|---------|-----|-----|--------|---------|
| `bm25-cold=default` | MIND | 0.5032 | 0.2406 | 0.2133 | 0.2772 |
| `bm25-cold=p1` | MIND | 0.5032 | 0.2406 | 0.2133 | 0.2772 |
| `bm25-cold=p2` | MIND | 0.5032 | 0.2406 | 0.2133 | 0.2772 |
| `bm25-cold=p5` | MIND | 0.5032 | 0.2406 | 0.2133 | 0.2772 |
| `bm25-cold=p10` | MIND | 0.5032 | 0.2406 | 0.2133 | 0.2772 |
| `semantic-cold=default` | MIND | 0.5950 | 0.3036 | 0.2846 | 0.3460 |
| `semantic-cold=p1` | MIND | 0.5950 | 0.3036 | 0.2846 | 0.3460 |
| `semantic-cold=p2` | MIND | 0.5950 | 0.3036 | 0.2846 | 0.3460 |
| `semantic-cold=p5` | MIND | 0.5950 | 0.3036 | 0.2846 | 0.3460 |
| `semantic-cold=p10` | MIND | 0.5950 | 0.3036 | 0.2846 | 0.3460 |
| `bm25-cold=default` | EBNERD | 0.5000 | 0.3122 | 0.3439 | 0.4282 |
| `bm25-cold=p1` | EBNERD | 0.5000 | 0.3122 | 0.3439 | 0.4282 |
| `bm25-cold=p2` | EBNERD | 0.5000 | 0.3122 | 0.3439 | 0.4282 |
| `bm25-cold=p5` | EBNERD | 0.5000 | 0.3122 | 0.3439 | 0.4282 |
| `bm25-cold=p10` | EBNERD | 0.5000 | 0.3122 | 0.3439 | 0.4282 |
| `semantic-cold=default` | EBNERD | 0.4962 | 0.3214 | 0.3498 | 0.4353 |
| `semantic-cold=p1` | EBNERD | 0.4962 | 0.3214 | 0.3498 | 0.4353 |
| `semantic-cold=p2` | EBNERD | 0.4962 | 0.3214 | 0.3498 | 0.4353 |
| `semantic-cold=p5` | EBNERD | 0.4962 | 0.3214 | 0.3498 | 0.4353 |
| `semantic-cold=p10` | EBNERD | 0.4962 | 0.3214 | 0.3498 | 0.4353 |

### Retrieval Metrics

| Config | Dataset | Recall@50 | Recall@100 | Recall@200 |
|--------|---------|-----------|------------|------------|
| `bm25-cold=default` | MIND | 0.0021 | 0.0051 | 0.0088 |
| `bm25-cold=p1` | MIND | 0.0021 | 0.0051 | 0.0088 |
| `bm25-cold=p2` | MIND | 0.0021 | 0.0051 | 0.0088 |
| `bm25-cold=p5` | MIND | 0.0021 | 0.0051 | 0.0088 |
| `bm25-cold=p10` | MIND | 0.0021 | 0.0051 | 0.0088 |
| `semantic-cold=default` | MIND | 0.0034 | 0.0063 | 0.0114 |
| `semantic-cold=p1` | MIND | 0.0034 | 0.0063 | 0.0114 |
| `semantic-cold=p2` | MIND | 0.0034 | 0.0063 | 0.0114 |
| `semantic-cold=p5` | MIND | 0.0034 | 0.0063 | 0.0114 |
| `semantic-cold=p10` | MIND | 0.0034 | 0.0063 | 0.0114 |
| `bm25-cold=default` | EBNERD | 0.0073 | 0.0139 | 0.0248 |
| `bm25-cold=p1` | EBNERD | 0.0073 | 0.0139 | 0.0248 |
| `bm25-cold=p2` | EBNERD | 0.0073 | 0.0139 | 0.0248 |
| `bm25-cold=p5` | EBNERD | 0.0073 | 0.0139 | 0.0248 |
| `bm25-cold=p10` | EBNERD | 0.0073 | 0.0139 | 0.0248 |
| `semantic-cold=default` | EBNERD | 0.0079 | 0.0164 | 0.0311 |
| `semantic-cold=p1` | EBNERD | 0.0079 | 0.0164 | 0.0311 |
| `semantic-cold=p2` | EBNERD | 0.0079 | 0.0164 | 0.0311 |
| `semantic-cold=p5` | EBNERD | 0.0079 | 0.0164 | 0.0311 |
| `semantic-cold=p10` | EBNERD | 0.0079 | 0.0164 | 0.0311 |

---

## Phase 4 — Semantic Index Type  (history=20)

Compares FAISS exact brute-force (`flat`) against FAISS HNSW
approximate nearest-neighbour (`ann`) on the dense retrieval path.
Both use L2-normalised embeddings with cosine similarity.

### Ranking Metrics

| Config | Dataset | AUC | MRR | nDCG@5 | nDCG@10 |
|--------|---------|-----|-----|--------|---------|
| `semantic-flat` | MIND | 0.5950 | 0.3036 | 0.2846 | 0.3460 |
| `semantic-ann` | MIND | 0.5950 | 0.3036 | 0.2846 | 0.3460 |
| `semantic-flat` | EBNERD | 0.4962 | 0.3214 | 0.3498 | 0.4353 |
| `semantic-ann` | EBNERD | 0.4962 | 0.3214 | 0.3498 | 0.4353 |

### Retrieval Metrics

| Config | Dataset | Recall@50 | Recall@100 | Recall@200 |
|--------|---------|-----------|------------|------------|
| `semantic-flat` | MIND | 0.0034 | 0.0063 | 0.0114 |
| `semantic-ann` | MIND | 0.0036 | 0.0067 | 0.0122 |
| `semantic-flat` | EBNERD | 0.0079 | 0.0164 | 0.0311 |
| `semantic-ann` | EBNERD | 0.0079 | 0.0164 | 0.0311 |

---

## Notes

- All reported metrics are **mean** values from non-parametric bootstrap resampling.
- Semantic phase uses cached embeddings; pass `force_recompute=True` if you
  switch models.
- EB-NeRD `title+abstract+body` ablation indexes subtitle (EB-NeRD abstract
  equivalent) + body text; MIND has no body column so the flag is a no-op there.
