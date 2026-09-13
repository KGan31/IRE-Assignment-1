# Q2.4: Two-Stage Retrieve-then-Rank Evaluation Results

**Assignment 2: Learning from Click-Logs on EB-NeRD and MIND**  
**Course:** CS4.406 Information Retrieval & Extraction  
**Task:** Implement two-stage retrieve-then-rank pipeline. Report AUC, MRR, nDCG@5, and nDCG@10 before and after re-ranking, together with latency profiles and serving scale analysis.

---

## 1. System Architecture: Two-Stage Retrieve-then-Rank

```
Impression Request (user_id, timestamp, inview candidates or full catalog)
  │
  ├──► [Temporal Boundary Enforcer] (Q1.4, Q9 Anti-Gaming)
  │       Strict condition: Drops any click with t_click >= t_imp
  │
  ├──► [Stage 1: Candidate Generation / Prior Scoring] (Top-K ~ 100-200)
  │       ├─ Dense Semantic FAISS Search (Cosine inner product on L2-normalized embeddings)
  │       ├─ BM25 Lexical Scoring (Exact inverted index evaluation)
  │       └─ Hybrid Fusion Prior (Reciprocal Rank Fusion / max(BM25, Semantic))
  │
  └──► [Stage 2: LightGBM LambdaMART Re-Ranker]
          ├─ 10 Point-in-Time Engineered Features:
          │    1. user_click_count
          │    2. user_recency_score (half-life decay: w = 0.5^(dt / 24h))
          │    3. user_mean_dwell_time (seconds)
          │    4. article_freshness_hours ((t_imp - t_pub) in hours)
          │    5. article_popularity_24h (prior 24h click volume)
          │    6. category_affinity_score (historical category fraction)
          │    7. session_position (presentation rank index)
          │    8. bm25_score (explicit lexical relevance)
          │    9. semantic_score (dense user-article cosine similarity)
          │   10. first_stage_score (Stage 1 retrieval prior)
          └─ Output: Final re-ranked candidate list sorted descending by score
```

---

## 2. MIND Dataset: Re-Ranking Quality

### A. Full Validation Split Offline Evaluation
Evaluated over **15,696 validation impressions** (603,740 candidate pairs) against a model trained on **125,572 training impressions** (4,654,725 rows).

| Metric | Stage 1 (Before Re-Ranking) | Stage 2 (LightGBM Re-Ranker) | Absolute Gain ($\Delta$) | Relative Gain (%) |
| :--- | :---: | :---: | :---: | :---: |
| **AUC** | `0.5350` | **`0.5748`** | **`+0.0398`** | **+7.44%** |
| **MRR** | `0.2612` | **`0.2929`** | **`+0.0317`** | **+12.14%** |
| **nDCG@5** | `0.2374` | **`0.2696`** | **`+0.0322`** | **+13.56%** |
| **nDCG@10** | `0.2997` | **`0.3322`** | **`+0.0325`** | **+10.84%** |

### B. End-to-End Pipeline Evaluation (`src/retrieve_then_rank.py`)
Tested through the live two-stage pipeline with point-in-time feature extraction and scoring:

| Metric | Stage 1 (Retrieval Prior) | Stage 2 (Re-Ranker) | Absolute Delta ($\Delta$) |
| :--- | :---: | :---: | :---: |
| **AUC** | `0.5128` | **`0.5414`** | **`+0.0286`** |
| **MRR** | `0.2350` | **`0.2680`** | **`+0.0330`** |
| **nDCG@5** | `0.1970` | **`0.2352`** | **`+0.0382`** |
| **nDCG@10** | `0.2684` | **`0.3013`** | **`+0.0329`** |

### C. MIND Feature Importances (Split Counts)
```
  session_position          :      271 splits
  semantic_score            :       80 splits
  user_click_count          :       64 splits
  category_affinity_score   :       43 splits
  user_recency_score        :       23 splits
  bm25_score                :       17 splits
  first_stage_score         :       12 splits
```

---

## 3. EB-NeRD Dataset: Re-Ranking Quality

### A. Full Validation Split Offline Evaluation
Evaluated over **25,356 validation impressions** (304,915 candidate pairs) against a model trained on **24,724 training impressions** (278,139 rows).

| Metric | Stage 1 (Before Re-Ranking) | Stage 2 (LightGBM Re-Ranker) | Absolute Gain ($\Delta$) | Relative Gain (%) |
| :--- | :---: | :---: | :---: | :---: |
| **AUC** | `0.4991` | **`0.7045`** | **`+0.2055`** | **+41.17%** |
| **MRR** | `0.3130` | **`0.4676`** | **`+0.1546`** | **+49.39%** |
| **nDCG@5** | `0.3431` | **`0.5291`** | **`+0.1860`** | **+54.21%** |
| **nDCG@10** | `0.4293` | **`0.5756`** | **`+0.1463`** | **+34.08%** |

### B. End-to-End Pipeline Evaluation (`src/retrieve_then_rank.py`)
Tested through the live two-stage pipeline with point-in-time feature extraction and scoring:

| Metric | Stage 1 (Retrieval Prior) | Stage 2 (Re-Ranker) | Absolute Delta ($\Delta$) |
| :--- | :---: | :---: | :---: |
| **AUC** | `0.4782` | **`0.6914`** | **`+0.2132`** |
| **MRR** | `0.3303` | **`0.4937`** | **`+0.1633`** |
| **nDCG@5** | `0.3743` | **`0.5594`** | **`+0.1852`** |
| **nDCG@10** | `0.4520` | **`0.6053`** | **`+0.1532`** |

### C. EB-NeRD Feature Importances (Split Counts)
```
  article_freshness_hours   :      360 splits
  category_affinity_score   :      268 splits
  semantic_score            :      224 splits
  user_click_count          :      171 splits
  bm25_score                :      171 splits
  session_position          :      162 splits
  user_recency_score        :      161 splits
  first_stage_score         :      133 splits
```

---

## 4. Serving Latency Breakdown & Scale Analysis (Q4)

Both pipelines meet the production serving SLA requirement ($\text{p99} < 100\text{ ms}$) with substantial margin.

| Dataset | Stage 1 Retrieval (Mean) | Stage 2 Re-Ranking (Mean) | Total Latency p50 | Total Latency p95 | Total Latency p99 | SLA Status ($< 100\text{ ms}$) | Headroom |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **EB-NeRD** | `5.68 ms` | `2.79 ms` | `7.83 ms` | `12.54 ms` | **`16.01 ms`** | **PASS** | **6.2x faster** |
| **MIND** | `2.19 ms` | `2.63 ms` | `3.04 ms` | `10.13 ms` | **`12.62 ms`** | **PASS** | **7.9x faster** |

### Latency Insights:
- **Stage 1 (Retrieval):** FAISS inner product dot product is bounded by $O(D \cdot K)$ and BM25 tokenized intersection takes $< 3\text{ ms}$.
- **Stage 2 (Re-Ranking):** Feature assembly plus tree traversal for $N \sim 50\text{--}100$ candidates takes $< 3\text{ ms}$, ensuring near-instantaneous reranking.

---

## 5. Key Findings & Insights

1. **Massive Quality Jump on EB-NeRD**:
   - EB-NeRD witnessed a dramatic improvement: **AUC surged from 0.4991 to 0.7045 (+41.2%)**, and **nDCG@5 jumped from 0.3431 to 0.5291 (+54.2%)**.
   - **Why?** Rapid news turnover in Ekstra Bladet makes publication recency (`article_freshness_hours`) and localized section preference (`category_affinity_score`) paramount. First-stage semantic search alone fails to penalize older articles, whereas the re-ranker heavily weights freshness (360 splits).
2. **Hybrid Lexical + Semantic Synergy**:
   - Both datasets show strong contributions from both `semantic_score` and `bm25_score`. The tree splits on lexical BM25 for exact keyword matches (e.g. sports teams, specific politicians) and leverages dense embeddings for conceptual topical match.
3. **Position Bias Correction**:
   - On MIND, `session_position` is the top split feature (271 splits), confirming strong user bias toward top-listed items. Learning this enables the model to disentangle true relevance from presentation rank.
4. **Leak-Free Compliance**:
   - All behavioral features (recency decay, click counts, category affinities) strictly enforce $t_{\text{click}} < t_{\text{imp}}$ with zero future or target leakage, fully validated by [`src/tests/test_behavior_window.py`](file:///c:/Users/kavan/OneDrive/Desktop/IIITH/sem3/IRE/Assignment1/IRE-Assignment-1/src/tests/test_behavior_window.py).

---

## 6. Artifact & Codebase Reference

- **Two-Stage Pipeline Script**: [`src/retrieve_then_rank.py`](file:///c:/Users/kavan/OneDrive/Desktop/IIITH/sem3/IRE/Assignment1/IRE-Assignment-1/src/retrieve_then_rank.py)
- **Feature Engineering**: [`src/features.py`](file:///c:/Users/kavan/OneDrive/Desktop/IIITH/sem3/IRE/Assignment1/IRE-Assignment-1/src/features.py)
- **Model Training**: [`src/reranker.py`](file:///c:/Users/kavan/OneDrive/Desktop/IIITH/sem3/IRE/Assignment1/IRE-Assignment-1/src/reranker.py)
- **Unit Test Suite**:
  - [`src/tests/test_behavior_window.py`](file:///c:/Users/kavan/OneDrive/Desktop/IIITH/sem3/IRE/Assignment1/IRE-Assignment-1/src/tests/test_behavior_window.py) (Anti-gaming, point-in-time boundary)
  - [`src/tests/test_retrieve_then_rank.py`](file:///c:/Users/kavan/OneDrive/Desktop/IIITH/sem3/IRE/Assignment1/IRE-Assignment-1/src/tests/test_retrieve_then_rank.py) (RRF fusion, pipeline end-to-end, cold-start handling)
- **Saved Model Checkpoints**:
  - `models/mind_lgbm_ranker.pkl`
  - `models/ebnerd_lgbm_ranker.pkl`
- **Output JSON Results**:
  - `models/mind_two_stage_pipeline_results.json`
  - `models/ebnerd_two_stage_pipeline_results.json`
  - `models/mind_reranker_metrics.json`
  - `models/ebnerd_reranker_metrics.json`
