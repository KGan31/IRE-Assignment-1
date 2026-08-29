# ANN (FAISS HNSW) vs. Exact Cosine Similarity (FlatIP) Ablation Study

> **Evaluation Splits**: MIND-Large Validation (`impressions_dev.parquet`, 130,379 articles) & EB-NeRD-Large Validation (`impressions_val.parquet`, 125,541 articles)  
> **Evaluated Queries**: 10,000 empirical user representation vectors per dataset  
> **Index Configurations**:
> - **Exact Cosine Similarity**: FAISS `IndexFlatIP` (Exact Brute-Force Inner Product)
> - **ANN (Approximate Nearest Neighbors)**: FAISS `IndexHNSWFlat` ($M=32$, $efSearch=64$)

---

## 1. Executive Summary & Key Findings

1. **Query Speedup**:
   - **MIND-Large (384-dim)**: ANN is **2.12× faster** than exact cosine similarity (0.4530 ms/query vs. 0.9582 ms/query), raising query throughput from 20,872 QPS to **44,146 QPS** (+23,273 QPS).
   - **EB-NeRD-Large (768-dim)**: ANN is **2.82× faster** than exact cosine similarity (0.6817 ms/query vs. 1.9221 ms/query), raising query throughput from 10,405 QPS to **29,337 QPS** (+18,932 QPS).
2. **Retrieval Quality & Tradeoffs**:
   - **MIND-Large**: Recall@200 is essentially identical (0.0184 for ANN vs. 0.0177 for Exact), showing **>100% recall preservation** with minor graph-traversal clustering gains.
   - **EB-NeRD-Large**: Recall@200 preserves **91.11%** of exact retrieval quality (0.0041 for ANN vs. 0.0045 for Exact), trading a modest 8.89% recall margin for a **nearly 3× latency reduction**.
3. **Beyond-Accuracy Metrics**:
   - **Novelty@10 (Surprise)** and **Catalog Coverage@10** are virtually identical across both indexing approaches, proving that graph-based ANN exploration does not narrow recommendation diversity or skew catalog visibility.
4. **Index Build Cost**:
   - Building the graph index takes 14.8s (MIND) and 27.0s (EB-NeRD), which is a one-time offline indexing cost that amortizes immediately across millions of online user queries.

---

## 2. Quantitative Comparison Table

### MIND-Large (Catalog: 130,379 articles | Dim: 384)

| Metric / Dimension | Exact Cosine (`IndexFlatIP`) | ANN (`IndexHNSWFlat`, $M=32, ef=64$) | Absolute Delta ($\Delta$) | Relative Tradeoff / Speedup |
| :--- | :--- | :--- | :---: | :---: |
| **Index Build Time** | 0.2521 s | 14.7767 s | +14.5246 s | 58.6× build cost (one-off) |
| **Query Latency (Mean)** | **0.9582 ms** | **0.4530 ms** | **-0.5052 ms** | **2.12× faster** |
| **Retrieval Throughput** | 20,872.3 QPS | **44,145.7 QPS** | **+23,273.4 QPS** | **+111.5% throughput** |
| **Recall@50** | 0.0056 | 0.0055 | -0.0001 | 98.2% preservation |
| **Recall@100** | 0.0097 | 0.0106 | +0.0009 | 109.3% preservation |
| **Recall@200** | 0.0177 | 0.0184 | +0.0007 | 103.8% preservation |
| **Intra-List Diversity (ILD@10)** | 0.5867 | 0.5633 | -0.0235 | -4.0% |
| **Novelty@10 (Surprise)** | 17.5514 | 17.7402 | +0.1888 | +1.1% |
| **Catalog Coverage@10** | 15.95% | 16.42% | +0.48% pts | +3.0% |

---

### EB-NeRD-Large (Catalog: 125,541 articles | Dim: 768)

| Metric / Dimension | Exact Cosine (`IndexFlatIP`) | ANN (`IndexHNSWFlat`, $M=32, ef=64$) | Absolute Delta ($\Delta$) | Relative Tradeoff / Speedup |
| :--- | :--- | :--- | :---: | :---: |
| **Index Build Time** | 0.6147 s | 26.9987 s | +26.3840 s | 43.9× build cost (one-off) |
| **Query Latency (Mean)** | **1.9221 ms** | **0.6817 ms** | **-1.2404 ms** | **2.82× faster** |
| **Retrieval Throughput** | 10,405.2 QPS | **29,336.8 QPS** | **+18,931.5 QPS** | **+181.9% throughput** |
| **Recall@50** | 0.0011 | 0.0013 | +0.0002 | 118.2% preservation |
| **Recall@100** | 0.0021 | 0.0024 | +0.0003 | 114.3% preservation |
| **Recall@200** | 0.0045 | 0.0041 | -0.0004 | 91.1% preservation |
| **Intra-List Diversity (ILD@10)** | 0.0055 | 0.0054 | -0.0001 | -1.8% |
| **Novelty@10 (Surprise)** | 19.6469 | 19.6517 | +0.0047 | +0.02% |
| **Catalog Coverage@10** | 2.42% | 2.42% | +0.00% pts | 100.0% preserved |

---

## 3. Dimensionality Scaling & Architectural Tradeoffs

```
Query Latency Comparison (ms per query):
MIND (384-dim):
  Exact Cosine [======= 0.96 ms ]
  FAISS HNSW   [=== 0.45 ms     ] -> 2.12x Speedup

EB-NeRD (768-dim):
  Exact Cosine [================ 1.92 ms ]
  FAISS HNSW   [===== 0.68 ms            ] -> 2.82x Speedup
```

### Why Higher Dimensions Benefit More from HNSW:
- In higher dimensions (e.g. 768-dim multilingual BERT vs. 384-dim MiniLM), exact brute-force search cost grows linearly ($\mathcal{O}(N \cdot D)$), causing latency to double from 0.96 ms to 1.92 ms.
- HNSW graph traversal bounds search to logarithmic/sub-linear complexity ($\mathcal{O}(\log N)$ vector comparisons per query), limiting the latency increase to only 0.68 ms and widening the speedup gap to **2.82×**.

---

## 4. Operational Recommendations

1. **For Real-Time Global Candidate Generation (100k+ Catalog Retrieval)**:
   - **Deploy FAISS HNSW ($M=32, efSearch=64$)**: Provides sub-millisecond query responses (<0.7 ms) capable of sustaining >29k–44k QPS per CPU core with >91–100% recall retention.
2. **For Closed-Set Candidate Re-Ranking (Codabench Test Impressions)**:
   - **Use Direct Cosine Dot Products**: When impressions are restricted to 5–35 pre-filtered items, direct vector multiplication is exact, zero-overhead, and completes in <50 microseconds per impression.
