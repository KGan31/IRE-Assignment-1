# Assignment 2 TODO & Implementation Backlog

## 1. EB-NeRD Rich Engagement Signals (Scroll Depth & Dwell Time)
- [x] **Data Parsing (`src/parse_ebnerd.py`)**:
  - In `parse_history()`, load `scroll_percentage_fixed` and `read_time_fixed` from `history.parquet` alongside `article_id_fixed` and `impression_time_fixed`.
  - Explode them simultaneously into the unified schema as `scroll_percentage` and `dwell_time`.
  - Ensure compatibility with MIND (which lacks scroll tracking) by keeping these fields nullable (`None`).
- [ ] **Feature Engineering (`src/features.py`)**:
  - `user_mean_scroll_percentage`: Average historical scroll depth across prior clicked articles. Distinguishes shallow skimmers from deep readers.
  - `user_deep_read_ratio`: Proportion of clicks where `scroll_percentage >= 75%` or `dwell_time >= 30s` (filters out clickbait bounces).
  - `dwell_weighted_category_affinity`: Weight category preferences by dwell time and exponential recency decay ($\text{dwell\_time} \times 0.5^{\Delta t / 24\text{h}}$) rather than raw click counts.
  - Session context from `behaviors.parquet`: `device_type` (Desktop/Mobile/Tablet), `is_subscriber` (Subscriber vs. Anonymous).
- [ ] **Anti-Gaming & Serving-Time Safety (Q9)**:
  - **CRITICAL**: Use ONLY past historical scroll/read times ($t_{\text{click}} < t_{\text{imp}}$).
  - DO NOT use the current impression's `read_time` or `scroll_percentage` from `behaviors.parquet` as features, since they represent post-click engagement unavailable at serving time.
- [ ] **Ablation Study Candidate (Q3)**:
  - Benchmark baseline (click count + BM25 + semantic similarity) vs. improved (adding scroll/dwell signals).
  - Measure performance delta with a paired bootstrap 95% confidence interval excluding zero.

---

## 2. Re-Ranker Pipeline (Q2)
- [ ] **Dataset Preparation**:
  - Build vectorized impression dataset generator (`src/dataset.py` or `src/features.py`) creating tabular $(X, y)$ training arrays from impressions and candidate articles.
  - Positive instances: clicked articles ($y = 1$).
  - Negative instances: non-clicked candidates in the impression or top-100 retrieved non-clicked items ($y = 0$).
- [ ] **Model Implementation (`src/reranker.py`)**:
  - LightGBM `LGBMRanker` (with group query IDs) or `LGBMClassifier` trained on engineered features.
  - Save trained checkpoint to `models/`.
- [ ] **Two-Stage Evaluation**:
  - Retrieve top-100 candidates with Stage 1 (BM25 / FAISS Semantic).
  - Re-score candidates with Stage 2 Re-Ranker.
  - Report AUC, MRR, nDCG@5, nDCG@10 before and after re-ranking using `OfflineEvaluationHarness`.

---

## 3. Baseline Reproduction, Principled Improvement & Ablation (Q3)
- [ ] **Reproduce Baseline**:
  - Official starter baseline on both MIND and EB-NeRD.
- [ ] **Principled Improvement**:
  - Implement hybrid lexical + semantic signals, category affinity matching, and/or engagement weighting.
- [ ] **Ablation Study**:
  - Systematically ablate each feature group (e.g. without BM25, without semantic embeddings, without recency decay, without engagement).
- [ ] **Statistical Significance**:
  - Compute paired bootstrap 95% confidence interval ($\Delta = \text{Metric}_{\text{improved}} - \text{Metric}_{\text{baseline}}$) and verify CI excludes zero.

---

## 4. Serving & Scale Analysis (Q4)
- [ ] **Index & Feature Store Memory Footprint**:
  - Measure RAM footprint of FAISS ANN index and feature lookup tables.
- [ ] **Retrieval Latency Benchmark**:
  - Measure single-request latency distribution (p50, p95, p99) for Stage 1 retrieval + Stage 2 re-ranking.
- [ ] **Cost / QPS Projection**:
  - Back-of-envelope calculation for cost per 1000 queries at SLA (p99 < 100ms).
- [ ] **10x Scale Breakdown**:
  - Architectural analysis of what bottlenecks first under 10x traffic.

---

## 5. Submissions & Final Deliverables (Q5 - Q9)
- [ ] Run predictions on large test sets (`MINDlarge_test.zip` and `ebnerd_testset.zip`).
- [ ] Package and submit predictions to Codabench leaderboards (MIND #13967 and RecSys #2469).
- [ ] Capture leaderboard confirmation screenshots.
- [ ] Write 6-page Design Note PDF in `docs/design_note/`.
