# CS4.406: Information Retrieval & Extraction — Large-Scale News Recommendation Pipeline

A reproducible, high-throughput, leak-free news recommendation and retrieval engine built on the **MIND** (Microsoft News Dataset) and **EB-NeRD** (Ekstra Bladet News Recommendation Dataset) benchmarks.

This repository implements:
1. **Strict Temporal Data Pipeline**: Raw interaction logs -> unified schema -> chronological train/val/test splits -> point-in-time feature store with zero temporal or target leakage.
2. **First-Stage Candidate Retrieval**: Lexical BM25 (via `bm25s`) and Dense Semantic vector retrieval (via HuggingFace embeddings and FAISS Flat/HNSW indexes).
3. **Second-Stage Re-Ranking**: Gradient-boosted decision trees (LightGBM Ranker / LambdaMART) scoring point-in-time user engagement and article features.
4. **Neural News Recommendation**: Vanilla NRMS (Neural News Recommendation with Multi-Head Self-Attention) and Improved NRMS incorporating subcategory attention and rich engagement signals.
5. **Anti-Gaming & Serving Safety (Q9)**: Production position bias neutralization eliminating position feedback loops while preserving > 99.7% ranking accuracy.
6. **Serving Benchmark & 10x Scaling Projections (Q4)**: Single-node p50/p95/p99 latency benchmarks, memory footprint analysis, AWS EC2 cost projections, and scaling bottleneck analysis.
7. **Codabench Submission Pipelines**: Automated generation and validation of official competition prediction archives for MIND Large (Competition #13967) and EB-NeRD Large (Competition #2469).

---

## Table of Contents
- [Setup & Installation](#setup--installation)
- [Repository Layout](#repository-layout)
- [File Catalog & Architectural Explanation](#file-catalog--architectural-explanation)
- [How to Run: Complete Command Reference](#how-to-run-complete-command-reference)
  - [1. Data Pipeline Rebuild](#1-data-pipeline-rebuild)
  - [2. Unit & Temporal Leakage Tests](#2-unit--temporal-leakage-tests)
  - [3. Dataset Statistics & EDA Reports](#3-dataset-statistics--eda-reports)
  - [4. Candidate Retrieval Evaluation (BM25 & Embeddings)](#4-candidate-retrieval-evaluation-bm25--embeddings)
  - [5. Two-Stage Retrieve-then-Rank Evaluation (Q2)](#5-two-stage-retrieve-then-rank-evaluation-q2)
  - [6. Neural NRMS & Ablation Studies (Q3)](#6-neural-nrms--ablation-studies-q3)
  - [7. Comprehensive Offline Evaluation & Serving Benchmark (Q4)](#7-comprehensive-offline-evaluation--serving-benchmark-q4)
  - [8. Anti-Gaming Verification (Q9)](#8-anti-gaming-verification-q9)
  - [9. Codabench Competition Submission Pipelines](#9-codabench-competition-submission-pipelines)
- [Makefile Targets Cheatsheet](#makefile-targets-cheatsheet)

---

## Setup & Installation

### Prerequisites
- Python 3.10+ (tested on Python 3.10–3.14)
- Virtual environment recommended:

```bash
# Clone the repository
git clone https://github.com/KGan31/IRE-Assignment-1.git
cd IRE-Assignment-1

# Create and activate virtual environment
python -m venv venv
# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# On Linux/macOS:
source venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

### Quick Verification
Run the 53-item test suite to verify pipeline integrity:
```bash
python -m pytest src/tests/
```

---

## Repository Layout

```text
IRE-Assignment-1/
├── configs/
│   └── pipeline.yaml             # Central configuration: paths, split ratios, feature parameters
│
├── data/                         # Data layer (strictly gitignored raw/interim/processed)
│   ├── raw/                      # Downloaded and uncompressed source archives
│   ├── interim/                  # Parquet tables parsed into the unified schema
│   ├── processed/                # Chronologically split impressions & materialized feature stores
│   └── stats/                    # Generated exploratory data analysis JSON and Markdown reports
│
├── docs/                         # Assignment reports, theoretical write-ups, and problem sheets
│   ├── assignments/              # Assignment handouts (A1.pdf, A2.pdf) and prompt logs
│   ├── design_note/              # Assignment 1 design note and LaTeX source
│   ├── design_note_2/            # Assignment 2 design note, figures, and LaTeX publication source
│   ├── math_formulae/            # Mathematical formulations (Markdown and compiled LaTeX PDF)
│   ├── ablations/                # Ablation benchmark tables and JSON metrics
│   └── schema.md                 # Formal dataset schema specifications across pipeline stages
│
├── models/                       # Serialized model checkpoints and weights (gitignored)
│   ├── *.pkl                     # Trained LightGBM Ranker models
│   ├── *.pt                      # PyTorch Neural NRMS checkpoints
│   └── *.json                    # Benchmark metric dumps
│
├── notebooks/                    # Exploratory Data Analysis (EDA) Jupyter notebooks
│   ├── ebnerd_analysis.ipynb     # Interactive EDA for EB-NeRD interactions, dwell, and scroll
│   └── mind_analysis.ipynb       # Interactive EDA for MIND articles, behaviors, and categories
│
├── results/                      # Empirical outputs and evaluation logs
│   ├── logs/                     # Verbose execution dumps and CLI outputs (*.txt, gitignored)
│   └── notes/                    # Question answer write-ups (Q2_*.md, Q3_*.md, Q4.md, Q9_*.md)
│
├── scripts/                      # Executable CLI runners and entrypoints
│   ├── experiments/              # Offline evaluation harnesses and ablation study runners
│   ├── stats/                    # Dataset exploration and statistical summary analyzers
│   └── submissions/              # Official Codabench prediction generator scripts
│
├── src/                          # Core reusable library package
│   ├── bm25.py                   # High-speed BM25 candidate retrieval engine
│   ├── download.py               # Automated dataset download and extraction pipeline
│   ├── embeddings.py             # Dense semantic vector retrieval with FAISS Flat & HNSW
│   ├── eval_bm25.py              # BM25 parameter sweep evaluation CLI
│   ├── eval_embeddings.py        # Embedding retrieval evaluation CLI
│   ├── eval_harness.py           # Unified offline evaluation harness with user/item slicing
│   ├── evaluate_ebnerd_large_validation.py # EB-NeRD Large validation set evaluation
│   ├── evaluate_mind_large_validation.py   # MIND Large validation set evaluation
│   ├── extract_mind_large_features.py      # Feature extraction for large-scale MIND data
│   ├── feature_store.py          # Materialized feature store builder
│   ├── features.py               # Point-in-time behavioral feature extractors & leak guards
│   ├── metrics.py                # Formal ranking, retrieval, diversity, and novelty metrics
│   ├── nrms_docvec.py            # Neural NRMS model architecture in PyTorch
│   ├── parse_ebnerd.py           # EB-NeRD Parquet parser with schema normalization
│   ├── parse_mind.py             # MIND TSV parser with unified schema mapping
│   ├── reranker.py               # LightGBM pointwise and pairwise LambdaMART re-rankers
│   ├── retrieve_then_rank.py     # End-to-end two-stage retrieve-then-rank pipeline orchestration
│   ├── schema.py                 # Unified schema dataclass contracts & validation helpers
│   ├── split.py                  # Strict chronological dataset splitting engine
│   └── tests/                    # 53 pytest unit and leakage verification tests
│
├── submissions/                  # Generated competition prediction zip archives
├── .env                          # Local environment variable overrides
├── .gitignore                    # Git tracking exclusions
├── AGENTS.md                     # Agent development constraints and guidelines
├── Makefile                      # Standard automation recipes
├── README.md                     # Project documentation
├── requirements.txt              # Pinned Python package dependencies
└── todo.md                       # Assignment backlog and implementation notes
```

---

## File Catalog & Architectural Explanation

### 1. Core Library (`src/`)

- [src/schema.py](src/schema.py): Defines the unified schema contract for articles, impressions, and user click history across MIND and EB-NeRD. Guarantees consistent types and field naming.
- [src/download.py](src/download.py): Handles dataset downloading, streaming extraction, checksums, and folder structuring for small, demo, and large dataset variants.
- [src/parse_mind.py](src/parse_mind.py): Ingests raw MIND TSV files (`news.tsv`, `behaviors.tsv`), normalizes timestamps, cleans text tokens, and writes interim Parquet tables.
- [src/parse_ebnerd.py](src/parse_ebnerd.py): Parses EB-NeRD Parquet files (`articles.parquet`, `behaviors.parquet`, `history.parquet`) with defensive column mapping, extracting rich engagement signals (scroll percentage, dwell time).
- [src/split.py](src/split.py): Enforces chronological train/val/test splits without future leakage. Emits split-specific cutoffs and bounded user history slices.
- [src/features.py](src/features.py): High-performance Polars/NumPy feature engineering: point-in-time history filtering (`click_time < impression_time`), exponential recency decay, category affinity distributions, and anti-gaming session position indicators.
- [src/feature_store.py](src/feature_store.py): Materializes `article_features.parquet` and split-specific `user_features_{train,val,test}.parquet` for rapid offline training and online serving.
- [src/bm25.py](src/bm25.py): High-speed BM25 index and retrieval engine using `bm25s`, supporting customizable tokenization, document length normalisation (parameter `b`), term saturation (parameter `k1`), and history length clipping.
- [src/embeddings.py](src/embeddings.py): Dense semantic representation builder supporting pre-trained embeddings, HuggingFace transformers, and FAISS indexing (`IndexFlatIP`, `IndexHNSWFlat`).
- [src/reranker.py](src/reranker.py): LightGBM re-ranking framework supporting binary cross-entropy and LambdaMART `lambdarank` objectives with grouped queries.
- [src/retrieve_then_rank.py](src/retrieve_then_rank.py): Orchestrates the complete two-stage inference pipeline: Stage 1 candidate generation -> point-in-time feature extraction -> Stage 2 LightGBM re-ranking.
- [src/nrms_docvec.py](src/nrms_docvec.py): PyTorch implementation of the NRMS (Neural News Recommendation with Multi-Head Self-Attention) model, with document encoder, user history encoder, and category/affinity projection heads.
- [src/metrics.py](src/metrics.py): Mathematically rigorous evaluation metrics including AUC, MRR, nDCG@K, Recall@K, Intra-List Diversity (ILD@K), Novelty@K, Catalog Coverage@K, and paired bootstrap 95% confidence intervals.
- [src/eval_harness.py](src/eval_harness.py): Offline evaluation harness supporting user cohort slicing (Cold-Start vs. Warm users) and item popularity slicing (Head vs. Tail articles).
- [src/evaluate_ebnerd_large_validation.py](src/evaluate_ebnerd_large_validation.py): Large-scale validation evaluator for the EB-NeRD LightGBM re-ranking pipeline.
- [src/evaluate_mind_large_validation.py](src/evaluate_mind_large_validation.py): Large-scale validation evaluator for the MIND LightGBM re-ranking pipeline.
- [src/tests/](src/tests/): Comprehensive test suite covering behavioral window bounds, temporal leakage, feature calculation, metrics, BM25, embeddings, and NRMS architectures.

### 2. Experiment & Evaluation Runners (`scripts/experiments/`)

- [scripts/experiments/run_ablations.py](scripts/experiments/run_ablations.py): Executes multi-phase ablation sweeps: input fields (title vs. title+abstract vs. body), history lengths (`H in {10, 20, 30}`), cold-start thresholds, and vector index types.
- [scripts/experiments/run_ann_vs_exact_ablation.py](scripts/experiments/run_ann_vs_exact_ablation.py): Benchmarks FAISS Exact Flat search against Approximate Nearest Neighbor (HNSW), comparing query latency, QPS, Recall@K preservation, and diversity.
- [scripts/experiments/run_q4_eval.py](scripts/experiments/run_q4_eval.py): Comprehensive evaluation of BM25 vs. Dense Semantic retrieval across user cohorts (cold-start bottom 2% vs. warm top 98%) and article popularity (head top 20% vs. tail bottom 80%) with 1,000-sample bootstrap 95% CIs.
- [scripts/experiments/run_q4_serving_benchmark.py](scripts/experiments/run_q4_serving_benchmark.py): Measures Stage 1 retrieval, feature extraction, and Stage 2 re-ranking latency profiles (mean, p50, p90, p95, p99), index memory footprints, concurrency scaling, and cloud serving costs.
- [scripts/experiments/run_q2_extended_eval.py](scripts/experiments/run_q2_extended_eval.py): Evaluates the two-stage Retrieve-then-Rank pipeline on validation and test splits with cohort breakdown.
- [scripts/experiments/run_q3_ablation.py](scripts/experiments/run_q3_ablation.py): Comparative study of Mean Pooling Embedding Baseline vs. Vanilla NRMS vs. Improved NRMS.
- [scripts/experiments/run_q3_affinity.py](scripts/experiments/run_q3_affinity.py): Measures ranking deltas and paired bootstrap confidence intervals when incorporating dwell-time weighted category affinity on EB-NeRD.
- [scripts/experiments/run_q3_subcategory.py](scripts/experiments/run_q3_subcategory.py): Evaluates the impact of fine-grained subcategory attention projections on MIND.
- [scripts/experiments/run_antigaming_eval.py](scripts/experiments/run_antigaming_eval.py): Verifies Question 9 anti-gaming behavior by comparing rankings with logged impression position vs. neutralized position bias (`session_position = 0.0`).
- [scripts/experiments/run_nrms_baseline.py](scripts/experiments/run_nrms_baseline.py): Trains and evaluates the neural NRMS model on pre-computed document vectors.

### 3. Statistics & EDA Analyzers (`scripts/stats/`)

- [scripts/stats/processed_stats.py](scripts/stats/processed_stats.py): Analyzes processed splits, generating temporal bounds, CTR distributions, cold-start ratios, and catalog coverage statistics.
- [scripts/stats/ebnerd_stats.py](scripts/stats/ebnerd_stats.py): Analyzes raw EB-NeRD Parquet datasets.
- [scripts/stats/mind_stats.py](scripts/stats/mind_stats.py): Analyzes raw MIND TSV datasets.

### 4. Codabench Competition Generators (`scripts/submissions/`)

- [scripts/submissions/generate_mind_submission.py](scripts/submissions/generate_mind_submission.py): Generates official Codabench prediction zip archives using Lexical BM25 for MIND Large.
- [scripts/submissions/generate_mind_semantic_submission.py](scripts/submissions/generate_mind_semantic_submission.py): Generates Codabench predictions using Dense Semantic vector retrieval for MIND Large.
- [scripts/submissions/generate_mind_reranker_submission.py](scripts/submissions/generate_mind_reranker_submission.py): Generates Codabench predictions using the two-stage LightGBM Re-Ranker for MIND Large.
- [scripts/submissions/generate_ebnerd_submission.py](scripts/submissions/generate_ebnerd_submission.py): Generates official Codabench predictions using BM25 for EB-NeRD Large.
- [scripts/submissions/generate_ebnerd_semantic_submission.py](scripts/submissions/generate_ebnerd_semantic_submission.py): Generates Codabench predictions using Dense Semantic embeddings for EB-NeRD Large.
- [scripts/submissions/generate_ebnerd_reranker_submission.py](scripts/submissions/generate_ebnerd_reranker_submission.py): Generates Codabench predictions using the two-stage LightGBM Re-Ranker for EB-NeRD Large.
- [scripts/submissions/generate_nrms_codabench_submission.py](scripts/submissions/generate_nrms_codabench_submission.py): Generates Codabench predictions using the improved Neural NRMS model for MIND, EB-NeRD, or both.

---

## How to Run: Complete Command Reference

### 1. Data Pipeline Rebuild

To run the complete data pipeline from raw files to materialized feature store in one command:
```bash
make data
```

Or execute individual stages:
```bash
# Step 1: Download raw datasets
python src/download.py --config configs/pipeline.yaml

# Step 2: Parse into unified schema
python src/parse_mind.py --split train
python src/parse_mind.py --split dev
python src/parse_ebnerd.py --split train
python src/parse_ebnerd.py --split validation

# Step 3: Strict chronological split
python src/split.py --config configs/pipeline.yaml

# Step 4: Materialize feature store
python src/feature_store.py --config configs/pipeline.yaml

# Step 5: Compute dataset statistics
python scripts/stats/processed_stats.py
```

---

### 2. Unit & Temporal Leakage Tests

Run all 53 automated unit and leakage tests:
```bash
# Using Makefile
make test

# Or directly with pytest
python -m pytest src/tests/ -v
```

Verifies:
- **No Temporal Overlap**: `max(train_timestamp) <= min(val_timestamp) <= max(val_timestamp) <= min(test_timestamp)`.
- **No Impression Duplicate**: Impression IDs are strictly unique per split.
- **Behavioral Boundary**: Historical click features never include events where `click_time >= impression_time`.
- **Target Leakage Immunity**: Labels of candidate items never leak into impression features.

---

### 3. Dataset Statistics & EDA Reports

Generate dataset distribution metrics, CTR statistics, and temporal coverage reports:
```bash
# Processed splits summary (both datasets)
python scripts/stats/processed_stats.py --dataset all --output_json data/processed_stats.json --output_md data/processed_stats_report.md

# Raw MIND dataset EDA
python scripts/stats/mind_stats.py --output_md data/mind_stats_report.md --output_json data/mind_stats.json

# Raw EB-NeRD dataset EDA
python scripts/stats/ebnerd_stats.py --output_md data/ebnerd_stats_report.md --output_json data/ebnerd_stats.json
```

---

### 4. Candidate Retrieval Evaluation (BM25 & Embeddings)

Evaluate candidate retrieval engines across datasets:
```bash
# Run BM25 evaluation across both MIND and EB-NeRD validation splits
python src/eval_bm25.py --dataset all --split val --max_history_len 20 --eval_mode all

# Run Dense Semantic evaluation using FAISS HNSW
python src/eval_embeddings.py --dataset all --split val --index_type ann --max_history_len 20
```

---

### 5. Two-Stage Retrieve-then-Rank Evaluation (Q2)

Run the full Retrieve-then-Rank pipeline evaluation (BM25/Semantic Stage 1 -> LightGBM Stage 2):
```bash
# Evaluate on EB-NeRD test split with cold-start & head/tail slicing
python scripts/experiments/run_q2_extended_eval.py --dataset ebnerd --split test --n_bootstraps 1000

# Evaluate on MIND test split
python scripts/experiments/run_q2_extended_eval.py --dataset mind --split test --n_bootstraps 1000

# Fast dry-run on validation split
python scripts/experiments/run_q2_extended_eval.py --dataset mind --split val --n_bootstraps 100
```

---

### 6. Neural NRMS & Ablation Studies (Q3)

Train and evaluate Neural NRMS models and run statistical ablation studies:
```bash
# Run comparative ablation across all models (Mean Pooling vs Vanilla NRMS vs Improved NRMS)
python scripts/experiments/run_q3_ablation.py

# Run category & dwell-time affinity ablation on EB-NeRD
python scripts/experiments/run_q3_affinity.py

# Run fine-grained subcategory attention ablation on MIND
python scripts/experiments/run_q3_subcategory.py

# Train NRMS baseline with custom learning rate and epochs
python scripts/experiments/run_nrms_baseline.py --dataset ebnerd --epochs 5 --lr 0.0001
```

---

### 7. Comprehensive Offline Evaluation & Serving Benchmark (Q4)

Execute the production serving benchmark and multi-slice offline evaluation:
```bash
# Run comprehensive Q4 evaluation (BM25 vs Embeddings across cohorts with 1,000 bootstrap CIs)
python scripts/experiments/run_q4_eval.py

# Run FAISS ANN (HNSW) vs Exact Cosine latency, throughput, and recall ablation
python scripts/experiments/run_ann_vs_exact_ablation.py

# Run end-to-end serving-time benchmark (latency percentiles, memory, QPS, cloud cost)
python scripts/experiments/run_q4_serving_benchmark.py --sample_queries 2000
```

---

### 8. Anti-Gaming Verification (Q9)

Verify that serving-time position bias neutralization (`session_position = 0.0`) eliminates ranking manipulation with negligible accuracy delta:
```bash
# Run anti-gaming position neutralization evaluation on both datasets
python scripts/experiments/run_antigaming_eval.py --dataset all --split test --n_bootstraps 1000
```

---

### 9. Codabench Competition Submission Pipelines

Generate submission zip archives containing formatted `predictions.txt` files for official Codabench competition leaderboards:

#### MIND Large (Competition #13967)
```bash
# 1. Download MIND Large
python src/download.py --dataset mind_large

# 2. Parse MIND Large into unified schema
python src/parse_mind.py --dataset_type large --split all

# 3. Generate BM25 Submission
python scripts/submissions/generate_mind_submission.py --dataset_type large --eval_dev --output_dir submissions

# 4. Generate Dense Semantic Submission
python scripts/submissions/generate_mind_semantic_submission.py --dataset_type large --eval_dev --output_dir submissions_semantic

# 5. Generate LightGBM Two-Stage Re-Ranker Submission
python scripts/submissions/generate_mind_reranker_submission.py --dataset_type large --eval_dev --output_dir submissions/submissions_mind_reranker

# 6. Generate Neural NRMS Subcategory Submission
python scripts/submissions/generate_nrms_codabench_submission.py --dataset mind
```

#### EB-NeRD Large (Competition #2469)
```bash
# 1. Download EB-NeRD Large & Testset
python src/download.py --dataset ebnerd_large

# 2. Parse EB-NeRD Large into unified schema
python src/parse_ebnerd.py --dataset_type large --split all

# 3. Generate BM25 Title+Abstract Submission
python scripts/submissions/generate_ebnerd_submission.py --dataset_type large --eval_dev --output_dir submissions_ebnerd

# 4. Generate BM25 Fulltext (Title+Abstract+Body) Submission
python scripts/submissions/generate_ebnerd_submission.py --dataset_type large --include_body --eval_dev --output_dir submissions_ebnerd_fulltext

# 5. Generate Dense Semantic Submission
python scripts/submissions/generate_ebnerd_semantic_submission.py --dataset_type large --eval_dev --output_dir submissions_ebnerd_semantic

# 6. Generate LightGBM Two-Stage Re-Ranker Submission
python scripts/submissions/generate_ebnerd_reranker_submission.py --dataset_type large --output_dir submissions/submissions_ebnerd_reranker

# 7. Generate Neural NRMS Affinity Submission
python scripts/submissions/generate_nrms_codabench_submission.py --dataset ebnerd
```

---

## Makefile Targets Cheatsheet

| Target | Description | Underlying Command |
| :--- | :--- | :--- |
| `make data` | Rebuild full data pipeline | `download`, `parse`, `split`, `features`, `stats` |
| `make download` | Download raw MIND and EB-NeRD demo | `python src/download.py --config configs/pipeline.yaml` |
| `make parse` | Parse raw files to unified interim Parquet | `python src/parse_mind.py ...; python src/parse_ebnerd.py ...` |
| `make split` | Chronological train/val/test splits | `python src/split.py --config configs/pipeline.yaml` |
| `make features` | Materialize article and user feature stores | `python src/feature_store.py --config configs/pipeline.yaml` |
| `make stats` | Compute processed split statistics | `python scripts/stats/processed_stats.py` |
| `make bm25` | Evaluate global BM25 candidate retrieval | `python src/eval_bm25.py --dataset all --split val ...` |
| `make test` | Run complete unit and leakage test suite | `pytest src/tests/ -v` |
| `make submit-mind-bm25` | Generate MIND BM25 Codabench zip | `python scripts/submissions/generate_mind_submission.py ...` |
| `make submit-mind-semantic` | Generate MIND Semantic Codabench zip | `python scripts/submissions/generate_mind_semantic_submission.py ...` |
| `make submit-mind-reranker` | Generate MIND Re-Ranker Codabench zip | `python scripts/submissions/generate_mind_reranker_submission.py ...` |
| `make submit-ebnerd-bm25` | Generate EB-NeRD BM25 Codabench zip | `python scripts/submissions/generate_ebnerd_submission.py ...` |
| `make submit-ebnerd-bm25-fulltext` | Generate EB-NeRD BM25 + Body Codabench zip | `python scripts/submissions/generate_ebnerd_submission.py --include_body ...` |
| `make submit-ebnerd-semantic` | Generate EB-NeRD Semantic Codabench zip | `python scripts/submissions/generate_ebnerd_semantic_submission.py ...` |
| `make submit-ebnerd-reranker` | Generate EB-NeRD Re-Ranker Codabench zip | `python scripts/submissions/generate_ebnerd_reranker_submission.py ...` |
| `make submit-ebnerd-nrms-affinity` | Generate EB-NeRD NRMS Codabench zip | `python scripts/submissions/generate_nrms_codabench_submission.py --dataset ebnerd` |
| `make submit-mind-nrms-subcategory` | Generate MIND NRMS Codabench zip | `python scripts/submissions/generate_nrms_codabench_submission.py --dataset mind` |
| `make submit-nrms-both` | Generate NRMS Codabench zips for both | `python scripts/submissions/generate_nrms_codabench_submission.py --dataset both` |
| `make clean` | Remove intermediate & processed datasets | `rm -rf data/interim/* data/processed/*` |
