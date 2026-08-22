# CS4.406 Assignment 1 — Q1 Data Pipeline

Reproducible pipeline: raw MIND-small + EB-NeRD-demo files → unified schema →
temporal train/val/test split → feature store. One command rebuilds everything.

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## Run the full pipeline

```bash
make data
```

This runs, in order:
1. `download` — pulls MIND-small (train/dev) and EB-NeRD-demo zips into `data/raw/`
2. `parse` — parses both datasets into the unified schema (`src/schema.py`),
   writes to `data/interim/{mind,ebnerd}/`
3. `split` — temporal (never random) train/val/test split per dataset,
   writes to `data/processed/{mind,ebnerd}/`
4. `features` — builds `article_features.parquet` and per-split
   `user_features_{train,val,test}.parquet`

Run stages individually with `make download`, `make parse`, `make split`,
`make features`.

## Codabench Submission Pipelines (MIND & EB-NeRD Large)

For official competition submissions on Codabench:

### 1. MIND Large (Competition #13967)
```bash
# Download and unzip MIND large
python src/download.py --dataset mind_large

# Parse into unified schema (train, dev, test)
python src/parse_mind.py --dataset_type large --split all

# Evaluate on dev and generate Codabench prediction.zip
python src/generate_mind_submission.py --dataset_type large --eval_dev --output_dir submissions
```

### 2. EB-NeRD Large (Competition #2469)
```bash
# Download and unzip EB-NeRD large & testset
python src/download.py --dataset ebnerd_large

# Parse into unified schema (train, validation, test)
python src/parse_ebnerd.py --dataset_type large --split all

# Evaluate on validation and generate Codabench predictions.zip
python src/generate_ebnerd_submission.py --dataset_type large --eval_dev --output_dir submissions_ebnerd
```

## Run the leakage tests

```bash
make test
```


Verifies (Q9 requirement):
- No timestamp overlap between train/val/test impressions
- No duplicate impressions across splits
- User click-history features never include clicks at/after the split's
  cutoff timestamp

## Layout

```
data/raw/         downloaded zips, unzipped raw files (gitignored)
data/interim/     per-dataset parsed data in unified schema (gitignored)
data/processed/   split impressions + feature store (gitignored)
src/schema.py     unified schema contract — articles / impressions / user_history
src/download.py   dataset download + unzip
src/parse_mind.py     MIND TSVs -> unified schema
src/parse_ebnerd.py   EB-NeRD parquet -> unified schema
src/split.py      temporal split + leak-free history cutoff helper
src/feature_store.py  article + user feature tables
src/tests/        pytest leakage tests
configs/pipeline.yaml  all paths, URLs, split window sizes, feature params
```

## Design notes / known simplifications (flag these in the Q6 write-up)

- EB-NeRD column names have drifted slightly across dataset releases —
  `parse_ebnerd.py` reads defensively (`col()` helper) but verify against
  your actual downloaded schema with `df.columns.tolist()` before trusting it.
- MIND does not timestamp individual history clicks; `parse_mind.py`
  approximates each history click's time as the owning impression's
  timestamp. This is an upper bound, not the true click time.
- `feature_store.py` computes user features once per split using the
  split's *earliest* impression timestamp as the cutoff (conservative,
  simple). A tighter version would recompute history per-impression —
  worth mentioning as a scale/precision tradeoff in the design note.
- Article embeddings (EB-NeRD word2vec/BERT artifacts) are not loaded
  yet — `embedding` is left `None` in `parse_ebnerd.py`; wire this up in Q3.

## Gitignore reminder (Q8)

Make sure raw/interim/processed data and checkpoints are excluded:

```
data/
*.zip
*.pt
*.ckpt
__pycache__/
```
