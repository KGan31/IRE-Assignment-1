# Unified Dataset & Pipeline Schema Documentation

This document provides a comprehensive reference of all data files across the **raw**, **interim**, and **processed** pipeline stages for both the **MIND** and **EB-NeRD** datasets. It details the columns, schemas, semantics, similarities, differences, and why each stage is critical for recommendation and retrieval systems.

---

## 1. Overview of Pipeline Architecture

The pipeline processes diverse input formats into a unified, leak-free schema:

```
[ Raw Layer (data/raw/) ]
  ├── MIND: TSV files + Knowledge Graph vectors (train / dev / test)
  └── EB-NeRD: Parquet files + BERT/Word2Vec embeddings (train / validation / test)
              │
              ▼ (src/parse_mind.py & src/parse_ebnerd.py)
[ Interim Layer (data/interim/) ]
  Unified Schema per dataset & split:
  ├── articles_{split}.parquet
  ├── impressions_{split}.parquet
  └── history_{split}.parquet
              │
              ▼ (src/split.py & src/feature_store.py)
[ Processed Layer (data/processed/) ]
  Strict temporal split + feature store + index caches:
  ├── articles.parquet (global deduplicated catalog)
  ├── impressions_{train,val,test}.parquet
  ├── history_{train,val,test}.parquet
  ├── article_features.parquet
  ├── user_features_{train,val,test}.parquet
  ├── article_embeddings.npy
  └── article_ids.json
```

---

## 2. Raw Data Layer (`data/raw/`)

The raw layer contains the original uncompressed files released by Microsoft Research (MIND) and Ekstra Bladet (EB-NeRD).

### A. MIND Dataset (`data/raw/mind/` & `data/raw/mind_large/`)
Formats: TSV (tab-separated values) and `.vec` text files.

#### 1. `news.tsv` (Article Catalog)
- **`news_id`** (`str`): Unique article identifier in Microsoft News (e.g., `N12345`).
- **`category`** (`str`): High-level topical section (e.g., `news`, `sports`, `finance`).
- **`subcategory`** (`str`): Granular topic category (e.g., `football_nfl`, `personalfinance`).
- **`title`** (`str`): Headline of the news article.
- **`abstract`** (`str`): Brief synopsis/summary of the news article.
- **`url`** (`str`): Web URL of the article on MSN.
- **`title_entities`** (`str`): JSON string of extracted Wikipedia/Wikidata entities in the title, with entity name, Wikidata ID, character offsets, and confidence.
- **`abstract_entities`** (`str`): JSON string of extracted Wikipedia/Wikidata entities in the abstract.

#### 2. `behaviors.tsv` (Impression & User Behavior Logs)
- **`impression_id`** (`int` / `str`): Unique identifier for a page impression log.
- **`user_id`** (`str`): Anonymized user identifier (e.g., `U13740`).
- **`time`** (`str`): Timestamp string of the impression (format: `%m/%d/%Y %I:%M:%S %p`).
- **`history`** (`str`): Space-separated list of previously clicked news IDs (e.g., `N123 N456 N789`).
- **`impressions`** (`str`): Space-separated list of candidate news IDs shown to the user in this session, with a click label suffix `-0` (not clicked) or `-1` (clicked) (e.g., `N123-0 N456-1 N789-0`).

#### 3. `entity_embedding.vec` & `relation_embedding.vec` (Knowledge Graph Embeddings)
- Tab-delimited files containing 100-dimensional TransE embeddings for Wikidata entities and relation types present in `news.tsv`.

---

### B. EB-NeRD Dataset (`data/raw/ebnerd/` & `data/raw/ebnerd_large/`)
Formats: Apache Parquet files.

#### 1. `articles.parquet` (Article Catalog)
- **`article_id`** (`int32`): Numeric ID assigned to the Ekstra Bladet article.
- **`title`** (`str`): Article headline.
- **`subtitle`** (`str`): Secondary headline / lead paragraph (equivalent to MIND's abstract).
- **`body`** (`str`): Full article body text (Danish).
- **`category`** (`int16`): Numerical category ID.
- **`subcategory`** (`list[int16]`): List of numerical subcategory tags.
- **`category_str`** (`str`): Human-readable category label (e.g., `sport`, `krimi`, `nyheder`).
- **`published_time`** (`datetime`): Exact publication timestamp.
- **`last_modified_time`** (`datetime`): Last editorial update timestamp.
- **`premium`** (`bool`): Whether the article requires a paid subscription.
- **`article_type`** (`str`): Format type (e.g., `standard`, `gallery`, `video`).
- **`url`** (`str`): Canonical publication URL.
- **`ner_clusters`** (`list[str]`): Named entities clustered across documents.
- **`entity_groups`** (`list[str]`): High-level entity classification categories.
- **`topics`** (`list[str]`): Editorial topical tags.
- **`image_ids`** (`list[int64]`): Attached image asset IDs.
- **`total_inviews`** (`int32`): Aggregate impressions count across the platform.
- **`total_pageviews`** (`int32`): Aggregate article clicks/pageviews.
- **`total_read_time`** (`float32`): Aggregate seconds users spent reading the article.
- **`sentiment_score`** (`float32`): Continuous sentiment polarity score.
- **`sentiment_label`** (`str`): Categorical sentiment classification (`Positive`, `Neutral`, `Negative`).

#### 2. `behaviors.parquet` (Session Impression Logs)
- **`impression_id`** (`uint32`): Unique identifier for the impression.
- **`article_id`** (`int32`): Current article where the impression list was rendered.
- **`impression_time`** (`datetime`): Timestamp of the impression.
- **`user_id`** (`uint32`): Anonymized user ID.
- **`article_ids_inview`** (`list[int32]`): List of candidate article IDs displayed on screen.
- **`article_ids_clicked`** (`list[int32]`): List of article IDs clicked from the candidate list.
- **`read_time`** (`float32`): Reading duration in seconds for the session.
- **`scroll_percentage`** (`float32`): Maximum scroll depth percentage on the page.
- **`device_type`** (`int8`): User device category (1=desktop, 2=mobile, 3=tablet).
- **`is_sso_user`** (`bool`): Whether the user is logged into Single Sign-On.
- **`is_subscriber`** (`bool`): Whether the user is an active paid subscriber.
- **`gender`**, **`age`**, **`postcode`** (`int8`): Anonymized demographic buckets.
- **`session_id`** (`uint32`): Browser session identifier.
- **`next_read_time`**, **`next_scroll_percentage`** (`float32`): Engagement metrics on subsequently navigated pages.

#### 3. `history.parquet` (Historical Interaction Logs)
- **`user_id`** (`uint32`): Anonymized user ID.
- **`article_id_fixed`** (`list[int32]`): Chronological array of previously clicked article IDs.
- **`impression_time_fixed`** (`list[datetime]`): Exact timestamps corresponding to each historical click.
- **`read_time_fixed`** (`list[float32]`): Dwell time in seconds for each historical click.
- **`scroll_percentage_fixed`** (`list[float32]`): Scroll depth for each historical click.

#### 4. `bert_base_multilingual_cased.parquet` (Official Embeddings)
- **`article_id`** (`int32`): Article ID.
- **`google-bert/bert-base-multilingual-cased`** (`list[float32]`): 768-dimensional multilingual BERT embedding vector.

---

## 3. Interim Data Layer (`data/interim/`)

The interim layer transforms raw data into a strictly standardized schema defined in `src/schema.py`. All columns are normalized, namespaces (`mind_` / `ebnerd_`) are prepended to IDs to prevent collisions, and all complex structures are normalized into Apache Parquet.

### Table 1: `articles_{split}.parquet`
| Column Name | Data Type | Semantics & Description |
| :--- | :--- | :--- |
| **`article_id`** | `str` | Globally unique namespaced ID (e.g., `mind_N12345`, `ebnerd_123456`). |
| **`dataset`** | `str` | Origin dataset tag (`mind` or `ebnerd`). |
| **`title`** | `str` | Article headline (clean text). |
| **`abstract`** | `str` | Article synopsis or summary (for EB-NeRD, populated from `subtitle`). |
| **`body`** | `str` | Full article body text (empty string for MIND, full text for EB-NeRD). |
| **`category`** | `str` | Hierarchical category path: `category/subcategory` for MIND, `category_str` for EB-NeRD. |
| **`published_time`** | `datetime64[ns]` | Publication timestamp (null for MIND, exact for EB-NeRD). |
| **`entities`** | `list[str]` | Unified list of extracted named entity names. |
| **`embedding`** | `object` | Optional pre-computed embedding vector (or null). |

### Table 2: `impressions_{split}.parquet`
| Column Name | Data Type | Semantics & Description |
| :--- | :--- | :--- |
| **`impression_id`** | `str` | Globally unique namespaced impression ID (e.g., `mind_101`, `ebnerd_202`). |
| **`dataset`** | `str` | Origin dataset tag (`mind` or `ebnerd`). |
| **`user_id`** | `str` | Globally unique namespaced user ID (e.g., `mind_U123`, `ebnerd_456`). |
| **`timestamp`** | `datetime64[ns]` | Exact timestamp when the recommendation slate was presented. |
| **`candidate_article_ids`** | `list[str]` | List of namespaced candidate article IDs shown to the user. |
| **`clicked_article_ids`** | `list[str]` | Subset of candidate article IDs that the user actually clicked. |
| **`session_context`** | `dict` / `null` | Contextual metadata (e.g., device, read time, scroll depth if available). |

### Table 3: `history_{split}.parquet`
| Column Name | Data Type | Semantics & Description |
| :--- | :--- | :--- |
| **`dataset`** | `str` | Origin dataset tag (`mind` or `ebnerd`). |
| **`user_id`** | `str` | Namespaced user ID. |
| **`clicked_article_id`** | `str` | Namespaced article ID that the user read. |
| **`click_time`** | `datetime64[ns]` | Timestamp when the click occurred (exact for EB-NeRD; impression timestamp approximation for MIND). |

---

## 4. Processed Data Layer (`data/processed/`)

The processed layer represents the finalized, leakage-free datasets ready for model training, candidate retrieval, feature stores, and offline evaluation.

### A. Core Dataset Partitions
1. **`articles.parquet`**:
   - Single, global, deduplicated catalog of all unique articles across train, val, and test.
   - Same schema as `articles_{split}.parquet`.
2. **`impressions_train.parquet`**, **`impressions_val.parquet`**, **`impressions_test.parquet`**:
   - Impressions partitioned strictly by timestamp (never random sampling).
   - Guarantees: `train_timestamps < val_timestamps < test_timestamps`.
3. **`history_train.parquet`**, **`history_val.parquet`**, **`history_test.parquet`**:
   - Unrolled click interaction tables filtered by strict temporal cutoff to prevent data leakage.
   - `history_train` contains clicks where `click_time < train_cutoff`.
   - `history_val` contains clicks where `click_time < val_cutoff` (cumulative past history).

---

### B. Feature Store Tables (`src/feature_store.py`)

#### 1. `article_features.parquet`
- Contains static metadata features per article: `article_id`, `dataset`, `title`, `abstract`, `body`, `category`, `published_time`, `entities`.

#### 2. `user_features_{train,val,test}.parquet`
Built using only history available prior to that split's cutoff timestamp:
- **`user_id`** (`str`): Namespaced user ID.
- **`click_history`** (`list[str]`): List of previously clicked article IDs.
- **`n_clicks`** (`uint32`): Total number of historical clicks.
- **`recency_score`** (`float64`): Exponentially decayed click score:
  `weight = 0.5 ** (age_hours / half_life_hours)`.
  Accounts for user interest freshness and velocity.

---

### C. Dense Retrieval Index Caches (`src/embeddings.py`)

1. **`article_embeddings.npy`**:
   - Binary NumPy matrix of shape `(N, D)` where `N` is the number of articles and `D` is embedding dimensionality (384 for MiniLM).
   - L2-normalized float32 vectors.
2. **`article_ids.json`**:
   - Array of strings mapping row index `i` in `article_embeddings.npy` to `article_id`.

---

## 5. Similarities and Differences Between Datasets

### Similarities
1. **Underlying Task**: Both datasets formulate news recommendation as candidate ranking: given user history and candidate slate, rank the articles to maximize Click-Through Rate (CTR) and ranking metrics (AUC, MRR, nDCG).
2. **Impression Structure**: Both datasets capture impression slates with candidates (`inview`) and binary engagement signals (`clicked`).
3. **Sequential History**: Both datasets track sequential past clicks per user.

### Key Differences
| Characteristic | MIND (Microsoft News) | EB-NeRD (Ekstra Bladet) |
| :--- | :--- | :--- |
| **Language** | English | Danish |
| **Raw Format** | Tab-separated TSVs | Columnar Apache Parquet |
| **Article Body Text** | ❌ None (Only Title + Abstract) | ✅ Full article body text included |
| **Click Timestamps in History** | ❌ No individual click timestamps (clicks grouped as string list) | ✅ Exact timestamp for every historical click (`impression_time_fixed`) |
| **Dwell & Engagement Signals** | ❌ Binary click only | ✅ Continuous read time, scroll percentage, and session IDs |
| **Knowledge Graph** | ✅ Pre-linked Wikidata entities and TransE vectors | ❌ None (only NER clusters and editorial topics) |
| **Publication Times** | ❌ Not provided | ✅ Exact article publication and update times |

---

## 6. Why Each Layer is Important

### 1. Why the Raw Layer is Important
- **Source of Truth**: Preserves untouched original provider logs for reproducibility and provenance.
- **Auditing**: Allows verifying parsing logic and recovering fields that were initially unparsed.

### 2. Why the Interim Layer is Important
- **Format Homogenization**: Converts disparate TSV strings and nested Parquet arrays into identical columnar structures.
- **Namespacing & Collision Avoidance**: Prefixes like `mind_` and `ebnerd_` ensure datasets can be pooled into a single model without ID collisions.
- **Validation**: Enforces strict schema contracts (`src/schema.py`) to catch corrupted rows, missing columns, or type mismatches early.

### 3. Why the Processed Layer is Important
- **Zero Temporal Leakage**: Enforces strict chronological splits (`t_train < t_val < t_test`). In real production systems, models never have access to tomorrow's data. Random splitting in time-series data produces overly optimistic, invalid evaluation metrics.
- **Deterministic Feature Store**: Freezes offline feature computations (recency decay, click counts, embeddings) so all candidate retrieval and ranking models train and evaluate on identical, reproducible inputs.
- **High-Throughput Retrieval**: Binary embedding caches (`.npy`) and pre-parsed catalog Parquet files reduce retrieval setup latency from minutes to milliseconds.
