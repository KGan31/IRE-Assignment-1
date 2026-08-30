# Mathematical Formulations and Explanations

This document provides a comprehensive reference of all mathematical formulas, definitions, objective functions, evaluation metrics, and statistical methods implemented across the project codebase (MIND and EB-NeRD news recommendation and candidate retrieval system).

---

## 1. Lexical Candidate Generation & Scoring: Okapi BM25
*Implemented in [`src/bm25.py`](file:///c:/Users/kavan/OneDrive/Desktop/IIITH/sem3/IRE/Assignment1/IRE-Assignment-1/src/bm25.py), [`src/generate_mind_submission.py`](file:///c:/Users/kavan/OneDrive/Desktop/IIITH/sem3/IRE/Assignment1/IRE-Assignment-1/src/generate_mind_submission.py), and [`src/generate_ebnerd_submission.py`](file:///c:/Users/kavan/OneDrive/Desktop/IIITH/sem3/IRE/Assignment1/IRE-Assignment-1/src/generate_ebnerd_submission.py).*

### 1.1 Robertson–Spärck Jones Inverse Document Frequency (IDF)
$$\text{IDF}(t) = \ln\left(1 + \frac{N - \text{df}(t) + 0.5}{\text{df}(t) + 0.5}\right)$$

#### Parameter Definitions
* $N$: Total number of articles in the indexed catalog.
* $\text{df}(t)$: Document frequency of term $t$ (the number of documents containing term $t$).
* $+0.5$: Robertson–Spärck Jones smoothing constant preventing division by zero and negative weights for high-frequency terms.

#### Significance & Mechanism
Measures the specificity and informativeness of term $t$. Rare, discriminative keywords receive large positive weights, whereas ubiquitous stopwords with high $\text{df}(t)$ asymptote to zero, preventing them from dominating candidate matching.

---

### 1.2 Okapi BM25 Scoring Function
$$\text{Score}_{\text{BM25}}(Q, D) = \sum_{t \in Q \cap D} \text{IDF}(t) \cdot \frac{\text{TF}(t, D) \cdot (k_1 + 1)}{\text{TF}(t, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}$$

#### Parameter Definitions
* $\text{TF}(t, D)$: Term frequency (count of occurrences of term $t$ in candidate document $D$).
* $|D|$: Length of candidate document $D$ (total token count in title + abstract).
* $\text{avgdl}$: Average document length across all $N$ catalog articles:
  $$\text{avgdl} = \frac{1}{N} \sum_{i=1}^N |D_i|$$
* $k_1 \in [0, \infty)$ (set to $1.5$): Controls term frequency saturation non-linearity. As $\text{TF}(t, D)$ increases, its contribution saturates towards $(k_1 + 1)$.
* $b \in [0, 1]$ (set to $0.75$): Controls document length penalization ($b=0 \implies$ no length normalization; $b=1 \implies$ full length scaling).

#### Significance & Mechanism
Calculates lexical similarity between the pseudo-query $Q$ (concatenated recent click history of the user) and a candidate article $D$. It balances non-linear term frequency saturation against length verbosity penalty.

---

## 2. Dense Semantic Retrieval & User Modeling
*Implemented in [`src/embeddings.py`](file:///c:/Users/kavan/OneDrive/Desktop/IIITH/sem3/IRE/Assignment1/IRE-Assignment-1/src/embeddings.py) and [`src/eval_embeddings.py`](file:///c:/Users/kavan/OneDrive/Desktop/IIITH/sem3/IRE/Assignment1/IRE-Assignment-1/src/eval_embeddings.py).*

### 2.1 Vector $L_2$ Normalization
$$\mathbf{\hat{v}} = \frac{\mathbf{v}}{\|\mathbf{v}\|_2 + \epsilon} = \frac{\mathbf{v}}{\sqrt{\sum_{k=1}^d v_k^2} + \epsilon}$$

#### Parameter Definitions
* $\mathbf{v} \in \mathbb{R}^d$: Unnormalized dense embedding vector ($d=384$ for MiniLM-L6-v2 / Multilingual BERT).
* $\epsilon = 10^{-12}$: Small constant preventing division by zero for null vectors.

#### Significance & Mechanism
Projects all article and user embedding representations onto the unit hypersphere $S^{d-1}$. This guarantees that inner products directly equal cosine similarities, allowing high-throughput Maximum Inner Product Search (MIPS) in FAISS.

---

### 2.2 User Representation via History Centroid Pooling
$$\mathbf{u} = \text{normalize}\left(\frac{\sum_{i=1}^{m} w_i \cdot \mathbf{e}_{a_i}}{\sum_{i=1}^{m} w_i + \epsilon}\right)$$

#### Parameter Definitions
* $H_u = [a_1, a_2, \dots, a_m]$: Ordered sequence of up to $m = 20$ most recent clicked article IDs prior to the impression cutoff.
* $\mathbf{e}_{a_i} \in \mathbb{R}^d$: Precomputed $L_2$-normalized embedding of historical article $a_i$.
* $w_i$: Historical click weighting coefficient (uniform $w_i = 1.0$ or exponential decay weight).

#### Significance & Mechanism
Synthesizes the user's multi-article reading trajectory into a compact semantic vector representing their current topical interest centroid, strictly eliminating temporal lookahead leakage.

---

### 2.3 Dense Cosine Similarity Scoring
$$\text{Score}_{\text{Dense}}(u, D) = \langle \mathbf{\hat{u}}, \mathbf{\hat{e}}_D \rangle = \sum_{k=1}^d \hat{u}_k \cdot \hat{e}_{D,k}$$

#### Significance & Mechanism
Measures continuous semantic alignment between user preference vector $\mathbf{\hat{u}}$ and candidate news article $\mathbf{\hat{e}}_D$. Exact nearest neighbors are retrieved via `faiss.IndexFlatIP` (exact) or `faiss.IndexHNSWFlat` (approximate graph search).

---

## 3. Feature Engineering: Exponential Temporal Recency
*Implemented in [`src/feature_store.py`](file:///c:/Users/kavan/OneDrive/Desktop/IIITH/sem3/IRE/Assignment1/IRE-Assignment-1/src/feature_store.py).*

### 3.1 Temporal Decay Recency Score
$$\text{Recency Score}(u) = \sum_{i \in H_u} 0.5^{\frac{\text{age}(i)}{h}}$$

#### Parameter Definitions
* $\text{age}(i)$: Age in hours of click $i$ measured relative to split cutoff timestamp $t_{\text{as\_of}}$:
  $$\text{age}(i) = \frac{t_{\text{as\_of}} - t_{\text{click}, i}}{3600}$$
* $h = 24.0\text{ hours}$: Half-life parameter governing the decay rate.

#### Significance & Mechanism
Quantifies user interaction freshness. A click from 24 hours ago receives a weight of $0.5$, while a click from 48 hours ago receives $0.25$. This differentiates actively engaged users from dormant accounts for downstream ranking.

---

## 4. Ranking & Accuracy Evaluation Metrics
*Implemented in [`src/metrics.py`](file:///c:/Users/kavan/OneDrive/Desktop/IIITH/sem3/IRE/Assignment1/IRE-Assignment-1/src/metrics.py) and [`src/eval_harness.py`](file:///c:/Users/kavan/OneDrive/Desktop/IIITH/sem3/IRE/Assignment1/IRE-Assignment-1/src/eval_harness.py).*

### 4.1 Area Under the ROC Curve (AUC - Wilcoxon–Mann–Whitney Statistic)
$$\text{AUC} = \frac{\sum_{i \in \text{Pos}} \text{rank}(i) - \frac{n_{\text{pos}}(n_{\text{pos}} + 1)}{2}}{n_{\text{pos}} \cdot n_{\text{neg}}}$$

#### Parameter Definitions
* $\text{rank}(i)$: 1-based rank position of the $i$-th positive (clicked) candidate article when the candidate list is sorted in **ascending** order of predicted scores (with average rank tie adjustments).
* $n_{\text{pos}}$: Total number of positive items in the impression ($y=1$).
* $n_{\text{neg}}$: Total number of negative items in the impression ($y=0$).

#### Significance & Mechanism
Evaluates pairwise discrimination quality. It equals the exact probability that a randomly chosen positive item receives a higher predicted score than a randomly chosen negative item:
$$P(s_{\text{pos}} > s_{\text{neg}})$$

---

### 4.2 Mean Reciprocal Rank (MRR)
$$\text{RR} = \begin{cases} \frac{1}{\text{rank}_{\text{first\_pos}}}, & \text{if } n_{\text{pos}} \ge 1 \\ 0.0, & \text{if } n_{\text{pos}} = 0 \end{cases}, \qquad \text{MRR} = \frac{1}{|U|} \sum_{u=1}^{|U|} \text{RR}_u$$

#### Parameter Definitions
* $\text{rank}_{\text{first\_pos}}$: 1-based position of the highest-ranked positive item in descending score order.
* $|U|$: Total number of evaluated impressions.

#### Significance & Mechanism
Evaluates the retrieval effort required before the user encounters their very first relevant recommendation.

---

### 4.3 Normalized Discounted Cumulative Gain (nDCG@K)
$$\text{DCG}@K = \sum_{i=1}^{K} \frac{y_i}{\log_2(i + 1)}, \qquad \text{IDCG}@K = \sum_{i=1}^{\min(K, n_{\text{pos}})} \frac{1}{\log_2(i + 1)}$$
$$\text{nDCG}@K = \frac{\text{DCG}@K}{\text{IDCG}@K}$$

#### Parameter Definitions
* $y_i \in \{0, 1\}$: Binary relevance label of the item ranked at position $i$.
* $\frac{1}{\log_2(i + 1)}$: Position discount factor dampening gains for lower ranks.
* $\text{IDCG}@K$: Ideal Discounted Cumulative Gain (maximum achievable DCG when all positive items occupy top ranks).

#### Significance & Mechanism
Measures top-heavy ranking performance. Positions near rank 1 provide significantly higher score rewards than lower positions, normalized to $[0, 1]$.

---

### 4.4 Candidate Retrieval Recall@K
$$\text{Recall}@K = \frac{|\text{Retrieved}_{@K} \cap \text{GroundTruth}|}{|\text{GroundTruth}|}$$

#### Parameter Definitions
* $\text{Retrieved}_{@K}$: Set of top-$K$ candidate articles retrieved from the full catalog ($K \in \{50, 100, 200\}$).
* $\text{GroundTruth}$: Set of true clicked articles in the impression.

#### Significance & Mechanism
Quantifies the candidate generation funnel capacity: what percentage of all relevant articles are retained for subsequent stage reranking.

---

## 5. Beyond-Accuracy Recommendation Metrics
*Implemented in [`src/metrics.py`](file:///c:/Users/kavan/OneDrive/Desktop/IIITH/sem3/IRE/Assignment1/IRE-Assignment-1/src/metrics.py).*

### 5.1 Intra-List Diversity (ILD@K)
$$\text{ILD}@K = \frac{2}{K(K - 1)} \sum_{1 \le i < j \le K} \left(1 - \langle \mathbf{\hat{e}}_i, \mathbf{\hat{e}}_j \rangle\right)$$

#### Parameter Definitions
* $\mathbf{\hat{e}}_i, \mathbf{\hat{e}}_j$: Unit $L_2$-normalized embedding vectors of items ranked at positions $i$ and $j$.
* $1 - \langle \mathbf{\hat{e}}_i, \mathbf{\hat{e}}_j \rangle$: Pairwise cosine distance.
* $\frac{2}{K(K - 1)}$: Normalization factor for all $\binom{K}{2}$ pairs.

#### Significance & Mechanism
Measures the semantic dispersion of items within a single user's recommendation slate. High diversity indicates varied topic coverage, preventing echo chambers and narrow filter bubbles.

---

### 5.2 Novelty@K (Self-Information Surprise)
$$\text{Novelty}@K = \frac{1}{K} \sum_{i=1}^{K} -\log_2 P(a_i)$$

#### Parameter Definitions
* $P(a_i)$: Empirical prior click probability (global popularity) of article $a_i$ across training history:
  $$P(a_i) = \frac{\text{Clicks}(a_i)}{\sum_{j \in C} \text{Clicks}(a_j)}$$
  (Clipped to $P_{\text{min}} = 10^{-6}$ for newly introduced/unseen articles).

#### Significance & Mechanism
Quantifies recommendation serendipity and surprise in bits of information. Recommending viral head articles yields low novelty, while surfacing relevant niche tail articles yields high novelty.

---

### 5.3 Catalog Coverage@K
$$\text{Coverage}@K = \frac{\left|\bigcup_{u \in U} \text{TopK}_u\right|}{|C|}$$

#### Parameter Definitions
* $\text{TopK}_u$: Set of top-$K$ recommended articles presented to user $u$.
* $|C|$: Total number of unique articles in the entire indexed catalog.

#### Significance & Mechanism
Measures system-wide exploration and exposure fairness: the fraction of the total catalog that receives at least one recommendation across the user base.

---

## 6. Statistical Inference: Non-Parametric Bootstrap Confidence Intervals
*Implemented in [`src/metrics.py`](file:///c:/Users/kavan/OneDrive/Desktop/IIITH/sem3/IRE/Assignment1/IRE-Assignment-1/src/metrics.py).*

Given metric observations $S = [s_1, s_2, \dots, s_N]$ over $N$ impressions:

1. **Resampling**: Draw $B = 1000$ bootstrap datasets $S^{*b}$ of size $N$ with replacement from $S$.
2. **Replicate Means**: Compute the sample mean for each resample:
   $$\bar{\theta}^{*b} = \frac{1}{N} \sum_{j=1}^N s^{*b}_j, \quad \forall b \in \{1, 2, \dots, B\}$$
3. **Percentile Interval**: Determine the empirical $95\%$ Confidence Interval:
   $$\text{CI}_{95\%} = \left[ \text{Percentile}\left(\bar{\theta}^*, 2.5\%\right), \; \text{Percentile}\left(\bar{\theta}^*, 97.5\%\right) \right]$$

#### Significance & Mechanism
Provides distribution-free, statistically sound confidence bounds for ranking and retrieval benchmarks without relying on Gaussian normality assumptions.
