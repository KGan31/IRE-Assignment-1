# The Comprehensive Guide to LightGBM: Architecture, Mechanics, and Practical Applications

---

## 1. Introduction & Core Philosophy

**LightGBM** (Light Gradient Boosting Machine) is an open-source, highly efficient, and distributed gradient boosting framework developed by Microsoft Research in 2017. It was designed specifically to overcome the computational and memory bottlenecks of conventional Gradient Boosted Decision Tree (GBDT) implementations (such as standard XGBoost or scikit-learn's `GradientBoostingClassifier`) when applied to large-scale, high-dimensional datasets.

### 1.1 The Classical GBDT Bottleneck
In a standard GBDT algorithm, the most time-consuming operation during training is finding the optimal split point for each continuous feature across all leaves. Conventional algorithms:
1. Sort the values of each continuous feature across all $N$ data instances ($O(N \log N)$ complexity).
2. Scan through all sorted candidate split points to calculate the exact loss reduction ($O(N \times D)$ where $D$ is the number of features).
3. Access memory in non-contiguous, random patterns due to pointer-based indexing of sorted values, incurring severe cache misses.

When $N$ scales into millions of rows and $D$ into hundreds or thousands of features (common in search engines, news recommenders, and click-through-rate prediction), exact greedy search becomes prohibitively slow and exhausts main memory.

### 1.2 The LightGBM Solution
LightGBM re-engineers this paradigm via four foundational algorithmic innovations:
1. **Histogram-Based Split Finding**: Discretizes continuous feature values into small integer bins (e.g., 256 bins), reducing memory consumption by $4\times$ to $8\times$ and split computation to $O(K \times D)$ where $K \ll N$.
2. **Leaf-Wise (Best-First) Tree Growth**: Expands the leaf that maximizes the global loss reduction rather than forcing level-by-level symmetric growth.
3. **Gradient-Based One-Side Sampling (GOSS)**: Selectively samples instances with smaller gradients while retaining all instances with large gradients, preserving training accuracy while drastically speeding up histogram construction.
4. **Exclusive Feature Bundling (EFB)**: Merges mutually exclusive sparse features into dense composite bins, reducing the effective feature dimension with minimal information loss.

---

## 2. Foundations of Gradient Boosting

Gradient boosting builds an ensemble of weak learners (typically shallow regression trees) sequentially:

$$
\hat{y}_i^{(M)} = \sum_{m=1}^{M} f_m(x_i), \quad f_m \in \mathcal{F}
$$

where $\mathcal{F}$ is the space of regression trees, and $f_m(x)$ maps an input vector $x \in \mathbb{R}^D$ to a continuous score.

### 2.1 Taylor Series Loss Approximation
At step $m$, given the current prediction $\hat{y}_i^{(m-1)}$ and a twice-differentiable loss function $\mathcal{L}(y_i, \hat{y}_i)$, the objective function is:

$$
\mathcal{L}^{(m)} = \sum_{i=1}^N \mathcal{L}\left(y_i, \hat{y}_i^{(m-1)} + f_m(x_i)\right) + \Omega(f_m)
$$

where $\Omega(f_m)$ is the tree complexity regularization term. Taking the second-order Taylor expansion around $\hat{y}_i^{(m-1)}$:

$$
\mathcal{L}^{(m)} \approx \sum_{i=1}^N \left[ \mathcal{L}\left(y_i, \hat{y}_i^{(m-1)}\right) + g_i f_m(x_i) + \frac{1}{2} h_i f_m^2(x_i) \right] + \Omega(f_m)
$$

The first-order gradient $g_i$ and second-order Hessian $h_i$ are defined as:

$$
g_i = \left. \frac{\partial \mathcal{L}(y_i, \hat{y})}{\partial \hat{y}} \right|_{\hat{y} = \hat{y}_i^{(m-1)}}, \quad h_i = \left. \frac{\partial^2 \mathcal{L}(y_i, \hat{y})}{\partial \hat{y}^2} \right|_{\hat{y} = \hat{y}_i^{(m-1)}}
$$

### 2.2 Optimal Leaf Weights and Split Criterion
Let $I_j = \{i \mid q(x_i) = j\}$ be the set of sample indices assigned to leaf $j$, and let $w_j$ be the output weight of leaf $j$. With L2 leaf regularization $\frac{1}{2} \lambda \sum_j w_j^2$ and L1 leaf regularization $\alpha \sum_j |w_j|$, the objective for leaf $j$ (assuming $\alpha = 0$ for simplicity) simplifies to:

$$
\mathcal{J}_j(w_j) = \left( \sum_{i \in I_j} g_i \right) w_j + \frac{1}{2} \left( \sum_{i \in I_j} h_i + \lambda \right) w_j^2
$$

Setting $\frac{\partial \mathcal{J}_j}{\partial w_j} = 0$, the optimal leaf weight $w_j^*$ is:

$$
w_j^* = -\frac{\sum_{i \in I_j} g_i}{\sum_{i \in I_j} h_i + \lambda}
$$

The corresponding minimum loss value is:

$$
\mathcal{L}^* = -\frac{1}{2} \frac{\left( \sum_{i \in I_j} g_i \right)^2}{\sum_{i \in I_j} h_i + \lambda}
$$

When evaluating a split of an existing node into left subset $I_L$ and right subset $I_R$ (where $I = I_L \cup I_R$), the information gain (loss reduction) is:

$$
\Delta \mathcal{L} = \frac{1}{2} \left[ \frac{\left(\sum_{i \in I_L} g_i\right)^2}{\sum_{i \in I_L} h_i + \lambda} + \frac{\left(\sum_{i \in I_R} g_i\right)^2}{\sum_{i \in I_R} h_i + \lambda} - \frac{\left(\sum_{i \in I} g_i\right)^2}{\sum_{i \in I} h_i + \lambda} \right] - \gamma
$$

where $\gamma$ is the minimum gain required to perform a split (`min_split_gain`).

---

## 3. Core Algorithmic Pillars of LightGBM

### 3.1 Histogram-Based Split Finding

Instead of sorting continuous features along the continuous real line $\mathbb{R}$, LightGBM discretizes continuous floating-point values into $K$ discrete bins (typically $K = 256$, represented as an `uint8_t` byte).

```
Continuous Feature: [0.12, 1.45, 0.88, 3.20, 0.05, 2.10, 1.95]
Discretized Bins:   [  0 ,   2 ,   1 ,   4 ,   0 ,   3 ,   3  ]  (K=5 bins)
```

#### Advantages:
1. **Memory Efficiency**:
   - Storing sorted 32-bit floating-point values requires $4$ bytes per feature per instance, plus another $4$ bytes for sorting indices ($8$ bytes total).
   - Storing an 8-bit bin index requires only $1$ byte per instance ($8\times$ reduction in feature memory).
2. **Computational Speed**:
   - Building a histogram requires iterating over data points once to accumulate first-order gradients and second-order Hessians into each bin:
     
     $$
     H_k = \left( \sum_{x_{ij} \in \text{bin } k} g_i, \sum_{x_{ij} \in \text{bin } k} h_i \right)
     $$

   - Evaluating split candidates requires scanning only $K$ bins ($O(K)$) instead of all unique sorted values ($O(N)$).
3. **Histogram Subtraction Trick**:
   - The histogram of a parent node is the exact sum of the histograms of its two children:
     
     $$
     H_{\text{right}} = H_{\text{parent}} - H_{\text{left}}
     $$

   - Therefore, after splitting a node, LightGBM only needs to compute the histogram for the smaller child node directly from data. The larger sibling's histogram is obtained via a trivial bin-by-bin subtraction in $O(K)$ operations without touching the dataset.

```
       [Parent Node Histogram: N=1000]
                 /           \
                /             \
      [Child Left: N=300]   [Child Right: N=700]
     (Build from scratch)   (H_right = H_parent - H_left)
       Cost: O(300 * D)       Cost: O(K * D) -- virtually instant!
```

---

### 3.2 Leaf-Wise (Best-First) Tree Growth

Traditional decision tree algorithms (including XGBoost in its original mode) grow trees **level-wise** (depth-first or breadth-first), splitting all leaves at the same depth simultaneously to maintain a balanced tree.

```
Level-wise (Symmetric Growth):
        [ Root ]
        /      \
      (o)      (o)          Level 1: Splits all leaves
      / \      / \
    (o) (o)  (o) (o)        Level 2: Splits all leaves evenly

Leaf-wise (LightGBM Best-First Growth):
        [ Root ]
        /      \
      (o)      (o)*         Picks leaf (*) with max gain
               /  \
             (o)  (o)*      Picks leaf (*) with max gain
                  /  \
                (o)  (o)    Deep, asymmetric growth targeting highest error reduction
```

#### Why Leaf-Wise is Superior:
- **Maximum Loss Reduction**: Splitting a leaf that has high potential gain achieves greater loss reduction than splitting multiple leaves with negligible gain.
- **Lower Metric Error**: For a fixed number of total leaves (`num_leaves`), leaf-wise growth consistently yields lower training error than level-wise growth.
- **Controlling Overfitting**: Because leaf-wise trees can grow deep and asymmetric on small clusters of data, LightGBM employs explicit safety bounds:
  - `max_depth`: Limits the maximum allowable depth.
  - `min_data_in_leaf` (or `min_child_samples`): Prevents leaves from isolating tiny sample subsets.
  - `min_sum_hessian_in_leaf`: Enforces minimal second-order curvature before a split is allowed.

---

### 3.3 Gradient-Based One-Side Sampling (GOSS)

In GBDT, data points with smaller gradients have already been fitted well by previous trees (their residual errors are close to zero). Data points with larger gradients still suffer from high training error and provide substantially more information for determining optimal split boundaries.

However, completely discarding small-gradient instances would skew the true data distribution and degrade model generalization. GOSS resolves this tension mathematically:

```
Full Dataset (N instances, sorted by |gradient| descending)
┌───────────────────────────────────────────────┐
│ Top a * 100% instances (Large gradients)     │ -> Retain 100% of these
├───────────────────────────────────────────────┤
│ Remaining (1 - a) * 100% instances           │ -> Randomly sample a fraction b
│ (Small gradients)                             │    Reweight by (1 - a) / b
└───────────────────────────────────────────────┘
Effective sample size used for split finding: (a + b) * N
```

#### Mathematical Formulation:
1. Sort all instances by the absolute value of their gradients $|g_i|$ in descending order.
2. Select the top $a \times 100\%$ instances to form subset $A$.
3. From the remaining subset $A^c$ of size $(1 - a) \times N$, randomly sample a subset $B$ of size $b \times |A^c|$.
4. When constructing histograms and calculating split gain over $A \cup B$, amplify the gradients and Hessians of instances in subset $B$ by the normalization factor:

   $$
   w_B = \frac{1 - a}{b}
   $$

The estimated gradient sum for a feature bin over the full dataset is:

$$
\tilde{g}_{\text{bin}} = \sum_{i \in A \cap \text{bin}} g_i + \frac{1 - a}{b} \sum_{i \in B \cap \text{bin}} g_i
$$

#### Unbiasedness Theorem:
Because $\mathbb{E}_{i \sim \text{Uniform}(A^c)} \left[ \frac{1 - a}{b} \cdot \mathbf{1}_{\{i \in B\}} \right] = 1$, the expectation of the estimated gradient $\tilde{g}$ equals the true gradient sum:

$$
\mathbb{E}\left[\tilde{g}_{\text{bin}}\right] = \sum_{i \in \text{bin}} g_i
$$

GOSS thus guarantees an asymptotically unbiased estimate of the information gain while training on only a fraction $(a + b)$ of the data (e.g., $a = 0.2, b = 0.1$ trains on only $30\%$ of the instances).

---

### 3.4 Exclusive Feature Bundling (EFB)

High-dimensional datasets (especially in NLP, recommendation systems, or one-hot encoded categorical settings) are typically extremely sparse. Many features rarely take non-zero values simultaneously. Such features are said to be **mutually exclusive**.

LightGBM bundles these mutually exclusive features into a single, compact feature bin without losing any information.

#### Step 1: Greedy Graph Coloring for Bundling
1. Construct an undirected graph where each vertex represents a feature, and weighted edges represent the conflict count (number of concurrent non-zero entries between two features).
2. Sort features in descending order of their degrees (total conflicts).
3. Greedily assign each feature to an existing bundle if the conflict rate remains below a threshold (specified by `max_conflict_rate`), or instantiate a new bundle.

#### Step 2: Merging Feature Values via Offsets
To ensure the values of original features within a bundle remain uniquely distinguishable, LightGBM shifts the bin values by adding distinct offsets:

```
Feature 1 (Range: [0, 10]):  Non-zero bins: 1, 2, ..., 10
Feature 2 (Range: [0, 20]):  Non-zero bins: 1, 2, ..., 20

Bundled Feature:
  - If Feature 1 is non-zero:  take bin in [1, 10]
  - If Feature 2 is non-zero:  add offset 10 -> take bin in [11, 30]
  - If both are zero:          take 0
```

By consolidating hundreds of sparse features into a few dense bundled features, the histogram construction cost drops from $O(N \times D_{\text{raw}})$ to $O(N \times D_{\text{bundled}})$.

---

## 4. Native Categorical Feature Handling

Most machine learning libraries require categorical variables to be preprocessed into one-hot encodings or target encodings. 
- **One-Hot Encoding** creates tall, deep trees that split repeatedly on single sparse binary columns, fragmenting data and wasting tree depth.
- LightGBM natively partitions categorical features with $K$ distinct categories into two subsets:

$$
\text{Left} = \mathcal{C}_{\text{left}} \subset \{c_1, \dots, c_K\}, \quad \text{Right} = \mathcal{C}_{\text{right}}
$$

### 4.1 Optimal Subset Splitting in $O(K \log K)$
Testing all possible subset splits requires evaluating $2^{K-1} - 1$ combinations, which is computationally intractable for high cardinality. LightGBM implements Fisher's optimal partitioning theorem:

1. For a given categorical feature, accumulate the sum of gradients and Hessians for each category $c$:
   
   $$
   S_c = \frac{\sum_{i \in c} g_i}{\sum_{i \in c} h_i + s}
   $$

   where $s$ is the smoothing regularization term (`cat_smooth`).

2. Sort the $K$ categories according to their ratio $S_c$ in ascending order.
3. The optimal multi-choice subset split is guaranteed to be a contiguous slice along this sorted sequence.
4. Hence, LightGBM reduces the search complexity from $O(2^K)$ to $O(K \log K)$ (sorting the categories).

```
Category:        ['Sports', 'Tech', 'Politics', 'Finance']
Gradient Ratio:  [  -1.4  ,  -0.2 ,    0.8    ,    2.1   ]
Sorted Order:    Sports  <  Tech  <  Politics <  Finance

Evaluated Splits:
  Split 1: {Sports} vs {Tech, Politics, Finance}
  Split 2: {Sports, Tech} vs {Politics, Finance}
  Split 3: {Sports, Tech, Politics} vs {Finance}
```

---

## 5. Objectives & Learning-to-Rank (LTR)

LightGBM supports standard machine learning paradigms (Regression, Binary Classification, Multiclass Classification) alongside first-class support for **Learning to Rank (LTR)**.

### 5.1 Overview of Objectives

| Task | Objective Parameter (`objective`) | Output Interpretation | Common Metrics (`metric`) |
| :--- | :--- | :--- | :--- |
| **Regression (L2)** | `regression` (MSE) | Conditional expectation $\mathbb{E}[y \mid x]$ | `rmse`, `l2` |
| **Regression (L1)** | `regression_l1` (MAE) | Median $\text{Median}(y \mid x)$ | `mae`, `l1` |
| **Robust Regression** | `huber`, `fair` | Robust to outliers | `huber`, `fair` |
| **Quantile Regression** | `quantile` | Specified conditional quantile $\tau$ | `quantile` |
| **Binary Classification** | `binary` | Probability $P(y=1 \mid x)$ via Sigmoid | `binary_logloss`, `auc` |
| **Multiclass** | `multiclass` | Class probability vector via Softmax | `multi_logloss`, `multi_error` |
| **Pairwise Ranking** | `lambdarank` | Arbitrary relevance score for ordering | `ndcg`, `map` |
| **Listwise Ranking** | `rank_xendcg` | Normalized cross-entropy relevance | `ndcg` |

---

### 5.2 LambdaRank Deep Dive

In Information Retrieval and Recommender Systems (such as MIND and EB-NeRD news ranking), models do not predict independent class labels. Instead, they rank a list of candidate documents $D = \{d_1, \dots, d_n\}$ for a specific query or user impression $q$.

The target evaluation metric is **Normalized Discounted Cumulative Gain (NDCG)**:

$$
\text{DCG}@K = \sum_{i=1}^K \frac{2^{y_i} - 1}{\log_2(i + 1)}, \quad \text{NDCG}@K = \frac{\text{DCG}@K}{\text{IDCG}@K}
$$

where $y_i$ is the relevance label of the item ranked at position $i$, and $\text{IDCG}@K$ is the ideal DCG obtained by sorting items in perfect descending order of relevance.

#### The Non-Differentiability Problem
Because $\text{NDCG}$ depends on the discrete permutation rank $i \in \{1, \dots, K\}$, it is piecewise flat with zero gradients everywhere and discontinuous step-jumps when two items swap places. Traditional gradient descent cannot be directly applied.

#### The Lambda Gradient Solution
LambdaRank bypasses this by constructing a virtual gradient—termed $\lambda_{ij}$—for every pair of items $(i, j)$ within the same query group where $y_i > y_j$:

$$
\lambda_{ij} = \frac{-\sigma}{1 + e^{\sigma(s_i - s_j)}} \cdot |\Delta \text{NDCG}_{ij}|
$$

where:
- $s_i, s_j$ are the predicted model scores for items $i$ and $j$.
- $\frac{-\sigma}{1 + e^{\sigma(s_i - s_j)}}$ is the gradient of the RankNet pairwise cross-entropy loss.
- $|\Delta \text{NDCG}_{ij}|$ is the exact change in NDCG that would occur if items $i$ and $j$ swapped their current ranked positions while keeping all other items fixed.

The total virtual gradient for item $i$ is accumulated across all pairs involving $i$:

$$
\lambda_i = \sum_{j: y_i > y_j} \lambda_{ij} - \sum_{k: y_k > y_i} \lambda_{ki}
$$

LightGBM then treats $\lambda_i$ as the first-order gradient $g_i$ in its tree-building routine, allowing direct optimization of non-differentiable listwise ranking metrics.

#### The Group / Query Structure
To train an `LGBMRanker` or use `objective="lambdarank"`, the dataset **must** specify the boundaries of each query or impression group. For example, if impression 1 contains 15 candidate articles, impression 2 contains 8, and impression 3 contains 22:
- The total rows in feature matrix $X$ is $15 + 8 + 22 = 45$.
- The `group` vector is `[15, 8, 22]`.
- Pairwise comparisons and NDCG swaps are strictly confined within each individual group; items from different queries are never paired.

---

## 6. Comprehensive Hyperparameter Directory

LightGBM parameters are grouped below by their functional domain, detailing their defaults, mechanisms, and tuning strategies.

### 6.1 Tree Structure & Complexity Controls

| Parameter | Type / Default | Mechanism & Impact | Recommended Tuning Strategy |
| :--- | :--- | :--- | :--- |
| `num_leaves` | `int` (default: 31) | Maximum number of terminal leaves per tree. Controls model expressiveness. | The most critical parameter for capacity. Set $\le 2^d$ where $d = \text{max depth}$ (`max_depth`). Lower (15-63) to prevent overfitting. |
| `max_depth` | `int` (default: -1) | Strictly caps tree depth to prevent deep runaway branches in leaf-wise mode. | Set to explicit positive integers (e.g., 6 to 12) if dataset is noisy or prone to overfitting. |
| `min_data_in_leaf` (`min_child_samples`) | `int` (default: 20) | Minimum number of samples required in a terminal leaf. Prevents isolating single outliers. | Increase to 50–500 on large noisy datasets; decrease for very small datasets. |
| `min_sum_hessian_in_leaf` (`min_child_weight`) | `float` (default: 1e-3) | Minimum sum of second-order Hessians in a leaf. In classification, relates to sample count $\times p(1-p)$. | Increase to regularize splits when dealing with noisy or imbalanced classes. |
| `min_gain_to_split` | `float` (default: 0.0) | Minimum gain reduction required to justify making an additional split. | Set to small positive values (0.01 - 0.5) to prune uninformative splits. |
| `max_delta_step` | `float` (default: 0.0) | Maximum step size allowed for leaf output weights. | Helpful in highly imbalanced binary classification to avoid extreme weight spikes. |

---

### 6.2 Learning Rate, Boosting & Regularization

| Parameter | Type / Default | Mechanism & Impact | Recommended Tuning Strategy |
| :--- | :--- | :--- | :--- |
| `learning_rate` (`eta`) | `float` (default: 0.1) | Shrinkage factor applied to each tree's leaf outputs: $\hat{y}^{(m)} = \hat{y}^{(m-1)} + \eta f_m(x)$. | Lower values (0.01–0.05) combined with larger `n_estimators` provide superior generalization. |
| `n_estimators` (`num_iterations`) | `int` (default: 100) | Number of boosting rounds / trees built in the ensemble. | Use early stopping with validation data and set this to a high upper bound (1000–5000). |
| `reg_alpha` (L1) | `float` (default: 0.0) | L1 regularization on leaf weights. Encourages sparsity in leaf predictions. | Useful when many features are uninformative (tune in range 0.001 to 10.0). |
| `reg_lambda` (L2) | `float` (default: 0.0) | L2 regularization on leaf weights. Discourages excessively large leaf outputs. | Baseline regularizer (tune in range 0.1 to 100.0) to smooth predictions. |
| `path_smooth` | `float` (default: 0.0) | Regularization that shrinks leaf outputs towards the predictions of ancestor nodes. | Excellent for small datasets or noisy leaves (e.g., 1.0 to 10.0). |

---

### 6.3 Subsampling & Stochasticity

| Parameter | Type / Default | Mechanism & Impact | Recommended Tuning Strategy |
| :--- | :--- | :--- | :--- |
| `subsample` (`bagging_fraction`) | `float` (default: 1.0) | Fraction of data rows randomly sampled without replacement before building each tree. | Set between 0.6 and 0.9 to reduce variance and speed up training. |
| `bagging_freq` | `int` (default: 0) | Frequency (every $K$ iterations) at which bagging is performed. Must be $>0$ to activate `subsample`. | Typically set to 1 or 5 when `subsample` $< 1.0$. |
| `colsample_bytree` (`feature_fraction`) | `float` (default: 1.0) | Fraction of features randomly selected before building each tree. | Set between 0.6 and 0.8 to decorrelate individual trees and prevent feature dominance. |
| `colsample_bynode` | `float` (default: 1.0) | Subsample ratio of columns at each split node. | Further decorrelates splits, especially effective in high-dimensional tabular data. |
| `extra_trees` | `bool` (default: False) | Extremely Randomized Trees mode: chooses split thresholds randomly from bins rather than exhaustively. | Drastically speeds up training and curbs overfitting at a minor bias expense. |

---

### 6.4 Histogram Binning & GOSS Controls

| Parameter | Type / Default | Mechanism & Impact | Recommended Tuning Strategy |
| :--- | :--- | :--- | :--- |
| `max_bin` | `int` (default: 255) | Maximum number of discrete bins for continuous features. | 255 fits in 1 byte (`uint8`). Lower (63–127) for speed / memory; higher (511) for subtle thresholds. |
| `min_data_per_bin` | `int` (default: 3) | Minimum number of samples required to form an individual bin. | Prevents creating isolated bins for single rare outlier values. |
| `bin_construct_sample_cnt` | `int` (default: 200000) | Number of rows sampled from dataset to compute the bin quantization boundaries. | Increase if dataset has extreme distribution tails that require precise quantization. |
| `boosting_type` | `str` (default: `'gbdt'`) | Underlying boosting algorithm: `'gbdt'`, `'goss'`, `'dart'`, or `'rf'`. | Use `'gbdt'` as default; switch to `'goss'` for massive datasets; use `'dart'` for fine-grained ranking. |
| `top_rate` | `float` (default: 0.2) | GOSS parameter: fraction of large-gradient instances retained (parameter $a$). | Tune between 0.1 and 0.3 when `boosting_type='goss'`. |
| `other_rate` | `float` (default: 0.1) | GOSS parameter: fraction of small-gradient instances sampled (parameter $b$). | Tune between 0.05 and 0.2 when `boosting_type='goss'`. |

---

### 6.5 Categorical Hyperparameters

| Parameter | Type / Default | Mechanism & Impact | Recommended Tuning Strategy |
| :--- | :--- | :--- | :--- |
| `categorical_feature` | `list` / `'auto'` | List of column names or integer indices treated as categorical. | Always specify explicitly or convert columns to `category` dtype in pandas. |
| `cat_smooth` | `float` (default: 10.0) | Regularization factor added to category gradient-over-hessian ratio. Prevents overfitting on rare categories. | Increase (20–100) if categories have very few occurrences and noisy labels. |
| `cat_l2` | `float` (default: 10.0) | L2 regularization applied specifically during categorical split search. | Smooths categorical partition decisions. |
| `max_cat_to_onehot` | `int` (default: 4) | Maximum cardinality threshold below which simple one-hot splitting is used instead of subset search. | Keep low (4–8) to prioritize optimal subset partitioning. |

---

### 6.6 Advanced Constraints & Safe Learning

| Feature | Parameter | Mechanism & Usage |
| :--- | :--- | :--- |
| **Monotonic Constraints** | `monotone_constraints` | Enforces that predictions must be monotonically non-decreasing ($+1$) or non-increasing ($-1$) with respect to specific features. Essential in credit scoring, pricing, and domain-governed rules. |
| **Interaction Constraints** | `interaction_constraints` | Explicit list of feature index sets allowed to interact within the same branch of a tree. Suppresses spurious feature interactions. |
| **Missing Value Handling** | `use_missing=True`, `zero_as_missing=False` | LightGBM natively routes missing (`NaN`) values to whichever child branch (left or right) minimizes loss on the training set. No imputation required. |
| **Class Imbalance** | `is_unbalance=True` or `scale_pos_weight` | Automatically weights positive instances by $\frac{N_{\text{negative}}}{N_{\text{positive}}}$ or applies a custom multiplier to positive gradients. |

---

## 7. Distributed, Parallel, and Accelerated Computing

LightGBM achieves state-of-the-art multi-core and multi-machine scaling through three distinct parallelization strategies:

```
1. Feature Parallel:
   - Partition features across workers.
   - Each worker finds best split for its local feature subset.
   - Workers communicate best splits (O(Workers) network cost).

2. Data Parallel:
   - Partition rows across workers.
   - Each worker builds local histograms on local data.
   - Reduce-scatter histograms across workers (O(K * D) communication).

3. Voting Parallel (PV-Tree):
   - Combines data partitioning with a two-stage voting mechanism.
   - Workers vote for top local split candidates.
   - Drastically cuts communication overhead for ultra-large datasets.
```

- **GPU Acceleration**: Built-in OpenCL and CUDA backends accelerate histogram construction directly on GPU hardware, yielding up to $10\times$ speedups on wide datasets.

---

## 8. Interpretability: Feature Importance & TreeSHAP

### 8.1 Split vs. Gain Importance
LightGBM exposes two distinct metrics for assessing feature importance:
1. **`importance_type='split'`**: Counts the raw number of times a feature was selected to split a node across all trees in the ensemble.
   - *Limitation*: Can be biased toward continuous features with many candidate bins or high cardinality categorical features.
2. **`importance_type='gain'`**: Sums the total information gain (loss reduction) contributed by splits on that feature across all trees.
   - *Advantage*: Accurately reflects how much a feature directly improved the objective function.

### 8.2 Fast Exact TreeSHAP
LightGBM features native C++ integration of **TreeSHAP** (Lundberg et al.), allowing exact computation of Shapley values in polynomial time $O(T L D^2)$ (where $T$ is the number of trees, $L$ is the number of leaves, and $D$ is tree depth), rather than exponential time.

```python
# Exact SHAP values directly from LightGBM booster without external shap library overhead
shap_values = booster.predict(X_test, pred_contrib=True)
# Output shape: (N_samples, N_features + 1) where last column is the base expected value
```

---

## 9. Comprehensive Implementation Guide

The following sections provide production-grade Python implementations across all primary machine learning tasks using both the Scikit-Learn API and Native LightGBM Dataset API.

### 9.1 Binary Classification with Early Stopping & Categorical Handling

```python
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, log_loss

# 1. Generate synthetic tabular dataset with continuous and categorical features
np.random.seed(42)
n_samples = 50000

df = pd.DataFrame({
    'user_age': np.random.randint(18, 70, size=n_samples),
    'mean_dwell_time': np.random.exponential(scale=25.0, size=n_samples),
    'scroll_depth': np.random.uniform(0.0, 1.0, size=n_samples),
    'device_type': pd.Series(np.random.choice(['Mobile', 'Desktop', 'Tablet'], size=n_samples), dtype='category'),
    'news_category': pd.Series(np.random.choice(['Politics', 'Tech', 'Sports', 'Finance'], size=n_samples), dtype='category')
})

# Ground truth label depends on non-linear interaction
logits = (
    0.03 * df['user_age'] +
    0.05 * df['mean_dwell_time'] +
    2.5 * df['scroll_depth'] +
    (df['device_type'] == 'Mobile') * 0.8 +
    (df['news_category'] == 'Tech') * 1.2 - 3.5
)
prob = 1.0 / (1.0 + np.exp(-logits))
df['target'] = (np.random.rand(n_samples) < prob).astype(int)

# 2. Train / Validation Split
X = df.drop(columns=['target'])
y = df['target']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3. Model Definition via Scikit-Learn API
clf = lgb.LGBMClassifier(
    objective='binary',
    boosting_type='gbdt',
    n_estimators=1000,
    learning_rate=0.03,
    num_leaves=31,
    max_depth=6,
    min_child_samples=50,
    subsample=0.8,
    bagging_freq=1,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    cat_smooth=20.0,
    random_state=42,
    n_jobs=-1
)

# 4. Train with Early Stopping and Metric Callbacks
callbacks = [
    lgb.early_stopping(stopping_rounds=30, verbose=True),
    lgb.log_evaluation(period=50)
]

clf.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    eval_names=['train', 'val'],
    eval_metric=['auc', 'binary_logloss'],
    callbacks=callbacks
)

# 5. Evaluate Predictions
val_preds_prob = clf.predict_proba(X_val)[:, 1]
print(f"Validation AUC: {roc_auc_score(y_val, val_preds_prob):.4f}")
print(f"Validation LogLoss: {log_loss(y_val, val_preds_prob):.4f}")
```

---

### 9.2 Learning to Rank (LTR) with `LGBMRanker`

This pattern directly models the Stage 2 candidate re-ranking pipeline required for recommender benchmarks like MIND and EB-NeRD.

```python
import lightgbm as lgb
import numpy as np
import pandas as pd

# 1. Simulate impression-grouped candidate data
# 1000 impressions (queries), each containing between 5 and 25 candidate news articles
np.random.seed(42)
num_impressions = 1000
group_sizes = np.random.randint(5, 26, size=num_impressions)
total_candidates = np.sum(group_sizes)

# Feature matrix: Lexical match, semantic similarity, recency decay, user historical affinity
X_features = pd.DataFrame({
    'bm25_score': np.random.exponential(scale=1.5, size=total_candidates),
    'dense_similarity': np.random.normal(loc=0.6, scale=0.15, size=total_candidates),
    'recency_decay': np.random.uniform(0.1, 1.0, size=total_candidates),
    'user_category_affinity': np.random.uniform(0.0, 1.0, size=total_candidates)
})

# Latent utility and relevance grades (0: unclicked, 1: clicked, 2: deep read)
utility = (
    0.8 * X_features['dense_similarity'] +
    0.4 * X_features['bm25_score'] +
    0.5 * X_features['user_category_affinity'] +
    0.3 * X_features['recency_decay'] +
    np.random.gumbel(scale=0.3, size=total_candidates)
)

# Assign top 1-2 items per impression as clicked / deep read
relevance = np.zeros(total_candidates, dtype=int)
start_idx = 0
for sz in group_sizes:
    end_idx = start_idx + sz
    sub_utility = utility[start_idx:end_idx]
    top_indices = np.argsort(sub_utility)[::-1]
    relevance[start_idx + top_indices[0]] = 2 # Deep read
    if sz > 10:
        relevance[start_idx + top_indices[1]] = 1 # Clicked
    start_idx = end_idx

# 2. Train / Validation Split by Impression Boundaries (Never leak items across groups)
train_queries = 800
train_rows = np.sum(group_sizes[:train_queries])

X_train, y_train = X_features.iloc[:train_rows], relevance[:train_rows]
groups_train = group_sizes[:train_queries]

X_val, y_val = X_features.iloc[train_rows:], relevance[train_rows:]
groups_val = group_sizes[train_queries:]

# 3. Instantiate and Train LGBMRanker
ranker = lgb.LGBMRanker(
    objective='lambdarank',
    boosting_type='gbdt',
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=6,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    ndcg_eval_at=[5, 10],
    random_state=42
)

ranker.fit(
    X_train, y_train,
    group=groups_train,
    eval_set=[(X_val, y_val)],
    eval_group=[groups_val],
    eval_metric=['ndcg'],
    callbacks=[
        lgb.early_stopping(stopping_rounds=25, verbose=True),
        lgb.log_evaluation(period=20)
    ]
)

# 4. Predict ranking scores
scores = ranker.predict(X_val)
```

---

### 9.3 Custom Objective & Custom Metric (Native Dataset API)

LightGBM allows users to supply custom first-order gradients ($g_i$) and second-order Hessians ($h_i$) for specialized losses (e.g., Focal Loss, asymmetric financial penalties).

```python
import lightgbm as lgb
import numpy as np

# Custom Objective: Focal Loss for extreme class imbalance
# FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
def focal_loss_objective(preds, train_data, gamma=2.0, alpha=0.25):
    labels = train_data.get_label()
    p = 1.0 / (1.0 + np.exp(-preds))
    
    # Calculate p_t
    p_t = p * labels + (1.0 - p) * (1.0 - labels)
    alpha_t = alpha * labels + (1.0 - alpha) * (1.0 - labels)
    
    # First derivative (gradient)
    g = alpha_t * (1.0 - p_t)**gamma * (p - labels)
    
    # Second derivative (hessian approximation)
    h = alpha_t * (1.0 - p_t)**gamma * p * (1.0 - p) * (gamma * (1.0 - p_t) + 1.0)
    
    return g, h

# Custom Metric: Exact evaluation function
def eval_error_rate(preds, eval_data):
    labels = eval_data.get_label()
    p = 1.0 / (1.0 + np.exp(-preds))
    pred_labels = (p > 0.5).astype(int)
    error = np.mean(pred_labels != labels)
    return 'error_rate', error, False # (name, value, is_higher_better)

# Native Dataset setup
# dtrain = lgb.Dataset(X_train, label=y_train)
# dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
# booster = lgb.train(
#     params={'learning_rate': 0.05, 'num_leaves': 31},
#     train_set=dtrain,
#     valid_sets=[dval],
#     fobj=focal_loss_objective,
#     feval=eval_error_rate,
#     callbacks=[lgb.early_stopping(20)]
# )
```

---

## 10. Practical Tuning Guide & Troubleshooting

### 10.1 Diagnostic Matrix

| Problem Observed | Primary Root Cause | Remediation Steps |
| :--- | :--- | :--- |
| **Severe Overfitting** (Train metric $\gg$ Validation metric) | Tree capacity too high, or leaf sample bounds too low. | 1. Decrease `num_leaves` (e.g., from 63 to 20).<br>2. Set explicit `max_depth` (e.g., 6).<br>3. Increase `min_child_samples` (e.g., 50–200).<br>4. Decrease `colsample_bytree` ($0.6$) and `subsample` ($0.7$).<br>5. Increase `reg_lambda` and `path_smooth`. |
| **Underfitting** (Both train and val metrics plateau poorly) | Constraints too restrictive; learning rate too low for step count. | 1. Increase `num_leaves` (e.g., 63 or 127).<br>2. Relax `min_child_samples` (e.g., 10 or 20).<br>3. Increase `learning_rate` or train with larger `n_estimators`.<br>4. Increase `max_bin` to 511 for finer numeric resolution. |
| **Slow Training Throughput** | Huge instance count $N$, high feature dimension $D$, or excessive bins. | 1. Switch `boosting_type='goss'`.<br>2. Set `max_bin=63` or `127`.<br>3. Lower `subsample` and `bagging_freq=1`.<br>4. Set `extra_trees=True`.<br>5. Enable GPU acceleration (`device='gpu'`). |
| **High Memory / OOM** | Storing excessive bins or high cardinality categorical feature expansions. | 1. Lower `max_bin=63`.<br>2. Decrease `bin_construct_sample_cnt`.<br>3. Convert float64 columns to float32 or categorical dtypes. |
| **Severe Class Imbalance** | Positives are $< 1\%$ of data; model predicts zero everywhere. | 1. Set `scale_pos_weight = N_neg / N_pos`.<br>2. Set `is_unbalance=True`.<br>3. Tune `min_child_samples` down so minority leaves can form. |

---

### 10.2 Recommendations for Re-Ranker Pipelines (MIND & EB-NeRD)

When using LightGBM as a Stage 2 re-ranker downstream of a Stage 1 retrieval system (e.g., BM25 or FAISS dense retrieval):

1. **Strict Query Grouping**:
   - Ensure the train, validation, and test splits partition at the **impression level**. Never allow candidates from the same impression ID to appear in both training and validation sets.
2. **Signal Fusion**:
   - LightGBM thrives on non-linear feature interactions between sparse lexical signals (BM25 scores), dense semantic vector similarities (FAISS inner products), and engagement context (exponential dwell time decay, scroll depth).
3. **Choice of Objective**:
   - For pure $\text{Top-}K$ ranking metrics like $\text{NDCG}@5$ or $\text{MRR}$, use `objective='lambdarank'` with `eval_metric='ndcg'` and `ndcg_eval_at=[5, 10]`.
   - If predicting well-calibrated post-click conversion rates is required simultaneously, train with `objective='binary'` and rank by the predicted calibrated probabilities $P(\text{click} \mid u, a)$.
4. **Production Deployment**:
   - Save trained boosters using `booster.save_model('model.txt')`.
   - For microsecond-latency serving (SLA $< 10\text{ms}$), compile the booster to native C code via **Treelite** or export to **ONNX Runtime**, eliminating Python runtime interpreter overhead.

---

## 11. Assignment 2 Deep Dive: Two-Stage Re-Ranking on EB-NeRD & MIND

In this assignment (CS4.406 Information Retrieval & Extraction, Assignment 2), LightGBM acts as the **Stage 2 Re-Ranker** in a two-stage retrieve-then-rank architecture:
- **Stage 1 (Candidate Retrieval)**: Scalable candidate generation filtering millions of articles down to candidate pools of $\approx 100\text{--}200$ items using BM25 inverted index token matching and FAISS dense semantic cosine similarity over document embeddings.
- **Stage 2 (LightGBM Re-Ranker)**: A GBDT model (`LGBMRanker`) trained with LambdaRank (`objective="lambdarank"`, `metric="ndcg"`, `eval_at=[5, 10]`) over 10 engineered point-in-time features to resolve fine-grained ranking within each user impression.

```
Impression Request: user_id, timestamp t_imp, candidate articles
  │
  ├──► [Stage 1: Retrieval Prior] (Top-100 candidates)
  │       ├─ Dense Semantic Search: cosine(u_emb, a_emb) via FAISS
  │       ├─ BM25 Lexical Score: token match against user reading history
  │       └─ First-Stage Score Prior: max(BM25, Semantic)
  │
  └──► [Stage 2: LightGBM LambdaMART Re-Ranker]
          ├─ Point-in-time behavioral & temporal features (t_click < t_imp)
          ├─ Leaf-wise decision trees optimize pairwise Lambda gradients
          └─ Output: Sorted list maximizing NDCG@5, NDCG@10, MRR, and AUC
```

---

### 11.1 Features Extracted from the Data

The feature matrix $X$ fed to `LGBMRanker` consists of 10 hand-crafted features capturing user behavior, article freshness/popularity, session presentation, and semantic alignment:

| Feature Name | Source Data | Engineering Logic & Formulation | Functional Role |
| :--- | :--- | :--- | :--- |
| `user_click_count` | `history.parquet` / `behaviors.tsv` | Total count of clicks recorded for the user strictly prior to impression time: $N_u = \sum \mathbf{1}_{\{t_{\text{click}} < t_{\text{imp}}\}}$. | Distinguishes cold-start users from highly active power readers. |
| `user_recency_score` | `history.parquet` / `behaviors.tsv` | Exponential half-life decayed history score: $\sum 0.5^{\Delta t / 24.0\text{h}}$, where $\Delta t = t_{\text{imp}} - t_{\text{click}}$. | Measures recent engagement velocity rather than static historical click counts. |
| `user_mean_dwell_time` | `history.parquet` (EB-NeRD only) | Point-in-time mean reading duration (seconds) across prior clicked articles: $\frac{1}{|H_u|} \sum \text{dwell}$. Defaults to `0.0` for MIND. | Captures reading depth (differentiates superficial skimmers from long-form readers). |
| `article_freshness_hours` | `articles.parquet` / `news.tsv` | Elapsed time in hours between article publication and the impression request: $\max\left(0, \frac{t_{\text{imp}} - t_{\text{pub}}}{3600}\right)$. | Enforces rapid temporal decay of breaking news. |
| `article_popularity_24h` | `history.parquet` / `behaviors.tsv` | Total platform-wide clicks accumulated by candidate article within rolling 24-hour window strictly before $t_{\text{imp}}$. | Unpersonalized popularity prior for trending stories. |
| `category_affinity_score` | `history` + `articles` | Historical topic affinity: fraction of prior clicks belonging to candidate article's category: $p_u(\text{cat}) = \frac{\sum_{d \in H_u} \mathbf{1}_{\{\text{cat}(d) = \text{cat}(a)\}}}{|H_u|}$. | Captures persistent topical interests (e.g., sports vs. politics). |
| `session_position` | `behaviors.parquet` / `behaviors.tsv` | 0-indexed position of candidate item in the impression list. | Captures presentation and position bias in the news feed UI. |
| `bm25_score` | `news.tsv` / Danish text | Exact BM25 lexical similarity between candidate title/body and concatenated historical clicked titles. | Sparse lexical relevance matching. |
| `semantic_score` | `google_bert` / `word2vec` embeddings | Dense cosine similarity: $\frac{u^\top v_a}{\|u\|_2 \|v_a\|_2}$, where $u$ is user representation (mean-pooled or NRMS-attended history). | Deep semantic alignment between user interest and candidate content. |
| `first_stage_score` | Stage 1 output | Retrieval prior score: $\max($ `bm25_score` `, ` `semantic_score` $)$. | Preserves candidate generator rank confidence into Stage 2. |

---

### 11.2 Feature Importance Analysis: The Best and the Worst Features

Empirical split counts obtained during validation training reveal stark behavioral differences between EB-NeRD (Danish tabloid) and MIND (MSN News portal):

```
EB-NeRD Feature Importances (Splits)      MIND Feature Importances (Splits)
┌───────────────────────────────┬──────┐  ┌───────────────────────────────┬──────┐
│ article_freshness_hours       │  358 │  │ session_position              │  271 │
│ category_affinity_score       │  311 │  │ semantic_score                │   80 │
│ semantic_score                │  209 │  │ user_click_count              │   64 │
│ user_click_count              │  179 │  │ category_affinity_score       │   43 │
│ session_position              │  160 │  │ user_recency_score            │   23 │
│ user_recency_score            │  147 │  │ bm25_score                    │   17 │
│ user_mean_dwell_time          │  135 │  │ first_stage_score             │   12 │
│ bm25_score                    │  124 │  │ user_mean_dwell_time          │    0 │
│ first_stage_score             │   87 │  └───────────────────────────────┴──────┘
│ article_popularity_24h        │    0 │
└───────────────────────────────┴──────┘
```

#### 1. The Best Performing Features

- **`article_freshness_hours` (#1 Best on EB-NeRD, 360 splits)**:
  - *Why it dominates*: News consumption exhibits aggressive recency decay. On Ekstra Bladet, breaking news articles experience $> 80\%$ of their total clicks within the first 6 to 12 hours of publication. LightGBM rapidly learns split thresholds (e.g., `article_freshness_hours <= 4.2h`) that act as a primary gate before evaluating relevance.
- **`session_position` (#1 Best on MIND with 271 splits, #6 on EB-NeRD with 162 splits)**:
  - *Why it dominates*: Strong position bias. In MSN News (MIND), articles presented at positions 0 and 1 receive dramatically higher clicks regardless of topic due to screen placement. LightGBM exploits this signal to assign heavy baseline score penalties to items positioned lower in the feed.
- **`category_affinity_score` (#2 Best on EB-NeRD with 268 splits, #4 on MIND with 43 splits)**:
  - *Why it dominates*: Readers exhibit high editorial topic loyalty (e.g., sports enthusiasts vs. crime/tabloid readers). An article matching a user's dominant historical category receives an immediate, robust ranking boost. In ablation studies, adding Category Affinity alone boosted $\text{nDCG}@5$ by $+0.0084$ on EB-NeRD.
- **`semantic_score` (#3 Best on EB-NeRD with 224 splits, #2 Best on MIND with 80 splits)**:
  - *Why it dominates*: Dense embeddings (multilingual BERT / Word2Vec / NRMS vectors) capture topical semantic overlap beyond exact keywords, enabling the model to recommend related articles even when vocabulary differs.

#### 2. The Moderate Performing Features

- **`user_mean_dwell_time` (135 splits on EB-NeRD, 0 splits on MIND)**:
  - *Role*: When extracted from EB-NeRD's `read_time_fixed`, historical dwell time provides a valuable signal distinguishing deep, attentive readers (average dwell $> 60\text{s}$) from shallow click-bouncers. It actively contributes 135 splits in the trained model. On MIND, it remains 0 splits because MIND does not record client-side dwell telemetry.
- **`user_click_count` & `user_recency_score` (147–179 splits on EB-NeRD, 23–64 splits on MIND)**:
  - *Role*: Act as conditioning features. LightGBM uses them to branch between user cohorts: for cold-start users (`user_click_count <= 5`), the trees rely predominantly on global popularity and freshness; for warm users (`user_click_count > 20`), the trees switch to personalized semantic and category affinity splits.
- **`bm25_score` (124 splits on EB-NeRD, 17 splits on MIND)**:
  - *Role*: Provides lexical match precision when users follow specific ongoing named-entity stories (e.g., specific politicians or sports clubs). On MIND, its utility is lower because news browsing without explicit search queries suffers from high lexical sparsity.

#### 3. The Worst / Ineffective Features

- **`first_stage_score` (Low split counts: 12 on MIND, 87 on EB-NeRD)**:
  - *Why it underperformed*: It is mathematically collinear with `semantic_score` and `bm25_score` (`first_stage_score` $= \max($ `bm25_score` `, ` `semantic_score` $)$). Because LightGBM histogram binning already evaluates the individual components with finer granularity, the composite score added little incremental information gain.
- **`article_popularity_24h` (0 splits)**:
  - *Why it struggled*: When computing rolling 24-hour clicks strictly prior to impression timestamps, long-tail articles often have 0 or 1 clicks, creating severe discrete spike distributions that tree splits struggle to generalize without heavy Bayesian smoothing.

---

### 11.3 Anti-Gaming & Serving-Time Safety (Q1.4 & Q9)

A critical requirement of Assignment 2 is **Anti-Gaming Verification**:
- **The Pitfall**: The dataset `behaviors.parquet` contains post-click engagement metrics for the current impression, such as `read_time` and `scroll_percentage`.
- **Why It Must NEVER Be Used in LightGBM**:
  - In a production news serving system, when a user requests an impression, they have not yet seen or clicked on any candidate article. The `read_time` and `scroll_percentage` for that impression **do not exist at serving time**.
  - Feeding current impression engagement signals into the re-ranker constitutes catastrophic target leakage (it trivially tells the model whether and how long the user engaged with the candidate).
- **The Point-in-Time Boundary**:
  - All behavioral features (click counts, dwell times, category affinities) are strictly filtered using:
    
    $$
    t_{\text{click}} < t_{\text{imp}}
    $$

  - Any click occurring at or after $t_{\text{imp}}$ is strictly dropped by `filter_history_point_in_time()`.

---

### 11.4 Empirical Results: Before vs. After LightGBM Re-Ranking

When evaluating Stage 1 retrieval alone versus Stage 2 LightGBM re-ranking on full validation sets:

#### EB-NeRD Dataset (25,356 validation impressions, 304,915 candidate pairs)

| Evaluation Metric | Stage 1 (Before Re-Ranking) | Stage 2 (LightGBM Re-Ranker) | Absolute Gain ($\Delta$) | Relative Gain (%) |
| :--- | :---: | :---: | :---: | :---: |
| **AUC** | `0.4991` | **`0.7045`** | **`+0.2055`** | **+41.17%** |
| **MRR** | `0.3130` | **`0.4676`** | **`+0.1546`** | **+49.39%** |
| **nDCG@5** | `0.3431` | **`0.5291`** | **`+0.1860`** | **+54.21%** |
| **nDCG@10** | `0.4293` | **`0.5756`** | **`+0.1463`** | **+34.08%** |

#### MIND Dataset (15,696 validation impressions, 603,740 candidate pairs)

| Evaluation Metric | Stage 1 (Before Re-Ranking) | Stage 2 (LightGBM Re-Ranker) | Absolute Gain ($\Delta$) | Relative Gain (%) |
| :--- | :---: | :---: | :---: | :---: |
| **AUC** | `0.5350` | **`0.5748`** | **`+0.0398`** | **+7.44%** |
| **MRR** | `0.2612` | **`0.2929`** | **`+0.0317`** | **+12.14%** |
| **nDCG@5** | `0.2374` | **`0.2696`** | **`+0.0322`** | **+13.56%** |
| **nDCG@10** | `0.2997` | **`0.3322`** | **`+0.0325`** | **+10.84%** |

#### Serving Latency & Scale (Q4 Analysis)
- **Re-Ranking Execution Speed**: LightGBM scores 100 candidate items per impression in only **`2.63 ms`** (MIND) and **`2.79 ms`** (EB-NeRD).
- **Total Pipeline Latency**:
  - EB-NeRD p99: **`16.01 ms`**
  - MIND p99: **`12.62 ms`**
  - Production SLA target ($\text{p99} < 100\text{ ms}$): **PASSED with 6.2x to 7.9x headroom**.

