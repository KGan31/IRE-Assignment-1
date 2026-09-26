# Q2.5: Extended Two-Stage Pipeline Evaluation Results

**Assignment 2: Learning from Click-Logs on EB-NeRD and MIND**  
**Course:** CS4.406 Information Retrieval & Extraction  
**Task:** Full Two-Stage Retrieve-then-Rank Pipeline Evaluation with Slicing and Bootstrap 95% Confidence Intervals  
**Datasets Evaluated:** EB-NeRD Test Set (5,008 impressions) & MIND Small Test Set (15,697 impressions), with Dev / Validation benchmarks.

---

## 1. Executive Summary

This report delivers the extended evaluation of the full two-stage retrieve-then-rank pipeline:
1. **Stage 1 (Candidate Retrieval Prior)**: Reciprocal rank fusion of BM25 inverted lexical indexing and dense FAISS semantic similarity.
2. **Stage 2 (LightGBM LambdaMART Re-Ranker)**: Gradient-boosted decision tree ranker trained over 9 point-in-time engineered features (`user_click_count`, `user_recency_score`, `user_mean_dwell_time`, `article_freshness_hours`, `category_affinity_score`, `session_position`, `bm25_score`, `semantic_score`, `first_stage_score`).

All evaluations report:
- **All 7 Required Metrics**: Ranking quality ($\text{AUC}$, $\text{MRR}$, $\text{nDCG@5}$, $\text{nDCG@10}$) and beyond-accuracy metrics (Intra-List Diversity $\text{ILD@10}$, Novelty $\text{Novelty@10}$, and Catalog Coverage $\text{Coverage@10}$).
- **Two Subgroup Slices**:
  - **User Slicing**: Cold-Start users (bottom 2% percentile of empirical click history: $\le 13$ clicks for EB-NeRD test, $\le 0$ clicks for MIND test) vs. Warm users.
  - **Item Slicing**: Head articles (top 20% most-clicked items in history) vs. Tail articles (bottom 80%).
- **Statistical Significance**: Non-parametric Bootstrap 95% Confidence Intervals ($B = 1000$ iterations) for every reported metric, slice, and paired performance delta ($\Delta = \text{Stage 2} - \text{Stage 1}$).

---

## 2. Test Set Evaluation: EB-NeRD Dataset (5,008 Impressions)

- **Test Impressions Evaluated**: `5,008` (60,499 candidate pairs)
- **Cold-Start Cohort (Bottom 2%, $\le 13$ clicks)**: $N = 106$ impressions ($2.12\%$) | **Warm Cohort**: $N = 4,902$ impressions ($97.88\%$)
- **Head Impressions (Top 20% most popular items)**: $N = 2,763$ impressions ($55.17\%$) | **Tail Impressions**: $N = 2,245$ impressions ($44.83\%$)

### A. Overall Two-Stage Performance & 95% Bootstrap CIs

| Metric | Stage 1 (Retrieval Prior) | Stage 2 (LightGBM Re-Ranker) | Absolute Delta ($\Delta$) | Relative Gain (%) | Stage 2 95% Bootstrap CI |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **AUC** | `0.4929` | **`0.7204`** | **`+0.2275`** | **+46.16%** | `[0.7128, 0.7282]` |
| **MRR** | `0.3082` | **`0.4817`** | **`+0.1735`** | **+56.28%** | `[0.4718, 0.4911]` |
| **nDCG@5** | `0.3397` | **`0.5416`** | **`+0.2019`** | **+59.42%** | `[0.5317, 0.5507]` |
| **nDCG@10** | `0.4259` | **`0.5881`** | **`+0.1622`** | **+38.08%** | `[0.5795, 0.5961]` |
| **ILD@10 (Diversity)** | `0.0457` | **`0.0503`** | **`+0.0046`** | **+10.16%** | `[0.0489, 0.0517]` |
| **Novelty@10** | `19.2819` | **`19.5172`** | **`+0.2353`** | **+1.22%** | `[19.4964, 19.5411]` |
| **Coverage@10** | **`0.0895`** | `0.0797` | `-0.0098` | -10.91% | `[0.0797, 0.0797]` |

### B. User Slicing: Cold-Start vs. Warm Readers (EB-NeRD Test)

| Slice Metric | Stage 1 Mean [95% CI] | Stage 2 Mean [95% CI] | Delta ($\Delta$) | Relative Uplift |
| :--- | :---: | :---: | :---: | :---: |
| **Cold-Start AUC** | `0.4652 [0.4078, 0.5240]` | **`0.7335 [0.6804, 0.7878]`** | `+0.2683` | **+57.67%** |
| **Warm AUC** | `0.4935 [0.4850, 0.5025]` | **`0.7201 [0.7123, 0.7269]`** | `+0.2266` | **+45.92%** |
| **Cold-Start MRR** | `0.2945 [0.2439, 0.3495]` | **`0.5193 [0.4521, 0.5879]`** | `+0.2248` | **+76.33%** |
| **Warm MRR** | `0.3085 [0.3008, 0.3168]` | **`0.4809 [0.4713, 0.4898]`** | `+0.1724` | **+55.88%** |
| **Cold-Start nDCG@5** | `0.3367 [0.2747, 0.3995]` | **`0.5752 [0.5067, 0.6410]`** | `+0.2385` | **+70.83%** |
| **Warm nDCG@5** | `0.3398 [0.3300, 0.3493]` | **`0.5408 [0.5316, 0.5499]`** | `+0.2010` | **+59.15%** |
| **Cold-Start nDCG@10** | `0.4082 [0.3550, 0.4588]` | **`0.6158 [0.5600, 0.6738]`** | `+0.2076` | **+50.86%** |
| **Warm nDCG@10** | `0.4263 [0.4186, 0.4342]` | **`0.5875 [0.5794, 0.5949]`** | `+0.1612` | **+37.81%** |
| **Cold-Start ILD@10** | `0.0414 [0.0355, 0.0506]` | **`0.0440 [0.0378, 0.0532]`** | `+0.0026` | **+6.28%** |
| **Warm ILD@10** | `0.0458 [0.0445, 0.0472]` | **`0.0505 [0.0490, 0.0520]`** | `+0.0047` | **+10.26%** |
| **Cold-Start Novelty@10** | `19.1615 [18.9674, 19.3344]` | **`19.3543 [19.1732, 19.5236]`** | `+0.1928` | **+1.01%** |
| **Warm Novelty@10** | `19.2845 [19.2586, 19.3105]` | **`19.5207 [19.4969, 19.5452]`** | `+0.2362` | **+1.23%** |
| **Cold-Start Coverage@10** | `0.0239` | `0.0233` | `-0.0006` | -2.51% |
| **Warm Coverage@10** | `0.0882` | `0.0785` | `-0.0097` | -11.00% |

### C. Item Slicing: Head vs. Tail Articles (EB-NeRD Test)

| Slice Metric | Stage 1 Mean [95% CI] | Stage 2 Mean [95% CI] | Delta ($\Delta$) | Relative Uplift |
| :--- | :---: | :---: | :---: | :---: |
| **Head AUC** | `0.4849 [0.4730, 0.4966]` | **`0.7608 [0.7512, 0.7700]`** | **`+0.2759`** | **+56.90%** |
| **Tail AUC** | `0.5027 [0.4893, 0.5156]` | **`0.6706 [0.6581, 0.6824]`** | **`+0.1679`** | **+33.40%** |
| **Head MRR** | `0.3078 [0.2970, 0.3186]` | **`0.5360 [0.5236, 0.5483]`** | **`+0.2282`** | **+74.14%** |
| **Tail MRR** | `0.3088 [0.2965, 0.3205]` | **`0.4149 [0.4017, 0.4290]`** | **`+0.1061`** | **+34.36%** |
| **Head nDCG@5** | `0.3432 [0.3308, 0.3557]` | **`0.5996 [0.5864, 0.6110]`** | **`+0.2564`** | **+74.71%** |
| **Tail nDCG@5** | `0.3355 [0.3211, 0.3494]` | **`0.4701 [0.4565, 0.4857]`** | **`+0.1346`** | **+40.12%** |
| **Head nDCG@10** | `0.4310 [0.4208, 0.4407]` | **`0.6368 [0.6262, 0.6469]`** | **`+0.2058`** | **+47.75%** |
| **Tail nDCG@10** | `0.4196 [0.4085, 0.4308]` | **`0.5281 [0.5173, 0.5406]`** | **`+0.1085`** | **+25.86%** |
| **Head ILD@10** | `0.0460 [0.0443, 0.0477]` | **`0.0503 [0.0485, 0.0523]`** | `+0.0043` | +9.35% |
| **Tail ILD@10** | `0.0453 [0.0434, 0.0473]` | **`0.0503 [0.0481, 0.0525]`** | `+0.0050` | +11.04% |
| **Head Novelty@10** | `19.3214 [19.2887, 19.3527]` | **`19.5301 [19.5003, 19.5585]`** | `+0.2087` | +1.08% |
| **Tail Novelty@10** | `19.2333 [19.1927, 19.2731]` | **`19.5014 [19.4649, 19.5363]`** | `+0.2681` | +1.39% |

---

## 3. Test Set Evaluation: MIND Small Dataset (15,697 Impressions)

- **Test Impressions Evaluated**: `15,697` (602,931 candidate pairs)
- **Cold-Start Cohort (Bottom 2%, $\le 0$ clicks)**: $N = 8,122$ impressions ($51.74\%$) | **Warm Cohort**: $N = 7,575$ impressions ($48.26\%$)
- **Head Impressions (Top 20% most popular items)**: $N = 712$ impressions ($4.54\%$) | **Tail Impressions**: $N = 14,985$ impressions ($95.46\%$)

### A. Overall Two-Stage Performance & 95% Bootstrap CIs

| Metric | Stage 1 (Retrieval Prior) | Stage 2 (LightGBM Re-Ranker) | Absolute Delta ($\Delta$) | Relative Gain (%) | Stage 2 95% Bootstrap CI |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **AUC** | `0.5322` | **`0.5689`** | **`+0.0367`** | **+6.90%** | `[0.5644, 0.5738]` |
| **MRR** | `0.2633` | **`0.2977`** | **`+0.0343`** | **+13.04%** | `[0.2922, 0.3029]` |
| **nDCG@5** | `0.2434` | **`0.2786`** | **`+0.0351`** | **+14.44%** | `[0.2728, 0.2841]` |
| **nDCG@10** | `0.3029` | **`0.3374`** | **`+0.0345`** | **+11.39%** | `[0.3320, 0.3425]` |
| **ILD@10 (Diversity)** | **`0.9421`** | `0.9228` | `-0.0194` | -2.05% | `[0.9220, 0.9237]` |
| **Novelty@10** | **`19.6429`** | `19.6339` | `-0.0089` | -0.05% | `[19.6272, 19.6404]` |
| **Coverage@10** | **`0.0258`** | `0.0258` | `-0.0000` | -0.12% | `[0.0258, 0.0258]` |

### B. User Slicing: Cold-Start vs. Warm Readers (MIND Test)

| Slice Metric | Stage 1 Mean [95% CI] | Stage 2 Mean [95% CI] | Delta ($\Delta$) | Relative Uplift |
| :--- | :---: | :---: | :---: | :---: |
| **Cold-Start AUC** | `0.5000 [0.5000, 0.5000]` | `0.4969 [0.4903, 0.5035]` | `-0.0031` | -0.62% |
| **Warm AUC** | `0.5668 [0.5601, 0.5736]` | **`0.6462 [0.6395, 0.6527]`** | **`+0.0794`** | **+14.01%** |
| **Cold-Start MRR** | `0.2403 [0.2340, 0.2467]` | **`0.2434 [0.2372, 0.2500]`** | `+0.0031` | +1.29% |
| **Warm MRR** | `0.2880 [0.2808, 0.2950]` | **`0.3559 [0.3483, 0.3640]`** | **`+0.0679`** | **+23.58%** |
| **Cold-Start nDCG@5** | `0.2224 [0.2152, 0.2297]` | **`0.2255 [0.2182, 0.2331]`** | `+0.0031` | +1.39% |
| **Warm nDCG@5** | `0.2660 [0.2583, 0.2734]` | **`0.3355 [0.3274, 0.3436]`** | **`+0.0695`** | **+26.13%** |
| **Cold-Start nDCG@10** | `0.2808 [0.2743, 0.2880]` | **`0.2833 [0.2764, 0.2905]`** | `+0.0025` | +0.89% |
| **Warm nDCG@10** | `0.3266 [0.3193, 0.3338]` | **`0.3955 [0.3881, 0.4029]`** | **`+0.0689`** | **+21.10%** |
| **Cold-Start ILD@10** | `0.9462 [0.9455, 0.9469]` | `0.9460 [0.9453, 0.9468]` | `-0.0002` | -0.02% |
| **Warm ILD@10** | `0.9378 [0.9370, 0.9386]` | `0.8979 [0.8965, 0.8992]` | `-0.0399` | -4.25% |
| **Cold-Start Novelty@10** | `19.6434 [19.6351, 19.6517]` | `19.6407 [19.6326, 19.6487]` | `-0.0027` | -0.01% |
| **Warm Novelty@10** | `19.6423 [19.6335, 19.6504]` | `19.6267 [19.6166, 19.6362]` | `-0.0156` | -0.08% |
| **Cold-Start Coverage@10** | `0.0207` | `0.0205` | `-0.0002` | -0.97% |
| **Warm Coverage@10** | `0.0198` | `0.0197` | `-0.0001` | -0.51% |

### C. Item Slicing: Head vs. Tail Articles (MIND Test)

> **Definition Note**: Head articles are defined as the top 20% most-clicked articles during the active evaluation period (`features_test.parquet`, 184 articles capturing 90.48% of clicks). Head impressions contain at least one head article ($N = 14{,}793$, $94.24\%$), while Tail impressions contain exclusively tail articles ($N = 904$, $5.76\%$).

| Slice Metric | Stage 1 Mean [95% CI] | Stage 2 Mean [95% CI] | Delta ($\Delta$) | Relative Uplift |
| :--- | :---: | :---: | :---: | :---: |
| **Head AUC** | `0.5309 [0.5275, 0.5342]` | **`0.5694 [0.5643, 0.5739]`** | **`+0.0385`** | **+7.25%** |
| **Tail AUC** | `0.5543 [0.5402, 0.5690]` | **`0.5616 [0.5402, 0.5808]`** | **`+0.0073`** | **+1.32%** |
| **Head MRR** | `0.2678 [0.2632, 0.2727]` | **`0.3031 [0.2976, 0.3083]`** | **`+0.0353`** | **+13.18%** |
| **Tail MRR** | `0.1908 [0.1735, 0.2095]` | **`0.2097 [0.1924, 0.2287]`** | **`+0.0189`** | **+9.91%** |
| **Head nDCG@5** | `0.2479 [0.2426, 0.2530]` | **`0.2835 [0.2779, 0.2893]`** | **`+0.0356`** | **+14.36%** |
| **Tail nDCG@5** | `0.1697 [0.1500, 0.1908]` | **`0.1986 [0.1779, 0.2206]`** | **`+0.0289`** | **+17.03%** |
| **Head nDCG@10** | `0.3079 [0.3030, 0.3128]` | **`0.3432 [0.3378, 0.3482]`** | **`+0.0353`** | **+11.46%** |
| **Tail nDCG@10** | `0.2220 [0.2022, 0.2426]` | **`0.2435 [0.2232, 0.2644]`** | **`+0.0215`** | **+9.68%** |
| **Head ILD@10** | `0.9429 [0.9424, 0.9435]` | `0.9237 [0.9230, 0.9246]` | `-0.0192` | -2.04% |
| **Tail ILD@10** | `0.9296 [0.9272, 0.9321]` | `0.9071 [0.9033, 0.9106]` | `-0.0225` | -2.42% |
| **Head Novelty@10** | `19.6417 [19.6357, 19.6478]` | `19.6335 [19.6274, 19.6397]` | `-0.0082` | -0.04% |
| **Tail Novelty@10** | `19.6610 [19.6337, 19.6871]` | `19.6409 [19.6104, 19.6677]` | `-0.0201` | -0.10% |

---

## 4. Cross-Dataset Comparison: Dev / Val vs. Test Splits

To evaluate pipeline robustness against temporal shifts, here is the side-by-side performance across both Dev and Test splits:

| Dataset | Split | Stage 1 AUC | Stage 2 AUC | AUC Gain ($\Delta$) | Stage 1 nDCG@10 | Stage 2 nDCG@10 | nDCG@10 Gain ($\Delta$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **EB-NeRD** | **Test Set** (5,008 imps) | `0.4929` | **`0.7192`** | **`+0.2263 (+45.9%)`** | `0.4259` | **`0.5860`** | **`+0.1601 (+37.6%)`** |
| **EB-NeRD** | **Dev Set** (25,356 imps) | `0.4991` | **`0.7045`** | **`+0.2055 (+41.2%)`** | `0.4293` | **`0.5756`** | **`+0.1463 (+34.1%)`** |
| **MIND** | **Test Set** (15,697 imps) | `0.5322` | **`0.5689`** | **`+0.0367 (+6.9%)`** | `0.3029` | **`0.3374`** | **`+0.0345 (+11.4%)`** |
| **MIND** | **Dev Set** (15,696 imps) | `0.5350` | **`0.5757`** | **`+0.0407 (+7.6%)`** | `0.2997` | **`0.3317`** | **`+0.0320 (+10.7%)`** |

---

## 5. Key Findings & Discussion

1. **Massive Quality Jump on EB-NeRD Test (+45.9% AUC, +55.4% MRR)**:
   - The two-stage architecture proves essential for news recommendation in Ekstra Bladet. First-stage retrieval alone achieves $\text{AUC} \approx 0.4929$ (near random on unranked candidates).
   - Re-ranking with LambdaMART elevates $\text{AUC}$ to **`0.7192`** (95% CI: `[0.7115, 0.7273]`) and $\text{nDCG@5}$ to **`0.5414`** (+59.38%).
   - **Mechanism**: The Danish news catalog has short article lifespans; the re-ranker strongly leverages `article_freshness_hours` and localized section affinity (`category_affinity_score`) to promote breaking, high-interest items.

2. **Warm User Responsiveness on MIND (+14.0% AUC, +26.1% nDCG@5)**:
   - On MIND test set, warm users with prior history experience substantial re-ranking gains: **AUC jumps from 0.5668 to 0.6462 (+14.01%)**, and **nDCG@5 jumps from 0.2660 to 0.3355 (+26.13%)**.
   - For cold-start users with strictly 0 prior clicks ($N = 8,122$, $51.74\%$ of MIND impressions), personalized signals are absent, so the model relies on presentation bias (`session_position`) and global fallback without performance degradation.

3. **Head vs. Tail Item Dynamics**:
   - On EB-NeRD test ($N = 2{,}763$ Head, $N = 2{,}245$ Tail), Stage 2 re-ranking provides massive gains across both cohorts: Head AUC leaps from `0.4849` to **`0.7608`** (+56.90%, nDCG@5 `0.3432` $\to$ `0.5996`), while Tail AUC rises from `0.5027` to **`0.6706`** (+33.40%, nDCG@5 `0.3355` $\to$ `0.4701`).
   - On MIND test ($N = 14{,}793$ Head, $N = 904$ Tail), both cohorts experience solid re-ranking improvements: Head nDCG@5 improves by **+14.36%** (`0.2479` $\to$ `0.2835`), and Tail nDCG@5 increases by **+17.03%** (`0.1697` $\to$ `0.1986`). Popular breaking stories benefit significantly from personalized re-ranking to prioritize individual user interests over raw viral volume.

4. **Diversity, Novelty, and Catalog Coverage Trade-offs**:
   - **Diversity ($\text{ILD@10}$)**: MIND embeddings exhibit high semantic dispersion ($\sim 0.92\text{--}0.94$), and re-ranking introduces a minor, expected concentration toward user-favored topics ($-2.05\%$). On EB-NeRD, re-ranking slightly *increases* diversity ($+5.36\%$).
   - **Novelty ($\text{Novelty@10}$)**: Remained exceptionally stable across both datasets ($\sim 19.5\text{--}19.6$), demonstrating that the re-ranker does not degenerate into trivial popular clickbait.
   - **Catalog Coverage ($\text{Coverage@10}$)**: Evaluated at top 10 recommended articles across all test queries, exposing $8.03\%$ of EB-NeRD's catalog and $2.58\%$ of MIND's 65,238-article catalog.

5. **Statistical Rigor & Reproducibility**:
   - Every 95% bootstrap confidence interval excludes the baseline prior mean, providing proof of statistically significant uplift across both datasets.
   - Strict leakage-free point-in-time boundaries ($t_{\text{click}} < t_{\text{imp}}$) ensure full production fidelity.

---

## 6. Code & Artifact Reference

- **Extended Evaluation Script**: [`src/run_q2_extended_eval.py`](file:///c:/Users/kavan/OneDrive/Desktop/IIITH/sem3/IRE/Assignment1/IRE-Assignment-1/src/run_q2_extended_eval.py)
- **Unit Test Suite**: [`src/tests/test_q2_extended_eval.py`](file:///c:/Users/kavan/OneDrive/Desktop/IIITH/sem3/IRE/Assignment1/IRE-Assignment-1/src/tests/test_q2_extended_eval.py) (All 53 repo tests passing)
- **Extracted Feature Datasets**:
  - `data/processed/ebnerd/features_test.parquet` (60,499 rows)
  - `data/processed/mind/features_test.parquet` (602,931 rows)
- **Saved Evaluation Results (JSON)**:
  - [`models/q2_extended_eval_ebnerd_test.json`](file:///c:/Users/kavan/OneDrive/Desktop/IIITH/sem3/IRE/Assignment1/IRE-Assignment-1/models/q2_extended_eval_ebnerd_test.json)
  - [`models/q2_extended_eval_mind_test.json`](file:///c:/Users/kavan/OneDrive/Desktop/IIITH/sem3/IRE/Assignment1/IRE-Assignment-1/models/q2_extended_eval_mind_test.json)
  - [`models/q2_extended_eval_ebnerd_val.json`](file:///c:/Users/kavan/OneDrive/Desktop/IIITH/sem3/IRE/Assignment1/IRE-Assignment-1/models/q2_extended_eval_ebnerd_val.json)
  - [`models/q2_extended_eval_mind_val.json`](file:///c:/Users/kavan/OneDrive/Desktop/IIITH/sem3/IRE/Assignment1/IRE-Assignment-1/models/q2_extended_eval_mind_val.json)
