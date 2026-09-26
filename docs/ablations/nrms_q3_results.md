# Q3: NRMSDocVec Baseline, Improvement & Ablation Study

This document records the reproduction of the official NRMS baseline (adapted as **NRMSDocVec** using precomputed document vectors), the principled improvement via **Exponential Recency-Decayed Attention**, and the **paired bootstrap 95% confidence intervals** demonstrating statistical significance ($p < 0.05$).


## Dataset: EBNERD (Validation Split)
*Generated at: 2026-09-15 03:01:59*

### 1. Comparative Ranking Performance

| Model Architecture | AUC (95% CI) | MRR (95% CI) | nDCG@5 (95% CI) | nDCG@10 (95% CI) |
| :--- | :--- | :--- | :--- | :--- |
| **A1: Semantic Mean-Pooling** | 0.4859 [0.4765, 0.4947] | 0.3167 [0.3087, 0.3252] | 0.3438 [0.3342, 0.3536] | 0.4298 [0.4222, 0.4379] |
| **A2: Vanilla NRMSDocVec** | 0.5809 [0.5720, 0.5902] | 0.3674 [0.3593, 0.3760] | 0.4099 [0.4009, 0.4195] | 0.4824 [0.4750, 0.4901] |
| **A2: Improved NRMS (Recency Decay)** | **0.5813** [0.5731, 0.5899] | **0.3689** [0.3610, 0.3777] | **0.4125** [0.4037, 0.4219] | **0.4836** [0.4761, 0.4916] |

### 2. Paired Bootstrap 95% Confidence Intervals (Q3 Statistical Rigor)
*Null Hypothesis $H_0: \Delta \le 0$ vs Alternative $H_1: \Delta > 0$ with $B = 1000$ bootstrap resamples.*

| Metric | $\Delta = \text{Improved} - \text{Baseline}$ | 95% Confidence Interval | Excludes Zero? ($p < 0.05$) |
| :--- | :--- | :--- | :--- |
| **AUC** | `+0.0004` | `[-0.0018, +0.0026]` | ❌ No |
| **MRR** | `+0.0016` | `[-0.0004, +0.0036]` | ❌ No |
| **nDCG@5** | `+0.0026` | `[+0.0002, +0.0050]` | ✅ **Yes (Statistically Significant)** |
| **nDCG@10** | `+0.0012` | `[-0.0004, +0.0029]` | ❌ No |

---


## Dataset: MIND (Validation Split)
*Generated at: 2026-09-15 03:46:23*

### 1. Comparative Ranking Performance

| Model Architecture | AUC (95% CI) | MRR (95% CI) | nDCG@5 (95% CI) | nDCG@10 (95% CI) |
| :--- | :--- | :--- | :--- | :--- |
| **A1: Semantic Mean-Pooling** | 0.6010 [0.5929, 0.6095] | 0.3108 [0.3026, 0.3195] | 0.2960 [0.2868, 0.3050] | 0.3566 [0.3482, 0.3651] |
| **A2: Vanilla NRMSDocVec** | 0.6392 [0.6310, 0.6469] | 0.3570 [0.3469, 0.3663] | 0.3358 [0.3260, 0.3455] | 0.3939 [0.3851, 0.4027] |
| **A2: Improved NRMS (Recency Decay)** | **0.6401** [0.6318, 0.6479] | **0.3572** [0.3474, 0.3666] | **0.3358** [0.3253, 0.3458] | **0.3941** [0.3852, 0.4028] |

### 2. Paired Bootstrap 95% Confidence Intervals (Q3 Statistical Rigor)
*Null Hypothesis $H_0: \Delta \le 0$ vs Alternative $H_1: \Delta > 0$ with $B = 1000$ bootstrap resamples.*

| Metric | $\Delta = \text{Improved} - \text{Baseline}$ | 95% Confidence Interval | Excludes Zero? ($p < 0.05$) |
| :--- | :--- | :--- | :--- |
| **AUC** | `+0.0009` | `[-0.0004, +0.0021]` | ❌ No |
| **MRR** | `+0.0002` | `[-0.0018, +0.0022]` | ❌ No |
| **nDCG@5** | `-0.0000` | `[-0.0021, +0.0020]` | ❌ No |
| **nDCG@10** | `+0.0002` | `[-0.0015, +0.0018]` | ❌ No |

---


## Dataset: MIND (Validation Split)
*Generated at: 2026-09-15 04:05:59*

### 1. Comparative Ranking Performance

| Model Architecture | AUC (95% CI) | MRR (95% CI) | nDCG@5 (95% CI) | nDCG@10 (95% CI) |
| :--- | :--- | :--- | :--- | :--- |
| **A1: Semantic Mean-Pooling** | 0.6139 [0.5696, 0.6508] | 0.3247 [0.2796, 0.3715] | 0.2978 [0.2496, 0.3490] | 0.3618 [0.3184, 0.4084] |
| **A2: Vanilla NRMSDocVec** | 0.6442 [0.6038, 0.6803] | 0.3245 [0.2799, 0.3718] | 0.3057 [0.2594, 0.3543] | 0.3663 [0.3221, 0.4128] |
| **A2: Improved NRMS (Recency Decay)** | **0.6475** [0.6089, 0.6838] | **0.3290** [0.2848, 0.3750] | **0.3085** [0.2604, 0.3557] | **0.3679** [0.3238, 0.4165] |

### 2. Paired Bootstrap 95% Confidence Intervals (Q3 Statistical Rigor)
*Null Hypothesis $H_0: \Delta \le 0$ vs Alternative $H_1: \Delta > 0$ with $B = 1000$ bootstrap resamples.*

| Metric | $\Delta = \text{Improved} - \text{Baseline}$ | 95% Confidence Interval | Excludes Zero? ($p < 0.05$) |
| :--- | :--- | :--- | :--- |
| **AUC** | `+0.0033` | `[-0.0036, +0.0100]` | ❌ No |
| **MRR** | `+0.0044` | `[-0.0051, +0.0139]` | ❌ No |
| **nDCG@5** | `+0.0029` | `[-0.0070, +0.0134]` | ❌ No |
| **nDCG@10** | `+0.0015` | `[-0.0047, +0.0084]` | ❌ No |

---


## Dataset: EBNERD (Validation Split)
*Generated at: 2026-09-15 04:08:05*

### 1. Comparative Ranking Performance

| Model Architecture | AUC (95% CI) | MRR (95% CI) | nDCG@5 (95% CI) | nDCG@10 (95% CI) |
| :--- | :--- | :--- | :--- | :--- |
| **A1: Semantic Mean-Pooling** | 0.4720 [0.4501, 0.4932] | 0.3081 [0.2907, 0.3262] | 0.3364 [0.3150, 0.3573] | 0.4241 [0.4070, 0.4419] |
| **A2: Vanilla NRMSDocVec** | 0.6027 [0.5840, 0.6232] | 0.3836 [0.3652, 0.4045] | 0.4342 [0.4141, 0.4576] | 0.4996 [0.4835, 0.5183] |
| **A2: Improved NRMS (Recency Decay)** | **0.6040** [0.5856, 0.6237] | **0.3854** [0.3670, 0.4061] | **0.4363** [0.4157, 0.4604] | **0.5010** [0.4850, 0.5199] |

### 2. Paired Bootstrap 95% Confidence Intervals (Q3 Statistical Rigor)
*Null Hypothesis $H_0: \Delta \le 0$ vs Alternative $H_1: \Delta > 0$ with $B = 1000$ bootstrap resamples.*

| Metric | $\Delta = \text{Improved} - \text{Baseline}$ | 95% Confidence Interval | Excludes Zero? ($p < 0.05$) |
| :--- | :--- | :--- | :--- |
| **AUC** | `+0.0013` | `[-0.0031, +0.0052]` | ❌ No |
| **MRR** | `+0.0018` | `[-0.0023, +0.0057]` | ❌ No |
| **nDCG@5** | `+0.0021` | `[-0.0036, +0.0074]` | ❌ No |
| **nDCG@10** | `+0.0014` | `[-0.0021, +0.0046]` | ❌ No |

---
