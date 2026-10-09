# MetaEns: Automatic Unsupervised Ensemble Outlier Model Selection

Official implementation of **"Automatic Unsupervised Ensemble Outlier Model Selection"** (ICML 2026).

**Paper:** [arXiv:2605.16567](https://arxiv.org/abs/2605.16567) (extended version)

Hong-Phuc Phan\*, Tuan-Anh Vu\*, Tung Kieu\*, Son Ha Xuan, Bin Yang, Christian S. Jensen (\*equal contribution)

---

## Table of Contents

- [Overview](#overview)
- [How It Works](#how-it-works)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Reproduce Paper Results](#reproduce-paper-results)
- [Benchmark Datasets](#benchmark-datasets)
- [Citation](#citation)

---

## Overview

**MetaEns** is a meta-learning framework that automatically constructs small, high-quality outlier detection ensembles without any labels on the target dataset. Given a pool of pre-trained detectors and their raw anomaly scores on a new dataset, MetaEns selects a compact, diverse subset that tends to outperform individual detectors and naive ensemble averaging.

**Key capabilities:**

- **Unsupervised** — no ground-truth labels required at test time
- **Score-only** — operates on detector output scores, independent of the input modality (tabular, image, text)
- **Plug-and-play** — scikit-learn-style `fit` / `select` / `predict` API
- **Extensible** — works with any detector that outputs a scalar anomaly score per sample

---

## How It Works

MetaEns operates in two phases:

### Offline (meta-training)

Given a collection of historical datasets with ground-truth labels, MetaEns learns a two-part meta-model (ExtraTrees classifier + regressor) that predicts the marginal AP gain of adding a candidate detector to a growing ensemble.  Meta-features are computed from raw score statistics.

### Online (test time)

For a new, unlabeled dataset:

1. **Primary selection** — A kNN search over historical datasets (optionally guided by [ELECT](https://github.com/yzhao062/ELECT)) identifies a strong single detector as a starting point.
2. **Partner selection** — The meta-model iteratively proposes candidates and predicts their marginal gain.  Two regularisers discourage degenerate ensembles:
   - **Submodular diversity discount** — penalises candidates whose top-ranked samples heavily overlap with already-selected detectors (Jaccard similarity).
   - **Family-risk prior** — penalises selecting multiple detectors from the same algorithmic family (e.g. multiple IForest variants), using an Empirical-Bayes prior calibrated from historical data.
3. **Adaptive stopping** — Partners are added only when the predicted (discounted) gain exceeds a threshold.

---

## Installation

### Requirements

- Python 3.9 or later (3.10 recommended)
- Dependencies listed in `requirements.txt`

### Setup

```bash
git clone https://github.com/ph-phuc/MetaEns.git
cd MetaEns

python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Core dependencies:

| Package | Purpose |
|---|---|
| `numpy`, `scipy`, `pandas` | Numerical computing and data handling |
| `scikit-learn` | ExtraTrees meta-model |
| `polars`, `pyarrow` | Fast score-matrix I/O |
| `lightgbm` | ELECT primary selector (optional) |
| `networkx`, `tqdm` | ELECT graph features (optional) |

> **Note:** `lightgbm`, `networkx`, and `tqdm` are only required if you use ELECT as the primary selector (full paper configuration).  The default kNN-based primary selector has no additional dependencies.

---

## Quick Start

The following snippet demonstrates the full MetaEns workflow on your own detector pool.

```python
import numpy as np
from metaens import MetaEns

# ------------------------------------------------------------------
# 1. Collect score matrices from your detector pool
#    historical_scores : dict[dataset_name -> np.ndarray (n_samples, n_models)]
#    historical_labels : dict[dataset_name -> np.ndarray (n_samples,)]  0/1
# ------------------------------------------------------------------
historical_scores = {
    "cardio":  scores_cardio,    # shape (n_samples, n_models) — one column per detector
    "shuttle": scores_shuttle,
    # ...
}
historical_labels = {
    "cardio":  y_cardio,
    "shuttle": y_shuttle,
    # ...
}
model_names = ["IForest_1", "LOF_1", "HBOS_1", ...]  # must align with columns

# ------------------------------------------------------------------
# 2. Fit the meta-model (offline phase)
# ------------------------------------------------------------------
selector = MetaEns(model_names=model_names, seed=42)
selector.fit(historical_scores, historical_labels)

# ------------------------------------------------------------------
# 3. Apply to a new, unlabelled dataset (online phase)
#    X_test : np.ndarray (n_samples, n_models) — same column order as model_names
# ------------------------------------------------------------------
selected_detectors = selector.select(X_test)       # e.g. ["IForest_1", "LOF_1"]
anomaly_scores     = selector.predict(X_test)      # np.ndarray shape (n_samples,)
```

**Run the included demo** (requires benchmark data — see [Benchmark Datasets](#benchmark-datasets)):

```bash
# Hold out 'annthyroid' as test, kNN primary selector; cache meta-features for fast reruns
python3 example_usage.py --cache cache/cache_train_meta_ens.npz

# Specify a different test dataset
python3 example_usage.py --cache cache/cache_train_meta_ens.npz --test-dataset mnist

# Full paper configuration: use ELECT as the primary selector
python3 example_usage.py --cache cache/cache_train_meta_ens.npz --elect
```

> On the first run the meta-feature cache is built automatically; subsequent runs reuse it.

---

## Reproduce Paper Results

`run_metaens.py` reproduces the LODO evaluation reported in Table 1 of the paper.

**Leave-One-Dataset-Out (LODO) benchmark — 39 datasets, pool of 297 detectors:**

| Method | AP | Avg. Rank |
|---|---|---|
| **MetaEns (ours)** | **0.4308** | **59.3** |
| ELECT Top-1 | 0.413 | 87.0 |
| ELECT Top-10 | 0.414 | 86.0 |
| Mega Ensemble (all 297) | 0.397 | 100.0 |
| IForest (best single) | 0.398 | 102.0 |

> Numbers above are from the paper (mean over seeds [0, 1, 2, 3, 4, 5, 6, 42, 100, 1000]). To closely match the paper's reported numbers, set `SEEDS` in `run_metaens.py` to that list. (Minor numerical differences may still arise due to library versions and hardware.)

The default `SEEDS = [100]` runs a single seed for a quick check.

```bash
python3 run_metaens.py
```

This runs Leave-One-Dataset-Out cross-validation across all 39 benchmark datasets with ELECT as the primary selector (full paper configuration).

The meta-feature cache in `cache/` is built automatically on the first run; subsequent runs reuse it.

**Output files:**

| File | Contents |
|---|---|
| `results/baseline_results.csv` | Summary: AP, ROC-AUC, Median AP Rank per method |
| `results/metaens_elect_detailed.csv` | Per-dataset breakdown |

---

## Benchmark Datasets

The benchmark uses **39 real-world tabular datasets** drawn from the [ELECT benchmark (ICDM 2022)](https://github.com/yzhao062/ELECT), originally sourced from the [ODDS](http://odds.cs.stonybrook.edu/) and [UCI Machine Learning Repository](https://archive.ics.uci.edu/) repositories.  Datasets cover a range of domains and sizes, with contamination rates from 0.03% to 35%.

The detector pool comprises **297 detectors** from **8 algorithm families** with diverse hyperparameter configurations:

| Family | Algorithm |
|---|---|
| `ABOD` | Angle-Based Outlier Detection |
| `COF` | Connectivity-Based Outlier Factor |
| `HBOS` | Histogram-Based Outlier Score |
| `IForest` | Isolation Forest |
| `kNN` | k-Nearest Neighbour distance |
| `LODA` | Lightweight Online Detector of Anomalies |
| `LOF` | Local Outlier Factor |
| `OCSVM` | One-Class Support Vector Machine |

### Download

The precomputed score matrices and ground-truth labels are hosted on Google Drive:

**[Download benchmark data (Google Drive)](https://drive.google.com/file/d/1ubFKuRHuv6iQ5BlsmMyvISaqLzyqCoXT/view?usp=drive_link)**

After downloading, extract the archive so the directory layout matches:

```
MetaEns/
└── datasets/
    └── benchmark/
        ├── data/                    # ground-truth label CSVs
        └── intermediate_files/      # score matrices, datasets.txt, models.txt, ...
```

---

## Citation

If you use MetaEns in your research, please cite:

```bibtex
@inproceedings{metaens2026,
  title     = {Automatic Unsupervised Ensemble Outlier Model Selection},
  author    = {Hong-Phuc Phan and Tuan-Anh Vu and Tung Kieu and Son Ha Xuan and Bin Yang and Christian S. Jensen},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning (ICML)},
  year      = {2026},
  note      = {Extended version: arXiv:2605.16567},
}
```