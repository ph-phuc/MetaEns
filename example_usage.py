"""
The high-level MetaEns API (fit / select / predict) using the
benchmark datasets downloaded to datasets/benchmark/.

Run from the repo root:
    python3 example_usage.py

Optional: persist the meta-feature cache so repeated runs are fast:
    python3 example_usage.py --cache /tmp/metaens_cache.npz

By default this demo uses MetaEns with the built-in kNN primary selector.
Pass ``--elect`` to enable ELECT-powered primary selection (the full paper
configuration), which requires the precomputed ELECT wild data under
datasets/benchmark/.
"""

import argparse
import os
import sys
import time

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))

from metaens import MetaEns
from utils.core import BASE_PATH, load_dataset_scores


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_benchmark_data():
    """Load all benchmark datasets from datasets/benchmark/."""
    ds_file  = os.path.join(BASE_PATH, "intermediate_files", "datasets.txt")
    mdl_file = os.path.join(BASE_PATH, "intermediate_files", "models.txt")

    with open(ds_file) as f:
        all_datasets = f.read().splitlines()
    with open(mdl_file) as f:
        all_models = f.read().splitlines()

    score_matrices, labels = {}, {}
    for ds in all_datasets:
        sc, y = load_dataset_scores(ds)
        if sc is not None:
            score_matrices[ds] = sc.to_numpy()   # shape (n_samples, n_models)
            labels[ds]         = y                # shape (n_samples,)  0/1

    return score_matrices, labels, all_models


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MetaEns API example")
    parser.add_argument("--test-dataset", default="annthyroid",
                        help="Dataset to hold out as the test set (default: annthyroid)")
    parser.add_argument("--cache", default=None,
                        help="Path for the meta-feature cache (.npz).  "
                             "If the file exists it is reused; otherwise it is "
                             "generated and saved here (speeds up future runs).")
    parser.add_argument("--elect", action="store_true",
                        help="Use ELECT as the primary selector (full paper configuration).")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1.  Load data
    # ------------------------------------------------------------------
    print("Loading benchmark datasets …")
    score_matrices, labels, all_models = load_benchmark_data()
    print(f"  {len(score_matrices)} datasets,  {len(all_models)} detectors in pool")

    test_ds = args.test_dataset
    if test_ds not in score_matrices:
        sys.exit(f"Dataset '{test_ds}' not found.  "
                 f"Available: {sorted(score_matrices)}")

    train_scores = {k: v for k, v in score_matrices.items() if k != test_ds}
    train_labels = {k: v for k, v in labels.items()         if k != test_ds}
    X_test       = score_matrices[test_ds]   # (n_samples, n_models)
    y_test       = labels[test_ds]           # (n_samples,)

    print(f"  Train: {len(train_scores)} datasets")
    print(f"  Test : '{test_ds}'  shape={X_test.shape}  "
          f"anomaly rate={y_test.mean():.1%}")

    # ------------------------------------------------------------------
    # 2.  (Optional) ELECT primary selector
    # ------------------------------------------------------------------
    primary_selector = None
    if args.elect:
        from algorithms.elect import ELECT
        primary_selector = ELECT(base_path=BASE_PATH, n_selection=1, verbose=False)
        print("\nUsing ELECT as primary selector (full paper configuration).")

    # ------------------------------------------------------------------
    # 3.  Create and fit MetaEns
    # ------------------------------------------------------------------
    selector = MetaEns(
        model_names=all_models,
        seed=100,
        primary_selector=primary_selector,
    )

    print("\nFitting meta-model …")
    t0 = time.time()
    selector.fit(train_scores, train_labels, cache_path=args.cache, verbose=True)
    print(f"  fit() completed in {time.time() - t0:.1f}s")
    if args.cache:
        print(f"  Cache saved/reused at: {args.cache}")

    # ------------------------------------------------------------------
    # 4.  Select ensemble for the test dataset
    # ------------------------------------------------------------------
    print("\nSelecting ensemble …")
    t1 = time.time()
    selected = selector.select(X_test)
    print(f"  select() completed in {time.time() - t1:.2f}s")
    print(f"  Selected {len(selected)} detectors:")
    for i, name in enumerate(selected):
        print(f"    [{i}] {name}")

    # ------------------------------------------------------------------
    # 5.  Predict combined anomaly scores
    # ------------------------------------------------------------------
    anomaly_scores = selector.predict(X_test)   # (n_samples,)

    # ------------------------------------------------------------------
    # 6.  Evaluate
    # ------------------------------------------------------------------
    ap  = average_precision_score(y_test, anomaly_scores)
    auc = roc_auc_score(y_test, anomaly_scores)

    # Reference baselines — min-max normalize per column (same as MetaEns internally)
    X_norm = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0).astype(float)
    col_min = X_norm.min(axis=0)
    col_max = X_norm.max(axis=0)
    rng = col_max - col_min
    rng[rng == 0] = 1.0
    X_norm = (X_norm - col_min) / rng

    mean_all_ap  = average_precision_score(y_test, X_norm.mean(axis=1))
    mean_all_auc = roc_auc_score(y_test, X_norm.mean(axis=1))

    # Vectorized AP across all columns (avoids 297 individual sklearn calls)
    order          = np.argsort(-X_norm, axis=0)
    y_ranked       = y_test[order]
    cum_pos        = np.cumsum(y_ranked, axis=0)
    prec_at_k      = cum_pos / np.arange(1, len(y_test) + 1)[:, None]
    n_pos          = max(float(y_test.sum()), 1.0)
    col_aps        = (prec_at_k * y_ranked).sum(axis=0) / n_pos
    best_single_ap = float(col_aps.max())
    best_single_auc = float(roc_auc_score(y_test, X_norm[:, int(col_aps.argmax())]))

    # Primary model (the single detector MetaEns starts from)
    primary = selected[0]
    p_idx = all_models.index(primary)
    primary_ap  = average_precision_score(y_test, X_norm[:, p_idx])
    primary_auc = roc_auc_score(y_test, X_norm[:, p_idx])

    print(f"\n{'=' * 50}")
    print(f"  Results on '{test_ds}'")
    print(f"{'=' * 50}")
    print(f"  MetaEns AP            : {ap:.4f}   ROC-AUC : {auc:.4f}")
    print(f"  --- baselines ---")
    print(f"  Primary ({primary})")
    print(f"    AP : {primary_ap:.4f}   ROC-AUC : {primary_auc:.4f}")
    print(f"  Mean of all detectors : {mean_all_ap:.4f}   ROC-AUC : {mean_all_auc:.4f}")
    print(f"  Best single (oracle)  : {best_single_ap:.4f}   ROC-AUC : {best_single_auc:.4f}")


if __name__ == "__main__":
    main()

