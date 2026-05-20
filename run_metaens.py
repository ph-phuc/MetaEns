"""
run_metaens.py  –  LODO benchmark evaluation
=============================================
Reproduces the main paper results using Leave-One-Dataset-Out (LODO)
cross-validation across all 39 benchmark datasets.

Run from the repo root:
    python3 run_metaens.py
"""

import os
import time

import numpy as np
import pandas as pd

from algorithms.elect import ELECT
from algorithms.metaens_selector import MetaEnsSelector
from utils.core import BASE_PATH, SCRIPT_DIR, create_score_loader
from utils.evaluation import evaluate_selector, compute_results_dataframe
from utils.metrics import compute_all_metrics_for_dataset

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

INTERMEDIATE_FILES_FOLDER = 'intermediate_files'

DATASET_FILE_PATH  = os.path.join(BASE_PATH, INTERMEDIATE_FILES_FOLDER, 'datasets.txt')
MODELS_FILE_PATH   = os.path.join(BASE_PATH, INTERMEDIATE_FILES_FOLDER, 'models.txt')
METAENS_CACHE_PATH = os.path.join(SCRIPT_DIR, 'cache/cache_train_meta_ens.npz')

# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

SEEDS = [100]

# ELECT shared instance — class-level cache avoids recomputing IPMs across folds.
_ELECT = ELECT(base_path=BASE_PATH, n_selection=1, verbose=True)

# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def _run_one_seed(seed, available_datasets, selection_pool_models, results_df, score_loader):
    """Run LODO evaluation for one seed; return (ap, roc_auc, median_rank, df)."""
    df = evaluate_selector(
        selector_class=MetaEnsSelector,
        selector_params={
            "train_meta_cache": METAENS_CACHE_PATH,
            "primary_selector": _ELECT,
            "seed": seed,
        },
        available_datasets=available_datasets,
        available_models=selection_pool_models,
        results_df=results_df,
        score_loader=score_loader,
        verbose=True,
        normalization='minmax',
        combining_method='mean',
    )
    if df.empty:
        return 0.0, float('nan'), float('nan'), df
    ap   = float(df['ap'].mean())
    auc  = float(df['roc_auc'].mean())  if 'roc_auc'  in df.columns else float('nan')
    rank = float(df['rank_ap'].median()) if 'rank_ap' in df.columns else float('nan')
    return ap, auc, rank, df


def main():
    # -----------------------------------------------------------------------
    # Load lists
    # -----------------------------------------------------------------------
    with open(DATASET_FILE_PATH) as f:
        available_datasets = [line.strip() for line in f if line.strip()]
    with open(MODELS_FILE_PATH) as f:
        selection_pool_models = [line.strip() for line in f if line.strip()]

    print(f"Datasets : {len(available_datasets)}")
    print(f"Models   : {len(selection_pool_models)}")
    print(f"Seeds    : {SEEDS}")

    score_loader = create_score_loader(INTERMEDIATE_FILES_FOLDER)

    # Pre-compute per-model AP / ROC-AUC used for ranking inside evaluate_selector
    print("\nPre-computing model metrics …")
    results_df = compute_results_dataframe(
        process_dataset_func=compute_all_metrics_for_dataset,
        available_datasets=available_datasets,
        available_models=selection_pool_models,
        intermediate_files_folder=INTERMEDIATE_FILES_FOLDER,
        verbose=False,
    )

    # -----------------------------------------------------------------------
    # Run MetaEns+ELECT across all seeds
    # -----------------------------------------------------------------------
    aps, aucs, ranks, run_dfs = [], [], [], []
    t0 = time.time()

    for i, seed in enumerate(SEEDS):
        print(f"\n  Seed {i+1}/{len(SEEDS)}  (seed={seed})")
        ap, auc, rank, df = _run_one_seed(
            seed, available_datasets, selection_pool_models, results_df, score_loader,
        )
        aps.append(ap); aucs.append(auc); ranks.append(rank)
        df['seed'] = seed
        run_dfs.append(df)
        print(f"    AP={ap:.4f}  ROC-AUC={auc:.4f}  Median-Rank={rank:.1f}")

    elapsed = time.time() - t0
    print(f"\n  Final: AP={np.mean(aps):.4f} \u00b1 {np.std(aps):.4f}  "
          f"ROC-AUC={np.nanmean(aucs):.4f}  "
          f"Median-Rank={np.nanmean(ranks):.1f}  "
          f"Time={elapsed:.1f}s")

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    os.makedirs(os.path.join(SCRIPT_DIR, 'results'), exist_ok=True)

    detailed_path = os.path.join(SCRIPT_DIR, 'results', 'metaens_elect_detailed.csv')
    pd.concat(run_dfs, ignore_index=True).to_csv(detailed_path, index=False)
    print(f"  Saved: {detailed_path}")

    summary_path = os.path.join(SCRIPT_DIR, 'results', 'baseline_results.csv')
    pd.DataFrame([{
        'Method':         'MetaEns+ELECT',
        'AP (Mean)':      float(np.mean(aps)),
        'AP (Std)':       float(np.std(aps)),
        'ROC-AUC (Mean)': float(np.nanmean(aucs)),
        'Median AP Rank': float(np.nanmean(ranks)),
        'Time (s)':       elapsed,
    }]).to_csv(summary_path, index=False)
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
