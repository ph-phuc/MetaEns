import pandas as pd
import numpy as np
from typing import List, Dict, Type, Any, Callable, Union, Tuple, Optional
from scipy.stats import rankdata
import time
import multiprocessing as mp
import os
from functools import partial
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)

from algorithms.base import SingleModelSelector

def _round_if_enabled(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Round AP values to a fixed number of decimal places for reproducibility."""
    try:
        dec = int(os.environ.get("ROUND_AP_DECIMALS", "4") or "4")
    except Exception:
        dec = 0
    if dec <= 0:
        return x
    try:
        if isinstance(x, np.ndarray):
            return np.round(x.astype(float), decimals=dec)
        return float(np.round(float(x), decimals=dec))
    except Exception:
        return x

def get_rank_from_scores(scores, target_index):
    """
    Calculates competition rank (1 is best, ties share best possible rank).
    Uses method='min' (standard in ML competitions/benchmarks).
    Explicitly rounds scores to ROUND_AP_DECIMALS to handle floating-point ties.
    """
    scores_arr = np.array(scores)
    
    # Consistent rounding to match historical_df and ensemble results
    try:
        dec = int(os.environ.get("ROUND_AP_DECIMALS", "4") or "4")
    except Exception:
        dec = 4
    
    if dec > 0:
        scores_arr = np.round(scores_arr.astype(float), decimals=dec)
        
    # Use rankdata on negated scores to get descending effect
    # method='min' assigns 1 to the smallest value (which is the largest original).
    ranks = rankdata(-scores_arr, method='min')
    return ranks[target_index]

def normalize_scores(scores: np.ndarray, method: str) -> np.ndarray:
    """
    Normalizes scores using the specified method.
    Args:
        scores: 2D numpy array of scores (n_samples, n_models)
        method: 'minmax', 'zscore', or 'rank'
    Returns:
        Normalized scores
    """
    if method is None or method.lower() == 'none':
        return scores
        
    if method == 'minmax':
        # Min-Max scaling to [0, 1]
        min_val = scores.min(axis=0)
        max_val = scores.max(axis=0)
        # Avoid division by zero
        range_val = max_val - min_val
        range_val[range_val == 0] = 1.0
        return (scores - min_val) / range_val
        
    elif method == 'zscore':
        # Z-Score standardization
        mean_val = scores.mean(axis=0)
        std_val = scores.std(axis=0)
        # Avoid division by zero
        std_val[std_val == 0] = 1.0
        return (scores - mean_val) / std_val
        
    elif method == 'rank':
        # Rank normalization (scaled to [0, 1])
        # rankdata returns 1..n
        ranks = np.apply_along_axis(rankdata, 0, scores)
        return ranks / len(scores)
        
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def calculate_additional_metrics(y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
    """Calculates ROC-AUC for the given scores."""
    metrics = {}
    try:
        metrics['roc_auc'] = roc_auc_score(y_true, y_score)
    except Exception:
        metrics['roc_auc'] = 0.0
    return metrics

def evaluate_selector(
    selector_class: Type[SingleModelSelector],
    selector_params: Dict[str, Any],
    available_datasets: List[str],
    available_models: List[str],
    results_df: pd.DataFrame,
    score_loader: Callable[[str], Tuple[np.ndarray, np.ndarray]] = None,
    combining_method: str = 'mean',
    normalization: str = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Evaluates a SingleModelSelector using Leave-One-Dataset-Out (LODO) cross-validation.

    Args:
        selector_class: The class of the selector to evaluate.
        selector_params: Dictionary of parameters to pass to the selector's __init__.
        available_datasets: List of dataset names to evaluate on.
        available_models: List of available model names.
        results_df: DataFrame containing ground truth AP scores (columns: dataset, model, ap).
        score_loader: Optional function that takes dataset_name and returns (scores_matrix, y_true).
                      Required for evaluating ensemble selections (list of models) AND for calculating
                      additional metrics (ROC AUC, F1, etc.) for single models.
        combining_method: Method to combine scores for ensemble ('mean' or 'max'). Default 'mean'.
        normalization: Method to normalize scores before combining ('minmax', 'zscore', 'rank', or None).
        verbose: Whether to print progress and results.

    Returns:
        DataFrame containing evaluation results for each dataset.
    """
    
    if verbose:
        print("="*80)
        print(f"EVALUATION: {selector_class.__name__}")
        print("="*80)
        print(f"Datasets: {len(available_datasets)}")
        print(f"Models:   {len(available_models)}")
        print(f"Combine:  {combining_method}")
        print(f"Norm:     {normalization}")
        print("-" * 120)
    final_results = []
    start_time = time.time()
    
    # Flag to detect if we are doing ensemble selection (K>1)
    is_ensemble_run = False

    for ds_i, dataset_name in enumerate(available_datasets):
        fold_t0 = time.time()
        if verbose:
            print(f"[{ds_i+1:>3d}/{len(available_datasets):<3d}] dataset={dataset_name} | fold_start", flush=True)
        
        # Initialize timing metrics
        time_fit = 0.0
        time_select = 0.0
        time_inference_total = 0.0
        
        # 1. Prepare Training Data (Leave-One-Out)
        # Exclude the current dataset from the history used by the selector
        historical_df = results_df[results_df['dataset'] != dataset_name]
        
        # 2. Instantiate and Fit Selector (Offline Phase - Training)
        fit_t0 = time.time()
        selector = selector_class(**selector_params)
        if verbose:
            print(f"[{ds_i+1:>3d}/{len(available_datasets):<3d}] dataset={dataset_name} | fitting selector...", flush=True)
        selector.fit(historical_df, available_models)
        time_fit = time.time() - fit_t0
        if verbose:
            print(f"[{ds_i+1:>3d}/{len(available_datasets):<3d}] dataset={dataset_name} | fit_done (elapsed_s={time_fit:.2f})", flush=True)
        
        # 3. Select (Online Phase - Inference)
        select_t0 = time.time()
        if verbose:
            print(f"[{ds_i+1:>3d}/{len(available_datasets):<3d}] dataset={dataset_name} | selecting...", flush=True)
        selected_model = selector.select(dataset_name)
        time_select = time.time() - select_t0
        time_inference_total = time_select  # Base inference time
        
        if verbose:
            sel_desc = selected_model
            if isinstance(selected_model, list):
                sel_desc = f"ensemble(k={len(selected_model)})"
            print(
                f"[{ds_i+1:>3d}/{len(available_datasets):<3d}] dataset={dataset_name} | select_done -> {sel_desc} (elapsed_s={time_select:.2f})",
                flush=True,
            )
            print("--> Selected Model(s):", selected_model, flush=True)
        
        if selected_model is None:
            if verbose:
                # Skip this fold when selection returns None.
                print(f"[{ds_i+1:>3d}/{len(available_datasets):<3d}] dataset={dataset_name} | fold_skip (selected_model=None)", flush=True)
            continue
            
        # Check if it's an ensemble run
        if isinstance(selected_model, list) and len(selected_model) > 1:
            is_ensemble_run = True
            
        # -------------------------------------------------------------------------
        # 4. Evaluate Performance & Calculate Gains
        # -------------------------------------------------------------------------
        # Load scores if available (needed for ensemble evaluation and consistent AP calc)
        scores_matrix = None
        y_true = None
        if score_loader:
            try:
                scores_matrix, y_true = score_loader(dataset_name)
            except Exception as e:
                if verbose:
                    print(f"[warn] dataset={dataset_name} | score_loader failed: {e}", flush=True)
        # IMPORTANT: Scores are loaded from CSVs with no headers, so column order MUST match
        # the order of `available_models` (typically read from models.txt). If these diverge,
        # all AP/rank/gain computations that index by model position will be incorrect.
        if scores_matrix is not None:
            try:
                n_cols = int(getattr(scores_matrix, "shape", [None, None])[1])
                if n_cols is not None and n_cols != len(available_models) and verbose:
                    print(
                        f"[warn] dataset={dataset_name} | score_matrix_cols={n_cols} != len(available_models)={len(available_models)}. "
                        "This code assumes score CSV column order matches models.txt order; mismatches will corrupt metrics.",
                        flush=True,
                    )
            except Exception:
                # best-effort warning only
                pass

        # Build AP map for this dataset.
        # IMPORTANT: If scores_matrix/y_true are available, compute APs from them using
        # the same normalization as ensembles. This avoids apples-to-oranges comparisons
        # where ensembles are evaluated on normalized scores but single models use precomputed AP.
        ds_res = results_df[results_df['dataset'] == dataset_name]
        model_ap_map = dict(zip(ds_res['model'], ds_res['ap']))
        if scores_matrix is not None and y_true is not None:
            try:
                pool_scores = scores_matrix
                if normalization:
                    pool_scores = normalize_scores(pool_scores, normalization)
                # Guard against accidental shape mismatches; we can only map APs for columns we have.
                m = min(int(pool_scores.shape[1]), int(len(available_models)))
                ap_vals = [average_precision_score(y_true, pool_scores[:, j]) for j in range(m)]
                ap_vals = _round_if_enabled(np.asarray(ap_vals, dtype=float)).tolist()
                model_ap_map = dict(zip(available_models, ap_vals))
            except Exception:
                # fall back to results_df APs if something goes wrong
                pass

        # Get all APs for ranking comparison
        all_aps = [model_ap_map.get(m, 0.0) for m in available_models]

        selected_ap = 0.0
        rank = -1.0
        display_name = ""

        # Gain Metrics
        primary_model_name = None
        primary_ap = 0.0
        primary_rank = -1.0
        gain_vs_primary = 0.0

        # Dictionary to hold all metrics
        metrics = {'ap': 0.0, 'roc_auc': 0.0}
        
        if isinstance(selected_model, list):
            # Ensemble Selection
            # ------------------
            if not selected_model:
                 selected_ap = 0.0
                 rank = len(available_models)
                 display_name = "Empty"
            else:
                # Identify Primary Model (first in list)
                primary_model_name = selected_model[0]
                primary_ap = model_ap_map.get(primary_model_name, 0.0)
                try:
                    p_idx = available_models.index(primary_model_name)
                    primary_rank = get_rank_from_scores(all_aps, p_idx)
                except Exception:
                    primary_rank = -1.0

                # Detailed Name
                # Truncate long names for display if needed, but keep full list for record
                # e.g. "IForest(..)+HBOS(..)"
                short_names = [str(m) for m in selected_model]
                
                # Join with ' + ' for readability
                display_name = " + ".join(short_names)
                
                if scores_matrix is None or y_true is None:
                    print(f"Error: Scores required for ensemble evaluation on {dataset_name}")
                    selected_ap = 0.0
                    rank = -1.0
                    display_name = "Error"
                else:
                    try:
                        # Time ensemble combination (part of inference)
                        ensemble_t0 = time.time()
                        
                        # Find indices of selected models
                        indices = [available_models.index(m) for m in selected_model if m in available_models]
                        
                        if indices:
                            # Extract scores for selected models
                            selected_scores = scores_matrix[:, indices]
                            
                            # Normalize if requested
                            if normalization:
                                selected_scores = normalize_scores(selected_scores, normalization)
                            
                            # Combine scores
                            if combining_method == 'mean':
                                ensemble_scores = selected_scores.mean(axis=1)
                            elif combining_method == 'max':
                                ensemble_scores = selected_scores.max(axis=1)
                            else:
                                raise ValueError(f"Unknown combining_method: {combining_method}")
                            
                            # Track ensemble combination time
                            time_ensemble_combine = time.time() - ensemble_t0
                            time_inference_total += time_ensemble_combine

                            # Calculate AP
                            selected_ap = average_precision_score(y_true, ensemble_scores)
                            selected_ap = float(_round_if_enabled(selected_ap))
                            metrics['ap'] = selected_ap

                            # Calculate ROC-AUC
                            add_metrics = calculate_additional_metrics(y_true, ensemble_scores)
                            metrics.update(add_metrics)

                            # Calculate Rank
                            all_aps_with_ensemble = all_aps + [selected_ap]
                            rank = get_rank_from_scores(all_aps_with_ensemble, len(all_aps_with_ensemble)-1)
                            
                            # Calculate Gains
                            gain_vs_primary = selected_ap - primary_ap
                            
                        else:
                            selected_ap = 0.0
                            rank = len(available_models) # Worst rank
                    except Exception as e:
                        print(f"Error evaluating ensemble for {dataset_name}: {e}")
                        selected_ap = 0.0
                        rank = -1.0
                        display_name = "Error"
                
        else:
            # Single Model Selection
            # ----------------------
            selected_ap = model_ap_map.get(selected_model, 0.0)
            metrics['ap'] = selected_ap

            primary_model_name = selected_model
            primary_ap = selected_ap
            
            # Calculate Additional Metrics if scores are available
            if scores_matrix is not None and y_true is not None:
                try:
                    # Try to find in available_models (selection pool)
                    p_idx = available_models.index(selected_model)
                    p_scores = scores_matrix[:, p_idx]
                except ValueError:
                    # Model not in selection pool - load from score_loader instead
                    try:
                        if score_loader:
                            full_scores, _ = score_loader(dataset_name)
                            # Find model in results_df to get its index in full scores
                            all_models_in_df = results_df[results_df['dataset'] == dataset_name]['model'].unique().tolist()
                            if selected_model in all_models_in_df:
                                model_idx = all_models_in_df.index(selected_model)
                                p_scores = full_scores[:, model_idx]
                            else:
                                p_scores = None
                        else:
                            p_scores = None
                    except Exception:
                        p_scores = None
                
                if p_scores is not None:
                    if normalization:
                        p_scores_2d = p_scores.reshape(-1, 1)
                        p_scores = normalize_scores(p_scores_2d, normalization).flatten()

                    add_metrics = calculate_additional_metrics(y_true, p_scores)
                    metrics.update(add_metrics)

            # Calculate rank: compare model's AP against selection pool APs
            try:
                p_idx = available_models.index(selected_model)
                rank = get_rank_from_scores(all_aps, p_idx)
                primary_rank = rank
            except ValueError:
                all_aps_with_model = all_aps + [selected_ap]
                rank = get_rank_from_scores(all_aps_with_model, len(all_aps))
                primary_rank = rank

            display_name = str(selected_model)
            gain_vs_primary = 0.0

        # Rank by AP
        metric_ranks = {'rank_ap': rank}

        # Construct result entry
        result_entry = {
            'dataset': dataset_name,
            'model': str(selected_model),
            'display_name': display_name,
            'primary_model': str(primary_model_name),
            'primary_ap': primary_ap,
            'primary_rank': primary_rank,
            'gain_vs_primary': gain_vs_primary,
            'time_fit': time_fit,
            'time_select': time_select,
            'time_inference_total': time_inference_total,
            'time_total': time.time() - fold_t0
        }
        result_entry.update(metrics)
        result_entry.update(metric_ranks)
        final_results.append(result_entry)
        if verbose:
            print(
                f"[{ds_i+1:>3d}/{len(available_datasets):<3d}] dataset={dataset_name} | fold_done (elapsed_s={time.time()-fold_t0:.2f})",
                flush=True,
            )
        
    # -------------------------------------------------------------------------
    # Print Results Table (After collecting all results to know if K>1)
    # -------------------------------------------------------------------------
    if verbose:
        if is_ensemble_run:
            # K > 1 Format
            header = f"{'Dataset':<20} {'Selected Models':<50} {'Pri AP':<8} {'Pri Rank':<8} {'Ens AP':<8} {'Ens Rank':<8} {'Gain(Pri)':<10}"
            print(header)
            print("-" * 126)
            for res in final_results:
                d_name = res['display_name']
                # Truncate if too long
                if len(d_name) > 48: d_name = d_name[:46] + ".."
                
                print(f"{res['dataset']:<20} {d_name:<50} {res['primary_ap']:<8.4f} {res['primary_rank']:<8.2f} {res['ap']:<8.4f} {res['rank_ap']:<8.2f} {res['gain_vs_primary']:<+10.4f}")
        else:
            # K = 1 Format
            header = f"{'Dataset':<20} {'Selected Model':<40} {'AP':<8} {'Rank':<6}"
            print(header)
            print("-" * 80)
            for res in final_results:
                d_name = res['display_name']
                if len(d_name) > 38: d_name = d_name[:36] + ".."
                print(f"{res['dataset']:<20} {d_name:<40} {res['ap']:<8.4f} {res['rank_ap']:<6.2f}")
        
        print("-" * 126)

    if verbose:
        print("-" * 120)

    # Summary
    df_res = pd.DataFrame(final_results)
    if not df_res.empty:
        # Calculate mean metrics (only for columns that exist)
        def _mean_if_exists(col: str) -> float:
            return df_res[col].mean() if col in df_res.columns else float('nan')

        avg_metrics = {
            'ap': _mean_if_exists('ap'),
            'roc_auc': _mean_if_exists('roc_auc'),
            'median_rank_ap': df_res['rank_ap'].median() if 'rank_ap' in df_res.columns else float('nan'),
        }

        if verbose:
            print("\n" + "="*40)
            print("FINAL SUMMARY")
            print("="*40)
            print(f"Evaluated Datasets: {len(df_res)}")
            print("-" * 65)
            print(f"{'Metric':<20} {'Average Score':<15}")
            print("-" * 65)
            print(f"{'AP':<20} {avg_metrics['ap']:<15.4f}")
            print(f"{'ROC-AUC':<20} {avg_metrics['roc_auc']:<15.4f}")
            print(f"{'Median AP Rank':<20} {avg_metrics['median_rank_ap']:<15.2f}")
            print("-" * 65)
            
            print(f"Time Taken:         {time.time() - start_time:.2f}s")
            print("-" * 65)
            print("TIMING BREAKDOWN (avg per dataset)")
            print("-" * 65)
            if 'time_fit' in df_res.columns:
                print(f"Fit Time (Training):      {df_res['time_fit'].mean():.4f}s")
            if 'time_select' in df_res.columns:
                print(f"Select Time (Inference):  {df_res['time_select'].mean():.4f}s")
            if 'time_inference_total' in df_res.columns:
                print(f"Total Inference Time:     {df_res['time_inference_total'].mean():.4f}s")
            if 'time_total' in df_res.columns:
                print(f"Total per Dataset:        {df_res['time_total'].mean():.4f}s")
            print("="*40)
            
    return df_res


def compute_results_dataframe(
    process_dataset_func: Callable,
    available_datasets: List[str],
    available_models: List[str],
    intermediate_files_folder: str = 'intermediate_files',
    verbose: bool = True
) -> pd.DataFrame:
    """
    Computes results DataFrame by processing all datasets in parallel using multiprocessing.

    Args:
        process_dataset_func: Function that takes (dataset_name, available_models, intermediate_files_folder)
                             and returns a list of result dictionaries
        available_datasets: List of dataset names to process
        available_models: List of model names to evaluate
        intermediate_files_folder: Which intermediate files folder to use (default: 'intermediate_files')
        verbose: Whether to print progress messages

    Returns:
        DataFrame containing metrics for all model-dataset pairs
    """
    if verbose:
        print("\n" + "="*80)
        print("Computing metrics for all models on all datasets...")
        print(f"Using: {intermediate_files_folder}")
        print("="*80)

    # Use configurable CPU cores for parallel processing.
    # Default is mp.cpu_count(), but you can override with N_JOBS to avoid overload/hangs.
    num_cores = mp.cpu_count()
    try:
        v = str(os.environ.get("N_JOBS", "")).strip()
        if v:
            num_cores = int(v)
    except Exception:
        pass
    num_cores = max(1, int(num_cores))
    process_func = partial(process_dataset_func,
                          available_models=available_models,
                          intermediate_files_folder=intermediate_files_folder)

    # Use imap_unordered for progress visibility.
    results_nested: List[List[Dict[str, Any]]] = []
    t0 = time.time()
    with mp.Pool(processes=num_cores) as pool:
        it = pool.imap_unordered(process_func, available_datasets)
        last_print = 0.0
        done = 0
        total = len(available_datasets)
        for res in it:
            results_nested.append(res)
            done += 1
            if verbose:
                now = time.time()
                # Print at most once per ~2s or on completion
                if (now - last_print) >= 2.0 or done == total:
                    rate = done / max(1e-9, (now - t0))
                    print(f"[metrics] done {done}/{total} datasets (workers={num_cores}, {rate:.2f} ds/s)")
                    last_print = now

    # Flatten nested results
    all_results = [item for sublist in results_nested for item in sublist]

    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)

    # Guard: if nothing was computed, avoid KeyError and provide actionable info.
    if results_df.empty:
        if verbose:
            non_empty = sum(1 for sub in results_nested if sub)
            print("No metrics were computed (results are empty).")
            print(f"   - datasets attempted: {len(available_datasets)} (non-empty returns: {non_empty})")
            print(f"   - intermediate_files_folder: {intermediate_files_folder}")
            print("   - Likely causes: missing score CSVs and/or missing label files for the datasets listed.")
            print("     * scores: dataset/benchmark/<folder>/scores/<dataset>.csv (or <dataset>.mat.csv)")
            print("     * labels: dataset/benchmark/data/<dataset>_y.csv (or <dataset>.mat_y.csv)")
        # Return an empty frame WITH expected columns so downstream code can handle it cleanly.
        return pd.DataFrame(columns=["dataset", "model"])

    required_cols = {"dataset", "model"}
    missing = required_cols.difference(set(results_df.columns))
    if missing:
        # Avoid pandas KeyError later; give a helpful error with a sample of raw items.
        sample = all_results[:3]
        raise ValueError(
            "process_dataset_func returned rows that are missing required keys "
            f"{sorted(required_cols)}. Missing: {sorted(missing)}. "
            f"Got columns: {results_df.columns.tolist()}. Sample items: {sample}"
        )

    # Filter to requested datasets/models
    results_df = results_df[results_df["dataset"].isin(available_datasets)]
    results_df = results_df[results_df["model"].isin(available_models)]

    if verbose:
        print(f"Computation complete! Processed {len(all_results)} model-dataset pairs")
        print(f"   Metrics available: {results_df.columns.tolist()}")

    return results_df


