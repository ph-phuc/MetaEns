import ast
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import entropy, kurtosis, rankdata, skew
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.metrics import average_precision_score, roc_auc_score

from utils.core import BASE_PATH, create_score_loader, load_dataset_dims

def _precompute_stats(scores_df: pd.DataFrame, model_names: List[str]) -> Dict[str, Any]:
    """
    Pre-computes various statistics (kurtosis, skewness, correlations, etc.)
    once for an entire dataset.
    """
    # 1. Convert to numpy
    raw_scores = scores_df.to_numpy()
    
    # 2. Slice columns
    n_cols_to_use = min(raw_scores.shape[1], len(model_names))
    
    scores = raw_scores[:, :n_cols_to_use]
    n_samples, n_models = scores.shape
    
    # 3. Pre-compute Moments
    kurt_vals = kurtosis(scores, axis=0)
    skew_vals = skew(scores, axis=0)
    
    # 4. Pre-compute Pearson Correlation
    with np.errstate(divide='ignore', invalid='ignore'):
        corr_matrix = np.corrcoef(scores, rowvar=False)
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

    # 5. Pre-compute Spearman Rank Correlation
    ranks = np.apply_along_axis(lambda x: rankdata(x, method='average'), 0, scores)
    with np.errstate(divide='ignore', invalid='ignore'):
        rank_corr_matrix = np.corrcoef(ranks, rowvar=False)
        rank_corr_matrix = np.nan_to_num(rank_corr_matrix, nan=0.0)

    # 6. Pre-compute Top-K Boolean Mask & Jaccard Matrix
    k_10pct = max(1, int(n_samples * 0.1))
    
    # Identify top-k indices
    top_k_indices = np.argpartition(scores, -k_10pct, axis=0)[-k_10pct:, :]
    
    # Create Boolean Mask
    bool_mask = np.zeros((n_samples, n_models), dtype=bool)
    for col_idx in range(n_models):
        rows = top_k_indices[:, col_idx]
        bool_mask[rows, col_idx] = True
        
    # Matrix Multiplication for Intersection
    intersection_matrix = bool_mask.T @ bool_mask.astype(np.float32)
    
    # Calculate Jaccard Matrix
    union_matrix = (2 * k_10pct) - intersection_matrix
    with np.errstate(divide='ignore', invalid='ignore'):
        jaccard_matrix = intersection_matrix / union_matrix
        jaccard_matrix = np.nan_to_num(jaccard_matrix, nan=0.0)

    # 7. Pre-compute Normalized Scores for L1/L2 diff
    min_vals = scores.min(axis=0)
    max_vals = scores.max(axis=0)
    ranges = max_vals - min_vals
    ranges[ranges < 1e-10] = 1.0 
    norm_scores = (scores - min_vals) / ranges
    
    # 8. Pre-compute Additional Stats (Std, Entropy, Centrality)
    # Standard Deviation
    std_vals = np.std(norm_scores, axis=0)
    
    # Entropy
    entropy_vals = np.zeros(n_models)
    for i in range(n_models):
        hist, _ = np.histogram(norm_scores[:, i], bins=10, density=True)
        entropy_vals[i] = entropy(hist + 1e-9)
        
    # Centrality (Correlation with pool mean)
    pool_mean = np.mean(norm_scores, axis=1)
    centrality_vals = np.zeros(n_models)
    with np.errstate(divide='ignore', invalid='ignore'):
        for i in range(n_models):
            if std_vals[i] > 1e-9:
                c = np.corrcoef(norm_scores[:, i], pool_mean)[0, 1]
                centrality_vals[i] = 0.0 if np.isnan(c) else c
    
    return {
        'scores': scores,
        'norm_scores': norm_scores,
        'kurtosis': kurt_vals,
        'skewness': skew_vals,
        'std': std_vals,
        'entropy': entropy_vals,
        'centrality': centrality_vals,
        'corr_matrix': corr_matrix,
        'rank_corr_matrix': rank_corr_matrix,
        'jaccard_matrix': jaccard_matrix,
        'top_k_indices': top_k_indices,
        'n_models': n_models
    }

def _compute_pair_features(stats: Dict[str, Any], 
                            p_idx: int, q_idx: int, 
                            p_name: str, q_name: str, 
                            n_samples: int, n_features: int) -> Dict[str, float]:
    """
    Extracts features for a specific pair of models using precomputed stats.
    """
    # Fast Lookups
    pearson = stats['corr_matrix'][p_idx, q_idx]
    jaccard = stats['jaccard_matrix'][p_idx, q_idx]
    
    # Tail Pearson & Disagreement (Top 10% of Primary)
    p_top_idx = stats['top_k_indices'][:, p_idx]
    q_top_idx = stats['top_k_indices'][:, q_idx]
    union_indices = np.union1d(p_top_idx, q_top_idx)
    
    tail_pearson = 0.0
    tail_pos_disagreement = 0.0
    tail_divergence = 0.0
    
    if len(union_indices) > 2:
        tail_p_scores = stats['scores'][union_indices, p_idx]
        tail_q_scores = stats['scores'][union_indices, q_idx]
        
        # Tail Pearson
        if np.std(tail_p_scores) > 1e-9 and np.std(tail_q_scores) > 1e-9:
                tail_pearson = np.corrcoef(tail_p_scores, tail_q_scores)[0, 1]
        if np.isnan(tail_pearson): tail_pearson = 0.0
        
        # Tail Disagreement / Divergence
        tail_p_norm = stats['norm_scores'][union_indices, p_idx]
        tail_q_norm = stats['norm_scores'][union_indices, q_idx]
        
        diffs = tail_q_norm - tail_p_norm
        tail_pos_disagreement = np.mean(np.maximum(0, diffs))
        tail_divergence = np.mean(np.abs(diffs))

    # Cosine Distance
    p_vec = stats['norm_scores'][:, p_idx]
    q_vec = stats['norm_scores'][:, q_idx]
    dot_prod = np.dot(p_vec, q_vec)
    norm_prod = np.linalg.norm(p_vec) * np.linalg.norm(q_vec)
    cosine_dist = 1 - (dot_prod / (norm_prod + 1e-9))
    
    # Score Dist L2 (Sorted)
    score_dist_l2 = np.linalg.norm(np.sort(p_vec) - np.sort(q_vec))

    # Moments
    ref_kurt = stats['kurtosis'][p_idx]
    if ref_kurt < 1e-3: ref_kurt = 1e-3
    rel_kurtosis = stats['kurtosis'][q_idx] / ref_kurt
    
    # Pseudo AP/ROC (Proxy Objective)
    pseudo_y_true = np.zeros(n_samples)
    pseudo_y_true[p_top_idx] = 1
    
    try:
        pseudo_ap = average_precision_score(pseudo_y_true, q_vec)
        pseudo_roc = roc_auc_score(pseudo_y_true, q_vec)
    except Exception:
        pseudo_ap = 0.5
        pseudo_roc = 0.5

    # Family Check
    fam1 = p_name.split('_')[0] if '_' in p_name else p_name
    fam2 = q_name.split('_')[0] if '_' in q_name else q_name
    is_same_family = 1.0 if fam1 == fam2 else 0.0

    # Additional Features
    spearman = stats['rank_corr_matrix'][p_idx, q_idx]
    
    diff = stats['norm_scores'][:, p_idx] - stats['norm_scores'][:, q_idx]
    l1_diff = np.mean(np.abs(diff))
    l2_diff = np.sqrt(np.mean(diff**2))

    kurt_diff = stats['kurtosis'][q_idx] - stats['kurtosis'][p_idx]
    skew_diff = stats['skewness'][q_idx] - stats['skewness'][p_idx]

    return {
        'pearson': pearson,
        'tail_pearson': tail_pearson,
        'jaccard': jaccard,
        'rel_kurtosis': rel_kurtosis,
        'tail_pos_disagreement': tail_pos_disagreement,
        'centrality': stats['centrality'][q_idx],
        'cand_std': stats['std'][q_idx],
        'pseudo_ap': pseudo_ap,
        'pseudo_roc': pseudo_roc,
        'tail_entropy': stats['entropy'][q_idx],
        'score_dist_l2': score_dist_l2,
        'cosine_dist': cosine_dist,
        'tail_divergence': tail_divergence,
        'same_family': is_same_family,
        'same_as_ensemble_count': 0, 
        
        # The following keys are not used by the default meta-model feature set
        # but are available for downstream analysis.
        'spearman_corr': spearman,
        'l1_diff': l1_diff,
        'l2_diff': l2_diff,
        'kurtosis_diff': kurt_diff,
        'skewness_diff': skew_diff,
        'n_samples': n_samples,
        'n_features': n_features
    }


def _model_family(model_name: Optional[str]) -> Optional[str]:
    """Extract model family from model name."""
    if model_name is None:
        return None
    s = str(model_name).strip()
    if not s:
        return None
    if s.startswith("(") and s.endswith(")"):
        try:
            val = ast.literal_eval(s)
            if isinstance(val, tuple) and len(val) >= 1 and isinstance(val[0], str):
                return val[0]
        except Exception:
            pass
    if "_" in s:
        return s.split("_", 1)[0]
    return s


def _read_lines(path: str) -> List[str]:
    """Read lines from a file."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def generate_meta_cache(
    cache_path: str,
    intermediate_files_folder: str = "intermediate_files",
    selection_pool_folder: str = "intermediate_files",
    datasets_file: Optional[str] = None,
    models_file: Optional[str] = None,
    train_n_primary: Optional[int] = None,
    single_train_cand_topm: Optional[int] = None,
    verbose: bool = True,
) -> bool:
    """
    Build and save the meta-feature training cache to ``cache_path``.

    Computes pairwise (primary, candidate) gain examples for every dataset
    in ``datasets_file``, then saves a compressed ``.npz`` archive that
    ``MetaEnsSelector.fit`` loads at training time.

    Parameters
    ----------
    cache_path : str
        Destination path for the ``.npz`` cache file.
    intermediate_files_folder : str
        Sub-folder under ``datasets/benchmark/`` containing ``scores/``,
        ``datasets.txt``, and ``models.txt``.
    selection_pool_folder : str
        Sub-folder containing ``models.txt`` for the selection pool
        (defaults to ``intermediate_files_folder``).
    datasets_file : str, optional
        Override path for the dataset-list file.
    models_file : str, optional
        Override path for the model-list file.
    train_n_primary : int, optional
        Number of oracle primary models per dataset used during training
        (default: 5).
    single_train_cand_topm : int, optional
        Maximum candidate pool size for marginal-gain rollout rows
        (default: 120).
    verbose : bool
        Print progress messages (default: True).

    Returns
    -------
    bool
        ``True`` on success, ``False`` if an exception occurred.
    """
    # Check if cache already exists
    if os.path.exists(cache_path):
        if verbose:
            print(f"Cache file already exists: {cache_path}")
        return True
    
    if verbose:
        print(f"Cache file not found: {cache_path}")
        print("Generating cache file... This may take several minutes.")
        print("=" * 80)
    
    try:
        # Setup paths
        default_datasets = os.path.join(BASE_PATH, intermediate_files_folder, "datasets.txt")
        default_models = os.path.join(BASE_PATH, selection_pool_folder, "models.txt")
        
        datasets_path = datasets_file or default_datasets
        models_path = models_file or default_models
        
        all_datasets = _read_lines(datasets_path)
        models = _read_lines(models_path)
        
        if verbose:
            print(f"Loaded {len(all_datasets)} datasets and {len(models)} models")
        
        # Default hyperparameters for meta-cache generation
        seed = 42
        np.random.seed(seed)
        
        if train_n_primary is None:
            train_n_primary = 5
        
        # Need at least 2 models (1 primary + 1 candidate); cap at half the pool.
        train_n_primary = min(train_n_primary, max(1, len(models) // 2))
        
        if single_train_cand_topm is None:
            single_train_cand_topm = 120
        
        # Clip topm to number of available candidate models
        single_train_cand_topm = min(single_train_cand_topm, len(models))
        
        single_train_max_set = 2
        gain_clip = 0.1
        
        # Feature set: extended (20 base features + 5 primary stats)
        feature_cols = [
            "pearson", "tail_pearson", "jaccard", "rel_kurtosis", "tail_pos_disagreement",
            "centrality", "cand_std", "pseudo_ap", "pseudo_roc", "tail_entropy",
            "score_dist_l2", "cosine_dist", "tail_divergence", "same_family", "same_as_ensemble_count",
            "prim_std", "prim_entropy", "prim_centrality", "prim_kurtosis", "prim_skewness",
        ]
        
        score_loader = create_score_loader(intermediate_files_folder)
        SCORE_CLIP = 1e6
        
        # Cache per dataset
        cache_raw: Dict[str, np.ndarray] = {}
        cache_norm: Dict[str, np.ndarray] = {}
        cache_y: Dict[str, np.ndarray] = {}
        cache_dims: Dict[str, Tuple[int, int]] = {}
        cache_stats: Dict[str, Dict[str, object]] = {}
        cache_oracle_primary_idxs: Dict[str, List[int]] = {}
        
        def get_dataset_cached(dname: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[int, int], Dict[str, object]]:
            if dname in cache_raw:
                return cache_raw[dname], cache_norm[dname], cache_y[dname], cache_dims[dname], cache_stats[dname]
            
            scores, y = score_loader(dname)
            m = min(scores.shape[1], len(models))
            scores = scores[:, :m]
            scores = np.nan_to_num(scores, nan=0.0, posinf=SCORE_CLIP, neginf=-SCORE_CLIP)
            scores = np.clip(scores, -SCORE_CLIP, SCORE_CLIP)
            
            # Normalize scores (minmax per column)
            norm_scores = np.zeros_like(scores, dtype=float)
            for j in range(scores.shape[1]):
                col = scores[:, j]
                col_min = np.min(col)
                col_max = np.max(col)
                if col_max > col_min:
                    norm_scores[:, j] = (col - col_min) / (col_max - col_min)
                else:
                    norm_scores[:, j] = 0.0
            norm_scores = np.nan_to_num(norm_scores, nan=0.0, posinf=1e9, neginf=-1e9)
            
            dims = load_dataset_dims(dname)
            # Fallback: X-file may not exist (e.g. in-memory API usage); use score matrix rows
            if dims[0] == 0:
                dims = (scores.shape[0], dims[1])
            stats = _precompute_stats(pd.DataFrame(scores), models[:m])
            
            cache_raw[dname] = scores
            cache_norm[dname] = norm_scores
            cache_y[dname] = y
            cache_dims[dname] = dims
            cache_stats[dname] = stats
            return scores, norm_scores, y, dims, stats
        
        def get_train_primary_indices(dname: str) -> List[int]:
            """Oracle mode: top-5 best single models on that dataset."""
            if dname in cache_oracle_primary_idxs:
                return cache_oracle_primary_idxs[dname][:train_n_primary]
            
            scores_raw, scores_norm, y_true, _, _ = get_dataset_cached(dname)
            m = scores_raw.shape[1]
            if m <= 0:
                return []
            
            aps = []
            for j in range(m):
                try:
                    ap = float(average_precision_score(y_true, scores_norm[:, j]))
                except Exception:
                    ap = 0.0
                aps.append((ap, j))
            aps.sort(reverse=True, key=lambda t: t[0])
            top = [j for _, j in aps[:train_n_primary]]
            cache_oracle_primary_idxs[dname] = top
            return top
        
        def enrich_feats_with_primary_stats(stats: Dict[str, object], p_idx: int, feats: Dict[str, float]) -> Dict[str, float]:
            """Add primary-context features."""
            out = dict(feats)
            try:
                out["prim_std"] = float(stats["std"][p_idx])
            except Exception:
                out["prim_std"] = 0.0
            try:
                out["prim_entropy"] = float(stats["entropy"][p_idx])
            except Exception:
                out["prim_entropy"] = 0.0
            try:
                out["prim_centrality"] = float(stats["centrality"][p_idx])
            except Exception:
                out["prim_centrality"] = 0.0
            try:
                out["prim_kurtosis"] = float(stats["kurtosis"][p_idx])
            except Exception:
                out["prim_kurtosis"] = 0.0
            try:
                out["prim_skewness"] = float(stats["skewness"][p_idx])
            except Exception:
                out["prim_skewness"] = 0.0
            return out
        
        # Generate training examples
        if verbose:
            print("Precomputing train meta-table (in-memory): mode=oracle n_primary=5 feature_set=extended ...")
        
        t0_meta = time.time()
        X_rows_all: List[List[float]] = []
        y_rows_all: List[float] = []
        ds_rows_all: List[str] = []
        pf_rows_all: List[object] = []
        cf_rows_all: List[object] = []
        
        for d2_i, d2 in enumerate(all_datasets):
            if verbose and (d2_i % 5 == 0 or d2_i == len(all_datasets) - 1):
                print(f"  Progress: {d2_i + 1}/{len(all_datasets)} datasets processed...")
            
            scores2_raw, scores2_norm, y2, (n_samples, n_features), stats = get_dataset_cached(d2)
            m = scores2_raw.shape[1]
            if m <= 1:
                continue
            
            # Similarity matrix for set-summary features
            sim_mat_r = np.asarray(stats["jaccard_matrix"], dtype=float)
            
            # In-memory per-dataset cache: p_idx -> (m, F) pair-features
            feats_mat_cache: Dict[int, np.ndarray] = {}
            
            def _get_feats_mat_for_p(p_idx_local: int, p_name_local: str) -> np.ndarray:
                p_idx_local = int(p_idx_local)
                if p_idx_local in feats_mat_cache:
                    return feats_mat_cache[p_idx_local]
                mat = np.zeros((int(m), int(len(feature_cols))), dtype=float)
                for qj in range(int(m)):
                    if int(qj) == int(p_idx_local):
                        continue
                    feats = _compute_pair_features(
                        stats=stats,
                        p_idx=int(p_idx_local),
                        q_idx=int(qj),
                        p_name=str(p_name_local),
                        q_name=models[int(qj)],
                        n_samples=int(n_samples),
                        n_features=int(n_features),
                    )
                    feats = enrich_feats_with_primary_stats(stats, int(p_idx_local), feats)
                    mat[int(qj), :] = np.asarray([float(feats.get(c, 0.0)) for c in feature_cols], dtype=float)
                mat = np.nan_to_num(mat, nan=0.0, posinf=1e9, neginf=-1e9)
                feats_mat_cache[p_idx_local] = mat
                return mat
            
            p2_idxs = get_train_primary_indices(d2)
            p2_idxs = [pi for pi in p2_idxs if isinstance(pi, int) and 0 <= pi < m]
            if not p2_idxs:
                continue
            
            for p2_idx in p2_idxs:
                p2_name = models[p2_idx]
                p2_fam = _model_family(str(p2_name))
                f_base = int(len(feature_cols))
                zeros_f = np.zeros((f_base,), dtype=float)
                
                ap_p = float(average_precision_score(y2, scores2_norm[:, p2_idx]))
                s_gain_list: List[Tuple[float, int]] = []
                feats_mat_p = _get_feats_mat_for_p(int(p2_idx), str(p2_name))
                
                # Generate step1 rows (set_size=0)
                for q_idx in range(m):
                    if q_idx == p2_idx:
                        continue
                    ens_scores = 0.5 * (scores2_norm[:, p2_idx] + scores2_norm[:, q_idx])
                    ens_scores = np.nan_to_num(ens_scores, nan=0.0, posinf=1e9, neginf=-1e9)
                    ap_e = float(average_precision_score(y2, ens_scores))
                    g = float(ap_e - ap_p)
                    s_gain_list.append((float(g), int(q_idx)))
                    
                    pq_vec = np.asarray(feats_mat_p[int(q_idx), :], dtype=float)
                    feats_vec = (
                        np.concatenate([pq_vec, zeros_f, zeros_f, np.asarray([0.0, 0.0, 0.0], dtype=float)], axis=0)
                        .astype(float)
                        .tolist()
                    )
                    
                    yv = float(np.clip(g, -gain_clip, gain_clip))
                    X_rows_all.append(feats_vec)
                    y_rows_all.append(yv)
                    ds_rows_all.append(d2)
                    pf_rows_all.append(p2_fam)
                    cf_rows_all.append(_model_family(str(models[int(q_idx)])))
        
        # Second pass: model rollout policy for marginal rows
        if verbose:
            print("single_rollout(model): fitting step1 policy on set_size=0 rows...")
        
        X_step1 = np.asarray(X_rows_all, dtype=float)
        y_step1 = np.asarray(y_rows_all, dtype=float)
        X_step1 = np.nan_to_num(X_step1, nan=0.0, posinf=1e9, neginf=-1e9)
        
        # Fit step1 model using expected_gain objective (two-stage: classifier + regressor)
        f_base = int(len(feature_cols))
        set_size_col = 3 * f_base
        ss = np.asarray(X_step1[:, int(set_size_col)], dtype=float)
        mask0 = ss <= 0.0
        
        X_train_step1 = X_step1[mask0]
        y_train_step1 = y_step1[mask0]
        
        # Stage 1: Classifier for P(gain > 0)
        y_succ = (y_train_step1 > 0.0).astype(int)
        clf = ExtraTreesClassifier(
            n_estimators=500,
            random_state=seed,
            n_jobs=-1,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
        )
        clf.fit(X_train_step1, y_succ)
        
        # Stage 2: Regressor for E[gain | gain > 0]
        pos_mask = y_succ == 1
        reg = ExtraTreesRegressor(
            n_estimators=800,
            random_state=seed,
            n_jobs=-1,
            max_depth=None,
            min_samples_leaf=2,
        )
        
        if int(np.sum(pos_mask)) < 50:
            # Not enough positive examples, fallback to simple regressor
            reg.fit(X_train_step1, y_train_step1)
            rollout_model = reg
        else:
            # Fit regressor on positive examples only
            y_pos = np.maximum(0.0, y_train_step1[pos_mask])
            X_pos = X_train_step1[pos_mask]
            reg.fit(X_pos, y_pos)
            rollout_model = {"clf": clf, "reg": reg}
        
        if verbose:
            print("single_rollout(model): generating marginal rows conditioned on model-picked partner1...")
        
        added_rows = 0
        for d2_i, d2 in enumerate(all_datasets):
            scores2_raw, scores2_norm, y2, (n_samples, n_features), stats = get_dataset_cached(d2)
            m = scores2_raw.shape[1]
            if m <= 1:
                continue
            
            sim_mat_r = np.asarray(stats["jaccard_matrix"], dtype=float)
            feats_mat_cache2: Dict[int, np.ndarray] = {}
            
            def _get_feats_mat_for_p2(p_idx_local: int, p_name_local: str) -> np.ndarray:
                p_idx_local = int(p_idx_local)
                if p_idx_local in feats_mat_cache2:
                    return feats_mat_cache2[p_idx_local]
                mat = np.zeros((int(m), int(len(feature_cols))), dtype=float)
                for qj in range(int(m)):
                    if int(qj) == int(p_idx_local):
                        continue
                    feats = _compute_pair_features(
                        stats=stats,
                        p_idx=int(p_idx_local),
                        q_idx=int(qj),
                        p_name=str(p_name_local),
                        q_name=models[int(qj)],
                        n_samples=int(n_samples),
                        n_features=int(n_features),
                    )
                    feats = enrich_feats_with_primary_stats(stats, int(p_idx_local), feats)
                    mat[int(qj), :] = np.asarray([float(feats.get(c, 0.0)) for c in feature_cols], dtype=float)
                mat = np.nan_to_num(mat, nan=0.0, posinf=1e9, neginf=-1e9)
                feats_mat_cache2[p_idx_local] = mat
                return mat
            
            p2_idxs = get_train_primary_indices(d2)
            p2_idxs = [pi for pi in p2_idxs if isinstance(pi, int) and 0 <= pi < m]
            if not p2_idxs:
                continue
            
            for p2_idx in p2_idxs:
                p2_name = models[int(p2_idx)]
                p2_fam = _model_family(str(p2_name))
                ap_p = float(average_precision_score(y2, scores2_norm[:, p2_idx]))
                feats_mat_p = _get_feats_mat_for_p2(int(p2_idx), str(p2_name))
                
                # Use step1 policy to pick partner1
                cand_q = [q for q in range(int(m)) if int(q) != int(p2_idx)]
                pq_block = np.asarray([feats_mat_p[int(q), :] for q in cand_q], dtype=float)
                zeros_f = np.zeros((int(len(feature_cols)),), dtype=float)
                X0 = np.concatenate(
                    [
                        pq_block,
                        np.repeat(zeros_f[None, :], repeats=len(cand_q), axis=0),
                        np.repeat(zeros_f[None, :], repeats=len(cand_q), axis=0),
                        np.zeros((len(cand_q), 3), dtype=float),
                    ],
                    axis=1,
                )
                X0 = np.nan_to_num(X0, nan=0.0, posinf=1e9, neginf=-1e9)
                
                # Predict using two-stage expected_gain model
                if isinstance(rollout_model, dict):
                    # Two-stage: P(gain > 0) * E[gain | gain > 0]
                    clf = rollout_model["clf"]
                    reg = rollout_model["reg"]
                    p = np.asarray(clf.predict_proba(X0)[:, 1], dtype=float)
                    r = np.asarray(reg.predict(X0), dtype=float)
                    r = np.maximum(0.0, r)
                    preds0 = p * r
                else:
                    # Simple regressor
                    preds0 = np.asarray(rollout_model.predict(X0), dtype=float)
                
                best_local = int(np.argmax(preds0)) if len(preds0) else 0
                s1_idx = int(cand_q[best_local]) if cand_q else None
                if s1_idx is None:
                    continue
                if float(preds0[best_local]) < 0.0:
                    continue
                
                feats_mat_s1 = _get_feats_mat_for_p2(int(s1_idx), str(models[int(s1_idx)]))
                scores_ps = 0.5 * (scores2_norm[:, p2_idx] + scores2_norm[:, int(s1_idx)])
                scores_ps = np.nan_to_num(scores_ps, nan=0.0, posinf=1e9, neginf=-1e9)
                ap_ps = float(average_precision_score(y2, scores_ps))
                
                # Restrict candidate pool to top-120 by true gain vs primary
                cand_pool = [q for q in range(int(m)) if int(q) != int(p2_idx) and int(q) != int(s1_idx)]
                if int(single_train_cand_topm) > 0:
                    gains_vs_p = []
                    for q in cand_pool:
                        ens = 0.5 * (scores2_norm[:, p2_idx] + scores2_norm[:, q])
                        ens = np.nan_to_num(ens, nan=0.0, posinf=1e9, neginf=-1e9)
                        ap_e = float(average_precision_score(y2, ens))
                        gains_vs_p.append((float(ap_e - ap_p), int(q)))
                    gains_vs_p.sort(reverse=True, key=lambda t: t[0])
                    cand_pool = [q for _, q in gains_vs_p[: int(single_train_cand_topm)]]
                
                # Generalized rollouts: generate marginal rows for set_size=1..2
                selected_s: List[int] = [int(s1_idx)]
                feats_mat_by_s: Dict[int, np.ndarray] = {int(s1_idx): feats_mat_s1}
                cur_sum = np.asarray(scores2_norm[:, p2_idx], dtype=float) + np.asarray(scores2_norm[:, int(s1_idx)], dtype=float)
                cur_ap = float(ap_ps)
                
                for set_k in range(1, int(single_train_max_set) + 1):
                    p_sel_vec = np.mean(
                        np.vstack([np.asarray(feats_mat_p[int(s), :], dtype=float) for s in selected_s]),
                        axis=0,
                    )
                    mats = [np.asarray(feats_mat_by_s[int(s)], dtype=float) for s in selected_s if int(s) in feats_mat_by_s]
                    g_by_q: Dict[int, float] = {}
                    
                    for q2 in cand_pool:
                        q2 = int(q2)
                        if q2 == int(p2_idx) or q2 in selected_s:
                            continue
                        pq_vec = np.asarray(feats_mat_p[int(q2), :], dtype=float)
                        if mats:
                            sel_q_vec = np.mean(np.stack([m[int(q2), :] for m in mats], axis=0), axis=0)
                        else:
                            sel_q_vec = np.zeros_like(pq_vec)
                        
                        sims = []
                        if sim_mat_r is not None and selected_s:
                            for s in selected_s:
                                try:
                                    sims.append(float(sim_mat_r[int(q2), int(s)]))
                                except Exception:
                                    sims.append(0.0)
                        sim_max = float(np.max(sims)) if sims else 0.0
                        sim_mean = float(np.mean(sims)) if sims else 0.0
                        feats_vec = np.concatenate(
                            [pq_vec, p_sel_vec, sel_q_vec, np.asarray([float(set_k), sim_max, sim_mean], dtype=float)], axis=0
                        ).astype(float)
                        
                        # True marginal gain
                        ens_scores = (cur_sum + scores2_norm[:, q2]) / float(len(selected_s) + 2)
                        ens_scores = np.nan_to_num(ens_scores, nan=0.0, posinf=1e9, neginf=-1e9)
                        ap_e = float(average_precision_score(y2, ens_scores))
                        g = float(ap_e - float(cur_ap))
                        g_by_q[int(q2)] = float(g)
                        
                        X_rows_all.append(feats_vec.tolist())
                        y_rows_all.append(float(np.clip(g, -gain_clip, gain_clip)))
                        ds_rows_all.append(d2)
                        pf_rows_all.append(p2_fam)
                        cf_rows_all.append(_model_family(str(models[int(q2)])))
                        added_rows += 1
                    
                    if not g_by_q:
                        break
                    best_next = max(g_by_q.items(), key=lambda kv: float(kv[1]))[0]
                    if float(g_by_q.get(int(best_next), -1e9)) <= 0.0:
                        break
                    
                    # Extend selected set
                    selected_s.append(int(best_next))
                    feats_mat_by_s[int(best_next)] = _get_feats_mat_for_p2(int(best_next), str(models[int(best_next)]))
                    cur_sum = cur_sum + np.asarray(scores2_norm[:, int(best_next)], dtype=float)
                    cur_scores_tmp = cur_sum / float(len(selected_s) + 1)
                    cur_scores_tmp = np.nan_to_num(cur_scores_tmp, nan=0.0, posinf=1e9, neginf=-1e9)
                    cur_ap = float(average_precision_score(y2, cur_scores_tmp))
        
        if verbose:
            print(f"Generated {len(y_rows_all)} training examples")
            print(f"single_rollout(model): added_rows={int(added_rows)} (set_size>=1)")
        
        # Convert to numpy arrays
        pre_meta_X = np.asarray(X_rows_all, dtype=float)
        pre_meta_y = np.asarray(y_rows_all, dtype=float)
        pre_meta_ds = np.asarray(ds_rows_all, dtype=object)
        pre_meta_pf = np.asarray(pf_rows_all, dtype=object)
        pre_meta_cf = np.asarray(cf_rows_all, dtype=object)
        pre_meta_X = np.nan_to_num(pre_meta_X, nan=0.0, posinf=1e9, neginf=-1e9)
        
        if verbose:
            print(f"Feature dimensions: {pre_meta_X.shape[1]}")
            print(f"Precompute done: rows={int(pre_meta_X.shape[0])} elapsed_s={time.time()-t0_meta:.2f}")
        
        # Save cache
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        meta = {
            "models_sig": "public_v1",
            "feature_set": "extended",
            "train_primary_mode": "oracle",
            "train_n_primary": 5,
            "task": "regress",
            "comb": "mean",
            "gain_clip": 0.1,
            "good_threshold": 0.0,
            "rank_objective": "quantile_bins",
            "step_aware_mode": "single_model_set_summary",
            "step2_train_topk": 5,
            "redundancy_sim": "jaccard",
            "single_train_max_set": 2,
            "single_train_cand_topm": 120,
            "single_rollout_policy": "model",
            "has_family_meta": True,
        }
        meta_json = json.dumps(meta, sort_keys=True)

        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        np.savez_compressed(
            cache_path,
            meta_json=np.asarray([meta_json], dtype=object),
            X=pre_meta_X,
            y=pre_meta_y,
            ds=pre_meta_ds,
            pf=pre_meta_pf,
            cf=pre_meta_cf,
            w=np.asarray([], dtype=float),
            group_sizes=np.asarray([], dtype=int),
            group_ds=np.asarray([], dtype=object),
        )
        
        if verbose:
            print(f"Saving to: {cache_path}")
            print("=" * 80)
            print(f"Successfully generated cache file: {cache_path}")
            print(f"  Total examples: {len(y_rows_all)}")
            print(f"  Feature dimensions: {pre_meta_X.shape[1]}")
            print(f"  Datasets: {len(all_datasets)}")
            print("=" * 80)
        
        return True
        
    except Exception as e:
        print(f"ERROR: Exception during cache generation: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False
