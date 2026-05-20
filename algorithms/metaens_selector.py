
"""
MetaEnsSelector: partner-selection meta-model for MetaEns.

This module implements the two-phase ensemble-building algorithm:
  1. A meta-model (ExtraTrees classifier + regressor) is trained offline
     on pre-computed pairwise gain examples to predict the marginal AP
     gain of adding a candidate detector to a growing ensemble.
  2. At inference time, ``_select_with_scores`` iteratively proposes
     partners using staged candidate evaluation, a submodular diversity
     discount, and an Empirical Bayes family-risk prior.

Typical usage is through the high-level :class:`metaens.MetaEns` API.
Direct use is also supported for LODO benchmark evaluation via
``run_metaens.py``.
"""

import os
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.metrics import average_precision_score

from .base import SingleModelSelector
from .elect import ELECT
from utils.core import BASE_PATH, create_score_loader, load_dataset_dims
from utils.evaluation import compute_results_dataframe
from utils.metrics import compute_all_metrics_for_dataset
from algorithms.meta_features import generate_meta_cache, _precompute_stats, _compute_pair_features, _model_family


def _sim_vec_to_primary(stats: Dict[str, object], idxs: List[int], p_idx: int, metric: str) -> np.ndarray:
    metric = (metric or "jaccard").lower()
    if not idxs:
        return np.zeros((0,), dtype=float)
    if metric == "jaccard":
        v = np.asarray(stats["jaccard_matrix"], dtype=float)[idxs, p_idx]
        return np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    if metric == "pearson_abs":
        v = np.abs(np.asarray(stats["corr_matrix"], dtype=float)[idxs, p_idx])
        return np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    if metric == "spearman_abs":
        v = np.abs(np.asarray(stats["rank_corr_matrix"], dtype=float)[idxs, p_idx])
        return np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    raise ValueError(f"Unknown similarity metric: {metric}")



@dataclass(frozen=True)
class _MetaCache:
    X: np.ndarray
    y: np.ndarray
    ds: np.ndarray
    pf: Optional[np.ndarray]
    cf: Optional[np.ndarray]
    rank_df: Optional[pd.DataFrame] = None  # embedded in .npz; loaded on first access


class MetaEnsSelector(SingleModelSelector):
    """
    MetaEns partner-selection model: selects a primary detector and iteratively
    adds ensemble partners using a trained meta-model.

    Parameters
    ----------
    train_meta_cache : str
        Path to the pre-generated ``.npz`` meta-feature cache produced by
        :func:`algorithms.meta_features.generate_meta_cache`.
    intermediate_files_folder : str, default ``"intermediate_files"``
        Sub-folder under ``datasets/benchmark/`` containing ``scores/``,
        ``datasets.txt``, and ``models.txt``.
    primary_selector : SingleModelSelector
        Pre-fitted (or fittable) selector that picks the primary detector.
        Any :class:`~algorithms.base.SingleModelSelector` is accepted
        (e.g. :class:`~algorithms.elect.ELECT`).
    n_selection : int, default 0
        Maximum ensemble size including the primary.  ``0`` means adaptive
        (partners are added until the predicted gain falls below the threshold).
    stages : str, default ``"40,120,0"``
        Comma-separated staged candidate-evaluation schedule.  Each value is
        the cumulative number of candidates evaluated per step; ``0`` means
        "all remaining candidates".
    redundancy_sim : str, default ``"jaccard"``
        Similarity metric used for the submodular diversity discount.
        One of ``"jaccard"``, ``"pearson_abs"``, ``"spearman_abs"``.
    use_submodular : bool, default True
        Enable submodular diversity-discount optimisation (paper's method).
        When False, uses greedy selection.
    submodular_beta : float, default 1.0
        Saturation strength β for the submodular discount.  Higher = stronger
        diminishing returns for redundant models.
    submodular_step1_beta : float, default 0.001
        Saturation strength for the first-partner step (near-zero = soft
        diversity enforcement, prioritising prediction quality).
    submodular_lazy : bool, default True
        Use lazy greedy evaluation for the submodular selection step.
        Speeds up partner selection with no effect on results.
    lambda_family_prior : float, default 0.2
        Weight λ_fam for the family-risk prior (Empirical Bayes, §3.2 of the
        paper).  Set to 0 to disable.
    use_family_prior : bool, default True
        Enable the family-risk prior.  Requires ``pf``/``cf`` arrays in the
        meta-feature cache.
    step2_family_prior_mode : str, default ``"cand_only"``
        Prior key: ``"cand_only"`` indexes by candidate family; ``"pair"``
        indexes by (primary_family, candidate_family).
    step2_family_prior_stat : str, default ``"p10"``
        Statistic used to build the prior: ``"mean"``, ``"median"``, or
        ``"p10"`` (10th percentile — more conservative).
    step2_family_prior_min_count : int, default 800
        Minimum training-sample count required to include a bucket in the
        prior map.
    min_pred_gain : float, default 0.001
        Minimum predicted gain to add the *first* partner.
    min_pred_gain_extra : float, default 0.005
        Minimum predicted gain to add each subsequent partner.
    good_threshold : float, default 0.0
        Threshold used to split the classifier target during training
        (``y > good_threshold`` → positive class).
    single_step2_weight : float, default 2.0
        Sample weight applied to marginal (set_size > 0) training rows.
    use_two_part_model : bool, default True
        Use the two-part meta-model (ExtraTrees classifier + regressor) from
        the paper.  When False, trains a single regressor.
    seed : int, default 100
        Random seed for the meta-model.
    debug_selection : bool, default False
        Print a detailed selection trace (useful for debugging).
    debug_topk : int, default 5
        Number of top candidates shown in the debug trace.
    """

    _meta_cache_by_path: Dict[str, _MetaCache] = {}

    def _dbg(self, msg: str) -> None:
        if self.debug_selection:
            print(msg, flush=True)

    def __init__(
        self,
        *,
        train_meta_cache: str,
        intermediate_files_folder: str = "intermediate_files",
        # Primary selector — required.  Pass any SingleModelSelector instance,
        # e.g. ELECT(...) or a PyOD wrapper implementing fit/select.
        primary_selector: Optional[SingleModelSelector] = None,
        # Debugging (prints detailed selection trace when enabled)
        debug_selection: bool = False,
        debug_topk: int = 5,
        # Selection policy
        n_selection: int = 0,  # 0 => unbounded selection (stopping rules)
        stages: str = "40,120,0",
        redundancy_sim: str = "jaccard",
        # Submodular optimization
        use_submodular: bool = True,   # Enable submodular optimization
        submodular_beta: float = 1.0,  # Saturation strength (higher = more diminishing returns)
        submodular_step1_beta: Optional[float] = 0.001, # Beta for first partner selection
        submodular_lazy: bool = True,  # Use lazy greedy evaluation
        # Bayesian priors
        lambda_family_prior: float = 0.2,     # Family-risk prior (Empirical Bayes)
        # Gating / stop rules
        min_pred_gain: float = 0.001,
        min_pred_gain_extra: float = 0.005,
        good_threshold: float = 0.0,
        # Training parameters
        single_step2_weight: float = 2.0,
        seed: int = 100,
        # Family-risk prior configuration
        use_family_prior: bool = True,  # Enable/disable family-risk prior
        step2_family_prior_mode: str = "cand_only",  # "pair" or "cand_only"
        step2_family_prior_stat: str = "p10",  # "mean"|"median"|"p10"
        step2_family_prior_min_count: int = 800,
        # Advanced meta-model configuration
        use_two_part_model: bool = True,  # Use two-part (classifier + regressor) vs single regressor
        clf_class: Optional[Any] = None,
        reg_class: Optional[Any] = None,
        clf_params: Optional[Dict[str, Any]] = None,
        reg_params: Optional[Dict[str, Any]] = None,
    ):
        self.train_meta_cache = str(train_meta_cache)
        self.intermediate_files_folder = str(intermediate_files_folder)
        self.primary_selector_template = primary_selector


        self.debug_selection = bool(debug_selection)
        self.debug_topk = int(debug_topk) if int(debug_topk) > 0 else 5

        self.n_selection_requested = int(n_selection)
        self.unbounded_selection = self.n_selection_requested <= 0
        self.n_selection = max(1, int(n_selection))

        self.stages = str(stages)
        self.redundancy_sim = str(redundancy_sim or "jaccard").lower()
        
        # Submodular optimization parameters
        self.use_submodular = bool(use_submodular)
        self.submodular_beta = float(submodular_beta)
        self.submodular_step1_beta = float(submodular_step1_beta) if submodular_step1_beta is not None else None
        self.submodular_lazy = bool(submodular_lazy)
        
        self.lambda_fam = float(lambda_family_prior)

        self.min_pred_gain = float(min_pred_gain)
        self.min_pred_gain_extra = float(min_pred_gain_extra)
        self.good_threshold = float(good_threshold)
        self.single_step2_weight = float(single_step2_weight)
        self.seed = int(seed)

        self.step2_family_prior = bool(use_family_prior)
        self.step2_family_prior_mode = str(step2_family_prior_mode or "cand_only").strip().lower()
        self.step2_family_prior_stat = str(step2_family_prior_stat or "mean").strip().lower()
        self.step2_family_prior_min_count = int(step2_family_prior_min_count)

        if self.step2_family_prior_mode not in ("pair", "cand_only"):
            raise ValueError("step2_family_prior_mode must be pair|cand_only")
        if self.step2_family_prior_stat not in ("mean", "median", "p10"):
            raise ValueError("step2_family_prior_stat must be mean|median|p10")
        if self.step2_family_prior_min_count < 1:
            raise ValueError("step2_family_prior_min_count must be >= 1")
        if self.lambda_fam < 0:
            raise ValueError("lambda_family_prior must be >= 0")
        
        # Advanced meta-model configuration
        self.use_two_part_model = bool(use_two_part_model)
        self.clf_class = clf_class or ExtraTreesClassifier
        self.reg_class = reg_class or ExtraTreesRegressor
        self.clf_params = clf_params or {
            "n_estimators": 500, 
            "random_state": self.seed, 
            "n_jobs": -1,
            "max_depth": None,
            "min_samples_leaf": 2,
            "class_weight": "balanced_subsample"
        }
        self.reg_params = reg_params or {
            "n_estimators": 800, 
            "random_state": self.seed, 
            "n_jobs": -1,
            "max_depth": None,
            "min_samples_leaf": 2
        }

        # Populated in fit()
        self._models: Optional[List[str]] = None
        self._score_loader = create_score_loader(self.intermediate_files_folder)
        self._primary_selector: Optional[SingleModelSelector] = None
        self._model: Optional[Union[Any, Dict[str, Any]]] = None
        self._step2_family_penalty_map: Optional[Dict[object, float]] = None
        self._f_base: Optional[int] = None
        self._set_size_col: Optional[int] = None
        self._rank_df: Optional[pd.DataFrame] = None
        self._rank_cache: Dict[str, Tuple[Dict[str, List[str]], List[str]]] = {}

    def _compute_candidate_ranking(self, historical_df: pd.DataFrame, available_models: List[str]) -> pd.DataFrame:
        """
        Compute partner-transfer rankings: for each (target_dataset, primary, partner) triple,
        evaluate how reliably a good partner transfers across other datasets.

        Returns a DataFrame with columns:
            primary_family, partner_model, transfer_success_rate, target_dataset
        """
        # Load ALL datasets from file (not from historical_df which excludes current dataset in LOOCV)
        datasets_file = os.path.join(BASE_PATH, self.intermediate_files_folder, "datasets.txt")
        with open(datasets_file, 'r') as f:
            all_datasets = sorted([line.strip() for line in f if line.strip()])
        
        # Cache for normalized scores to avoid recomputation
        cache_norm_scores = {}
        cache_y = {}
        
        def get_norm_scores_and_y(dataset_name: str):
            """Get normalized scores and labels for a dataset with caching."""
            if dataset_name in cache_norm_scores:
                return cache_norm_scores[dataset_name], cache_y[dataset_name]
            
            try:
                scores, y = self._score_loader(dataset_name)
                m = min(scores.shape[1], len(available_models))
                scores = scores[:, :m]
                scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
                scores = np.clip(scores, -1e6, 1e6)
                
                # Minmax normalization per column
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
                
                cache_norm_scores[dataset_name] = norm_scores
                cache_y[dataset_name] = y
                return norm_scores, y
            except Exception as e:
                warnings.warn(f"Could not load dataset {dataset_name}: {e}")
                return None, None
        
        def ensemble_ap_mean_two_cols(norm_scores: np.ndarray, y: np.ndarray, p_idx: int, q_idx: int) -> float:
            """Compute AP for mean ensemble of two models."""
            try:
                ens = 0.5 * (norm_scores[:, p_idx] + norm_scores[:, q_idx])
                ens = np.nan_to_num(ens, nan=0.0, posinf=1e9, neginf=-1e9)
                return float(average_precision_score(y, ens))
            except Exception:
                return 0.0
        
        threshold = 0.01  # Good partner threshold for target dataset (gain over primary)
        
        # Create a fresh ELECT selector with ALL datasets (not LOOCV-fitted)
        # We need to compute a full results_df with all datasets for proper ELECT selection
        full_results_df = compute_results_dataframe(
            process_dataset_func=compute_all_metrics_for_dataset,
            available_datasets=all_datasets,
            available_models=available_models,
            intermediate_files_folder=self.intermediate_files_folder,
            verbose=False
        )
        
        # Create and fit a fresh ELECT selector with all datasets
        elect_selector = ELECT(n_selection=1)
        elect_selector.fit(full_results_df, available_models)
        
        # Use the fresh ELECT selector to get primary models
        primary_selections = {}
        for dataset in all_datasets:
            selected = elect_selector.select(dataset)
            if selected:
                primary_selections[dataset] = selected if isinstance(selected, str) else selected[0]
        
        results = []
        for d_i, target_dataset in enumerate(all_datasets):
            # Get primary model for this dataset
            if target_dataset not in primary_selections:
                continue
                
            primary_model = primary_selections[target_dataset]
            if primary_model not in available_models:
                continue
                
            p_idx = available_models.index(primary_model)
            primary_family = _model_family(primary_model)
            
            # Load target dataset scores
            scores_target, y_target = get_norm_scores_and_y(target_dataset)
            if scores_target is None:
                continue
            
            m = scores_target.shape[1]
            
            try:
                ap_primary = float(average_precision_score(y_target, scores_target[:, p_idx]))
            except Exception:
                ap_primary = 0.0
            
            # Find "good" partners on target dataset (gain > threshold)
            good_partners = []
            for q_idx in range(m):
                if q_idx == p_idx:
                    continue
                
                ap_ens = ensemble_ap_mean_two_cols(scores_target, y_target, p_idx, q_idx)
                gain = ap_ens - ap_primary
                
                if gain > threshold:
                    good_partners.append({
                        'q_idx': q_idx,
                        'partner_model': available_models[q_idx],
                        'gain_on_target': gain,
                        'ensemble_ap_on_target': ap_ens
                    })
            
            # Evaluate transfer of each good partner to other datasets
            for partner_info in good_partners:
                q_idx = partner_info['q_idx']
                partner_model = partner_info['partner_model']
                partner_family = _model_family(partner_model)
                same_family = (primary_family is not None and primary_family == partner_family)
                
                # Evaluate on other datasets
                other_datasets = [d for d in all_datasets if d != target_dataset]
                transfer_success = 0
                gains_other = []
                
                for other_dataset in other_datasets:
                    scores_other, y_other = get_norm_scores_and_y(other_dataset)
                    if scores_other is None or scores_other.shape[1] <= max(p_idx, q_idx):
                        continue
                    
                    try:
                        ap_p_other = float(average_precision_score(y_other, scores_other[:, p_idx]))
                    except Exception:
                        ap_p_other = 0.0
                    
                    ap_ens_other = ensemble_ap_mean_two_cols(scores_other, y_other, p_idx, q_idx)
                    gain_other = ap_ens_other - ap_p_other
                    gains_other.append(gain_other)
                    
                    if gain_other > threshold:
                        transfer_success += 1
                
                n_other = len(gains_other)
                transfer_success_rate = transfer_success / n_other if n_other > 0 else 0.0
                avg_gain_other = float(np.mean(gains_other)) if gains_other else 0.0
                median_gain_other = float(np.median(gains_other)) if gains_other else 0.0
                
                # Average/median gain on successful transfers only
                gains_success = [g for g in gains_other if g > threshold]
                avg_gain_success = float(np.mean(gains_success)) if gains_success else 0.0
                median_gain_success = float(np.median(gains_success)) if gains_success else 0.0
                
                results.append({
                    'target_dataset': target_dataset,
                    'primary_model': primary_model,
                    'primary_family': primary_family,
                    'partner_model': partner_model,
                    'partner_family': partner_family,
                    'same_family': bool(same_family),
                    'threshold': threshold,
                    'gain_on_target': partner_info['gain_on_target'],
                    'ensemble_ap_on_target': partner_info['ensemble_ap_on_target'],
                    'n_other_datasets': n_other,
                    'transfer_success_count': transfer_success,
                    'transfer_success_rate': transfer_success_rate,
                    'avg_gain_other': avg_gain_other,
                    'median_gain_other': median_gain_other,
                    'avg_gain_success_other': avg_gain_success,
                    'median_gain_success_other': median_gain_success,
                })
        
        
        df_results = pd.DataFrame(results)
        return df_results

    @staticmethod
    def _load_meta_cache(path: str) -> _MetaCache:
        if path in MetaEnsSelector._meta_cache_by_path:
            return MetaEnsSelector._meta_cache_by_path[path]
        z = np.load(path, allow_pickle=True)
        X = np.asarray(z["X"], dtype=float)
        y = np.asarray(z["y"], dtype=float)
        ds = np.asarray(z["ds"], dtype=object)
        pf = np.asarray(z["pf"], dtype=object) if "pf" in z and len(z["pf"]) else None
        cf = np.asarray(z["cf"], dtype=object) if "cf" in z and len(z["cf"]) else None
        rank_df = None
        if "rank_primary_family" in z:
            try:
                rank_df = pd.DataFrame({
                    "primary_family": np.asarray(z["rank_primary_family"], dtype=object).astype(str),
                    "partner_model": np.asarray(z["rank_partner_model"], dtype=object).astype(str),
                    "transfer_success_rate": np.asarray(z["rank_transfer_success_rate"], dtype=float),
                    "target_dataset": np.asarray(z["rank_target_dataset"], dtype=object).astype(str),
                })
            except Exception:
                rank_df = None
        cache = _MetaCache(X=X, y=y, ds=ds, pf=pf, cf=cf, rank_df=rank_df)
        MetaEnsSelector._meta_cache_by_path[path] = cache
        return cache

    @staticmethod
    def _append_ranking_to_cache(path: str, rank_df: pd.DataFrame) -> None:
        """Re-save the .npz with partner-transfer ranking arrays embedded."""
        z = np.load(path, allow_pickle=True)
        data = {k: z[k] for k in z.files}
        data["rank_primary_family"] = rank_df["primary_family"].to_numpy(dtype=object)
        data["rank_partner_model"] = rank_df["partner_model"].to_numpy(dtype=object)
        data["rank_transfer_success_rate"] = rank_df["transfer_success_rate"].to_numpy(dtype=float)
        data["rank_target_dataset"] = rank_df["target_dataset"].to_numpy(dtype=object)
        np.savez_compressed(path, **data)
        MetaEnsSelector._meta_cache_by_path.pop(path, None)

    @staticmethod
    def _feature_cols_extended() -> List[str]:
        base = [
            "pearson",
            "tail_pearson",
            "jaccard",
            "rel_kurtosis",
            "tail_pos_disagreement",
            "centrality",
            "cand_std",
            "pseudo_ap",
            "pseudo_roc",
            "tail_entropy",
            "score_dist_l2",
            "cosine_dist",
            "tail_divergence",
            "same_family",
            "same_as_ensemble_count",
        ]
        return base + ["prim_std", "prim_entropy", "prim_centrality", "prim_kurtosis", "prim_skewness"]

    def _enrich_feats_with_primary_stats(self, stats: Dict[str, object], p_idx: int, feats: Dict[str, float]) -> Dict[str, float]:
        out = dict(feats)
        try:
            out["prim_std"] = float(np.asarray(stats["std"], dtype=float)[p_idx])
        except Exception:
            out["prim_std"] = 0.0
        try:
            out["prim_entropy"] = float(np.asarray(stats["entropy"], dtype=float)[p_idx])
        except Exception:
            out["prim_entropy"] = 0.0
        try:
            out["prim_centrality"] = float(np.asarray(stats["centrality"], dtype=float)[p_idx])
        except Exception:
            out["prim_centrality"] = 0.0
        try:
            out["prim_kurtosis"] = float(np.asarray(stats["kurtosis"], dtype=float)[p_idx])
        except Exception:
            out["prim_kurtosis"] = 0.0
        try:
            out["prim_skewness"] = float(np.asarray(stats["skewness"], dtype=float)[p_idx])
        except Exception:
            out["prim_skewness"] = 0.0
        return out

    def _build_test_X1_full(
        self,
        *,
        ordered_q_idxs: List[int],
        p_idx_local: int,
        stats_local: Dict[str, object],
        n_samples_local: int,
        n_features_local: int,
        primary_name: str,
    ) -> np.ndarray:
        feature_cols = self._feature_cols_extended()
        if not ordered_q_idxs:
            return np.zeros((0, len(feature_cols)), dtype=float)
        X_rows: List[List[float]] = []
        assert self._models is not None
        for q_idx in ordered_q_idxs:
            feats = _compute_pair_features(
                stats=stats_local,
                p_idx=int(p_idx_local),
                q_idx=int(q_idx),
                p_name=str(primary_name),
                q_name=self._models[int(q_idx)],
                n_samples=int(n_samples_local),
                n_features=int(n_features_local),
            )
            feats = self._enrich_feats_with_primary_stats(stats_local, int(p_idx_local), feats)
            X_rows.append([float(feats.get(c, 0.0)) for c in feature_cols])
        X_full = np.nan_to_num(np.asarray(X_rows, dtype=float), nan=0.0, posinf=1e9, neginf=-1e9)
        return X_full

    @staticmethod
    def _build_test_X_set_summary(
        *,
        pq_block: np.ndarray,
        p_sel_mean_vec: Optional[np.ndarray],
        sel_q_mean_block: Optional[np.ndarray],
        set_size: int,
        sim_sel_max: Optional[np.ndarray],
        sim_sel_mean: Optional[np.ndarray],
    ) -> np.ndarray:
        pq_block = np.asarray(pq_block, dtype=float)
        n = int(pq_block.shape[0])
        f = int(pq_block.shape[1])
        if n <= 0:
            return np.zeros((0, 3 * f + 3), dtype=float)
        if p_sel_mean_vec is None:
            p_sel_block = np.zeros((n, f), dtype=float)
        else:
            p_sel_block = np.repeat(np.asarray(p_sel_mean_vec, dtype=float)[None, :], repeats=n, axis=0)
        if sel_q_mean_block is None:
            sel_q_block = np.zeros((n, f), dtype=float)
        else:
            sel_q_block = np.asarray(sel_q_mean_block, dtype=float)
            if sel_q_block.shape != (n, f):
                sel_q_block = np.zeros((n, f), dtype=float)
        ss = np.full((n, 1), float(set_size), dtype=float)
        if sim_sel_max is None:
            sim_mx = np.zeros((n, 1), dtype=float)
        else:
            sim_mx = np.asarray(sim_sel_max, dtype=float).reshape(n, 1)
        if sim_sel_mean is None:
            sim_mu = np.zeros((n, 1), dtype=float)
        else:
            sim_mu = np.asarray(sim_sel_mean, dtype=float).reshape(n, 1)
        X = np.concatenate([pq_block, p_sel_block, sel_q_block, ss, sim_mx, sim_mu], axis=1)
        X = np.nan_to_num(np.asarray(X, dtype=float), nan=0.0, posinf=1e9, neginf=-1e9)
        return X

    @staticmethod
    def _compute_saturation_factor(
        redundancy_to_selected: np.ndarray,
        beta: float
    ) -> np.ndarray:
        """
        Compute saturation factor γ(q,S) = 1 / (1 + β·sim_max(q,S)).
        
        This creates submodular structure by enforcing diminishing returns for redundant models.
        
        Args:
            redundancy_to_selected: Max similarity to any model in selected set [n_candidates]
            beta: Saturation strength β. Higher = stronger diminishing returns.
            
        Returns:
            Saturation factors γ ∈ (0, 1] for each candidate
        """
        redundancy_to_selected = np.asarray(redundancy_to_selected, dtype=float)
        redundancy_to_selected = np.clip(redundancy_to_selected, 0.0, 1.0)
        alpha = 1.0 / (1.0 + float(beta) * redundancy_to_selected)
        return np.asarray(alpha, dtype=float)

    @staticmethod
    def _predict_expected_gain(mdl: Union[ExtraTreesRegressor, Dict[str, Any]], X: np.ndarray) -> np.ndarray:
        X = np.nan_to_num(np.asarray(X, dtype=float), nan=0.0, posinf=1e9, neginf=-1e9)
        if isinstance(mdl, dict) and "clf" in mdl and "reg" in mdl:
            p = np.asarray(mdl["clf"].predict_proba(X)[:, 1], dtype=float)
            mag = np.asarray(mdl["reg"].predict(X), dtype=float)
            mag = np.maximum(0.0, mag)
            return p * mag
        return np.asarray(mdl.predict(X), dtype=float)

    @staticmethod
    def _predict_prob(
        mdl: Union[ExtraTreesRegressor, Dict[str, Any]], X: np.ndarray
    ) -> np.ndarray:
        """Extract predicted probability P(gain > 0) from classifier component."""
        X = np.nan_to_num(np.asarray(X, dtype=float), nan=0.0, posinf=1e9, neginf=-1e9)
        if isinstance(mdl, dict) and "clf" in mdl:
            return np.asarray(mdl["clf"].predict_proba(X)[:, 1], dtype=float)
        # No classifier component, return ones (always accept)
        return np.ones(X.shape[0], dtype=float)

    @staticmethod
    def _predict_expected_gain_mean_std(
        mdl: Union[ExtraTreesRegressor, Dict[str, Any]], X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        X = np.nan_to_num(np.asarray(X, dtype=float), nan=0.0, posinf=1e9, neginf=-1e9)

        def _tree_mean_std(reg) -> Tuple[np.ndarray, np.ndarray]:
            if hasattr(reg, "estimators_") and getattr(reg, "estimators_", None):
                preds = [np.asarray(est.predict(X), dtype=float) for est in reg.estimators_]
                mat = np.vstack(preds)
                return np.mean(mat, axis=0), np.std(mat, axis=0)
            y = np.asarray(reg.predict(X), dtype=float)
            return y, np.zeros_like(y)

        if isinstance(mdl, dict) and "clf" in mdl and "reg" in mdl:
            p = np.asarray(mdl["clf"].predict_proba(X)[:, 1], dtype=float)
            mag_mean, mag_std = _tree_mean_std(mdl["reg"])
            mag_mean = np.maximum(0.0, mag_mean)
            mag_std = np.maximum(0.0, mag_std)
            return p * mag_mean, p * mag_std

        return _tree_mean_std(mdl)

    def _submodular_select_next(
        self,
        *,
        candidate_idxs: List[int],
        selected_idxs: List[int],
        predicted_gains: np.ndarray,
        stats: Dict[str, Any],
        p_idx: int,
        use_lazy: bool = True,
    ) -> Tuple[Optional[int], float]:
        """
        Select next model using submodular proxy maximization.
        
        Diversity is handled via saturation factor γ(q,S) = 1/(1 + β·sim_max(q,S)).
        
        For step1 (first partner): A very soft saturation (submodular_step1_beta ≈ 0) is
            applied, effectively prioritising prediction quality over diversity.
        For step2+ (additional partners): Full saturation γ enforces diminishing returns.
        
        Returns (selected_idx, marginal_value) or (None, 0.0) if no valid candidate.
        """
        if not candidate_idxs:
            return None, 0.0
        
        # Compute max similarity to any member already in the set {primary} U {selected}
        reference_idxs = [int(p_idx)] + [int(i) for i in selected_idxs]
        
        max_sim_to_set = np.zeros(len(candidate_idxs), dtype=float)
        for ref_idx in reference_idxs:
            sim_to_ref = _sim_vec_to_primary(stats, candidate_idxs, ref_idx, self.redundancy_sim)
            max_sim_to_set = np.maximum(max_sim_to_set, sim_to_ref)
        
        # Compute saturation factors α(m, S) based on redundancy to primary + selected ensemble.
        # A softened β is used for the first partner to prioritize prediction quality over diversity.
        effective_beta = self.submodular_beta
        if not selected_idxs and self.submodular_step1_beta is not None:
            effective_beta = self.submodular_step1_beta
        
        alpha = self._compute_saturation_factor(max_sim_to_set, effective_beta)
        
        # Submodular objective: g(m | S) * α(m, S)
        submodular_values = predicted_gains[:len(candidate_idxs)] * alpha
        
        best_idx_local = int(np.argmax(submodular_values))
        best_value = float(submodular_values[best_idx_local])
        return candidate_idxs[best_idx_local], best_value

    def fit(self, historical_df: pd.DataFrame, available_models: List[str]):
        """
        Train the meta-model on historical (leave-one-out) data.

        Parameters
        ----------
        historical_df : DataFrame
            Columns [``dataset``, ``model``, ``ap``].  In LODO evaluation
            the current test dataset is excluded by the evaluation harness
            before this method is called.
        available_models : list of str
            Ordered pool of candidate model names.  Column order must match
            the score matrices used at selection time.
        """
        self._models = list(available_models)

        # Primary selector — must be provided via primary_selector parameter.
        # Any SingleModelSelector (e.g. ELECT, a PyOD wrapper) is accepted as long
        # as it implements fit(historical_df, models) and select(dataset_name).
        if self.primary_selector_template is None:
            raise ValueError(
                "primary_selector is required. Pass an ELECT instance (or any "
                "SingleModelSelector) when constructing MetaEnsSelector."
            )
        self._primary_selector = self.primary_selector_template
        self._primary_selector.fit(historical_df, self._models)

        # Generate meta cache if missing
        if not os.path.exists(self.train_meta_cache):
            print(f"\n{'='*80}")
            print(f"[MetaEnsSelector] Cache file not found: {self.train_meta_cache}")
            print(f"[MetaEnsSelector] Generating cache file automatically...")
            print(f"{'='*80}\n")
            
            # Extract configuration from intermediate_files_folder to pass to cache generator
            success = generate_meta_cache(
                cache_path=self.train_meta_cache,
                intermediate_files_folder=self.intermediate_files_folder,
                selection_pool_folder=self.intermediate_files_folder,
                verbose=True,
            )
            
            if not success:
                raise RuntimeError(
                    f"Failed to generate cache file: {self.train_meta_cache}. "
                    f"Please check the error messages above."
                )
            
            print(f"\n{'='*80}")
            print(f"[MetaEnsSelector] Cache generation completed successfully!")
            print(f"{'='*80}\n")

        # Load meta cache and build train split based on train datasets (LODO-safe via evaluation harness)
        meta = self._load_meta_cache(self.train_meta_cache)

        # Load or compute partner-transfer ranking (embedded in .npz; computed once then cached).
        self._rank_df = meta.rank_df
        self._rank_cache = {}
        if self._rank_df is None:
            print(f"\n{'='*80}")
            print("[MetaEnsSelector] Partner ranking not in cache — computing now (one-time, ~minutes)...")
            print(f"{'='*80}\n")
            rank_df = self._compute_candidate_ranking(historical_df, available_models)
            self._rank_df = rank_df
            self._append_ranking_to_cache(self.train_meta_cache, rank_df)
            print(f"\n{'='*80}")
            print("[MetaEnsSelector] Partner ranking embedded in cache.")
            print(f"{'='*80}\n")

        train_ds = set(pd.unique(historical_df["dataset"]).tolist())
        mask = np.isin(meta.ds.astype(str), np.asarray(sorted(train_ds), dtype=str))
        X_train = np.asarray(meta.X[mask], dtype=float)
        y_train = np.asarray(meta.y[mask], dtype=float)

        # Infer base feature width and set_size col (matches 3*f + 3 layout)
        d = int(X_train.shape[1])
        f_base = int((d - 3) // 3)
        set_size_col = 3 * f_base
        self._f_base = f_base
        self._set_size_col = set_size_col

        # Sample weights: upweight marginal rows (set_size>0)
        sample_weight = np.ones((len(y_train),), dtype=float)
        if np.isfinite(self.single_step2_weight) and float(self.single_step2_weight) != 1.0:
            try:
                ss = np.asarray(X_train[:, int(set_size_col)], dtype=float)
                sample_weight = np.where(ss > 0.0, float(self.single_step2_weight), 1.0)
            except Exception:
                sample_weight = np.ones((len(y_train),), dtype=float)

        # Train expected_gain model: clf for success, reg for positive magnitude
        thr = float(self.good_threshold)
        
        # Model variant: single-part (regressor only) vs two-part (classifier + regressor, paper default)
        if not self.use_two_part_model:
            # Single-regressor variant
            reg = self.reg_class(**self.reg_params)
            try:
                reg.fit(X_train, y_train, sample_weight=sample_weight)
            except TypeError:
                reg.fit(X_train, y_train)
            self._model = reg
        else:
            # Two-part model (paper's method): classifier for P(G > 0) + regressor for E[G | G > 0]
            y_succ = (np.asarray(y_train, dtype=float) > thr).astype(int)
            clf = self.clf_class(**self.clf_params)
            try:
                clf.fit(X_train, y_succ, sample_weight=sample_weight)
            except TypeError:
                clf.fit(X_train, y_succ)

            pos_mask = y_succ == 1
            reg = self.reg_class(**self.reg_params)
            if int(np.sum(pos_mask)) < 50:
                # Fallback to raw-gain regression if not enough positives
                try:
                    reg.fit(X_train, y_train, sample_weight=sample_weight)
                except TypeError:
                    reg.fit(X_train, y_train)
                self._model = reg
            else:
                X_pos = X_train[pos_mask]
                y_pos = np.maximum(0.0, np.asarray(y_train, dtype=float)[pos_mask])
                sw_pos = sample_weight[pos_mask]
                try:
                    reg.fit(X_pos, y_pos, sample_weight=sw_pos)
                except TypeError:
                    reg.fit(X_pos, y_pos)
                self._model = {"clf": clf, "reg": reg}

        # Optional family-aware penalty map (learned from training marginal rows only).
        # Store an unscaled family risk score pi_f and apply lambda_fam at selection time.
        self._step2_family_penalty_map = None
        if self.step2_family_prior:
            if meta.pf is None or meta.cf is None:
                raise RuntimeError(
                    "step2_family_prior=True requires family metadata in train_meta_cache (pf/cf arrays)."
                )
            pf_train = np.asarray(meta.pf[mask], dtype=object)
            cf_train = np.asarray(meta.cf[mask], dtype=object)
            ss = np.asarray(X_train[:, int(set_size_col)], dtype=float)
            marg_mask = ss > 0.0
            pf_m = pf_train[marg_mask]
            cf_m = cf_train[marg_mask]
            y_m = np.asarray(y_train[marg_mask], dtype=float)

            buckets: Dict[object, List[float]] = {}
            for pfv, cfv, gv in zip(pf_m.tolist(), cf_m.tolist(), y_m.tolist()):
                key: object
                if self.step2_family_prior_mode == "cand_only":
                    key = cfv
                else:
                    key = (pfv, cfv)
                buckets.setdefault(key, []).append(float(gv))

            def _stat(vals: List[float]) -> float:
                arr = np.asarray(vals, dtype=float)
                if self.step2_family_prior_stat == "median":
                    return float(np.median(arr))
                if self.step2_family_prior_stat == "p10":
                    return float(np.quantile(arr, 0.10))
                return float(np.mean(arr))

            pen_map: Dict[object, float] = {}
            for key, vals in buckets.items():
                if len(vals) < int(self.step2_family_prior_min_count):
                    continue
                s = _stat(vals)
                # pi_f := max(0, -stat_f)  (unscaled); lambda_fam is applied later.
                pi_f = max(0.0, -float(s))
                if pi_f > 0:
                    pen_map[key] = float(pi_f)
            self._step2_family_penalty_map = pen_map

    def _get_candidate_ranking_for_target(self, dname: str) -> Tuple[Dict[str, List[str]], List[str]]:
        if dname in self._rank_cache:
            return self._rank_cache[dname]
        if self._rank_df is None:
            self._rank_cache[dname] = ({}, [])
            return self._rank_cache[dname]

        df_rank = self._rank_df.copy()
        if "target_dataset" in df_rank.columns:
            df_rank = df_rank[df_rank["target_dataset"].astype(str) != str(dname)]

        metric = "transfer_success_rate"
        need_cols = {"primary_family", "partner_model", metric}
        if not need_cols.issubset(set(df_rank.columns)):
            self._rank_cache[dname] = ({}, [])
            return self._rank_cache[dname]

        df_rank = df_rank.replace([np.inf, -np.inf], np.nan)
        df_rank = df_rank.dropna(subset=["primary_family", "partner_model", metric])

        rank_by_family: Dict[str, List[str]] = {}
        global_rank: List[str] = []
        if not df_rank.empty:
            agg = df_rank.groupby(["primary_family", "partner_model"])[metric].mean().reset_index()
            for fam, sub in agg.groupby("primary_family"):
                sub2 = sub.sort_values(metric, ascending=False)
                rank_by_family[str(fam)] = sub2["partner_model"].astype(str).tolist()
            g = df_rank.groupby("partner_model")[metric].mean().sort_values(ascending=False)
            global_rank = g.index.astype(str).tolist()

        self._rank_cache[dname] = (rank_by_family, global_rank)
        return self._rank_cache[dname]

    def select(self, dataset_name: str) -> Optional[Union[str, List[str]]]:
        """Select an ensemble for *dataset_name* using the configured primary selector and file-based score loading."""
        if self._models is None or self._primary_selector is None or self._model is None:
            return None

        primary = self._primary_selector.select(dataset_name)
        if primary is None or str(primary) not in set(self._models):
            return None

        scores, _ = self._score_loader(dataset_name)
        dims = load_dataset_dims(dataset_name)
        return self._select_with_scores(primary, scores, dims, dataset_name=dataset_name)

    def _select_with_scores(
        self,
        primary: str,
        scores: np.ndarray,
        dims: tuple = (0, 0),
        dataset_name: Optional[str] = None,
    ) -> Optional[Union[str, List[str]]]:
        """
        Core ensemble-partner selection given a pre-loaded score matrix.

        This method can be called directly (bypassing ELECT primary-model selection
        and file-based data loading) to integrate MetaEns into custom pipelines.

        Parameters
        ----------
        primary : str
            Name of the primary model (must be in ``self._models``).
        scores : np.ndarray, shape (n_samples, n_models)
            Raw detector-score matrix.  Column order must match ``self._models``.
        dims : tuple of (n_samples, n_features)
            Shape of the original input data.  ``n_features`` may be 0 when unknown.
        dataset_name : str, optional
            Identifier used for LODO-safe candidate filtering and debug messages.
            Pass ``None`` for unseen datasets.
        """
        if self._models is None or self._model is None:
            return None
        if primary is None or str(primary) not in set(self._models):
            return None

        label = dataset_name or "<new>"
        self._dbg(f"\n{'-'*100}\n[MetaEns] TARGET dataset={label}")
        self._dbg(f"[MetaEns] primary={primary} family={_model_family(str(primary))} n_selection={self.n_selection_requested} stages={self.stages}")

        m = min(scores.shape[1], len(self._models))
        if m <= 0:
            return primary

        scores = np.asarray(scores[:, :m], dtype=float)
        scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
        scores = np.clip(scores, -1e6, 1e6)

        stats = _precompute_stats(pd.DataFrame(scores), self._models)

        # Primary index
        try:
            p_idx = self._models.index(str(primary))
        except Exception:
            return primary
        if p_idx >= m:
            return primary

        # Candidate list (ranked first, then remaining in pool order)
        rank_by_family, global_rank = self._get_candidate_ranking_for_target(dataset_name or "")
        p_family = _model_family(str(primary))

        ordered_names: List[str] = []
        if p_family and p_family in rank_by_family:
            ordered_names = rank_by_family[p_family]
        elif global_rank:
            ordered_names = global_rank

        ordered_idxs: List[int] = []
        seen = set()

        def _eligible(q_idx: int) -> bool:
            if q_idx == p_idx:
                return False
            if q_idx >= m:
                return False
            return True

        if ordered_names:
            for name in ordered_names:
                if name not in self._models:
                    continue
                q_idx = self._models.index(name)
                if q_idx in seen:
                    continue
                if _eligible(q_idx):
                    ordered_idxs.append(q_idx)
                    seen.add(q_idx)

        # Add the rest in pool order
        for q_idx in range(m):
            if q_idx in seen:
                continue
            if _eligible(q_idx):
                ordered_idxs.append(q_idx)
                seen.add(q_idx)

        if not ordered_idxs:
            return primary

        # Stages
        stages_raw = [s.strip() for s in str(self.stages).split(",") if s.strip()]
        stages: List[int] = []
        for s in stages_raw:
            try:
                stages.append(int(s))
            except Exception:
                pass
        if not stages:
            stages = [0]
        
        # Early exit: if n_selection=1, return just the primary model (no partners)
        if not self.unbounded_selection and int(self.n_selection) <= 1:
            self._dbg(f"[MetaEns] n_selection={int(self.n_selection)} <= 1: returning primary only (no partners)")
            return primary
        
        effective_min_pred_gain = float(self.min_pred_gain)
        effective_min_pred_gain_extra = float(self.min_pred_gain_extra)

        total_candidates = len(ordered_idxs)
        cum_sizes: List[int] = []
        cur = 0
        for s in stages:
            if s <= 0:
                cur = total_candidates
            else:
                cur = min(total_candidates, cur + s)
            if cur <= 0:
                continue
            if not cum_sizes or cur > cum_sizes[-1]:
                cum_sizes.append(cur)
        if not cum_sizes:
            cum_sizes = [total_candidates]

        # Precompute X1_full for (primary,q)
        feature_cols = self._feature_cols_extended()
        X1_full = self._build_test_X1_full(
            ordered_q_idxs=ordered_idxs,
            p_idx_local=int(p_idx),
            stats_local=stats,
            n_samples_local=int(dims[0]),
            n_features_local=int(dims[1]),
            primary_name=str(primary),
        )

        # Step1: score with set-summary features (|S|=0)
        X_step1 = self._build_test_X_set_summary(
            pq_block=X1_full,
            p_sel_mean_vec=None,
            sel_q_mean_block=None,
            set_size=0,
            sim_sel_max=None,
            sim_sel_mean=None,
        )
        
        # Use mean+std for uncertainty-aware selection
        preds_mean_full, preds_std_full = self._predict_expected_gain_mean_std(self._model, X_step1)
        preds_full = np.asarray(preds_mean_full, dtype=float)
        preds_prob_full = self._predict_prob(self._model, X_step1)

        if self.debug_selection:
            topk = min(int(self.debug_topk), len(ordered_idxs))
            if topk > 0:
                order = np.argsort(-np.asarray(preds_full, dtype=float))[:topk].tolist()
                self._dbg(f"[MetaEns] step1 candidates (top{topk} by pred_gain, gain_thresh={float(effective_min_pred_gain):.4f}):")
                for r, j in enumerate(order, start=1):
                    q_idx = int(ordered_idxs[int(j)])
                    std_v = float(preds_std_full[int(j)])
                    prob_v = float(preds_prob_full[int(j)])
                    self._dbg(f"  {r:>2d}. {self._models[q_idx]} pred_gain={float(preds_full[int(j)]):+.6f} std={std_v:.4f} prob={prob_v:.4f}")

        # Pick partner1 in staged order
        partner1_idx: Optional[int] = None
        ordered_pos = {int(q): i for i, q in enumerate(ordered_idxs)}
        for stage_n in cum_sizes:
            n = int(stage_n)
            if n <= 0:
                continue
            cand_subset = ordered_idxs[:n]
            preds = preds_full[:n]
            preds_std = preds_std_full[:n]
            preds_prob = preds_prob_full[:n]
            if len(cand_subset) == 0:
                continue
            
            # Selection method: submodular or greedy
            if self.use_submodular:
                # Submodular selection for first partner
                best_idx, best_value = self._submodular_select_next(
                    candidate_idxs=cand_subset,
                    selected_idxs=[],
                    predicted_gains=preds,
                    stats=stats,
                    p_idx=p_idx,
                    use_lazy=self.submodular_lazy,
                )
                if best_idx is not None and best_value >= float(effective_min_pred_gain):
                    partner1_idx = int(best_idx)
                    best_local = cand_subset.index(best_idx)
                    best_std = float(preds_std[best_local])
                    best_prob = float(preds_prob[best_local])
                    self._dbg(f"[MetaEns|SUBMODULAR] step1_pick: partner1={self._models[int(partner1_idx)]} submod_value={best_value:+.6f} std={best_std:.4f} prob={best_prob:.4f} stage_used={int(stage_n)}")
                    break
                else:
                    self._dbg(f"[MetaEns|SUBMODULAR] step1_stage_skip: stage_n={int(stage_n)} value={best_value:+.6f} < {float(effective_min_pred_gain):.4f}")
            else:
                # Greedy selection (baseline)
                best_local = int(np.argmax(preds))
                best_pred = float(preds[best_local])
                best_std = float(preds_std[best_local])
                best_prob = float(preds_prob[best_local])
                
                # Check gain threshold
                if best_pred >= float(effective_min_pred_gain):
                    partner1_idx = int(cand_subset[best_local])
                    self._dbg(f"[MetaEns|GREEDY] step1_pick: partner1={self._models[int(partner1_idx)]} pred_gain={best_pred:+.6f} std={best_std:.4f} prob={best_prob:.4f} stage_used={int(stage_n)}")
                    break
                else:
                    self._dbg(f"[MetaEns|GREEDY] step1_stage_skip: stage_n={int(stage_n)} gain={best_pred:+.6f} < {float(effective_min_pred_gain):.4f}")

        if partner1_idx is None:
            self._dbg("[MetaEns] step1_abstain: no candidate met min_pred_gain; returning primary only")
            return primary

        selected_idxs_step: List[int] = [int(partner1_idx)]
        selected_partners: List[str] = [self._models[int(partner1_idx)]]

        if self.unbounded_selection:
            remaining_slots = max(0, len(ordered_idxs))
        else:
            # n_selection is total ensemble size (primary + all partners)
            # Already selected: primary (1) + partner1 (1) = 2
            # remaining_slots = n_selection - 2
            remaining_slots = max(0, int(self.n_selection) - 2)

        # Cache anchor full feature matrices for pair(s,q)
        X_anchor_full_cache: Dict[int, np.ndarray] = {}

        def _p_sel_mean_vec(selected_idxs: List[int]) -> Optional[np.ndarray]:
            if not selected_idxs:
                return None
            vecs = []
            for s in selected_idxs:
                s = int(s)
                if s in ordered_pos:
                    vecs.append(np.asarray(X1_full[int(ordered_pos[s])], dtype=float))
                else:
                    feats_ps = _compute_pair_features(
                        stats=stats,
                        p_idx=int(p_idx),
                        q_idx=int(s),
                        p_name=str(primary),
                        q_name=self._models[int(s)],
                        n_samples=int(dims[0]),
                        n_features=int(dims[1]),
                    )
                    feats_ps = self._enrich_feats_with_primary_stats(stats, int(p_idx), feats_ps)
                    vecs.append(np.asarray([float(feats_ps.get(c, 0.0)) for c in feature_cols], dtype=float))
            if not vecs:
                return None
            return np.mean(np.vstack(vecs), axis=0)

        def _sel_q_mean_block(selected_idxs: List[int], ii: List[int]) -> Optional[np.ndarray]:
            if not selected_idxs:
                return None
            mats = []
            for s in selected_idxs:
                s = int(s)
                if s not in X_anchor_full_cache:
                    X_anchor_full_cache[s] = self._build_test_X1_full(
                        ordered_q_idxs=ordered_idxs,
                        p_idx_local=int(s),
                        stats_local=stats,
                        n_samples_local=int(dims[0]),
                        n_features_local=int(dims[1]),
                        primary_name=str(self._models[int(s)]),
                    )
                mats.append(np.asarray(X_anchor_full_cache[s][ii], dtype=float))
            if not mats:
                return None
            return np.mean(np.stack(mats, axis=0), axis=0)

        # step2+ loop
        for _ in range(int(remaining_slots)):
            new_idx = None
            for stage_n in cum_sizes:
                stage_prefix = ordered_idxs[: int(stage_n)]
                cand_subset = [q for q in stage_prefix if int(q) not in set(selected_idxs_step)]
                if not cand_subset:
                    continue
                ii = [int(ordered_pos[int(q)]) for q in cand_subset if int(q) in ordered_pos]
                if not ii:
                    continue

                pq_block = np.asarray(X1_full[ii], dtype=float)
                p_sel_vec = _p_sel_mean_vec(selected_idxs_step)
                sel_q_block = _sel_q_mean_block(selected_idxs_step, ii)

                idxs_U = list(map(int, cand_subset))

                # sim_max/sim_mean are intentionally excluded from the meta-model's feature
                # vector: the meta-model captures pairwise redundancy through its learned
                # features.  The submodular saturation factor γ uses sim_max separately
                # (inside _submodular_select_next) to enforce diminishing returns.
                sim_sel_max = None
                sim_sel_mean = None

                X_step = self._build_test_X_set_summary(
                    pq_block=pq_block,
                    p_sel_mean_vec=p_sel_vec,
                    sel_q_mean_block=sel_q_block,
                    set_size=int(len(selected_idxs_step)),
                    sim_sel_max=sim_sel_max,
                    sim_sel_mean=sim_sel_mean,
                )

                preds_mean, preds_std = self._predict_expected_gain_mean_std(self._model, X_step)
                preds_prob = self._predict_prob(self._model, X_step)
                step_score = np.asarray(preds_mean, dtype=float)

                # Family-risk prior: lambda_family_prior * pi_f (Empirical Bayes tail control, step2+ only)
                # The meta-model provides the likelihood (mean prediction), pi_f provides family-level prior
                pi_arr = None
                if self._step2_family_penalty_map is not None:
                    pi = []
                    for q_idx in idxs_U:
                        cf = _model_family(str(self._models[int(q_idx)]))
                        key: object
                        if self.step2_family_prior_mode == "cand_only":
                            key = cf
                        else:
                            key = (p_family, cf)
                        pi.append(float(self._step2_family_penalty_map.get(key, 0.0)))
                    pi_arr = np.asarray(pi, dtype=float)
                    step_score = np.asarray(step_score, dtype=float) - float(self.lambda_fam) * pi_arr

                # Effective score: meta-model prediction minus family-risk prior.
                # Saturation factor γ for diversity is applied inside _submodular_select_next.
                eff = step_score

                # Eligibility gating on raw predicted marginal gain
                preds_mean_arr = np.asarray(preds_mean, dtype=float)
                preds_prob_arr = np.asarray(preds_prob, dtype=float)
                eligible_locals = [
                    i for i in range(len(preds_mean_arr))
                    if float(preds_mean_arr[i]) >= float(effective_min_pred_gain_extra)
                ]
                if not eligible_locals:
                    continue

                if self.debug_selection:
                    topk = min(int(self.debug_topk), len(eligible_locals))
                    if topk > 0:
                        # Rank within this stage by effective score
                        elig_sorted = sorted(eligible_locals, key=lambda i: float(eff[int(i)]), reverse=True)[:topk]
                        mode_label = "SUBMODULAR" if self.use_submodular else "GREEDY"
                        self._dbg(
                            f"[MetaEns|{mode_label}] step{len(selected_idxs_step)+1} stage_n={int(stage_n)} eligible={len(eligible_locals)}/{len(cand_subset)} "
                            f"(gain_thresh={float(effective_min_pred_gain_extra):.4f}) top{topk}:"
                        )
                        for r, i_local in enumerate(elig_sorted, start=1):
                            q_idx = int(cand_subset[int(i_local)])
                            pi_v = float(pi_arr[int(i_local)]) if pi_arr is not None else 0.0
                            prob_v = float(preds_prob_arr[int(i_local)])
                            self._dbg(
                                f"  {r:>2d}. {self._models[q_idx]} "
                                f"pred_marg={float(preds_mean[int(i_local)]):+.6f} "
                                f"prob={prob_v:.4f} pi={pi_v:.4f} eff={float(eff[int(i_local)]):+.6f}"
                            )

                # Selection method: submodular or greedy
                if self.use_submodular:
                    # Submodular selection for additional partners.
                    # Filter to eligible candidates, then score with eff (includes family priors).
                    eligible_cand_idxs = [cand_subset[i] for i in eligible_locals]
                    eligible_eff_scores = np.asarray([eff[i] for i in eligible_locals], dtype=float)
                    
                    best_idx, best_value = self._submodular_select_next(
                        candidate_idxs=eligible_cand_idxs,
                        selected_idxs=selected_idxs_step,
                        predicted_gains=eligible_eff_scores,  # effective scores, not raw predictions
                        stats=stats,
                        p_idx=p_idx,
                        use_lazy=self.submodular_lazy,
                    )
                    
                    if best_idx is not None and best_value > 0.0:
                        new_idx = int(best_idx)
                        # Find the local index for logging
                        best_local = cand_subset.index(new_idx)
                        best_prob = float(preds_prob_arr[best_local])
                        self._dbg(f"[MetaEns|SUBMODULAR] step{len(selected_idxs_step)+1}_pick: partner={self._models[int(new_idx)]} submod_value={best_value:+.6f} prob={best_prob:.4f} stage_used={int(stage_n)}")
                    else:
                        continue
                else:
                    # Greedy selection (baseline)
                    best_local = int(max(eligible_locals, key=lambda i: float(eff[i])))
                    if float(eff[int(best_local)]) <= 0.0:
                        continue
                    new_idx = int(cand_subset[int(best_local)])
                    best_prob = float(preds_prob_arr[int(best_local)])
                    self._dbg(f"[MetaEns|GREEDY] step{len(selected_idxs_step)+1}_pick: partner={self._models[int(new_idx)]} eff={float(eff[int(best_local)]):+.6f} prob={best_prob:.4f} stage_used={int(stage_n)}")
                
                break

            if new_idx is None:
                self._dbg(f"[MetaEns] stop: no eligible partner met thresholds; selected_partners={len(selected_partners)}")
                break
            selected_idxs_step.append(int(new_idx))
            selected_partners.append(self._models[int(new_idx)])

        # Return primary + partners as an ensemble list
        return [str(primary)] + selected_partners


