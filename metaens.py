"""
MetaEns: Automatic Unsupervised Ensemble Outlier Model Selection
================================================================
High-level API.

Notes
-----
*   Column order in every score matrix must align with ``model_names``.
*   MetaEns only uses the score matrices (not the raw feature space), so it
    works for any data modality.
*   ``n_features`` in :meth:`select` / :meth:`predict` is optional; it is used
    as one meta-feature and defaults to 0 when unknown.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import entropy as sp_entropy
from scipy.stats import kurtosis, skew
from sklearn.metrics import average_precision_score

from algorithms.meta_features import generate_meta_cache
from algorithms.metaens_selector import MetaEnsSelector
from algorithms.base import SingleModelSelector


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

class _NoOpPrimarySelector(SingleModelSelector):
    """Stub selector used when MetaEns handles primary selection via its own kNN.

    MetaEnsSelector.fit() requires a primary_selector_template; this placeholder
    satisfies the interface when kNN handles primary selection.
    """

    def fit(self, historical_df: pd.DataFrame, available_models: List[str]):
        pass

    def select(self, dataset_name: str, scores=None):  # never called via the high-level API
        return None


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class MetaEns:
    """
    Automatic Unsupervised Ensemble Outlier Model Selection.

    Provides a scikit-learn-style ``fit`` / ``select`` / ``predict`` interface
    on top of the MetaEns algorithm described in the ICML 2026 paper.

    Parameters
    ----------
    model_names : list of str
        Names of the detectors in the pool, in the same column order as the
        score matrices that will be passed to :meth:`fit`, :meth:`select`,
        and :meth:`predict`.
    n_selection : int, default 0
        Maximum ensemble size (primary + partners).  0 means adaptive
        (MetaEns stops adding partners once marginal gain falls below the
        threshold).
    seed : int, default 42
        Random seed for the meta-model.
    n_neighbors : int, default 5
        Number of nearest historical datasets used for primary-model
        selection (kNN-ELECT).
    combining_method : {"mean", "max"}, default "mean"
        How to aggregate selected detectors' scores in :meth:`predict`.
    stages : str, default "40,120,0"
        Staged candidate evaluation schedule.
    use_submodular : bool, default True
        Enable submodular diversity-discount optimisation.
    submodular_beta : float, default 1.0
        Saturation strength for the submodular discount.
    submodular_step1_beta : float, default 0.001
        Saturation strength for the first-partner step.
    lambda_family_prior : float, default 0.2
        Family-risk prior weight λ_fam (Empirical Bayes, §3.2).
    step2_family_prior_stat : {"mean", "median", "p10"}, default "p10"
        Statistic used to build the family-risk prior.
    min_pred_gain : float, default 0.001
        Minimum predicted AP gain required to add a first partner.
    min_pred_gain_extra : float, default 0.005
        Minimum predicted AP gain required to add subsequent partners.
    use_two_part_model : bool, default True
        Use the two-part (classifier + regressor) meta-model from the paper.
    """

    def __init__(
        self,
        model_names: List[str],
        *,
        n_selection: int = 0,
        seed: int = 42,
        n_neighbors: int = 5,
        combining_method: str = "mean",
        primary_selector: Optional[SingleModelSelector] = None,
        # --- Advanced MetaEns hyperparameters ---
        stages: str = "40,120,0",
        use_submodular: bool = True,
        submodular_beta: float = 1.0,
        submodular_step1_beta: float = 0.001,
        lambda_family_prior: float = 0.2,
        step2_family_prior_stat: str = "p10",
        min_pred_gain: float = 0.001,
        min_pred_gain_extra: float = 0.005,
        use_two_part_model: bool = True,
    ):
        if not model_names:
            raise ValueError(
                "model_names must be a non-empty list of detector names."
            )
        if combining_method not in ("mean", "max"):
            raise ValueError(
                f"combining_method must be 'mean' or 'max', got {combining_method!r}."
            )
        if n_selection < 0:
            raise ValueError(
                f"n_selection must be >= 0 (0 = adaptive), got {n_selection}."
            )
        if n_neighbors < 1:
            raise ValueError(
                f"n_neighbors must be >= 1, got {n_neighbors}."
            )

        self.model_names = list(model_names)
        self.n_selection = n_selection
        self.seed = seed
        self.n_neighbors = n_neighbors
        self.combining_method = combining_method

        # kwargs forwarded verbatim to MetaEnsSelector
        self._selector_kwargs: Dict = dict(
            stages=stages,
            use_submodular=use_submodular,
            submodular_beta=submodular_beta,
            submodular_step1_beta=submodular_step1_beta,
            submodular_lazy=True,
            lambda_family_prior=lambda_family_prior,
            step2_family_prior_stat=step2_family_prior_stat,
            min_pred_gain=min_pred_gain,
            min_pred_gain_extra=min_pred_gain_extra,
            good_threshold=0.0,
            single_step2_weight=2.0,
            use_two_part_model=use_two_part_model,
        )

        # Optional pluggable primary selector (e.g. ELECT).
        # When None, the default kNN fingerprint-based selection is used.
        self._primary_selector: Optional[SingleModelSelector] = primary_selector

        # Populated after fit()
        self._selector: Optional[MetaEnsSelector] = None
        self._historical_df: Optional[pd.DataFrame] = None
        # kNN primary-selection structures (used when _primary_selector is None)
        self._fingerprints: Optional[np.ndarray] = None   # (n_datasets, d_fp)
        self._fp_names: Optional[List[str]] = None         # one name per row
        self._fp_ap_table: Optional[np.ndarray] = None     # (n_datasets, n_models)
        # Persistent temp directory (keeps cache file alive for the object lifetime)
        self._workdir: Optional[tempfile.TemporaryDirectory] = None
        self._cache_path: Optional[str] = None

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(
        self,
        score_matrices: Dict[str, np.ndarray],
        labels: Dict[str, np.ndarray],
        *,
        cache_path: Optional[str] = None,
        verbose: bool = True,
    ) -> "MetaEns":
        """
        Train the meta-model on historical datasets.

        Parameters
        ----------
        score_matrices : dict[str, ndarray]
            Maps dataset name → 2-D array of shape ``(n_samples, n_models)``.
            Columns must align with ``model_names`` given at construction.
        labels : dict[str, ndarray]
            Maps dataset name → 1-D binary array ``(n_samples,)``,
            where ``1`` = anomaly and ``0`` = normal.
        cache_path : str, optional
            Path to save / load the pre-computed meta-feature cache (.npz).
            If ``None``, a temporary file is used (deleted when this object
            is garbage-collected).
        verbose : bool, default True
            Print progress messages.

        Returns
        -------
        self
        """
        datasets = sorted(set(score_matrices) & set(labels))
        if not datasets:
            raise ValueError(
                "No common keys found between score_matrices and labels."
            )

        n_models = len(self.model_names)

        # ------------------------------------------------------------------ #
        # Write data to a temporary directory so generate_meta_cache can read #
        # ------------------------------------------------------------------ #
        self._workdir = tempfile.TemporaryDirectory(prefix="metaens_")
        workdir = Path(self._workdir.name)

        scores_dir = workdir / "dataset" / "benchmark" / "intermediate_files" / "scores"
        data_dir   = workdir / "dataset" / "benchmark" / "data"
        meta_dir   = workdir / "dataset" / "benchmark" / "intermediate_files"
        scores_dir.mkdir(parents=True)
        data_dir.mkdir(parents=True)

        for ds in datasets:
            mat = np.asarray(score_matrices[ds], dtype=float)
            y   = np.asarray(labels[ds], dtype=int)
            m   = min(mat.shape[1], n_models)
            pd.DataFrame(mat[:, :m]).to_csv(
                scores_dir / f"{ds}.csv", header=False, index=False
            )
            pd.Series(y).to_csv(data_dir / f"{ds}_y.csv", header=False, index=False)

        (meta_dir / "datasets.txt").write_text("\n".join(datasets))
        (meta_dir / "models.txt").write_text("\n".join(self.model_names))

        # ------------------------------------------------------------------ #
        # Generate (or reuse) the meta-feature cache                          #
        # ------------------------------------------------------------------ #
        if cache_path is None:
            cache_path = str(workdir / "cache_train_meta_ens.npz")
        self._cache_path = cache_path

        # Patch utils.core.BASE_PATH so score-loading helpers find the temp data.
        import utils.core as _core_mod
        _orig_base = _core_mod.BASE_PATH
        _core_mod.BASE_PATH = str(workdir / "dataset" / "benchmark")

        try:
            if not os.path.exists(cache_path):
                ok = generate_meta_cache(
                    cache_path=cache_path,
                    intermediate_files_folder="intermediate_files",
                    datasets_file=str(meta_dir / "datasets.txt"),
                    models_file=str(meta_dir / "models.txt"),
                    verbose=verbose,
                )
                if not ok:
                    raise RuntimeError(
                        "Meta-feature cache generation failed.  "
                        "Check that score_matrices and labels are non-empty and correctly shaped."
                    )
        finally:
            _core_mod.BASE_PATH = _orig_base

        # ------------------------------------------------------------------ #
        # Build per-dataset AP table and fingerprints for kNN primary selection
        # ------------------------------------------------------------------ #
        rows: List[Dict] = []
        fp_names: List[str] = []
        fp_vecs: List[np.ndarray] = []
        ap_rows: List[List[float]] = []

        for ds in datasets:
            mat = np.asarray(score_matrices[ds], dtype=float)
            y   = np.asarray(labels[ds], dtype=int)
            m   = min(mat.shape[1], n_models)

            # historical AP DataFrame (used inside MetaEnsSelector.fit)
            for j, mname in enumerate(self.model_names[:m]):
                try:
                    ap = float(average_precision_score(y, mat[:, j]))
                except Exception:
                    ap = 0.0
                rows.append({"dataset": ds, "model": mname, "ap": ap})

            # kNN fingerprint
            fp_names.append(ds)
            fp_vecs.append(_compute_fingerprint(mat, n_models))

            # per-model AP vector (one entry per model in pool)
            aps = []
            for j in range(n_models):
                if j < m:
                    try:
                        aps.append(float(average_precision_score(y, mat[:, j])))
                    except Exception:
                        aps.append(0.0)
                else:
                    aps.append(0.0)
            ap_rows.append(aps)

        self._historical_df  = pd.DataFrame(rows)
        self._fp_names       = fp_names
        self._fingerprints   = np.vstack(fp_vecs)
        self._fp_ap_table    = np.array(ap_rows, dtype=float)

        # ------------------------------------------------------------------ #
        # Build an in-memory score loader for MetaEnsSelector                 #
        # ------------------------------------------------------------------ #
        _score_lookup: Dict[str, Tuple[np.ndarray, np.ndarray]] = {
            ds: (
                np.asarray(score_matrices[ds], dtype=float),
                np.asarray(labels[ds], dtype=int),
            )
            for ds in datasets
        }

        def _mem_loader(ds_name: str) -> Tuple[np.ndarray, np.ndarray]:
            if ds_name not in _score_lookup:
                raise ValueError(
                    f"Dataset '{ds_name}' was not provided during fit()."
                )
            return _score_lookup[ds_name]

        # ------------------------------------------------------------------ #
        # Create MetaEnsSelector (primary selection handled via kNN above)    #
        # ------------------------------------------------------------------ #
        # Patch BASE_PATH again for the MetaEnsSelector.fit() call.
        _core_mod.BASE_PATH = str(workdir / "dataset" / "benchmark")
        try:
            self._selector = MetaEnsSelector(
                train_meta_cache=cache_path,
                intermediate_files_folder="intermediate_files",
                n_selection=self.n_selection,
                seed=self.seed,
                primary_selector=_NoOpPrimarySelector(),
                **self._selector_kwargs,
            )
            # Override the default file-based loader with our in-memory one.
            self._selector._score_loader = _mem_loader
            self._selector.fit(self._historical_df, self.model_names)
        finally:
            _core_mod.BASE_PATH = _orig_base

        # Fit the pluggable primary selector (if provided) with score matrices
        # so it can compute IPMs and work on new unseen datasets.
        if self._primary_selector is not None:
            try:
                self._primary_selector.fit(
                    self._historical_df,
                    self.model_names,
                    score_matrices=score_matrices,
                )
            except TypeError:
                # Selector does not accept score_matrices kwarg — fall back
                self._primary_selector.fit(self._historical_df, self.model_names)

        return self

    # ------------------------------------------------------------------
    # select
    # ------------------------------------------------------------------

    def select(
        self,
        score_matrix: np.ndarray,
        *,
        n_features: int = 0,
        top_k: Optional[int] = None,
    ) -> List[str]:
        """
        Select an ensemble of detectors for a new (unlabelled) dataset.

        Parameters
        ----------
        score_matrix : ndarray, shape (n_samples, n_models)
            Score matrix from the full detector pool on the target dataset.
            Column order must match ``model_names`` given at construction.
        n_features : int, optional
            Dimensionality of the *original* input data.  Used as one
            meta-feature; defaults to 0 when unknown (has negligible effect).
        top_k : int, optional
            Override ``n_selection`` for this single call.

        Returns
        -------
        list of str
            Names of the selected models (primary model first, then partners).
        """
        if self._selector is None:
            raise RuntimeError("Call fit() before select().")

        score_matrix = np.asarray(score_matrix, dtype=float)

        # Primary model selection: pluggable selector or kNN fallback
        if self._primary_selector is not None:
            raw = self._primary_selector.select("_new_", scores=score_matrix)
            if raw is None:
                primary = self._select_primary(score_matrix)  # kNN fallback
            elif isinstance(raw, list):
                primary = raw[0]
            else:
                primary = str(raw)
        else:
            primary = self._select_primary(score_matrix)

        dims = (score_matrix.shape[0], n_features)

        result = self._selector._select_with_scores(primary, score_matrix, dims)
        if result is None:
            return [primary]
        return result if isinstance(result, list) else [str(result)]

    # ------------------------------------------------------------------
    # predict
    # ------------------------------------------------------------------

    def predict(
        self,
        score_matrix: np.ndarray,
        *,
        n_features: int = 0,
        combining_method: Optional[str] = None,
    ) -> np.ndarray:
        """
        Select an ensemble and return the combined anomaly-score vector.

        Parameters
        ----------
        score_matrix : ndarray, shape (n_samples, n_models)
            Score matrix from the full detector pool.
        n_features : int, optional
            Dimensionality of the original input data (meta-feature).
        combining_method : {"mean", "max"}, optional
            Overrides the instance-level ``combining_method`` for this call.

        Returns
        -------
        ndarray, shape (n_samples,)
            Combined anomaly scores.  Higher = more anomalous.
        """
        selected = self.select(score_matrix, n_features=n_features)
        method = combining_method if combining_method is not None else self.combining_method
        if method not in ("mean", "max"):
            raise ValueError(
                f"combining_method must be 'mean' or 'max', got {method!r}."
            )

        score_matrix = np.asarray(score_matrix, dtype=float)
        sel_set  = set(selected)
        sel_cols = [j for j, m in enumerate(self.model_names) if m in sel_set]
        if not sel_cols:
            sel_cols = list(range(min(score_matrix.shape[1], len(self.model_names))))

        mat = score_matrix[:, sel_cols].copy()

        # Min-max normalise each detector before combining (matches training protocol)
        mn  = mat.min(axis=0)
        mx  = mat.max(axis=0)
        rng = mx - mn
        rng[rng < 1e-10] = 1.0
        mat = (mat - mn) / rng

        if method == "max":
            return mat.max(axis=1)
        return mat.mean(axis=1)

    # ------------------------------------------------------------------
    # Convenience / compatibility
    # ------------------------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        """``True`` after :meth:`fit` has been called successfully."""
        return self._selector is not None

    def decision_function(
        self,
        score_matrix: np.ndarray,
        *,
        n_features: int = 0,
        combining_method: Optional[str] = None,
    ) -> np.ndarray:
        """
        Alias for :meth:`predict` following the PyOD / scikit-learn convention.

        Returns raw anomaly scores (higher value = more anomalous).

        Parameters
        ----------
        score_matrix : ndarray, shape (n_samples, n_models)
            Score matrix from the full detector pool.
        n_features : int, optional
            Dimensionality of the original input data (meta-feature).
        combining_method : {"mean", "max"}, optional
            Overrides the instance-level ``combining_method`` for this call.

        Returns
        -------
        ndarray, shape (n_samples,)
            Combined anomaly scores.
        """
        return self.predict(
            score_matrix,
            n_features=n_features,
            combining_method=combining_method,
        )

    def get_params(self, deep: bool = True) -> Dict:
        """
        Return the constructor parameters of this estimator.

        Compatible with scikit-learn's ``get_params`` convention, enabling
        use in ``GridSearchCV`` and similar tools.

        Parameters
        ----------
        deep : bool, default True
            Ignored (no sub-estimators); kept for API compatibility.

        Returns
        -------
        dict
            Parameter names mapped to their current values.
        """
        kw = self._selector_kwargs
        return {
            "model_names":            self.model_names,
            "n_selection":            self.n_selection,
            "seed":                   self.seed,
            "n_neighbors":            self.n_neighbors,
            "combining_method":       self.combining_method,
            "stages":                 kw["stages"],
            "use_submodular":         kw["use_submodular"],
            "submodular_beta":        kw["submodular_beta"],
            "submodular_step1_beta":  kw["submodular_step1_beta"],
            "lambda_family_prior":    kw["lambda_family_prior"],
            "step2_family_prior_stat":kw["step2_family_prior_stat"],
            "min_pred_gain":          kw["min_pred_gain"],
            "min_pred_gain_extra":    kw["min_pred_gain_extra"],
            "use_two_part_model":     kw["use_two_part_model"],
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _select_primary(self, score_matrix: np.ndarray) -> str:
        """
        Pick the primary detector via k-nearest-neighbour lookup over the
        historical dataset fingerprints (lightweight ELECT analogue).

        Finds the *k* most similar historical datasets (by Euclidean distance
        in fingerprint space) and returns the detector with the highest mean
        AP across those neighbours.
        """
        if self._fingerprints is None or len(self._fingerprints) == 0:
            raise RuntimeError("Call fit() before select().")

        fp    = _compute_fingerprint(score_matrix, len(self.model_names))
        dists = np.linalg.norm(self._fingerprints - fp[np.newaxis, :], axis=1)
        k     = min(self.n_neighbors, len(dists))
        nn    = np.argpartition(dists, k - 1)[:k]

        mean_ap      = self._fp_ap_table[nn].mean(axis=0)   # (n_models,)
        best_j       = int(np.argmax(mean_ap))
        return self.model_names[best_j]

    def __repr__(self) -> str:
        fitted = self._selector is not None
        return (
            f"MetaEns(n_models={len(self.model_names)}, "
            f"n_selection={self.n_selection}, seed={self.seed}, fitted={fitted})"
        )


# ---------------------------------------------------------------------------
# Module-level helper: dataset fingerprint
# ---------------------------------------------------------------------------

def _compute_fingerprint(score_matrix: np.ndarray, n_pool_models: int) -> np.ndarray:
    """
    Compute a compact fixed-length vector that characterises a dataset's
    score-matrix distribution.  Used for kNN-based primary-model selection.

    The fingerprint contains 9 scalar statistics:
        [avg_model_std, avg_kurtosis, avg_skewness,
         mean_pairwise_pearson, std_pairwise_pearson,
         mean_pairwise_jaccard,
         avg_score_entropy,
         log1p(n_samples), log1p(n_models_in_matrix)]
    """
    mat = np.asarray(score_matrix, dtype=float)
    m   = min(mat.shape[1], n_pool_models)
    if m == 0:
        return np.zeros(9, dtype=float)
    mat = mat[:, :m]
    mat = np.nan_to_num(mat, nan=0.0, posinf=1e6, neginf=-1e6)

    # Per-column min-max normalise
    mn  = mat.min(axis=0)
    mx  = mat.max(axis=0)
    rng = mx - mn
    rng[rng < 1e-10] = 1.0
    norm = (mat - mn) / rng

    feat: List[float] = [
        float(np.mean(np.std(norm, axis=0))),
        float(np.mean(kurtosis(norm, axis=0))),
        float(np.mean(skew(norm, axis=0))),
    ]

    # Mean/std of upper-triangle pairwise Pearson correlations
    if m >= 2:
        with np.errstate(divide="ignore", invalid="ignore"):
            C = np.corrcoef(norm, rowvar=False)
            C = np.nan_to_num(C, nan=0.0)
        upper = C[np.triu_indices(m, k=1)]
        feat.append(float(np.mean(upper)))
        feat.append(float(np.std(upper)))
    else:
        feat += [0.0, 0.0]

    # Mean pairwise Jaccard (top-10 % agreement)
    k_top = max(1, int(mat.shape[0] * 0.1))
    top_k = np.argpartition(norm, -k_top, axis=0)[-k_top:]
    mask  = np.zeros_like(norm, dtype=bool)
    for j in range(m):
        mask[top_k[:, j], j] = True
    inter = (mask.T @ mask).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        jacc = inter / (2 * k_top - inter)
        jacc = np.nan_to_num(jacc, nan=0.0)
    if m >= 2:
        feat.append(float(np.mean(jacc[np.triu_indices(m, k=1)])))
    else:
        feat.append(0.0)

    # Mean score entropy per detector
    ent = []
    for j in range(m):
        h, _ = np.histogram(norm[:, j], bins=10, density=True)
        ent.append(float(sp_entropy(h + 1e-9)))
    feat.append(float(np.mean(ent)))

    feat.append(float(np.log1p(mat.shape[0])))   # log n_samples
    feat.append(float(np.log1p(m)))              # log n_models

    return np.array(feat, dtype=float)
