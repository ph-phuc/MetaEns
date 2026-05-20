"""
Implementation of the ELECT algorithm (ICDM 2022).

Algorithm overview
------------------
1. Build Internal Performance Measures (IPMs) per dataset via PageRank, HITS,
   and mean correlation on the model-correlation graph.
2. Train a LightGBM meta-learner (LODO) to predict pairwise AP differences
   from IPM features.
3. At inference time, use Bayesian Optimisation (Expected Improvement) guided
   by Weighted Kendall's Tau dataset similarity to select the best model.

Two operating modes
-------------------
**LODO benchmark mode** (default, used by run_metaens.py):
    fit(historical_df, available_models)
        Lazily initialises the shared _ELECTCore processor (loads pre-built
        imp_wild/X_wild/y_wild when present, rebuilds from score CSVs otherwise).
        LightGBM is re-trained per fold inside select().
    select(dataset_name)
        LODO-safe: excludes the test dataset from LightGBM training.

**Production mode** (used by MetaEns high-level API or custom setups):
    fit(historical_df, available_models, score_matrices=score_dict)
        Computes IPMs for every historical dataset from raw score matrices,
        trains LightGBM on ALL historical data (no LODO), and caches the result.
    select(dataset_name, scores=score_matrix)
        Computes the IPM for the new score matrix on the fly and runs Bayesian
        Optimisation against historical AP/IPM data.  dataset_name is only a
        label; it does not need to exist in the historical data.
"""

import itertools
import os
import warnings
from typing import Any, Dict, List, Optional, Union

import lightgbm as lgb
import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.preprocessing import RobustScaler
from tqdm import tqdm

from .base import SingleModelSelector

# ---------------------------------------------------------------------------
# Thread configuration (determinism)
# ---------------------------------------------------------------------------
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

warnings.filterwarnings('ignore', category=UserWarning)

# ---------------------------------------------------------------------------
# ELECT algorithm constants (from the original paper)
# ---------------------------------------------------------------------------
N_SIM_DATASETS   = 5    # Number of similar datasets used for model scoring
ELECT_ITERATIONS = 50   # Max Bayesian Optimisation iterations
ELECT_CONV_ITER  = 50   # Convergence window (stop when similar-dataset set stabilises)

# Initial model set for the first BO iteration (indices into the 297-model pool).
# When running on a different pool size these are replaced with a deterministic sample.
_DEFAULT_INITIAL_MODELS = [179, 253, 22, 16, 54, 214, 273, 291]


# =============================================================================
# Part 1 — IPM computation helpers
# =============================================================================

def _compute_ipm_block(scores: np.ndarray, n_models: int) -> Optional[np.ndarray]:
    """
    Compute the 3-row IPM block (PageRank, HITS, mean-correlation) for a
    (n_samples × n_models) score matrix.

    Returns a (3, n_models) array, or None if the input is invalid.
    """
    if scores.shape[0] < 2 or not np.isfinite(scores).all():
        return None

    scores = np.clip(scores, -1e6, 1e6)

    # Fix zero/infinite variance columns
    variances = np.var(scores, axis=0)
    zero_var = (variances == 0) | ~np.isfinite(variances)
    if zero_var.any():
        rng = np.random.RandomState(42)
        for col in np.where(zero_var)[0]:
            scores[:, col] = scores[:, col] + rng.normal(0, 1e-6, scores.shape[0])

    try:
        norm_scores = RobustScaler().fit_transform(scores)
    except Exception:
        return None

    post_var = np.var(norm_scores, axis=0)
    near_zero = post_var < 1e-10
    if near_zero.any():
        rng = np.random.RandomState(42)
        for col in np.where(near_zero)[0]:
            norm_scores[:, col] = norm_scores[:, col] + rng.normal(0, 1e-8, norm_scores.shape[0])

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        corr = np.corrcoef(norm_scores, rowvar=False)

    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr, 1.0)
    corr = (corr + corr.T) / 2

    G = nx.from_numpy_array(np.abs(corr), create_using=nx.Graph)
    try:
        pagerank = nx.pagerank(
            G, weight='weight', tol=1e-10, max_iter=1000, alpha=0.85,
            nstart={k: 1.0 / n_models for k in range(n_models)},
        )
        _, hits = nx.hits(
            G, tol=1e-10, max_iter=1000,
            nstart={k: 1.0 / n_models for k in range(n_models)},
        )
        mean_sim = np.mean(corr, axis=1)
    except Exception:
        return None

    block = np.zeros((3, n_models), dtype=np.float64)
    block[0] = [pagerank.get(k, 0.0) for k in range(n_models)]
    block[1] = [hits.get(k, 0.0) for k in range(n_models)]
    block[2] = mean_sim
    return block


def _process_dataset_for_ipm(args):
    """Compute the 3-row IPM block for one dataset (reads scores from CSV)."""
    i, dataset_name, n_models, base_path, intermediate_folder = args
    scores_path = os.path.join(base_path, intermediate_folder, 'scores')
    score_file = os.path.join(scores_path, f"{dataset_name}.csv")
    if not os.path.exists(score_file):
        score_file = os.path.join(scores_path, f"{dataset_name.lower()}.mat.csv")
        if not os.path.exists(score_file):
            return i, None
    try:
        scores = pd.read_csv(score_file, header=None, dtype=np.float64).to_numpy()
    except Exception:
        return i, None
    return i, _compute_ipm_block(scores, n_models)


def _build_ipm_matrix(datasets, n_models, base_path, intermediate_folder='intermediate_files'):
    """Build the full IPM matrix (shape: n_datasets*3 × n_models)."""
    n = len(datasets)
    ipm_matrix = np.zeros((n * 3, n_models), dtype=np.float64)
    valid_indices = []
    tasks = [(i, ds, n_models, base_path, intermediate_folder) for i, ds in enumerate(datasets)]
    for i, block in tqdm(
        (_process_dataset_for_ipm(t) for t in tasks),
        total=n, desc="Building IPM matrix",
    ):
        if block is not None:
            ipm_matrix[i * 3: (i + 1) * 3] = block
            valid_indices.append(i)
    return ipm_matrix, valid_indices


def _build_meta_training_data(ap_values, ipm_matrix, valid_indices, n_models):
    """Generate (X, y) pairwise training data from IPMs and AP values."""
    pairs = sorted(itertools.combinations(range(n_models), 2))
    X, y = [], []
    for i in tqdm(valid_indices, desc="Building meta-training data"):
        ipms = ipm_matrix[i * 3: (i + 1) * 3]
        for m1, m2 in pairs:
            diff = float(ap_values[i, m1] - ap_values[i, m2])
            fwd = np.concatenate([ipms[:, m1], ipms[:, m2], ipms[:, m1] - ipms[:, m2]]).round(8)
            rev = np.concatenate([ipms[:, m2], ipms[:, m1], ipms[:, m2] - ipms[:, m1]]).round(8)
            X += [fwd, rev]
            y += [diff, -diff]
    return np.array(X, dtype=np.float64), np.array(y, dtype=np.float64).round(8)


# =============================================================================
# Part 2 — LightGBM helper
# =============================================================================

def _fit_lgbm(X: np.ndarray, y: np.ndarray, random_state: int = 42) -> Optional[lgb.LGBMRegressor]:
    """Train and return a LightGBM regressor with the ELECT hyperparameters."""
    if len(X) == 0:
        return None
    rng = np.random.RandomState(random_state)
    idx = np.arange(len(X))
    rng.shuffle(idx)
    clf = lgb.LGBMRegressor(
        random_state=random_state, verbose=-1, n_jobs=1,
        num_leaves=64, max_depth=20, learning_rate=0.01,
        n_estimators=200, objective='huber', min_data_in_leaf=1000,
        deterministic=True, force_col_wise=True, max_bin=255,
        min_data_in_bin=3, feature_fraction=1.0,
        bagging_fraction=1.0, bagging_freq=0,
    )
    clf.fit(X[idx], y[idx])
    return clf


# =============================================================================
# Part 3 — Core algorithm
# =============================================================================

class _ELECTCore:
    """
    Internal ELECT processor: loads pre-built meta-data, trains LightGBM per
    LODO fold, and runs Bayesian Optimisation at inference time.

    Users should interact with the public ``ELECT`` class below.
    """

    def __init__(
        self,
        random_state: int = 42,
        base_path: str = '.',
        intermediate_folder: Optional[str] = None,
    ):
        self.random_state = random_state
        self.clf: Optional[lgb.LGBMRegressor] = None
        self.n_iter = ELECT_ITERATIONS
        self.n_sim_datasets = N_SIM_DATASETS
        self.conv_iter = ELECT_CONV_ITER

        if intermediate_folder is None:
            intermediate_folder = os.environ.get('ELECT_INTERMEDIATE_FOLDER', 'intermediate_files')
        self.intermediate_folder = intermediate_folder

        self._meta_data: Dict[str, Any] = self._build_meta_data(base_path)
        self.valid_datasets: int = len(self._meta_data['valid_indices'])
        self.model_names: List[str] = self._meta_data['model_names']

    # ------------------------------------------------------------------
    # Meta-data construction
    # ------------------------------------------------------------------

    def _build_meta_data(self, base_path: str) -> Dict[str, Any]:
        inter    = os.path.join(base_path, self.intermediate_folder)
        ap_file  = os.path.join(inter, 'AP_full.xlsx')
        ipm_file = os.path.join(inter, 'imp_wild.npy')
        x_file   = os.path.join(inter, 'X_wild.npy')
        y_file   = os.path.join(inter, 'y_wild.npy')

        if not os.path.exists(ap_file):
            raise FileNotFoundError(f"AP performance file not found: {ap_file}")

        ap_df = pd.read_excel(ap_file, engine='openpyxl', dtype_backend='numpy_nullable')
        ap_df.iloc[:, 1:] = ap_df.iloc[:, 1:].astype(np.float64)
        datasets  = ap_df['Dataset'].tolist()
        ap_values = ap_df.to_numpy()[:, 1:].astype(np.float64)
        n_models  = ap_values.shape[1]

        # Use pre-built ELECT artifacts (imp_wild.npy, X_wild.npy, y_wild.npy) when present.
        if os.path.exists(ipm_file) and os.path.exists(x_file) and os.path.exists(y_file):
            ipm_matrix    = np.load(ipm_file)
            X_meta        = np.load(x_file)
            y_meta        = np.load(y_file)
            valid_indices = list(range(len(datasets)))
        else:
            ipm_matrix, valid_indices = _build_ipm_matrix(
                datasets, n_models, base_path, self.intermediate_folder
            )
            X_meta, y_meta = _build_meta_training_data(
                ap_values, ipm_matrix, valid_indices, n_models
            )

        return {
            'X_meta': X_meta,
            'y_meta': y_meta,
            'ipm_matrix': ipm_matrix,
            'valid_indices': valid_indices,
            'model_names': list(ap_df.columns[1:]),
        }

    # ------------------------------------------------------------------
    # Offline training (one per LODO fold)
    # ------------------------------------------------------------------

    def offline_training(self, n_datasets: int, n_models: int, test_dataset_idx: int):
        """Train the LightGBM meta-learner, excluding the test dataset (LODO)."""
        valid    = self._meta_data['valid_indices']
        n_valid  = len(valid)
        rows_per_ds = len(self._meta_data['X_meta']) // n_valid if n_valid > 0 else 0

        idx_map = {ds_idx: pos for pos, ds_idx in enumerate(valid)}
        if test_dataset_idx in idx_map:
            pos       = idx_map[test_dataset_idx]
            test_rows = set(range(pos * rows_per_ds, (pos + 1) * rows_per_ds))
        else:
            test_rows = set()

        train_rows = sorted(set(range(n_valid * rows_per_ds)) - test_rows)
        X_train    = self._meta_data['X_meta'][train_rows]
        y_train    = self._meta_data['y_meta'][train_rows]
        self.clf   = _fit_lgbm(X_train, y_train, self.random_state)

    # ------------------------------------------------------------------
    # Online model selection (Bayesian Optimisation)
    # ------------------------------------------------------------------

    def online_model_selection(
        self,
        ap_values: np.ndarray,
        train_index: List[int],
        test_index: List[int],
        n_models: int,
        internal_measure_mat: Optional[np.ndarray] = None,
    ) -> List[int]:
        """
        Select the best model for the test dataset via Bayesian Optimisation.

        Returns a list of selected model indices (length 1 by default).
        """
        if self.clf is None:
            return [int(np.argsort(np.mean(ap_values[train_index], axis=0))[-1])]

        if internal_measure_mat is None:
            internal_measure_mat = self._meta_data['ipm_matrix']

        test_ds_idx = test_index[0]
        all_models  = list(range(n_models))

        # Initial model set — randomise when pool size differs from paper
        if any(m >= n_models for m in _DEFAULT_INITIAL_MODELS):
            rng     = np.random.RandomState(self.random_state + n_models)
            n_start = min(len(_DEFAULT_INITIAL_MODELS), n_models)
            curr_models = sorted(rng.choice(all_models, n_start, replace=False).tolist())
        else:
            curr_models = sorted(_DEFAULT_INITIAL_MODELS)

        left_models = sorted(set(all_models) - set(curr_models))
        best_model: List[int] = []
        sim_history: List[List[int]] = []

        for _ in range(self.n_iter):
            pairs_local = sorted(itertools.combinations(range(len(curr_models)), 2))
            if not pairs_local:
                break

            test_ipm = internal_measure_mat[test_ds_idx * 3: (test_ds_idx + 1) * 3][:, curr_models]
            fwd = np.zeros((len(pairs_local), 9))
            rev = np.zeros((len(pairs_local), 9))
            for k, (a, b) in enumerate(pairs_local):
                fwd[k] = np.concatenate([test_ipm[:, a], test_ipm[:, b], test_ipm[:, a] - test_ipm[:, b]])
                rev[k] = np.concatenate([test_ipm[:, b], test_ipm[:, a], test_ipm[:, b] - test_ipm[:, a]])

            pred      = self.clf.predict(fwd)
            pred_rev  = self.clf.predict(rev)
            pred_comb = 0.5 * (pred - pred_rev)  # Symmetrised prediction

            # Keep only non-contradictory pairs
            consistent = np.sign(pred) == np.sign(-pred_rev)
            kept = np.where(consistent)[0]
            if len(kept) == 0:
                break

            test_norm = self._normalise_diff(pred_comb[kept])
            pairs_arr = np.array(pairs_local)

            dataset_sim = np.zeros(ap_values.shape[0])
            for i in train_index:
                train_vals = ap_values[i, curr_models]
                a_vals     = train_vals[pairs_arr[kept, 0]]
                b_vals     = train_vals[pairs_arr[kept, 1]]
                train_norm = self._normalise_diff(a_vals - b_vals)
                dataset_sim[i] = self._weighted_kendall(test_norm, train_norm)

            similar      = np.sort(np.argsort(-dataset_sim)[:self.n_sim_datasets])
            neighbor_aps = np.mean(ap_values[similar], axis=0)
            best_model   = [int(np.argmax(neighbor_aps))]

            if not left_models:
                break

            mu    = np.array([np.mean(ap_values[similar, m]) for m in left_models])
            sigma = np.maximum(np.array([np.std(ap_values[similar, m]) for m in left_models]), 1e-8)
            z     = np.clip((mu - np.max(neighbor_aps[curr_models])) / sigma, -35, 35)
            ei    = (mu - np.max(neighbor_aps[curr_models])) * norm.cdf(z) + sigma * norm.pdf(z)
            ei[sigma < 1e-8] = 0.0

            if ei.max() <= 0:
                break

            chosen = left_models[int(np.argmax(ei))]
            curr_models = sorted(curr_models + [chosen])
            left_models.remove(chosen)

            sim_history.append(similar.tolist())
            if len(sim_history) >= self.conv_iter and np.all(
                np.array(sim_history[-self.conv_iter:]) == similar
            ):
                break  # Convergence

        if not best_model:
            best_model = [int(np.argmax(np.mean(ap_values[train_index], axis=0)))]
        return best_model

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_diff(arr: np.ndarray) -> np.ndarray:
        scale = np.max(np.abs(arr))
        return arr / (scale + np.finfo(float).eps)

    @staticmethod
    def _weighted_kendall(a: np.ndarray, b: np.ndarray) -> float:
        eps   = np.finfo(float).eps
        c1    = np.abs(a) <= np.abs(b)
        c     = np.where(c1, a / (b + eps), b / (a + eps))
        denom = np.sum(np.abs(c))
        return float(np.sum(c) / denom) if denom > 0 else 0.0


# =============================================================================
# Part 4 — Public selector
# =============================================================================


class ELECT(SingleModelSelector):
    """
    Primary model selector using the full ELECT algorithm.

    Parameters
    ----------
    base_path : str, optional
        Root directory containing ``intermediate_folder/`` with score files
        and ``AP_full.xlsx``.  Defaults to ``datasets/benchmark`` relative
        to the package root.
    intermediate_folder : str
        Name of the intermediate-files sub-directory (default:
        ``"intermediate_files"``).
    n_selection : int
        Number of models to return (default: 1).
    random_state : int
        Random seed (default: 42).
    verbose : bool
        Print progress messages (default: False).
    """

    # Class-level cache: share one _ELECTCore instance across all LODO folds.
    _processor_cache: dict = {}

    def __init__(
        self,
        base_path: Optional[str] = None,
        intermediate_folder: str = 'intermediate_files',
        n_selection: int = 1,
        random_state: int = 42,
        verbose: bool = False,
    ):
        self.n_selection         = n_selection
        self.random_state        = random_state
        self.verbose             = verbose
        self.intermediate_folder = intermediate_folder

        if base_path is None:
            pkg_root  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            base_path = os.path.join(pkg_root, 'datasets', 'benchmark')
        self.base_path = base_path

        # Load canonical dataset list (needed for LODO index mapping)
        datasets_file = os.path.join(base_path, intermediate_folder, 'datasets.txt')
        if os.path.exists(datasets_file):
            with open(datasets_file) as f:
                self.canonical_datasets: Optional[List[str]] = [
                    ln.strip() for ln in f if ln.strip()
                ]
        else:
            self.canonical_datasets = None

        self._core: Optional[_ELECTCore] = None
        self._available_models: Optional[List[str]] = None

        # Production-mode state (populated when score_matrices is passed to fit)
        self._prod_clf:         Optional[lgb.LGBMRegressor] = None
        self._prod_ap_matrix:   Optional[np.ndarray]        = None   # (n_ds, n_models)
        self._prod_ipm_matrix:  Optional[np.ndarray]        = None   # (n_ds*3, n_models)
        self._prod_ds_names:    Optional[List[str]]         = None
        self._prod_model_names: Optional[List[str]]         = None

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        historical_df: pd.DataFrame,
        available_models: List[str],
        score_matrices: Optional[Dict[str, np.ndarray]] = None,
    ):
        """
        Prepare the selector.

        Parameters
        ----------
        historical_df : DataFrame
            Columns ['dataset', 'model', 'ap'].
        available_models : list of str
            Ordered pool of candidate model names.
        score_matrices : dict[str -> ndarray], optional
            Raw detector scores per dataset (columns aligned with
            available_models).  Activates **production mode**.
        """
        self._available_models = list(available_models)

        if score_matrices is not None:
            self._fit_production(historical_df, available_models, score_matrices)

        # Lazy-init: share one _ELECTCore instance across all LODO folds
        cache_key = (self.random_state, self.base_path, self.intermediate_folder)
        if cache_key not in ELECT._processor_cache:
            if self.verbose:
                print("[ELECT] Initialising core processor (one-time)…")
            try:
                core = _ELECTCore(
                    random_state=self.random_state,
                    base_path=self.base_path,
                    intermediate_folder=self.intermediate_folder,
                )
                ELECT._processor_cache[cache_key] = core
                if self.verbose:
                    print(f"[ELECT] Ready — {core.valid_datasets} datasets")
            except Exception as exc:
                if self.verbose:
                    print(f"[ELECT] Warning: could not init core: {exc}")
                ELECT._processor_cache[cache_key] = None
        elif self.verbose and ELECT._processor_cache[cache_key] is not None:
            print("[ELECT] Reusing cached core processor")

        self._core = ELECT._processor_cache[cache_key]

    def _fit_production(
        self,
        historical_df: pd.DataFrame,
        available_models: List[str],
        score_matrices: Dict[str, np.ndarray],
    ):
        """Train LightGBM on ALL historical data using freshly-computed IPMs."""
        n_models      = len(available_models)
        hist_datasets = set(historical_df['dataset'].unique())
        datasets      = [ds for ds in score_matrices if ds in hist_datasets]
        if not datasets:
            return

        pivot = (
            historical_df[historical_df['dataset'].isin(datasets)]
            .pivot(index='dataset', columns='model', values='ap')
            .reindex(index=datasets, columns=available_models)
            .fillna(0.0)
        )
        ap_matrix = pivot.to_numpy(dtype=float)  # (n_ds, n_models)

        n_ds       = len(datasets)
        ipm_matrix = np.zeros((n_ds * 3, n_models), dtype=np.float64)
        valid_indices = []
        for pos, ds in enumerate(datasets):
            mat = np.asarray(score_matrices[ds], dtype=float)
            m   = min(mat.shape[1], n_models)
            if m < n_models:
                mat = np.hstack([mat[:, :m], np.zeros((mat.shape[0], n_models - m))])
            block = _compute_ipm_block(mat, n_models)
            if block is not None:
                ipm_matrix[pos * 3: (pos + 1) * 3] = block
                valid_indices.append(pos)

        if not valid_indices:
            if self.verbose:
                print("[ELECT][prod] No valid IPM blocks — production mode inactive.")
            return

        if self.verbose:
            print(f"[ELECT][prod] Building meta-training pairs "
                  f"({len(valid_indices)} datasets × {n_models} models)…")

        X, y = _build_meta_training_data(ap_matrix, ipm_matrix, valid_indices, n_models)
        if len(X) == 0:
            return

        self._prod_clf         = _fit_lgbm(X, y, self.random_state)
        self._prod_ap_matrix   = ap_matrix
        self._prod_ipm_matrix  = ipm_matrix
        self._prod_ds_names    = datasets
        self._prod_model_names = list(available_models)

        if self.verbose:
            print(f"[ELECT][prod] Ready — {len(valid_indices)} training datasets.")

    # ------------------------------------------------------------------
    # Select
    # ------------------------------------------------------------------

    def select(
        self,
        dataset_name: str,
        scores: Optional[np.ndarray] = None,
    ) -> Optional[Union[str, List[str]]]:
        """
        Select the best model.

        Parameters
        ----------
        dataset_name : str
            Label for the target dataset.
        scores : ndarray of shape (n_samples, n_models), optional
            Raw detector scores for a new, unseen dataset.  Triggers
            production mode.  When omitted, falls back to LODO benchmark
            mode (dataset_name must be in the canonical dataset list).
        """
        if self._available_models is None:
            raise RuntimeError("Call fit() before select().")

        if scores is not None:
            return self._select_production(scores)
        return self._select_lodo(dataset_name)

    # ------------------------------------------------------------------
    # Production-mode selection
    # ------------------------------------------------------------------

    def _select_production(self, scores: np.ndarray) -> Optional[Union[str, List[str]]]:
        if self._prod_clf is None or self._prod_ap_matrix is None:
            if self.verbose:
                print("[ELECT][prod] Not ready — fit with score_matrices first.")
            return None

        n_models = len(self._prod_model_names)
        scores   = np.asarray(scores, dtype=float)
        m = min(scores.shape[1], n_models)
        if m < n_models:
            scores = np.hstack([scores[:, :m], np.zeros((scores.shape[0], n_models - m))])

        new_ipm = _compute_ipm_block(scores, n_models)
        if new_ipm is None:
            return None

        n_hist  = len(self._prod_ds_names)
        aug_ipm = np.vstack([self._prod_ipm_matrix, new_ipm])
        aug_ap  = np.vstack([self._prod_ap_matrix, np.zeros((1, n_models))])

        selected_indices = self._run_online_selection(
            clf=self._prod_clf,
            ap_values=aug_ap,
            train_index=list(range(n_hist)),
            test_index=[n_hist],
            n_models=n_models,
            ipm_matrix=aug_ipm,
        )
        if not selected_indices:
            return None

        selected = [self._prod_model_names[i] for i in selected_indices[:self.n_selection]]
        return selected[0] if self.n_selection == 1 else selected

    def _run_online_selection(self, clf, ap_values, train_index, test_index, n_models, ipm_matrix):
        """Run BO using the given clf, reusing the shared core when available."""
        if self._core is not None:
            orig_clf       = self._core.clf
            self._core.clf = clf
            result = self._core.online_model_selection(
                ap_values=ap_values,
                train_index=train_index,
                test_index=test_index,
                n_models=n_models,
                internal_measure_mat=ipm_matrix,
            )
            self._core.clf = orig_clf
            return result

        # No shared core available: build a thin proxy
        proxy = _ELECTCore.__new__(_ELECTCore)
        proxy.clf            = clf
        proxy.random_state   = self.random_state
        proxy.n_iter         = ELECT_ITERATIONS
        proxy.n_sim_datasets = N_SIM_DATASETS
        proxy.conv_iter      = ELECT_CONV_ITER
        proxy._meta_data     = {}
        return proxy.online_model_selection(
            ap_values=ap_values,
            train_index=train_index,
            test_index=test_index,
            n_models=n_models,
            internal_measure_mat=ipm_matrix,
        )

    # ------------------------------------------------------------------
    # LODO benchmark mode
    # ------------------------------------------------------------------

    def _select_lodo(self, dataset_name: str) -> Optional[Union[str, List[str]]]:
        if self._core is None:
            return None
        if self.canonical_datasets is None or dataset_name not in self.canonical_datasets:
            if self.verbose:
                print(f"[ELECT] '{dataset_name}' not in canonical list.")
            return None

        test_idx      = self.canonical_datasets.index(dataset_name)
        n_canonical   = len(self.canonical_datasets)
        all_train_idx = [i for i in range(n_canonical) if i != test_idx]

        full_pool    = self._core.model_names
        pool_indices = [full_pool.index(m) for m in self._available_models if m in full_pool]
        valid_models = [m for m in self._available_models if m in full_pool]
        if not pool_indices:
            return None

        n_models = len(valid_models)

        # Load / cache canonical AP matrix
        ap_full_path = os.path.join(self.base_path, self.intermediate_folder, 'AP_full.xlsx')
        if ap_full_path not in ELECT._processor_cache:
            if not os.path.exists(ap_full_path):
                return None
            ap_df = pd.read_excel(ap_full_path, engine='openpyxl')
            ap_df = (
                ap_df.set_index('Dataset')
                .reindex(self.canonical_datasets)
                .reset_index()
            )
            ELECT._processor_cache[ap_full_path] = ap_df
        ap_df = ELECT._processor_cache[ap_full_path]

        ap_cols = [m for m in valid_models if m in ap_df.columns]
        if not ap_cols:
            return None
        ap_values = ap_df[ap_cols].to_numpy(dtype=float)

        subset_ipm = self._core._meta_data['ipm_matrix'][:, pool_indices]

        self._core.offline_training(
            n_datasets=n_canonical,
            n_models=n_models,
            test_dataset_idx=test_idx,
        )
        selected_indices = self._core.online_model_selection(
            ap_values=ap_values,
            train_index=all_train_idx,
            test_index=[test_idx],
            n_models=n_models,
            internal_measure_mat=subset_ipm,
        )
        if not selected_indices:
            return None

        selected = [valid_models[i] for i in selected_indices[:self.n_selection]]
        return selected[0] if self.n_selection == 1 else selected
