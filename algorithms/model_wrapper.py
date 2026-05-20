from typing import List, Optional
import numpy as np
import pandas as pd
from .base import SingleModelSelector


class SinglePredictiveModelWrapper(SingleModelSelector):
    """
    A wrapper to include a standalone predictive model
    in the selector evaluation framework.

    This class bypasses the training (since the model is assumed pre-trained /
    pre-calculated) and simply returns its own name as the "selected model".
    """

    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name

    def fit(self, historical_df: pd.DataFrame, available_models: List[str], **kwargs):
        pass  # No training needed.

    def select(self, dataset_name: str, scores: Optional[np.ndarray] = None) -> str:
        return self.model_name

    def __repr__(self) -> str:
        return f"SinglePredictiveModelWrapper(model_name={self.model_name!r})"


class PyODPrimarySelector(SingleModelSelector):
    """
    Wraps any PyOD detector and uses it as a primary selector.

    At ``fit`` time nothing happens (no historical data is needed).
    At ``select`` time the detector is fitted on the raw score matrix
    ``scores`` (shape ``n_samples × n_models``) and the pool model whose
    scores are most correlated (Pearson r) with the detector's output is
    returned.

    Parameters
    ----------
    detector :
        An instantiated PyOD detector (e.g. ``IForest()``, ``LOF()``).
        Must implement ``fit(X)`` and expose ``decision_scores_``.
    """

    def __init__(self, detector, **kwargs):
        self.detector = detector

    def fit(self, historical_df: pd.DataFrame, available_models: List[str], **kwargs):
        self._available_models = available_models  # stored for reference

    def select(self, dataset_name: str, scores: Optional[np.ndarray] = None) -> str:
        if scores is None:
            raise ValueError(
                "PyODPrimarySelector.select() requires the score matrix via the "
                "'scores' keyword argument."
            )
        scores = np.asarray(scores, dtype=float)
        # Replace NaN / Inf before fitting
        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

        self.detector.fit(scores)
        detector_scores = self.detector.decision_scores_   # shape (n_samples,)

        # Correlate each pool-model column with the detector output
        correlations = np.array([
            float(np.corrcoef(scores[:, j], detector_scores)[0, 1])
            for j in range(scores.shape[1])
        ])
        # Replace NaN (constant column) with -inf so argmax skips them
        correlations = np.where(np.isfinite(correlations), correlations, -np.inf)
        best_idx = int(np.argmax(correlations))

        if hasattr(self, '_available_models') and best_idx < len(self._available_models):
            return self._available_models[best_idx]
        return str(best_idx)

    def __repr__(self) -> str:
        return f"PyODPrimarySelector(detector={self.detector!r})"
