from abc import ABC, abstractmethod
from typing import List, Optional, Union
import numpy as np
import pandas as pd

class SingleModelSelector(ABC):
    @abstractmethod
    def fit(self, historical_df: pd.DataFrame, available_models: List[str]):
        """
        Offline phase: Prepare the selector using historical/training data and available models.

        Note: In LODO evaluation, `historical_df` is typically already filtered to exclude
        the current test dataset (i.e., the caller performs the split).
        """
        pass

    @abstractmethod
    def select(
        self,
        dataset_name: str,
        scores: Optional[np.ndarray] = None,
    ) -> Optional[Union[str, List[str]]]:
        """
        Online phase: Select a primary model (or list of models) for the given dataset.

        Parameters
        ----------
        dataset_name : str
            Name / label for the target dataset.
        scores : ndarray of shape (n_samples, n_models), optional
            Raw detector score matrix for the target dataset.  When provided,
            selectors that support it (e.g. ELECT) will compute
            internal performance measures on the fly so they can handle
            completely unseen datasets without pre-built lookup files.
        """
        pass
