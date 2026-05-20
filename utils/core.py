import os
import polars as pl
import numpy as np
from typing import List, Optional

# Resolve paths relative to the project root or this utils package
SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_PATH = os.path.join(SCRIPT_DIR, 'datasets/benchmark')

def create_score_loader(intermediate_files_folder: str = 'intermediate_files'):
    """
    Creates a score_loader function configured for a specific intermediate files folder.
    
    Args:
        intermediate_files_folder: Which intermediate files folder to use
        
    Returns:
        A score_loader function that uses the specified folder
    """
    def score_loader(dataset_name: str):
        """
        Load dataset scores and ground truth labels.

        Args:
            dataset_name: Name of the dataset to load

        Returns:
            tuple: (scores as numpy array, ground truth labels)

        Raises:
            ValueError: If dataset scores cannot be loaded
        """
        df_scores, y_true = load_dataset_scores(dataset_name, intermediate_files_folder)
        if df_scores is None:
            raise ValueError(f"Could not load scores for {dataset_name}")
        return df_scores.to_numpy(), y_true
    
    return score_loader


def get_ground_truth_labels(dataset_name: str, base_path: str = BASE_PATH) -> Optional[pl.Series]:
    """
    Loads ground truth labels for a specific dataset using Polars.
    """
    labels_dir = os.path.join(base_path, 'data')
    # Try standard naming convention first
    file_path = os.path.join(labels_dir, f"{dataset_name}_y.csv")
    
    if not os.path.exists(file_path):
        # Try lowercase .mat convention
        file_path = os.path.join(labels_dir, f"{dataset_name.lower()}.mat_y.csv")
    
    if not os.path.exists(file_path):
        return None

    try:
        labels = pl.read_csv(file_path, has_header=False).to_series()
        # Standardize labels to 0 (normal) and 1 (anomaly)
        # Assumes -1 is 0 (normal)
        if labels.dtype in [pl.Int64, pl.Int32, pl.Float64] and labels.min() == -1:
             labels = labels.map_elements(lambda x: 0 if x == -1 else 1, return_dtype=pl.Int64)
        return labels
    except Exception as e:
        print(f"Error loading labels for {dataset_name}: {e}")
        return None

def load_dataset_scores(dataset_name: str, intermediate_files_folder: str = 'intermediate_files') -> tuple[Optional[pl.DataFrame], Optional[np.ndarray]]:
    """
    Efficiently loads the entire score matrix for a dataset.
    
    Args:
        dataset_name: Name of the dataset
        intermediate_files_folder: Which intermediate files folder to use (default: 'intermediate_files')
    
    Returns: 
        (df_scores, y_true_array) or (None, None) if failed.
    """
    scores_dir = os.path.join(BASE_PATH, intermediate_files_folder, 'scores')

    file_path = os.path.join(scores_dir, f"{dataset_name}.csv")
    
    if not os.path.exists(file_path):
        file_path = os.path.join(scores_dir, f"{dataset_name.lower()}.mat.csv")
    
    if not os.path.exists(file_path):
        return None, None

    try:
        # Load scores
        df_scores = pl.read_csv(file_path, has_header=False)
        
        # Load labels
        labels_series = get_ground_truth_labels(dataset_name, BASE_PATH)
        
        if labels_series is None:
            return None, None
            
        return df_scores, labels_series.to_numpy()
    except Exception as e:
        print(f"Error loading scores for {dataset_name}: {e}")
        return None, None


def load_dataset_dims(dataset_name: str) -> tuple[int, int]:
    """
    Loads dataset dimensions (n_samples, n_features).
    Returns (0, 0) if file not found.
    """
    data_dir = os.path.join(BASE_PATH, 'data')
    
    # Try standard naming
    file_path = os.path.join(data_dir, f"{dataset_name}_X.csv")
    if not os.path.exists(file_path):
        # Try .mat naming
        file_path = os.path.join(data_dir, f"{dataset_name.lower()}.mat_X.csv")
    
    if not os.path.exists(file_path):
        return 0, 0
        
    try:
        # Read only first row to get shape if possible, but for n_samples we need full read
        # Polars is fast enough
        df = pl.read_csv(file_path, has_header=False)
        return df.shape
    except Exception as e:
        print(f"Error loading dims for {dataset_name}: {e}")
        return 0, 0
