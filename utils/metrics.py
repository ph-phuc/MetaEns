import numpy as np
from typing import List, Dict, Any
from sklearn.metrics import average_precision_score, roc_auc_score

from .core import load_dataset_scores



def compute_all_metrics_for_dataset(dataset_name: str, available_models: List[str], intermediate_files_folder: str = 'intermediate_files') -> List[Dict[str, Any]]:
    """
    Computes AP and ROC-AUC for all models on a dataset.

    Args:
        dataset_name: Name of the dataset to process.
        available_models: List of model names to evaluate.
        intermediate_files_folder: Subfolder under datasets/benchmark/ containing scores/
            (default: 'intermediate_files').

    Returns:
        List of dicts with keys: dataset, model, ap, roc_auc.
    """
    try:
        df_scores, y_true = load_dataset_scores(dataset_name, intermediate_files_folder)
        if df_scores is None:
            return []

        scores_array = df_scores.to_numpy()

        dataset_results = []
        for idx, model_name in enumerate(available_models):
            if idx >= scores_array.shape[1]:
                continue
            y_scores = scores_array[:, idx]
            dataset_results.append({
                'dataset': dataset_name,
                'model': model_name,
                'ap': average_precision_score(y_true, y_scores),
                'roc_auc': roc_auc_score(y_true, y_scores),
            })
    except Exception as e:
        print(f"Error processing {dataset_name}: {e}")
        return []

    return dataset_results
