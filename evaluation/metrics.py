"""
Evaluation metrics for Customer Lifetime Value Prediction.

Based on: Wang et al. (2019) "A Deep Probabilistic Model for Customer Lifetime Value Prediction"
Paper: https://arxiv.org/abs/1912.07753

Implements metrics from Section 4 of the paper:
1. Normalized Gini Coefficient (Section 4.1, pages 6-7)
2. Spearman's Rank Correlation (Table 1, page 9)
3. Decile-level MAPE (Table 3, page 11)
4. AUC-PR for binary classification (Table 4)
"""

import numpy as np
import torch
from typing import Tuple, Dict
from scipy.stats import spearmanr
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
    average_precision_score
)


def gini_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Gini coefficient for measuring discrimination.

    The Gini coefficient measures how well predictions rank-order the true values.
    It ranges from 0 (no discrimination) to 1 (perfect discrimination).

    Implementation based on the Lorenz curve approach:
    Gini = 2 * (AUC of Lorenz curve - 0.5)

    Parameters
    ----------
    y_true : np.ndarray
        True LTV values
    y_pred : np.ndarray
        Predicted LTV values

    Returns
    -------
    float
        Gini coefficient (0 to 1)

    References
    ----------
    Wang et al. (2019), Section 4.1, Equation 5
    """
    # Sort by predicted values
    sorted_indices = np.argsort(y_pred)
    y_true_sorted = y_true[sorted_indices]

    # Calculate cumulative sums
    n = len(y_true)
    cumsum_true = np.cumsum(y_true_sorted)

    # Gini coefficient formula
    # G = (2 * sum(i * y_i)) / (n * sum(y_i)) - (n + 1) / n
    sum_yi = cumsum_true[-1]
    if sum_yi == 0:
        return 0.0

    gini = (2.0 * np.sum((np.arange(1, n + 1) * y_true_sorted))) / (n * sum_yi) - (n + 1) / n

    return gini


def normalized_gini_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Normalized Gini Coefficient.

    The Normalized Gini Coefficient is the ratio of the model's Gini coefficient
    to the Gini coefficient of a perfect model (where predictions = true values).

    Normalized Gini = Gini(y_true, y_pred) / Gini(y_true, y_true)

    This metric is the PRIMARY evaluation metric used in the paper (Section 4.1).

    Parameters
    ----------
    y_true : np.ndarray
        True LTV values
    y_pred : np.ndarray
        Predicted LTV values

    Returns
    -------
    float
        Normalized Gini coefficient (0 to 1)

    References
    ----------
    Wang et al. (2019), Section 4.1, pages 6-7
    "The normalized Gini coefficient is defined as the ratio of the Gini coefficient
    of the model to the Gini coefficient of the perfect model."

    Examples
    --------
    Paper Results (Table 1, page 9):
    - DNN-ZILN: 0.368
    - DNN-MSE: 0.330
    - Relative improvement: 11.4%
    """
    gini_model = gini_coefficient(y_true, y_pred)
    gini_perfect = gini_coefficient(y_true, y_true)

    if gini_perfect == 0:
        return 0.0

    normalized_gini = gini_model / gini_perfect

    return normalized_gini


def decile_mape(y_true: np.ndarray, y_pred: np.ndarray, return_by_decile: bool = False) -> float:
    """
    Calculate Decile-level Mean Absolute Percentage Error (MAPE).

    This metric measures calibration by:
    1. Sorting samples by predicted LTV into 10 deciles
    2. Computing MAPE for each decile
    3. Averaging MAPE across deciles

    This is a key calibration metric from the paper (Table 3, page 11).

    Parameters
    ----------
    y_true : np.ndarray
        True LTV values
    y_pred : np.ndarray
        Predicted LTV values
    return_by_decile : bool
        If True, return MAPE for each decile (default: False)

    Returns
    -------
    float or Tuple[float, np.ndarray]
        Average MAPE across deciles, or (average, decile_mapes) if return_by_decile=True

    References
    ----------
    Wang et al. (2019), Section 4.2, Table 3, page 11

    Examples
    --------
    Paper Results (Table 3, page 11):
    - DNN-ZILN: 22.6%
    - DNN-MSE: 72.8%
    - Improvement: 68.9% reduction in error
    """
    # Sort by predicted values
    sorted_indices = np.argsort(y_pred)
    y_true_sorted = y_true[sorted_indices]
    y_pred_sorted = y_pred[sorted_indices]

    # Split into 10 deciles
    n = len(y_true)
    decile_size = n // 10

    decile_mapes = []

    for i in range(10):
        start_idx = i * decile_size
        if i == 9:  # Last decile includes remainder
            end_idx = n
        else:
            end_idx = (i + 1) * decile_size

        y_true_decile = y_true_sorted[start_idx:end_idx]
        y_pred_decile = y_pred_sorted[start_idx:end_idx]

        # Calculate MAPE for this decile
        # MAPE = mean(|y_true - y_pred| / |y_true|)
        # Add small epsilon to avoid division by zero
        eps = 1e-8
        mape = np.mean(np.abs(y_true_decile - y_pred_decile) / (np.abs(y_true_decile) + eps))
        decile_mapes.append(mape * 100)  # Convert to percentage

    avg_mape = np.mean(decile_mapes)

    if return_by_decile:
        return avg_mape, np.array(decile_mapes)

    return avg_mape


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_binary: np.ndarray = None
) -> Dict[str, float]:
    """
    Compute all evaluation metrics used in the paper.

    Parameters
    ----------
    y_true : np.ndarray
        True LTV values
    y_pred : np.ndarray
        Predicted LTV values
    y_binary : np.ndarray, optional
        Binary labels (0: non-returning, 1: returning) for AUC-PR calculation
        If None, derived from y_true (0 if y_true == 0, else 1)

    Returns
    -------
    Dict[str, float]
        Dictionary containing all metrics:
        - normalized_gini: Normalized Gini Coefficient (PRIMARY METRIC)
        - spearman: Spearman's Rank Correlation
        - decile_mape: Decile-level MAPE (%)
        - mae: Mean Absolute Error
        - rmse: Root Mean Squared Error
        - auc_pr: Area Under Precision-Recall Curve

    Examples
    --------
    >>> metrics = compute_all_metrics(y_true, y_pred)
    >>> print(f"Normalized Gini: {metrics['normalized_gini']:.3f}")
    >>> print(f"Decile MAPE: {metrics['decile_mape']:.1f}%")
    """
    # Check for NaN or Inf values
    if not np.all(np.isfinite(y_true)):
        nan_count = np.sum(~np.isfinite(y_true))
        raise ValueError(
            f"y_true contains {nan_count} NaN or Inf values. "
            f"Please check your data preprocessing."
        )

    if not np.all(np.isfinite(y_pred)):
        nan_count = np.sum(np.isnan(y_pred))
        inf_count = np.sum(np.isinf(y_pred))
        max_finite = np.max(y_pred[np.isfinite(y_pred)]) if np.any(np.isfinite(y_pred)) else 0

        error_msg = (
            f"Predictions contain invalid values:\n"
            f"  NaN values: {nan_count}\n"
            f"  Inf values: {inf_count}\n"
            f"  Max finite value: {max_finite:.2f}\n\n"
            f"This usually happens when:\n"
            f"  1. Model outputs (mu, sigma) become too large → exp() overflows\n"
            f"  2. Learning rate is too high → gradients explode\n"
            f"  3. Loss function has numerical instability\n\n"
            f"Solutions:\n"
            f"  1. Reduce learning rate (try 0.0001 instead of 0.001)\n"
            f"  2. Check model initialization\n"
            f"  3. Enable gradient clipping (already enabled in trainer)\n"
            f"  4. Inspect model outputs (mu, sigma) for extreme values"
        )
        raise ValueError(error_msg)

    metrics = {}

    # Primary metric from paper
    metrics['normalized_gini'] = normalized_gini_coefficient(y_true, y_pred)

    # Ranking metric
    spearman_corr, _ = spearmanr(y_true, y_pred)
    metrics['spearman'] = spearman_corr

    # Calibration metric
    metrics['decile_mape'] = decile_mape(y_true, y_pred)

    # Standard regression metrics
    metrics['mae'] = mean_absolute_error(y_true, y_pred)
    metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))

    # Binary classification metric (for returning vs non-returning)
    if y_binary is None:
        y_binary = (y_true > 0).astype(int)

    if len(np.unique(y_binary)) > 1:  # Need at least 2 classes
        metrics['auc_pr'] = average_precision_score(y_binary, y_pred)
    else:
        metrics['auc_pr'] = 0.0

    return metrics


def compute_metrics_from_model(
    model,
    data_loader,
    model_type: str = 'ziln',
    device: str = 'cpu'
) -> Dict[str, float]:
    """
    Compute all metrics directly from a model and data loader.

    Parameters
    ----------
    model : torch.nn.Module
        Trained model
    data_loader : torch.utils.data.DataLoader
        Data loader with (features, labels)
    model_type : str
        Type of model ('ziln' or 'simple')
    device : str
        Device to run on ('cpu' or 'cuda')

    Returns
    -------
    Dict[str, float]
        Dictionary of all metrics
    """
    model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x = batch_x.to(device)

            if model_type == 'ziln':
                # Get predictions from ZILN model
                p, mu, sigma = model(batch_x)
                # E[LTV] = p * exp(mu + sigma^2/2)
                y_pred = p * torch.exp(mu + sigma ** 2 / 2)
            else:
                # Simple model directly predicts LTV
                y_pred = model(batch_x)

            all_predictions.append(y_pred.cpu().numpy())
            all_labels.append(batch_y.numpy())

    y_pred = np.concatenate(all_predictions)
    y_true = np.concatenate(all_labels)

    return compute_all_metrics(y_true, y_pred)


def print_metrics_comparison(metrics_dict: Dict[str, Dict[str, float]]):
    """
    Print a formatted comparison of metrics across different models.

    Parameters
    ----------
    metrics_dict : Dict[str, Dict[str, float]]
        Dictionary mapping model names to their metrics

    Examples
    --------
    >>> metrics_dict = {
    ...     'DNN-ZILN': {'normalized_gini': 0.368, 'decile_mape': 22.6},
    ...     'DNN-MSE': {'normalized_gini': 0.330, 'decile_mape': 72.8}
    ... }
    >>> print_metrics_comparison(metrics_dict)
    """
    print("\n" + "="*80)
    print("METRICS COMPARISON")
    print("="*80)

    # Get all metric names from first model
    first_model = list(metrics_dict.keys())[0]
    metric_names = list(metrics_dict[first_model].keys())

    # Print header
    print(f"{'Metric':<20}", end="")
    for model_name in metrics_dict.keys():
        print(f"{model_name:>15}", end="")
    print()
    print("-"*80)

    # Print each metric
    for metric_name in metric_names:
        print(f"{metric_name:<20}", end="")
        for model_name in metrics_dict.keys():
            value = metrics_dict[model_name][metric_name]
            if 'mape' in metric_name.lower():
                print(f"{value:>14.1f}%", end="")
            else:
                print(f"{value:>15.4f}", end="")
        print()

    print("="*80 + "\n")
