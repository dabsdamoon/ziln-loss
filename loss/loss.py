"""
Loss functions for LTV prediction.

Implements:
- ZILN (Zero-Inflated Lognormal) Loss
- MSE Loss (for comparison)
- Additional losses for experimentation

Based on: Wang et al. (2019) "A Deep Probabilistic Model for Customer Lifetime Value Prediction"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ZILNLoss(nn.Module):
    """
    Zero-Inflated Lognormal Loss Function.

    The loss combines:
    - Binary Cross Entropy for zero vs non-zero classification
    - Lognormal loss for non-zero values

    Reference: Equation 3, Page 5 of Wang et al. (2019)
    """

    def __init__(self, eps: float = 1e-8):
        """
        Parameters
        ----------
        eps : float
            Small constant for numerical stability (default: 1e-8)
        """
        super(ZILNLoss, self).__init__()
        self.eps = eps

    def forward(
        self,
        y_true: torch.Tensor,
        p: torch.Tensor,
        mu: torch.Tensor,
        sigma: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute ZILN loss.

        Parameters
        ----------
        y_true : torch.Tensor
            True LTV values (shape: [batch_size])
        p : torch.Tensor
            Predicted probability of returning (shape: [batch_size])
        mu : torch.Tensor
            Predicted mean parameter of lognormal (shape: [batch_size])
        sigma : torch.Tensor
            Predicted std parameter of lognormal (shape: [batch_size])

        Returns
        -------
        torch.Tensor
            Mean loss over batch
        """
        # Create masks for zero and non-zero values
        zero_mask = (y_true == 0).float()
        nonzero_mask = (y_true > 0).float()

        # Binary Cross Entropy loss for zero vs non-zero
        # For zeros: -log(1-p), For non-zeros: -log(p)
        bce_loss = -zero_mask * torch.log(1 - p + self.eps) - \
                    nonzero_mask * torch.log(p + self.eps)

        # Lognormal loss for non-zero values
        # L_Lognormal(x; �, �) = log(x�2�) + (log x - �)�/(2ò)
        log_y = torch.log(y_true + self.eps)  # Add eps to avoid log(0)

        lognormal_loss = torch.log(sigma + self.eps) + \
                         0.5 * torch.log(torch.tensor(2 * np.pi)) + \
                         torch.log(y_true + self.eps) + \
                         ((log_y - mu) ** 2) / (2 * sigma ** 2 + self.eps)

        # Only apply lognormal loss to non-zero values
        lognormal_loss = nonzero_mask * lognormal_loss

        # Combine losses
        total_loss = bce_loss + lognormal_loss

        return total_loss.mean()


class MSELossForZILN(nn.Module):
    """
    MSE Loss for comparison with ZILN loss.

    For fair comparison with ZILN model, this loss expects the same
    output format (p, mu, sigma) but only uses the mean prediction.

    E[LTV] = p � exp(� + ò/2)
    Loss = MSE(y_true, E[LTV])
    """

    def __init__(self):
        super(MSELossForZILN, self).__init__()
        self.mse = nn.MSELoss()

    def forward(
        self,
        y_true: torch.Tensor,
        p: torch.Tensor,
        mu: torch.Tensor,
        sigma: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute MSE loss using predicted mean LTV.

        Parameters
        ----------
        y_true : torch.Tensor
            True LTV values
        p : torch.Tensor
            Predicted probability of returning
        mu : torch.Tensor
            Predicted mean parameter of lognormal
        sigma : torch.Tensor
            Predicted std parameter of lognormal

        Returns
        -------
        torch.Tensor
            MSE loss
        """
        # Compute expected LTV: p * exp(� + ò/2)
        expected_ltv = p * torch.exp(mu + sigma ** 2 / 2)

        return self.mse(expected_ltv, y_true)


class SimpleMSELoss(nn.Module):
    """
    Simple MSE Loss that directly predicts LTV (not using ZILN outputs).

    This is the traditional baseline approach.
    """

    def __init__(self):
        super(SimpleMSELoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute simple MSE loss.

        Parameters
        ----------
        y_true : torch.Tensor
            True LTV values
        y_pred : torch.Tensor
            Predicted LTV values

        Returns
        -------
        torch.Tensor
            MSE loss
        """
        return self.mse(y_pred, y_true)


class QuantileLoss(nn.Module):
    """
    Quantile Loss (Pinball Loss) for robust regression.

    This loss is less sensitive to outliers than MSE.
    At quantile=0.5, it becomes equivalent to MAE (median regression).
    """

    def __init__(self, quantile: float = 0.5):
        """
        Parameters
        ----------
        quantile : float
            Target quantile to predict (default: 0.5 for median)
        """
        super(QuantileLoss, self).__init__()
        assert 0 < quantile < 1, "Quantile must be between 0 and 1"
        self.quantile = quantile

    def forward(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute quantile loss.

        Parameters
        ----------
        y_true : torch.Tensor
            True values
        y_pred : torch.Tensor
            Predicted values

        Returns
        -------
        torch.Tensor
            Quantile loss
        """
        errors = y_true - y_pred
        loss = torch.max(
            self.quantile * errors,
            (self.quantile - 1) * errors
        )
        return loss.mean()


class HuberLoss(nn.Module):
    """
    Huber Loss for robust regression.

    Combines MSE for small errors and MAE for large errors.
    Less sensitive to outliers than pure MSE.
    """

    def __init__(self, delta: float = 1.0):
        """
        Parameters
        ----------
        delta : float
            Threshold at which to change between quadratic and linear loss
        """
        super(HuberLoss, self).__init__()
        self.delta = delta

    def forward(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Huber loss.

        Parameters
        ----------
        y_true : torch.Tensor
            True values
        y_pred : torch.Tensor
            Predicted values

        Returns
        -------
        torch.Tensor
            Huber loss
        """
        error = torch.abs(y_true - y_pred)
        quadratic = torch.min(error, torch.tensor(self.delta))
        linear = error - quadratic
        loss = 0.5 * quadratic ** 2 + self.delta * linear
        return loss.mean()


class LogCoshLoss(nn.Module):
    """
    Log-Cosh Loss for regression.

    Smoother than Huber loss and approximately equal to MAE for large errors.
    """

    def __init__(self):
        super(LogCoshLoss, self).__init__()

    def forward(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute log-cosh loss.

        Parameters
        ----------
        y_true : torch.Tensor
            True values
        y_pred : torch.Tensor
            Predicted values

        Returns
        -------
        torch.Tensor
            Log-cosh loss
        """
        error = y_pred - y_true
        loss = torch.log(torch.cosh(error))
        return loss.mean()


def get_loss_function(loss_name: str, **kwargs):
    """
    Factory function to get loss function by name.

    Parameters
    ----------
    loss_name : str
        Name of the loss function
    **kwargs : dict
        Additional arguments for the loss function

    Returns
    -------
    nn.Module
        Loss function

    Examples
    --------
    >>> ziln_loss = get_loss_function('ziln')
    >>> mse_loss = get_loss_function('mse')
    >>> quantile_loss = get_loss_function('quantile', quantile=0.9)
    """
    loss_registry = {
        'ziln': ZILNLoss,
        'mse': MSELossForZILN,
        'simple_mse': SimpleMSELoss,
        'quantile': QuantileLoss,
        'huber': HuberLoss,
        'logcosh': LogCoshLoss
    }

    if loss_name not in loss_registry:
        raise ValueError(
            f"Unknown loss function: {loss_name}. "
            f"Available: {list(loss_registry.keys())}"
        )

    return loss_registry[loss_name](**kwargs)
