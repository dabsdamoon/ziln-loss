"""
Deep Neural Network Model for Customer Lifetime Value Prediction.

Based on: Wang et al. (2019) "A Deep Probabilistic Model for Customer Lifetime Value Prediction"
Paper: https://arxiv.org/abs/1912.07753

This model outputs three parameters (p, μ, σ) that can be used with
different loss functions (ZILN, MSE, etc.) for comparison.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List


class ZILNModel(nn.Module):
    """
    Deep Neural Network for LTV prediction with probabilistic outputs.

    Architecture:
    - Input features -> Hidden layers -> 3 output heads (p, mu, sigma)
    - p: probability of returning (sigmoid activation)
    - mu: mean of lognormal distribution (identity activation)
    - sigma: std of lognormal distribution (softplus activation)

    This architecture can be used with different loss functions:
    - ZILN Loss: Fully utilizes all three outputs
    - MSE Loss: Uses p, mu, sigma to compute E[LTV] = p * exp(mu + sigma²/2)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [64, 32],
        dropout_rate: float = 0.2,
        use_batch_norm: bool = True
    ):
        """
        Parameters
        ----------
        input_dim : int
            Number of input features
        hidden_dims : List[int]
            List of hidden layer dimensions (default: [64, 32])
        dropout_rate : float
            Dropout rate for regularization (default: 0.2)
        use_batch_norm : bool
            Whether to use batch normalization (default: True)
        """
        super(ZILNModel, self).__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm

        # Build hidden layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim

        self.hidden_layers = nn.Sequential(*layers)

        # Output heads (3 logits)
        self.fc_p = nn.Linear(prev_dim, 1)      # Probability of returning
        self.fc_mu = nn.Linear(prev_dim, 1)     # Mean of lognormal
        self.fc_sigma = nn.Linear(prev_dim, 1)  # Std of lognormal

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights using He initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with numerical stability.

        Parameters
        ----------
        x : torch.Tensor
            Input features (shape: [batch_size, input_dim])

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            p: probability of returning (shape: [batch_size])
            mu: mean parameter (shape: [batch_size])
            sigma: std parameter (shape: [batch_size])
        """
        # Shared hidden layers
        h = self.hidden_layers(x)

        # Output heads with appropriate activations and bounds for numerical stability
        p = torch.sigmoid(self.fc_p(h)).squeeze(-1)  # [0, 1]

        # Clip mu to prevent overflow in exp(mu + sigma^2/2)
        # exp(15) ≈ 3.3M, exp(-15) ≈ 0.0000003 - wide enough for most LTV ranges
        mu = torch.clamp(self.fc_mu(h).squeeze(-1), min=-15.0, max=15.0)

        # Clip sigma to prevent overflow and ensure minimum variance
        # Max sigma=5 means exp(5^2/2) = exp(12.5) ≈ 268k (safe from overflow)
        # Min sigma=0.01 prevents numerical instability with zero variance
        sigma = torch.clamp(F.softplus(self.fc_sigma(h)).squeeze(-1), min=0.01, max=5.0)

        return p, mu, sigma

    def predict_ltv(
        self,
        x: torch.Tensor,
        return_components: bool = False
    ) -> torch.Tensor:
        """
        Predict expected LTV with numerical stability.

        E[LTV] = p * E[LTV | returning] = p * exp(μ + σ²/2)

        Parameters
        ----------
        x : torch.Tensor
            Input features
        return_components : bool
            If True, return (expected_ltv, p, expected_ltv_returning)

        Returns
        -------
        torch.Tensor or Tuple
            Expected LTV or (expected_ltv, p, expected_ltv_returning)
        """
        self.eval()
        with torch.no_grad():
            p, mu, sigma = self.forward(x)

            # Expected value of lognormal: E[X] = exp(μ + σ²/2)
            # Clamp the exponent to prevent overflow (already bounded by forward, but extra safety)
            exponent = mu + sigma ** 2 / 2
            exponent = torch.clamp(exponent, max=20.0)  # exp(20) ≈ 485M, safe upper bound
            expected_ltv_returning = torch.exp(exponent)

            # Expected LTV: p * E[LTV | returning]
            expected_ltv = p * expected_ltv_returning

            # Replace any NaN or Inf with 0 as a safety measure
            expected_ltv = torch.where(
                torch.isfinite(expected_ltv),
                expected_ltv,
                torch.zeros_like(expected_ltv)
            )

            if return_components:
                return expected_ltv, p, expected_ltv_returning
            return expected_ltv

    def predict_quantile(
        self,
        x: torch.Tensor,
        quantile: float = 0.5
    ) -> torch.Tensor:
        """
        Predict LTV quantile.

        For lognormal distribution, the q-th quantile is:
        Q(q) = exp(μ + σ * Φ^(-1)(q))

        where Φ^(-1) is the inverse CDF of standard normal.

        Parameters
        ----------
        x : torch.Tensor
            Input features
        quantile : float
            Quantile to predict (default: 0.5 for median)

        Returns
        -------
        torch.Tensor
            Predicted LTV quantile
        """
        self.eval()
        with torch.no_grad():
            p, mu, sigma = self.forward(x)

            # Inverse CDF of standard normal
            z = torch.erfinv(2 * torch.tensor(quantile) - 1) * np.sqrt(2)

            # Quantile of lognormal
            quantile_returning = torch.exp(mu + sigma * z)

            # Adjust by probability of returning
            # (This is a simplification; exact quantile of mixture is more complex)
            return p * quantile_returning


class SimpleMLPModel(nn.Module):
    """
    Simple MLP model for direct LTV prediction (for comparison with ZILN).

    This model directly predicts LTV value without the probabilistic framework.
    Used as a baseline for comparison with ZILN model.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [64, 32],
        dropout_rate: float = 0.2,
        use_batch_norm: bool = True
    ):
        """
        Parameters
        ----------
        input_dim : int
            Number of input features
        hidden_dims : List[int]
            List of hidden layer dimensions (default: [64, 32])
        dropout_rate : float
            Dropout rate for regularization (default: 0.2)
        use_batch_norm : bool
            Whether to use batch normalization (default: True)
        """
        super(SimpleMLPModel, self).__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm

        # Build hidden layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        # ReLU to ensure non-negative predictions
        layers.append(nn.ReLU())

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights using He initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input features (shape: [batch_size, input_dim])

        Returns
        -------
        torch.Tensor
            Predicted LTV values (shape: [batch_size])
        """
        return self.network(x).squeeze(-1)

    def predict_ltv(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict LTV (same as forward for consistency with ZILNModel API).

        Parameters
        ----------
        x : torch.Tensor
            Input features

        Returns
        -------
        torch.Tensor
            Predicted LTV values
        """
        self.eval()
        with torch.no_grad():
            return self.forward(x)
