"""
Linear Regression model wrapper for LTV prediction.

Simple baseline model using sklearn's LinearRegression with optional regularization.
Provides consistent interface with other models for easy comparison.
"""

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from typing import Optional, Literal
import pickle


class LinearRegressionModel:
    """
    Linear Regression wrapper for LTV prediction.

    This class provides a consistent interface with the PyTorch and XGBoost models
    for easy comparison and evaluation.

    Supports:
    - Standard Linear Regression (OLS)
    - Ridge Regression (L2 regularization)
    - Lasso Regression (L1 regularization)
    """

    def __init__(
        self,
        regularization: Literal['none', 'ridge', 'lasso'] = 'none',
        alpha: float = 1.0,
        fit_intercept: bool = True,
        **kwargs
    ):
        """
        Parameters
        ----------
        regularization : str
            Type of regularization ('none', 'ridge', 'lasso')
            - 'none': Standard linear regression (OLS)
            - 'ridge': Ridge regression (L2 regularization)
            - 'lasso': Lasso regression (L1 regularization)
        alpha : float
            Regularization strength (default: 1.0)
            Only used if regularization != 'none'
        fit_intercept : bool
            Whether to fit intercept (default: True)
        **kwargs
            Additional parameters passed to sklearn model
        """
        self.regularization = regularization
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.kwargs = kwargs

        # Create appropriate sklearn model
        if regularization == 'none':
            self.model = LinearRegression(fit_intercept=fit_intercept, **kwargs)
        elif regularization == 'ridge':
            self.model = Ridge(alpha=alpha, fit_intercept=fit_intercept, **kwargs)
        elif regularization == 'lasso':
            self.model = Lasso(alpha=alpha, fit_intercept=fit_intercept, **kwargs)
        else:
            raise ValueError(f"Unknown regularization: {regularization}. Use 'none', 'ridge', or 'lasso'")

        self.is_fitted = False

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        verbose: bool = True
    ):
        """
        Fit linear regression model.

        Parameters
        ----------
        X_train : np.ndarray
            Training features
        y_train : np.ndarray
            Training labels
        X_val : np.ndarray, optional
            Validation features (not used, for interface compatibility)
        y_val : np.ndarray, optional
            Validation labels (not used, for interface compatibility)
        verbose : bool
            Whether to print training info (default: True)
        """
        if verbose:
            print(f"\nTraining Linear Regression (regularization={self.regularization})")
            print(f"  Training samples: {len(X_train):,}")
            print(f"  Features: {X_train.shape[1]}")

        # Fit model
        self.model.fit(X_train, y_train)
        self.is_fitted = True

        if verbose:
            # Compute R² score
            train_score = self.model.score(X_train, y_train)
            print(f"  Training R²: {train_score:.4f}")

            if X_val is not None and y_val is not None:
                val_score = self.model.score(X_val, y_val)
                print(f"  Validation R²: {val_score:.4f}")

            # Print coefficient statistics
            coef_stats = {
                'mean': np.mean(np.abs(self.model.coef_)),
                'max': np.max(np.abs(self.model.coef_)),
                'non_zero': np.sum(self.model.coef_ != 0)
            }
            print(f"  Coefficient stats: mean_abs={coef_stats['mean']:.4f}, "
                  f"max_abs={coef_stats['max']:.4f}, non_zero={coef_stats['non_zero']}")

            if self.fit_intercept:
                print(f"  Intercept: {self.model.intercept_:.4f}")

        print(f"\nLinear regression training completed")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict LTV values.

        Parameters
        ----------
        X : np.ndarray
            Input features

        Returns
        -------
        np.ndarray
            Predicted LTV values
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet. Call fit() first.")

        predictions = self.model.predict(X)

        # Ensure non-negative predictions for LTV
        predictions = np.maximum(predictions, 0)

        return predictions

    def predict_ltv(self, X: np.ndarray) -> np.ndarray:
        """
        Predict LTV (alias for predict to match other models' interface).

        Parameters
        ----------
        X : np.ndarray
            Input features

        Returns
        -------
        np.ndarray
            Predicted LTV values
        """
        return self.predict(X)

    def get_feature_importance(self) -> np.ndarray:
        """
        Get feature importance (absolute coefficient values).

        Returns
        -------
        np.ndarray
            Array of absolute coefficient values
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet. Call fit() first.")

        return np.abs(self.model.coef_)

    def get_coefficients(self) -> dict:
        """
        Get model coefficients.

        Returns
        -------
        dict
            Dictionary with 'coefficients' and 'intercept'
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet. Call fit() first.")

        return {
            'coefficients': self.model.coef_,
            'intercept': self.model.intercept_ if self.fit_intercept else 0.0
        }

    def save_model(self, filepath: str):
        """
        Save model to file.

        Parameters
        ----------
        filepath : str
            Path to save the model
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet. Call fit() first.")

        model_data = {
            'model': self.model,
            'regularization': self.regularization,
            'alpha': self.alpha,
            'fit_intercept': self.fit_intercept,
            'is_fitted': self.is_fitted
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

    def load_model(self, filepath: str):
        """
        Load model from file.

        Parameters
        ----------
        filepath : str
            Path to load the model from
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        self.model = model_data['model']
        self.regularization = model_data['regularization']
        self.alpha = model_data['alpha']
        self.fit_intercept = model_data['fit_intercept']
        self.is_fitted = model_data['is_fitted']

    def __repr__(self) -> str:
        """String representation of the model."""
        if self.regularization == 'none':
            return "LinearRegressionModel(OLS)"
        else:
            return f"LinearRegressionModel({self.regularization}, alpha={self.alpha})"
