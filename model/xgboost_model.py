"""
XGBoost model wrapper for LTV prediction with consistent interface.

This wrapper provides a consistent interface with PyTorch models,
allowing for easy comparison between XGBoost and neural network approaches.

Supports:
- Standard regression (predicts mean)
- Quantile regression (predicts quantiles for uncertainty estimation)
"""

import numpy as np
import xgboost as xgb
from typing import Optional, Dict, Any, List, Tuple
import pickle


class XGBoostLTVModel:
    """
    XGBoost wrapper for LTV prediction.

    This class provides a consistent interface with the PyTorch models
    for easy comparison and evaluation.
    """

    def __init__(
        self,
        objective: str = 'reg:squarederror',
        max_depth: int = 6,
        learning_rate: float = 0.1,
        n_estimators: int = 100,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_weight: int = 1,
        gamma: float = 0.0,
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0,
        random_state: int = 42,
        n_jobs: int = -1,
        **kwargs
    ):
        """
        Parameters
        ----------
        objective : str
            Learning objective (default: 'reg:squarederror')
        max_depth : int
            Maximum tree depth (default: 6)
        learning_rate : float
            Boosting learning rate (default: 0.1)
        n_estimators : int
            Number of boosting rounds (default: 100)
        subsample : float
            Subsample ratio of training instances (default: 0.8)
        colsample_bytree : float
            Subsample ratio of columns when constructing each tree (default: 0.8)
        min_child_weight : int
            Minimum sum of instance weight needed in a child (default: 1)
        gamma : float
            Minimum loss reduction required to make a split (default: 0.0)
        reg_alpha : float
            L1 regularization term on weights (default: 0.0)
        reg_lambda : float
            L2 regularization term on weights (default: 1.0)
        random_state : int
            Random seed (default: 42)
        n_jobs : int
            Number of parallel threads (default: -1 for all cores)
        **kwargs
            Additional XGBoost parameters
        """
        self.params = {
            'objective': objective,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'n_estimators': n_estimators,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'min_child_weight': min_child_weight,
            'gamma': gamma,
            'reg_alpha': reg_alpha,
            'reg_lambda': reg_lambda,
            'random_state': random_state,
            'n_jobs': n_jobs,
            **kwargs
        }

        self.model = None
        self.feature_names = None
        self.is_fitted = False

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        early_stopping_rounds: int = 10,
        verbose: bool = True
    ):
        """
        Fit XGBoost model.

        Parameters
        ----------
        X_train : np.ndarray
            Training features
        y_train : np.ndarray
            Training labels
        X_val : np.ndarray, optional
            Validation features for early stopping
        y_val : np.ndarray, optional
            Validation labels for early stopping
        early_stopping_rounds : int
            Number of rounds for early stopping (default: 10)
        verbose : bool
            Whether to print training progress (default: True)
        """
        # Create DMatrix for training
        dtrain = xgb.DMatrix(X_train, label=y_train)

        # Prepare evaluation sets
        evals = [(dtrain, 'train')]
        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val)
            evals.append((dval, 'val'))

        # Prepare parameters
        params = self.params.copy()
        n_estimators = params.pop('n_estimators')

        # Add verbosity control
        if verbose:
            verbose_eval = 10
        else:
            verbose_eval = False

        # Train model
        self.evals_result = {}

        if X_val is not None and y_val is not None:
            self.model = xgb.train(
                params=params,
                dtrain=dtrain,
                num_boost_round=n_estimators,
                evals=evals,
                early_stopping_rounds=early_stopping_rounds,
                evals_result=self.evals_result,
                verbose_eval=verbose_eval
            )
        else:
            self.model = xgb.train(
                params=params,
                dtrain=dtrain,
                num_boost_round=n_estimators,
                evals=evals,
                evals_result=self.evals_result,
                verbose_eval=verbose_eval
            )

        self.is_fitted = True

        if verbose:
            print(f"\nXGBoost training completed")
            print(f"Best iteration: {self.model.best_iteration if hasattr(self.model, 'best_iteration') else n_estimators}")

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

        dmatrix = xgb.DMatrix(X)
        predictions = self.model.predict(dmatrix)

        # Ensure non-negative predictions for LTV
        predictions = np.maximum(predictions, 0)

        return predictions

    def predict_ltv(self, X: np.ndarray) -> np.ndarray:
        """
        Predict LTV (alias for predict to match PyTorch model interface).

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

    def get_feature_importance(
        self,
        importance_type: str = 'weight'
    ) -> Dict[str, float]:
        """
        Get feature importance.

        Parameters
        ----------
        importance_type : str
            Type of importance ('weight', 'gain', 'cover', 'total_gain', 'total_cover')

        Returns
        -------
        Dict[str, float]
            Dictionary of feature importances
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet. Call fit() first.")

        return self.model.get_score(importance_type=importance_type)

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
            'params': self.params,
            'feature_names': self.feature_names,
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
        self.params = model_data['params']
        self.feature_names = model_data['feature_names']
        self.is_fitted = model_data['is_fitted']

    def __repr__(self) -> str:
        """String representation of the model."""
        params_str = ', '.join([f'{k}={v}' for k, v in list(self.params.items())[:5]])
        return f"XGBoostLTVModel({params_str}, ...)"


class XGBoostQuantileModel:
    """
    XGBoost Quantile Regression for LTV prediction with uncertainty estimation.

    This model trains multiple XGBoost models to predict different quantiles,
    allowing you to estimate prediction intervals and capture uncertainty.

    Example:
        model = XGBoostQuantileModel(quantiles=[0.1, 0.5, 0.9])
        model.fit(X_train, y_train, X_val, y_val)

        # Predict mean and intervals
        predictions = model.predict(X_test)
        # Returns: {
        #     'q_0.5': median predictions,
        #     'q_0.1': lower bound (10th percentile),
        #     'q_0.9': upper bound (90th percentile),
        #     'mean': average of quantiles
        # }
    """

    def __init__(
        self,
        quantiles: List[float] = [0.1, 0.5, 0.9],
        max_depth: int = 6,
        learning_rate: float = 0.1,
        n_estimators: int = 100,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_weight: int = 1,
        gamma: float = 0.0,
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0,
        random_state: int = 42,
        n_jobs: int = -1,
        **kwargs
    ):
        """
        Parameters
        ----------
        quantiles : List[float]
            List of quantiles to predict (default: [0.1, 0.5, 0.9])
            0.5 = median, 0.1 = 10th percentile, 0.9 = 90th percentile
        max_depth : int
            Maximum tree depth (default: 6)
        learning_rate : float
            Boosting learning rate (default: 0.1)
        n_estimators : int
            Number of boosting rounds (default: 100)
        subsample : float
            Subsample ratio of training instances (default: 0.8)
        colsample_bytree : float
            Subsample ratio of columns (default: 0.8)
        min_child_weight : int
            Minimum sum of instance weight needed in a child (default: 1)
        gamma : float
            Minimum loss reduction required to make a split (default: 0.0)
        reg_alpha : float
            L1 regularization (default: 0.0)
        reg_lambda : float
            L2 regularization (default: 1.0)
        random_state : int
            Random seed (default: 42)
        n_jobs : int
            Number of parallel threads (default: -1)
        """
        self.quantiles = sorted(quantiles)
        self.params = {
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'n_estimators': n_estimators,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'min_child_weight': min_child_weight,
            'gamma': gamma,
            'reg_alpha': reg_alpha,
            'reg_lambda': reg_lambda,
            'random_state': random_state,
            'n_jobs': n_jobs,
            **kwargs
        }

        self.models = {}  # Will store one model per quantile
        self.is_fitted = False

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        early_stopping_rounds: int = 10,
        verbose: bool = True
    ):
        """
        Fit quantile regression models for each quantile.

        Parameters
        ----------
        X_train : np.ndarray
            Training features
        y_train : np.ndarray
            Training labels
        X_val : np.ndarray, optional
            Validation features
        y_val : np.ndarray, optional
            Validation labels
        early_stopping_rounds : int
            Early stopping patience (default: 10)
        verbose : bool
            Print training progress (default: True)
        """
        dtrain = xgb.DMatrix(X_train, label=y_train)
        evals = [(dtrain, 'train')]

        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val)
            evals.append((dval, 'val'))

        if verbose:
            print(f"\nTraining XGBoost Quantile Regression for quantiles: {self.quantiles}")

        # Train one model per quantile
        for q in self.quantiles:
            if verbose:
                print(f"\n  Training quantile {q:.2f}...")

            # Set up quantile regression objective
            params = self.params.copy()
            params['objective'] = 'reg:quantileerror'
            params['quantile_alpha'] = q
            n_estimators = params.pop('n_estimators')

            evals_result = {}

            if X_val is not None and y_val is not None:
                model = xgb.train(
                    params=params,
                    dtrain=dtrain,
                    num_boost_round=n_estimators,
                    evals=evals,
                    early_stopping_rounds=early_stopping_rounds,
                    evals_result=evals_result,
                    verbose_eval=False
                )
            else:
                model = xgb.train(
                    params=params,
                    dtrain=dtrain,
                    num_boost_round=n_estimators,
                    evals=evals,
                    evals_result=evals_result,
                    verbose_eval=False
                )

            self.models[q] = model

            if verbose and hasattr(model, 'best_iteration'):
                print(f"    Best iteration: {model.best_iteration}")

        self.is_fitted = True

        if verbose:
            print(f"\nQuantile regression training completed")
            print(f"Trained {len(self.quantiles)} models (one per quantile)")

    def predict(
        self,
        X: np.ndarray,
        return_all: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Predict quantiles for input features.

        Parameters
        ----------
        X : np.ndarray
            Input features
        return_all : bool
            If True, return all quantiles. If False, return only median.

        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary with keys:
            - 'q_{quantile}': predictions for each quantile
            - 'mean': average across quantiles
            - 'median': median prediction (q_0.5 if available)
            - 'lower': lowest quantile prediction
            - 'upper': highest quantile prediction
            - 'width': prediction interval width (upper - lower)
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet. Call fit() first.")

        dmatrix = xgb.DMatrix(X)
        predictions = {}

        # Predict for each quantile
        for q in self.quantiles:
            preds = self.models[q].predict(dmatrix)
            preds = np.maximum(preds, 0)  # Ensure non-negative for LTV
            predictions[f'q_{q}'] = preds

        if return_all:
            # Compute derived statistics
            all_quantile_preds = np.array([predictions[f'q_{q}'] for q in self.quantiles])

            predictions['mean'] = np.mean(all_quantile_preds, axis=0)
            predictions['lower'] = predictions[f'q_{self.quantiles[0]}']
            predictions['upper'] = predictions[f'q_{self.quantiles[-1]}']
            predictions['width'] = predictions['upper'] - predictions['lower']

            # Median is either q_0.5 or the middle quantile
            if 0.5 in self.quantiles:
                predictions['median'] = predictions['q_0.5']
            else:
                mid_idx = len(self.quantiles) // 2
                predictions['median'] = predictions[f'q_{self.quantiles[mid_idx]}']

        return predictions

    def predict_ltv(self, X: np.ndarray) -> np.ndarray:
        """
        Predict LTV using median quantile (for compatibility with standard interface).

        Parameters
        ----------
        X : np.ndarray
            Input features

        Returns
        -------
        np.ndarray
            Median LTV predictions
        """
        preds = self.predict(X, return_all=True)
        return preds['median']

    def get_prediction_intervals(
        self,
        X: np.ndarray,
        confidence: float = 0.8
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get prediction intervals at specified confidence level.

        Parameters
        ----------
        X : np.ndarray
            Input features
        confidence : float
            Confidence level (default: 0.8 for 80% interval)

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            (median, lower_bound, upper_bound)
        """
        # Find quantiles that match the confidence level
        lower_q = (1 - confidence) / 2
        upper_q = 1 - lower_q

        # Find closest quantiles in our trained models
        lower_idx = np.argmin([abs(q - lower_q) for q in self.quantiles])
        upper_idx = np.argmin([abs(q - upper_q) for q in self.quantiles])

        preds = self.predict(X, return_all=True)

        lower = preds[f'q_{self.quantiles[lower_idx]}']
        upper = preds[f'q_{self.quantiles[upper_idx]}']
        median = preds['median']

        return median, lower, upper

    def save_model(self, filepath: str):
        """Save all quantile models to file."""
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet. Call fit() first.")

        model_data = {
            'models': self.models,
            'quantiles': self.quantiles,
            'params': self.params,
            'is_fitted': self.is_fitted
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

    def load_model(self, filepath: str):
        """Load all quantile models from file."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        self.models = model_data['models']
        self.quantiles = model_data['quantiles']
        self.params = model_data['params']
        self.is_fitted = model_data['is_fitted']

    def __repr__(self) -> str:
        """String representation."""
        return f"XGBoostQuantileModel(quantiles={self.quantiles}, n_models={len(self.models)})"
