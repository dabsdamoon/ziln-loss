"""
Training script for LTV prediction models with flexible loss functions.

This script allows for comparison between different loss functions as done in the paper:
- ZILN Loss
- MSE Loss
- Other robust losses (Huber, Quantile, etc.)

Usage:
    # Train with ZILN loss (default)
    python train_ziln_model.py --loss_name ziln

    # Train with MSE loss for comparison
    python train_ziln_model.py --loss_name mse

    # Compare multiple losses
    python train_ziln_model.py --loss_name ziln --compare_with mse
"""

import argparse
import os
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from typing import Optional, Dict, List
import json
from datetime import datetime

from model import ZILNModel, SimpleMLPModel, XGBoostLTVModel
from loss import get_loss_function, ZILNLoss, MSELossForZILN, SimpleMSELoss
from evaluation import compute_all_metrics, compute_metrics_from_model


class LTVTrainer:
    """
    Flexible trainer for LTV prediction models.

    This trainer can use any loss function, allowing for easy comparison
    between different approaches (ZILN, MSE, etc.) as done in the paper.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        model_type: str = 'ziln',  # 'ziln' or 'simple'
        learning_rate: float = 0.001,
        weight_decay: float = 0.0,
        device: str = 'cpu',
        log_dir: Optional[str] = None,
        eval_interval: int = 5
    ):
        """
        Parameters
        ----------
        model : torch.nn.Module
            The model to train (ZILNModel or SimpleMLPModel)
        criterion : torch.nn.Module
            Loss function to optimize
        model_type : str
            Type of model ('ziln' or 'simple')
        learning_rate : float
            Learning rate for optimizer (default: 0.001)
        weight_decay : float
            L2 regularization strength (default: 0.0)
        device : str
            Device to use ('cpu' or 'cuda')
        log_dir : str, optional
            Directory for TensorBoard logs
        eval_interval : int
            Interval (in epochs) for computing comprehensive evaluation metrics
        """
        self.model = model.to(device)
        self.criterion = criterion
        self.model_type = model_type
        self.device = device
        self.eval_interval = eval_interval
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.train_losses = []
        self.val_losses = []
        self.eval_history = []

        # Initialize TensorBoard writer
        self.writer = None
        if log_dir is not None:
            self.writer = SummaryWriter(log_dir=log_dir)
            print(f"TensorBoard logging enabled: {log_dir}")
            print(f"  View logs with: tensorboard --logdir {Path(log_dir).parent}")

    def train_epoch(
        self,
        train_loader: torch.utils.data.DataLoader
    ) -> float:
        """
        Train for one epoch.

        Parameters
        ----------
        train_loader : DataLoader
            Training data loader

        Returns
        -------
        float
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            # Forward pass - different for ZILN vs simple models
            if self.model_type == 'ziln':
                p, mu, sigma = self.model(batch_x)

                # Different losses expect different inputs
                if isinstance(self.criterion, (ZILNLoss, MSELossForZILN)):
                    loss = self.criterion(batch_y, p, mu, sigma)
                else:
                    # For other losses, compute E[LTV] = p * exp(mu + sigma^2/2)
                    y_pred = p * torch.exp(mu + sigma ** 2 / 2)
                    loss = self.criterion(batch_y, y_pred)
            else:
                # Simple model directly predicts LTV
                y_pred = self.model(batch_x)
                loss = self.criterion(batch_y, y_pred)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Clip gradients to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        self.train_losses.append(avg_loss)

        return avg_loss

    def validate(
        self,
        val_loader: torch.utils.data.DataLoader
    ) -> float:
        """
        Validate model.

        Parameters
        ----------
        val_loader : DataLoader
            Validation data loader

        Returns
        -------
        float
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                # Forward pass
                if self.model_type == 'ziln':
                    p, mu, sigma = self.model(batch_x)

                    if isinstance(self.criterion, (ZILNLoss, MSELossForZILN)):
                        loss = self.criterion(batch_y, p, mu, sigma)
                    else:
                        y_pred = p * torch.exp(mu + sigma ** 2 / 2)
                        loss = self.criterion(batch_y, y_pred)
                else:
                    y_pred = self.model(batch_x)
                    loss = self.criterion(batch_y, y_pred)

                total_loss += loss.item()
                n_batches += 1

        avg_loss = total_loss / n_batches
        self.val_losses.append(avg_loss)

        return avg_loss

    def compute_comprehensive_metrics(
        self,
        data_loader: torch.utils.data.DataLoader,
        split_name: str = 'val'
    ) -> Dict[str, float]:
        """
        Compute comprehensive evaluation metrics with numerical stability checks.

        Parameters
        ----------
        data_loader : DataLoader
            Data loader for evaluation
        split_name : str
            Name of the split ('train', 'val', 'test')

        Returns
        -------
        Dict[str, float]
            Dictionary of metrics including:
            - normalized_gini (PRIMARY)
            - spearman
            - decile_mape
            - mae
            - rmse
            - auc_pr
        """
        self.model.eval()

        all_predictions = []
        all_labels = []
        all_mu = []
        all_sigma = []

        with torch.no_grad():
            for batch_x, batch_y in data_loader:
                batch_x = batch_x.to(self.device)

                if self.model_type == 'ziln':
                    # Get predictions from ZILN model
                    p, mu, sigma = self.model(batch_x)

                    # Store mu and sigma for diagnostics
                    all_mu.append(mu.cpu().numpy())
                    all_sigma.append(sigma.cpu().numpy())

                    # E[LTV] = p * exp(mu + sigma^2/2)
                    # Use model's predict_ltv which has safeguards
                    y_pred = self.model.predict_ltv(batch_x)
                else:
                    # Simple model directly predicts LTV
                    y_pred = self.model(batch_x)

                all_predictions.append(y_pred.cpu().numpy())
                all_labels.append(batch_y.numpy())

        y_pred = np.concatenate(all_predictions)
        y_true = np.concatenate(all_labels)

        # Diagnostic info for ZILN model
        if self.model_type == 'ziln' and len(all_mu) > 0:
            mu_all = np.concatenate(all_mu)
            sigma_all = np.concatenate(all_sigma)

            # Check for extreme values
            if np.any(np.abs(mu_all) > 10):
                print(f"\n⚠️  Warning: Large mu values detected (max={np.max(np.abs(mu_all)):.2f})")
                print(f"   This may indicate training instability")

            if np.any(sigma_all > 4):
                print(f"\n⚠️  Warning: Large sigma values detected (max={np.max(sigma_all):.2f})")
                print(f"   This may indicate training instability")

        # Check predictions before computing metrics
        if not np.all(np.isfinite(y_pred)):
            nan_count = np.sum(np.isnan(y_pred))
            inf_count = np.sum(np.isinf(y_pred))
            print(f"\n❌ ERROR: Invalid predictions detected:")
            print(f"   NaN: {nan_count}, Inf: {inf_count}")

            if self.model_type == 'ziln' and len(all_mu) > 0:
                print(f"\n   Model output statistics:")
                print(f"   mu: min={np.min(mu_all):.2f}, max={np.max(mu_all):.2f}, mean={np.mean(mu_all):.2f}")
                print(f"   sigma: min={np.min(sigma_all):.2f}, max={np.max(sigma_all):.2f}, mean={np.mean(sigma_all):.2f}")

        # Compute all metrics using the evaluation module
        try:
            metrics = compute_all_metrics(y_true, y_pred)
        except ValueError as e:
            print(f"\n❌ Metric computation failed: {e}")
            # Return dummy metrics to allow training to continue
            metrics = {
                'normalized_gini': 0.0,
                'spearman': 0.0,
                'decile_mape': 100.0,
                'mae': float('inf'),
                'rmse': float('inf'),
                'auc_pr': 0.0
            }

        return metrics

    def fit(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: Optional[torch.utils.data.DataLoader] = None,
        epochs: int = 100,
        early_stopping_patience: int = 10,
        verbose: bool = True,
        save_dir: Optional[Path] = None
    ):
        """
        Train the model with TensorBoard logging and interval-based evaluation.

        Parameters
        ----------
        train_loader : DataLoader
            Training data loader
        val_loader : DataLoader, optional
            Validation data loader
        epochs : int
            Maximum number of epochs (default: 100)
        early_stopping_patience : int
            Patience for early stopping (default: 10)
        verbose : bool
            Whether to print progress (default: True)
        save_dir : Path, optional
            Directory to save evaluation results
        """
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            # Train
            train_loss = self.train_epoch(train_loader)

            # Validate
            if val_loader is not None:
                val_loss = self.validate(val_loader)

                # Log basic losses to TensorBoard
                if self.writer is not None:
                    self.writer.add_scalar('Loss/train', train_loss, epoch)
                    self.writer.add_scalar('Loss/val', val_loss, epoch)

                # Compute comprehensive metrics at intervals
                should_eval = (epoch + 1) % self.eval_interval == 0 or epoch == 0 or epoch == epochs - 1
                if should_eval:
                    if verbose:
                        print(f"\n--- Epoch {epoch+1}/{epochs} - Comprehensive Evaluation ---")

                    # Compute metrics on validation set
                    val_metrics = self.compute_comprehensive_metrics(val_loader, 'val')

                    # Log to TensorBoard
                    if self.writer is not None:
                        for metric_name, metric_value in val_metrics.items():
                            self.writer.add_scalar(f'Metrics/val_{metric_name}', metric_value, epoch)

                    # Save evaluation results
                    eval_result = {
                        'epoch': epoch + 1,
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                        **{f'val_{k}': v for k, v in val_metrics.items()}
                    }
                    self.eval_history.append(eval_result)

                    if save_dir is not None:
                        eval_df = pd.DataFrame(self.eval_history)
                        eval_df.to_csv(save_dir / 'eval_history.csv', index=False)

                    if verbose:
                        print(f"  Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
                        print(f"  Normalized Gini: {val_metrics['normalized_gini']:.4f} (PRIMARY)")
                        print(f"  Spearman Corr: {val_metrics['spearman']:.4f}")
                        print(f"  Decile MAPE: {val_metrics['decile_mape']:.1f}%")
                        print(f"  MAE: {val_metrics['mae']:.2f}, RMSE: {val_metrics['rmse']:.2f}")
                        print(f"  AUC-PR: {val_metrics['auc_pr']:.4f}")
                else:
                    if verbose:
                        print(f"Epoch {epoch+1}/{epochs} - "
                              f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model
                    self.best_model_state = self.model.state_dict()
                else:
                    patience_counter += 1

                if patience_counter >= early_stopping_patience:
                    if verbose:
                        print(f"\nEarly stopping at epoch {epoch+1}")
                    # Restore best model
                    self.model.load_state_dict(self.best_model_state)
                    break
            else:
                if verbose:
                    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}")

        # Close TensorBoard writer
        if self.writer is not None:
            self.writer.close()


def prepare_features(df: pd.DataFrame, scalers: dict = None, is_train: bool = True):
    """
    Prepare features for training with one-hot encoding and min-max normalization.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    scalers : dict, optional
        Dictionary of fitted scalers/encoders (for test set)
    is_train : bool
        Whether this is training data

    Returns
    -------
    np.ndarray, np.ndarray, dict
        X, y, scalers
    """
    df = df.copy()

    # Separate features and target
    target_col = 'future_12m_purchase_value'

    # Define categorical and numerical columns
    categorical_cols = ['chain', 'initial_category', 'initial_brand']
    numerical_cols = ['initial_purchase_amount', 'initial_num_items']

    # Handle productmeasure if present
    if 'initial_productmeasure' in df.columns:
        categorical_cols.append('initial_productmeasure')

    # Filter to only available columns
    categorical_cols = [col for col in categorical_cols if col in df.columns]
    numerical_cols = [col for col in numerical_cols if col in df.columns]

    if is_train:
        scalers = {}

        # One-hot encoding for categorical features
        if len(categorical_cols) > 0:
            # Convert to string and handle missing values
            for col in categorical_cols:
                df[col] = df[col].astype(str).fillna('missing')

            # Fit one-hot encoder
            onehot_encoder = OneHotEncoder(
                sparse_output=False,
                handle_unknown='ignore',  # Handle unseen categories in test set
                dtype=np.float32
            )
            categorical_encoded = onehot_encoder.fit_transform(df[categorical_cols])
            scalers['onehot_encoder'] = onehot_encoder

            # Get feature names for debugging
            try:
                feature_names = onehot_encoder.get_feature_names_out(categorical_cols)
                scalers['onehot_feature_names'] = feature_names
            except:
                pass
        else:
            categorical_encoded = np.array([]).reshape(len(df), 0)

        # Min-max scaling for numerical features
        if len(numerical_cols) > 0:
            minmax_scaler = MinMaxScaler()
            numerical_scaled = minmax_scaler.fit_transform(df[numerical_cols])
            scalers['minmax_scaler'] = minmax_scaler
        else:
            numerical_scaled = np.array([]).reshape(len(df), 0)

        # Combine features
        X = np.hstack([numerical_scaled, categorical_encoded]).astype(np.float32)

    else:
        # Transform using fitted scalers/encoders

        # One-hot encoding for categorical features
        if len(categorical_cols) > 0 and 'onehot_encoder' in scalers:
            # Convert to string and handle missing values
            for col in categorical_cols:
                df[col] = df[col].astype(str).fillna('missing')

            categorical_encoded = scalers['onehot_encoder'].transform(df[categorical_cols])
        else:
            categorical_encoded = np.array([]).reshape(len(df), 0)

        # Min-max scaling for numerical features
        if len(numerical_cols) > 0 and 'minmax_scaler' in scalers:
            numerical_scaled = scalers['minmax_scaler'].transform(df[numerical_cols])
        else:
            numerical_scaled = np.array([]).reshape(len(df), 0)

        # Combine features
        X = np.hstack([numerical_scaled, categorical_encoded]).astype(np.float32)

    y = df[target_col].values.astype(np.float32)

    return X, y, scalers


def create_dataloaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int = 256,
    val_split: float = 0.2
):
    """
    Create PyTorch dataloaders.

    Parameters
    ----------
    X_train, y_train : np.ndarray
        Training data
    X_test, y_test : np.ndarray
        Test data
    batch_size : int
        Batch size (default: 256)
    val_split : float
        Validation split ratio (default: 0.2)

    Returns
    -------
    train_loader, val_loader, test_loader
    """
    # Split train into train and validation
    n_val = int(len(X_train) * val_split)
    indices = np.random.permutation(len(X_train))

    val_indices = indices[:n_val]
    train_indices = indices[n_val:]

    X_train_split = X_train[train_indices]
    y_train_split = y_train[train_indices]
    X_val = X_train[val_indices]
    y_val = y_train[val_indices]

    # Create datasets
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train_split),
        torch.FloatTensor(y_train_split)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.FloatTensor(y_val)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test),
        torch.FloatTensor(y_test)
    )

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def evaluate_model(model, test_loader, model_type='ziln', device='cpu'):
    """
    Evaluate model on test set using comprehensive metrics.

    Parameters
    ----------
    model : torch.nn.Module
        Trained model
    test_loader : DataLoader
        Test data loader
    model_type : str
        Type of model ('ziln' or 'simple')
    device : str
        Device to use

    Returns
    -------
    dict, np.ndarray, np.ndarray
        Evaluation metrics, predictions, true values
    """
    model.eval()
    all_preds = []
    all_true = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)

            if model_type == 'ziln':
                # Get predictions from ZILN model
                p, mu, sigma = model(batch_x)
                # E[LTV] = p * exp(mu + sigma^2/2)
                preds = p * torch.exp(mu + sigma ** 2 / 2)
            else:
                preds = model.predict_ltv(batch_x)

            all_preds.append(preds.cpu().numpy())
            all_true.append(batch_y.numpy())

    all_preds = np.concatenate(all_preds)
    all_true = np.concatenate(all_true)

    # Use comprehensive evaluation metrics from the paper
    metrics = compute_all_metrics(all_true, all_preds)

    return metrics, all_preds, all_true


def plot_results(train_losses, val_losses, output_dir, loss_name):
    """Plot training curves."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation Loss ({loss_name})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(Path(output_dir) / f'training_curves_{loss_name}.png', dpi=300, bbox_inches='tight')
    plt.close()


def train_xgboost_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    args,
    run_path: Path
):
    """
    Train XGBoost model for comparison with eval_history tracking.

    Parameters
    ----------
    X_train, y_train : np.ndarray
        Training data
    X_val, y_val : np.ndarray
        Validation data
    X_test, y_test : np.ndarray
        Test data
    args : argparse.Namespace
        Training arguments
    run_path : Path
        Directory to save results

    Returns
    -------
    XGBoostLTVModel, dict, np.ndarray, np.ndarray
        Trained model, metrics, predictions, true values
    """
    print("\n" + "=" * 60)
    print("Training XGBoost Model")
    print("=" * 60)

    # Create XGBoost model
    model = XGBoostLTVModel(
        objective='reg:squarederror',
        max_depth=args.xgb_max_depth,
        learning_rate=args.learning_rate,
        n_estimators=args.xgb_n_estimators,
        subsample=args.xgb_subsample,
        colsample_bytree=args.xgb_colsample_bytree,
        min_child_weight=args.xgb_min_child_weight,
        gamma=args.xgb_gamma,
        reg_alpha=args.xgb_reg_alpha,
        reg_lambda=args.xgb_reg_lambda,
        random_state=42
    )

    print(f"Model: {model}")

    # Train model with periodic evaluation
    print("\nTraining with periodic evaluation...")
    eval_interval = args.eval_interval if hasattr(args, 'eval_interval') else 10
    n_estimators = args.xgb_n_estimators

    # Initialize eval history
    eval_history = []

    # Train incrementally to track metrics at intervals
    import xgboost as xgb
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    params = {
        'objective': 'reg:squarederror',
        'max_depth': args.xgb_max_depth,
        'learning_rate': args.learning_rate,
        'subsample': args.xgb_subsample,
        'colsample_bytree': args.xgb_colsample_bytree,
        'min_child_weight': args.xgb_min_child_weight,
        'gamma': args.xgb_gamma,
        'reg_alpha': args.xgb_reg_alpha,
        'reg_lambda': args.xgb_reg_lambda,
        'random_state': 42,
        'n_jobs': -1
    }

    evals_result = {}
    xgb_model = None
    best_iteration = 0
    best_val_loss = float('inf')
    patience_counter = 0

    for current_round in range(0, n_estimators, eval_interval):
        rounds_to_train = min(eval_interval, n_estimators - current_round)

        # Train for eval_interval rounds
        xgb_model = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=rounds_to_train,
            evals=[(dtrain, 'train'), (dval, 'val')],
            xgb_model=xgb_model,  # Continue from previous model
            evals_result=evals_result,
            verbose_eval=False
        )

        iteration = current_round + rounds_to_train

        # Get train and val loss
        train_loss = evals_result['train']['rmse'][-1]
        val_loss = evals_result['val']['rmse'][-1]

        # Compute comprehensive metrics on validation set
        val_preds = xgb_model.predict(dval)
        val_preds = np.maximum(val_preds, 0)  # Ensure non-negative
        val_metrics = compute_all_metrics(y_val, val_preds)

        # Save to eval history
        eval_result = {
            'iteration': iteration,
            'train_loss': train_loss,
            'val_loss': val_loss,
            **{f'val_{k}': v for k, v in val_metrics.items()}
        }
        eval_history.append(eval_result)

        # Print progress
        if iteration % (eval_interval * 2) == 0 or iteration == n_estimators:
            print(f"\n--- Iteration {iteration}/{n_estimators} - Comprehensive Evaluation ---")
            print(f"  Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            print(f"  Normalized Gini: {val_metrics['normalized_gini']:.4f} (PRIMARY)")
            print(f"  Spearman Corr: {val_metrics['spearman']:.4f}")
            print(f"  Decile MAPE: {val_metrics['decile_mape']:.1f}%")
            print(f"  MAE: {val_metrics['mae']:.2f}, RMSE: {val_metrics['rmse']:.2f}")
            print(f"  AUC-PR: {val_metrics['auc_pr']:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_iteration = iteration
            patience_counter = 0
        else:
            patience_counter += eval_interval

        if patience_counter >= args.early_stopping_patience:
            print(f"\nEarly stopping at iteration {iteration}")
            break

    # Store the trained model in our wrapper
    model.model = xgb_model
    model.is_fitted = True
    model.evals_result = evals_result

    # Save eval_history.csv (same format as PyTorch models)
    eval_df = pd.DataFrame(eval_history)
    eval_history_path = run_path / 'eval_history.csv'
    eval_df.to_csv(eval_history_path, index=False)
    print(f"\nEvaluation history saved to: {eval_history_path}")

    print(f"Best iteration: {best_iteration} (val_loss: {best_val_loss:.4f})")

    # Plot training curves (XGBoost style)
    if hasattr(model, 'evals_result') and model.evals_result:
        plt.figure(figsize=(10, 6))

        # Get loss metric name (first metric in results)
        metric_name = list(model.evals_result['train'].keys())[0]

        train_losses = model.evals_result['train'][metric_name]
        val_losses = model.evals_result['val'][metric_name] if 'val' in model.evals_result else None

        plt.plot(train_losses, label='Train Loss')
        if val_losses:
            plt.plot(val_losses, label='Val Loss')

        plt.xlabel('Boosting Round')
        plt.ylabel('Loss')
        plt.title('XGBoost Training and Validation Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(run_path / 'training_curves_xgboost.png', dpi=300, bbox_inches='tight')
        plt.close()

    # Evaluate on test set
    print("\n" + "=" * 60)
    print("FINAL EVALUATION ON TEST SET")
    print("=" * 60)

    test_preds = model.predict(X_test)
    metrics = compute_all_metrics(y_test, test_preds)

    print("\nTest Set Metrics (from paper):")
    print("-" * 60)
    print(f"  Normalized Gini: {metrics['normalized_gini']:.4f} (PRIMARY METRIC)")
    print(f"  Spearman Correlation: {metrics['spearman']:.4f}")
    print(f"  Decile MAPE: {metrics['decile_mape']:.1f}%")
    print(f"  MAE: {metrics['mae']:.2f}")
    print(f"  RMSE: {metrics['rmse']:.2f}")
    print(f"  AUC-PR: {metrics['auc_pr']:.4f}")
    print("-" * 60)

    # Save model
    model_path = run_path / 'model_best.pkl'
    model.save_model(str(model_path))
    print(f"\nModel saved to: {model_path}")

    # Save predictions
    results_df = pd.DataFrame({
        'true_ltv': y_test,
        'predicted_ltv': test_preds
    })
    predictions_path = run_path / 'test_predictions.csv'
    results_df.to_csv(predictions_path, index=False)
    print(f"Predictions saved to: {predictions_path}")

    # Save feature importance
    try:
        importance = model.get_feature_importance(importance_type='gain')
        importance_df = pd.DataFrame([
            {'feature': k, 'importance': v}
            for k, v in importance.items()
        ]).sort_values('importance', ascending=False)

        importance_path = run_path / 'feature_importance.csv'
        importance_df.to_csv(importance_path, index=False)
        print(f"Feature importance saved to: {importance_path}")

        # Plot feature importance
        plt.figure(figsize=(10, 6))
        importance_df_top = importance_df.head(20)
        plt.barh(range(len(importance_df_top)), importance_df_top['importance'])
        plt.yticks(range(len(importance_df_top)), importance_df_top['feature'])
        plt.xlabel('Importance (Gain)')
        plt.title('Top 20 Feature Importance')
        plt.tight_layout()
        plt.savefig(run_path / 'feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"Could not save feature importance: {e}")

    # Save training configuration
    config_path = run_path / 'config.json'
    metrics_serializable = {k: float(v) for k, v in metrics.items()}

    config = {
        'model_type': 'xgboost',
        'max_depth': args.xgb_max_depth,
        'learning_rate': args.learning_rate,
        'n_estimators': args.xgb_n_estimators,
        'subsample': args.xgb_subsample,
        'colsample_bytree': args.xgb_colsample_bytree,
        'min_child_weight': args.xgb_min_child_weight,
        'gamma': args.xgb_gamma,
        'reg_alpha': args.xgb_reg_alpha,
        'reg_lambda': args.xgb_reg_lambda,
        'input_dim': int(X_train.shape[1]),
        'train_samples': int(len(X_train)),
        'test_samples': int(len(X_test)),
        'final_metrics': metrics_serializable
    }
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Configuration saved to: {config_path}")

    return model, metrics, test_preds, y_test


def main():
    parser = argparse.ArgumentParser(
        description='Train LTV prediction model with flexible loss functions'
    )

    parser.add_argument(
        '--train_path',
        type=str,
        default='data/processed/train.parquet',
        help='Path to training data'
    )
    parser.add_argument(
        '--test_path',
        type=str,
        default='data/processed/test.parquet',
        help='Path to test data'
    )
    parser.add_argument(
        '--loss_name',
        type=str,
        default='ziln',
        choices=['ziln', 'mse', 'simple_mse', 'quantile', 'huber', 'logcosh'],
        help='Loss function to use (default: ziln)'
    )
    parser.add_argument(
        '--model_type',
        type=str,
        default='auto',
        choices=['auto', 'ziln', 'simple', 'xgboost'],
        help='Model type (auto, ziln, simple, or xgboost). Auto chooses based on loss.'
    )
    parser.add_argument(
        '--hidden_dims',
        type=int,
        nargs='+',
        default=[64, 32],
        help='Hidden layer dimensions'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=256,
        help='Batch size'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Maximum number of epochs'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.001,
        help='Learning rate'
    )
    parser.add_argument(
        '--dropout_rate',
        type=float,
        default=0.2,
        help='Dropout rate'
    )
    parser.add_argument(
        '--early_stopping_patience',
        type=int,
        default=100,
        help='Early stopping patience'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        help='Device to use (cpu or cuda)'
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        default=None,
        help='TensorBoard log directory (default: runs/<loss_name>_<timestamp>)'
    )
    parser.add_argument(
        '--eval_interval',
        type=int,
        default=5,
        help='Interval (in epochs) for comprehensive evaluation (default: 5)'
    )

    # XGBoost-specific arguments
    parser.add_argument(
        '--xgb_max_depth',
        type=int,
        default=6,
        help='XGBoost max tree depth (default: 6)'
    )
    parser.add_argument(
        '--xgb_n_estimators',
        type=int,
        default=100,
        help='XGBoost number of boosting rounds (default: 100)'
    )
    parser.add_argument(
        '--xgb_subsample',
        type=float,
        default=0.8,
        help='XGBoost subsample ratio (default: 0.8)'
    )
    parser.add_argument(
        '--xgb_colsample_bytree',
        type=float,
        default=0.8,
        help='XGBoost column subsample ratio (default: 0.8)'
    )
    parser.add_argument(
        '--xgb_min_child_weight',
        type=int,
        default=1,
        help='XGBoost min child weight (default: 1)'
    )
    parser.add_argument(
        '--xgb_gamma',
        type=float,
        default=0.0,
        help='XGBoost gamma (min split loss) (default: 0.0)'
    )
    parser.add_argument(
        '--xgb_reg_alpha',
        type=float,
        default=0.0,
        help='XGBoost L1 regularization (default: 0.0)'
    )
    parser.add_argument(
        '--xgb_reg_lambda',
        type=float,
        default=1.0,
        help='XGBoost L2 regularization (default: 1.0)'
    )

    args = parser.parse_args()

    # Create TensorBoard log directory (this will be our main experiment directory)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if args.log_dir is None:
        model_name = args.model_type if args.model_type != 'auto' else args.loss_name
        log_dir = f'runs/{model_name}_{timestamp}'
    else:
        log_dir = args.log_dir

    # Create the run directory (this is our only output directory now)
    run_path = Path(log_dir)
    run_path.mkdir(parents=True, exist_ok=True)

    model_display = args.model_type.upper() if args.model_type != 'auto' else args.loss_name.upper()
    print("=" * 60)
    print(f"LTV Prediction Model Training - {model_display}")
    print("=" * 60)
    print(f"Experiment directory: {log_dir}")
    print(f"  - TensorBoard logs: {log_dir}/")
    print(f"  - Model checkpoints: {log_dir}/")
    print(f"  - Results: {log_dir}/")

    # Load data
    print("\nLoading data...")
    train_df = pd.read_parquet(args.train_path)
    test_df = pd.read_parquet(args.test_path)

    print(f"Train set: {len(train_df):,} samples")
    print(f"Test set: {len(test_df):,} samples")

    # Prepare features
    print("\nPreparing features...")
    print("  - Numerical features: Min-Max normalization [0, 1]")
    print("  - Categorical features: One-hot encoding")
    X_train, y_train, scalers = prepare_features(train_df, is_train=True)
    X_test, y_test, _ = prepare_features(test_df, scalers=scalers, is_train=False)

    # Print feature engineering details
    print(f"\nFeature Engineering Summary:")
    print(f"  Total features: {X_train.shape[1]}")

    # Count numerical vs categorical features
    if 'minmax_scaler' in scalers:
        n_numerical = scalers['minmax_scaler'].n_features_in_
        print(f"  - Numerical (min-max scaled): {n_numerical}")
    else:
        n_numerical = 0

    if 'onehot_encoder' in scalers:
        n_categorical_original = scalers['onehot_encoder'].n_features_in_
        n_categorical_encoded = len(scalers['onehot_encoder'].get_feature_names_out())
        print(f"  - Categorical (original): {n_categorical_original}")
        print(f"  - Categorical (one-hot encoded): {n_categorical_encoded}")
    else:
        n_categorical_encoded = 0

    print(f"  Formula: {n_numerical} + {n_categorical_encoded} = {X_train.shape[1]}")

    # Auto-select model type based on loss
    if args.model_type == 'auto':
        model_type = 'ziln' if args.loss_name in ['ziln', 'mse'] else 'simple'
    else:
        model_type = args.model_type

    # Handle XGBoost training separately (doesn't use PyTorch dataloaders)
    if model_type == 'xgboost':
        # Split train into train and validation
        n_val = int(len(X_train) * 0.2)
        indices = np.random.permutation(len(X_train))
        val_indices = indices[:n_val]
        train_indices = indices[n_val:]

        X_train_split = X_train[train_indices]
        y_train_split = y_train[train_indices]
        X_val = X_train[val_indices]
        y_val = y_train[val_indices]

        # Train XGBoost model
        model, metrics, preds, true = train_xgboost_model(
            X_train=X_train_split,
            y_train=y_train_split,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            args=args,
            run_path=run_path
        )

        print("\n" + "=" * 60)
        print("Training Complete!")
        print("=" * 60)
        print(f"\nExperiment saved to: {run_path}")
        print(f"\nContents:")
        print(f"  - Model checkpoint: model_best.pkl")
        print(f"  - Predictions: test_predictions.csv")
        print(f"  - Evaluation history: eval_history.csv")
        print(f"  - Training curves: training_curves_xgboost.png")
        print(f"  - Feature importance: feature_importance.csv, feature_importance.png")
        print(f"  - Configuration: config.json")
        print("=" * 60)

        return

    # Create dataloaders for PyTorch models
    print("\nCreating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        X_train, y_train, X_test, y_test, batch_size=args.batch_size
    )

    # Create model
    print(f"\nInitializing {model_type.upper()} model...")
    if model_type == 'ziln':
        model = ZILNModel(
            input_dim=X_train.shape[1],
            hidden_dims=args.hidden_dims,
            dropout_rate=args.dropout_rate
        )
    else:
        model = SimpleMLPModel(
            input_dim=X_train.shape[1],
            hidden_dims=args.hidden_dims,
            dropout_rate=args.dropout_rate
        )

    print(f"Model architecture: {model}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create loss function
    print(f"\nUsing {args.loss_name.upper()} loss")
    criterion = get_loss_function(args.loss_name)

    # Create trainer
    trainer = LTVTrainer(
        model=model,
        criterion=criterion,
        model_type=model_type,
        learning_rate=args.learning_rate,
        device=args.device,
        log_dir=log_dir,
        eval_interval=args.eval_interval
    )

    # Train
    print("\nTraining...")
    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        early_stopping_patience=args.early_stopping_patience,
        verbose=True,
        save_dir=run_path  # Save to runs directory
    )

    # Plot training curves
    plot_results(trainer.train_losses, trainer.val_losses, run_path, args.loss_name)

    # Evaluate on test set
    print("\n" + "=" * 60)
    print("FINAL EVALUATION ON TEST SET")
    print("=" * 60)
    metrics, preds, true = evaluate_model(model, test_loader, model_type, args.device)

    print("\nTest Set Metrics (from paper):")
    print("-" * 60)
    print(f"  Normalized Gini: {metrics['normalized_gini']:.4f} (PRIMARY METRIC)")
    print(f"  Spearman Correlation: {metrics['spearman']:.4f}")
    print(f"  Decile MAPE: {metrics['decile_mape']:.1f}%")
    print(f"  MAE: {metrics['mae']:.2f}")
    print(f"  RMSE: {metrics['rmse']:.2f}")
    print(f"  AUC-PR: {metrics['auc_pr']:.4f}")
    print("-" * 60)

    # Save model to runs directory
    model_path = run_path / 'model_best.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_type': model_type,
        'scalers': scalers,
        'input_dim': X_train.shape[1],
        'hidden_dims': args.hidden_dims,
        'loss_name': args.loss_name,
        'metrics': metrics,
        'args': vars(args)  # Save all training arguments
    }, model_path)

    print(f"\nModel saved to: {model_path}")

    # Save predictions to runs directory
    results_df = pd.DataFrame({
        'true_ltv': true,
        'predicted_ltv': preds
    })
    predictions_path = run_path / 'test_predictions.csv'
    results_df.to_csv(predictions_path, index=False)
    print(f"Predictions saved to: {predictions_path}")

    # Save training configuration
    config_path = run_path / 'config.json'

    # Convert metrics to native Python types for JSON serialization
    metrics_serializable = {k: float(v) for k, v in metrics.items()}

    config = {
        'model_type': model_type,
        'loss_name': args.loss_name,
        'hidden_dims': args.hidden_dims,
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'dropout_rate': args.dropout_rate,
        'input_dim': int(X_train.shape[1]),
        'train_samples': int(len(train_df)),
        'test_samples': int(len(test_df)),
        'timestamp': timestamp if args.log_dir is None else 'custom',
        'final_metrics': metrics_serializable
    }
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Configuration saved to: {config_path}")

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"\nExperiment saved to: {run_path}")
    print(f"\nContents:")
    print(f"  - TensorBoard logs (view with: tensorboard --logdir {run_path.parent})")
    print(f"  - Model checkpoint: model_best.pt")
    print(f"  - Predictions: test_predictions.csv")
    print(f"  - Evaluation history: eval_history.csv")
    print(f"  - Training curves: training_curves_{args.loss_name}.png")
    print(f"  - Configuration: config.json")
    print("=" * 60)


if __name__ == "__main__":
    main()
