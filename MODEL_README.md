# ZILN Model for Customer Lifetime Value Prediction

This implementation is based on the paper: **"A Deep Probabilistic Model for Customer Lifetime Value Prediction"** by Wang et al. (2019) from Google Research.

Paper: https://arxiv.org/abs/1912.07753

## Overview

The Zero-Inflated Lognormal (ZILN) model is specifically designed to handle the challenges of LTV prediction:

1. **Zero-inflation**: Many customers never return (LTV = 0)
2. **Heavy-tailed distribution**: Returning customers have highly skewed LTV values
3. **Uncertainty quantification**: Provides full probabilistic predictions

## Model Architecture

```
Input Features → Hidden Layers → Three Output Heads:
                                   ├─ p (probability of returning) - Sigmoid
                                   ├─ μ (lognormal mean) - Identity
                                   └─ σ (lognormal std) - Softplus
```

### Loss Function

The ZILN loss combines two components:

1. **Binary Cross Entropy**: For zero vs non-zero classification
2. **Lognormal Loss**: For non-zero LTV values

```
L_ZILN(x; p, μ, σ) = -1{x=0} log(1-p) - 1{x>0}(log p - L_Lognormal(x; μ, σ))
```

where:

```
L_Lognormal(x; μ, σ) = log(xσ√2π) + (log x - μ)²/(2σ²)
```

### Prediction

Expected LTV is computed as:

```
E[LTV] = p × exp(μ + σ²/2)
```

- `p`: Probability of returning
- `exp(μ + σ²/2)`: Expected LTV for returning customers (lognormal mean)

## Usage

### 1. Training the Model

```bash
# Basic usage with defaults
conda run -n ziln-loss python train_ziln_model.py

# Custom hyperparameters
conda run -n ziln-loss python train_ziln_model.py \
    --train_path data/processed/train.parquet \
    --test_path data/processed/test.parquet \
    --output_dir models/ \
    --hidden_dims 64 32 \
    --batch_size 256 \
    --epochs 100 \
    --learning_rate 0.001 \
    --dropout_rate 0.2 \
    --early_stopping_patience 10 \
    --device cpu
```

### 2. Using the Model Programmatically

```python
import torch
from model import ZILNModel, ZILNTrainer
import pandas as pd

# Load preprocessed data
train_df = pd.read_parquet('data/processed/train.parquet')

# Prepare features (see train_ziln_model.py for full preprocessing)
X_train = ...  # Your feature matrix
y_train = ...  # Your target values

# Create model
model = ZILNModel(
    input_dim=X_train.shape[1],
    hidden_dims=[64, 32],
    dropout_rate=0.2
)

# Train
trainer = ZILNTrainer(model, learning_rate=0.001)
trainer.fit(train_loader, val_loader, epochs=100)

# Predict
model.eval()
with torch.no_grad():
    X_test_tensor = torch.FloatTensor(X_test)

    # Get expected LTV
    expected_ltv = model.predict_ltv(X_test_tensor)

    # Get components
    expected_ltv, p_return, ltv_returning = model.predict_ltv(
        X_test_tensor,
        return_components=True
    )

    # Get median (50th percentile)
    median_ltv = model.predict_quantile(X_test_tensor, quantile=0.5)

    # Get 90th percentile
    ltv_90th = model.predict_quantile(X_test_tensor, quantile=0.9)
```

### 3. Loading Saved Model

```python
import torch
from model import ZILNModel

# Load checkpoint
checkpoint = torch.load('models/ziln_model.pt')

# Recreate model
model = ZILNModel(
    input_dim=checkpoint['input_dim'],
    hidden_dims=checkpoint['hidden_dims']
)
model.load_state_dict(checkpoint['model_state_dict'])

# Use for prediction
model.eval()
predictions = model.predict_ltv(X_test_tensor)
```

## Advantages over Traditional Methods

### vs. MSE Loss
- **Handles zero-inflation**: Explicitly models probability of returning
- **Robust to outliers**: Lognormal loss is less sensitive to extreme values
- **Better calibration**: More accurate predictions across all LTV ranges

### vs. Two-Stage Models
- **Single unified model**: No need for separate classification and regression models
- **Joint optimization**: Both tasks share representations and benefit from multi-task learning
- **Simpler deployment**: One model to maintain instead of two

### vs. Simple Mean Prediction
- **Uncertainty quantification**: Provides full distribution, not just point estimate
- **Better discrimination**: Can separate high-value from low-value customers
- **Probabilistic framework**: Natural handling of zero-inflation

## Evaluation Metrics

The paper recommends:

1. **Normalized Gini Coefficient**: Measures model's ability to rank customers by LTV
2. **Decile Charts**: Visual assessment of calibration
3. **Spearman Correlation**: Rank correlation between predicted and actual LTV
4. **AUC-PR**: For binary classification of returning vs non-returning

## Key Features

### Input Features (from preprocessing)
- Initial purchase amount
- Number of items in initial purchase
- Store chain
- Product category
- Product brand
- Product size measure

### Hyperparameters
- `input_dim`: Number of input features
- `hidden_dims`: List of hidden layer sizes (default: [64, 32])
- `dropout_rate`: Dropout for regularization (default: 0.2)
- `use_batch_norm`: Whether to use batch normalization (default: True)
- `learning_rate`: Adam optimizer learning rate (default: 0.001)

### Output
- **Point prediction**: Expected LTV = p × E[LTV | returning]
- **Probability**: p (probability of being a returning customer)
- **Uncertainty**: Full lognormal distribution for returning customers
- **Quantiles**: Any percentile of the LTV distribution

## Results on Test Data

After training, the model will output:

```
Test Set Metrics:
============================================================
MAE: [Mean Absolute Error]
RMSE: [Root Mean Squared Error]
Spearman: [Rank correlation]
AUC: [Area Under ROC Curve for returning prediction]
AUC_PR: [Area Under Precision-Recall Curve]
```

## File Structure

```
model/
├── __init__.py          # Package initialization
└── model.py             # ZILN model implementation
    ├── ZILNLoss         # Loss function
    ├── ZILNModel        # Neural network architecture
    └── ZILNTrainer      # Training utilities

train_ziln_model.py      # Training script
MODEL_README.md          # This file
```

## References

Wang, X., Liu, T., & Miao, J. (2019). A Deep Probabilistic Model for Customer Lifetime Value Prediction. *arXiv preprint arXiv:1912.07753*.

## Tips for Better Performance

1. **Feature Engineering**:
   - Add more customer attributes if available
   - Create interaction features
   - Use embeddings for high-cardinality categorical features

2. **Hyperparameter Tuning**:
   - Try deeper/wider networks: `--hidden_dims 128 64 32`
   - Adjust learning rate: `--learning_rate 0.0002`
   - Tune dropout: `--dropout_rate 0.3`

3. **Data Preprocessing**:
   - Handle outliers in features
   - Normalize/standardize numerical features
   - Encode categorical features appropriately

4. **Training**:
   - Use early stopping to prevent overfitting
   - Monitor validation loss
   - Use gradient clipping (already implemented)

## Common Issues

### Numerical Stability
- The model includes `eps=1e-8` for numerical stability
- Uses softplus for σ to ensure positivity
- Gradient clipping prevents exploding gradients

### Convergence
- If loss doesn't decrease, try lower learning rate
- Check for NaN/Inf in features
- Ensure target values are non-negative

### Overfitting
- Increase dropout rate
- Add more regularization (weight_decay)
- Use more training data
- Reduce model complexity
