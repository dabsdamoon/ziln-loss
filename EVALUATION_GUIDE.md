# Evaluation Metrics & TensorBoard Comparison Guide

This guide explains how to use the comprehensive evaluation metrics and TensorBoard logging to compare ZILN and MSE loss functions, replicating the experiments from Wang et al. (2019).

## Overview of Implemented Features

### 1. Comprehensive Evaluation Metrics (`evaluation/metrics.py`)

All metrics from the paper are now implemented:

- **Normalized Gini Coefficient** (PRIMARY METRIC) - Section 4.1, pages 6-7
  - Measures discrimination capability
  - Range: 0 to 1 (higher is better)
  - Paper result: ZILN (0.368) vs MSE (0.330) = 11.4% improvement

- **Spearman's Rank Correlation** - Table 1, page 9
  - Measures ranking quality
  - Range: -1 to 1 (higher is better)
  - Paper result: ZILN (0.484) vs MSE (0.327) = 48.0% improvement

- **Decile-level MAPE** - Table 3, page 11
  - Measures calibration quality
  - Range: 0 to ∞ (lower is better)
  - Paper result: ZILN (22.6%) vs MSE (72.8%) = 68.9% reduction

- **AUC-PR** - Table 4
  - Area under precision-recall curve for binary classification
  - Range: 0 to 1 (higher is better)

- **Standard Metrics**
  - MAE (Mean Absolute Error)
  - RMSE (Root Mean Squared Error)

### 2. TensorBoard Logging

The trainer now logs:
- Training and validation losses (every epoch)
- All comprehensive metrics (at specified intervals)
- Evaluation history saved to CSV

### 3. Comparison Tools (`compare_tensorboards.py`)

Utilities for comparing models:
- Launch TensorBoard for interactive comparison
- Generate comparison plots from evaluation histories
- Print formatted metric comparison tables
- Compare against paper results

## Quick Start

### Step 1: Train ZILN Model

```bash
python train_ziln_model.py \
    --loss_name ziln \
    --epochs 50 \
    --eval_interval 5 \
    --log_dir runs/ziln_experiment
```

This will:
- Train with ZILN loss
- Evaluate comprehensive metrics every 5 epochs
- Save TensorBoard logs to `runs/ziln_experiment`
- Save evaluation history to `models/eval_history.csv`

### Step 2: Train MSE Model for Comparison

```bash
python train_ziln_model.py \
    --loss_name mse \
    --epochs 50 \
    --eval_interval 5 \
    --log_dir runs/mse_experiment
```

### Step 3: Compare Results

#### Option A: Interactive TensorBoard

```bash
# Launch TensorBoard to view both runs
python compare_tensorboards.py --launch --logdir runs

# Or simply:
tensorboard --logdir runs --port 6006
```

Then open your browser to `http://localhost:6006` to see:
- Loss curves side-by-side
- All metrics over time
- Smoothing and filtering options

#### Option B: Generate Comparison Plots

```bash
python compare_tensorboards.py \
    --plot \
    --eval1 models/eval_history.csv \
    --eval2 models/eval_history.csv \
    --label1 ZILN \
    --label2 MSE \
    --output comparison_ziln_vs_mse.png
```

This generates a 6-panel comparison plot showing all metrics over training.

#### Option C: Print Metric Comparison Table

The comparison script automatically prints a formatted table comparing:
- Your results (ZILN vs MSE)
- Paper results (for reference)
- Relative improvements

## Training Options

### Basic Usage

```bash
python train_ziln_model.py --loss_name ziln
```

### Advanced Options

```bash
python train_ziln_model.py \
    --loss_name ziln \
    --hidden_dims 128 64 32 \
    --batch_size 512 \
    --epochs 100 \
    --learning_rate 0.0005 \
    --dropout_rate 0.3 \
    --eval_interval 10 \
    --log_dir runs/ziln_$(date +%Y%m%d_%H%M%S) \
    --device cuda
```

### Available Loss Functions

- `ziln`: Zero-Inflated Lognormal loss (from paper)
- `mse`: MSE loss for ZILN model (for comparison)
- `simple_mse`: Simple MSE with MLP baseline
- `quantile`: Quantile loss (robust to outliers)
- `huber`: Huber loss (robust to outliers)
- `logcosh`: Log-Cosh loss (smooth robust loss)

## Understanding the Output

### During Training

Every `eval_interval` epochs, you'll see:

```
--- Epoch 5/50 - Comprehensive Evaluation ---
  Train Loss: 0.8234, Val Loss: 0.7891
  Normalized Gini: 0.3521 (PRIMARY)
  Spearman Corr: 0.4123
  Decile MAPE: 28.3%
  MAE: 12.45, RMSE: 18.76
  AUC-PR: 0.7821
```

### After Training

```
============================================================
FINAL EVALUATION ON TEST SET
============================================================

Test Set Metrics (from paper):
------------------------------------------------------------
  Normalized Gini: 0.3680 (PRIMARY METRIC)
  Spearman Correlation: 0.4840
  Decile MAPE: 22.6%
  MAE: 10.23
  RMSE: 15.67
  AUC-PR: 0.8234
------------------------------------------------------------
```

## File Organization

After training, each experiment is self-contained in its own runs directory:

```
ziln-loss/
├── runs/
│   ├── ziln_20250125_123456/          # Complete ZILN experiment
│   │   ├── events.out.tfevents.*      # TensorBoard logs
│   │   ├── model_best.pt              # Best model checkpoint
│   │   ├── eval_history.csv           # Metrics over time
│   │   ├── test_predictions.csv       # Test predictions
│   │   ├── training_curves_ziln.png   # Loss plots
│   │   └── config.json                # Training configuration
│   │
│   ├── mse_20250125_123457/           # Complete MSE experiment
│   │   ├── events.out.tfevents.*      # TensorBoard logs
│   │   ├── model_best.pt              # Best model checkpoint
│   │   ├── eval_history.csv           # Metrics over time
│   │   ├── test_predictions.csv       # Test predictions
│   │   ├── training_curves_mse.png    # Loss plots
│   │   └── config.json                # Training configuration
│   │
│   └── comparison_20250125_123500.png # Comparison plot
│
└── evaluation/
    ├── __init__.py
    └── metrics.py                     # All evaluation metric implementations
```

**Benefits of this structure:**
- Each experiment is self-contained
- Easy to compare different runs
- No overwriting of previous experiments
- TensorBoard can show all runs together

## Interpreting Results

### Normalized Gini Coefficient (PRIMARY)

- **What it measures**: How well the model discriminates between high and low LTV customers
- **Range**: 0 (random) to 1 (perfect)
- **Paper target**: ZILN should achieve ~0.368, MSE ~0.330
- **Interpretation**: A 10% improvement means significantly better customer ranking

### Decile MAPE (Calibration)

- **What it measures**: Average prediction error within each decile of predicted LTV
- **Range**: 0% (perfect) to ∞
- **Paper target**: ZILN should achieve ~22.6%, MSE ~72.8%
- **Interpretation**: Lower values mean better calibrated predictions

### Spearman Correlation (Ranking)

- **What it measures**: Monotonic relationship between predictions and true values
- **Range**: -1 to 1 (higher is better)
- **Paper target**: ZILN should achieve ~0.484, MSE ~0.327
- **Interpretation**: Higher correlation means better rank ordering

## Troubleshooting

### TensorBoard not found

```bash
pip install tensorboard
```

### Port already in use

```bash
python compare_tensorboards.py --launch --logdir runs --port 6007
```

### Metrics look incorrect

Check that:
1. Data preprocessing is consistent between train/val/test
2. Feature scaling is properly applied
3. Model has converged (check loss curves)
4. Eval interval is appropriate (not too sparse)

## References

Wang et al. (2019). "A Deep Probabilistic Model for Customer Lifetime Value Prediction"
- Section 4: Evaluation Metrics
- Table 1: Normalized Gini and Spearman comparison
- Table 3: Decile-level MAPE comparison
- Paper: https://arxiv.org/abs/1912.07753

## Next Steps

1. **Experiment with hyperparameters**: Try different architectures, learning rates, dropout rates
2. **Analyze predictions**: Use the saved predictions CSVs to understand model behavior
3. **Feature engineering**: Add more features from the initial purchase data
4. **Cross-validation**: Implement k-fold CV for more robust comparisons
5. **Production deployment**: Export best model for serving

## Example Workflow

```bash
# 1. Train both models (each gets its own timestamped directory)
python train_ziln_model.py --loss_name ziln --epochs 50
python train_ziln_model.py --loss_name mse --epochs 50

# 2. View training progress in real-time (shows all runs)
tensorboard --logdir runs

# 3. After training, compare results
# Note the timestamped directory names from training output
python compare_tensorboards.py --plot \
    --eval1 runs/ziln_20250126_123456/eval_history.csv \
    --eval2 runs/mse_20250126_123500/eval_history.csv \
    --label1 ZILN --label2 MSE

# 4. Analyze predictions from specific runs
python -c "
import pandas as pd
ziln = pd.read_csv('runs/ziln_20250126_123456/test_predictions.csv')
mse = pd.read_csv('runs/mse_20250126_123500/test_predictions.csv')
print('ZILN predictions:', ziln.describe())
print('MSE predictions:', mse.describe())
"

# 5. Load a specific model for inference
python -c "
import torch
checkpoint = torch.load('runs/ziln_20250126_123456/model_best.pt')
print('Model info:', checkpoint['args'])
print('Final metrics:', checkpoint['metrics'])
"
```

This workflow replicates the paper's comparison experiments and allows you to verify whether ZILN loss provides significant improvements over MSE for your specific dataset.
