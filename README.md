# ZILN Loss for Customer Lifetime Value Prediction

Implementation of the Zero-Inflated Lognormal (ZILN) loss function for customer lifetime value (LTV) prediction, based on the paper:

**"A Deep Probabilistic Model for Customer Lifetime Value Prediction"**
Wang et al. (2019) - [arXiv:1912.07753](https://arxiv.org/abs/1912.07753)

## Overview

This project implements a deep neural network with ZILN loss to predict customer lifetime value. The model handles the zero-inflated, heavy-tailed distribution of customer purchase behavior by combining:
- **Binary classification**: Will the customer return? (probability p)
- **Lognormal regression**: How much will they spend? (parameters μ, σ)

**Key Features:**
- Complete implementation of ZILN loss and model architecture from the paper
- Comprehensive evaluation metrics (Normalized Gini, Spearman, Decile MAPE, etc.)
- TensorBoard integration for training visualization
- Memory-optimized preprocessing for large datasets
- Automated experiment tracking and comparison

## Architecture

### Model Components

1. **ZILNModel** (`model/model.py`)
   - Deep neural network with shared hidden layers
   - Three output heads: p (return probability), μ (mean), σ (std)
   - Numerical stability bounds to prevent overflow

2. **SimpleMLPModel** (`model/model.py`)
   - Simple MLP for direct LTV prediction
   - Used as baseline for comparison

3. **XGBoostLTVModel** (`model/xgboost_model.py`)
   - Gradient boosting baseline for comparison
   - Feature importance analysis
   - Standard tree-based regression (predicts mean only)

4. **XGBoostQuantileModel** (`model/xgboost_model.py`)
   - XGBoost with quantile regression
   - Predicts full distribution via multiple quantiles
   - Captures uncertainty with prediction intervals
   - Alternative to ZILN for uncertainty estimation

5. **Loss Functions** (`loss/loss.py`)
   - `ZILNLoss`: Zero-Inflated Lognormal loss (main)
   - `MSELossForZILN`: MSE baseline for comparison
   - Additional robust losses: Huber, Quantile, LogCosh

6. **Evaluation Metrics** (`evaluation/metrics.py`)
   - Normalized Gini Coefficient (PRIMARY)
   - Spearman's Rank Correlation
   - Decile-level MAPE
   - Standard metrics: MAE, RMSE, AUC-PR

### Project Structure

```
ziln-loss/
├── data/
│   └── processed/          # Preprocessed train/test data
├── runs/                   # Experiment outputs (auto-generated)
│   └── <loss>_<timestamp>/ # Self-contained experiment
│       ├── model_best.pt
│       ├── eval_history.csv
│       ├── test_predictions.csv
│       └── config.json
├── model/                  # Model architecture
├── loss/                   # Loss functions
├── evaluation/             # Evaluation metrics
├── preprocessor/           # Data preprocessing
└── utils/                  # Utilities (CSV→Parquet conversion)
```

## Quick Start

### 1. Environment Setup

```bash
# Create conda environment
conda env create -f environment.yml
conda activate ziln-loss

# Or install dependencies manually
pip install torch pandas numpy scipy scikit-learn matplotlib seaborn tensorboard psutil tqdm pyarrow
```

### 2. Data Pipeline

#### Step 1: Convert CSV to Parquet (Optional, for large datasets)

```bash
# Convert large CSV files to efficient Parquet format
python utils/convert_to_parquet.py \
    --csv_path /path/to/transactions.csv.gz \
    --output_dir /path/to/parquet/
```

**Purpose**: Converts 20GB+ CSV files to Parquet for faster loading
**Output**: Multiple `part_XXXX.parquet` files

#### Step 2: Preprocess Data

```bash
# Process transactions into customer-level features
python preprocessor/preprocess.py \
    --parquet_dir /mnt/d/datasets/acquire-valued-shoppers-challenge/parquet \
    --output_dir ./data/processed \
    --n_files 20 \
    --use_chunked_loading  # For large datasets (>20 files)
```

**What it does:**
- Loads transaction data from parquet files
- Creates features from initial purchase (amount, items, store, category, brand)
- Calculates target: 12-month future purchase value
- Splits into train/test (80/20) by company
- Saves to `data/processed/train.parquet` and `test.parquet`

**Feature Engineering:**
- **Numerical features** (initial_purchase_amount, initial_num_items): Min-Max normalization [0, 1]
- **Categorical features** (chain, initial_category, initial_brand, initial_productmeasure): One-hot encoding
- Handles unseen categories in test set gracefully

**Memory optimization options:**
```bash
# For limited RAM
python preprocessor/preprocess.py \
    --n_files 50 \
    --chunk_size 3 \          # Load 3 files at a time
    --memory_threshold 75.0   # Stop if RAM exceeds 75%
```

**Output:**
- `data/processed/train.parquet` (~16k customers)
- `data/processed/test.parquet` (~3k customers)
- `data/processed/data_summary.txt`

#### Step 3: Exploratory Data Analysis (Optional)

```bash
# Analyze target distribution
python eda_target_distribution.py
```

**Generates:**
- Distribution plots showing heavy-tail and right-skewness
- Statistics confirming zero-inflation
- Saved to `eda_plots/target_distribution.png`

### 3. Training

The training script automatically applies proper feature engineering:
- **Numerical features**: Min-Max normalization to [0, 1]
- **Categorical features**: One-hot encoding with unknown category handling

#### Basic Training

```bash
# Train ZILN model
python train_ziln_model.py \
    --loss_name ziln \
    --epochs 50 \
    --eval_interval 5
```

**Output directory:** `runs/ziln_<timestamp>/`
- TensorBoard logs
- `model_best.pt` - Best model checkpoint
- `eval_history.csv` - Metrics over time
- `test_predictions.csv` - Final predictions
- `config.json` - Training configuration

**Training output example:**
```
Preparing features...
  - Numerical features: Min-Max normalization [0, 1]
  - Categorical features: One-hot encoding

Feature Engineering Summary:
  Total features: 125
  - Numerical (min-max scaled): 2
  - Categorical (original): 3
  - Categorical (one-hot encoded): 123
  Formula: 2 + 123 = 125
```

#### Training Options

```bash
# Full options
python train_ziln_model.py \
    --loss_name ziln \              # Loss: ziln, mse, huber, quantile
    --hidden_dims 128 64 32 \       # Network architecture
    --learning_rate 0.001 \         # Learning rate
    --batch_size 256 \              # Batch size
    --epochs 100 \                  # Number of epochs
    --dropout_rate 0.2 \            # Dropout rate
    --eval_interval 5 \             # Eval metrics every N epochs
    --log_dir runs/my_experiment    # Custom output directory
```

#### Train XGBoost for Comparison

You can now train an XGBoost model as a baseline for comparison:

```bash
# Train XGBoost model
python train_ziln_model.py \
    --model_type xgboost \
    --learning_rate 0.1 \
    --xgb_n_estimators 100 \
    --xgb_max_depth 6
```

**XGBoost-specific options:**
```bash
python train_ziln_model.py \
    --model_type xgboost \
    --learning_rate 0.1 \           # Boosting learning rate
    --xgb_n_estimators 100 \        # Number of trees
    --xgb_max_depth 6 \             # Max tree depth
    --xgb_subsample 0.8 \           # Row sampling ratio
    --xgb_colsample_bytree 0.8 \   # Column sampling ratio
    --xgb_min_child_weight 1 \      # Minimum child weight
    --xgb_gamma 0.0 \               # Min split loss
    --xgb_reg_alpha 0.0 \           # L1 regularization
    --xgb_reg_lambda 1.0            # L2 regularization
```

**Output directory:** `runs/xgboost_<timestamp>/`
- `model_best.pkl` - Trained XGBoost model
- `test_predictions.csv` - Final predictions
- `eval_history.csv` - Metrics tracked over boosting rounds
- `feature_importance.csv` & `.png` - Feature importance analysis
- `training_curves_xgboost.png` - Training curves
- `config.json` - Training configuration

**Note:** XGBoost uses `--eval_interval` (default: 10) to evaluate comprehensive metrics every N boosting rounds, creating an `eval_history.csv` file with the same format as PyTorch models. This allows for fair comparison of training dynamics across all model types.

#### XGBoost Quantile Regression (Capture Uncertainty)

**Problem:** Standard XGBoost only predicts the mean, not the variation/uncertainty.

**Solution:** XGBoost Quantile Regression predicts the full distribution by training multiple models for different quantiles (10th, 50th, 90th percentiles, etc.).

```python
from model import XGBoostQuantileModel
from train_ziln_model import prepare_features
import pandas as pd

# Load and prepare data
train_df = pd.read_parquet('data/processed/train.parquet')
test_df = pd.read_parquet('data/processed/test.parquet')
X_train, y_train, scalers = prepare_features(train_df, is_train=True)
X_test, y_test, _ = prepare_features(test_df, scalers=scalers, is_train=False)

# Train quantile regression (predicts 10th, 50th, 90th percentiles)
model = XGBoostQuantileModel(
    quantiles=[0.1, 0.5, 0.9],  # Lower bound, median, upper bound
    n_estimators=100
)
model.fit(X_train, y_train)

# Get predictions with uncertainty
predictions = model.predict(X_test)
# Returns: {
#   'q_0.5': median predictions,
#   'q_0.1': lower bound (10th percentile),
#   'q_0.9': upper bound (90th percentile),
#   'mean': average across quantiles,
#   'width': prediction interval width
# }

# Or get 80% prediction interval
median, lower, upper = model.get_prediction_intervals(X_test, confidence=0.8)
```

**Full example:**
```bash
python example_quantile_xgboost.py
```

This creates visualizations showing:
- Prediction intervals (uncertainty bands)
- All quantile predictions
- Calibration analysis

**Comparison with ZILN:**
- ZILN: Predicts distribution via parameters (p, μ, σ) - single model
- XGBoost Quantile: Predicts distribution via multiple quantiles - multiple models
- Both capture uncertainty, but different approaches

#### Compare ZILN vs MSE vs XGBoost

```bash
# Automated comparison (trains both neural models)
./run_comparison_experiment.sh
```

Or manually:
```bash
# Train ZILN
python train_ziln_model.py --loss_name ziln --epochs 50

# Train MSE baseline
python train_ziln_model.py --loss_name mse --epochs 50

# Train XGBoost baseline
python train_ziln_model.py --model_type xgboost --xgb_n_estimators 100

# Compare results
python compare_tensorboards.py --plot \
    --eval1 runs/ziln_<timestamp>/eval_history.csv \
    --eval2 runs/mse_<timestamp>/eval_history.csv \
    --label1 ZILN --label2 MSE
```

### 4. Monitoring Training

#### TensorBoard

```bash
# View all experiments
tensorboard --logdir runs

# Open browser to http://localhost:6006
```

**What you'll see:**
- Loss curves (train/val)
- Metrics over time (Normalized Gini, Spearman, etc.)
- Side-by-side comparison of different runs

#### Training Output

```
--- Epoch 5/50 - Comprehensive Evaluation ---
  Train Loss: 0.8234, Val Loss: 0.7891
  Normalized Gini: 0.3521 (PRIMARY)
  Spearman Corr: 0.4123
  Decile MAPE: 28.3%
  MAE: 12.45, RMSE: 18.76
  AUC-PR: 0.7821
```

### 5. Loading Trained Models

```python
import torch
from model import ZILNModel

# Load checkpoint
checkpoint = torch.load('runs/ziln_20250126_123456/model_best.pt')

# Recreate model
model = ZILNModel(
    input_dim=checkpoint['input_dim'],
    hidden_dims=checkpoint['hidden_dims']
)
model.load_state_dict(checkpoint['model_state_dict'])

# Get scalers for preprocessing
scalers = checkpoint['scalers']

# Make predictions
model.eval()
predictions = model.predict_ltv(new_features)
```

## Evaluation Metrics

Based on the paper's evaluation methodology:

| Metric | Description | Better |
|--------|-------------|--------|
| **Normalized Gini** | Discrimination capability (PRIMARY) | Higher |
| **Spearman Correlation** | Ranking quality | Higher |
| **Decile MAPE** | Calibration quality | Lower |
| **MAE** | Mean absolute error | Lower |
| **RMSE** | Root mean squared error | Lower |
| **AUC-PR** | Binary classification (returning vs not) | Higher |

**Paper Results:**
- ZILN Normalized Gini: 0.368 vs MSE: 0.330 (+11.4%)
- ZILN Spearman: 0.484 vs MSE: 0.327 (+48.0%)
- ZILN Decile MAPE: 22.6% vs MSE: 72.8% (-68.9%)

**Evaluation History:**

All models (ZILN, MLP, XGBoost) generate an `eval_history.csv` file tracking metrics over time:
- **PyTorch models**: Metrics computed every `--eval_interval` epochs (default: 5)
- **XGBoost**: Metrics computed every `--eval_interval` boosting rounds (default: 10)

Example `eval_history.csv`:
```csv
iteration,train_loss,val_loss,val_normalized_gini,val_spearman,val_decile_mape,val_mae,val_rmse,val_auc_pr
10,0.8234,0.7891,0.3521,0.4123,28.3,12.45,18.76,0.7821
20,0.7456,0.7234,0.3789,0.4456,25.1,11.23,17.89,0.8012
...
```

This allows you to:
- Track learning progress over time
- Compare training dynamics across different models
- Use the same comparison tools for all model types

## Memory Management

For large datasets, the preprocessing script includes memory optimization:

```bash
# Automatic chunked loading for >20 files
python preprocessor/preprocess.py --n_files 100

# Manual configuration
python preprocessor/preprocess.py \
    --n_files 100 \
    --chunk_size 5 \              # Load 5 files at a time
    --memory_threshold 85.0 \     # Stop if RAM > 85%
    --use_chunked_loading
```

**Features:**
- Real-time memory monitoring
- Chunked file loading
- Data type optimization (50% memory reduction)
- Automatic garbage collection
- Prevents out-of-memory crashes

See `MEMORY_OPTIMIZATION_GUIDE.md` for details.

## Numerical Stability

The model includes safeguards against overflow/underflow:

- **Output bounds**: μ ∈ [-15, 15], σ ∈ [0.01, 5]
- **Gradient clipping**: max_norm=1.0
- **NaN/Inf detection**: Real-time monitoring and error reporting
- **Loss clamping**: Prevents extreme loss values

If you encounter infinity errors, reduce learning rate:
```bash
python train_ziln_model.py --learning_rate 0.0001  # 10x smaller
```

See `NUMERICAL_STABILITY_FIXES.md` for details.

## Troubleshooting

### Issue: Out of Memory
```bash
# Use chunked loading
python preprocessor/preprocess.py --use_chunked_loading --chunk_size 3
```

### Issue: Infinity in predictions
```bash
# Reduce learning rate
python train_ziln_model.py --learning_rate 0.0001
```

### Issue: Poor convergence
```bash
# Increase regularization
python train_ziln_model.py --dropout_rate 0.3 --learning_rate 0.0005
```

### Issue: Can't find experiment files
All files are in `runs/<loss>_<timestamp>/`. Check the output at the end of training.

## Documentation

- `EVALUATION_GUIDE.md` - Complete guide to metrics and evaluation
- `MEMORY_OPTIMIZATION_GUIDE.md` - Memory management for large datasets
- `NUMERICAL_STABILITY_FIXES.md` - Handling infinity/NaN errors
- `DIRECTORY_STRUCTURE.md` - Experiment organization
- `DATA.md` - Data schema and preprocessing details

## Citation

If you use this implementation, please cite the original paper:

```bibtex
@article{wang2019deep,
  title={A Deep Probabilistic Model for Customer Lifetime Value Prediction},
  author={Wang, Xiaojing and Liu, Tianqi and Miao, Jingang},
  journal={arXiv preprint arXiv:1912.07753},
  year={2019}
}
```

## License

This implementation is for educational and research purposes.

## Example End-to-End Workflow

```bash
# 1. Setup
conda activate ziln-loss

# 2. Preprocess data (first time only)
python preprocessor/preprocess.py \
    --parquet_dir /path/to/parquet \
    --n_files 20

# 3. Train ZILN model
python train_ziln_model.py --loss_name ziln --epochs 50

# 4. Train MSE baseline
python train_ziln_model.py --loss_name mse --epochs 50

# 5. Train XGBoost baseline
python train_ziln_model.py --model_type xgboost --xgb_n_estimators 100

# 6. View results
tensorboard --logdir runs

# 7. Compare models
python compare_tensorboards.py --plot \
    --eval1 runs/ziln_<timestamp>/eval_history.csv \
    --eval2 runs/mse_<timestamp>/eval_history.csv
```

**Expected training time:**
- Preprocessing: 5-15 minutes (20 files)
- Training (Neural): 10-30 minutes (50 epochs)
- Training (XGBoost): 2-5 minutes (100 trees)
- Total: ~30-50 minutes for complete comparison

## Key Results to Expect

If the implementation is working correctly, you should see:
- ZILN outperforms MSE on Normalized Gini (+10-15%)
- ZILN outperforms MSE on Spearman correlation (+40-50%)
- ZILN outperforms MSE on Decile MAPE (-60-70%)
- Training converges smoothly without NaN/Inf errors
- Validation metrics improve over time

## Contact & Issues

For questions or issues, please refer to the documentation files or check the training output for diagnostic information.
