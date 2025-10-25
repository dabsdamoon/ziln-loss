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

2. **Loss Functions** (`loss/loss.py`)
   - `ZILNLoss`: Zero-Inflated Lognormal loss (main)
   - `MSELossForZILN`: MSE baseline for comparison
   - Additional robust losses: Huber, Quantile, LogCosh

3. **Evaluation Metrics** (`evaluation/metrics.py`)
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

#### Compare ZILN vs MSE

```bash
# Automated comparison (trains both models)
./run_comparison_experiment.sh
```

Or manually:
```bash
# Train ZILN
python train_ziln_model.py --loss_name ziln --epochs 50

# Train MSE baseline
python train_ziln_model.py --loss_name mse --epochs 50

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

# 5. View results
tensorboard --logdir runs

# 6. Compare models
python compare_tensorboards.py --plot \
    --eval1 runs/ziln_<timestamp>/eval_history.csv \
    --eval2 runs/mse_<timestamp>/eval_history.csv
```

**Expected training time:**
- Preprocessing: 5-15 minutes (20 files)
- Training: 10-30 minutes (50 epochs)
- Total: ~30-45 minutes for complete comparison

## Key Results to Expect

If the implementation is working correctly, you should see:
- ZILN outperforms MSE on Normalized Gini (+10-15%)
- ZILN outperforms MSE on Spearman correlation (+40-50%)
- ZILN outperforms MSE on Decile MAPE (-60-70%)
- Training converges smoothly without NaN/Inf errors
- Validation metrics improve over time

## Contact & Issues

For questions or issues, please refer to the documentation files or check the training output for diagnostic information.
