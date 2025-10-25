# Directory Structure Update

## Problem

Previously, training outputs were split across multiple directories:
- TensorBoard logs → `runs/<loss_name>_<timestamp>/`
- Model checkpoints → `models/`
- Predictions → `models/`
- Evaluation history → `models/`

**Issues:**
- Multiple runs would overwrite files in `models/`
- No way to distinguish between different experiments
- TensorBoard logs were separate from model/results
- Difficult to track which model corresponds to which logs

## Solution

All experiment artifacts are now saved together in a single timestamped directory within `runs/`:

```
runs/<loss_name>_<timestamp>/
├── events.out.tfevents.*      # TensorBoard logs
├── model_best.pt              # Model checkpoint
├── eval_history.csv           # Metrics over time
├── test_predictions.csv       # Test predictions
├── training_curves_*.png      # Loss plots
└── config.json                # Training configuration
```

## Directory Structure

### Complete Structure

```
ziln-loss/
├── runs/                          # All experiments (NEVER OVERWRITTEN)
│   ├── ziln_20250126_123456/      # ZILN experiment
│   │   ├── events.out.tfevents.*
│   │   ├── model_best.pt
│   │   ├── eval_history.csv
│   │   ├── test_predictions.csv
│   │   ├── training_curves_ziln.png
│   │   └── config.json
│   │
│   ├── mse_20250126_123500/       # MSE experiment
│   │   ├── events.out.tfevents.*
│   │   ├── model_best.pt
│   │   ├── eval_history.csv
│   │   ├── test_predictions.csv
│   │   ├── training_curves_mse.png
│   │   └── config.json
│   │
│   ├── ziln_20250127_090000/      # Another ZILN experiment (different hyperparams)
│   │   └── ...
│   │
│   └── comparison_20250126_123500.png  # Comparison plots
│
├── data/                          # Processed datasets
│   └── processed/
│       ├── train.parquet
│       └── test.parquet
│
└── [source code files]

Note: The `models/` directory is no longer created or used. All outputs go to `runs/`.
```

### File Descriptions

#### `model_best.pt`
Complete model checkpoint containing:
- `model_state_dict`: Model weights
- `model_type`: 'ziln' or 'simple'
- `scalers`: Feature preprocessing scalers
- `input_dim`: Number of input features
- `hidden_dims`: Model architecture
- `loss_name`: Loss function used
- `metrics`: Final test metrics
- `args`: All training arguments

#### `eval_history.csv`
Evaluation metrics computed at intervals during training:
```csv
epoch,train_loss,val_loss,val_normalized_gini,val_spearman,val_decile_mape,val_mae,val_rmse,val_auc_pr
1,0.8234,0.7891,0.3521,0.4123,28.3,12.45,18.76,0.7821
5,0.6543,0.6234,0.3678,0.4456,25.1,11.23,16.89,0.8012
...
```

#### `test_predictions.csv`
Final predictions on test set:
```csv
true_ltv,predicted_ltv
0.0,2.34
15.67,14.23
...
```

#### `config.json`
Complete training configuration:
```json
{
  "model_type": "ziln",
  "loss_name": "ziln",
  "hidden_dims": [64, 32],
  "learning_rate": 0.001,
  "batch_size": 256,
  "epochs": 50,
  "dropout_rate": 0.2,
  "input_dim": 6,
  "train_samples": 16053,
  "test_samples": 3300,
  "timestamp": "20250126_123456",
  "final_metrics": {
    "normalized_gini": 0.368,
    "spearman": 0.484,
    ...
  }
}
```

## Usage

### Training

```bash
# Automatic timestamped directory
python train_ziln_model.py --loss_name ziln --epochs 50
# Creates: runs/ziln_20250126_123456/

# Custom directory name
python train_ziln_model.py --loss_name ziln --log_dir runs/my_experiment
# Creates: runs/my_experiment/
```

### Comparing Runs

```bash
# TensorBoard (shows all runs)
tensorboard --logdir runs

# Compare specific runs
python compare_tensorboards.py --plot \
    --eval1 runs/ziln_20250126_123456/eval_history.csv \
    --eval2 runs/mse_20250126_123500/eval_history.csv \
    --label1 ZILN --label2 MSE
```

### Loading Models

```python
import torch

# Load checkpoint
checkpoint = torch.load('runs/ziln_20250126_123456/model_best.pt')

# Inspect configuration
print("Training config:", checkpoint['args'])
print("Final metrics:", checkpoint['metrics'])

# Recreate model
from model import ZILNModel
model = ZILNModel(
    input_dim=checkpoint['input_dim'],
    hidden_dims=checkpoint['hidden_dims']
)
model.load_state_dict(checkpoint['model_state_dict'])
```

## Benefits

### 1. No Overwriting
Each experiment gets a unique timestamped directory. Previous experiments are never overwritten.

### 2. Self-Contained
Everything for one experiment is in one place:
- Want to see training curves? Check the run directory.
- Want to load the model? It's in the same directory.
- Want to see what hyperparameters were used? Check config.json.

### 3. Easy Comparison
TensorBoard automatically shows all experiments in the runs directory:
```bash
tensorboard --logdir runs
# Shows all experiments with color-coded plots
```

### 4. Reproducibility
Each run's `config.json` contains complete training configuration. You can exactly reproduce any experiment.

### 5. Organized by Time
Runs are automatically sorted chronologically by their timestamp in the directory name.

## Migration from Old Structure (If Applicable)

If you previously had experiments saved in a `models/` directory, you can organize them:

```bash
# Create a run directory for old experiment
mkdir -p runs/legacy_experiment

# Move old files (if they exist)
[ -f models/model_ziln.pt ] && mv models/model_ziln.pt runs/legacy_experiment/model_best.pt
[ -f models/test_predictions_ziln.csv ] && mv models/test_predictions_ziln.csv runs/legacy_experiment/test_predictions.csv
# etc.

# Remove empty models directory
[ -d models ] && rmdir models 2>/dev/null || true
```

Note: The `models/` directory is **no longer created** by the training script. Everything goes to `runs/`.

## Automated Comparison Script

The `run_comparison_experiment.sh` script now automatically:
1. Trains ZILN model → `runs/ziln_<timestamp>/`
2. Trains MSE model → `runs/mse_<timestamp>/`
3. Compares them → `runs/comparison_<timestamp>.png`

All with a single command:
```bash
./run_comparison_experiment.sh
```

Output:
```
Experiment Complete!
========================================

Results organized in runs directory:

ZILN Model:
  Directory: runs/ziln_20250126_123456/
    - TensorBoard logs
    - Model: model_best.pt
    - Predictions: test_predictions.csv
    - Evaluation: eval_history.csv
    - Config: config.json

MSE Model:
  Directory: runs/mse_20250126_123456/
    - TensorBoard logs
    - Model: model_best.pt
    - Predictions: test_predictions.csv
    - Evaluation: eval_history.csv
    - Config: config.json

Comparison:
  Plot: runs/comparison_20250126_123456.png

To view TensorBoard (both models):
  tensorboard --logdir runs
```

## Best Practices

### 1. Use Descriptive Log Dirs for Experiments
```bash
# Instead of relying on timestamp
python train_ziln_model.py --loss_name ziln --log_dir runs/ziln_lr0001_dropout02

# Or with notes
python train_ziln_model.py --loss_name ziln --log_dir runs/ziln_higherlr_experiment
```

### 2. Keep TensorBoard Running
```bash
# Start TensorBoard in background
tensorboard --logdir runs &

# Continue training, experiments automatically appear
python train_ziln_model.py --loss_name ziln --epochs 50
```

### 3. Clean Up Old Experiments
```bash
# List experiments
ls -lth runs/

# Remove old unsuccessful experiments
rm -rf runs/ziln_20250120_failed/
```

### 4. Archive Completed Experiments
```bash
# Create archive directory
mkdir -p experiments/completed

# Move finished experiments
mv runs/ziln_20250126_123456/ experiments/completed/
```

## Summary

✅ **New Structure:**
- Everything in `runs/<loss_name>_<timestamp>/`
- Self-contained experiments
- No overwriting
- Easy comparison with TensorBoard

✅ **Benefits:**
- Reproducibility (config.json)
- Organization (timestamped directories)
- Comparison (TensorBoard integration)
- Tracking (all artifacts together)

✅ **Usage:**
- Training: Same commands, better organization
- Comparison: Point to specific run directories
- TensorBoard: Just use `--logdir runs`
