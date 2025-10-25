#!/bin/bash
# Automated comparison experiment: ZILN vs MSE
# This script trains both models and generates comparison visualizations

echo "=========================================="
echo "ZILN vs MSE Comparison Experiment"
echo "=========================================="
echo ""

# Configuration
EPOCHS=50
EVAL_INTERVAL=5
BATCH_SIZE=256
LEARNING_RATE=0.001

# Create timestamp for this experiment
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "Timestamp: ${TIMESTAMP}"
echo "Epochs: ${EPOCHS}"
echo "Eval interval: ${EVAL_INTERVAL}"
echo ""

# Note: Each training run will create its own directory in runs/
# We'll track the run directories to compare them later

# Train ZILN model
echo "=========================================="
echo "1/3: Training ZILN model..."
echo "=========================================="
ZILN_RUN_DIR="runs/ziln_${TIMESTAMP}"
python train_ziln_model.py \
    --loss_name ziln \
    --epochs ${EPOCHS} \
    --eval_interval ${EVAL_INTERVAL} \
    --batch_size ${BATCH_SIZE} \
    --learning_rate ${LEARNING_RATE} \
    --log_dir "${ZILN_RUN_DIR}"

if [ $? -ne 0 ]; then
    echo "Error: ZILN training failed"
    exit 1
fi

echo ""
echo "=========================================="
echo "2/3: Training MSE model..."
echo "=========================================="
MSE_RUN_DIR="runs/mse_${TIMESTAMP}"
python train_ziln_model.py \
    --loss_name mse \
    --epochs ${EPOCHS} \
    --eval_interval ${EVAL_INTERVAL} \
    --batch_size ${BATCH_SIZE} \
    --learning_rate ${LEARNING_RATE} \
    --log_dir "${MSE_RUN_DIR}"

if [ $? -ne 0 ]; then
    echo "Error: MSE training failed"
    exit 1
fi

echo ""
echo "=========================================="
echo "3/3: Generating comparison visualizations..."
echo "=========================================="
python compare_tensorboards.py \
    --plot \
    --eval1 "${ZILN_RUN_DIR}/eval_history.csv" \
    --eval2 "${MSE_RUN_DIR}/eval_history.csv" \
    --label1 ZILN \
    --label2 MSE \
    --output "runs/comparison_${TIMESTAMP}.png"

if [ $? -ne 0 ]; then
    echo "Error: Comparison plot generation failed"
    exit 1
fi

echo ""
echo "=========================================="
echo "Experiment Complete!"
echo "=========================================="
echo ""
echo "Results organized in runs directory:"
echo ""
echo "ZILN Model:"
echo "  Directory: ${ZILN_RUN_DIR}/"
echo "    - TensorBoard logs"
echo "    - Model: model_best.pt"
echo "    - Predictions: test_predictions.csv"
echo "    - Evaluation: eval_history.csv"
echo "    - Config: config.json"
echo ""
echo "MSE Model:"
echo "  Directory: ${MSE_RUN_DIR}/"
echo "    - TensorBoard logs"
echo "    - Model: model_best.pt"
echo "    - Predictions: test_predictions.csv"
echo "    - Evaluation: eval_history.csv"
echo "    - Config: config.json"
echo ""
echo "Comparison:"
echo "  Plot: runs/comparison_${TIMESTAMP}.png"
echo ""
echo "To view TensorBoard (both models):"
echo "  tensorboard --logdir runs"
echo ""
echo "To view comparison plot:"
echo "  open runs/comparison_${TIMESTAMP}.png"
echo ""
echo "=========================================="
