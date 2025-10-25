"""
Compare TensorBoard logs from two different training runs (e.g., ZILN vs MSE).

This script provides utilities for visualizing and comparing two models trained
with different loss functions to replicate the paper's comparison experiments.

Usage:
    # Compare ZILN and MSE logs
    python compare_tensorboards.py --log1 runs/ziln_20250125_123456 --log2 runs/mse_20250125_123457

    # Launch TensorBoard to compare multiple runs
    python compare_tensorboards.py --launch --logdir runs

    # Generate comparison plots from saved evaluation histories
    python compare_tensorboards.py --plot --eval1 models/eval_history_ziln.csv --eval2 models/eval_history_mse.csv
"""

import argparse
import subprocess
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Optional


def launch_tensorboard(logdir: str, port: int = 6006, host: str = 'localhost'):
    """
    Launch TensorBoard to compare multiple runs.

    Parameters
    ----------
    logdir : str
        Parent directory containing multiple run logs
    port : int
        Port to run TensorBoard on (default: 6006)
    host : str
        Host to bind to (default: localhost)
    """
    print(f"\nLaunching TensorBoard...")
    print(f"  Log directory: {logdir}")
    print(f"  URL: http://{host}:{port}")
    print("\nPress Ctrl+C to stop TensorBoard\n")

    try:
        subprocess.run([
            'tensorboard',
            '--logdir', logdir,
            '--port', str(port),
            '--host', host
        ])
    except KeyboardInterrupt:
        print("\nTensorBoard stopped.")
    except FileNotFoundError:
        print("Error: TensorBoard not found. Install with: pip install tensorboard")
        sys.exit(1)


def plot_comparison(
    eval_paths: List[str],
    labels: List[str],
    output_path: Optional[str] = None
):
    """
    Generate comparison plots from evaluation history CSV files.

    Parameters
    ----------
    eval_paths : List[str]
        Paths to evaluation history CSV files
    labels : List[str]
        Labels for each evaluation run
    output_path : str, optional
        Path to save comparison plots
    """
    # Load evaluation histories
    eval_dfs = []
    for path, label in zip(eval_paths, labels):
        df = pd.read_csv(path)
        df['model'] = label
        eval_dfs.append(df)

    if not eval_dfs:
        print("Error: No evaluation histories found")
        return

    # Create figure with subplots
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('Model Comparison: ZILN vs MSE', fontsize=16, fontweight='bold')

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    # Define metrics to plot
    metrics = [
        ('val_normalized_gini', 'Normalized Gini Coefficient', 'higher is better', True),
        ('val_spearman', "Spearman's Rank Correlation", 'higher is better', True),
        ('val_decile_mape', 'Decile-level MAPE (%)', 'lower is better', False),
        ('val_mae', 'Mean Absolute Error', 'lower is better', False),
        ('val_rmse', 'Root Mean Squared Error', 'lower is better', False),
        ('val_auc_pr', 'Area Under PR Curve', 'higher is better', True)
    ]

    for idx, (metric_name, title, direction, higher_better) in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]

        for i, (df, label) in enumerate(zip(eval_dfs, labels)):
            if metric_name in df.columns:
                ax.plot(
                    df['epoch'],
                    df[metric_name],
                    label=label,
                    color=colors[i % len(colors)],
                    linewidth=2,
                    marker='o',
                    markersize=4
                )

        ax.set_xlabel('Epoch', fontsize=10)
        ax.set_ylabel(title, fontsize=10)
        ax.set_title(f'{title}\n({direction})', fontsize=11, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nComparison plot saved to: {output_path}")
    else:
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        print("\nComparison plot saved to: model_comparison.png")

    plt.show()


def print_final_metrics_comparison(eval_paths: List[str], labels: List[str]):
    """
    Print a formatted table comparing final metrics.

    Parameters
    ----------
    eval_paths : List[str]
        Paths to evaluation history CSV files
    labels : List[str]
        Labels for each evaluation run
    """
    print("\n" + "=" * 80)
    print("FINAL METRICS COMPARISON (Last Epoch)")
    print("=" * 80)

    # Load final metrics from each run
    metrics_dict = {}
    for path, label in zip(eval_paths, labels):
        df = pd.read_csv(path)
        final_row = df.iloc[-1]

        metrics_dict[label] = {
            'normalized_gini': final_row.get('val_normalized_gini', 0),
            'spearman': final_row.get('val_spearman', 0),
            'decile_mape': final_row.get('val_decile_mape', 0),
            'mae': final_row.get('val_mae', 0),
            'rmse': final_row.get('val_rmse', 0),
            'auc_pr': final_row.get('val_auc_pr', 0)
        }

    # Print header
    print(f"{'Metric':<25}", end="")
    for label in labels:
        print(f"{label:>15}", end="")
    if len(labels) == 2:
        print(f"{'Improvement':>15}", end="")
    print()
    print("-" * 80)

    # Print each metric
    metric_names = [
        ('normalized_gini', 'Normalized Gini', False),
        ('spearman', "Spearman's Correlation", False),
        ('decile_mape', 'Decile MAPE (%)', True),
        ('mae', 'MAE', False),
        ('rmse', 'RMSE', False),
        ('auc_pr', 'AUC-PR', False)
    ]

    for metric_key, metric_display, is_percentage in metric_names:
        print(f"{metric_display:<25}", end="")

        values = [metrics_dict[label][metric_key] for label in labels]

        for value in values:
            if 'mape' in metric_key.lower() or is_percentage:
                print(f"{value:>14.1f}%", end="")
            else:
                print(f"{value:>15.4f}", end="")

        # Calculate improvement if comparing 2 models
        if len(labels) == 2:
            if values[1] != 0:
                if 'mape' in metric_key.lower() or metric_key in ['mae', 'rmse']:
                    # Lower is better
                    improvement = ((values[1] - values[0]) / values[1]) * 100
                else:
                    # Higher is better
                    improvement = ((values[0] - values[1]) / values[1]) * 100

                print(f"{improvement:>14.1f}%", end="")

        print()

    print("=" * 80)

    # Print paper comparison if labels match
    if set(labels) == {'ZILN', 'MSE'} or set(labels) == {'ziln', 'mse'}:
        print("\nPaper Results (Wang et al., 2019):")
        print("-" * 80)
        print(f"{'Metric':<25}{'DNN-ZILN':>15}{'DNN-MSE':>15}{'Improvement':>15}")
        print("-" * 80)
        print(f"{'Normalized Gini':<25}{0.368:>15.3f}{0.330:>15.3f}{11.4:>14.1f}%")
        print(f"{'Spearman Correlation':<25}{0.484:>15.3f}{0.327:>15.3f}{48.0:>14.1f}%")
        print(f"{'Decile MAPE (%)':<25}{22.6:>14.1f}%{72.8:>14.1f}%{-68.9:>14.1f}%")
        print("=" * 80)


def create_comparison_script(logdir: str = 'runs'):
    """
    Create a shell script to easily launch TensorBoard for comparison.

    Parameters
    ----------
    logdir : str
        Directory containing TensorBoard logs
    """
    script_content = f"""#!/bin/bash
# TensorBoard Comparison Script
# Auto-generated by compare_tensorboards.py

echo "Launching TensorBoard for model comparison..."
echo "Log directory: {logdir}"
echo ""
echo "Access TensorBoard at: http://localhost:6006"
echo "Press Ctrl+C to stop"
echo ""

tensorboard --logdir {logdir} --port 6006
"""

    script_path = Path('launch_tensorboard.sh')
    script_path.write_text(script_content)
    script_path.chmod(0o755)  # Make executable

    print(f"\nCreated script: {script_path}")
    print(f"Run with: ./{script_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Compare TensorBoard logs from different training runs'
    )

    parser.add_argument(
        '--launch',
        action='store_true',
        help='Launch TensorBoard for interactive comparison'
    )
    parser.add_argument(
        '--logdir',
        type=str,
        default='runs',
        help='Parent directory containing multiple run logs (default: runs)'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=6006,
        help='Port for TensorBoard (default: 6006)'
    )
    parser.add_argument(
        '--plot',
        action='store_true',
        help='Generate comparison plots from evaluation histories'
    )
    parser.add_argument(
        '--eval1',
        type=str,
        help='Path to first evaluation history CSV'
    )
    parser.add_argument(
        '--eval2',
        type=str,
        help='Path to second evaluation history CSV'
    )
    parser.add_argument(
        '--label1',
        type=str,
        default='ZILN',
        help='Label for first model (default: ZILN)'
    )
    parser.add_argument(
        '--label2',
        type=str,
        default='MSE',
        help='Label for second model (default: MSE)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='model_comparison.png',
        help='Output path for comparison plot'
    )
    parser.add_argument(
        '--create-script',
        action='store_true',
        help='Create a shell script to launch TensorBoard'
    )

    args = parser.parse_args()

    # Launch TensorBoard
    if args.launch:
        launch_tensorboard(args.logdir, args.port)

    # Generate comparison plots
    elif args.plot:
        if not args.eval1 or not args.eval2:
            print("Error: Both --eval1 and --eval2 are required for plotting")
            print("\nNote: With the new structure, eval files are in runs directories:")
            print("  Example: runs/ziln_20250126_123456/eval_history.csv")
            sys.exit(1)

        eval_paths = [args.eval1, args.eval2]
        labels = [args.label1, args.label2]

        # Check if files exist
        for path in eval_paths:
            if not Path(path).exists():
                print(f"Error: File not found: {path}")
                print(f"\nNote: Evaluation history is now saved in runs directories.")
                print(f"Look for: runs/<loss_name>_<timestamp>/eval_history.csv")
                sys.exit(1)

        # Print final metrics comparison
        print_final_metrics_comparison(eval_paths, labels)

        # Generate plots
        plot_comparison(eval_paths, labels, args.output)

    # Create shell script
    elif args.create_script:
        create_comparison_script(args.logdir)

    # Default: show usage
    else:
        parser.print_help()
        print("\n" + "=" * 80)
        print("QUICK START GUIDE")
        print("=" * 80)
        print("\n1. Train two models with different losses:")
        print("   python train_ziln_model.py --loss_name ziln --epochs 50")
        print("   python train_ziln_model.py --loss_name mse --epochs 50")
        print("\n2. Launch TensorBoard to compare interactively:")
        print("   python compare_tensorboards.py --launch --logdir runs")
        print("\n3. Generate comparison plots:")
        print("   python compare_tensorboards.py --plot \\")
        print("       --eval1 models/eval_history.csv \\")
        print("       --eval2 models/eval_history.csv \\")
        print("       --label1 ZILN --label2 MSE")
        print("\n4. Create a launch script:")
        print("   python compare_tensorboards.py --create-script")
        print("   ./launch_tensorboard.sh")
        print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
