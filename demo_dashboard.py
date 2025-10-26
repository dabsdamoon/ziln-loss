"""
Interactive Demo Dashboard for ZILN-Loss Project

This Streamlit dashboard provides:
1. EDA Results Visualization (cached)
2. Model Runs Comparison
3. Prediction Analysis
4. Performance Metrics

Usage:
    streamlit run demo_dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import json
from datetime import datetime
import sys

# Page configuration
st.set_page_config(
    page_title="ZILN-Loss Demo Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_data
def load_eda_data():
    """Load and cache EDA results from preprocessed data."""
    try:
        train_df = pd.read_parquet('data/processed/train.parquet')
        test_df = pd.read_parquet('data/processed/test.parquet')

        # Combine for overall statistics
        full_df = pd.concat([train_df, test_df], ignore_index=True)

        eda_results = {
            'train_size': len(train_df),
            'test_size': len(test_df),
            'total_size': len(full_df),
            'target_stats': {
                'mean': float(full_df['future_12m_purchase_value'].mean()),
                'median': float(full_df['future_12m_purchase_value'].median()),
                'std': float(full_df['future_12m_purchase_value'].std()),
                'min': float(full_df['future_12m_purchase_value'].min()),
                'max': float(full_df['future_12m_purchase_value'].max()),
                'zero_rate': float((full_df['future_12m_purchase_value'] == 0).mean())
            },
            'train_df': train_df,
            'test_df': test_df,
            'full_df': full_df
        }

        return eda_results
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None


@st.cache_data
def get_available_runs():
    """Get all available model runs from runs/ directory."""
    runs_dir = Path('runs')
    if not runs_dir.exists():
        return []

    runs = []
    for run_path in runs_dir.iterdir():
        if run_path.is_dir():
            config_path = run_path / 'config.json'
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = json.load(f)

                runs.append({
                    'name': run_path.name,
                    'path': run_path,
                    'config': config,
                    'timestamp': run_path.name.split('_')[-1] if '_' in run_path.name else 'unknown'
                })

    return sorted(runs, key=lambda x: x['timestamp'], reverse=True)


def load_run_results(run_path):
    """Load results from a specific run."""
    results = {}

    # Load config
    config_path = run_path / 'config.json'
    if config_path.exists():
        with open(config_path, 'r') as f:
            results['config'] = json.load(f)

    # Load predictions
    pred_path = run_path / 'test_predictions.csv'
    if pred_path.exists():
        results['predictions'] = pd.read_csv(pred_path)

    # Load eval history (if available)
    eval_path = run_path / 'eval_history.csv'
    if eval_path.exists():
        results['eval_history'] = pd.read_csv(eval_path)

    return results


def plot_target_distribution(eda_results):
    """Plot target variable distribution."""
    df = eda_results['full_df']
    target = df['future_12m_purchase_value']

    # Create figure with secondary y-axis
    fig = go.Figure()

    # Histogram with more bins for better granularity
    fig.add_trace(go.Histogram(
        x=target,
        nbinsx=150,  # Increased from 50 to 150 for finer detail
        name='Distribution',
        marker_color='lightblue',
        opacity=0.7
    ))

    fig.update_layout(
        title='Target Variable Distribution (12-Month Future Purchase Value)',
        xaxis_title='LTV ($)',
        yaxis_title='Count',
        height=400,
        showlegend=True
    )

    return fig


def plot_zero_inflation(eda_results):
    """Plot zero-inflation analysis."""
    df = eda_results['full_df']
    target = df['future_12m_purchase_value']

    zero_count = (target == 0).sum()
    non_zero_count = (target > 0).sum()

    fig = go.Figure(data=[
        go.Pie(
            labels=['Zero LTV', 'Non-Zero LTV'],
            values=[zero_count, non_zero_count],
            hole=0.4,
            marker_colors=['#ff7f0e', '#1f77b4']
        )
    ])

    fig.update_layout(
        title=f'Zero-Inflation Analysis (Zero Rate: {eda_results["target_stats"]["zero_rate"]:.1%})',
        height=400
    )

    return fig


def plot_metrics_comparison(runs_data):
    """Plot metrics comparison across different models."""
    metrics_to_plot = [
        'normalized_gini',
        'spearman',
        'mae',
        'rmse',
        'decile_mape',
        'auc_pr'
    ]

    data = []
    for run_name, run_results in runs_data.items():
        if 'config' in run_results and 'final_metrics' in run_results['config']:
            metrics = run_results['config']['final_metrics']
            model_type = run_results['config'].get('model_type', 'unknown')

            for metric in metrics_to_plot:
                if metric in metrics:
                    data.append({
                        'Model': run_name.split('_')[0].upper(),
                        'Metric': metric.replace('_', ' ').title(),
                        'Value': metrics[metric]
                    })

    if not data:
        return None

    df = pd.DataFrame(data)

    # Create subplots for different metrics
    fig = px.bar(
        df,
        x='Model',
        y='Value',
        color='Model',
        facet_col='Metric',
        facet_col_wrap=3,
        title='Model Performance Comparison',
        height=600
    )

    fig.update_yaxes(matches=None)

    return fig


def plot_predictions_comparison(runs_data):
    """Plot prediction comparisons."""
    fig = go.Figure()

    colors = px.colors.qualitative.Plotly

    for idx, (run_name, run_results) in enumerate(runs_data.items()):
        if 'predictions' in run_results:
            preds_df = run_results['predictions']

            # Sort by true value for better visualization
            preds_df = preds_df.sort_values('true_ltv').reset_index(drop=True)

            # Sample for performance
            if len(preds_df) > 500:
                preds_df = preds_df.sample(500).sort_values('true_ltv').reset_index(drop=True)

            model_name = run_name.split('_')[0].upper()

            fig.add_trace(go.Scatter(
                x=preds_df['true_ltv'],
                y=preds_df['predicted_ltv'],
                mode='markers',
                name=model_name,
                marker=dict(size=5, color=colors[idx % len(colors)], opacity=0.6)
            ))

    # Add perfect prediction line
    max_val = max([run_results.get('predictions', pd.DataFrame({'true_ltv': [0]}))['true_ltv'].max()
                   for run_results in runs_data.values()])

    fig.add_trace(go.Scatter(
        x=[0, max_val],
        y=[0, max_val],
        mode='lines',
        name='Perfect Prediction',
        line=dict(color='black', dash='dash'),
        showlegend=True
    ))

    fig.update_layout(
        title='Predicted vs True LTV',
        xaxis_title='True LTV ($)',
        yaxis_title='Predicted LTV ($)',
        height=500,
        hovermode='closest'
    )

    return fig


def plot_training_curves(runs_data):
    """Plot training curves for models with eval_history."""
    fig = go.Figure()

    colors = px.colors.qualitative.Plotly

    for idx, (run_name, run_results) in enumerate(runs_data.items()):
        if 'eval_history' in run_results:
            eval_df = run_results['eval_history']
            model_name = run_name.split('_')[0].upper()

            # Get iteration column (could be 'epoch' or 'iteration')
            iter_col = 'iteration' if 'iteration' in eval_df.columns else 'epoch'

            # Plot validation Gini (primary metric)
            if 'val_normalized_gini' in eval_df.columns:
                fig.add_trace(go.Scatter(
                    x=eval_df[iter_col],
                    y=eval_df['val_normalized_gini'],
                    mode='lines+markers',
                    name=model_name,
                    line=dict(color=colors[idx % len(colors)], width=2),
                    marker=dict(size=4)
                ))

    fig.update_layout(
        title='Training Progress: Normalized Gini (Primary Metric)',
        xaxis_title='Iteration/Epoch',
        yaxis_title='Normalized Gini',
        height=400,
        hovermode='x unified'
    )

    return fig


def main():
    # Header
    st.markdown('<h1 class="main-header">📊 ZILN-Loss Demo Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("### Interactive visualization of EDA results and model comparison")
    st.markdown("---")

    # Sidebar
    st.sidebar.header("⚙️ Settings")
    page = st.sidebar.radio(
        "Navigate to:",
        ["🏠 Overview", "📈 EDA Results", "🤖 Model Comparison", "📊 Detailed Analysis"]
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📁 About")
    st.sidebar.info(
        "This dashboard visualizes results from the ZILN-Loss project, "
        "comparing different LTV prediction models."
    )

    # Load data
    eda_results = load_eda_data()
    runs = get_available_runs()

    # Page: Overview
    if page == "🏠 Overview":
        st.header("📋 Project Overview")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Total Samples",
                f"{eda_results['total_size']:,}" if eda_results else "N/A"
            )

        with col2:
            st.metric(
                "Models Trained",
                len(runs)
            )

        with col3:
            st.metric(
                "Zero-Inflation Rate",
                f"{eda_results['target_stats']['zero_rate']:.1%}" if eda_results else "N/A"
            )

        st.markdown("---")

        # Dataset info
        st.subheader("📊 Dataset Information")
        if eda_results:
            col1, col2 = st.columns(2)

            with col1:
                st.write("**Training Set:**")
                st.write(f"- Samples: {eda_results['train_size']:,}")

                st.write("**Test Set:**")
                st.write(f"- Samples: {eda_results['test_size']:,}")

            with col2:
                st.write("**Target Statistics:**")
                stats = eda_results['target_stats']
                st.write(f"- Mean: ${stats['mean']:.2f}")
                st.write(f"- Median: ${stats['median']:.2f}")
                st.write(f"- Std Dev: ${stats['std']:.2f}")
                st.write(f"- Range: ${stats['min']:.2f} - ${stats['max']:.2f}")

        st.markdown("---")

        # Available runs
        st.subheader("🤖 Available Model Runs")
        if runs:
            runs_df = pd.DataFrame([
                {
                    'Model': r['name'].split('_')[0].upper(),
                    'Run Name': r['name'],
                    'Type': r['config'].get('model_type', 'N/A'),
                    'Samples': r['config'].get('train_samples', 'N/A'),
                    'Timestamp': r['timestamp']
                }
                for r in runs
            ])
            st.dataframe(runs_df, width='stretch')
        else:
            st.warning("No trained models found in `runs/` directory. Train some models first!")

    # Page: EDA Results
    elif page == "📈 EDA Results":
        st.header("📈 Exploratory Data Analysis")

        if not eda_results:
            st.error("Could not load data. Make sure `data/processed/train.parquet` exists.")
            return

        # Target distribution
        st.subheader("Distribution of Target Variable")
        fig = plot_target_distribution(eda_results)
        st.plotly_chart(fig, config={'displayModeBar': True})

        col1, col2 = st.columns(2)

        with col1:
            # Zero-inflation
            st.subheader("Zero-Inflation Analysis")
            fig = plot_zero_inflation(eda_results)
            st.plotly_chart(fig, config={'displayModeBar': True})

        with col2:
            # Summary statistics
            st.subheader("Summary Statistics")
            stats = eda_results['target_stats']

            stats_df = pd.DataFrame({
                'Metric': ['Mean', 'Median', 'Std Dev', 'Min', 'Max', 'Zero Rate'],
                'Value': [
                    f"${stats['mean']:.2f}",
                    f"${stats['median']:.2f}",
                    f"${stats['std']:.2f}",
                    f"${stats['min']:.2f}",
                    f"${stats['max']:.2f}",
                    f"{stats['zero_rate']:.1%}"
                ]
            })
            st.dataframe(stats_df, width='stretch', hide_index=True)

        # Distribution by quantiles
        st.subheader("Distribution by Quantiles")
        target = eda_results['full_df']['future_12m_purchase_value']
        quantiles = [0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]
        quantile_values = target.quantile(quantiles)

        quantile_df = pd.DataFrame({
            'Quantile': [f'{q:.0%}' for q in quantiles],
            'Value': [f'${v:.2f}' for v in quantile_values]
        })
        st.dataframe(quantile_df, width='stretch', hide_index=True)

    # Page: Model Comparison
    elif page == "🤖 Model Comparison":
        st.header("🤖 Model Performance Comparison")

        if not runs:
            st.warning("No trained models found. Train some models first!")
            return

        # Select models to compare
        st.subheader("Select Models to Compare")
        selected_runs = st.multiselect(
            "Choose models:",
            [r['name'] for r in runs],
            default=[r['name'] for r in runs[:min(3, len(runs))]]
        )

        if not selected_runs:
            st.info("Select at least one model to compare.")
            return

        # Load selected runs data
        runs_data = {}
        for run_name in selected_runs:
            run = next(r for r in runs if r['name'] == run_name)
            runs_data[run_name] = load_run_results(run['path'])

        # Metrics comparison
        st.subheader("📊 Performance Metrics")
        fig = plot_metrics_comparison(runs_data)
        if fig:
            st.plotly_chart(fig, config={'displayModeBar': True})
        else:
            st.warning("No metrics data available for selected models.")

        # Metrics table
        st.subheader("📋 Detailed Metrics Table")
        metrics_data = []
        for run_name, run_results in runs_data.items():
            if 'config' in run_results and 'final_metrics' in run_results['config']:
                metrics = run_results['config']['final_metrics']
                model_name = run_name.split('_')[0].upper()

                metrics_data.append({
                    'Model': model_name,
                    'Gini': f"{metrics.get('normalized_gini', 0):.4f}",
                    'Spearman': f"{metrics.get('spearman', 0):.4f}",
                    'MAE': f"{metrics.get('mae', 0):.2f}",
                    'RMSE': f"{metrics.get('rmse', 0):.2f}",
                    'MAPE': f"{metrics.get('decile_mape', 0):.1f}%",
                    'AUC-PR': f"{metrics.get('auc_pr', 0):.4f}"
                })

        if metrics_data:
            st.dataframe(pd.DataFrame(metrics_data), width='stretch', hide_index=True)

        # Training curves
        st.subheader("📈 Training Progress")
        fig = plot_training_curves(runs_data)
        st.plotly_chart(fig, config={'displayModeBar': True})

    # Page: Detailed Analysis
    elif page == "📊 Detailed Analysis":
        st.header("📊 Detailed Prediction Analysis")

        if not runs:
            st.warning("No trained models found. Train some models first!")
            return

        # Select models
        selected_runs = st.multiselect(
            "Choose models to analyze:",
            [r['name'] for r in runs],
            default=[r['name'] for r in runs[:min(3, len(runs))]]
        )

        if not selected_runs:
            st.info("Select at least one model to analyze.")
            return

        # Load data
        runs_data = {}
        for run_name in selected_runs:
            run = next(r for r in runs if r['name'] == run_name)
            runs_data[run_name] = load_run_results(run['path'])

        # Predictions scatter plot
        st.subheader("Predicted vs True LTV")
        fig = plot_predictions_comparison(runs_data)
        st.plotly_chart(fig, config={'displayModeBar': True})

        # Error analysis
        st.subheader("📉 Error Analysis")

        for run_name, run_results in runs_data.items():
            if 'predictions' in run_results:
                st.write(f"**{run_name.split('_')[0].upper()}**")

                preds_df = run_results['predictions']
                errors = preds_df['predicted_ltv'] - preds_df['true_ltv']

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Mean Error", f"${errors.mean():.2f}")

                with col2:
                    st.metric("MAE", f"${np.abs(errors).mean():.2f}")

                with col3:
                    st.metric("RMSE", f"${np.sqrt((errors**2).mean()):.2f}")

                with col4:
                    st.metric("Max Error", f"${np.abs(errors).max():.2f}")

                # Error histogram
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=errors,
                    nbinsx=50,
                    name='Error Distribution',
                    marker_color='lightcoral'
                ))
                fig.update_layout(
                    title=f'Prediction Error Distribution - {run_name.split("_")[0].upper()}',
                    xaxis_title='Error (Predicted - True)',
                    yaxis_title='Count',
                    height=300
                )
                st.plotly_chart(fig, config={'displayModeBar': True})

                st.markdown("---")


if __name__ == "__main__":
    main()
