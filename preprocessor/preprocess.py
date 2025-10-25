"""
Preprocessing module for customer purchase value prediction.

This module processes transaction data to predict each customer's total purchase
value in the next 12 months following their initial purchase.

Features:
- Initial purchase amount
- Number of items purchased in initial purchase
- Store chain
- Product category
- Product brand
- Product size measure
"""

import argparse
import os
import gc
import psutil
from pathlib import Path
from typing import Tuple, Optional, List
import pandas as pd
import numpy as np
from datetime import timedelta
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


def get_memory_usage():
    """
    Get current memory usage information.

    Returns
    -------
    dict
        Dictionary with memory statistics
    """
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    virtual_mem = psutil.virtual_memory()

    return {
        'process_mb': mem_info.rss / 1024 / 1024,
        'available_mb': virtual_mem.available / 1024 / 1024,
        'percent_used': virtual_mem.percent,
        'total_mb': virtual_mem.total / 1024 / 1024
    }


def check_memory_limit(threshold_percent: float = 85.0):
    """
    Check if memory usage exceeds threshold.

    Parameters
    ----------
    threshold_percent : float
        Threshold percentage for memory usage (default: 85%)

    Returns
    -------
    bool
        True if memory usage is below threshold, False otherwise
    """
    mem = get_memory_usage()
    if mem['percent_used'] > threshold_percent:
        print(f"\n⚠️  WARNING: High memory usage detected!")
        print(f"   Process: {mem['process_mb']:.1f} MB")
        print(f"   System: {mem['percent_used']:.1f}% used")
        print(f"   Available: {mem['available_mb']:.1f} MB")
        return False
    return True


def load_parquet_files_chunked(
    parquet_dir: str,
    n_files: int = 20,
    chunk_size: int = 5,
    memory_threshold: float = 85.0
) -> pd.DataFrame:
    """
    Load parquet files in chunks to avoid memory issues.

    Parameters
    ----------
    parquet_dir : str
        Directory containing parquet files
    n_files : int
        Number of parquet files to load
    chunk_size : int
        Number of files to load at once (default: 5)
    memory_threshold : float
        Memory usage threshold percentage to trigger warnings (default: 85%)

    Returns
    -------
    pd.DataFrame
        Combined transaction data
    """
    print(f"Loading {n_files} parquet files in chunks of {chunk_size}...")

    # Initial memory check
    mem = get_memory_usage()
    print(f"Initial memory: Process={mem['process_mb']:.1f} MB, "
          f"Available={mem['available_mb']:.1f} MB, "
          f"System={mem['percent_used']:.1f}%")

    all_dfs = []
    files_loaded = 0

    for chunk_start in tqdm(range(0, n_files, chunk_size), desc="Loading file chunks"):
        chunk_end = min(chunk_start + chunk_size, n_files)

        # Check memory before loading chunk
        if not check_memory_limit(memory_threshold):
            print(f"\n⚠️  Memory threshold exceeded before loading chunk {chunk_start}-{chunk_end}")
            print(f"   Stopping at {files_loaded} files to prevent OOM")
            break

        chunk_dfs = []
        for i in range(chunk_start, chunk_end):
            file_path = Path(parquet_dir) / f"part_{i:04d}.parquet"

            if file_path.exists():
                try:
                    df = pd.read_parquet(file_path)
                    chunk_dfs.append(df)
                    files_loaded += 1
                except Exception as e:
                    print(f"\n⚠️  Error loading {file_path}: {e}")
                    continue
            else:
                print(f"\n⚠️  Warning: {file_path} not found, skipping...")

        if chunk_dfs:
            # Combine chunk
            chunk_combined = pd.concat(chunk_dfs, ignore_index=True)
            all_dfs.append(chunk_combined)

            # Clear chunk data
            del chunk_dfs
            gc.collect()

            # Memory status
            mem = get_memory_usage()
            print(f"   After chunk {chunk_start//chunk_size + 1}: "
                  f"Process={mem['process_mb']:.1f} MB, "
                  f"System={mem['percent_used']:.1f}%")

    if not all_dfs:
        raise RuntimeError("No data loaded. Check parquet directory and file names.")

    print(f"\nCombining all chunks...")
    combined_df = pd.concat(all_dfs, ignore_index=True)

    # Clear intermediate data
    del all_dfs
    gc.collect()

    # Final memory check
    mem = get_memory_usage()
    print(f"Final memory: Process={mem['process_mb']:.1f} MB, "
          f"Available={mem['available_mb']:.1f} MB, "
          f"System={mem['percent_used']:.1f}%")
    print(f"Loaded {len(combined_df):,} transactions from {files_loaded} files")

    return combined_df


def load_parquet_files(parquet_dir: str, n_files: int = 20) -> pd.DataFrame:
    """
    Load first n parquet files and combine them.

    Parameters
    ----------
    parquet_dir : str
        Directory containing parquet files
    n_files : int, optional
        Number of parquet files to load (default: 20)

    Returns
    -------
    pd.DataFrame
        Combined transaction data
    """
    print(f"Loading {n_files} parquet files from {parquet_dir}...")

    dfs = []
    for i in tqdm(range(n_files), desc="Loading files"):
        file_path = Path(parquet_dir) / f"part_{i:04d}.parquet"
        if file_path.exists():
            df = pd.read_parquet(file_path)
            dfs.append(df)
        else:
            print(f"Warning: {file_path} not found, skipping...")

    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(combined_df):,} transactions from {len(dfs)} files")

    return combined_df


def preprocess_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess transaction data with memory optimization.

    Parameters
    ----------
    df : pd.DataFrame
        Raw transaction data

    Returns
    -------
    pd.DataFrame
        Preprocessed transactions with parsed dates
    """
    print("\nPreprocessing transactions...")

    # Memory before
    mem_before = get_memory_usage()

    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'])

    # Optimize data types to reduce memory
    print("Optimizing data types...")
    if 'id' in df.columns:
        df['id'] = df['id'].astype('int32')
    if 'company' in df.columns:
        df['company'] = df['company'].astype('int16')
    if 'chain' in df.columns:
        df['chain'] = df['chain'].astype('int16')
    if 'purchasequantity' in df.columns:
        df['purchasequantity'] = df['purchasequantity'].astype('int16')

    # Convert categorical columns to category dtype to save memory
    categorical_cols = ['category', 'brand', 'productmeasure']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype('category')

    # Sort by customer and date
    print("Sorting by customer and date...")
    df = df.sort_values(['id', 'date']).reset_index(drop=True)

    # Memory after
    mem_after = get_memory_usage()
    mem_saved = mem_before['process_mb'] - mem_after['process_mb']

    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Number of unique customers: {df['id'].nunique():,}")
    print(f"Number of unique companies: {df['company'].nunique():,}")
    print(f"Memory optimization: {abs(mem_saved):.1f} MB {'saved' if mem_saved > 0 else 'added'}")

    # Force garbage collection
    gc.collect()

    return df


def create_customer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features for each customer based on their initial purchase and
    calculate target (total purchase in next 12 months).

    Memory-optimized version that processes data efficiently.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed transaction data

    Returns
    -------
    pd.DataFrame
        Customer-level features and targets
    """
    print("\nCreating customer features...")

    # Memory check
    check_memory_limit()

    # Get first purchase date for each customer
    print("Computing first purchase dates...")
    customer_first_purchase = df.groupby('id')['date'].min().reset_index()
    customer_first_purchase.columns = ['id', 'first_purchase_date']

    # Merge back to get first purchase info
    print("Merging first purchase dates...")
    df = df.merge(customer_first_purchase, on='id', how='left')

    # Clear temporary data
    del customer_first_purchase
    gc.collect()

    # Define 12 months window after first purchase
    df['twelve_months_after'] = df['first_purchase_date'] + timedelta(days=365)

    # Separate initial purchases from future purchases
    print("Separating initial and future purchases...")
    initial_purchases = df[df['date'] == df['first_purchase_date']].copy()
    future_purchases = df[
        (df['date'] > df['first_purchase_date']) &
        (df['date'] <= df['twelve_months_after'])
    ].copy()

    # Clear original dataframe to free memory
    del df
    gc.collect()

    print(f"Initial purchase transactions: {len(initial_purchases):,}")
    print(f"Future purchase transactions (within 12 months): {len(future_purchases):,}")

    # Memory check after split
    check_memory_limit()

    # Aggregate initial purchase features by customer
    print("\nAggregating initial purchase features...")
    initial_features = initial_purchases.groupby('id').agg({
        'purchaseamount': 'sum',  # Total initial purchase amount
        'purchasequantity': 'sum',  # Total items purchased
        'chain': 'first',  # Store chain (assuming same for initial purchase)
        'company': 'first',  # Company (for train/test split)
        'date': 'first'  # First purchase date
    }).reset_index()

    initial_features.columns = [
        'id', 'initial_purchase_amount', 'initial_num_items',
        'chain', 'company', 'first_purchase_date'
    ]

    # Add categorical features from initial purchases (one-hot encode approach)
    # Get most common category, brand, and productmeasure for each customer's initial purchase
    categorical_features = initial_purchases.groupby('id').agg({
        'category': lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0],
        'brand': lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0],
        'productmeasure': lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0]
    }).reset_index()

    categorical_features.columns = ['id', 'initial_category', 'initial_brand', 'initial_productmeasure']

    # Merge categorical features
    initial_features = initial_features.merge(categorical_features, on='id')

    # Calculate target: total purchase amount in next 12 months
    print("\nCalculating target variable (12-month purchase value)...")
    future_totals = future_purchases.groupby('id')['purchaseamount'].sum().reset_index()
    future_totals.columns = ['id', 'future_12m_purchase_value']

    # Merge features with target
    print("Merging features with targets...")
    customer_data = initial_features.merge(future_totals, on='id', how='left')

    # Clear intermediate dataframes
    del initial_features, future_totals, initial_purchases, future_purchases
    gc.collect()

    # Fill NaN targets with 0 (customers who didn't make future purchases)
    customer_data['future_12m_purchase_value'] = customer_data['future_12m_purchase_value'].fillna(0)

    print(f"\nFinal dataset shape: {customer_data.shape}")
    print(f"Number of customers: {len(customer_data):,}")
    print(f"Customers with repeat purchases: {(customer_data['future_12m_purchase_value'] > 0).sum():,}")
    print(f"Average 12-month purchase value: ${customer_data['future_12m_purchase_value'].mean():.2f}")

    # Final memory check
    mem = get_memory_usage()
    print(f"Memory after feature creation: Process={mem['process_mb']:.1f} MB, "
          f"System={mem['percent_used']:.1f}%")

    return customer_data


def train_test_split_by_company(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into train and test sets, stratified by company.
    For each company, randomly select test_size fraction of customers.

    Parameters
    ----------
    df : pd.DataFrame
        Customer-level features and targets
    test_size : float, optional
        Fraction of customers to use for testing (default: 0.2)
    random_state : int, optional
        Random seed for reproducibility (default: 42)

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        Training and test dataframes
    """
    print(f"\nSplitting data by company ({int((1-test_size)*100)}% train, {int(test_size*100)}% test)...")

    np.random.seed(random_state)

    train_dfs = []
    test_dfs = []

    for company_id in tqdm(df['company'].unique(), desc="Processing companies"):
        company_data = df[df['company'] == company_id].copy()

        # Randomly shuffle customers for this company
        company_data = company_data.sample(frac=1, random_state=random_state)

        # Split into train and test
        n_test = int(len(company_data) * test_size)
        test_data = company_data.iloc[:n_test]
        train_data = company_data.iloc[n_test:]

        train_dfs.append(train_data)
        test_dfs.append(test_data)

    train_df = pd.concat(train_dfs, ignore_index=True)
    test_df = pd.concat(test_dfs, ignore_index=True)

    print(f"\nTrain set: {len(train_df):,} customers")
    print(f"Test set: {len(test_df):,} customers")
    print(f"Train companies: {train_df['company'].nunique()}")
    print(f"Test companies: {test_df['company'].nunique()}")

    return train_df, test_df


def save_processed_data(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: str
) -> None:
    """
    Save processed train and test data.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training data
    test_df : pd.DataFrame
        Test data
    output_dir : str
        Directory to save processed data
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    train_path = output_path / "train.parquet"
    test_path = output_path / "test.parquet"

    print(f"\nSaving processed data to {output_dir}...")
    train_df.to_parquet(train_path, index=False, compression='snappy')
    test_df.to_parquet(test_path, index=False, compression='snappy')

    print(f" Train data saved: {train_path} ({len(train_df):,} rows)")
    print(f" Test data saved: {test_path} ({len(test_df):,} rows)")

    # Save data summary
    summary_path = output_path / "data_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("Data Preprocessing Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Train set:\n")
        f.write(f"  - Customers: {len(train_df):,}\n")
        f.write(f"  - Companies: {train_df['company'].nunique()}\n")
        f.write(f"  - Avg future purchase value: ${train_df['future_12m_purchase_value'].mean():.2f}\n")
        f.write(f"  - Customers with repeat purchases: {(train_df['future_12m_purchase_value'] > 0).sum():,}\n\n")

        f.write(f"Test set:\n")
        f.write(f"  - Customers: {len(test_df):,}\n")
        f.write(f"  - Companies: {test_df['company'].nunique()}\n")
        f.write(f"  - Avg future purchase value: ${test_df['future_12m_purchase_value'].mean():.2f}\n")
        f.write(f"  - Customers with repeat purchases: {(test_df['future_12m_purchase_value'] > 0).sum():,}\n\n")

        f.write(f"Features:\n")
        for col in train_df.columns:
            f.write(f"  - {col}: {train_df[col].dtype}\n")

    print(f" Summary saved: {summary_path}")


def main():
    """Main preprocessing pipeline."""
    parser = argparse.ArgumentParser(
        description="Preprocess transaction data for customer value prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--parquet_dir',
        type=str,
        default='/mnt/d/datasets/acquire-valued-shoppers-challenge/parquet',
        help='Directory containing parquet files (default: /mnt/d/datasets/acquire-valued-shoppers-challenge/parquet)'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default='./data/processed',
        help='Directory to save processed data (default: ./data/processed)'
    )

    parser.add_argument(
        '--n_files',
        type=int,
        default=20,
        help='Number of parquet files to load (default: 20)'
    )

    parser.add_argument(
        '--test_size',
        type=float,
        default=0.2,
        help='Fraction of customers for test set (default: 0.2)'
    )

    parser.add_argument(
        '--random_state',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )

    parser.add_argument(
        '--chunk_size',
        type=int,
        default=5,
        help='Number of files to load at once (default: 5, reduce if OOM)'
    )

    parser.add_argument(
        '--memory_threshold',
        type=float,
        default=85.0,
        help='Memory usage threshold percentage (default: 85.0)'
    )

    parser.add_argument(
        '--use_chunked_loading',
        action='store_true',
        help='Use chunked loading to prevent OOM (recommended for large datasets)'
    )

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.parquet_dir):
        parser.error(f"Parquet directory does not exist: {args.parquet_dir}")

    print("=" * 60)
    print("Customer Purchase Value Prediction - Data Preprocessing")
    print("=" * 60)

    # Display system info
    mem = get_memory_usage()
    print(f"\nSystem Information:")
    print(f"  Total RAM: {mem['total_mb']:.1f} MB")
    print(f"  Available RAM: {mem['available_mb']:.1f} MB")
    print(f"  Current usage: {mem['percent_used']:.1f}%")
    print(f"  Memory threshold: {args.memory_threshold}%")

    # Decide whether to use chunked loading
    use_chunked = args.use_chunked_loading or args.n_files > 20

    if use_chunked:
        print(f"\n⚙️  Using chunked loading (chunk_size={args.chunk_size})")
        print(f"   Recommended for large datasets to prevent OOM")

    # Step 1: Load data
    if use_chunked:
        df = load_parquet_files_chunked(
            args.parquet_dir,
            args.n_files,
            chunk_size=args.chunk_size,
            memory_threshold=args.memory_threshold
        )
    else:
        print(f"\n⚙️  Using standard loading")
        print(f"   Note: Use --use_chunked_loading if you encounter OOM errors")
        df = load_parquet_files(args.parquet_dir, args.n_files)

    # Step 2: Preprocess
    df = preprocess_transactions(df)

    # Step 3: Create features and target
    customer_data = create_customer_features(df)

    # Step 4: Train/test split by company
    train_df, test_df = train_test_split_by_company(
        customer_data,
        test_size=args.test_size,
        random_state=args.random_state
    )

    # Step 5: Save processed data
    save_processed_data(train_df, test_df, args.output_dir)

    print("\n" + "=" * 60)
    print("Preprocessing complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
