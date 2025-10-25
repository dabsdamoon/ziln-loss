"""
Convert large CSV/CSV.GZ files to Parquet format in chunks.

This module provides functionality to efficiently convert large CSV files
(compressed or uncompressed) to Parquet format by processing them in chunks.
"""

import argparse
import os
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from typing import Optional


def convert_csv_to_parquet(
    csv_path: str,
    parquet_dir: str,
    chunksize: int = 1_000_000,
    compression: Optional[str] = None
) -> None:
    """
    Convert a CSV file to multiple Parquet files in chunks.

    Parameters
    ----------
    csv_path : str
        Path to the input CSV file (can be .csv or .csv.gz)
    parquet_dir : str
        Directory where Parquet files will be saved
    chunksize : int, optional
        Number of rows per chunk (default: 1,000,000)
    compression : str, optional
        Compression type ('gzip', 'bz2', 'zip', 'xz', None)
        If None, will auto-detect from file extension
    """
    # Create output directory if it doesn't exist
    parquet_path = Path(parquet_dir)
    parquet_path.mkdir(parents=True, exist_ok=True)

    # Auto-detect compression from file extension
    if compression is None and csv_path.endswith('.gz'):
        compression = 'gzip'

    print(f"Converting {csv_path} to Parquet format...")
    print(f"Output directory: {parquet_dir}")
    print(f"Chunk size: {chunksize:,} rows")
    print(f"Compression: {compression}")
    print("-" * 60)

    total_rows = 0
    chunk_count = 0

    try:
        for i, chunk in tqdm(enumerate(pd.read_csv(
            csv_path,
            chunksize=chunksize,
            compression=compression
        ))):
            # Generate output filename
            output_file = parquet_path / f"part_{i:04d}.parquet"

            # Write chunk to parquet
            chunk.to_parquet(
                output_file,
                engine='pyarrow',
                index=False,
                compression='snappy'
            )

            total_rows += len(chunk)
            chunk_count += 1

            print(f"Chunk {i:4d}: {len(chunk):,} rows -> {output_file.name}")

        print("-" * 60)
        print(f"Conversion complete!")
        print(f"Total chunks: {chunk_count}")
        print(f"Total rows: {total_rows:,}")
        print(f"Output directory: {parquet_dir}")

    except Exception as e:
        print(f"Error during conversion: {e}")
        raise


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Convert large CSV/CSV.GZ files to Parquet format in chunks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            %(prog)s input.csv.gz ./output_parquet/
            %(prog)s /path/to/data.csv /path/to/parquet/ --chunksize 500000
        """
    )

    parser.add_argument(
        '--csv_path',
        type=str,
        help='Path to input CSV file (can be .csv or .csv.gz)'
    )

    parser.add_argument(
        '--parquet_dir',
        type=str,
        help='Directory where Parquet files will be saved'
    )

    parser.add_argument(
        '--chunksize',
        type=int,
        default=1_000_000,
        help='Number of rows per chunk (default: 1,000,000)'
    )

    parser.add_argument(
        '--compression',
        type=str,
        choices=['gzip', 'bz2', 'zip', 'xz', 'none'],
        default=None,
        help='Compression type (auto-detected from extension if not specified)'
    )

    args = parser.parse_args()

    # Validate input file exists
    if not os.path.exists(args.csv_path):
        parser.error(f"Input file does not exist: {args.csv_path}")

    # Convert 'none' string to None
    compression = None if args.compression == 'none' else args.compression

    # Run conversion
    convert_csv_to_parquet(
        csv_path=args.csv_path,
        parquet_dir=args.parquet_dir,
        chunksize=args.chunksize,
        compression=compression
    )


if __name__ == "__main__":
    main()
