"""
Load all CRSP data and create adjusted close prices DataFrame
"""

import pandas as pd
import numpy as np
import os
import glob

start = 2020
end = 2022

def create_adjusted_close_dataset(data_dir='crsp_data_by_year', start_year=start, end_year=end):
    """
    Load CRSP data for specified years and create a DataFrame with adjusted close prices

    Args:
        data_dir (str): Directory containing the CRSP data files
        start_year (int): First year to include (inclusive)
        end_year (int): Last year to include (inclusive)

    Returns:
        DataFrame with dates as index and tickers as columns (adjusted close prices)
    """

    print("=" * 60)
    print("CRSP ADJUSTED CLOSE PRICE AGGREGATOR")
    print("=" * 60)

    # Get all CSV files and filter by year range
    all_files = sorted(glob.glob(os.path.join(data_dir, 'crsp_daily_*.csv')))
    
    if not all_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")
    
    # Filter files by year range
    csv_files = []
    for file in all_files:
        try:
            year = int(os.path.basename(file).split('_')[-1].replace('.csv', ''))
            if start_year <= year <= end_year:
                csv_files.append((file, year))
        except (ValueError, IndexError):
            continue
    
    if not csv_files:
        raise ValueError(f"No files found in the specified year range {start_year}-{end_year}")
    
    print(f"\nFound {len(csv_files)} files for years {start_year} to {end_year}")
    print(f"Loading data from {data_dir}...\n")

    # Load all data
    all_data = []
    for file, year in csv_files:
        print(f"Loading {year}...", end=' ')

        df_year = pd.read_csv(file)
        print(f"✓ {len(df_year):,} records")

        all_data.append(df_year)

    # Combine all years
    print("\n" + "=" * 60)
    print("Combining all years...")
    df_all = pd.concat(all_data, ignore_index=True)

    print(f"Total records loaded: {len(df_all):,}")

    # Convert date to datetime
    print("Processing dates...")
    df_all['date'] = pd.to_datetime(df_all['date'])

    # Calculate adjusted close price
    print("Calculating adjusted close prices...")
    df_all['price'] = df_all['price'].abs()  # Make sure prices are positive
    df_all['adj_close'] = df_all['price'] / df_all['cum_factor_price']

    # Keep only necessary columns
    df_clean = df_all[['date', 'permno', 'adj_close', 'share_code']].copy()

    # Only keep ordinary shares
    df_clean = df_clean[df_clean['share_code'] in ['10', '11']]

    print(f"\nData Summary:")
    print(f"  Date range: {df_clean['date'].min()} to {df_clean['date'].max()}")
    print(f"  Trading days: {df_clean['date'].nunique():,}")
    print(f"  Unique stocks (PERMCOs): {df_clean['permno'].nunique():,}")
    print(f"  Total price records: {len(df_clean):,}")
    print(f"  Date range of data: {df_clean['date'].min().date()} to {df_clean['date'].max().date()}")

    # Pivot to wide format: dates as rows, PERMCOs as columns
    print("\nPivoting to wide format (this may take a few minutes)...")
    df_pivot = df_clean.pivot_table(
        index='date',
        columns='permno',  # Using PERMCO as the unique company identifier
        values='adj_close',
        aggfunc='first'  # In case of duplicates, take first
    )

    print(f"\nFinal DataFrame shape: {df_pivot.shape}")
    print(f"  Rows (dates): {len(df_pivot):,}")
    print(f"  Columns (stocks): {len(df_pivot.columns):,}")

    # Calculate data completeness
    total_cells = df_pivot.shape[0] * df_pivot.shape[1]
    non_null_cells = df_pivot.notna().sum().sum()
    completeness = (non_null_cells / total_cells) * 100

    print(f"\nData completeness: {completeness:.2f}%")

    print("\n" + "=" * 60)
    print("COMPLETE!")
    print("=" * 60)

    return df_pivot


def save_dataframe_pickle(df, filename):
    """
    Save DataFrame to a pickle file
    Args:
        df: DataFrame to save
        filename: Name of the pickle file (with .pkl extension)
    """
    df.to_pickle(filename)
