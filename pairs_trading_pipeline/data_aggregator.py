"""
Load all CRSP data and create adjusted close prices DataFrame
"""

import pandas as pd
import numpy as np
import os
import glob
import pickle as cp

start = 2019
end = 2024

def create_adjusted_close_dataset(
    data_dir='pairs_trading_pipeline/crsp_data_by_year',
    start_year=start,
    end_year=end,
    gics_filename='gics_sector_classification.csv',
    output_parquet='gics_sector_time_series.parquet'
):
    """
    Load CRSP data for specified years, compute adjusted close, and bucket by GICS sector.

    """

    print("=" * 60)
    print("CRSP ADJUSTED CLOSE PRICE AGGREGATOR")
    print("=" * 60)

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
    print("Processing dates...")
    df_all['date'] = pd.to_datetime(df_all['date'])

    # Calculate adjusted close price
    print("Calculating adjusted close prices...")
    df_all['price'] = df_all['price'].abs()  # Make sure prices are positive
    df_all['adj_close'] = df_all['price'] / df_all['cum_factor_price']
    df_clean = df_all[['date', 'permno', 'adj_close', 'share_code']].copy()
    df_clean = df_clean[df_clean['share_code'].isin([10, 11])]

    # Load GICS mapping 
    gics_path = os.path.join(data_dir, gics_filename)
    if not os.path.exists(gics_path):
        raise FileNotFoundError(f"GICS mapping not found at {gics_path}")

    gics_df = pd.read_csv(gics_path)
    gics_df = gics_df[['permno', 'gics_sector']].dropna()
    gics_df['gics_sector'] = gics_df['gics_sector'].astype(str)
    df_clean = df_clean.merge(gics_df, on='permno', how='inner')

    print(f"\nData Summary:")
    print(f"  Date range of data: {df_clean['date'].min().date()} to {df_clean['date'].max().date()}")
    print(f"  Trading days: {df_clean['date'].nunique():,}")
    print(f"  Unique stocks (PERMNOs): {df_clean['permno'].nunique():,}")
    print(f"  GICS sectors represented: {df_clean['gics_sector'].nunique():,}")
    
    
    # Create output directory for GICS-filtered Pickle files
    output_dir = os.path.join(os.getcwd(), 'pairs_trading_pipeline/GICS_Filtered_Equities')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"\nCreated output directory: {output_dir}")

    # Split by GICS sector and save each as a separate Pickle file
    for sector, df_sector in df_clean.groupby('gics_sector'):
        print(f"\nProcessing GICS sector {sector}...")
        df_pivot = df_sector.pivot_table(
            index='date',
            columns='permno',
            values='adj_close',
            aggfunc='first'
        ).sort_index()

        # Save the DataFrame to a Pickle file
        sector_file = os.path.join(output_dir, f"GICS_{sector}.pkl")
        with open(sector_file, 'wb') as f:
            cp.dump(df_pivot, f)
        print(f"Saved Pickle file for GICS sector {sector}: {sector_file}")

    print("\n" + "=" * 60)
    print("COMPLETE!")
    print("=" * 60)
