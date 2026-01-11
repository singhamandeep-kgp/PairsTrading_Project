"""
CRSP Daily Stock Price Data Retrieval (1962 onwards)
This script pulls daily stock price data from CRSP database via WRDS
"""

import wrds
import pandas as pd
from datetime import datetime
import os

class CRSPDataPuller:
    """Class to handle CRSP daily data retrieval from WRDS"""
    
    def __init__(self, username=None):
        """
        Initialize WRDS connection
        
        Args:
            username: WRDS username (if None, will use saved credentials)
        """
        self.username = username
        self.db = None
        
    def connect(self):
        """Establish connection to WRDS"""
        try:
            if self.username:
                self.db = wrds.Connection(wrds_username=self.username)
            else:
                self.db = wrds.Connection()
            print("Successfully connected to WRDS!")
            return True
        except Exception as e:
            print(f"Error connecting to WRDS: {e}")
            print("Make sure you have WRDS credentials set up.")
            return False
    
    def get_daily_stock_data(self, start_date='1962-01-01', end_date=None, 
                            exchanges=['NYSE', 'NASDAQ', 'AMEX'],
                            save_to_csv=True, output_file='crsp_daily_1962_onwards.csv'):
        """
        Pull daily stock price data from CRSP starting 1962
        
        Args:
            start_date: Start date (default: '1962-01-01')
            end_date: End date (YYYY-MM-DD). If None, gets latest available
            exchanges: List of exchanges to include
            save_to_csv: Whether to save results to CSV
            output_file: Output filename
            
        Returns:
            pandas DataFrame with daily stock price data
        """
        if not self.db:
            print("Not connected to WRDS. Connecting...")
            if not self.connect():
                return None
        
        # Set end date to today if not specified
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        print("="*60)
        print("Querying CRSP Daily Stock File (DSF)")
        print("="*60)
        print(f"Date range: {start_date} to {end_date}")
        print(f"Exchanges: {', '.join(exchanges)}")
        print("\nThis may take several minutes for large date ranges...")
        print("="*60)
        
        # Build SQL query for daily data
        query = f"""
        SELECT 
            a.permno,
            a.permco,
            a.date,
            b.ticker,
            b.comnam as company_name,
            b.exchcd,
            b.shrcd as share_code,
            a.prc as price,
            a.vol as volume,
            a.ret as return,
            a.retx as return_ex_dividend,
            a.shrout as shares_outstanding,
            a.openprc as open_price,
            a.askhi as ask_high,
            a.bidlo as bid_low,
            a.cfacshr as cum_factor_shares,
            a.cfacpr as cum_factor_price
        FROM 
            crsp.dsf AS a
        LEFT JOIN 
            crsp.dsenames AS b
        ON 
            a.permno = b.permno
            AND b.namedt <= a.date
            AND a.date <= b.nameendt
        WHERE 
            a.date >= '{start_date}'
            AND a.date <= '{end_date}'
        """
        
        # Add exchange filter (1=NYSE, 2=AMEX, 3=NASDAQ)
        exchange_codes = []
        if 'NYSE' in exchanges:
            exchange_codes.append('1')
        if 'AMEX' in exchanges:
            exchange_codes.append('2')
        if 'NASDAQ' in exchanges:
            exchange_codes.append('3')
        
        if exchange_codes:
            query += f" AND b.exchcd IN ({','.join(exchange_codes)})"
        
        query += " ORDER BY a.date, a.permno"
        
        try:
            # Execute query
            print("\nExecuting query...")
            df = self.db.raw_sql(query)
            
            print(f"\n{'='*60}")
            print("QUERY RESULTS:")
            print(f"{'='*60}")
            print(f"Total records retrieved: {len(df):,}")
            print(f"Date range in data: {df['date'].min()} to {df['date'].max()}")
            print(f"Number of unique stocks (PERMNOs): {df['permno'].nunique():,}")
            print(f"Number of trading days: {df['date'].nunique():,}")
            
            # Clean and process data
            print("\nProcessing data...")
            df = self._process_data(df)
            
            # Display summary statistics
            self._display_summary(df)
            
            # Save to CSV if requested
            if save_to_csv:
                print(f"\nSaving data to {output_file}...")
                df.to_csv(output_file, index=False)
                file_size = os.path.getsize(output_file) / (1024 * 1024)  # MB
                print(f"✓ Data saved successfully!")
                print(f"  File: {output_file}")
                print(f"  Size: {file_size:.2f} MB")
            
            return df
            
        except Exception as e:
            print(f"\n✗ Error retrieving data: {e}")
            return None
    
    def get_daily_data_by_year_range(self, start_year=1962, end_year=None,
                                     exchanges=['NYSE', 'NASDAQ', 'AMEX'],
                                     output_dir='crsp_data_by_year'):
        """
        Pull daily data year by year and save separately (for very large datasets)
        """
        if not self.db:
            if not self.connect():
                return None
        
        if end_year is None:
            end_year = datetime.now().year
        
        # Create output directory
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created directory: {output_dir}")
        
        all_data = {}
        
        for year in range(start_year, end_year + 1):
            print(f"\n{'='*60}")
            print(f"Processing Year: {year}")
            print(f"{'='*60}")
            
            start_date = f"{year}-01-01"
            end_date = f"{year}-12-31"
            output_file = os.path.join(output_dir, f"crsp_daily_{year}.csv")
            
            df_year = self.get_daily_stock_data(
                start_date=start_date,
                end_date=end_date,
                exchanges=exchanges,
                save_to_csv=True,
                output_file=output_file
            )
            
            if df_year is not None:
                all_data[year] = df_year
        
        print(f"\n{'='*60}")
        print(f"COMPLETE! Processed {len(all_data)} years")
        print(f"Files saved in: {output_dir}/")
        print(f"{'='*60}")
        
        return all_data
    
    def _process_data(self, df):
        """Process and clean the retrieved data"""
        # Convert price from negative to positive (CRSP stores negative for bid/ask average)
        df['price'] = df['price'].abs()
        
        # Calculate market cap (price * shares outstanding in thousands)
        df['market_cap'] = df['price'] * df['shares_outstanding'] * 1000
        
        # Convert date to datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Add exchange name
        exchange_map = {1: 'NYSE', 2: 'AMEX', 3: 'NASDAQ'}
        df['exchange_name'] = df['exchcd'].map(exchange_map)
        
        # Sort by date and permno
        df = df.sort_values(['date', 'permno']).reset_index(drop=True)
        
        return df
    
    def _display_summary(self, df):
        """Display summary statistics of the data"""
        print(f"\n{'='*60}")
        print("DATA SUMMARY:")
        print(f"{'='*60}")
        
        # Exchange distribution
        print("\nRecords by Exchange:")
        exchange_counts = df['exchange_name'].value_counts()
        for exchange, count in exchange_counts.items():
            print(f"  {exchange}: {count:,}")
        
        # Time period coverage
        print(f"\nTime Period:")
        print(f"  First date: {df['date'].min().strftime('%Y-%m-%d')}")
        print(f"  Last date: {df['date'].max().strftime('%Y-%m-%d')}")
        print(f"  Years covered: {df['date'].dt.year.nunique()}")
        
        # Data completeness
        print(f"\nData Completeness:")
        print(f"  Records with price: {df['price'].notna().sum():,} ({df['price'].notna().sum()/len(df)*100:.1f}%)")
        print(f"  Records with volume: {df['volume'].notna().sum():,} ({df['volume'].notna().sum()/len(df)*100:.1f}%)")
        print(f"  Records with return: {df['return'].notna().sum():,} ({df['return'].notna().sum()/len(df)*100:.1f}%)")
        
        # Sample stocks
        print(f"\nSample stocks in dataset:")
        sample_stocks = df[df['ticker'].notna()].groupby('ticker').first().head(10)
        for ticker in sample_stocks.index[:10]:
            company = sample_stocks.loc[ticker, 'company_name']
            print(f"  {ticker}: {company}")
    
    def get_stock_list(self, date='2024-01-01'):
        """
        Get list of all stocks available on a specific date
        """
        if not self.db:
            if not self.connect():
                return None
        
        query = f"""
        SELECT DISTINCT
            permno,
            ticker,
            comnam as company_name,
            exchcd,
            shrcd
        FROM 
            crsp.dsenames
        WHERE 
            '{date}' BETWEEN namedt AND nameendt
        ORDER BY 
            ticker
        """
        
        try:
            df = self.db.raw_sql(query)
            exchange_map = {1: 'NYSE', 2: 'AMEX', 3: 'NASDAQ'}
            df['exchange_name'] = df['exchcd'].map(exchange_map)
            return df
        except Exception as e:
            print(f"Error: {e}")
            return None
    
    def get_gics_sector_classification(self, output_file='/Users/amandeepsingh/Desktop/Quant codes/PairsTrading_Project/gics_sector_classification.csv'):
        """
        Retrieve GICS sector classification from WRDS and save to a CSV file.
        """
        if not self.db:
            if not self.connect():
                return None

        print("\nRetrieving GICS sector classification from WRDS...")

        query = """
        SELECT DISTINCT
            l.lpermno AS permno,
            c.gsector AS gics_sector, 
            c.gind AS gics_industry,
            c.gsubind AS gics_subindustry
        FROM 
            comp.company AS c
        INNER JOIN 
            crsp.ccmxpf_linktable AS l
        ON 
            c.gvkey = l.gvkey
        WHERE 
            c.gsector IS NOT NULL
            AND l.lpermno IS NOT NULL
            AND l.linktype IN ('LU', 'LC')
            AND l.linkprim IN ('P', 'C')
        ORDER BY 
            c.gsector, c.gind, c.gsubind
        """

        try:

            df = self.db.raw_sql(query)
            print(f"\nSaving GICS sector classification to {output_file}...")
            df.to_csv(output_file, index=False)
            print(f"✓ Data saved successfully to {output_file}")

            return df
        except Exception as e:
            print(f"Error retrieving GICS sector classification: {e}")
            return None
    
    def close(self):

        if self.db:
            self.db.close()
            print("\nConnection closed.")