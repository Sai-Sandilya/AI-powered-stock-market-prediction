import os
import pandas as pd
import requests
import sys
from io import StringIO
from datetime import datetime
import numpy as np

# Add the 'dags' directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config_manager import load_config, get_api_key

def _generate_sample_insider_data(symbols: list) -> pd.DataFrame:
    """Generates sample insider transaction data if the Alpha Vantage key is not available."""
    print("⚠️ Alpha Vantage API key not found. Generating sample insider transactions.")
    
    all_transactions = []
    config = load_config()
    date_range = pd.date_range(start=config['data_fetching']['start_date'], end=config['data_fetching']['end_date'], freq='D')
    np.random.seed(42)

    for symbol in symbols:
        # Create a few dummy transactions for each symbol
        for date in date_range[::60]: # One transaction every 60 days
            transaction_type = np.random.choice(['P-Purchase', 'S-Sale'])
            shares = np.random.randint(100, 5000)
            price = np.random.uniform(100, 500)
            all_transactions.append({
                'symbol': symbol,
                'transactionDate': date,
                'reportingName': 'Sample Insider',
                'transactionType': transaction_type,
                'transactionShares': shares,
                'price': price,
            })
    
    if not all_transactions:
        return pd.DataFrame()
        
    df = pd.DataFrame(all_transactions)
    df.rename(columns={'transactionDate': 'Date'}, inplace=True)
    return df

def fetch_insider_transactions(symbols: list):
    """
    Fetches insider trading transactions from Alpha Vantage, with a fallback to sample data.
    """
    print(f"Fetching Alpha Vantage insider transactions for {len(symbols)} symbols...")
    
    try:
        api_key = get_api_key('alpha_vantage')
        all_transactions = []
        
        for symbol in symbols:
            print(f"Fetching for {symbol}...")
            url = (f'https://www.alphavantage.co/query?function=INSIDER_TRADING&symbol={symbol}'
                   f'&apikey={api_key}')
            r = requests.get(url)
            r.raise_for_status()
            csv_data = r.text
            
            if "Our standard API call frequency is" in csv_data:
                print(f"API limit reached for Alpha Vantage. Skipping {symbol}.")
                continue

            if not csv_data:
                print(f"No insider data for {symbol}.")
                continue

            df = pd.read_csv(StringIO(csv_data))
            if not df.empty:
                all_transactions.append(df)
        
        if not all_transactions:
            print("No transactions found from Alpha Vantage.")
            return _generate_sample_insider_data(symbols)
        
        full_df = pd.concat(all_transactions, ignore_index=True)
        full_df.rename(columns={'transactionDate': 'Date'}, inplace=True)
        full_df['Date'] = pd.to_datetime(full_df['Date'])
        
        print(f"✅ Successfully fetched {len(full_df)} total insider transactions.")
        return full_df
        
    except (ValueError, ImportError):
        return _generate_sample_insider_data(symbols)
    except Exception as e:
        print(f"An unexpected error occurred with Alpha Vantage: {e}")
        return _generate_sample_insider_data(symbols)

def save_raw_data(df: pd.DataFrame, raw_data_path: str):
    """Saves the raw DataFrame to a Parquet file."""
    if not df.empty:
        output_path = os.path.join(raw_data_path, "insider_transactions_data.parquet")
        df.to_parquet(output_path)
        print(f"💾 Saved raw insider transaction data to {output_path}")

if __name__ == '__main__':
    config = load_config()
    
    symbols_to_test = config['stocks']
    raw_path = config['storage']['raw_path']
    
    os.makedirs(raw_path, exist_ok=True)
    
    df = fetch_insider_transactions(symbols_to_test)
    
    if not df.empty:
        save_raw_data(df, raw_path)
        print("\n--- Sample Data ---")
        print(df.head())
        print("\n--- Data Info ---")
        df.info() 