import os
import pandas as pd
import numpy as np
import sys
from datetime import datetime

try:
    from fredapi import Fred
except ImportError:
    Fred = None

# Add the 'dags' directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config_manager import load_config, get_api_key

def _generate_sample_macro_data(series_ids: list, start_date: str, end_date: str) -> pd.DataFrame:
    """Generates sample macroeconomic data if the FRED API key is not available."""
    print("⚠️ FRED API key not found. Generating sample macroeconomic data.")
    
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    data = {}
    
    # Create some plausible, random-walk data for each series
    np.random.seed(42)
    initial_values = {'DGS10': 1.5, 'T10Y2Y': 0.2, 'VIXCLS': 20, 'DFF': 0.1}
    
    for series_id in series_ids:
        initial = initial_values.get(series_id, 50)
        daily_changes = np.random.randn(len(date_range)) * 0.05
        data[series_id] = initial + np.cumsum(daily_changes)
        
    df = pd.DataFrame(data, index=date_range)
    df.index.name = 'Date'
    df.ffill(inplace=True) # Macro data is often sticky
    return df

def fetch_fred_data(series_ids: list, start_date: str, end_date: str):
    """
    Fetches macroeconomic data from FRED, with a fallback to sample data.
    """
    print(f"Fetching FRED data for {len(series_ids)} series...")
    
    try:
        api_key = get_api_key('fred')
        if not Fred:
            raise ImportError("fredapi is not installed.")
            
        fred = Fred(api_key=api_key)
        all_series = []
        for series_id in series_ids:
            series = fred.get_series(series_id, start_date, end_date)
            all_series.append(series.rename(series_id))
            
        df = pd.concat(all_series, axis=1)
        df.ffill(inplace=True)
        df.index.name = 'Date'
        
        print(f"✅ Successfully fetched {len(df)} records for {len(series_ids)} FRED series.")
        return df
        
    except (ValueError, ImportError):
        return _generate_sample_macro_data(series_ids, start_date, end_date)
    except Exception as e:
        print(f"An unexpected error occurred with FRED: {e}")
        return _generate_sample_macro_data(series_ids, start_date, end_date)

def save_raw_data(df: pd.DataFrame, raw_data_path: str):
    """Saves the raw DataFrame to a Parquet file."""
    if not df.empty:
        output_path = os.path.join(raw_data_path, "macroeconomic_data.parquet")
        df.to_parquet(output_path)
        print(f"💾 Saved raw macroeconomic data to {output_path}")

if __name__ == '__main__':
    # This allows running the script directly for testing
    config = load_config()
    
    series = config['data_fetching']['fred_series']
    start = config['data_fetching']['start_date']
    end = config['data_fetching']['end_date']
    raw_path = config['storage']['raw_path']
    
    # Create the directory if it doesn't exist
    os.makedirs(raw_path, exist_ok=True)
    
    df = fetch_fred_data(series, start, end)
    
    if not df.empty:
        save_raw_data(df, raw_path)
        print("\n--- Sample Data ---")
        print(df.head())
        print("\n--- Data Info ---")
        df.info() 