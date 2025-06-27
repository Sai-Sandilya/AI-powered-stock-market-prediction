import os
import pandas as pd
import sys
from datetime import datetime

# Try to import polygon, but don't fail if it's not there
try:
    from polygon import RESTClient
except ImportError:
    RESTClient = None

# Try to import yfinance as a fallback
try:
    import yfinance as yf
except ImportError:
    yf = None

# Add the 'dags' directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config_manager import load_config, get_api_key

def _fetch_yfinance_fallback(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fallback to yfinance if Polygon API key is not available."""
    print(f"⚠️ Polygon API key not found. Falling back to yfinance for {symbol}.")
    if not yf:
        raise ImportError("yfinance is not installed. Please run 'pip install yfinance'.")
        
    stock = yf.Ticker(symbol)
    df = stock.history(start=start_date, end=end_date)
    
    # yfinance data is already in the desired format, just ensure column names are capitalized
    df.rename(columns={
        'Open': 'Open', 'High': 'High', 'Low': 'Low', 'Close': 'Close', 'Volume': 'Volume'
    }, inplace=True)
    
    return df[['Open', 'High', 'Low', 'Close', 'Volume']]

def fetch_polygon_data(symbol: str, start_date: str, end_date: str):
    """
    Fetches historical stock data from Polygon.io, with a fallback to yfinance.
    """
    print(f"Fetching price data for {symbol} from {start_date} to {end_date}...")
    
    try:
        api_key = get_api_key('polygon')
        if not RESTClient:
            raise ImportError("polygon-api-client is not installed.")
            
        client = RESTClient(api_key)
        resp = client.get_aggs(
            ticker=symbol, multiplier=1, timespan="day", from_=start_date, to=end_date, limit=50000
        )
        
        if not resp:
            print(f"No data returned from Polygon for {symbol}. Trying yfinance.")
            return _fetch_yfinance_fallback(symbol, start_date, end_date)
            
        df = pd.DataFrame(resp)
        df.rename(columns={
            'v': 'Volume', 'o': 'Open', 'c': 'Close', 'h': 'High', 'l': 'Low', 't': 'Timestamp'
        }, inplace=True)
        df['Date'] = pd.to_datetime(df['Timestamp'], unit='ms')
        df.set_index('Date', inplace=True)
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        
        print(f"✅ Successfully fetched {len(df)} records for {symbol} from Polygon.")
        return df
        
    except (ValueError, ImportError):
        # This catches both missing API key and missing library
        return _fetch_yfinance_fallback(symbol, start_date, end_date)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return _fetch_yfinance_fallback(symbol, start_date, end_date)

def save_raw_data(df: pd.DataFrame, symbol: str, raw_data_path: str):
    """Saves the raw DataFrame to a Parquet file."""
    if not df.empty:
        output_path = os.path.join(raw_data_path, f"{symbol}_price_data.parquet")
        df.to_parquet(output_path)
        print(f"💾 Saved raw data for {symbol} to {output_path}")

if __name__ == '__main__':
    config = load_config()
    symbol_to_test = config['stocks'][0]
    start = config['data_fetching']['start_date']
    end = config['data_fetching']['end_date']
    raw_path = config['storage']['raw_path']
    
    os.makedirs(raw_path, exist_ok=True)
    df = fetch_polygon_data(symbol_to_test, start, end)
    
    if not df.empty:
        save_raw_data(df, symbol_to_test, raw_path)
        print("\n--- Sample Data ---")
        print(df.head())
        print("\n--- Data Info ---")
        df.info() 