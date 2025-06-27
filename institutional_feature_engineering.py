import pandas as pd
import numpy as np
import os

def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates standard technical indicators.
    """
    # Simple Moving Averages
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()

    # Exponential Moving Average
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()

    # Bollinger Bands
    df['BB_std'] = df['Close'].rolling(window=20).std()
    df['BB_upper'] = df['SMA_20'] + (df['BB_std'] * 2)
    df['BB_lower'] = df['SMA_20'] - (df['BB_std'] * 2)

    # Relative Strength Index (RSI)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # MACD
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    return df

def calculate_contextual_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates features based on macro, news, and insider data.
    """
    # These are the columns created by our ETL pipeline
    macro_cols = ['DGS10', 'T10Y2Y', 'VIXCLS', 'DFF']
    context_cols = ['news_volume', 'insider_trades']
    
    # Create rolling averages for macro indicators
    for col in macro_cols:
        if col in df.columns:
            df[f'{col}_SMA_10'] = df[col].rolling(window=10).mean()
            df[f'{col}_SMA_30'] = df[col].rolling(window=30).mean()

    # Create rolling sums for news and insider activity
    for col in context_cols:
        if col in df.columns:
            df[f'{col}_rolling_7'] = df[col].rolling(window=7).sum()
            df[f'{col}_rolling_30'] = df[col].rolling(window=30).sum()
            
    return df

def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Main function to generate all features.
    """
    # 1. Calculate basic technical indicators
    df = calculate_technical_indicators(df)
    
    # 2. Calculate features from contextual data
    df = calculate_contextual_features(df)
    
    # Drop rows with NaN values created by rolling windows
    df.dropna(inplace=True)
    
    return df


if __name__ == '__main__':
    # Configuration
    PROCESSED_DATA_PATH = 'data/processed_data'
    FEATURE_DATA_PATH = 'data/featured_data'
    SYMBOL_TO_TEST = 'AAPL' # Example symbol
    
    # Create output directory
    os.makedirs(FEATURE_DATA_PATH, exist_ok=True)
    
    # Load processed data
    input_file = os.path.join(PROCESSED_DATA_PATH, f"{SYMBOL_TO_TEST}_processed_data.parquet")
    if not os.path.exists(input_file):
        print(f"❌ Error: Processed data for {SYMBOL_TO_TEST} not found.")
        print("Please run the ETL pipeline first (e.g., `python3 dry_run_etl.py`).")
    else:
        print(f"✅ Loading processed data from {input_file}...")
        stock_df = pd.read_parquet(input_file)
        
        print("🛠️  Generating features...")
        features_df = generate_features(stock_df.copy())
        
        # Save the result
        output_file = os.path.join(FEATURE_DATA_PATH, f"{SYMBOL_TO_TEST}_featured_data.parquet")
        features_df.to_parquet(output_file)
        
        print(f"✅ Successfully generated and saved featured data to {output_file}")
        print("\n--- Feature DataFrame Info ---")
        features_df.info()
        print("\n--- Sample of Featured Data ---")
        print(features_df.tail())