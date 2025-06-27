"""
Test Enhanced Training System
Simple test script to run the enhanced training with proper error handling.
"""

import pandas as pd
import numpy as np
import warnings
import traceback
from datetime import datetime, timedelta

# Import our modules
from data_ingestion import fetch_yfinance
from feature_engineering import get_comprehensive_features
from train_improved_lstm import ImprovedLSTMTrainer

warnings.filterwarnings('ignore')

def test_data_preparation():
    """
    Test data preparation with comprehensive error handling.
    """
    print("🧪 Testing data preparation...")
    
    try:
        # Fetch basic data
        symbol = 'AAPL'
        start_date = '2020-01-01'
        end_date = datetime.now().strftime('%Y-%m-%d')
        
        print(f"📊 Fetching data for {symbol}...")
        df = fetch_yfinance(symbol, start_date, end_date)
        
        if df.empty:
            raise ValueError("No data fetched")
        
        print(f"✅ Fetched {len(df)} records")
        print(f"📊 Columns: {list(df.columns)}")
        
        # Check for basic data quality
        print(f"📊 Date range: {df['Date'].min()} to {df['Date'].max()}")
        print(f"📊 Price range: ${df['Close'].min():.2f} to ${df['Close'].max():.2f}")
        
        # Add features
        print("🔧 Adding features...")
        featured_df = get_comprehensive_features(df)
        
        print(f"✅ Added features. Total columns: {len(featured_df.columns)}")
        
        # Check for NaN values
        nan_counts = featured_df.isna().sum()
        nan_cols = nan_counts[nan_counts > 0]
        
        if len(nan_cols) > 0:
            print(f"⚠️ Found NaN values in columns: {list(nan_cols.index)}")
            print(f"📊 NaN counts: {dict(nan_cols)}")
        
        # Clean data
        print("🧹 Cleaning data...")
        initial_rows = len(featured_df)
        featured_df = featured_df.dropna()
        print(f"📊 Removed {initial_rows - len(featured_df)} rows with NaN values")
        
        # Check for infinite values
        featured_df = featured_df.replace([np.inf, -np.inf], np.nan)
        featured_df = featured_df.dropna()
        print(f"📊 Removed infinite values")
        
        # Final check
        if len(featured_df) < 100:
            raise ValueError(f"Insufficient data after cleaning: {len(featured_df)} rows")
        
        print(f"✅ Data preparation successful! Final dataset: {len(featured_df)} rows, {len(featured_df.columns)} columns")
        
        return featured_df
        
    except Exception as e:
        print(f"❌ Data preparation failed: {e}")
        traceback.print_exc()
        return None

def test_model_training(df):
    """
    Test model training with the prepared data.
    """
    print("\n🧪 Testing model training...")
    
    try:
        # Initialize trainer
        trainer = ImprovedLSTMTrainer()
        
        # Train model
        print("🚀 Starting model training...")
        results = trainer.train_model(df)
        
        print("✅ Model training completed!")
        print(f"📊 Training results: {results}")
        
        return results
        
    except Exception as e:
        print(f"❌ Model training failed: {e}")
        traceback.print_exc()
        return None

def main():
    """
    Main test function.
    """
    print("🚀 Starting Enhanced Training System Test")
    print("=" * 50)
    
    # Test data preparation
    df = test_data_preparation()
    
    if df is not None:
        # Test model training
        results = test_model_training(df)
        
        if results is not None:
            print("\n🎉 All tests passed! Enhanced training system is working.")
        else:
            print("\n❌ Model training failed.")
    else:
        print("\n❌ Data preparation failed.")

if __name__ == "__main__":
    main() 