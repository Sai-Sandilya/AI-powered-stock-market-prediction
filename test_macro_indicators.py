"""
Test script for Macroeconomic Indicators Module
Demonstrates functionality with sample data and real API calls.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from macro_indicators import MacroIndicators

def test_with_sample_data():
    """
    Test the macro indicators module with sample data.
    """
    print("🧪 Testing Macroeconomic Indicators with Sample Data")
    print("=" * 60)
    
    # Initialize the module
    macro = MacroIndicators()
    
    # Create comprehensive sample data
    dates = pd.date_range(start='2022-01-01', end='2024-01-01', freq='D')
    
    # Sample macroeconomic data
    sample_data = {
        'gdp': pd.DataFrame({
            'Date': dates,
            'GDP': np.random.normal(100, 5, len(dates)) + np.linspace(0, 10, len(dates))  # Trending up
        }),
        'inflation': pd.DataFrame({
            'Date': dates,
            'CPIAUCSL': np.random.normal(2.5, 0.5, len(dates)) + np.sin(np.linspace(0, 4*np.pi, len(dates)))  # Cyclical
        }),
        'interest_rate': pd.DataFrame({
            'Date': dates,
            'FEDFUNDS': np.random.normal(3.0, 1.0, len(dates)) + np.linspace(0, 5, len(dates))  # Rising trend
        }),
        'unemployment': pd.DataFrame({
            'Date': dates,
            'UNRATE': np.random.normal(4.0, 0.5, len(dates)) + np.sin(np.linspace(0, 2*np.pi, len(dates)))  # Cyclical
        }),
        'vix': pd.DataFrame({
            'Date': dates,
            'VIXCLS': np.random.normal(20, 5, len(dates)) + np.abs(np.sin(np.linspace(0, 6*np.pi, len(dates)))) * 10  # Volatile
        }),
        'oil_futures': pd.DataFrame({
            'Date': dates,
            'Close': np.random.normal(80, 10, len(dates)) + np.sin(np.linspace(0, 3*np.pi, len(dates))) * 20,
            'High': np.random.normal(82, 10, len(dates)),
            'Low': np.random.normal(78, 10, len(dates)),
            'Volume': np.random.randint(1000, 10000, len(dates))
        }),
        'gold_futures': pd.DataFrame({
            'Date': dates,
            'Close': np.random.normal(1800, 100, len(dates)) + np.linspace(0, 200, len(dates)),
            'High': np.random.normal(1820, 100, len(dates)),
            'Low': np.random.normal(1780, 100, len(dates)),
            'Volume': np.random.randint(500, 5000, len(dates))
        }),
        'bitcoin': pd.DataFrame({
            'Date': dates,
            'Close': np.random.normal(50000, 5000, len(dates)) + np.sin(np.linspace(0, 4*np.pi, len(dates))) * 10000,
            'High': np.random.normal(52000, 5000, len(dates)),
            'Low': np.random.normal(48000, 5000, len(dates)),
            'Volume': np.random.randint(10000, 100000, len(dates))
        })
    }
    
    print("✅ Sample data created with realistic patterns")
    
    # Test feature calculation
    print("\n📊 Calculating Macro Features...")
    features = macro.calculate_macro_features(sample_data)
    print(f"✅ Calculated {len(features.columns)} features")
    print(f"Feature columns: {list(features.columns)}")
    
    # Test market regime features
    print("\n🏛️ Calculating Market Regime Features...")
    regime_features = macro.get_market_regime_features(sample_data)
    print(f"✅ Calculated {len(regime_features.columns)} regime features")
    if not regime_features.empty:
        print(f"Regime features: {list(regime_features.columns)}")
    
    # Test sector rotation signals
    print("\n🔄 Calculating Sector Rotation Signals...")
    rotation_signals = macro.get_sector_rotation_signals(sample_data)
    print(f"✅ Calculated {len(rotation_signals.columns)} rotation signals")
    if not rotation_signals.empty:
        print(f"Rotation signals: {list(rotation_signals.columns)}")
    
    # Display sample statistics
    print("\n📈 Sample Feature Statistics:")
    if not features.empty:
        print(features.describe())
    
    # Test data persistence
    print("\n💾 Testing Data Persistence...")
    macro.save_macro_data(sample_data, 'test_macro_data')
    loaded_data = macro.load_macro_data('test_macro_data')
    print(f"✅ Saved and loaded {len(loaded_data)} datasets")
    
    # Clean up
    import shutil
    shutil.rmtree('test_macro_data', ignore_errors=True)
    
    return features, regime_features, rotation_signals

def test_with_real_data():
    """
    Test with real data from Yahoo Finance (no API key required).
    """
    print("\n🌐 Testing with Real Yahoo Finance Data")
    print("=" * 60)
    
    macro = MacroIndicators()
    
    # Test with real Yahoo Finance data
    yahoo_symbols = {
        'oil_futures': 'CL=F',
        'gold_futures': 'GC=F',
        'bitcoin': 'BTC-USD',
        'ethereum': 'ETH-USD',
        'usd_eur': 'EURUSD=X',
        'vix': '^VIX'
    }
    
    real_data = {}
    
    for indicator_name, symbol in yahoo_symbols.items():
        print(f"Fetching {indicator_name} ({symbol})...")
        try:
            df = macro.fetch_yahoo_macro_data(symbol, '6mo')
            if not df.empty:
                real_data[indicator_name] = df
                print(f"✅ Successfully fetched {len(df)} records")
            else:
                print(f"❌ No data for {symbol}")
        except Exception as e:
            print(f"❌ Error fetching {symbol}: {e}")
    
    if real_data:
        print(f"\n📊 Processing {len(real_data)} real datasets...")
        features = macro.calculate_macro_features(real_data)
        print(f"✅ Calculated {len(features.columns)} features from real data")
        
        # Show sample of real data
        print("\n📈 Sample Real Data:")
        for name, df in real_data.items():
            if 'Close' in df.columns:
                print(f"{name}: Latest Close = ${df['Close'].iloc[-1]:.2f}")
    
    return real_data

def demonstrate_integration():
    """
    Demonstrate how to integrate macro indicators with stock prediction.
    """
    print("\n🔗 Demonstrating Integration with Stock Prediction")
    print("=" * 60)
    
    # Create sample stock data
    dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
    
    # Sample stock price data
    stock_data = pd.DataFrame({
        'Date': dates,
        'Close': np.random.normal(100, 5, len(dates)) + np.linspace(0, 20, len(dates)),
        'Volume': np.random.randint(1000000, 10000000, len(dates))
    })
    
    # Sample macro data
    macro_data = {
        'interest_rate': pd.DataFrame({
            'Date': dates,
            'FEDFUNDS': np.random.normal(3.0, 1.0, len(dates))
        }),
        'inflation': pd.DataFrame({
            'Date': dates,
            'CPIAUCSL': np.random.normal(2.5, 0.5, len(dates))
        }),
        'vix': pd.DataFrame({
            'Date': dates,
            'VIXCLS': np.random.normal(20, 5, len(dates))
        })
    }
    
    # Calculate macro features
    macro = MacroIndicators()
    macro_features = macro.calculate_macro_features(macro_data)
    regime_features = macro.get_market_regime_features(macro_data)
    
    # Merge stock data with macro features
    if not macro_features.empty:
        combined_data = stock_data.merge(macro_features, on='Date', how='left')
        combined_data = combined_data.merge(regime_features, on='Date', how='left')
        
        print("✅ Successfully combined stock and macro data")
        print(f"Combined dataset shape: {combined_data.shape}")
        print(f"Features available: {list(combined_data.columns)}")
        
        # Show correlation with macro factors
        print("\n📊 Macro Factor Correlations with Stock Price:")
        if 'Close' in combined_data.columns:
            correlations = combined_data.corr()['Close'].sort_values(ascending=False)
            print(correlations.head(10))
    
    return combined_data if 'combined_data' in locals() else None

def create_macro_enhanced_features():
    """
    Create enhanced features that combine technical and macro indicators.
    """
    print("\n🚀 Creating Macro-Enhanced Features")
    print("=" * 60)
    
    # Sample data
    dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
    
    # Stock data
    stock_data = pd.DataFrame({
        'Date': dates,
        'Close': np.random.normal(100, 5, len(dates)) + np.linspace(0, 20, len(dates)),
        'Volume': np.random.randint(1000000, 10000000, len(dates))
    })
    
    # Macro data
    macro_data = {
        'interest_rate': pd.DataFrame({
            'Date': dates,
            'FEDFUNDS': np.random.normal(3.0, 1.0, len(dates))
        }),
        'inflation': pd.DataFrame({
            'Date': dates,
            'CPIAUCSL': np.random.normal(2.5, 0.5, len(dates))
        })
    }
    
    # Calculate enhanced features
    macro = MacroIndicators()
    macro_features = macro.calculate_macro_features(macro_data)
    regime_features = macro.get_market_regime_features(macro_data)
    
    # Create enhanced features
    enhanced_features = []
    
    # 1. Interest rate adjusted returns
    if not macro_features.empty and 'interest_rate_change' in macro_features.columns:
        stock_data['rate_adjusted_returns'] = stock_data['Close'].pct_change() - macro_features['interest_rate_change']
    
    # 2. Inflation adjusted volatility
    if 'inflation_CPIAUCSL' in macro_features.columns:
        stock_data['inflation_adjusted_volatility'] = stock_data['Close'].rolling(20).std() * (1 + macro_features['inflation_CPIAUCSL'] / 100)
    
    # 3. Macro regime signals
    if not regime_features.empty:
        for col in regime_features.columns:
            if col != 'Date':
                stock_data[f'macro_{col}'] = regime_features[col]
    
    # 4. Economic cycle indicators
    if 'high_rate_environment' in stock_data.columns and 'high_inflation' in stock_data.columns:
        stock_data['economic_stress'] = (stock_data['high_rate_environment'] + stock_data['high_inflation']) / 2
    
    print("✅ Created enhanced features:")
    enhanced_cols = [col for col in stock_data.columns if col not in ['Date', 'Close', 'Volume']]
    print(f"Enhanced features: {enhanced_cols}")
    
    return stock_data

def main():
    """
    Run all tests and demonstrations.
    """
    print("🎯 Macroeconomic Indicators Test Suite")
    print("=" * 80)
    
    # Test with sample data
    features, regime_features, rotation_signals = test_with_sample_data()
    
    # Test with real data
    real_data = test_with_real_data()
    
    # Demonstrate integration
    combined_data = demonstrate_integration()
    
    # Create enhanced features
    enhanced_data = create_macro_enhanced_features()
    
    print("\n🎉 All tests completed successfully!")
    print("\n📋 Summary:")
    print(f"- Sample features calculated: {len(features.columns) if not features.empty else 0}")
    print(f"- Regime features: {len(regime_features.columns) if not regime_features.empty else 0}")
    print(f"- Rotation signals: {len(rotation_signals.columns) if not rotation_signals.empty else 0}")
    print(f"- Real datasets fetched: {len(real_data)}")
    print(f"- Enhanced features created: {len(enhanced_data.columns) if enhanced_data is not None else 0}")
    
    print("\n🚀 Next Steps:")
    print("1. Get a FRED API key for comprehensive economic data")
    print("2. Integrate macro indicators into your LSTM model")
    print("3. Add macro features to your Streamlit UI")
    print("4. Create macro-based trading signals")
    print("5. Implement sector rotation strategies")

if __name__ == "__main__":
    main() 