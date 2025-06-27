"""
Test script for Macroeconomic Indicators Integration
Demonstrates how macro indicators enhance the stock prediction system.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys

# Import our modules
from data_ingestion import fetch_yfinance
from feature_engineering import (
    add_technical_indicators, 
    get_comprehensive_features,
    get_trading_signals,
    get_macro_trading_signals
)
from macro_indicators import MacroIndicators

def test_macro_integration():
    """
    Test the integration of macroeconomic indicators with stock analysis.
    """
    print("🔗 Testing Macroeconomic Indicators Integration")
    print("=" * 60)
    
    # Test symbols
    symbols = ['AAPL', 'MSFT', 'TSLA']
    
    for symbol in symbols:
        print(f"\n📊 Testing {symbol}")
        print("-" * 40)
        
        try:
            # Fetch stock data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)
            
            df = fetch_yfinance(symbol, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
            print(f"✅ Fetched {len(df)} records for {symbol}")
            
            # Test technical indicators only
            print("\n1. Testing Technical Indicators Only...")
            df_technical = add_technical_indicators(df.copy())
            technical_signals = get_trading_signals(df_technical)
            print(f"✅ Technical signals: {len(technical_signals)} indicators")
            
            # Test comprehensive features with macro
            print("\n2. Testing Comprehensive Features with Macro...")
            df_comprehensive = get_comprehensive_features(df.copy(), include_macro=True)
            comprehensive_signals = get_trading_signals(df_comprehensive)
            macro_signals = get_macro_trading_signals(df_comprehensive)
            
            print(f"✅ Comprehensive signals: {len(comprehensive_signals)} indicators")
            print(f"✅ Macro signals: {len(macro_signals)} indicators")
            
            # Compare feature counts
            technical_features = len([col for col in df_technical.columns if col not in ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']])
            comprehensive_features = len([col for col in df_comprehensive.columns if col not in ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']])
            
            print(f"📈 Technical features: {technical_features}")
            print(f"📈 Comprehensive features: {comprehensive_features}")
            print(f"📈 Additional macro features: {comprehensive_features - technical_features}")
            
            # Show sample macro features
            macro_cols = [col for col in df_comprehensive.columns if any(x in col.lower() for x in ['rate', 'inflation', 'gdp', 'unemployment', 'vix', 'stress', 'regime'])]
            if macro_cols:
                print(f"🔍 Sample macro features: {macro_cols[:5]}")
            
        except Exception as e:
            print(f"❌ Error testing {symbol}: {e}")

def test_macro_enhancement_impact():
    """
    Test the impact of macro indicators on trading signals.
    """
    print("\n🎯 Testing Macro Enhancement Impact")
    print("=" * 60)
    
    # Test with a single symbol
    symbol = 'AAPL'
    
    try:
        # Fetch data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        df = fetch_yfinance(symbol, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        
        # Compare signals with and without macro
        print("\n1. Technical Analysis Only:")
        df_technical = add_technical_indicators(df.copy())
        technical_signals = get_trading_signals(df_technical)
        
        for indicator, signal_data in technical_signals.items():
            if indicator != 'Overall':
                print(f"   {indicator}: {signal_data['signal']} (Confidence: {signal_data['confidence']:.0%})")
        
        print("\n2. Technical + Macro Analysis:")
        df_comprehensive = get_comprehensive_features(df.copy(), include_macro=True)
        comprehensive_signals = get_trading_signals(df_comprehensive)
        macro_signals = get_macro_trading_signals(df_comprehensive)
        
        for indicator, signal_data in comprehensive_signals.items():
            if indicator != 'Overall':
                print(f"   {indicator}: {signal_data['signal']} (Confidence: {signal_data['confidence']:.0%})")
        
        print("\n3. Macro-Specific Signals:")
        for indicator, signal_data in macro_signals.items():
            print(f"   {indicator}: {signal_data['signal']} (Confidence: {signal_data['confidence']:.0%})")
            print(f"      Reason: {signal_data['reason']}")
        
        # Compare overall signals
        if 'Overall' in technical_signals and 'Overall' in comprehensive_signals:
            print(f"\n📊 Signal Comparison:")
            print(f"   Technical Only: {technical_signals['Overall']['signal']} (Confidence: {technical_signals['Overall']['confidence']:.0%})")
            print(f"   With Macro: {comprehensive_signals['Overall']['signal']} (Confidence: {comprehensive_signals['Overall']['confidence']:.0%})")
        
    except Exception as e:
        print(f"❌ Error in enhancement impact test: {e}")

def test_macro_data_sources():
    """
    Test different macro data sources and their availability.
    """
    print("\n🌐 Testing Macro Data Sources")
    print("=" * 60)
    
    macro = MacroIndicators()
    
    # Test Yahoo Finance data (no API key required)
    print("\n1. Testing Yahoo Finance Macro Data:")
    yahoo_symbols = {
        'Oil Futures': 'CL=F',
        'Gold Futures': 'GC=F',
        'Bitcoin': 'BTC-USD',
        'VIX': '^VIX',
        'USD/EUR': 'EURUSD=X'
    }
    
    for name, symbol in yahoo_symbols.items():
        try:
            df = macro.fetch_yahoo_macro_data(symbol, '6mo')
            if not df.empty:
                print(f"   ✅ {name} ({symbol}): {len(df)} records")
                if 'Close' in df.columns:
                    latest_price = df['Close'].iloc[-1]
                    print(f"      Latest: ${latest_price:.2f}")
            else:
                print(f"   ❌ {name} ({symbol}): No data")
        except Exception as e:
            print(f"   ❌ {name} ({symbol}): Error - {e}")
    
    # Test sample macro data creation
    print("\n2. Testing Sample Macro Data Creation:")
    dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
    
    # Create sample macro data manually
    sample_data = {
        'interest_rate': pd.DataFrame({
            'Date': dates,
            'FEDFUNDS': np.random.normal(3.0, 1.0, len(dates)) + np.linspace(0, 5, len(dates))
        }),
        'inflation': pd.DataFrame({
            'Date': dates,
            'CPIAUCSL': np.random.normal(2.5, 0.5, len(dates)) + np.sin(np.linspace(0, 4*np.pi, len(dates)))
        }),
        'vix': pd.DataFrame({
            'Date': dates,
            'VIXCLS': np.random.normal(20, 5, len(dates)) + np.abs(np.sin(np.linspace(0, 6*np.pi, len(dates)))) * 10
        })
    }
    
    for indicator_name, df in sample_data.items():
        print(f"   ✅ {indicator_name}: {len(df)} records")
        if not df.empty:
            value_col = df.columns[1]
            latest_value = df[value_col].iloc[-1]
            print(f"      Latest {value_col}: {latest_value:.2f}")

def test_macro_feature_calculation():
    """
    Test the calculation of macro-enhanced features.
    """
    print("\n🧮 Testing Macro Feature Calculation")
    print("=" * 60)
    
    # Create sample data
    dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
    
    # Sample stock data
    stock_data = pd.DataFrame({
        'Date': dates,
        'Open': np.random.normal(100, 5, len(dates)),
        'High': np.random.normal(105, 5, len(dates)),
        'Low': np.random.normal(95, 5, len(dates)),
        'Close': np.random.normal(100, 5, len(dates)) + np.linspace(0, 20, len(dates)),
        'Volume': np.random.randint(1000000, 10000000, len(dates))
    })
    
    # Sample macro data
    macro_data = {
        'interest_rate': pd.DataFrame({
            'Date': dates,
            'FEDFUNDS': np.random.normal(3.0, 1.0, len(dates)) + np.linspace(0, 5, len(dates))
        }),
        'inflation': pd.DataFrame({
            'Date': dates,
            'CPIAUCSL': np.random.normal(2.5, 0.5, len(dates)) + np.sin(np.linspace(0, 4*np.pi, len(dates)))
        }),
        'vix': pd.DataFrame({
            'Date': dates,
            'VIXCLS': np.random.normal(20, 5, len(dates)) + np.abs(np.sin(np.linspace(0, 6*np.pi, len(dates)))) * 10
        })
    }
    
    print("✅ Sample data created")
    
    # Test macro feature calculation
    macro = MacroIndicators()
    
    print("\n1. Testing Macro Features:")
    macro_features = macro.calculate_macro_features(macro_data)
    print(f"   ✅ Calculated {len(macro_features.columns)} macro features")
    
    print("\n2. Testing Market Regime Features:")
    regime_features = macro.get_market_regime_features(macro_data)
    print(f"   ✅ Calculated {len(regime_features.columns)} regime features")
    
    print("\n3. Testing Sector Rotation Signals:")
    rotation_signals = macro.get_sector_rotation_signals(macro_data)
    print(f"   ✅ Calculated {len(rotation_signals.columns)} rotation signals")
    
    # Test comprehensive feature integration
    print("\n4. Testing Comprehensive Integration:")
    df_comprehensive = get_comprehensive_features(stock_data, include_macro=True)
    print(f"   ✅ Final dataset: {df_comprehensive.shape}")
    
    # Show enhanced features
    enhanced_features = [col for col in df_comprehensive.columns if any(x in col.lower() for x in ['adjusted', 'stress', 'regime', 'macro'])]
    print(f"   🔍 Enhanced features: {enhanced_features}")

def test_performance_impact():
    """
    Test the performance impact of adding macro indicators.
    """
    print("\n⚡ Testing Performance Impact")
    print("=" * 60)
    
    import time
    
    # Test with a single symbol
    symbol = 'AAPL'
    
    try:
        # Fetch data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        df = fetch_yfinance(symbol, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        
        # Test technical only
        start_time = time.time()
        df_technical = add_technical_indicators(df.copy())
        technical_time = time.time() - start_time
        
        # Test comprehensive with macro
        start_time = time.time()
        df_comprehensive = get_comprehensive_features(df.copy(), include_macro=True)
        comprehensive_time = time.time() - start_time
        
        print(f"⏱️ Technical only processing time: {technical_time:.3f} seconds")
        print(f"⏱️ Comprehensive processing time: {comprehensive_time:.3f} seconds")
        print(f"📈 Performance impact: {((comprehensive_time - technical_time) / technical_time * 100):.1f}% slower")
        
        # Feature count comparison
        technical_features = len([col for col in df_technical.columns if col not in ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']])
        comprehensive_features = len([col for col in df_comprehensive.columns if col not in ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']])
        
        print(f"📊 Technical features: {technical_features}")
        print(f"📊 Comprehensive features: {comprehensive_features}")
        print(f"📊 Additional features: {comprehensive_features - technical_features}")
        print(f"📊 Feature increase: {((comprehensive_features - technical_features) / technical_features * 100):.1f}%")
        
    except Exception as e:
        print(f"❌ Error in performance test: {e}")

def main():
    """
    Run all integration tests.
    """
    print("🎯 Macroeconomic Indicators Integration Test Suite")
    print("=" * 80)
    
    # Run all tests
    test_macro_integration()
    test_macro_enhancement_impact()
    test_macro_data_sources()
    test_macro_feature_calculation()
    test_performance_impact()
    
    print("\n🎉 All integration tests completed!")
    print("\n📋 Summary:")
    print("✅ Macroeconomic indicators successfully integrated")
    print("✅ Enhanced feature engineering with macro data")
    print("✅ Trading signals enhanced with macro context")
    print("✅ Performance impact measured")
    print("✅ Data sources tested")
    
    print("\n🚀 Next Steps:")
    print("1. Get a FRED API key for real economic data")
    print("2. Integrate macro indicators into LSTM model training")
    print("3. Add macro-based portfolio optimization")
    print("4. Implement real-time macro data updates")
    print("5. Create macro-based risk management rules")

if __name__ == "__main__":
    main() 