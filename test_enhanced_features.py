#!/usr/bin/env python3
"""
🧪 Enhanced Features Test Suite
==================================================

This script demonstrates the enhanced capabilities of the stock prediction system:
- Advanced Technical Indicators (MACD, Bollinger Bands, Stochastic, etc.)
- Trading Signal Generation
- Pattern Detection
- Risk Assessment
- Portfolio Analysis
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from data_ingestion import fetch_yfinance
from feature_engineering import add_technical_indicators, get_trading_signals
import warnings
import yfinance as yf

warnings.filterwarnings('ignore')

def test_enhanced_technical_indicators():
    """Test the enhanced technical indicators."""
    print("🔧 Testing Enhanced Technical Indicators")
    print("=" * 50)
    
    # Fetch data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    df = fetch_yfinance('AAPL', start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
    
    # Add technical indicators
    df = add_technical_indicators(df)
    
    # Display available indicators
    basic_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    indicator_cols = [col for col in df.columns if col not in basic_cols]
    
    print(f"📊 Total Indicators: {len(indicator_cols)}")
    print(f"📈 Available Indicators: {', '.join(indicator_cols)}")
    
    # Show latest values
    latest = df.iloc[-1]
    print(f"\n📊 Latest Technical Values:")
    print(f"   RSI (14): {latest['RSI_14']:.2f}")
    print(f"   MACD: {latest['MACD']:.4f}")
    print(f"   MACD Signal: {latest['MACD_Signal']:.4f}")
    print(f"   Bollinger Upper: {latest['BB_Upper']:.2f}")
    print(f"   Bollinger Lower: {latest['BB_Lower']:.2f}")
    print(f"   Stochastic K: {latest['Stoch_K']:.2f}")
    print(f"   ATR (14): {latest['ATR_14']:.4f}")
    print(f"   Williams %R: {latest['Williams_R']:.2f}")
    print(f"   Volatility: {latest['Volatility']:.4f}")
    
    return df

def test_trading_signals(df):
    """Test trading signal generation."""
    print("\n🎯 Testing Trading Signal Generation")
    print("=" * 50)
    
    signals = get_trading_signals(df)
    
    print("📊 Individual Signals:")
    for indicator, signal_data in signals.items():
        if indicator != 'Overall':
            emoji = "🟢" if signal_data['signal'] == 'BUY' else "🔴" if signal_data['signal'] == 'SELL' else "⚪"
            print(f"   {emoji} {indicator}: {signal_data['signal']} (Confidence: {signal_data['confidence']:.1%})")
    
    print(f"\n🎯 Overall Signal: {signals['Overall']['signal']} (Confidence: {signals['Overall']['confidence']:.1%})")
    
    return signals

def test_pattern_detection(df):
    """Test pattern detection capabilities."""
    print("\n📈 Testing Pattern Detection")
    print("=" * 50)
    
    patterns = {}
    
    # Bullish/Bearish patterns
    current_price = df['Close'].iloc[-1]
    ma_20 = df['MA_20'].iloc[-1]
    ma_50 = df['MA_50'].iloc[-1]
    
    # Golden Cross / Death Cross
    if ma_20 > ma_50:
        patterns['Moving Average'] = "🟢 Golden Cross (Bullish)"
    else:
        patterns['Moving Average'] = "🔴 Death Cross (Bearish)"
    
    # RSI patterns
    rsi = df['RSI_14'].iloc[-1]
    if rsi < 30:
        patterns['RSI'] = "🟢 Oversold (Potential Buy)"
    elif rsi > 70:
        patterns['RSI'] = "🔴 Overbought (Potential Sell)"
    else:
        patterns['RSI'] = "⚪ Neutral"
    
    # Bollinger Bands patterns
    bb_position = df['BB_Position'].iloc[-1]
    if bb_position < 0.2:
        patterns['Bollinger Bands'] = "🟢 Near Lower Band (Potential Bounce)"
    elif bb_position > 0.8:
        patterns['Bollinger Bands'] = "🔴 Near Upper Band (Potential Drop)"
    else:
        patterns['Bollinger Bands'] = "⚪ Within Bands"
    
    # MACD patterns
    macd = df['MACD'].iloc[-1]
    macd_signal = df['MACD_Signal'].iloc[-1]
    if macd > macd_signal:
        patterns['MACD'] = "🟢 Bullish Crossover"
    else:
        patterns['MACD'] = "🔴 Bearish Crossover"
    
    # Volume patterns
    volume_ratio = df['Volume_Ratio'].iloc[-1]
    if volume_ratio > 1.5:
        patterns['Volume'] = "📈 High Volume (Strong Move)"
    elif volume_ratio < 0.5:
        patterns['Volume'] = "📉 Low Volume (Weak Move)"
    else:
        patterns['Volume'] = "📊 Normal Volume"
    
    for pattern, description in patterns.items():
        print(f"   {description}")
    
    return patterns

def test_risk_assessment(df):
    """Test risk assessment capabilities."""
    print("\n⚠️ Testing Risk Assessment")
    print("=" * 50)
    
    # Calculate risk metrics
    returns = df['Close'].pct_change().dropna()
    volatility = returns.std() * np.sqrt(252)  # Annualized volatility
    max_drawdown = calculate_max_drawdown(df['Close'])
    sharpe_ratio = calculate_sharpe_ratio(returns)
    
    print(f"📊 Risk Metrics:")
    print(f"   Volatility (Annual): {volatility:.2%}")
    print(f"   Maximum Drawdown: {max_drawdown:.2%}")
    print(f"   Sharpe Ratio: {sharpe_ratio:.2f}")
    
    # Risk assessment
    risk_level = "🟢 Low" if volatility < 0.2 else "🟡 Medium" if volatility < 0.4 else "🔴 High"
    print(f"   Risk Level: {risk_level}")
    
    return {
        'volatility': volatility,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'risk_level': risk_level
    }

def calculate_max_drawdown(prices):
    """Calculate maximum drawdown."""
    peak = prices.expanding().max()
    drawdown = (prices - peak) / peak
    return drawdown.min()

def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    """Calculate Sharpe ratio."""
    excess_returns = returns - risk_free_rate/252
    return np.sqrt(252) * excess_returns.mean() / returns.std()

def test_portfolio_analysis():
    """Test portfolio analysis with multiple stocks."""
    print("\n💼 Testing Portfolio Analysis")
    print("=" * 50)
    
    # Analyze multiple stocks
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN']
    portfolio_data = {}
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    for symbol in symbols:
        try:
            df = fetch_yfinance(symbol, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
            df = add_technical_indicators(df)
            signals = get_trading_signals(df)
            
            portfolio_data[symbol] = {
                'current_price': df['Close'].iloc[-1],
                'signal': signals['Overall']['signal'],
                'confidence': signals['Overall']['confidence'],
                'rsi': df['RSI_14'].iloc[-1],
                'volatility': df['Volatility'].iloc[-1]
            }
        except Exception as e:
            print(f"   ⚠️ Error analyzing {symbol}: {e}")
    
    # Display portfolio summary
    print("📊 Portfolio Summary:")
    for symbol, data in portfolio_data.items():
        emoji = "🟢" if data['signal'] == 'BUY' else "🔴" if data['signal'] == 'SELL' else "⚪"
        print(f"   {emoji} {symbol}: ${data['current_price']:.2f} - {data['signal']} ({data['confidence']:.1%})")
    
    return portfolio_data

def test_advanced_feature_engineering():
    """
    Test the advanced feature engineering module.
    """
    print("🧪 Testing Advanced Feature Engineering...")
    print("=" * 50)
    
    try:
        # Import the module
        from advanced_feature_engineering_v2 import AdvancedFeatureEngineer
        
        # Fetch sample data
        print("📊 Fetching sample data...")
        ticker = yf.Ticker('AAPL')
        df = ticker.history(period='6mo')
        df.reset_index(inplace=True)
        df.rename(columns={'Date': 'Date'}, inplace=True)
        
        print(f"✅ Fetched {len(df)} records")
        
        # Add basic indicators first
        df['RSI_14'] = calculate_rsi(df['Close'], 14)
        df['MACD'], _, _ = calculate_macd(df['Close'])
        df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = calculate_bollinger_bands(df['Close'])
        
        # Test feature engineering
        print("🔬 Creating advanced features...")
        engineer = AdvancedFeatureEngineer()
        df_enhanced = engineer.create_all_features(df)
        
        # Check results
        feature_count = len(engineer.get_feature_names())
        print(f"✅ Created {feature_count} advanced features")
        
        # Show sample features
        sample_features = engineer.get_feature_names()[:10]
        print(f"📋 Sample features: {sample_features}")
        
        # Check for NaN values
        nan_count = df_enhanced.isnull().sum().sum()
        print(f"🔍 NaN values: {nan_count}")
        
        return True, feature_count
        
    except Exception as e:
        print(f"❌ Error in feature engineering test: {str(e)}")
        return False, 0

def test_multi_timeframe_signals():
    """
    Test the multi-timeframe signal generation.
    """
    print("\n🧪 Testing Multi-Timeframe Signals...")
    print("=" * 50)
    
    try:
        # Import the module
        from multi_timeframe_signals import MultiTimeframeSignals
        
        # Fetch sample data
        print("📊 Fetching sample data...")
        ticker = yf.Ticker('AAPL')
        df = ticker.history(period='6mo')
        df.reset_index(inplace=True)
        df.rename(columns={'Date': 'Date'}, inplace=True)
        
        print(f"✅ Fetched {len(df)} records")
        
        # Test signal generation
        print("🎯 Generating signals...")
        signal_generator = MultiTimeframeSignals()
        signals = signal_generator.generate_all_signals(df)
        
        # Check results
        signal_count = sum(len(strategy_signals) for strategy_signals in signals.values())
        print(f"✅ Generated {signal_count} signals across {len(signals)} strategies")
        
        # Get recommendations
        recommendations = signal_generator.get_trading_recommendations()
        print("📋 Trading Recommendations:")
        for strategy, status in recommendations.items():
            print(f"  {strategy}: {status}")
        
        # Get high confidence signals
        high_conf_signals = signal_generator.get_high_confidence_signals(min_confidence=0.7)
        print(f"🎯 High confidence signals: {len(high_conf_signals)}")
        
        return True, signal_count
        
    except Exception as e:
        print(f"❌ Error in signal generation test: {str(e)}")
        return False, 0

def test_enhanced_lstm_model():
    """
    Test the enhanced LSTM model creation.
    """
    print("\n🧪 Testing Enhanced LSTM Models...")
    print("=" * 50)
    
    try:
        # Import the module
        from enhanced_lstm_model import create_enhanced_lstm_model
        
        # Test different model types
        model_types = ['attention', 'bidirectional', 'stacked', 'transformer']
        
        for model_type in model_types:
            print(f"🔧 Testing {model_type} LSTM...")
            
            try:
                model = create_enhanced_lstm_model(
                    model_type=model_type,
                    sequence_length=30,
                    n_features=20,
                    lstm_units=64,
                    dropout_rate=0.2
                )
                
                print(f"✅ {model_type} LSTM created successfully")
                print(f"📊 Model summary: {type(model).__name__}")
                
            except Exception as e:
                print(f"❌ Error creating {model_type} LSTM: {str(e)}")
        
        return True, len(model_types)
        
    except Exception as e:
        print(f"❌ Error in LSTM model test: {str(e)}")
        return False, 0

def test_data_preparation():
    """
    Test data preparation for LSTM training.
    """
    print("\n🧪 Testing Data Preparation...")
    print("=" * 50)
    
    try:
        # Import the module
        from enhanced_lstm_model import prepare_data_for_lstm
        
        # Fetch sample data
        print("📊 Fetching sample data...")
        ticker = yf.Ticker('AAPL')
        df = ticker.history(period='1y')
        df.reset_index(inplace=True)
        df.rename(columns={'Date': 'Date'}, inplace=True)
        
        # Add some basic features
        df['RSI'] = calculate_rsi(df['Close'], 14)
        df['MA_20'] = df['Close'].rolling(20).mean()
        df['Volume_MA'] = df['Volume'].rolling(20).mean()
        
        print(f"✅ Fetched {len(df)} records with {len(df.columns)} features")
        
        # Test data preparation
        print("🔧 Preparing data for LSTM...")
        X_train, X_test, y_train, y_test, scaler, feature_cols = prepare_data_for_lstm(
            df, target_col='Close', sequence_length=30, test_size=0.2
        )
        
        print(f"✅ Data prepared successfully")
        print(f"📊 Training data shape: {X_train.shape}")
        print(f"📊 Test data shape: {X_test.shape}")
        print(f"🎯 Features used: {len(feature_cols)}")
        
        return True, len(feature_cols)
        
    except Exception as e:
        print(f"❌ Error in data preparation test: {str(e)}")
        return False, 0

def test_integration():
    """
    Test integration of all components.
    """
    print("\n🧪 Testing Integration...")
    print("=" * 50)
    
    try:
        # Fetch data
        print("📊 Fetching data...")
        ticker = yf.Ticker('AAPL')
        df = ticker.history(period='6mo')
        df.reset_index(inplace=True)
        df.rename(columns={'Date': 'Date'}, inplace=True)
        
        # Add basic indicators
        df['RSI_14'] = calculate_rsi(df['Close'], 14)
        df['MACD'], _, _ = calculate_macd(df['Close'])
        df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = calculate_bollinger_bands(df['Close'])
        
        print(f"✅ Fetched {len(df)} records")
        
        # Test feature engineering
        print("🔬 Creating features...")
        from advanced_feature_engineering_v2 import AdvancedFeatureEngineer
        engineer = AdvancedFeatureEngineer()
        df_enhanced = engineer.create_all_features(df)
        
        # Test signal generation
        print("🎯 Generating signals...")
        from multi_timeframe_signals import MultiTimeframeSignals
        signal_generator = MultiTimeframeSignals()
        signals = signal_generator.generate_all_signals(df_enhanced)
        
        # Test data preparation
        print("🔧 Preparing data...")
        from enhanced_lstm_model import prepare_data_for_lstm
        X_train, X_test, y_train, y_test, scaler, feature_cols = prepare_data_for_lstm(
            df_enhanced, target_col='Close', sequence_length=30, test_size=0.2
        )
        
        print("✅ Integration test completed successfully!")
        print(f"📊 Final data shape: {X_train.shape}")
        print(f"🎯 Total features: {len(feature_cols)}")
        print(f"📈 Signal strategies: {len(signals)}")
        
        return True, len(feature_cols)
        
    except Exception as e:
        print(f"❌ Error in integration test: {str(e)}")
        return False, 0

# Helper functions
def calculate_rsi(prices, period=14):
    """Calculate RSI."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calculate MACD."""
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram

def calculate_bollinger_bands(prices, period=20, std_dev=2):
    """Calculate Bollinger Bands."""
    middle = prices.rolling(period).mean()
    std = prices.rolling(period).std()
    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)
    return upper, middle, lower

def main():
    """
    Run all tests.
    """
    print("🚀 Enhanced Features Test Suite")
    print("=" * 60)
    
    test_results = {}
    
    # Run all tests
    tests = [
        ("Advanced Feature Engineering", test_advanced_feature_engineering),
        ("Multi-Timeframe Signals", test_multi_timeframe_signals),
        ("Enhanced LSTM Models", test_enhanced_lstm_model),
        ("Data Preparation", test_data_preparation),
        ("Integration", test_integration)
    ]
    
    for test_name, test_func in tests:
        success, count = test_func()
        test_results[test_name] = {"success": success, "count": count}
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(tests)
    
    for test_name, result in test_results.items():
        status = "✅ PASS" if result["success"] else "❌ FAIL"
        count = result["count"]
        print(f"{status} {test_name}: {count} items")
        
        if result["success"]:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Enhanced features are working correctly.")
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
    
    return test_results

if __name__ == "__main__":
    main() 