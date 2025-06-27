"""
Simplified Backtesting Script
Tests prediction accuracy with a straightforward approach.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
from typing import Dict, List

# Import our modules
from data_ingestion import fetch_yfinance
from feature_engineering import get_comprehensive_features
from future_forecasting import FutureForecaster

warnings.filterwarnings('ignore')

def simple_backtest(symbol: str = 'AAPL', test_days: int = 30):
    """
    Simple backtest that compares predictions with actual data.
    
    Args:
        symbol: Stock symbol to test
        test_days: Number of days to test
    """
    print(f"🧪 Simple Backtest for {symbol}")
    print("=" * 50)
    
    # Get historical data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=test_days + 30)  # Extra data for features
    
    try:
        df = fetch_yfinance(symbol, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        print(f"✅ Fetched {len(df)} historical records")
    except Exception as e:
        print(f"❌ Error fetching data: {e}")
        return
    
    # Add features
    df = get_comprehensive_features(df, include_macro=True)
    
    # Split data: use first part for training, last part for testing
    split_point = len(df) - test_days
    train_data = df.iloc[:split_point].copy()
    test_data = df.iloc[split_point:].copy()
    
    print(f"📊 Training data: {len(train_data)} records")
    print(f"📊 Test data: {len(test_data)} records")
    
    # Initialize forecaster
    forecaster = FutureForecaster()
    
    # Generate predictions for test period
    print(f"\n🔮 Generating predictions for {test_days} days...")
    
    try:
        forecast_df = forecaster.forecast_future(
            symbol=symbol,
            forecast_days=test_days,
            include_macro=True
        )
        
        if not forecast_df.empty:
            print(f"✅ Generated {len(forecast_df)} predictions")
            
            # Compare with actual data
            actual_prices = test_data['Close'].values
            predicted_prices = forecast_df['Predicted_Close'].values[:len(actual_prices)]
            
            if len(predicted_prices) == len(actual_prices):
                # Calculate accuracy metrics
                mae = np.mean(np.abs(actual_prices - predicted_prices))
                mse = np.mean((actual_prices - predicted_prices) ** 2)
                rmse = np.sqrt(mse)
                mape = np.mean(np.abs((actual_prices - predicted_prices) / actual_prices)) * 100
                
                # Directional accuracy
                actual_direction = np.diff(actual_prices) > 0
                predicted_direction = np.diff(predicted_prices) > 0
                directional_accuracy = np.mean(actual_direction == predicted_direction) * 100
                
                # Correlation
                correlation = np.corrcoef(actual_prices, predicted_prices)[0, 1]
                
                # Hit rate (predictions within 5% of actual)
                hit_rate_5 = np.mean(np.abs((actual_prices - predicted_prices) / actual_prices) <= 0.05) * 100
                hit_rate_10 = np.mean(np.abs((actual_prices - predicted_prices) / actual_prices) <= 0.10) * 100
                
                # Print results
                print("\n📈 ACCURACY RESULTS:")
                print("-" * 30)
                print(f"MAE: ${mae:.2f}")
                print(f"RMSE: ${rmse:.2f}")
                print(f"MAPE: {mape:.2f}%")
                print(f"Directional Accuracy: {directional_accuracy:.2f}%")
                print(f"Correlation: {correlation:.4f}")
                print(f"Hit Rate (5%): {hit_rate_5:.2f}%")
                print(f"Hit Rate (10%): {hit_rate_10:.2f}%")
                
                # Trading performance
                actual_returns = np.diff(actual_prices) / actual_prices[:-1]
                predicted_returns = np.diff(predicted_prices) / predicted_prices[:-1]
                signals = np.where(predicted_returns > 0, 1, -1)
                strategy_returns = signals * actual_returns
                
                total_return = np.prod(1 + strategy_returns) - 1
                win_rate = np.mean(strategy_returns > 0) * 100
                
                print(f"\n💰 TRADING PERFORMANCE:")
                print("-" * 30)
                print(f"Total Return: {total_return:.2%}")
                print(f"Win Rate: {win_rate:.2f}%")
                
                # Key insights
                print(f"\n🎯 KEY INSIGHTS:")
                print("-" * 30)
                
                if directional_accuracy > 60:
                    print("✅ Good directional accuracy - model predicts price direction well")
                elif directional_accuracy > 50:
                    print("⚠️ Moderate directional accuracy - some room for improvement")
                else:
                    print("❌ Poor directional accuracy - model struggles with direction")
                
                if mape < 10:
                    print("✅ Good prediction accuracy (MAPE < 10%)")
                elif mape < 20:
                    print("⚠️ Moderate prediction accuracy (MAPE < 20%)")
                else:
                    print("❌ Poor prediction accuracy (MAPE > 20%)")
                
                if total_return > 0:
                    print("✅ Positive trading returns")
                else:
                    print("❌ Negative trading returns")
                
                # Show sample predictions
                print(f"\n📊 SAMPLE PREDICTIONS (First 10 days):")
                print("-" * 50)
                print("Date\t\tActual\t\tPredicted\tError\t\tError%")
                print("-" * 50)
                
                for i in range(min(10, len(actual_prices))):
                    actual = actual_prices[i]
                    predicted = predicted_prices[i]
                    error = predicted - actual
                    error_pct = (error / actual) * 100
                    date = test_data['Date'].iloc[i].strftime('%Y-%m-%d')
                    print(f"{date}\t${actual:.2f}\t\t${predicted:.2f}\t\t${error:.2f}\t\t{error_pct:.2f}%")
                
            else:
                print(f"❌ Mismatch in data lengths: {len(actual_prices)} actual vs {len(predicted_prices)} predicted")
                
        else:
            print("❌ No predictions generated")
            
    except Exception as e:
        print(f"❌ Error in backtest: {e}")
        import traceback
        traceback.print_exc()

def test_multiple_stocks():
    """Test backtesting on multiple stocks."""
    stocks = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN']
    
    print("🧪 Testing Multiple Stocks")
    print("=" * 50)
    
    results = {}
    
    for stock in stocks:
        print(f"\n📊 Testing {stock}...")
        try:
            simple_backtest(stock, test_days=30)
            results[stock] = "✅ Success"
        except Exception as e:
            print(f"❌ Error testing {stock}: {e}")
            results[stock] = "❌ Failed"
    
    print(f"\n📋 SUMMARY:")
    print("-" * 30)
    for stock, result in results.items():
        print(f"{stock}: {result}")

if __name__ == "__main__":
    # Test single stock
    simple_backtest('AAPL', test_days=30)
    
    # Uncomment to test multiple stocks
    # test_multiple_stocks() 