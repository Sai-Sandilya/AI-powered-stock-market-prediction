#!/usr/bin/env python3
"""
Test script to verify deployment fixes
Tests the future forecasting without tz_zone errors
"""

def test_forecasting_fix():
    """Test that forecasting works without tz_zone errors"""
    print("🧪 Testing Deployment Fix...")
    
    try:
        from future_forecasting import FutureForecaster
        print("✅ FutureForecaster import successful")
        
        # Initialize forecaster
        forecaster = FutureForecaster(symbol='AAPL')
        print("✅ FutureForecaster initialization successful")
        
        # Test short forecast to verify no tz_zone errors
        forecast_df = forecaster.forecast_future('AAPL', forecast_days=10, include_macro=False)
        print(f"✅ Forecast successful: {len(forecast_df)} predictions generated")
        
        if not forecast_df.empty:
            print(f"📊 First prediction: ${forecast_df['Predicted_Close'].iloc[0]:.2f}")
            print(f"📊 Last prediction: ${forecast_df['Predicted_Close'].iloc[-1]:.2f}")
            print("✅ All tests passed! Deployment should work now.")
            return True
        else:
            print("⚠️ Forecast returned empty DataFrame")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    test_forecasting_fix() 