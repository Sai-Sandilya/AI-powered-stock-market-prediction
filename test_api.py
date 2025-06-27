from fastapi import FastAPI, HTTPException
import uvicorn
import pandas as pd
import torch
from model_training import LSTMModel
from feature_engineering import add_technical_indicators
from data_ingestion import fetch_yfinance
import os
from datetime import datetime, timedelta

app = FastAPI(title="Stock Prediction API - Test Version", version="1.0.0")

@app.get('/')
def root():
    return {
        "message": "Stock Prediction API - Test Version", 
        "endpoints": ["/predict", "/health", "/docs"],
        "status": "operational"
    }

@app.get('/health')
def health():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get('/predict')
def predict(symbol: str, days: int = 30):
    try:
        # Fetch data using the data ingestion module
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)  # Get 1 year of data
        
        df = fetch_yfinance(symbol, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        
        if df.empty:
            raise HTTPException(status_code=404, detail=f"No data found for symbol {symbol}")
        
        # Add technical indicators
        df = add_technical_indicators(df)
        
        # Prepare sequence for LSTM (last 60 days)
        if len(df) < 60:
            raise HTTPException(status_code=400, detail=f"Insufficient data for {symbol}. Need at least 60 days, got {len(df)}")
        
        # Get the required features
        features = ['MA_20', 'MA_50', 'RSI_14', 'Volatility']
        
        # Check if all features exist
        missing_features = [f for f in features if f not in df.columns]
        if missing_features:
            raise HTTPException(status_code=500, detail=f"Missing features: {missing_features}")
        
        # Create sequence
        seq_data = df[features].values[-60:]
        seq = torch.tensor(seq_data, dtype=torch.float32).unsqueeze(0)
        
        # Create a new model (no need to load trained weights for demo)
        model = LSTMModel(input_dim=4)
        model.eval()
        
        # Generate sample forecasts (since we don't have a trained model)
        current_price = df['Close'].iloc[-1]
        
        # Simple trend-based prediction for demo
        import random
        forecasts = []
        trend = random.uniform(-0.02, 0.02)  # Random trend between -2% and +2% per day
        
        for i in range(1, days + 1):
            # Add some randomness to make it realistic
            daily_change = trend + random.uniform(-0.01, 0.01)
            new_price = current_price * (1 + daily_change) ** i
            forecasts.append(round(new_price, 2))
        
        return {
            'symbol': symbol,
            'forecast': forecasts,
            'current_price': round(current_price, 2),
            'last_updated': str(df.index[-1])[:10],
            'forecast_days': days,
            'message': 'Demo prediction (model not trained)',
            'technical_indicators': {
                'rsi': round(df['RSI_14'].iloc[-1], 2),
                'ma_20': round(df['MA_20'].iloc[-1], 2),
                'ma_50': round(df['MA_50'].iloc[-1], 2),
                'volatility': round(df['Volatility'].iloc[-1], 4)
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000) 