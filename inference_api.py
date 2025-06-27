from fastapi import FastAPI, HTTPException
import uvicorn
import pandas as pd
import torch
from model_training import LSTMModel
from feature_engineering import add_technical_indicators
from data_ingestion import fetch_yfinance
import os
from datetime import datetime, timedelta

app = FastAPI(title="Stock Prediction API", version="1.0.0")

@app.get('/')
def root():
    return {"message": "Stock Prediction API", "endpoints": ["/predict", "/docs"]}

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
        
        # Load model
    model = LSTMModel(input_dim=4)
        
        # Check if model file exists
        model_path = 'models/lstm_model.pth'
        if not os.path.exists(model_path):
            # Return sample prediction if model doesn't exist
            current_price = df['Close'].iloc[-1]
            forecasts = [current_price * (1 + 0.01 * i) for i in range(1, days + 1)]
            return {
                'symbol': symbol,
                'forecast': forecasts,
                'current_price': current_price,
                'message': 'Using sample prediction (model not trained)'
            }
        
        model.load_state_dict(torch.load(model_path))
    model.eval()
        
        # Generate forecasts
    forecasts = []
        current_seq = seq.clone()
        
        with torch.no_grad():
    for _ in range(days):
                pred = model(current_seq).item()
        forecasts.append(pred)
                
                # Update sequence for next prediction
                new_data = current_seq.squeeze(0).tolist()[1:]
                new_data.append([pred] * 4)  # Use prediction for all features
                current_seq = torch.tensor([new_data], dtype=torch.float32)
        
        return {
            'symbol': symbol,
            'forecast': forecasts,
            'current_price': df['Close'].iloc[-1],
            'last_updated': str(df.index[-1])[:10],  # Convert to string and take date part
            'forecast_days': days
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
