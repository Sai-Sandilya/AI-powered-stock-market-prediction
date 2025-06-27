"""
LSTM Training Script
Trains a real LSTM model on historical stock data with advanced features.
Saves the trained model and scaler for inference.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from data_ingestion import fetch_yfinance
from feature_engineering import get_comprehensive_features


def create_lstm_dataset(df, feature_cols, target_col='Close', window=20):
    X, y = [], []
    for i in range(window, len(df)):
        X.append(df[feature_cols].iloc[i-window:i].values)
        y.append(df[target_col].iloc[i])
    return np.array(X), np.array(y)


def train_lstm_model(symbol='AAPL',
                     start_date='2018-01-01',
                     end_date=None,
                     window=20,
                     test_size=0.2,
                     epochs=50,
                     batch_size=32,
                     lstm_units=64,
                     dropout=0.2,
                     save_dir='models'):
    print(f"📈 Training LSTM for {symbol}")
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    # Fetch data
    df = fetch_yfinance(symbol, start_date, end_date)
    print(f"✅ Fetched {len(df)} records")
    
    # Feature engineering
    df = get_comprehensive_features(df, include_macro=True)
    df = df.dropna().reset_index(drop=True)
    print(f"✅ Features generated: {df.shape[1]}")
    
    # Select features
    feature_cols = [col for col in df.columns if col not in ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
    print(f"✅ Using {len(feature_cols)} features")
    
    # Scale features
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    
    # Prepare LSTM dataset
    X, y = create_lstm_dataset(df, feature_cols, target_col='Close', window=window)
    print(f"✅ LSTM dataset: X={X.shape}, y={y.shape}")
    
    # Train/val split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, shuffle=False)
    print(f"✅ Train: {X_train.shape[0]}, Val: {X_val.shape[0]}")
    
    # Build model
    model = Sequential([
        LSTM(lstm_units, return_sequences=False, input_shape=(window, len(feature_cols))),
        Dropout(dropout),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.summary()
    
    # Train
    callbacks = [EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=2
    )
    
    # Evaluate
    val_pred = model.predict(X_val)
    val_rmse = np.sqrt(np.mean((val_pred.flatten() - y_val) ** 2))
    print(f"✅ Validation RMSE: {val_rmse:.2f}")
    
    # Save model and scaler
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, f'{symbol}_lstm.h5')
    scaler_path = os.path.join(save_dir, f'{symbol}_scaler.pkl')
    model.save(model_path)
    joblib.dump(scaler, scaler_path)
    print(f"✅ Model saved: {model_path}")
    print(f"✅ Scaler saved: {scaler_path}")
    
    return model_path, scaler_path


def main():
    # You can adjust these parameters as needed
    train_lstm_model(
        symbol='AAPL',
        start_date='2018-01-01',
        window=20,
        test_size=0.2,
        epochs=30,
        batch_size=32,
        lstm_units=64,
        dropout=0.2,
        save_dir='models'
    )

if __name__ == '__main__':
    main() 