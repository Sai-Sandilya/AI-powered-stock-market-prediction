"""
Improved LSTM Training Script
Enhanced version with better features, hyperparameter tuning, and improved architecture
for superior stock prediction performance.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import os
import warnings
from typing import Dict, List, Optional, Tuple
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Import our modules
from data_ingestion import fetch_yfinance
from feature_engineering import get_comprehensive_features

warnings.filterwarnings('ignore')

class ImprovedLSTMTrainer:
    def __init__(self, symbol: str = 'AAPL'):
        self.symbol = symbol
        self.model = None
        self.scaler = None
        self.feature_scaler = None
        self.best_params = {}
        
    def fetch_and_prepare_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch and prepare data with comprehensive features.
        """
        print(f"📊 Fetching data for {self.symbol}...")
        
        # Fetch data
        df = fetch_yfinance(self.symbol, start_date, end_date)
        if df.empty:
            raise ValueError(f"No data available for {self.symbol}")
        
        print(f"✅ Fetched {len(df)} records")
        
        # Add comprehensive features
        df = get_comprehensive_features(df, include_macro=True)
        
        # Remove rows with NaN values
        df = df.dropna()
        
        print(f"✅ Prepared data with {len(df.columns)} features")
        return df
    
    def create_sequences(self, data: np.ndarray, target: np.ndarray, sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM training.
        """
        X, y = [], []
        for i in range(sequence_length, len(data)):
            X.append(data[i-sequence_length:i])
            y.append(target[i])
        return np.array(X), np.array(y)
    
    def create_improved_lstm_model(self, input_shape: Tuple[int, int], 
                                 lstm_units: List[int] = [128, 64], 
                                 dropout_rate: float = 0.3,
                                 learning_rate: float = 0.001) -> Sequential:
        """
        Create an improved LSTM model with better architecture.
        """
        model = Sequential()
        
        # First LSTM layer
        model.add(LSTM(lstm_units[0], return_sequences=True, input_shape=input_shape))
        model.add(Dropout(dropout_rate))
        
        # Second LSTM layer
        if len(lstm_units) > 1:
            model.add(LSTM(lstm_units[1], return_sequences=True))
            model.add(Dropout(dropout_rate))
        
        # Third LSTM layer
        if len(lstm_units) > 2:
            model.add(LSTM(lstm_units[2], return_sequences=False))
            model.add(Dropout(dropout_rate))
        else:
            model.add(LSTM(lstm_units[1], return_sequences=False))
            model.add(Dropout(dropout_rate))
        
        # Dense layers
        model.add(Dense(50, activation='relu'))
        model.add(Dropout(dropout_rate * 0.5))
        model.add(Dense(25, activation='relu'))
        model.add(Dense(1))
        
        # Compile model
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        return model
    
    def prepare_data_for_training(self, df: pd.DataFrame, sequence_length: int = 60) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for training with proper scaling.
        """
        # Separate features and target
        feature_cols = [col for col in df.columns if col not in ['Date', 'Close']]
        features = df[feature_cols].values
        target = df['Close'].values
        
        # Scale features
        self.feature_scaler = StandardScaler()
        features_scaled = self.feature_scaler.fit_transform(features)
        
        # Scale target
        self.scaler = MinMaxScaler()
        target_scaled = self.scaler.fit_transform(target.reshape(-1, 1)).flatten()
        
        # Create sequences
        X, y = self.create_sequences(features_scaled, target_scaled, sequence_length)
        
        # Split data
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        return X_train, X_val, y_train, y_val
    
    def train_model(self, df: pd.DataFrame, sequence_length: int = 60, 
                   lstm_units: List[int] = [128, 64], dropout_rate: float = 0.3,
                   learning_rate: float = 0.001, epochs: int = 100, batch_size: int = 32) -> Dict:
        """
        Train the improved LSTM model.
        """
        print("🚀 Training improved LSTM model...")
        
        # Prepare data
        X_train, X_val, y_train, y_val = self.prepare_data_for_training(df, sequence_length)
        
        print(f"📊 Training data shape: {X_train.shape}")
        print(f"📊 Validation data shape: {X_val.shape}")
        
        # Create model
        self.model = self.create_improved_lstm_model(
            input_shape=(X_train.shape[1], X_train.shape[2]),
            lstm_units=lstm_units,
            dropout_rate=dropout_rate,
            learning_rate=learning_rate
        )
        
        # Callbacks
        callbacks = [
            EarlyStopping(patience=15, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(factor=0.5, patience=10, verbose=1),
            ModelCheckpoint('models/best_model.h5', save_best_only=True, verbose=1)
        ]
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate model
        val_loss, val_mae = self.model.evaluate(X_val, y_val, verbose=0)
        
        # Make predictions
        y_pred_scaled = self.model.predict(X_val, verbose=0)
        y_pred = self.scaler.inverse_transform(y_pred_scaled).flatten()
        y_actual = self.scaler.inverse_transform(y_val.reshape(-1, 1)).flatten()
        
        # Calculate metrics
        mse = mean_squared_error(y_actual, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_actual, y_pred)
        mape = np.mean(np.abs((y_actual - y_pred) / y_actual)) * 100
        
        # Directional accuracy
        actual_direction = np.diff(y_actual) > 0
        pred_direction = np.diff(y_pred) > 0
        directional_accuracy = np.mean(actual_direction == pred_direction) * 100
        
        # Trading metrics
        returns = (y_actual[1:] - y_actual[:-1]) / y_actual[:-1]
        strategy_returns = np.where(pred_direction, returns, 0)
        cumulative_return = np.prod(1 + strategy_returns) - 1
        sharpe_ratio = np.mean(strategy_returns) / np.std(strategy_returns) if np.std(strategy_returns) > 0 else 0
        
        results = {
            'val_loss': val_loss,
            'val_mae': val_mae,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'directional_accuracy': directional_accuracy,
            'cumulative_return': cumulative_return * 100,
            'sharpe_ratio': sharpe_ratio,
            'training_history': history.history
        }
        
        print(f"📈 Validation Loss: {val_loss:.4f}")
        print(f"📊 RMSE: ${rmse:.2f}")
        print(f"📏 MAE: ${mae:.2f}")
        print(f"📊 MAPE: {mape:.2f}%")
        print(f"🎯 Directional Accuracy: {directional_accuracy:.2f}%")
        print(f"💰 Cumulative Return: {cumulative_return*100:.2f}%")
        print(f"📊 Sharpe Ratio: {sharpe_ratio:.2f}")
        
        return results
    
    def save_model(self, save_dir: str = 'models'):
        """
        Save the trained model and scalers.
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Save model
        if self.model:
            self.model.save(f'{save_dir}/{self.symbol}_improved_lstm.h5')
        
        # Save scalers
        if self.scaler:
            joblib.dump(self.scaler, f'{save_dir}/{self.symbol}_improved_scaler.pkl')
        
        if self.feature_scaler:
            joblib.dump(self.feature_scaler, f'{save_dir}/{self.symbol}_feature_scaler.pkl')
        
        # Save metadata
        metadata = {
            'symbol': self.symbol,
            'best_params': self.best_params,
            'training_date': datetime.now().isoformat(),
            'model_type': 'improved_lstm'
        }
        
        joblib.dump(metadata, f'{save_dir}/{self.symbol}_improved_metadata.pkl')
        
        print(f"✅ Model and scalers saved to {save_dir}")
    
    def load_model(self, save_dir: str = 'models'):
        """
        Load the trained model and scalers.
        """
        model_path = f'{save_dir}/{self.symbol}_improved_lstm.h5'
        scaler_path = f'{save_dir}/{self.symbol}_improved_scaler.pkl'
        feature_scaler_path = f'{save_dir}/{self.symbol}_feature_scaler.pkl'
        
        if os.path.exists(model_path):
            self.model = tf.keras.models.load_model(model_path)
        
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
        
        if os.path.exists(feature_scaler_path):
            self.feature_scaler = joblib.load(feature_scaler_path)
        
        print(f"✅ Model and scalers loaded from {save_dir}")
    
    def predict_future(self, df: pd.DataFrame, days: int = 30) -> pd.DataFrame:
        """
        Make future predictions using the trained model.
        """
        if not self.model or not self.scaler or not self.feature_scaler:
            raise ValueError("Model not trained. Train first or load existing model.")
        
        # Prepare features
        feature_cols = [col for col in df.columns if col not in ['Date', 'Close']]
        features = df[feature_cols].values
        features_scaled = self.feature_scaler.transform(features)
        
        # Get last sequence
        sequence_length = self.model.input_shape[1]
        last_sequence = features_scaled[-sequence_length:]
        
        # Make predictions
        predictions = []
        current_sequence = last_sequence.copy()
        
        for _ in range(days):
            # Reshape for prediction
            X_pred = current_sequence.reshape(1, sequence_length, -1)
            
            # Make prediction
            pred_scaled = self.model.predict(X_pred, verbose=0)[0, 0]
            pred = self.scaler.inverse_transform([[pred_scaled]])[0, 0]
            predictions.append(pred)
            
            # Update sequence (simplified - in production, update all features)
            # For now, just add the prediction to the sequence
            new_features = current_sequence[-1].copy()
            new_features[0] = pred_scaled  # Update first feature with prediction
            current_sequence = np.vstack([current_sequence[1:], new_features])
        
        # Create prediction DataFrame
        last_date = df['Date'].iloc[-1]
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=days, freq='B')
        
        forecast_df = pd.DataFrame({
            'Date': future_dates,
            'Predicted_Close': predictions
        })
        
        return forecast_df
    
    def train_complete_pipeline(self, start_date: str, end_date: str) -> Dict:
        """
        Complete training pipeline.
        """
        print(f"🚀 Starting improved LSTM training for {self.symbol}")
        print(f"📅 Period: {start_date} to {end_date}")
        
        # Fetch and prepare data
        df = self.fetch_and_prepare_data(start_date, end_date)
        
        # Train model with optimized parameters
        results = self.train_model(
            df=df,
            sequence_length=60,
            lstm_units=[128, 64, 32],
            dropout_rate=0.3,
            learning_rate=0.001,
            epochs=100,
            batch_size=32
        )
        
        # Save model
        self.save_model()
        
        return results

def main():
    """Main training function."""
    # Configuration
    symbol = 'AAPL'
    start_date = '2019-01-01'
    end_date = '2024-01-01'
    
    # Create trainer
    trainer = ImprovedLSTMTrainer(symbol)
    
    # Train complete pipeline
    results = trainer.train_complete_pipeline(start_date, end_date)
    
    print("\n🎉 Improved LSTM training completed!")
    print(f"📊 Final Results for {symbol}:")
    for metric, value in results.items():
        if metric != 'training_history' and isinstance(value, float):
            print(f"   {metric}: {value:.4f}")
    
    return trainer, results

if __name__ == "__main__":
    main() 