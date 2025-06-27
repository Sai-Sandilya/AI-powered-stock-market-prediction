"""
Advanced LSTM Training Script
Trains optimized LSTM models with hyperparameter tuning, multiple architectures,
and comprehensive feature engineering for better stock prediction accuracy.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1_l2
import optuna
from optuna.integration import TFKerasPruningCallback
import warnings
warnings.filterwarnings('ignore')

from data_ingestion import fetch_yfinance
from feature_engineering import get_comprehensive_features
from macro_indicators import MacroIndicators


def create_advanced_dataset(df, feature_cols, target_col='Close', window=20, prediction_horizon=1):
    """Create dataset with multiple prediction horizons"""
    X, y = [], []
    for i in range(window, len(df) - prediction_horizon + 1):
        X.append(df[feature_cols].iloc[i-window:i].values)
        y.append(df[target_col].iloc[i:i+prediction_horizon].values)
    return np.array(X), np.array(y)


def build_model_architecture(architecture, input_shape, lstm_units, dropout, learning_rate):
    """Build different model architectures"""
    model = Sequential()
    
    if architecture == 'simple_lstm':
        model.add(LSTM(lstm_units, input_shape=input_shape, return_sequences=False))
        model.add(Dropout(dropout))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1))
        
    elif architecture == 'stacked_lstm':
        model.add(LSTM(lstm_units, input_shape=input_shape, return_sequences=True))
        model.add(Dropout(dropout))
        model.add(LSTM(lstm_units // 2, return_sequences=False))
        model.add(Dropout(dropout))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1))
        
    elif architecture == 'bidirectional_lstm':
        model.add(Bidirectional(LSTM(lstm_units, return_sequences=True), input_shape=input_shape))
        model.add(Dropout(dropout))
        model.add(Bidirectional(LSTM(lstm_units // 2, return_sequences=False)))
        model.add(Dropout(dropout))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(1))
        
    elif architecture == 'cnn_lstm':
        model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(LSTM(lstm_units, return_sequences=False))
        model.add(Dropout(dropout))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(1))
        
    elif architecture == 'deep_lstm':
        model.add(LSTM(lstm_units, input_shape=input_shape, return_sequences=True))
        model.add(Dropout(dropout))
        model.add(LSTM(lstm_units // 2, return_sequences=True))
        model.add(Dropout(dropout))
        model.add(LSTM(lstm_units // 4, return_sequences=False))
        model.add(Dropout(dropout))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1))
    
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    return model


def objective(trial, X_train, y_train, X_val, y_val, input_shape):
    """Optuna objective function for hyperparameter optimization"""
    
    # Hyperparameters to optimize
    architecture = trial.suggest_categorical('architecture', 
                                           ['simple_lstm', 'stacked_lstm', 'bidirectional_lstm', 'cnn_lstm', 'deep_lstm'])
    lstm_units = trial.suggest_int('lstm_units', 32, 256, step=32)
    dropout = trial.suggest_float('dropout', 0.1, 0.5, step=0.1)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
    epochs = trial.suggest_int('epochs', 50, 200)
    
    # Build model
    model = build_model_architecture(architecture, input_shape, lstm_units, dropout, learning_rate)
    
    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7),
        TFKerasPruningCallback(trial, 'val_loss')
    ]
    
    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=0
    )
    
    # Evaluate
    val_pred = model.predict(X_val, verbose=0)
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    
    return val_rmse


def train_advanced_lstm(symbol='AAPL',
                       start_date='2015-01-01',
                       end_date=None,
                       window=30,
                       test_size=0.2,
                       prediction_horizon=1,
                       use_optuna=True,
                       n_trials=50,
                       save_dir='models'):
    """
    Train advanced LSTM model with hyperparameter optimization
    """
    print(f"🚀 Advanced LSTM Training for {symbol}")
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    # Fetch data
    print("📊 Fetching stock data...")
    df = fetch_yfinance(symbol, start_date, end_date)
    print(f"✅ Fetched {len(df)} records")
    
    # Enhanced feature engineering
    print("🔧 Generating comprehensive features...")
    df = get_comprehensive_features(df, include_macro=True)
    
    # Add macro features if available
    try:
        macro = MacroIndicators()
        macro_data = macro.get_macro_indicators(start_date, end_date)
        if macro_data:
            macro_features = macro.calculate_macro_features(macro_data)
            if not macro_features.empty:
                # Drop duplicate 'Close' columns before merge
                if 'Close' in macro_features.columns and 'Close' in df.columns:
                    macro_features = macro_features.drop(columns=['Close'])
                df = df.merge(macro_features, on='Date', how='left')
                df = df.fillna(method='ffill')
                print("✅ Added macroeconomic features")
    except Exception as e:
        print(f"⚠️ Could not add macro features: {e}")
    
    # Additional advanced features
    print("🔧 Adding advanced features...")
    
    # Volatility features
    df['volatility_5d'] = df['Close'].rolling(5).std()
    df['volatility_20d'] = df['Close'].rolling(20).std()
    df['volatility_ratio'] = df['volatility_5d'] / df['volatility_20d']
    
    # Price momentum features
    df['price_momentum_1d'] = df['Close'].pct_change(1)
    df['price_momentum_5d'] = df['Close'].pct_change(5)
    df['price_momentum_20d'] = df['Close'].pct_change(20)
    
    # Volume features
    df['volume_ma_5'] = df['Volume'].rolling(5).mean()
    df['volume_ma_20'] = df['Volume'].rolling(20).mean()
    df['volume_ratio'] = df['Volume'] / df['volume_ma_20']
    
    # Technical pattern features
    df['higher_high'] = (df['High'] > df['High'].shift(1)).astype(int)
    df['lower_low'] = (df['Low'] < df['Low'].shift(1)).astype(int)
    df['gap_up'] = (df['Open'] > df['Close'].shift(1)).astype(int)
    df['gap_down'] = (df['Open'] < df['Close'].shift(1)).astype(int)
    
    # Market regime features
    df['trend_strength'] = abs(df['Close'] - df['Close'].rolling(20).mean()) / df['Close'].rolling(20).std()
    df['market_regime'] = np.where(df['trend_strength'] > 1, 'trending', 'ranging')
    
    # Remove NaN values
    df = df.dropna().reset_index(drop=True)
    print(f"✅ Final dataset shape: {df.shape}")
    
    # Select features (exclude date and target columns)
    exclude_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'market_regime']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    # Only keep numeric columns
    feature_cols = [col for col in feature_cols if pd.api.types.is_numeric_dtype(df[col])]
    print(f"✅ Using {len(feature_cols)} features")
    
    # Scale features
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    
    # Prepare dataset
    X, y = create_advanced_dataset(df, feature_cols, target_col='Close', 
                                 window=window, prediction_horizon=prediction_horizon)
    print(f"✅ Dataset: X={X.shape}, y={y.shape}")
    
    # Time series split for validation
    tscv = TimeSeriesSplit(n_splits=3)
    splits = list(tscv.split(X))
    
    # Use last split for final training
    train_idx, val_idx = splits[-1]
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    print(f"✅ Train: {X_train.shape[0]}, Val: {X_val.shape[0]}")
    
    if use_optuna:
        print("🔍 Running hyperparameter optimization...")
        study = optuna.create_study(direction='minimize')
        study.optimize(lambda trial: objective(trial, X_train, y_train, X_val, y_val, (window, len(feature_cols))), 
                      n_trials=n_trials)
        
        best_params = study.best_params
        print(f"✅ Best parameters: {best_params}")
        
        # Train final model with best parameters
        final_model = build_model_architecture(
            best_params['architecture'],
            (window, len(feature_cols)),
            best_params['lstm_units'],
            best_params['dropout'],
            best_params['learning_rate']
        )
        
        # Enhanced callbacks for final training
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-7),
            ModelCheckpoint(
                filepath=f'{save_dir}/{symbol}_best_model.h5',
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train final model with more epochs
        history = final_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=best_params['epochs'] * 2,  # Double the epochs for final training
            batch_size=best_params['batch_size'],
            callbacks=callbacks,
            verbose=2
        )
        
    else:
        # Use default parameters
        final_model = build_model_architecture(
            'stacked_lstm',
            (window, len(feature_cols)),
            128,
            0.3,
            0.001
        )
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7)
        ]
        
        history = final_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=150,
            batch_size=64,
            callbacks=callbacks,
            verbose=2
        )
    
    # Evaluate final model
    val_pred = final_model.predict(X_val, verbose=0)
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    val_mae = mean_absolute_error(y_val, val_pred)
    
    print(f"✅ Final Validation RMSE: {val_rmse:.4f}")
    print(f"✅ Final Validation MAE: {val_mae:.4f}")
    
    # Save model and scaler
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, f'{symbol}_advanced_lstm.h5')
    scaler_path = os.path.join(save_dir, f'{symbol}_advanced_scaler.pkl')
    
    final_model.save(model_path)
    joblib.dump(scaler, scaler_path)
    
    # Save feature columns for inference
    feature_info = {
        'feature_cols': feature_cols,
        'window': window,
        'prediction_horizon': prediction_horizon,
        'best_params': best_params if use_optuna else None
    }
    joblib.dump(feature_info, os.path.join(save_dir, f'{symbol}_feature_info.pkl'))
    
    print(f"✅ Model saved: {model_path}")
    print(f"✅ Scaler saved: {scaler_path}")
    print(f"✅ Feature info saved")
    
    return model_path, scaler_path, feature_info


def main():
    """Main training function"""
    # Train with hyperparameter optimization
    train_advanced_lstm(
        symbol='AAPL',
        start_date='2015-01-01',
        window=30,
        test_size=0.2,
        prediction_horizon=1,
        use_optuna=True,
        n_trials=30,  # Reduced for faster training
        save_dir='models'
    )


if __name__ == '__main__':
    main() 