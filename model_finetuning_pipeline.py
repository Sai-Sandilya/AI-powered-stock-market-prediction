import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout, Bidirectional, Conv1D, MaxPooling1D, Flatten
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.preprocessing import RobustScaler
import optuna
import joblib
import os
import warnings

# Import the new advanced feature engineer
from advanced_feature_engineering_v2 import AdvancedFeatureEngineer

warnings.filterwarnings('ignore')

class ModelFinetuningPipeline:
    def __init__(self, symbol='AAPL', start_date='2020-01-01', end_date=None):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        self.scaler = RobustScaler()
        self.feature_engineer = AdvancedFeatureEngineer()
        self.results = {}
        os.makedirs('models', exist_ok=True)

    def fetch_and_engineer_features(self):
        """Fetch data and apply advanced feature engineering."""
        print(f"📊 Fetching and engineering features for {self.symbol}...")
        stock = yf.Ticker(self.symbol)
        df = stock.history(start=self.start_date, end=self.end_date)
        
        if df.empty:
            raise ValueError(f"No data found for {self.symbol}")
        
        # Apply all advanced features
        df = self.feature_engineer.create_all_features(df)
        
        # Handle potential infinite values
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Drop rows with NaN values created by feature engineering
        df.dropna(inplace=True)
        
        print(f"✅ Fetched and engineered {len(df)} records with {len(df.columns)} features")
        return df

    def prepare_data(self, df, sequence_length=60, target_column='Close'):
        """Prepare data for LSTM training with engineered features."""
        exclude_columns = ['Open', 'High', 'Low', 'Volume', 'Dividends', 'Stock Splits']
        # One-hot encode categorical features
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        
        feature_columns = [col for col in df.columns if col not in exclude_columns]
        data = df[feature_columns].values
        
        scaled_data = self.scaler.fit_transform(data)
        
        X, y = [], []
        for i in range(sequence_length, len(scaled_data)):
            X.append(scaled_data[i-sequence_length:i])
            y.append(scaled_data[i, feature_columns.index(target_column)])
        
        X, y = np.array(X), np.array(y)
        
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        self.feature_columns = feature_columns
        return X_train, X_test, y_train, y_test

    def create_model(self, model_type, input_shape, lstm_units=128, dropout_rate=0.3, learning_rate=0.001):
        """Creates a specified type of neural network model."""
        if model_type == 'lstm':
            model = Sequential([
                LSTM(lstm_units, return_sequences=True, input_shape=input_shape),
                Dropout(dropout_rate),
                LSTM(lstm_units // 2, return_sequences=False),
                Dropout(dropout_rate),
                Dense(lstm_units // 4, activation='relu'),
                Dense(1, activation='linear')
            ])
        elif model_type == 'conv_lstm':
            model = Sequential([
                Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
                MaxPooling1D(pool_size=2),
                Dropout(dropout_rate),
                LSTM(lstm_units, return_sequences=False),
                Dropout(dropout_rate),
                Dense(lstm_units // 2, activation='relu'),
                Dense(1, activation='linear')
            ])
        elif model_type == 'bidirectional':
            model = Sequential([
                Bidirectional(LSTM(lstm_units, return_sequences=True), input_shape=input_shape),
                Dropout(dropout_rate),
                Bidirectional(LSTM(lstm_units // 2, return_sequences=False)),
                Dropout(dropout_rate),
                Dense(lstm_units // 4, activation='relu'),
                Dense(1, activation='linear')
            ])
        
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss='huber', metrics=['mae', 'mse'])
        return model

    def objective(self, trial, df):
        """Optuna objective function for hyperparameter optimization."""
        sequence_length = trial.suggest_categorical('sequence_length', [30, 60, 90])
        X_train, X_test, y_train, y_test = self.prepare_data(df, sequence_length)

        model_type = trial.suggest_categorical('model_type', ['lstm', 'conv_lstm', 'bidirectional'])
        lstm_units = trial.suggest_categorical('lstm_units', [64, 128, 256])
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
        
        model = self.create_model(
            model_type, (X_train.shape[1], X_train.shape[2]),
            lstm_units, dropout_rate, learning_rate
        )
        
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=50,
            batch_size=batch_size,
            callbacks=[early_stopping],
            verbose=0
        )
        
        val_loss = min(history.history['val_loss'])
        return val_loss

    def run_finetuning(self, n_trials=20):
        """Runs the complete finetuning pipeline."""
        df = self.fetch_and_engineer_features()
        
        study = optuna.create_study(direction='minimize')
        study.optimize(lambda trial: self.objective(trial, df), n_trials=n_trials)
        
        best_params = study.best_params
        print(f"🏆 Best Hyperparameters found: {best_params}")

        # Train final model with best parameters on the full dataset
        sequence_length = best_params['sequence_length']
        # Re-prepare data to get the full set correctly
        X_train, X_test, y_train, y_test = self.prepare_data(df, sequence_length)
        X_full = np.concatenate((X_train, X_test))
        y_full = np.concatenate((y_train, y_test))

        
        final_model = self.create_model(
            best_params['model_type'],
            (X_full.shape[1], X_full.shape[2]),
            best_params['lstm_units'],
            best_params['dropout_rate'],
            best_params['learning_rate']
        )

        checkpoint_path = f"models/{self.symbol}_finetuned_best.h5"
        model_checkpoint = ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_loss')
        
        final_model.fit(
            X_full, y_full,
            epochs=100, 
            batch_size=best_params['batch_size'],
            validation_split=0.1,
            callbacks=[
                EarlyStopping(patience=15),
                ReduceLROnPlateau(patience=5),
                model_checkpoint
            ],
            verbose=1
        )
        
        # Save the scaler and feature columns
        joblib.dump(self.scaler, f"models/{self.symbol}_finetuned_scaler.pkl")
        joblib.dump(self.feature_columns, f"models/{self.symbol}_feature_columns.pkl")
        
        print(f"✅ Final model and scaler for {self.symbol} saved successfully.")
        return best_params

def run_finetuning_for_stock(symbol, n_trials=20):
    """Convenience function to run the pipeline for a single stock."""
    pipeline = ModelFinetuningPipeline(symbol=symbol)
    pipeline.run_finetuning(n_trials=n_trials)

if __name__ == '__main__':
    # This allows running the finetuning for a specific stock from the command line
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, default="AAPL", help="Stock symbol to train on.")
    parser.add_argument("--trials", type=int, default=20, help="Number of Optuna trials.")
    args = parser.parse_args()
    
    run_finetuning_for_stock(args.symbol, args.trials) 