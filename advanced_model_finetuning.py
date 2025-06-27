import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Attention
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import optuna
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

class AdvancedModelFinetuner:
    def __init__(self, symbol='AAPL', start_date='2020-01-01', end_date=None):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        self.scaler = MinMaxScaler()
        self.models = {}
        self.results = {}
        
    def fetch_data(self):
        """Fetch comprehensive stock data"""
        print(f"📊 Fetching data for {self.symbol}...")
        stock = yf.Ticker(self.symbol)
        df = stock.history(start=self.start_date, end=self.end_date)
        
        if df.empty:
            raise ValueError(f"No data found for {self.symbol}")
            
        # Add technical indicators
        df = self.add_technical_indicators(df)
        print(f"✅ Fetched {len(df)} records with {len(df.columns)} features")
        return df
    
    def add_technical_indicators(self, df):
        """Add comprehensive technical indicators"""
        # Price-based indicators
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        # Volume indicators
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        
        # Price momentum
        df['Price_Change'] = df['Close'].pct_change()
        df['Price_Change_5'] = df['Close'].pct_change(5)
        df['Price_Change_10'] = df['Close'].pct_change(10)
        df['Price_Change_20'] = df['Close'].pct_change(20)
        
        # Volatility
        df['Volatility'] = df['Price_Change'].rolling(window=20).std()
        
        # Support and Resistance
        df['Support'] = df['Low'].rolling(window=20).min()
        df['Resistance'] = df['High'].rolling(window=20).max()
        df['Support_Distance'] = (df['Close'] - df['Support']) / df['Close']
        df['Resistance_Distance'] = (df['Resistance'] - df['Close']) / df['Close']
        
        # Remove NaN values
        df = df.dropna()
        return df
    
    def prepare_data(self, df, sequence_length=60, target_column='Close'):
        """Prepare data for LSTM training"""
        # Select features
        feature_columns = [col for col in df.columns if col not in ['Open', 'High', 'Low', 'Volume']]
        data = df[feature_columns].values
        
        # Scale the data
        scaled_data = self.scaler.fit_transform(data)
        
        # Create sequences
        X, y = [], []
        for i in range(sequence_length, len(scaled_data)):
            X.append(scaled_data[i-sequence_length:i])
            y.append(scaled_data[i, feature_columns.index(target_column)])
        
        X, y = np.array(X), np.array(y)
        
        # Split data
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        return X_train, X_test, y_train, y_test, feature_columns
    
    def create_advanced_model(self, input_shape, lstm_units=128, dropout_rate=0.3, learning_rate=0.001):
        """Create advanced LSTM model with attention"""
        model = Sequential([
            # First LSTM layer with return sequences
            LSTM(lstm_units, return_sequences=True, input_shape=input_shape),
            Dropout(dropout_rate),
            
            # Second LSTM layer
            LSTM(lstm_units // 2, return_sequences=True),
            Dropout(dropout_rate),
            
            # Third LSTM layer
            LSTM(lstm_units // 4, return_sequences=False),
            Dropout(dropout_rate),
            
            # Dense layers
            Dense(lstm_units // 8, activation='relu'),
            Dropout(dropout_rate / 2),
            Dense(1, activation='linear')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def objective(self, trial):
        """Optuna objective function for hyperparameter optimization"""
        # Hyperparameters to optimize
        lstm_units = trial.suggest_categorical('lstm_units', [64, 128, 256, 512])
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
        sequence_length = trial.suggest_categorical('sequence_length', [30, 60, 90, 120])
        
        try:
            # Prepare data
            X_train, X_test, y_train, y_test, _ = self.prepare_data(
                self.df, sequence_length=sequence_length
            )
            
            # Create model
            model = self.create_advanced_model(
                input_shape=(X_train.shape[1], X_train.shape[2]),
                lstm_units=lstm_units,
                dropout_rate=dropout_rate,
                learning_rate=learning_rate
            )
            
            # Callbacks
            callbacks = [
                EarlyStopping(patience=15, restore_best_weights=True),
                ReduceLROnPlateau(patience=10, factor=0.5, min_lr=1e-7),
                ModelCheckpoint(
                    f'models/temp_best_{trial.number}.h5',
                    save_best_only=True,
                    monitor='val_loss'
                )
            ]
            
            # Train model
            history = model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=100,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=0
            )
            
            # Evaluate
            val_loss = min(history.history['val_loss'])
            
            # Clean up
            if os.path.exists(f'models/temp_best_{trial.number}.h5'):
                os.remove(f'models/temp_best_{trial.number}.h5')
            
            return val_loss
            
        except Exception as e:
            print(f"Trial failed: {e}")
            return float('inf')
    
    def optimize_hyperparameters(self, n_trials=50):
        """Optimize hyperparameters using Optuna"""
        print("🔍 Starting hyperparameter optimization...")
        
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        # Create study
        study = optuna.create_study(direction='minimize')
        study.optimize(self.objective, n_trials=n_trials)
        
        print(f"✅ Best trial: {study.best_trial.value}")
        print(f"📊 Best parameters: {study.best_trial.params}")
        
        return study.best_trial.params
    
    def train_ensemble_models(self, best_params):
        """Train multiple models for ensemble"""
        print("🏗️ Training ensemble models...")
        
        # Prepare data
        X_train, X_test, y_train, y_test, feature_columns = self.prepare_data(
            self.df, sequence_length=best_params['sequence_length']
        )
        
        # Train multiple models with different architectures
        model_configs = [
            {'lstm_units': best_params['lstm_units'], 'dropout_rate': best_params['dropout_rate']},
            {'lstm_units': best_params['lstm_units'] // 2, 'dropout_rate': best_params['dropout_rate'] * 0.8},
            {'lstm_units': best_params['lstm_units'] * 2, 'dropout_rate': best_params['dropout_rate'] * 1.2},
        ]
        
        for i, config in enumerate(model_configs):
            print(f"Training model {i+1}/{len(model_configs)}...")
            
            model = self.create_advanced_model(
                input_shape=(X_train.shape[1], X_train.shape[2]),
                lstm_units=config['lstm_units'],
                dropout_rate=config['dropout_rate'],
                learning_rate=best_params['learning_rate']
            )
            
            callbacks = [
                EarlyStopping(patience=20, restore_best_weights=True),
                ReduceLROnPlateau(patience=15, factor=0.5, min_lr=1e-7),
                ModelCheckpoint(
                    f'models/{self.symbol}_ensemble_{i+1}.h5',
                    save_best_only=True,
                    monitor='val_loss'
                )
            ]
            
            history = model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=150,
                batch_size=best_params['batch_size'],
                callbacks=callbacks,
                verbose=1
            )
            
            self.models[f'ensemble_{i+1}'] = {
                'model': model,
                'history': history,
                'config': config
            }
        
        # Save scaler and feature info
        joblib.dump(self.scaler, f'models/{self.symbol}_ensemble_scaler.pkl')
        joblib.dump(feature_columns, f'models/{self.symbol}_ensemble_features.pkl')
        
        print("✅ Ensemble models trained successfully!")
    
    def backtest_ensemble(self, test_period_days=252):
        """Comprehensive backtesting of ensemble models"""
        print(f"📈 Starting comprehensive backtesting for {test_period_days} days...")
        
        # Prepare test data
        X_train, X_test, y_train, y_test, feature_columns = self.prepare_data(
            self.df, sequence_length=60
        )
        
        # Load scaler and features
        scaler = joblib.load(f'models/{self.symbol}_ensemble_scaler.pkl')
        feature_columns = joblib.load(f'models/{self.symbol}_ensemble_features.pkl')
        
        results = {}
        
        for model_name in self.models.keys():
            print(f"Testing {model_name}...")
            
            # Load model
            model = load_model(f'models/{self.symbol}_{model_name}.h5')
            
            # Make predictions
            predictions = model.predict(X_test)
            
            # Inverse transform predictions
            pred_reshaped = np.zeros((len(predictions), len(feature_columns)))
            pred_reshaped[:, feature_columns.index('Close')] = predictions.flatten()
            predictions_actual = scaler.inverse_transform(pred_reshaped)[:, feature_columns.index('Close')]
            
            # Inverse transform actual values
            actual_reshaped = np.zeros((len(y_test), len(feature_columns)))
            actual_reshaped[:, feature_columns.index('Close')] = y_test
            actual_actual = scaler.inverse_transform(actual_reshaped)[:, feature_columns.index('Close')]
            
            # Calculate metrics
            mse = mean_squared_error(actual_actual, predictions_actual)
            mae = mean_absolute_error(actual_actual, predictions_actual)
            rmse = np.sqrt(mse)
            mape = np.mean(np.abs((actual_actual - predictions_actual) / actual_actual)) * 100
            
            # Directional accuracy
            actual_direction = np.diff(actual_actual) > 0
            pred_direction = np.diff(predictions_actual) > 0
            directional_accuracy = np.mean(actual_direction == pred_direction) * 100
            
            # Trading returns
            returns = np.diff(actual_actual) / actual_actual[:-1]
            pred_returns = np.diff(predictions_actual) / predictions_actual[:-1]
            
            # Simple trading strategy
            position = np.where(pred_returns > 0, 1, -1)
            strategy_returns = position * returns
            
            cumulative_return = np.prod(1 + strategy_returns) - 1
            sharpe_ratio = np.mean(strategy_returns) / np.std(strategy_returns) if np.std(strategy_returns) > 0 else 0
            
            results[model_name] = {
                'MSE': mse,
                'MAE': mae,
                'RMSE': rmse,
                'MAPE': mape,
                'Directional_Accuracy': directional_accuracy,
                'Cumulative_Return': cumulative_return,
                'Sharpe_Ratio': sharpe_ratio,
                'Predictions': predictions_actual,
                'Actual': actual_actual
            }
        
        # Ensemble predictions (average of all models)
        ensemble_predictions = np.mean([results[model]['Predictions'] for model in results.keys()], axis=0)
        
        # Calculate ensemble metrics
        mse = mean_squared_error(results[list(results.keys())[0]]['Actual'], ensemble_predictions)
        mae = mean_absolute_error(results[list(results.keys())[0]]['Actual'], ensemble_predictions)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((results[list(results.keys())[0]]['Actual'] - ensemble_predictions) / results[list(results.keys())[0]]['Actual'])) * 100
        
        actual_direction = np.diff(results[list(results.keys())[0]]['Actual']) > 0
        pred_direction = np.diff(ensemble_predictions) > 0
        directional_accuracy = np.mean(actual_direction == pred_direction) * 100
        
        returns = np.diff(results[list(results.keys())[0]]['Actual']) / results[list(results.keys())[0]]['Actual'][:-1]
        pred_returns = np.diff(ensemble_predictions) / ensemble_predictions[:-1]
        position = np.where(pred_returns > 0, 1, -1)
        strategy_returns = position * returns
        cumulative_return = np.prod(1 + strategy_returns) - 1
        sharpe_ratio = np.mean(strategy_returns) / np.std(strategy_returns) if np.std(strategy_returns) > 0 else 0
        
        results['ensemble'] = {
            'MSE': mse,
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape,
            'Directional_Accuracy': directional_accuracy,
            'Cumulative_Return': cumulative_return,
            'Sharpe_Ratio': sharpe_ratio,
            'Predictions': ensemble_predictions,
            'Actual': results[list(results.keys())[0]]['Actual']
        }
        
        self.results = results
        return results
    
    def print_results(self):
        """Print comprehensive backtesting results"""
        print("\n" + "="*80)
        print(f"📊 COMPREHENSIVE BACKTESTING RESULTS FOR {self.symbol}")
        print("="*80)
        
        for model_name, metrics in self.results.items():
            print(f"\n🎯 {model_name.upper()}:")
            print(f"   📈 RMSE: ${metrics['RMSE']:.2f}")
            print(f"   📏 MAE: ${metrics['MAE']:.2f}")
            print(f"   📊 MAPE: {metrics['MAPE']:.2f}%")
            print(f"   🎯 Directional Accuracy: {metrics['Directional_Accuracy']:.2f}%")
            print(f"   💰 Cumulative Return: {metrics['Cumulative_Return']:.2%}")
            print(f"   📊 Sharpe Ratio: {metrics['Sharpe_Ratio']:.2f}")
        
        # Find best model
        best_model = max(self.results.keys(), key=lambda x: self.results[x]['Sharpe_Ratio'])
        print(f"\n🏆 BEST MODEL: {best_model}")
        print(f"   Sharpe Ratio: {self.results[best_model]['Sharpe_Ratio']:.2f}")
        print(f"   Cumulative Return: {self.results[best_model]['Cumulative_Return']:.2%}")
    
    def run_complete_pipeline(self, n_trials=30):
        """Run the complete fine-tuning and backtesting pipeline"""
        print("🚀 Starting complete model fine-tuning and backtesting pipeline...")
        
        # Fetch data
        self.df = self.fetch_data()
        
        # Optimize hyperparameters
        best_params = self.optimize_hyperparameters(n_trials=n_trials)
        
        # Train ensemble models
        self.train_ensemble_models(best_params)
        
        # Backtest
        results = self.backtest_ensemble()
        
        # Print results
        self.print_results()
        
        return results

if __name__ == "__main__":
    # Run fine-tuning for multiple stocks
    stocks = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
    
    for stock in stocks:
        print(f"\n{'='*60}")
        print(f"🎯 PROCESSING {stock}")
        print(f"{'='*60}")
        
        try:
            finetuner = AdvancedModelFinetuner(symbol=stock)
            results = finetuner.run_complete_pipeline(n_trials=20)
            
            # Save results
            joblib.dump(results, f'results/{stock}_finetuned_results.pkl')
            
        except Exception as e:
            print(f"❌ Error processing {stock}: {e}")
            continue 