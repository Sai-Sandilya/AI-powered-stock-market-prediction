"""
Enhanced Training Script v2.0
Comprehensive training with all advanced features and models
"""

import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# Import our enhanced modules
from advanced_feature_engineering_v2 import AdvancedFeatureEngineer
from multi_timeframe_signals import MultiTimeframeSignals
from enhanced_lstm_model import EnsembleLSTM, prepare_data_for_lstm

class EnhancedStockPredictor:
    def __init__(self, symbol: str = 'AAPL'):
        self.symbol = symbol
        self.feature_engineer = AdvancedFeatureEngineer()
        self.signal_generator = MultiTimeframeSignals()
        self.ensemble_model = None
        self.scaler = MinMaxScaler()
        self.feature_names = []
        
    def fetch_and_prepare_data(self, period: str = '2y') -> pd.DataFrame:
        """
        Fetch and prepare comprehensive dataset with all features.
        """
        print(f"📊 Fetching data for {self.symbol}...")
        
        # Fetch data
        ticker = yf.Ticker(self.symbol)
        df = ticker.history(period=period)
        df.reset_index(inplace=True)
        df.rename(columns={'Date': 'Date'}, inplace=True)
        
        print(f"✅ Fetched {len(df)} records")
        
        # Add basic technical indicators first
        df = self.add_basic_indicators(df)
        
        # Add advanced features
        df = self.feature_engineer.create_all_features(df)
        
        # Store feature names
        self.feature_names = self.feature_engineer.get_feature_names()
        
        print(f"🎯 Total features: {len(self.feature_names)}")
        return df
    
    def add_basic_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add basic technical indicators that advanced features depend on.
        """
        print("📈 Adding basic technical indicators...")
        
        # RSI
        df['RSI_14'] = self.calculate_rsi(df['Close'], 14)
        
        # MACD
        df['MACD'], df['MACD_Signal'], df['MACD_Histogram'] = self.calculate_macd(df['Close'])
        
        # Bollinger Bands
        df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = self.calculate_bollinger_bands(df['Close'])
        
        # Moving Averages
        for period in [5, 10, 20, 50, 200]:
            df[f'MA_{period}'] = df['Close'].rolling(period).mean()
        
        # Stochastic
        df['Stoch_K'], df['Stoch_D'] = self.calculate_stochastic(df['High'], df['Low'], df['Close'])
        
        return df
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
        """Calculate MACD."""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        histogram = macd - signal_line
        return macd, signal_line, histogram
    
    def calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: int = 2) -> tuple:
        """Calculate Bollinger Bands."""
        middle = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        return upper, middle, lower
    
    def calculate_stochastic(self, high: pd.Series, low: pd.Series, close: pd.Series, 
                           k_period: int = 14, d_period: int = 3) -> tuple:
        """Calculate Stochastic Oscillator."""
        lowest_low = low.rolling(k_period).min()
        highest_high = high.rolling(k_period).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(d_period).mean()
        return k_percent, d_percent
    
    def generate_signals(self, df: pd.DataFrame) -> dict:
        """
        Generate multi-timeframe trading signals.
        """
        print("🎯 Generating multi-timeframe signals...")
        
        # Generate all signals
        signals = self.signal_generator.generate_all_signals(df)
        
        # Get recommendations
        recommendations = self.signal_generator.get_trading_recommendations()
        
        print("📋 Trading Recommendations:")
        for strategy, status in recommendations.items():
            print(f"  {strategy}: {status}")
        
        return signals
    
    def prepare_training_data(self, df: pd.DataFrame, sequence_length: int = 60) -> tuple:
        """
        Prepare data for LSTM training with enhanced features.
        """
        print("🔧 Preparing training data...")
        
        # Prepare data using the enhanced function
        X_train, X_test, y_train, y_test, scaler, feature_cols = prepare_data_for_lstm(
            df, target_col='Close', sequence_length=sequence_length, test_size=0.2
        )
        
        self.scaler = scaler
        self.feature_names = feature_cols
        
        print(f"📊 Training data shape: {X_train.shape}")
        print(f"📊 Test data shape: {X_test.shape}")
        print(f"🎯 Features used: {len(feature_cols)}")
        
        return X_train, X_test, y_train, y_test
    
    def train_enhanced_model(self, X_train, y_train, X_test, y_test, 
                           sequence_length: int = 60, epochs: int = 100) -> dict:
        """
        Train enhanced ensemble LSTM model.
        """
        print("🚀 Training enhanced ensemble model...")
        
        # Create ensemble model
        self.ensemble_model = EnsembleLSTM(
            sequence_length=sequence_length,
            n_features=X_train.shape[2],
            n_outputs=1
        )
        
        # Build ensemble
        self.ensemble_model.build_ensemble(lstm_units=128, dropout_rate=0.3)
        
        # Compile models
        self.ensemble_model.compile_models(learning_rate=0.001)
        
        # Train ensemble
        training_history = self.ensemble_model.train_ensemble(
            X_train, y_train, X_test, y_test,
            epochs=epochs, batch_size=32
        )
        
        return training_history
    
    def evaluate_model(self, X_test, y_test) -> dict:
        """
        Evaluate model performance.
        """
        print("📊 Evaluating model performance...")
        
        # Make predictions
        ensemble_pred, individual_preds = self.ensemble_model.predict_ensemble(X_test)
        
        # Calculate metrics
        mse = np.mean((y_test - ensemble_pred) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_test - ensemble_pred))
        mape = np.mean(np.abs((y_test - ensemble_pred) / y_test)) * 100
        
        # Directional accuracy
        actual_direction = np.diff(y_test) > 0
        pred_direction = np.diff(ensemble_pred) > 0
        directional_accuracy = np.mean(actual_direction == pred_direction) * 100
        
        # Calculate returns
        actual_returns = np.diff(y_test) / y_test[:-1]
        pred_returns = np.diff(ensemble_pred) / ensemble_pred[:-1]
        
        # Trading performance
        cumulative_return = np.prod(1 + pred_returns) - 1
        sharpe_ratio = np.mean(pred_returns) / np.std(pred_returns) if np.std(pred_returns) > 0 else 0
        
        results = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'Directional_Accuracy': directional_accuracy,
            'Cumulative_Return': cumulative_return,
            'Sharpe_Ratio': sharpe_ratio,
            'Individual_Predictions': individual_preds
        }
        
        print(f"📈 RMSE: ${rmse:.2f}")
        print(f"📏 MAE: ${mae:.2f}")
        print(f"📊 MAPE: {mape:.2f}%")
        print(f"🎯 Directional Accuracy: {directional_accuracy:.2f}%")
        print(f"💰 Cumulative Return: {cumulative_return:.2%}")
        print(f"📊 Sharpe Ratio: {sharpe_ratio:.3f}")
        
        return results
    
    def save_model_and_data(self, base_path: str = "models"):
        """
        Save model, scaler, and feature information.
        """
        print("💾 Saving model and data...")
        
        # Create directory
        os.makedirs(base_path, exist_ok=True)
        
        # Save ensemble model
        self.ensemble_model.save_ensemble(base_path)
        
        # Save scaler
        scaler_path = os.path.join(base_path, f"{self.symbol}_enhanced_scaler.pkl")
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Save feature information
        feature_info = {
            'feature_names': self.feature_names,
            'n_features': len(self.feature_names),
            'symbol': self.symbol
        }
        feature_path = os.path.join(base_path, f"{self.symbol}_feature_info.pkl")
        with open(feature_path, 'wb') as f:
            pickle.dump(feature_info, f)
        
        print(f"✅ Saved model and data to {base_path}")
    
    def run_complete_training(self, symbol: str = None, period: str = '2y', 
                            sequence_length: int = 60, epochs: int = 100) -> dict:
        """
        Run complete enhanced training pipeline.
        """
        if symbol:
            self.symbol = symbol
        
        print(f"🎯 Starting enhanced training for {self.symbol}")
        print("=" * 60)
        
        # 1. Fetch and prepare data
        df = self.fetch_and_prepare_data(period)
        
        # 2. Generate signals
        signals = self.generate_signals(df)
        
        # 3. Prepare training data
        X_train, X_test, y_train, y_test = self.prepare_training_data(df, sequence_length)
        
        # 4. Train model
        training_history = self.train_enhanced_model(
            X_train, y_train, X_test, y_test, sequence_length, epochs
        )
        
        # 5. Evaluate model
        evaluation_results = self.evaluate_model(X_test, y_test)
        
        # 6. Save model
        self.save_model_and_data()
        
        # 7. Return comprehensive results
        results = {
            'symbol': self.symbol,
            'training_history': training_history,
            'evaluation_results': evaluation_results,
            'signals': signals,
            'feature_count': len(self.feature_names),
            'data_points': len(df)
        }
        
        print("\n" + "=" * 60)
        print("✅ Enhanced training completed successfully!")
        print(f"📊 Final Results for {self.symbol}:")
        print(f"  - RMSE: ${evaluation_results['RMSE']:.2f}")
        print(f"  - MAPE: {evaluation_results['MAPE']:.2f}%")
        print(f"  - Directional Accuracy: {evaluation_results['Directional_Accuracy']:.2f}%")
        print(f"  - Cumulative Return: {evaluation_results['Cumulative_Return']:.2%}")
        print(f"  - Sharpe Ratio: {evaluation_results['Sharpe_Ratio']:.3f}")
        
        return results

def main():
    """
    Main function to run enhanced training.
    """
    # Example usage
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'RELIANCE.NS', 'TCS.NS']
    
    for symbol in symbols:
        try:
            predictor = EnhancedStockPredictor(symbol)
            results = predictor.run_complete_training(
                symbol=symbol,
                period='2y',
                sequence_length=60,
                epochs=50  # Reduced for faster training
            )
            
            print(f"\n🎉 Training completed for {symbol}")
            print("-" * 40)
            
        except Exception as e:
            print(f"❌ Error training {symbol}: {str(e)}")
            continue

if __name__ == "__main__":
    main() 