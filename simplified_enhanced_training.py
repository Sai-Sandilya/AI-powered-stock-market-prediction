"""
Simplified Enhanced Training Script
Working enhanced training with key improvements
"""

import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

class SimplifiedEnhancedPredictor:
    def __init__(self, symbol: str = 'AAPL'):
        self.symbol = symbol
        self.model = None
        self.scaler = MinMaxScaler()
        self.feature_names = []
        
    def fetch_and_prepare_data(self, period: str = '2y') -> pd.DataFrame:
        """
        Fetch and prepare comprehensive dataset with enhanced features.
        """
        print(f"📊 Fetching data for {self.symbol}...")
        
        # Fetch data
        ticker = yf.Ticker(self.symbol)
        df = ticker.history(period=period)
        df.reset_index(inplace=True)
        df.rename(columns={'Date': 'Date'}, inplace=True)
        
        print(f"✅ Fetched {len(df)} records")
        
        # Add enhanced features
        df = self.add_enhanced_features(df)
        
        print(f"🎯 Total features: {len(self.feature_names)}")
        return df
    
    def add_enhanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add enhanced features for better prediction.
        """
        print("🔬 Adding enhanced features...")
        
        # Basic technical indicators
        df['RSI_14'] = self.calculate_rsi(df['Close'], 14)
        df['MACD'], df['MACD_Signal'], df['MACD_Histogram'] = self.calculate_macd(df['Close'])
        df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = self.calculate_bollinger_bands(df['Close'])
        
        # Moving averages
        for period in [5, 10, 20, 50, 200]:
            df[f'MA_{period}'] = df['Close'].rolling(period).mean()
        
        # Volume features
        df['Volume_MA_5'] = df['Volume'].rolling(5).mean()
        df['Volume_MA_20'] = df['Volume'].rolling(20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA_20']
        df['Volume_Spike'] = (df['Volume'] > df['Volume_MA_20'] * 2).astype(int)
        
        # Price-volume relationship
        df['Price_Volume_Trend'] = ((df['Close'] - df['Close'].shift(1)) * df['Volume']).cumsum()
        df['On_Balance_Volume'] = (np.sign(df['Close'].diff()) * df['Volume']).cumsum()
        
        # Advanced momentum indicators
        df['Price_Acceleration'] = df['Close'].diff().diff()
        df['Price_Momentum'] = df['Close'].diff(5)
        df['Price_Acceleration_MA'] = df['Price_Acceleration'].rolling(10).mean()
        
        # Support and Resistance
        df['Support_Level'] = df['Low'].rolling(20).min()
        df['Resistance_Level'] = df['High'].rolling(20).max()
        df['Price_vs_Support'] = (df['Close'] - df['Support_Level']) / df['Close']
        df['Price_vs_Resistance'] = (df['Resistance_Level'] - df['Close']) / df['Close']
        
        # Volatility measures
        df['Realized_Volatility'] = df['Close'].pct_change().rolling(20).std() * np.sqrt(252)
        df['ATR'] = self.calculate_atr(df['High'], df['Low'], df['Close'])
        
        # Statistical features
        for window in [5, 10, 20]:
            df[f'Rolling_Mean_{window}'] = df['Close'].rolling(window).mean()
            df[f'Rolling_Std_{window}'] = df['Close'].rolling(window).std()
            df[f'Z_Score_{window}'] = (df['Close'] - df[f'Rolling_Mean_{window}']) / df[f'Rolling_Std_{window}']
        
        # Mean reversion indicators
        df['Mean_Reversion_5'] = (df['Close'] - df['Close'].rolling(5).mean()) / df['Close'].rolling(5).std()
        df['Mean_Reversion_20'] = (df['Close'] - df['Close'].rolling(20).mean()) / df['Close'].rolling(20).std()
        
        # Pattern recognition
        df['Higher_High'] = (df['High'] > df['High'].shift(1)).astype(int)
        df['Lower_Low'] = (df['Low'] < df['Low'].shift(1)).astype(int)
        df['Higher_Low'] = (df['Low'] > df['Low'].shift(1)).astype(int)
        df['Lower_High'] = (df['High'] < df['High'].shift(1)).astype(int)
        
        # Gap analysis
        df['Gap_Up'] = (df['Open'] > df['Close'].shift(1)).astype(int)
        df['Gap_Down'] = (df['Open'] < df['Close'].shift(1)).astype(int)
        df['Gap_Size'] = abs(df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
        
        # Lag features
        for lag in range(1, 6):
            df[f'Price_Lag_{lag}'] = df['Close'].shift(lag)
            df[f'Volume_Lag_{lag}'] = df['Volume'].shift(lag)
            df[f'Returns_Lag_{lag}'] = df['Close'].pct_change().shift(lag)
        
        # Interaction features
        df['RSI_MACD_Interaction'] = df['RSI_14'] * df['MACD']
        df['Bollinger_RSI_Interaction'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower']) * df['RSI_14']
        df['Price_Volume_Correlation'] = df['Close'].rolling(20).corr(df['Volume'])
        
        # Market regime features (numeric instead of categorical)
        vol_quantile_75 = df['Realized_Volatility'].rolling(50).quantile(0.75)
        vol_quantile_25 = df['Realized_Volatility'].rolling(50).quantile(0.25)
        df['Volatility_Regime_Score'] = np.where(df['Realized_Volatility'] > vol_quantile_75, 2,
                                                np.where(df['Realized_Volatility'] < vol_quantile_25, 0, 1))
        
        trend_threshold = df['Close'].rolling(50).mean()
        df['Trend_Regime_Score'] = np.where(df['Close'] > trend_threshold * 1.02, 1,
                                           np.where(df['Close'] < trend_threshold * 0.98, -1, 0))
        
        # Clean up NaN values
        df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # Store feature names (only numeric columns)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        self.feature_names = [col for col in numeric_cols if col not in ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        
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
    
    def calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(period).mean()
    
    def prepare_training_data(self, df: pd.DataFrame, sequence_length: int = 60) -> tuple:
        """
        Prepare data for LSTM training.
        """
        print("🔧 Preparing training data...")
        
        # Select features (exclude Date and target column)
        feature_cols = [col for col in df.columns if col not in ['Date', 'Close']]
        
        # Scale features
        scaled_data = self.scaler.fit_transform(df[feature_cols])
        
        # Create sequences
        X, y = [], []
        for i in range(sequence_length, len(scaled_data)):
            X.append(scaled_data[i-sequence_length:i])
            y.append(df['Close'].iloc[i])
        
        X = np.array(X)
        y = np.array(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )
        
        self.feature_names = feature_cols
        
        print(f"📊 Training data shape: {X_train.shape}")
        print(f"📊 Test data shape: {X_test.shape}")
        print(f"🎯 Features used: {len(feature_cols)}")
        
        return X_train, X_test, y_train, y_test
    
    def build_enhanced_lstm(self, sequence_length: int, n_features: int, lstm_units: int = 128) -> keras.Model:
        """
        Build enhanced LSTM model with attention and residual connections.
        """
        inputs = layers.Input(shape=(sequence_length, n_features))
        
        # First LSTM layer
        lstm1 = layers.LSTM(lstm_units, return_sequences=True, activation='tanh')(inputs)
        lstm1 = layers.BatchNormalization()(lstm1)
        lstm1 = layers.Dropout(0.3)(lstm1)
        
        # Second LSTM layer with residual connection
        lstm2 = layers.LSTM(lstm_units, return_sequences=True, activation='tanh')(lstm1)
        lstm2 = layers.BatchNormalization()(lstm2)
        lstm2 = layers.Dropout(0.3)(lstm2)
        
        # Residual connection
        if lstm2.shape[-1] == lstm1.shape[-1]:
            lstm2 = layers.Add()([lstm2, lstm1])
        
        # Third LSTM layer
        lstm3 = layers.LSTM(lstm_units // 2, return_sequences=False, activation='tanh')(lstm2)
        lstm3 = layers.BatchNormalization()(lstm3)
        lstm3 = layers.Dropout(0.3)(lstm3)
        
        # Dense layers
        dense1 = layers.Dense(lstm_units, activation='relu')(lstm3)
        dense1 = layers.BatchNormalization()(dense1)
        dense1 = layers.Dropout(0.3)(dense1)
        
        dense2 = layers.Dense(lstm_units // 2, activation='relu')(dense1)
        dense2 = layers.BatchNormalization()(dense2)
        dense2 = layers.Dropout(0.3)(dense2)
        
        # Output layer
        outputs = layers.Dense(1, activation='linear')(dense2)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        return model
    
    def train_model(self, X_train, y_train, X_test, y_test, 
                   sequence_length: int = 60, epochs: int = 100) -> dict:
        """
        Train enhanced LSTM model.
        """
        print("🚀 Training enhanced LSTM model...")
        
        # Build model
        self.model = self.build_enhanced_lstm(
            sequence_length=sequence_length,
            n_features=X_train.shape[2],
            lstm_units=128
        )
        
        # Compile model
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        # Callbacks
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True
        )
        
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-6
        )
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=32,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        return history.history
    
    def evaluate_model(self, X_test, y_test) -> dict:
        """
        Evaluate model performance.
        """
        print("📊 Evaluating model performance...")
        
        # Make predictions
        predictions = self.model.predict(X_test, verbose=0).flatten()
        
        # Calculate metrics
        mse = np.mean((y_test - predictions) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_test - predictions))
        mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
        
        # Directional accuracy
        actual_direction = np.diff(y_test) > 0
        pred_direction = np.diff(predictions) > 0
        directional_accuracy = np.mean(actual_direction == pred_direction) * 100
        
        # Calculate returns
        actual_returns = np.diff(y_test) / y_test[:-1]
        pred_returns = np.diff(predictions) / predictions[:-1]
        
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
            'Predictions': predictions
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
        
        # Save model
        model_path = os.path.join(base_path, f"{self.symbol}_enhanced_model.h5")
        self.model.save(model_path)
        
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
        feature_path = os.path.join(base_path, f"{self.symbol}_enhanced_feature_info.pkl")
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
        
        # 2. Prepare training data
        X_train, X_test, y_train, y_test = self.prepare_training_data(df, sequence_length)
        
        # 3. Train model
        training_history = self.train_model(
            X_train, y_train, X_test, y_test, sequence_length, epochs
        )
        
        # 4. Evaluate model
        evaluation_results = self.evaluate_model(X_test, y_test)
        
        # 5. Save model
        self.save_model_and_data()
        
        # 6. Return comprehensive results
        results = {
            'symbol': self.symbol,
            'training_history': training_history,
            'evaluation_results': evaluation_results,
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
    symbols = ['AAPL', 'MSFT', 'RELIANCE.NS']
    
    for symbol in symbols:
        try:
            predictor = SimplifiedEnhancedPredictor(symbol)
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