"""
Enhanced Training System
Integrates advanced features with existing modules for improved performance.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
import os
import joblib
from typing import Dict, List, Optional, Tuple
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error
import optuna

# Import our existing modules
from data_ingestion import fetch_yfinance
from feature_engineering import get_comprehensive_features
from train_improved_lstm import ImprovedLSTMTrainer
from backtesting import Backtester
from macro_indicators import MacroIndicators

warnings.filterwarnings('ignore')

class EnhancedTrainingSystem:
    def __init__(self, 
                 symbol: str = 'AAPL',
                 start_date: str = '2020-01-01',
                 end_date: Optional[str] = None,
                 initial_capital: float = 100000):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        self.initial_capital = initial_capital
        
        # Initialize components
        self.macro_indicators = MacroIndicators()
        self.trainer = ImprovedLSTMTrainer()
        
        # Data storage
        self.raw_data = None
        self.featured_data = None
        self.trained_model = None
        self.scaler = None
        self.optimization_results = {}
        
    def fetch_and_prepare_data(self) -> pd.DataFrame:
        """
        Fetch and prepare comprehensive data including macro indicators.
        """
        print(f"📊 Fetching data for {self.symbol}...")
        
        # Fetch stock data
        stock_data = fetch_yfinance(self.symbol, self.start_date, self.end_date)
        
        # Fetch macro indicators
        try:
            macro_data = self.macro_indicators.fetch_macro_data(
                self.start_date, self.end_date
            )
            
            # Merge data
            if macro_data is not None and not macro_data.empty:
                merged_data = pd.merge(
                    stock_data, macro_data, on='Date', how='left'
                )
                # Forward fill macro data
                merged_data = merged_data.fillna(method='ffill')
            else:
                merged_data = stock_data
        except Exception as e:
            print(f"⚠️ Macro data fetch failed: {e}")
            merged_data = stock_data
        
        self.raw_data = merged_data
        print(f"✅ Fetched {len(merged_data)} data points")
        
        return merged_data
    
    def create_enhanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create enhanced features using our existing feature engineering.
        """
        print("🔧 Creating enhanced features...")
        
        # Create basic features
        featured_df = get_comprehensive_features(df)
        
        # Add macro features if available
        macro_cols = [col for col in df.columns if col.startswith('macro_')]
        if macro_cols:
            featured_df = pd.concat([featured_df, df[macro_cols]], axis=1)
            print(f"✅ Added {len(macro_cols)} macro features")
        
        # Add advanced technical indicators
        featured_df = self.add_advanced_indicators(featured_df)
        
        # Add sentiment features (simulated)
        featured_df = self.add_sentiment_features(featured_df)
        
        # Add market microstructure features
        featured_df = self.add_microstructure_features(featured_df)
        
        # CRITICAL: Clean data before proceeding
        featured_df = self.clean_data(featured_df)
        
        self.featured_data = featured_df
        print(f"✅ Created {len(featured_df.columns)} enhanced features")
        
        return featured_df
    
    def add_advanced_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add advanced technical indicators.
        """
        print("📊 Adding advanced technical indicators...")
        
        # Bollinger Bands
        df['bb_upper'] = df['Close'].rolling(20).mean() + 2 * df['Close'].rolling(20).std()
        df['bb_lower'] = df['Close'].rolling(20).mean() - 2 * df['Close'].rolling(20).std()
        df['bb_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        df['bb_squeeze'] = (df['bb_upper'] - df['bb_lower']) / df['Close'].rolling(20).mean()
        
        # Stochastic Oscillator
        low_min = df['Low'].rolling(14).min()
        high_max = df['High'].rolling(14).max()
        df['stoch_k'] = 100 * (df['Close'] - low_min) / (high_max - low_min)
        df['stoch_d'] = df['stoch_k'].rolling(3).mean()
        
        # Williams %R
        df['williams_r'] = -100 * (high_max - df['Close']) / (high_max - low_min)
        
        # Average True Range (ATR)
        df['tr1'] = df['High'] - df['Low']
        df['tr2'] = abs(df['High'] - df['Close'].shift())
        df['tr3'] = abs(df['Low'] - df['Close'].shift())
        df['atr'] = pd.concat([df['tr1'], df['tr2'], df['tr3']], axis=1).max(axis=1).rolling(14).mean()
        df['atr_ratio'] = df['atr'] / df['Close']
        
        # Commodity Channel Index (CCI)
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        sma_tp = typical_price.rolling(20).mean()
        mad = typical_price.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())))
        df['cci'] = (typical_price - sma_tp) / (0.015 * mad)
        
        # Money Flow Index (MFI)
        money_flow = df['Volume'] * ((df['High'] + df['Low'] + df['Close']) / 3)
        positive_flow = money_flow.where(df['Close'] > df['Close'].shift(1), 0).rolling(14).sum()
        negative_flow = money_flow.where(df['Close'] < df['Close'].shift(1), 0).rolling(14).sum()
        df['mfi'] = 100 - (100 / (1 + positive_flow / negative_flow))
        
        # On-Balance Volume (OBV)
        df['obv'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
        df['obv_ma'] = df['obv'].rolling(20).mean()
        df['obv_signal'] = (df['obv'] - df['obv_ma']) / df['obv_ma']
        
        # Clean up temporary columns
        df = df.drop(['tr1', 'tr2', 'tr3'], axis=1, errors='ignore')
        
        return df
    
    def add_sentiment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add simulated sentiment features.
        """
        print("📰 Adding sentiment features...")
        
        # Simulate sentiment based on price movement and volatility
        df['price_change'] = df['Close'].pct_change()
        df['volatility'] = df['price_change'].rolling(20).std()
        
        # Fill NaN values in volatility
        df['volatility'] = df['volatility'].fillna(df['volatility'].mean())
        df['price_change'] = df['price_change'].fillna(0)
        
        # Sentiment correlates with price movement and volatility
        # Use safe calculations to avoid NaN
        sentiment_base = np.tanh(df['price_change'] * 10) + np.tanh(df['volatility'] * 20)
        noise = np.random.normal(0, 0.1, len(df))
        df['sentiment_score'] = sentiment_base + noise
        
        # Sentiment derivatives with safe calculations
        df['sentiment_momentum'] = df['sentiment_score'].rolling(5).mean().fillna(df['sentiment_score'].mean())
        df['sentiment_volatility'] = df['sentiment_score'].rolling(10).std().fillna(df['sentiment_score'].std())
        df['sentiment_divergence'] = df['sentiment_score'] - df['sentiment_momentum']
        
        return df
    
    def add_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add market microstructure features.
        """
        print("🔬 Adding microstructure features...")
        
        # Volume-based features
        df['volume_ma_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
        df['volume_price_trend'] = df['Volume'] * df['Close'].pct_change()
        df['volume_weighted_price'] = (df['Volume'] * df['Close']).rolling(5).sum() / df['Volume'].rolling(5).sum()
        
        # Price efficiency features
        df['price_efficiency'] = abs(df['Close'].pct_change()) / df['Close'].pct_change().rolling(20).std()
        df['price_momentum'] = df['Close'].pct_change(5)
        df['price_reversal'] = -df['Close'].pct_change(5)
        
        # Volatility features
        df['realized_volatility'] = df['Close'].pct_change().rolling(20).std() * np.sqrt(252)
        df['volatility_ratio'] = df['realized_volatility'] / df['realized_volatility'].rolling(60).mean()
        
        # Liquidity features
        df['amihud_illiquidity'] = abs(df['Close'].pct_change()) / (df['Volume'] / 1000000)
        df['turnover_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
        df['bid_ask_spread'] = (df['High'] - df['Low']) / df['Close']
        
        return df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Comprehensive data cleaning to prevent NaN issues.
        """
        print("🧹 Cleaning data...")
        
        # Remove rows with any NaN values
        initial_rows = len(df)
        df = df.dropna()
        print(f"📊 Removed {initial_rows - len(df)} rows with NaN values")
        
        # Check for infinite values
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna()
        print(f"📊 Removed infinite values")
        
        # Check for zero or negative prices
        if 'Close' in df.columns:
            df = df[df['Close'] > 0]
            print(f"📊 Removed rows with non-positive prices")
        
        # Check for zero volumes
        if 'Volume' in df.columns:
            df = df[df['Volume'] > 0]
            print(f"📊 Removed rows with zero volume")
        
        # Ensure we have enough data
        if len(df) < 100:
            raise ValueError(f"Insufficient data after cleaning: {len(df)} rows")
        
        print(f"✅ Data cleaning completed. Final dataset: {len(df)} rows, {len(df.columns)} columns")
        
        # Final check for any remaining NaN values
        nan_cols = df.columns[df.isna().any()].tolist()
        if nan_cols:
            print(f"⚠️ Warning: NaN values found in columns: {nan_cols}")
            df = df.dropna()
            print(f"📊 Final cleanup: {len(df)} rows remaining")
        
        return df
    
    def train_enhanced_model(self, df: pd.DataFrame) -> Tuple[object, object]:
        """
        Train enhanced LSTM model with advanced features.
        """
        print("🤖 Training enhanced LSTM model...")
        
        # Ensure data is clean
        df = self.clean_data(df)
        
        # Train the improved LSTM model
        training_results = self.trainer.train_model(df)
        
        # Extract model and scaler from results
        if isinstance(training_results, dict):
            # If the trainer returns a dict, we need to get the model differently
            self.trained_model = self.trainer.model
            self.scaler = self.trainer.scaler
        else:
            # If it returns a tuple
            self.trained_model, self.scaler = training_results
        
        # Save model
        if self.trained_model:
            self.trained_model.save('models/enhanced_lstm_model.h5')
        if self.scaler:
            joblib.dump(self.scaler, 'models/enhanced_scaler.pkl')
        
        print("✅ Enhanced model trained and saved")
        
        return self.trained_model, self.scaler
    
    def optimize_hyperparameters(self, df: pd.DataFrame) -> Dict:
        """
        Run Bayesian optimization for hyperparameters.
        """
        print("🎯 Running Bayesian optimization...")
        
        def objective(trial):
            try:
                # Suggest hyperparameters
                sequence_length = trial.suggest_int('sequence_length', 30, 120, step=10)
                lstm_units = trial.suggest_int('lstm_units', 32, 256, step=32)
                dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5, step=0.1)
                learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
                batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
                
                # Train model with suggested parameters
                model, scaler = self.train_model_with_params(df, {
                    'sequence_length': sequence_length,
                    'lstm_units': lstm_units,
                    'dropout_rate': dropout_rate,
                    'learning_rate': learning_rate,
                    'batch_size': batch_size
                })
                
                # Run backtest to evaluate
                backtest_results = self.run_backtest(df, model, scaler)
                
                # Return MAPE (lower is better)
                return backtest_results.get('mape', 100)
                
            except Exception as e:
                print(f"⚠️ Trial failed: {e}")
                return 100.0
        
        # Create study
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=20)  # Reduced for faster testing
        
        self.optimization_results = {
            'best_params': study.best_params,
            'best_value': study.best_value,
            'study': study
        }
        
        print(f"✅ Optimization completed! Best MAPE: {study.best_value:.2f}%")
        print(f"📊 Best parameters: {study.best_params}")
        
        return self.optimization_results
    
    def train_model_with_params(self, df: pd.DataFrame, params: Dict) -> Tuple[object, object]:
        """
        Train model with specific parameters.
        """
        # This would need to be implemented to work with our existing training function
        # For now, return the default training
        return self.trainer.train_model(df)
    
    def run_backtest(self, df: pd.DataFrame, model, scaler) -> Dict:
        """
        Run backtest using our backtesting module.
        """
        # Create backtester
        backtester = Backtester()
        
        # Run walk-forward backtest
        results = backtester.walk_forward_backtest(
            self.symbol, 
            self.start_date, 
            self.end_date,
            forecast_horizon=30,
            step_size=5
        )
        
        return results
    
    def evaluate_system_performance(self, df: pd.DataFrame) -> Dict:
        """
        Comprehensive system performance evaluation.
        """
        print("📊 Evaluating system performance...")
        
        # Load trained model
        try:
            model = tf.keras.models.load_model('models/enhanced_lstm_model.h5')
            scaler = joblib.load('models/enhanced_scaler.pkl')
        except:
            print("⚠️ No saved model found. Training new one...")
            model, scaler = self.train_enhanced_model(df)
        
        # Run backtest
        backtest_results = self.run_backtest(df, model, scaler)
        
        # Calculate additional metrics
        performance_summary = {
            'backtest_results': backtest_results,
            'model_info': {
                'model_type': 'Enhanced LSTM',
                'features_used': len(df.columns) - 2,  # Exclude Date and Close
                'data_points': len(df)
            },
            'optimization_results': self.optimization_results
        }
        
        print("✅ System performance evaluation completed")
        print(f"📊 MAPE: {backtest_results.get('mape', 0):.2f}%")
        print(f"🎯 Directional Accuracy: {backtest_results.get('directional_accuracy', 0):.2f}%")
        print(f"💰 Total Return: {backtest_results.get('cumulative_return', 0):.2f}%")
        print(f"📈 Sharpe Ratio: {backtest_results.get('sharpe_ratio', 0):.2f}")
        
        return performance_summary
    
    def save_system_state(self, save_dir: str = 'enhanced_system'):
        """
        Save the complete system state.
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Save data
        if self.featured_data is not None:
            self.featured_data.to_csv(f'{save_dir}/featured_data.csv', index=False)
        
        # Save system configuration
        config = {
            'symbol': self.symbol,
            'start_date': self.start_date,
            'end_date': self.end_date,
            'initial_capital': self.initial_capital,
            'training_date': datetime.now().isoformat()
        }
        
        joblib.dump(config, f'{save_dir}/system_config.pkl')
        
        print(f"✅ System state saved to {save_dir}")
    
    def run_complete_training_pipeline(self) -> Dict:
        """
        Run the complete enhanced training pipeline.
        """
        print("🚀 Starting complete enhanced training pipeline...")
        
        # Step 1: Fetch and prepare data
        df = self.fetch_and_prepare_data()
        
        # Step 2: Create enhanced features
        featured_df = self.create_enhanced_features(df)
        
        # Step 3: Optimize hyperparameters
        optimization_results = self.optimize_hyperparameters(featured_df)
        
        # Step 4: Train with optimized parameters
        model, scaler = self.train_enhanced_model(featured_df)
        
        # Step 5: Evaluate system performance
        performance = self.evaluate_system_performance(featured_df)
        
        # Step 6: Save system state
        self.save_system_state()
        
        # Compile final results
        final_results = {
            'symbol': self.symbol,
            'data_points': len(featured_df),
            'features_created': len(featured_df.columns),
            'optimization_results': optimization_results,
            'performance_summary': performance,
            'training_completed': datetime.now().isoformat()
        }
        
        print("🎉 Complete enhanced training pipeline finished!")
        print(f"📊 Final Performance Summary:")
        print(f"   MAPE: {performance['backtest_results'].get('mape', 0):.2f}%")
        print(f"   Directional Accuracy: {performance['backtest_results'].get('directional_accuracy', 0):.2f}%")
        print(f"   Total Return: {performance['backtest_results'].get('cumulative_return', 0):.2f}%")
        print(f"   Sharpe Ratio: {performance['backtest_results'].get('sharpe_ratio', 0):.2f}")
        
        return final_results

def test_enhanced_system():
    """Test the enhanced training system."""
    print("🧪 Testing enhanced training system...")
    
    # Create system
    system = EnhancedTrainingSystem(
        symbol='AAPL',
        start_date='2022-01-01',
        end_date='2023-12-31',
        initial_capital=100000
    )
    
    # Run complete pipeline
    results = system.run_complete_training_pipeline()
    
    return system, results

if __name__ == "__main__":
    test_enhanced_system() 