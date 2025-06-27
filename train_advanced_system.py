"""
Advanced Training System
Integrates all advanced enhancements: ensemble methods, advanced features,
trading strategies, and Bayesian optimization for superior performance.
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

# Import our advanced modules
from advanced_ensemble import AdvancedEnsemble
from advanced_feature_engineering import AdvancedFeatureEngineer
from advanced_trading_strategy import AdvancedTradingStrategy
from bayesian_optimization import BayesianOptimizer
from data_ingestion import DataIngestion
from macro_indicators import MacroIndicators

warnings.filterwarnings('ignore')

class AdvancedTrainingSystem:
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
        self.data_ingestion = DataIngestion()
        self.feature_engineer = AdvancedFeatureEngineer()
        self.ensemble = AdvancedEnsemble()
        self.trading_strategy = AdvancedTradingStrategy(initial_capital=initial_capital)
        self.optimizer = BayesianOptimizer(n_trials=50)  # Reduced for faster testing
        self.macro_indicators = MacroIndicators()
        
        # Data storage
        self.raw_data = None
        self.featured_data = None
        self.trained_models = {}
        self.optimization_results = {}
        
    def fetch_and_prepare_data(self) -> pd.DataFrame:
        """
        Fetch and prepare comprehensive data including macro indicators.
        """
        print(f"📊 Fetching data for {self.symbol}...")
        
        # Fetch stock data
        stock_data = self.data_ingestion.fetch_stock_data(
            self.symbol, self.start_date, self.end_date
        )
        
        # Fetch macro indicators
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
        
        self.raw_data = merged_data
        print(f"✅ Fetched {len(merged_data)} data points")
        
        return merged_data
    
    def create_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create comprehensive advanced features.
        """
        print("🔧 Creating advanced features...")
        
        # Create advanced features
        featured_df = self.feature_engineer.create_advanced_features(df, self.symbol)
        
        self.featured_data = featured_df
        print(f"✅ Created {len(featured_df.columns)} features")
        
        return featured_df
    
    def train_ensemble_models(self, df: pd.DataFrame, params: Optional[Dict] = None) -> Dict:
        """
        Train the advanced ensemble models.
        """
        print("🤖 Training advanced ensemble models...")
        
        # Use provided parameters or defaults
        if params is None:
            params = {
                'sequence_length': 60,
                'n_features': len(df.columns) - 2,  # Exclude Date and Close
                'lstm_units_1': 128,
                'lstm_units_2': 64,
                'dropout_rate': 0.3,
                'learning_rate': 0.001,
                'batch_size': 32
            }
        
        # Train ensemble
        self.ensemble.train_ensemble(df)
        
        # Save models
        self.ensemble.save_ensemble('models/advanced_ensemble')
        
        self.trained_models['ensemble'] = self.ensemble
        print("✅ Ensemble models trained and saved")
        
        return {'ensemble': self.ensemble}
    
    def run_trading_backtest(self, df: pd.DataFrame, params: Optional[Dict] = None) -> Dict:
        """
        Run advanced trading strategy backtest.
        """
        print("📈 Running advanced trading backtest...")
        
        # Use provided parameters or defaults
        if params is None:
            params = {
                'max_position_size': 0.1,
                'risk_per_trade': 0.02,
                'stop_loss_pct': 0.03,
                'take_profit_pct': 0.06
            }
        
        # Update strategy parameters
        self.trading_strategy.max_position_size = params.get('max_position_size', 0.1)
        self.trading_strategy.risk_per_trade = params.get('risk_per_trade', 0.02)
        
        # Run backtest
        performance = self.trading_strategy.run_backtest(df, self.symbol)
        
        return performance
    
    def optimize_hyperparameters(self, df: pd.DataFrame) -> Dict:
        """
        Run Bayesian optimization for hyperparameters.
        """
        print("🎯 Running Bayesian optimization...")
        
        def model_trainer(data, params):
            """Model trainer function for optimization."""
            try:
                # Create ensemble with parameters
                ensemble = AdvancedEnsemble(
                    sequence_length=params.get('sequence_length', 60),
                    n_features=len(data.columns) - 2
                )
                
                # Train ensemble
                ensemble.train_ensemble(data)
                
                return ensemble
            except Exception as e:
                print(f"⚠️ Model training failed: {e}")
                return None
        
        def backtester(data, model, params):
            """Backtester function for optimization."""
            try:
                if model is None:
                    return {'mape': 100, 'total_return': -0.5, 'sharpe_ratio': 0}
                
                # Create trading strategy with parameters
                strategy = AdvancedTradingStrategy(
                    initial_capital=self.initial_capital,
                    max_position_size=params.get('max_position_size', 0.1),
                    risk_per_trade=params.get('risk_per_trade', 0.02)
                )
                
                # Run backtest
                performance = strategy.run_backtest(data, self.symbol)
                
                return {
                    'mape': performance.get('mape', 100),
                    'total_return': performance.get('total_return', 0),
                    'sharpe_ratio': performance.get('sharpe_ratio', 0)
                }
            except Exception as e:
                print(f"⚠️ Backtesting failed: {e}")
                return {'mape': 100, 'total_return': -0.5, 'sharpe_ratio': 0}
        
        # Run optimization
        optimization_results = self.optimizer.optimize_single_objective(
            df, model_trainer, backtester, 'accuracy'
        )
        
        self.optimization_results = optimization_results
        print("✅ Hyperparameter optimization completed")
        
        return optimization_results
    
    def train_with_optimized_params(self, df: pd.DataFrame) -> Dict:
        """
        Train models with optimized hyperparameters.
        """
        print("🚀 Training with optimized parameters...")
        
        if not self.optimization_results:
            print("⚠️ No optimization results found. Using default parameters.")
            return self.train_ensemble_models(df)
        
        best_params = self.optimization_results.get('best_params', {})
        
        # Train ensemble with optimized parameters
        ensemble = self.train_ensemble_models(df, best_params)
        
        # Run trading backtest with optimized parameters
        trading_performance = self.run_trading_backtest(df, best_params)
        
        return {
            'ensemble': ensemble,
            'trading_performance': trading_performance,
            'optimized_params': best_params
        }
    
    def evaluate_system_performance(self, df: pd.DataFrame) -> Dict:
        """
        Comprehensive system performance evaluation.
        """
        print("📊 Evaluating system performance...")
        
        # Load trained ensemble
        try:
            self.ensemble.load_ensemble('models/advanced_ensemble')
        except:
            print("⚠️ No saved ensemble found. Training new one...")
            self.train_ensemble_models(df)
        
        # Make predictions
        predictions = self.ensemble.predict_ensemble(df)
        
        if not predictions:
            return {'error': 'No predictions generated'}
        
        # Calculate prediction metrics
        actual_prices = df['Close'].values[-len(predictions['ensemble_prediction']):]
        pred_prices = predictions['ensemble_prediction']
        
        mse = mean_squared_error(actual_prices, pred_prices)
        mae = mean_absolute_error(actual_prices, pred_prices)
        mape = np.mean(np.abs((actual_prices - pred_prices) / actual_prices)) * 100
        
        # Directional accuracy
        actual_direction = np.diff(actual_prices) > 0
        pred_direction = np.diff(pred_prices) > 0
        directional_accuracy = np.mean(actual_direction == pred_direction) * 100
        
        # Trading performance
        trading_performance = self.run_trading_backtest(df)
        
        # Model confidence analysis
        confidence_stats = {
            'mean_confidence': np.mean(predictions['ensemble_confidence']),
            'confidence_std': np.std(predictions['ensemble_confidence']),
            'high_confidence_pct': np.mean(predictions['ensemble_confidence'] > 0.7) * 100
        }
        
        # Model weight analysis
        weight_stats = {}
        if 'model_weights' in predictions:
            for model, weight in predictions['model_weights'].items():
                weight_stats[f'{model}_weight'] = weight
        
        performance_summary = {
            'prediction_metrics': {
                'mse': mse,
                'mae': mae,
                'mape': mape,
                'directional_accuracy': directional_accuracy
            },
            'trading_performance': trading_performance,
            'confidence_stats': confidence_stats,
            'model_weights': weight_stats,
            'ensemble_confidence': predictions['ensemble_confidence'],
            'prediction_uncertainty': np.mean(predictions['prediction_std'])
        }
        
        print("✅ System performance evaluation completed")
        print(f"📊 MAPE: {mape:.2f}%")
        print(f"🎯 Directional Accuracy: {directional_accuracy:.2f}%")
        print(f"💰 Total Return: {trading_performance.get('total_return', 0):.2%}")
        print(f"📈 Sharpe Ratio: {trading_performance.get('sharpe_ratio', 0):.2f}")
        
        return performance_summary
    
    def save_system_state(self, save_dir: str = 'advanced_system'):
        """
        Save the complete system state.
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Save data
        if self.featured_data is not None:
            self.featured_data.to_csv(f'{save_dir}/featured_data.csv', index=False)
        
        # Save optimization results
        if self.optimization_results:
            self.optimizer.save_optimization_results(
                self.optimization_results, f'{save_dir}/optimization'
            )
        
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
    
    def load_system_state(self, save_dir: str = 'advanced_system'):
        """
        Load the complete system state.
        """
        # Load data
        data_path = f'{save_dir}/featured_data.csv'
        if os.path.exists(data_path):
            self.featured_data = pd.read_csv(data_path)
            self.featured_data['Date'] = pd.to_datetime(self.featured_data['Date'])
        
        # Load optimization results
        opt_dir = f'{save_dir}/optimization'
        if os.path.exists(opt_dir):
            self.optimization_results = self.optimizer.load_optimization_results(opt_dir)
        
        # Load system configuration
        config_path = f'{save_dir}/system_config.pkl'
        if os.path.exists(config_path):
            config = joblib.load(config_path)
            self.symbol = config.get('symbol', self.symbol)
            self.start_date = config.get('start_date', self.start_date)
            self.end_date = config.get('end_date', self.end_date)
            self.initial_capital = config.get('initial_capital', self.initial_capital)
        
        print(f"✅ System state loaded from {save_dir}")
    
    def run_complete_training_pipeline(self) -> Dict:
        """
        Run the complete advanced training pipeline.
        """
        print("🚀 Starting complete advanced training pipeline...")
        
        # Step 1: Fetch and prepare data
        df = self.fetch_and_prepare_data()
        
        # Step 2: Create advanced features
        featured_df = self.create_advanced_features(df)
        
        # Step 3: Optimize hyperparameters
        optimization_results = self.optimize_hyperparameters(featured_df)
        
        # Step 4: Train with optimized parameters
        training_results = self.train_with_optimized_params(featured_df)
        
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
            'training_results': training_results,
            'performance_summary': performance,
            'training_completed': datetime.now().isoformat()
        }
        
        print("🎉 Complete training pipeline finished!")
        print(f"📊 Final Performance Summary:")
        print(f"   MAPE: {performance['prediction_metrics']['mape']:.2f}%")
        print(f"   Directional Accuracy: {performance['prediction_metrics']['directional_accuracy']:.2f}%")
        print(f"   Total Return: {performance['trading_performance'].get('total_return', 0):.2%}")
        print(f"   Sharpe Ratio: {performance['trading_performance'].get('sharpe_ratio', 0):.2f}")
        
        return final_results

def test_advanced_system():
    """Test the advanced training system."""
    print("🧪 Testing advanced training system...")
    
    # Create system
    system = AdvancedTrainingSystem(
        symbol='AAPL',
        start_date='2022-01-01',
        end_date='2023-12-31',
        initial_capital=100000
    )
    
    # Run complete pipeline
    results = system.run_complete_training_pipeline()
    
    return system, results

if __name__ == "__main__":
    test_advanced_system() 