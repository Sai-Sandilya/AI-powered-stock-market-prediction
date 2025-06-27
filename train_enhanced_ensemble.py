"""
Enhanced Ensemble Training Script
Trains an ensemble of models with advanced features, hyperparameter optimization,
and improved trading strategies for superior stock prediction performance.
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
import optuna

# Import our modules
from data_ingestion import fetch_yfinance
from enhanced_feature_engineering import EnhancedFeatureEngineer
from ensemble_model import EnsembleModel

warnings.filterwarnings('ignore')

class EnhancedEnsembleTrainer:
    def __init__(self, symbol: str = 'AAPL'):
        self.symbol = symbol
        self.feature_engineer = EnhancedFeatureEngineer()
        self.ensemble = None
        self.best_params = {}
        
    def fetch_and_prepare_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch and prepare data with enhanced features.
        """
        print(f"📊 Fetching data for {self.symbol}...")
        
        # Fetch data
        df = fetch_yfinance(self.symbol, start_date, end_date)
        if df.empty:
            raise ValueError(f"No data available for {self.symbol}")
        
        print(f"✅ Fetched {len(df)} records")
        
        # Create enhanced features
        df = self.feature_engineer.create_enhanced_features(df, self.symbol)
        
        # Remove rows with NaN values
        df = df.dropna()
        
        print(f"✅ Prepared data with {len(df.columns)} features")
        return df
    
    def objective(self, trial: optuna.Trial, df: pd.DataFrame) -> float:
        """
        Optuna objective function for hyperparameter optimization.
        """
        # Suggest hyperparameters
        sequence_length = trial.suggest_int('sequence_length', 30, 120, step=10)
        lstm_units_1 = trial.suggest_int('lstm_units_1', 50, 200, step=25)
        lstm_units_2 = trial.suggest_int('lstm_units_2', 25, 100, step=25)
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5, step=0.1)
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
        epochs = trial.suggest_int('epochs', 20, 100, step=10)
        
        try:
            # Create ensemble with suggested parameters
            n_features = len([col for col in df.columns if col not in ['Date', 'Close']])
            ensemble = EnsembleModel(sequence_length=sequence_length, n_features=n_features)
            
            # Train with cross-validation
            tscv = TimeSeriesSplit(n_splits=3)
            cv_scores = []
            
            for train_idx, val_idx in tscv.split(df):
                train_df = df.iloc[train_idx]
                val_df = df.iloc[val_idx]
                
                # Train ensemble
                ensemble.train_ensemble(train_df, validation_split=0.2)
                
                # Make predictions
                predictions = ensemble.predict_ensemble(val_df)
                
                if predictions:
                    # Calculate metrics
                    actual = val_df['Close'].values[sequence_length:]
                    predicted = predictions['ensemble_prediction']
                    
                    if len(actual) == len(predicted):
                        mse = mean_squared_error(actual, predicted)
                        cv_scores.append(mse)
            
            # Return mean CV score
            if cv_scores:
                return np.mean(cv_scores)
            else:
                return float('inf')
                
        except Exception as e:
            print(f"⚠️ Trial failed: {e}")
            return float('inf')
    
    def optimize_hyperparameters(self, df: pd.DataFrame, n_trials: int = 50) -> Dict:
        """
        Optimize hyperparameters using Optuna.
        """
        print(f"🔍 Optimizing hyperparameters with {n_trials} trials...")
        
        study = optuna.create_study(direction='minimize')
        study.optimize(lambda trial: self.objective(trial, df), n_trials=n_trials)
        
        self.best_params = study.best_params
        print(f"✅ Best parameters: {self.best_params}")
        print(f"✅ Best CV score: {study.best_value:.4f}")
        
        return self.best_params
    
    def train_final_ensemble(self, df: pd.DataFrame) -> EnsembleModel:
        """
        Train the final ensemble with optimized parameters.
        """
        print("🚀 Training final ensemble model...")
        
        # Use best parameters if available
        if self.best_params:
            sequence_length = self.best_params.get('sequence_length', 60)
        else:
            sequence_length = 60
        
        n_features = len([col for col in df.columns if col not in ['Date', 'Close']])
        self.ensemble = EnsembleModel(sequence_length=sequence_length, n_features=n_features)
        
        # Train ensemble
        self.ensemble.train_ensemble(df, validation_split=0.2)
        
        print("✅ Final ensemble training completed!")
        return self.ensemble
    
    def evaluate_ensemble(self, df: pd.DataFrame) -> Dict:
        """
        Evaluate the trained ensemble.
        """
        print("📊 Evaluating ensemble performance...")
        
        if not self.ensemble:
            raise ValueError("No ensemble model available. Train first.")
        
        # Make predictions
        predictions = self.ensemble.predict_ensemble(df)
        
        if not predictions:
            return {}
        
        # Calculate metrics
        actual = df['Close'].values[self.ensemble.sequence_length:]
        predicted = predictions['ensemble_prediction']
        
        if len(actual) != len(predicted):
            # Align arrays
            min_len = min(len(actual), len(predicted))
            actual = actual[-min_len:]
            predicted = predicted[-min_len:]
        
        # Calculate metrics
        mse = mean_squared_error(actual, predicted)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actual, predicted)
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100
        
        # Directional accuracy
        actual_direction = np.diff(actual) > 0
        pred_direction = np.diff(predicted) > 0
        directional_accuracy = np.mean(actual_direction == pred_direction) * 100
        
        # Trading metrics
        returns = (actual[1:] - actual[:-1]) / actual[:-1]
        strategy_returns = np.where(pred_direction, returns, 0)
        cumulative_return = np.prod(1 + strategy_returns) - 1
        sharpe_ratio = np.mean(strategy_returns) / np.std(strategy_returns) if np.std(strategy_returns) > 0 else 0
        
        results = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'directional_accuracy': directional_accuracy,
            'cumulative_return': cumulative_return * 100,
            'sharpe_ratio': sharpe_ratio,
            'ensemble_confidence': predictions['ensemble_confidence'],
            'model_weights': predictions['model_weights']
        }
        
        print(f"📈 MSE: {mse:.4f}")
        print(f"📊 RMSE: {rmse:.4f}")
        print(f"📏 MAE: {mae:.4f}")
        print(f"📊 MAPE: {mape:.2f}%")
        print(f"🎯 Directional Accuracy: {directional_accuracy:.2f}%")
        print(f"💰 Cumulative Return: {cumulative_return*100:.2f}%")
        print(f"📊 Sharpe Ratio: {sharpe_ratio:.2f}")
        print(f"🎯 Ensemble Confidence: {predictions['ensemble_confidence']:.3f}")
        
        return results
    
    def save_models(self, save_dir: str = 'models'):
        """
        Save all models and metadata.
        """
        if not self.ensemble:
            raise ValueError("No ensemble model available. Train first.")
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Save ensemble
        self.ensemble.save_ensemble(save_dir)
        
        # Save metadata
        metadata = {
            'symbol': self.symbol,
            'best_params': self.best_params,
            'feature_columns': [col for col in self.ensemble.prepare_data(pd.DataFrame())[0].columns 
                              if col not in ['Date', 'Close']],
            'training_date': datetime.now().isoformat()
        }
        
        joblib.dump(metadata, f'{save_dir}/{self.symbol}_ensemble_metadata.pkl')
        
        print(f"✅ Models and metadata saved to {save_dir}")
    
    def train_complete_pipeline(self, start_date: str, end_date: str, 
                              optimize_hyperparams: bool = True, n_trials: int = 30) -> Dict:
        """
        Complete training pipeline.
        """
        print(f"🚀 Starting enhanced ensemble training for {self.symbol}")
        print(f"📅 Period: {start_date} to {end_date}")
        
        # Fetch and prepare data
        df = self.fetch_and_prepare_data(start_date, end_date)
        
        # Optimize hyperparameters
        if optimize_hyperparams:
            self.optimize_hyperparameters(df, n_trials)
        
        # Train final ensemble
        self.train_final_ensemble(df)
        
        # Evaluate performance
        results = self.evaluate_ensemble(df)
        
        # Save models
        self.save_models()
        
        return results

def main():
    """Main training function."""
    # Configuration
    symbol = 'AAPL'
    start_date = '2019-01-01'
    end_date = '2024-01-01'
    
    # Create trainer
    trainer = EnhancedEnsembleTrainer(symbol)
    
    # Train complete pipeline
    results = trainer.train_complete_pipeline(
        start_date=start_date,
        end_date=end_date,
        optimize_hyperparams=True,
        n_trials=20  # Reduced for faster training
    )
    
    print("\n🎉 Enhanced ensemble training completed!")
    print(f"📊 Final Results for {symbol}:")
    for metric, value in results.items():
        if isinstance(value, float):
            print(f"   {metric}: {value:.4f}")
        else:
            print(f"   {metric}: {value}")
    
    return trainer, results

if __name__ == "__main__":
    main() 