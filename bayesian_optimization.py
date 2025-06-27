"""
Bayesian Optimization Module
Implements advanced hyperparameter optimization using Bayesian methods
for multiple objectives including accuracy and returns.
"""

import pandas as pd
import numpy as np
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import Dict, List, Optional, Tuple, Union, Callable
import warnings
import joblib
import os
from datetime import datetime

warnings.filterwarnings('ignore')

class BayesianOptimizer:
    def __init__(self, 
                 n_trials: int = 100,
                 n_jobs: int = 1,
                 timeout: Optional[int] = None):
        self.n_trials = n_trials
        self.n_jobs = n_jobs
        self.timeout = timeout
        self.best_params = {}
        self.optimization_history = []
        
    def create_study(self, 
                    direction: str = 'minimize',
                    study_name: Optional[str] = None) -> optuna.Study:
        """
        Create an Optuna study with advanced sampling and pruning.
        """
        sampler = TPESampler(seed=42)
        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        
        study = optuna.create_study(
            direction=direction,
            sampler=sampler,
            pruner=pruner,
            study_name=study_name
        )
        
        return study
    
    def suggest_hyperparameters(self, trial: optuna.Trial) -> Dict:
        """
        Suggest hyperparameters for different model types.
        """
        params = {}
        
        # LSTM hyperparameters
        params['lstm_units_1'] = trial.suggest_int('lstm_units_1', 32, 256, step=32)
        params['lstm_units_2'] = trial.suggest_int('lstm_units_2', 16, 128, step=16)
        params['lstm_units_3'] = trial.suggest_int('lstm_units_3', 8, 64, step=8)
        params['dropout_rate'] = trial.suggest_float('dropout_rate', 0.1, 0.5, step=0.1)
        params['learning_rate'] = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        params['batch_size'] = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
        params['sequence_length'] = trial.suggest_int('sequence_length', 30, 120, step=10)
        
        # Feature engineering hyperparameters
        params['feature_window'] = trial.suggest_int('feature_window', 5, 50, step=5)
        params['technical_window'] = trial.suggest_int('technical_window', 10, 100, step=10)
        params['sentiment_weight'] = trial.suggest_float('sentiment_weight', 0.0, 1.0, step=0.1)
        
        # Ensemble hyperparameters
        params['ensemble_weight_lstm'] = trial.suggest_float('ensemble_weight_lstm', 0.1, 0.9, step=0.1)
        params['ensemble_weight_ml'] = trial.suggest_float('ensemble_weight_ml', 0.1, 0.9, step=0.1)
        params['confidence_threshold'] = trial.suggest_float('confidence_threshold', 0.1, 0.9, step=0.1)
        
        # Trading strategy hyperparameters
        params['position_size_multiplier'] = trial.suggest_float('position_size_multiplier', 0.5, 2.0, step=0.1)
        params['stop_loss_pct'] = trial.suggest_float('stop_loss_pct', 0.01, 0.05, step=0.005)
        params['take_profit_pct'] = trial.suggest_float('take_profit_pct', 0.02, 0.10, step=0.01)
        params['max_position_size'] = trial.suggest_float('max_position_size', 0.05, 0.25, step=0.01)
        
        return params
    
    def multi_objective_objective(self, 
                                trial: optuna.Trial,
                                df: pd.DataFrame,
                                model_trainer: Callable,
                                backtester: Callable) -> Tuple[float, float]:
        """
        Multi-objective optimization function for accuracy and returns.
        """
        try:
            # Suggest hyperparameters
            params = self.suggest_hyperparameters(trial)
            
            # Train model with suggested parameters
            model = model_trainer(df, params)
            
            # Run backtest
            backtest_results = backtester(df, model, params)
            
            # Extract objectives
            accuracy_metric = backtest_results.get('mape', 100)  # Lower is better
            return_metric = -backtest_results.get('total_return', 0)  # Negative because we minimize
            
            # Store trial results
            trial_result = {
                'trial_number': trial.number,
                'params': params,
                'accuracy': accuracy_metric,
                'returns': -return_metric,
                'timestamp': datetime.now().isoformat()
            }
            self.optimization_history.append(trial_result)
            
            return accuracy_metric, return_metric
            
        except Exception as e:
            print(f"⚠️ Trial {trial.number} failed: {e}")
            return float('inf'), float('inf')
    
    def optimize_single_objective(self, 
                                df: pd.DataFrame,
                                model_trainer: Callable,
                                backtester: Callable,
                                objective: str = 'accuracy') -> Dict:
        """
        Optimize for a single objective (accuracy or returns).
        """
        print(f"🔍 Optimizing for {objective}...")
        
        def objective_function(trial):
            try:
                params = self.suggest_hyperparameters(trial)
                model = model_trainer(df, params)
                results = backtester(df, model, params)
                
                if objective == 'accuracy':
                    return results.get('mape', 100)
                elif objective == 'returns':
                    return -results.get('total_return', 0)  # Negative for minimization
                else:
                    return results.get('mape', 100)
                    
            except Exception as e:
                print(f"⚠️ Trial {trial.number} failed: {e}")
                return float('inf')
        
        study = self.create_study(direction='minimize')
        study.optimize(objective_function, n_trials=self.n_trials, timeout=self.timeout)
        
        self.best_params = study.best_params
        print(f"✅ Best {objective}: {study.best_value:.4f}")
        print(f"📊 Best parameters: {self.best_params}")
        
        return {
            'best_params': self.best_params,
            'best_value': study.best_value,
            'study': study
        }
    
    def optimize_multi_objective(self, 
                               df: pd.DataFrame,
                               model_trainer: Callable,
                               backtester: Callable) -> Dict:
        """
        Optimize for multiple objectives using Pareto optimization.
        """
        print("🎯 Optimizing for multiple objectives (accuracy + returns)...")
        
        study = optuna.create_study(
            directions=['minimize', 'minimize'],  # Minimize both accuracy and negative returns
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        
        study.optimize(
            lambda trial: self.multi_objective_objective(trial, df, model_trainer, backtester),
            n_trials=self.n_trials,
            timeout=self.timeout
        )
        
        # Find Pareto optimal solutions
        pareto_front = optuna.visualization.plot_pareto_front(study)
        
        # Get best solutions
        best_trials = study.best_trials
        
        print(f"✅ Found {len(best_trials)} Pareto optimal solutions")
        
        return {
            'study': study,
            'pareto_front': pareto_front,
            'best_trials': best_trials,
            'optimization_history': self.optimization_history
        }
    
    def cross_validate_hyperparameters(self, 
                                     df: pd.DataFrame,
                                     params: Dict,
                                     model_trainer: Callable,
                                     n_splits: int = 5) -> Dict:
        """
        Perform time series cross-validation for hyperparameters.
        """
        print("🔄 Performing time series cross-validation...")
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        cv_scores = []
        
        for train_idx, val_idx in tscv.split(df):
            train_df = df.iloc[train_idx]
            val_df = df.iloc[val_idx]
            
            try:
                # Train model
                model = model_trainer(train_df, params)
                
                # Evaluate on validation set
                # This would need to be implemented based on your specific model
                val_score = self.evaluate_model(model, val_df)
                cv_scores.append(val_score)
                
            except Exception as e:
                print(f"⚠️ CV fold failed: {e}")
                cv_scores.append(float('inf'))
        
        cv_results = {
            'mean_score': np.mean(cv_scores),
            'std_score': np.std(cv_scores),
            'cv_scores': cv_scores,
            'params': params
        }
        
        print(f"📊 CV Mean Score: {cv_results['mean_score']:.4f} ± {cv_results['std_score']:.4f}")
        
        return cv_results
    
    def evaluate_model(self, model, val_df: pd.DataFrame) -> float:
        """
        Evaluate model performance (placeholder - implement based on your model).
        """
        # This is a placeholder - implement based on your specific model
        try:
            # Make predictions
            predictions = model.predict(val_df)
            actual = val_df['Close'].values
            
            # Calculate MSE
            mse = mean_squared_error(actual, predictions)
            return mse
            
        except Exception as e:
            print(f"⚠️ Model evaluation failed: {e}")
            return float('inf')
    
    def save_optimization_results(self, results: Dict, save_dir: str = 'optimization'):
        """
        Save optimization results and history.
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Save best parameters
        if 'best_params' in results:
            joblib.dump(results['best_params'], f'{save_dir}/best_params.pkl')
        
        # Save optimization history
        if 'optimization_history' in results:
            history_df = pd.DataFrame(results['optimization_history'])
            history_df.to_csv(f'{save_dir}/optimization_history.csv', index=False)
        
        # Save study object
        if 'study' in results:
            joblib.dump(results['study'], f'{save_dir}/study.pkl')
        
        print(f"✅ Optimization results saved to {save_dir}")
    
    def load_optimization_results(self, save_dir: str = 'optimization') -> Dict:
        """
        Load optimization results.
        """
        results = {}
        
        # Load best parameters
        best_params_path = f'{save_dir}/best_params.pkl'
        if os.path.exists(best_params_path):
            results['best_params'] = joblib.load(best_params_path)
        
        # Load optimization history
        history_path = f'{save_dir}/optimization_history.csv'
        if os.path.exists(history_path):
            results['optimization_history'] = pd.read_csv(history_path)
        
        # Load study object
        study_path = f'{save_dir}/study.pkl'
        if os.path.exists(study_path):
            results['study'] = joblib.load(study_path)
        
        print(f"✅ Optimization results loaded from {save_dir}")
        
        return results
    
    def plot_optimization_results(self, results: Dict):
        """
        Plot optimization results and convergence.
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            if 'optimization_history' in results:
                history_df = pd.DataFrame(results['optimization_history'])
                
                # Plot convergence
                plt.figure(figsize=(15, 10))
                
                # Accuracy convergence
                plt.subplot(2, 2, 1)
                plt.plot(history_df['trial_number'], history_df['accuracy'])
                plt.title('Accuracy Convergence')
                plt.xlabel('Trial Number')
                plt.ylabel('MAPE')
                plt.grid(True)
                
                # Returns convergence
                plt.subplot(2, 2, 2)
                plt.plot(history_df['trial_number'], history_df['returns'])
                plt.title('Returns Convergence')
                plt.xlabel('Trial Number')
                plt.ylabel('Total Return')
                plt.grid(True)
                
                # Parameter importance (if study available)
                if 'study' in results:
                    plt.subplot(2, 2, 3)
                    optuna.visualization.plot_param_importances(results['study'])
                    plt.title('Parameter Importance')
                
                # Pareto front (if multi-objective)
                if 'pareto_front' in results:
                    plt.subplot(2, 2, 4)
                    optuna.visualization.plot_pareto_front(results['study'])
                    plt.title('Pareto Front')
                
                plt.tight_layout()
                plt.savefig('optimization_results.png', dpi=300, bbox_inches='tight')
                plt.show()
                
        except ImportError:
            print("⚠️ Matplotlib not available for plotting")
        except Exception as e:
            print(f"⚠️ Plotting failed: {e}")

def test_bayesian_optimization():
    """Test the Bayesian optimization module."""
    print("🧪 Testing Bayesian optimization...")
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    
    dates = pd.date_range(start='2020-01-01', periods=n_samples, freq='D')
    returns = np.random.normal(0.001, 0.02, n_samples)
    prices = 100 * np.exp(np.cumsum(returns))
    volumes = np.random.lognormal(10, 0.5, n_samples)
    
    df = pd.DataFrame({
        'Date': dates,
        'Open': prices * (1 + np.random.normal(0, 0.005, n_samples)),
        'High': prices * (1 + np.abs(np.random.normal(0, 0.01, n_samples))),
        'Low': prices * (1 - np.abs(np.random.normal(0, 0.01, n_samples))),
        'Close': prices,
        'Volume': volumes
    })
    
    # Mock model trainer and backtester functions
    def mock_model_trainer(df, params):
        return {'model': 'mock', 'params': params}
    
    def mock_backtester(df, model, params):
        return {
            'mape': np.random.uniform(5, 15),
            'total_return': np.random.uniform(-0.2, 0.3),
            'sharpe_ratio': np.random.uniform(0.5, 2.0)
        }
    
    # Create optimizer
    optimizer = BayesianOptimizer(n_trials=10)  # Reduced for testing
    
    # Run single objective optimization
    results = optimizer.optimize_single_objective(
        df, mock_model_trainer, mock_backtester, 'accuracy'
    )
    
    print(f"✅ Optimization completed!")
    print(f"📊 Best parameters: {results['best_params']}")
    
    return optimizer, results

if __name__ == "__main__":
    test_bayesian_optimization() 