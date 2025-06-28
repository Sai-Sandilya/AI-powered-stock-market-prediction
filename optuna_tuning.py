"""
Optuna Hyperparameter Tuning for Stock Prediction Models
Advanced optimization for Random Forest, XGBoost, and other ML models
"""

import optuna
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

class OptunaTuner:
    """
    Advanced hyperparameter tuning using Optuna for stock prediction models
    """
    
    def __init__(self, X, y, model_type='random_forest', n_trials=100, cv_folds=5):
        """
        Initialize Optuna tuner
        
        Args:
            X: Feature matrix
            y: Target values  
            model_type: Type of model ('random_forest', 'xgboost', 'lightgbm')
            n_trials: Number of optimization trials
            cv_folds: Number of cross-validation folds
        """
        self.X = X
        self.y = y
        self.model_type = model_type
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.study = None
        self.best_model = None
        self.scaler = StandardScaler()
        
        # Prepare data
        self.X_scaled = self.scaler.fit_transform(X)
        
        # Use TimeSeriesSplit for financial data
        self.cv = TimeSeriesSplit(n_splits=cv_folds)
        
    def objective_random_forest(self, trial):
        """Objective function for Random Forest optimization"""
        # Suggest hyperparameters
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['auto', 'sqrt', 'log2']),
            'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
            'random_state': 42
        }
        
        # Create model
        model = RandomForestRegressor(**params)
        
        # Cross-validation score
        scores = cross_val_score(model, self.X_scaled, self.y, cv=self.cv, 
                               scoring='neg_mean_squared_error', n_jobs=-1)
        
        return -scores.mean()  # Return negative MSE (Optuna minimizes)
    
    def objective_xgboost(self, trial):
        """Objective function for XGBoost optimization"""
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost not available")
            
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
            'random_state': 42
        }
        
        model = xgb.XGBRegressor(**params)
        scores = cross_val_score(model, self.X_scaled, self.y, cv=self.cv,
                               scoring='neg_mean_squared_error', n_jobs=-1)
        
        return -scores.mean()
    
    def objective_lightgbm(self, trial):
        """Objective function for LightGBM optimization"""
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM not available")
            
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'num_leaves': trial.suggest_int('num_leaves', 10, 300),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
            'random_state': 42,
            'verbosity': -1
        }
        
        model = LGBMRegressor(**params)
        scores = cross_val_score(model, self.X_scaled, self.y, cv=self.cv,
                               scoring='neg_mean_squared_error', n_jobs=-1)
        
        return -scores.mean()
    
    def optimize(self, direction='minimize', show_progress=True):
        """
        Run hyperparameter optimization
        
        Args:
            direction: Optimization direction ('minimize' or 'maximize')
            show_progress: Whether to show progress bar
            
        Returns:
            dict: Best parameters and study results
        """
        # Create study
        study_name = f"{self.model_type}_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.study = optuna.create_study(direction=direction, study_name=study_name)
        
        # Select objective function
        if self.model_type == 'random_forest':
            objective = self.objective_random_forest
        elif self.model_type == 'xgboost':
            objective = self.objective_xgboost
        elif self.model_type == 'lightgbm':
            objective = self.objective_lightgbm
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        # Optimize with progress callback
        if show_progress:
            self.study.optimize(objective, n_trials=self.n_trials, 
                              callbacks=[self._progress_callback])
        else:
            self.study.optimize(objective, n_trials=self.n_trials)
        
        # Train best model
        self._train_best_model()
        
        return {
            'best_params': self.study.best_params,
            'best_value': self.study.best_value,
            'n_trials': len(self.study.trials),
            'study': self.study
        }
    
    def _progress_callback(self, study, trial):
        """Progress callback for optimization"""
        if trial.number % 10 == 0:
            print(f"Trial {trial.number}: Best value = {study.best_value:.6f}")
    
    def _train_best_model(self):
        """Train the best model with optimal parameters"""
        best_params = self.study.best_params
        
        if self.model_type == 'random_forest':
            self.best_model = RandomForestRegressor(**best_params)
        elif self.model_type == 'xgboost':
            self.best_model = xgb.XGBRegressor(**best_params)
        elif self.model_type == 'lightgbm':
            self.best_model = LGBMRegressor(**best_params)
        
        # Train on full dataset
        self.best_model.fit(self.X_scaled, self.y)
    
    def get_model_performance(self, X_test=None, y_test=None):
        """
        Evaluate model performance
        
        Args:
            X_test: Test features (optional)
            y_test: Test targets (optional)
            
        Returns:
            dict: Performance metrics
        """
        if self.best_model is None:
            raise ValueError("Model not trained yet. Run optimize() first.")
        
        if X_test is not None and y_test is not None:
            # Evaluate on test set
            X_test_scaled = self.scaler.transform(X_test)
            y_pred = self.best_model.predict(X_test_scaled)
            
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            return {
                'mse': mse,
                'rmse': np.sqrt(mse),
                'mae': mae,
                'r2': r2,
                'predictions': y_pred
            }
        else:
            # Cross-validation performance
            cv_scores = cross_val_score(self.best_model, self.X_scaled, self.y, 
                                      cv=self.cv, scoring='neg_mean_squared_error')
            
            return {
                'cv_mse': -cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'cv_scores': cv_scores
            }
    
    def get_feature_importance(self, feature_names=None):
        """
        Get feature importance from the best model
        
        Args:
            feature_names: List of feature names
            
        Returns:
            pandas.DataFrame: Feature importance
        """
        if self.best_model is None:
            raise ValueError("Model not trained yet. Run optimize() first.")
        
        if hasattr(self.best_model, 'feature_importances_'):
            importance = self.best_model.feature_importances_
            
            if feature_names is None:
                feature_names = [f'feature_{i}' for i in range(len(importance))]
            
            df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            return df
        else:
            return pd.DataFrame()  # Return empty DataFrame if no feature importance
    
    def save_model(self, filepath):
        """Save the best model and scaler"""
        if self.best_model is None:
            raise ValueError("Model not trained yet. Run optimize() first.")
        
        joblib.dump({
            'model': self.best_model,
            'scaler': self.scaler,
            'best_params': self.study.best_params,
            'model_type': self.model_type
        }, filepath)
    
    def load_model(self, filepath):
        """Load a saved model and scaler"""
        saved_data = joblib.load(filepath)
        self.best_model = saved_data['model']
        self.scaler = saved_data['scaler']
        return saved_data
    
    def predict(self, X):
        """Make predictions with the best model"""
        if self.best_model is None:
            raise ValueError("Model not trained yet. Run optimize() first.")
        
        X_scaled = self.scaler.transform(X)
        return self.best_model.predict(X_scaled)
    
    def get_optimization_history(self):
        """Get optimization history for visualization"""
        if self.study is None:
            return pd.DataFrame()
        
        trials_df = self.study.trials_dataframe()
        return trials_df

def quick_tune_random_forest(X, y, n_trials=50):
    """
    Quick function to tune Random Forest for stock prediction
    
    Args:
        X: Feature matrix
        y: Target values
        n_trials: Number of trials
        
    Returns:
        OptunaTuner: Trained tuner object
    """
    tuner = OptunaTuner(X, y, model_type='random_forest', n_trials=n_trials)
    results = tuner.optimize(show_progress=False)
    
    return tuner, results

def get_available_models():
    """Get list of available models for tuning"""
    models = ['random_forest']
    
    if XGBOOST_AVAILABLE:
        models.append('xgboost')
    
    if LIGHTGBM_AVAILABLE:
        models.append('lightgbm')
    
    return models

def tune_ensemble_models(X, y, n_trials=30):
    """
    Tune multiple models and return the best performing ensemble
    
    Args:
        X: Feature matrix
        y: Target values
        n_trials: Number of trials per model
        
    Returns:
        dict: Results for all models
    """
    available_models = get_available_models()
    results = {}
    
    for model_type in available_models:
        print(f"Tuning {model_type}...")
        tuner = OptunaTuner(X, y, model_type=model_type, n_trials=n_trials)
        optimization_result = tuner.optimize(show_progress=False)
        
        results[model_type] = {
            'tuner': tuner,
            'best_params': optimization_result['best_params'],
            'best_score': optimization_result['best_value'],
            'performance': tuner.get_model_performance()
        }
    
    # Find best model
    best_model_type = min(results.keys(), key=lambda x: results[x]['best_score'])
    results['best_model'] = best_model_type
    
    return results 