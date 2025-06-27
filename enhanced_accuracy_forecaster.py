#!/usr/bin/env python3
"""
Enhanced Accuracy Forecaster
Advanced ML-based stock prediction system for improved backtesting accuracy.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
import warnings
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import joblib
import os

# Optional imports
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

# Import our modules
from data_ingestion import fetch_yfinance
from feature_engineering import get_comprehensive_features, add_technical_indicators

warnings.filterwarnings('ignore')

class EnhancedAccuracyForecaster:
    """
    Enhanced forecasting system with advanced ML techniques for maximum accuracy.
    """
    
    def __init__(self, symbol: str = 'AAPL'):
        self.symbol = symbol
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.prediction_intervals = {}
        
        # Advanced parameters
        self.window_sizes = [5, 10, 20, 50]  # Multiple timeframes
        self.prediction_horizons = [1, 5, 10, 20]  # Multiple forecast horizons
        self.ensemble_weights = {}
        
        print(f"✅ Initialized Enhanced Accuracy Forecaster for {symbol}")
        
    def create_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create advanced technical and statistical features for improved prediction.
        """
        print("🔧 Creating advanced features...")
        
        # Start with comprehensive features
        df = get_comprehensive_features(df, include_macro=True)
        
        # Price-based features
        for window in [3, 5, 7, 10, 14, 21, 30]:
            # Price statistics
            df[f'Price_ROC_{window}'] = df['Close'].pct_change(window)
            df[f'Price_Momentum_{window}'] = df['Close'] / df['Close'].shift(window) - 1
            df[f'Price_Acceleration_{window}'] = df[f'Price_ROC_{window}'] - df[f'Price_ROC_{window}'].shift(1)
            
            # Volatility features
            df[f'Returns_Std_{window}'] = df['Close'].pct_change().rolling(window).std()
            df[f'Returns_Skew_{window}'] = df['Close'].pct_change().rolling(window).skew()
            df[f'Returns_Kurt_{window}'] = df['Close'].pct_change().rolling(window).kurt()
            
            # Volume features
            df[f'Volume_SMA_{window}'] = df['Volume'].rolling(window).mean()
            df[f'Volume_Ratio_{window}'] = df['Volume'] / df[f'Volume_SMA_{window}']
            df[f'Volume_Momentum_{window}'] = df['Volume'].pct_change(window)
            
            # High-Low features
            df[f'HL_Ratio_{window}'] = (df['High'] - df['Low']) / df['Close']
            df[f'HL_Position_{window}'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])
        
        # Advanced technical indicators
        
        # Bollinger Band variations
        for window in [10, 20, 50]:
            ma = df['Close'].rolling(window).mean()
            std = df['Close'].rolling(window).std()
            df[f'BB_Upper_{window}'] = ma + 2 * std
            df[f'BB_Lower_{window}'] = ma - 2 * std
            df[f'BB_Width_{window}'] = (df[f'BB_Upper_{window}'] - df[f'BB_Lower_{window}']) / ma
            df[f'BB_Position_{window}'] = (df['Close'] - df[f'BB_Lower_{window}']) / (df[f'BB_Upper_{window}'] - df[f'BB_Lower_{window}'])
        
        # Multiple RSI periods
        for window in [7, 14, 21, 30]:
            delta = df['Close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
            rs = gain / loss
            df[f'RSI_{window}'] = 100 - (100 / (1 + rs))
            
            # RSI divergence
            df[f'RSI_Divergence_{window}'] = df[f'RSI_{window}'] - df[f'RSI_{window}'].shift(5)
        
        # Stochastic variations
        for window in [14, 21]:
            low_min = df['Low'].rolling(window).min()
            high_max = df['High'].rolling(window).max()
            df[f'Stoch_K_{window}'] = 100 * (df['Close'] - low_min) / (high_max - low_min)
            df[f'Stoch_D_{window}'] = df[f'Stoch_K_{window}'].rolling(3).mean()
        
        # Williams %R variations
        for window in [14, 21]:
            high_max = df['High'].rolling(window).max()
            low_min = df['Low'].rolling(window).min()
            df[f'Williams_R_{window}'] = -100 * (high_max - df['Close']) / (high_max - low_min)
        
        # MACD variations
        for fast, slow, signal in [(12, 26, 9), (8, 21, 5), (19, 39, 9)]:
            ema_fast = df['Close'].ewm(span=fast).mean()
            ema_slow = df['Close'].ewm(span=slow).mean()
            macd = ema_fast - ema_slow
            macd_signal = macd.ewm(span=signal).mean()
            df[f'MACD_{fast}_{slow}'] = macd
            df[f'MACD_Signal_{fast}_{slow}'] = macd_signal
            df[f'MACD_Histogram_{fast}_{slow}'] = macd - macd_signal
        
        # Commodity Channel Index (CCI)
        for window in [14, 20]:
            tp = (df['High'] + df['Low'] + df['Close']) / 3
            sma_tp = tp.rolling(window).mean()
            mad = tp.rolling(window).apply(lambda x: np.mean(np.abs(x - np.mean(x))))
            df[f'CCI_{window}'] = (tp - sma_tp) / (0.015 * mad)
        
        # Average True Range (ATR)
        for window in [14, 21]:
            tr1 = df['High'] - df['Low']
            tr2 = np.abs(df['High'] - df['Close'].shift())
            tr3 = np.abs(df['Low'] - df['Close'].shift())
            tr = np.maximum(tr1, np.maximum(tr2, tr3))
            df[f'ATR_{window}'] = tr.rolling(window).mean()
            df[f'ATR_Ratio_{window}'] = df[f'ATR_{window}'] / df['Close']
        
        # Fibonacci retracements
        for window in [20, 50]:
            high_max = df['High'].rolling(window).max()
            low_min = df['Low'].rolling(window).min()
            diff = high_max - low_min
            df[f'Fib_23.6_{window}'] = high_max - 0.236 * diff
            df[f'Fib_38.2_{window}'] = high_max - 0.382 * diff
            df[f'Fib_61.8_{window}'] = high_max - 0.618 * diff
            
            # Distance to Fibonacci levels
            df[f'Dist_Fib_23.6_{window}'] = np.abs(df['Close'] - df[f'Fib_23.6_{window}']) / df['Close']
            df[f'Dist_Fib_38.2_{window}'] = np.abs(df['Close'] - df[f'Fib_38.2_{window}']) / df['Close']
            df[f'Dist_Fib_61.8_{window}'] = np.abs(df['Close'] - df[f'Fib_61.8_{window}']) / df['Close']
        
        # Support and Resistance levels
        for window in [20, 50]:
            df[f'Support_{window}'] = df['Low'].rolling(window).min()
            df[f'Resistance_{window}'] = df['High'].rolling(window).max()
            df[f'Support_Distance_{window}'] = (df['Close'] - df[f'Support_{window}']) / df['Close']
            df[f'Resistance_Distance_{window}'] = (df[f'Resistance_{window}'] - df['Close']) / df['Close']
        
        # Market microstructure features
        df['Bid_Ask_Spread'] = (df['High'] - df['Low']) / df['Close']  # Proxy for spread
        df['Order_Flow'] = df['Volume'] * np.sign(df['Close'].diff())  # Proxy for order flow
        df['Price_Impact'] = np.abs(df['Close'].pct_change()) / (df['Volume'] / df['Volume'].rolling(20).mean())
        
        # Regime indicators
        df['Volatility_Regime'] = df['Close'].pct_change().rolling(20).std() > df['Close'].pct_change().rolling(50).std()
        df['Trend_Regime'] = df['Close'] > df['Close'].rolling(50).mean()
        df['Volume_Regime'] = df['Volume'] > df['Volume'].rolling(20).mean()
        
        # Calendar effects
        df['DayOfWeek'] = df.index.dayofweek
        df['MonthOfYear'] = df.index.month
        df['QuarterOfYear'] = df.index.quarter
        df['IsMonthEnd'] = df.index.is_month_end.astype(int)
        df['IsQuarterEnd'] = df.index.is_quarter_end.astype(int)
        
        # Lagged features
        for lag in [1, 2, 3, 5, 10]:
            df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)
            df[f'Volume_Lag_{lag}'] = df['Volume'].shift(lag)
            df[f'Returns_Lag_{lag}'] = df['Close'].pct_change().shift(lag)
        
        # Cross-sectional features (if available)
        # Market cap, sector performance, etc. would go here
        
        print(f"✅ Created {len([col for col in df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume']])} advanced features")
        
        return df.dropna()
    
    def create_ensemble_models(self) -> Dict[str, any]:
        """
        Create ensemble of ML models for prediction.
        """
        models = {}
        
        # Tree-based models
        models['RandomForest'] = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        models['GradientBoosting'] = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        
        if XGBOOST_AVAILABLE:
            models['XGBoost'] = XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                min_child_weight=1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            )
        
        if LIGHTGBM_AVAILABLE:
            models['LightGBM'] = LGBMRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
        
        # Linear models with regularization
        models['Ridge'] = Ridge(alpha=1.0)
        models['Lasso'] = Lasso(alpha=0.1)
        models['ElasticNet'] = ElasticNet(alpha=0.1, l1_ratio=0.5)
        
        # Support Vector Machines
        models['SVR_RBF'] = SVR(kernel='rbf', C=1.0, gamma='scale')
        models['SVR_Linear'] = SVR(kernel='linear', C=1.0)
        
        print(f"✅ Created ensemble of {len(models)} models")
        return models
    
    def optimize_hyperparameters(self, X: pd.DataFrame, y: pd.Series, model_name: str, model: any) -> any:
        """
        Optimize hyperparameters using time series cross-validation.
        """
        print(f"🎯 Optimizing {model_name} hyperparameters...")
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=3)
        
        # Define parameter grids for different models
        param_grids = {
            'RandomForest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15],
                'min_samples_split': [2, 5, 10]
            },
            'GradientBoosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.05, 0.1, 0.2],
                'max_depth': [3, 6, 9]
            },
            'Ridge': {
                'alpha': [0.1, 1.0, 10.0, 100.0]
            },
            'Lasso': {
                'alpha': [0.01, 0.1, 1.0, 10.0]
            },
            'SVR_RBF': {
                'C': [0.1, 1.0, 10.0],
                'gamma': ['scale', 'auto', 0.01, 0.1]
            }
        }
        
        if model_name in param_grids:
            try:
                grid_search = GridSearchCV(
                    model,
                    param_grids[model_name],
                    cv=tscv,
                    scoring='neg_mean_absolute_percentage_error',
                    n_jobs=-1,
                    verbose=0
                )
                
                grid_search.fit(X, y)
                print(f"   Best MAPE: {-grid_search.best_score_:.2f}%")
                print(f"   Best params: {grid_search.best_params_}")
                return grid_search.best_estimator_
                
            except Exception as e:
                print(f"   Optimization failed: {e}, using default parameters")
                return model
        
        return model
    
    def train_ensemble(self, df: pd.DataFrame, target_col: str = 'Close', 
                      optimize: bool = True) -> Dict[str, any]:
        """
        Train ensemble of models with advanced feature engineering.
        """
        print("🤖 Training enhanced ensemble models...")
        
        # Create advanced features
        featured_df = self.create_advanced_features(df)
        
        # Prepare features and target
        feature_cols = [col for col in featured_df.columns 
                       if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'Date']]
        
        X = featured_df[feature_cols]
        y = featured_df[target_col]
        
        # Handle infinite and missing values
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.mean())
        
        print(f"📊 Training with {len(feature_cols)} features and {len(X)} samples")
        
        # Create scalers for each model type
        self.scalers['standard'] = StandardScaler()
        self.scalers['robust'] = RobustScaler()
        
        # Fit scalers
        X_standard = self.scalers['standard'].fit_transform(X)
        X_robust = self.scalers['robust'].fit_transform(X)
        
        # Create ensemble models
        base_models = self.create_ensemble_models()
        
        # Train models
        trained_models = {}
        model_scores = {}
        
        for model_name, model in base_models.items():
            try:
                print(f"   Training {model_name}...")
                
                # Choose appropriate scaling
                if model_name in ['Ridge', 'Lasso', 'ElasticNet', 'SVR_RBF', 'SVR_Linear']:
                    X_train = X_standard
                else:
                    X_train = X.values
                
                # Optimize hyperparameters if requested
                if optimize:
                    model = self.optimize_hyperparameters(X, y, model_name, model)
                
                # Train model
                model.fit(X_train, y)
                trained_models[model_name] = model
                
                # Calculate training score
                y_pred = model.predict(X_train)
                mape = mean_absolute_percentage_error(y, y_pred) * 100
                model_scores[model_name] = mape
                
                print(f"   ✅ {model_name}: MAPE = {mape:.2f}%")
                
                # Store feature importance if available
                if hasattr(model, 'feature_importances_'):
                    importance_df = pd.DataFrame({
                        'feature': feature_cols,
                        'importance': model.feature_importances_
                    }).sort_values('importance', ascending=False)
                    self.feature_importance[model_name] = importance_df
                
            except Exception as e:
                print(f"   ❌ Failed to train {model_name}: {e}")
                continue
        
        # Calculate ensemble weights based on inverse MAPE
        total_inverse_mape = sum(1 / score for score in model_scores.values())
        self.ensemble_weights = {
            name: (1 / score) / total_inverse_mape 
            for name, score in model_scores.items()
        }
        
        print(f"✅ Trained {len(trained_models)} models successfully")
        print("📊 Ensemble weights:")
        for name, weight in self.ensemble_weights.items():
            print(f"   {name}: {weight:.3f}")
        
        self.models = trained_models
        return trained_models
    
    def predict_ensemble(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make ensemble predictions with confidence intervals.
        """
        if not self.models:
            raise ValueError("No trained models available. Call train_ensemble first.")
        
        # Handle infinite and missing values
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.mean())
        
        predictions = []
        model_predictions = {}
        
        for model_name, model in self.models.items():
            try:
                # Choose appropriate scaling
                if model_name in ['Ridge', 'Lasso', 'ElasticNet', 'SVR_RBF', 'SVR_Linear']:
                    X_scaled = self.scalers['standard'].transform(X)
                else:
                    X_scaled = X.values
                
                pred = model.predict(X_scaled)
                model_predictions[model_name] = pred
                
                # Weight predictions
                weight = self.ensemble_weights.get(model_name, 1.0 / len(self.models))
                predictions.append(pred * weight)
                
            except Exception as e:
                print(f"⚠️ Error predicting with {model_name}: {e}")
                continue
        
        if not predictions:
            raise ValueError("No successful predictions from ensemble models")
        
        # Ensemble prediction
        ensemble_pred = np.sum(predictions, axis=0)
        
        # Calculate prediction intervals using model disagreement
        model_preds_array = np.array(list(model_predictions.values()))
        pred_std = np.std(model_preds_array, axis=0)
        
        # 95% confidence intervals
        confidence_intervals = np.array([
            ensemble_pred - 1.96 * pred_std,
            ensemble_pred + 1.96 * pred_std
        ]).T
        
        return ensemble_pred, confidence_intervals
    
    def walk_forward_backtest(self, df: pd.DataFrame, window_size: int = 252, 
                             step_size: int = 21) -> Dict[str, any]:
        """
        Perform walk-forward backtesting with enhanced accuracy measurement.
        """
        print("🔄 Performing walk-forward backtesting...")
        
        results = []
        feature_cols = None
        
        for i in range(window_size, len(df) - step_size, step_size):
            print(f"   Processing window {i // step_size + 1}...")
            
            # Training data
            train_data = df.iloc[i - window_size:i].copy()
            
            # Testing data
            test_data = df.iloc[i:i + step_size].copy()
            
            try:
                # Train ensemble on this window
                self.train_ensemble(train_data, optimize=False)  # Skip optimization for speed
                
                # Create features for test data
                test_featured = self.create_advanced_features(test_data)
                
                if feature_cols is None:
                    feature_cols = [col for col in test_featured.columns 
                                   if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'Date']]
                
                X_test = test_featured[feature_cols]
                y_test = test_featured['Close']
                
                # Make predictions
                predictions, confidence_intervals = self.predict_ensemble(X_test)
                
                # Calculate metrics
                mape = mean_absolute_percentage_error(y_test, predictions) * 100
                rmse = np.sqrt(mean_squared_error(y_test, predictions))
                mae = np.mean(np.abs(y_test - predictions))
                
                # Directional accuracy
                actual_direction = np.diff(y_test.values) > 0
                pred_direction = np.diff(predictions) > 0
                directional_accuracy = np.mean(actual_direction == pred_direction) * 100
                
                # Coverage of confidence intervals
                lower_bound = confidence_intervals[:, 0]
                upper_bound = confidence_intervals[:, 1]
                coverage = np.mean((y_test >= lower_bound) & (y_test <= upper_bound)) * 100
                
                results.append({
                    'Window': i // step_size + 1,
                    'Train_Start': train_data.index[0],
                    'Train_End': train_data.index[-1],
                    'Test_Start': test_data.index[0],
                    'Test_End': test_data.index[-1],
                    'MAPE': mape,
                    'RMSE': rmse,
                    'MAE': mae,
                    'Directional_Accuracy': directional_accuracy,
                    'Coverage_95': coverage,
                    'Predictions': predictions,
                    'Actual': y_test.values,
                    'Confidence_Intervals': confidence_intervals
                })
                
                print(f"      MAPE: {mape:.2f}%, Directional Accuracy: {directional_accuracy:.2f}%")
                
            except Exception as e:
                print(f"      ❌ Error in window {i // step_size + 1}: {e}")
                continue
        
        # Calculate overall metrics
        all_mapes = [r['MAPE'] for r in results]
        all_directional = [r['Directional_Accuracy'] for r in results]
        all_coverage = [r['Coverage_95'] for r in results]
        
        overall_results = {
            'individual_results': results,
            'overall_metrics': {
                'Mean_MAPE': np.mean(all_mapes),
                'Std_MAPE': np.std(all_mapes),
                'Best_MAPE': np.min(all_mapes),
                'Worst_MAPE': np.max(all_mapes),
                'Mean_Directional_Accuracy': np.mean(all_directional),
                'Mean_Coverage': np.mean(all_coverage),
                'Windows_Tested': len(results)
            }
        }
        
        print(f"✅ Walk-forward backtest completed!")
        print(f"📊 Overall MAPE: {overall_results['overall_metrics']['Mean_MAPE']:.2f}% ± {overall_results['overall_metrics']['Std_MAPE']:.2f}%")
        print(f"🎯 Mean Directional Accuracy: {overall_results['overall_metrics']['Mean_Directional_Accuracy']:.2f}%")
        print(f"📊 Confidence Interval Coverage: {overall_results['overall_metrics']['Mean_Coverage']:.2f}%")
        
        return overall_results
    
    def save_enhanced_system(self, filename: str = None):
        """
        Save the enhanced forecasting system.
        """
        if filename is None:
            filename = f"enhanced_forecaster_{self.symbol}.pkl"
        
        save_data = {
            'models': self.models,
            'scalers': self.scalers,
            'ensemble_weights': self.ensemble_weights,
            'feature_importance': self.feature_importance,
            'symbol': self.symbol
        }
        
        os.makedirs('models', exist_ok=True)
        joblib.dump(save_data, f"models/{filename}")
        print(f"💾 Enhanced system saved to models/{filename}")
    
    def load_enhanced_system(self, filename: str):
        """
        Load a saved enhanced forecasting system.
        """
        try:
            save_data = joblib.load(f"models/{filename}")
            self.models = save_data['models']
            self.scalers = save_data['scalers']
            self.ensemble_weights = save_data['ensemble_weights']
            self.feature_importance = save_data['feature_importance']
            self.symbol = save_data['symbol']
            print(f"✅ Enhanced system loaded from models/{filename}")
        except Exception as e:
            print(f"❌ Error loading system: {e}")


def test_enhanced_accuracy():
    """
    Test the enhanced accuracy forecasting system.
    """
    print("🧪 Testing Enhanced Accuracy Forecasting System")
    print("=" * 70)
    
    # Test symbols
    symbols = ['AAPL', 'MSFT', 'GOOGL']
    
    for symbol in symbols:
        print(f"\n📊 Testing Enhanced Forecasting for {symbol}")
        print("-" * 50)
        
        try:
            # Initialize enhanced forecaster
            forecaster = EnhancedAccuracyForecaster(symbol)
            
            # Fetch historical data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=800)  # More data for better training
            
            df = fetch_yfinance(symbol, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
            
            if len(df) < 300:
                print(f"⚠️ Insufficient data for {symbol}")
                continue
            
            # Perform walk-forward backtesting
            results = forecaster.walk_forward_backtest(df, window_size=252, step_size=21)
            
            # Save results
            os.makedirs('results', exist_ok=True)
            joblib.dump(results, f'results/{symbol}_enhanced_backtest.pkl')
            
            # Save the trained system
            forecaster.save_enhanced_system(f"enhanced_forecaster_{symbol}.pkl")
            
            print(f"✅ Enhanced backtesting completed for {symbol}")
            
        except Exception as e:
            print(f"❌ Error testing {symbol}: {e}")
            continue
    
    print("\n🎉 Enhanced accuracy testing completed!")


if __name__ == "__main__":
    test_enhanced_accuracy() 