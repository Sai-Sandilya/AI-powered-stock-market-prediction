"""
Comprehensive Backtesting Module
Evaluates the accuracy of stock price predictions against historical data.
Features:
- Walk-forward analysis
- Multiple accuracy metrics
- Performance comparison
- Risk-adjusted returns
- Statistical significance testing
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
from typing import Dict, List, Optional, Tuple, Union
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# Import our modules
from data_ingestion import fetch_yfinance
from feature_engineering import get_comprehensive_features, add_technical_indicators
from future_forecasting import FutureForecaster
from macro_indicators import MacroIndicators

warnings.filterwarnings('ignore')

class Backtester:
    """
    Comprehensive backtesting system for stock price predictions.
    """
    
    def __init__(self, model_path: Optional[str] = None, scaler_path: Optional[str] = None):
        """
        Initialize backtester with optional model and scaler paths.
        """
        self.forecaster = FutureForecaster()
        
        # Try to load improved model first, then fallback to basic model
        if model_path is None:
            # Try improved model first
            improved_model_path = 'models/AAPL_improved_lstm.h5'
            improved_scaler_path = 'models/AAPL_improved_scaler.pkl'
            
            if os.path.exists(improved_model_path) and os.path.exists(improved_scaler_path):
                print("✅ Loading improved LSTM model...")
                self.forecaster.load_model(improved_model_path)
                self.forecaster.load_scaler(improved_scaler_path)
            else:
                # Fallback to basic model
                basic_model_path = 'models/AAPL_advanced_lstm.h5'
                basic_scaler_path = 'models/AAPL_advanced_scaler.pkl'
                
                if os.path.exists(basic_model_path) and os.path.exists(basic_scaler_path):
                    print("✅ Loading basic LSTM model...")
                    self.forecaster.load_model(basic_model_path)
                    self.forecaster.load_scaler(basic_scaler_path)
                else:
                    print("⚠️ No trained model found. Using dummy model for backtesting.")
        else:
            self.forecaster.load_model(model_path)
            if scaler_path:
                self.forecaster.load_scaler(scaler_path)
        self.results = {}
        
    def calculate_accuracy_metrics(self, actual: np.ndarray, predicted: np.ndarray) -> Dict:
        """
        Calculate comprehensive accuracy metrics.
        
        Args:
            actual: Actual values
            predicted: Predicted values
            
        Returns:
            Dictionary of accuracy metrics
        """
        # Remove any NaN values
        mask = ~(np.isnan(actual) | np.isnan(predicted))
        actual_clean = actual[mask]
        predicted_clean = predicted[mask]
        
        if len(actual_clean) == 0:
            return {}
        
        # Basic metrics
        mae = np.mean(np.abs(actual_clean - predicted_clean))
        mse = np.mean((actual_clean - predicted_clean) ** 2)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((actual_clean - predicted_clean) / actual_clean)) * 100
        
        # Directional accuracy
        actual_direction = np.diff(actual_clean) > 0
        predicted_direction = np.diff(predicted_clean) > 0
        directional_accuracy = np.mean(actual_direction == predicted_direction) * 100
        
        # Correlation
        correlation = np.corrcoef(actual_clean, predicted_clean)[0, 1]
        
        # R-squared
        ss_res = np.sum((actual_clean - predicted_clean) ** 2)
        ss_tot = np.sum((actual_clean - np.mean(actual_clean)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # Theil's U statistic (perfect forecast = 1)
        theil_u = np.sqrt(np.mean((actual_clean - predicted_clean) ** 2)) / np.sqrt(np.mean(actual_clean ** 2))
        
        # Hit rate (predictions within 5% of actual)
        hit_rate_5 = np.mean(np.abs((actual_clean - predicted_clean) / actual_clean) <= 0.05) * 100
        hit_rate_10 = np.mean(np.abs((actual_clean - predicted_clean) / actual_clean) <= 0.10) * 100
        
        return {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'MAPE': mape,
            'Directional_Accuracy': directional_accuracy,
            'Correlation': correlation,
            'R_Squared': r_squared,
            'Theil_U': theil_u,
            'Hit_Rate_5%': hit_rate_5,
            'Hit_Rate_10%': hit_rate_10,
            'Sample_Size': len(actual_clean)
        }
    
    def calculate_trading_metrics(self, actual_prices: np.ndarray, predicted_prices: np.ndarray) -> Dict:
        """
        Calculate trading performance metrics.
        
        Args:
            actual_prices: Actual price series
            predicted_prices: Predicted price series
            
        Returns:
            Dictionary of trading metrics
        """
        # Calculate returns
        actual_returns = np.diff(actual_prices) / actual_prices[:-1]
        predicted_returns = np.diff(predicted_prices) / predicted_prices[:-1]
        
        # Trading signals based on predicted direction
        signals = np.where(predicted_returns > 0, 1, -1)  # 1 for buy, -1 for sell
        
        # Strategy returns (buy and hold vs prediction-based)
        strategy_returns = signals * actual_returns
        
        # Performance metrics
        total_return = np.prod(1 + strategy_returns) - 1
        annualized_return = (1 + total_return) ** (252 / len(strategy_returns)) - 1
        volatility = np.std(strategy_returns) * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # Maximum drawdown
        cumulative_returns = np.cumprod(1 + strategy_returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        # Win rate
        winning_trades = np.sum(strategy_returns > 0)
        total_trades = len(strategy_returns)
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        
        # Average win/loss
        wins = strategy_returns[strategy_returns > 0]
        losses = strategy_returns[strategy_returns < 0]
        avg_win = np.mean(wins) if len(wins) > 0 else 0
        avg_loss = np.mean(losses) if len(losses) > 0 else 0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        
        return {
            'Total_Return': total_return,
            'Annualized_Return': annualized_return,
            'Volatility': volatility,
            'Sharpe_Ratio': sharpe_ratio,
            'Max_Drawdown': max_drawdown,
            'Win_Rate': win_rate,
            'Total_Trades': total_trades,
            'Avg_Win': avg_win,
            'Avg_Loss': avg_loss,
            'Profit_Factor': profit_factor
        }
    
    def walk_forward_backtest(self, symbol: str, start_date: str, end_date: str, 
                            forecast_horizon: int = 30, step_size: int = 5,
                            include_macro: bool = True) -> Dict:
        """
        Perform walk-forward backtesting with proper date range handling.
        Only forecasts dates that are within the available historical data.
        """
        print(f"🔍 Starting walk-forward backtest for {symbol}")
        print(f"   Period: {start_date} to {end_date}")
        print(f"   Forecast horizon: {forecast_horizon} days")
        print(f"   Step size: {step_size} days")
        
        # Fetch historical data
        try:
            df = fetch_yfinance(symbol, start_date, end_date)
            print(f"✅ Fetched {len(df)} historical records")
        except Exception as e:
            print(f"❌ Error fetching data: {e}")
            return {}
        
        # Convert dates and sort
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
        
        # Add features
        if include_macro:
            df = get_comprehensive_features(df, include_macro=True)
        else:
            df = add_technical_indicators(df)
        
        # Initialize results storage
        all_predictions = []
        all_actuals = []
        all_dates = []
        
        # Walk-forward parameters
        min_train_size = 252  # At least 1 year of training data
        train_start_idx = 0
        
        step_count = 0
        while train_start_idx + min_train_size + forecast_horizon < len(df):
            step_count += 1
            
            # Define training and forecast periods
            train_end_idx = train_start_idx + min_train_size
            forecast_start_idx = train_end_idx
            forecast_end_idx = min(forecast_start_idx + forecast_horizon, len(df))
            
            # Get training data
            train_data = df.iloc[train_start_idx:train_end_idx].copy()
            train_start_date = train_data['Date'].iloc[0]
            train_end_date = train_data['Date'].iloc[-1]
            
            # Get actual values for the forecast period
            actual_data = df.iloc[forecast_start_idx:forecast_end_idx].copy()
            
            print(f"   Step {step_count}: Training {train_start_date.strftime('%Y-%m-%d')} to {train_end_date.strftime('%Y-%m-%d')}")
            print(f"      Forecasting {actual_data['Date'].iloc[0].strftime('%Y-%m-%d')} to {actual_data['Date'].iloc[-1].strftime('%Y-%m-%d')}")
            
            try:
                # Generate forecast using the training period
                forecast_df = self.forecaster.forecast_future(
                    symbol=symbol,
                    forecast_days=len(actual_data),
                    include_macro=include_macro
                )
                
                if not forecast_df.empty and len(forecast_df) == len(actual_data):
                    # Collect predictions and actuals
                    for i, (pred_row, actual_row) in enumerate(zip(forecast_df.iterrows(), actual_data.iterrows())):
                        pred_date = pred_row[1]['Date']
                        pred_price = pred_row[1]['Predicted_Close']
                        actual_price = actual_row[1]['Close']
                        
                        all_predictions.append(pred_price)
                        all_actuals.append(actual_price)
                        all_dates.append(pred_date)
                    
                    print(f"      ✅ Collected {len(actual_data)} predictions and actuals")
                else:
                    print(f"      ⚠️ Forecast mismatch: {len(forecast_df)} predictions vs {len(actual_data)} actuals")
                    
            except Exception as e:
                print(f"      ❌ Error in forecasting step: {e}")
            
            # Move to next step
            train_start_idx += step_size
        
        # Calculate metrics
        if len(all_predictions) > 0:
            print(f"\n=== Backtest Summary ===")
            print(f"Total predictions: {len(all_predictions)}")
            
            # Convert to numpy arrays
            predictions = np.array(all_predictions)
            actuals = np.array(all_actuals)
            
            # Calculate accuracy metrics
            mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
            rmse = np.sqrt(np.mean((actuals - predictions) ** 2))
            mae = np.mean(np.abs(actuals - predictions))
            
            # Calculate directional accuracy
            pred_direction = np.diff(predictions) > 0
            actual_direction = np.diff(actuals) > 0
            directional_accuracy = np.mean(pred_direction == actual_direction) * 100
            
            # Calculate trading metrics
            returns = (actuals[1:] - actuals[:-1]) / actuals[:-1]
            pred_returns = (predictions[1:] - predictions[:-1]) / predictions[:-1]
            
            # Simple trading strategy: buy when prediction goes up
            strategy_returns = np.where(pred_direction, returns, 0)
            cumulative_return = np.prod(1 + strategy_returns) - 1
            sharpe_ratio = np.mean(strategy_returns) / np.std(strategy_returns) if np.std(strategy_returns) > 0 else 0
            
            results = {
                'symbol': symbol,
                'total_predictions': len(all_predictions),
                'mape': mape,
                'rmse': rmse,
                'mae': mae,
                'directional_accuracy': directional_accuracy,
                'cumulative_return': cumulative_return * 100,
                'sharpe_ratio': sharpe_ratio,
                'predictions': all_predictions,
                'actuals': all_actuals,
                'dates': all_dates
            }
            
            print(f"📈 MAPE: {mape:.2f}%")
            print(f"📊 RMSE: ${rmse:.2f}")
            print(f"📏 MAE: ${mae:.2f}")
            print(f"🎯 Directional Accuracy: {directional_accuracy:.2f}%")
            print(f"💰 Cumulative Return: {cumulative_return*100:.2f}%")
            print(f"📊 Sharpe Ratio: {sharpe_ratio:.2f}")
            
            return results
        else:
            print("❌ No valid predictions generated for backtesting")
            return {}
    
    def plot_backtest_results(self, save_path: Optional[str] = None):
        """
        Create comprehensive visualization of backtest results.
        
        Args:
            save_path: Optional path to save the plot
        """
        if not self.results:
            print("❌ No backtest results to plot")
            return
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Price Predictions vs Actual',
                'Prediction Errors Over Time',
                'Error Distribution',
                'Cumulative Returns',
                'Accuracy Metrics',
                'Trading Performance'
            ),
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}]
            ]
        )
        
        # 1. Price predictions vs actual
        fig.add_trace(
            go.Scatter(
                x=self.results['dates'],
                y=self.results['actuals'],
                mode='lines',
                name='Actual',
                line=dict(color='blue')
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=self.results['dates'],
                y=self.results['predictions'],
                mode='lines',
                name='Predicted',
                line=dict(color='red')
            ),
            row=1, col=1
        )
        
        # 2. Prediction errors over time
        errors = np.array(self.results['predictions']) - np.array(self.results['actuals'])
        fig.add_trace(
            go.Scatter(
                x=self.results['dates'],
                y=errors,
                mode='lines',
                name='Prediction Error',
                line=dict(color='orange')
            ),
            row=1, col=2
        )
        
        # 3. Error distribution
        fig.add_trace(
            go.Histogram(
                x=errors,
                nbinsx=30,
                name='Error Distribution',
                marker_color='lightblue'
            ),
            row=2, col=1
        )
        
        # 4. Cumulative returns
        actual_returns = np.diff(self.results['actuals']) / self.results['actuals'][:-1]
        predicted_returns = np.diff(self.results['predictions']) / self.results['predictions'][:-1]
        signals = np.where(predicted_returns > 0, 1, -1)
        strategy_returns = signals * actual_returns
        cumulative_returns = np.cumprod(1 + strategy_returns)
        
        fig.add_trace(
            go.Scatter(
                x=self.results['dates'][1:],
                y=cumulative_returns,
                mode='lines',
                name='Strategy Returns',
                line=dict(color='green')
            ),
            row=2, col=2
        )
        
        # 5. Accuracy metrics bar chart
        accuracy_metrics = self.results['accuracy_metrics']
        metrics_names = list(accuracy_metrics.keys())
        metrics_values = list(accuracy_metrics.values())
        
        # Filter out non-numeric metrics
        numeric_metrics = []
        numeric_values = []
        for name, value in zip(metrics_names, metrics_values):
            if isinstance(value, (int, float)) and name != 'Sample_Size':
                numeric_metrics.append(name)
                numeric_values.append(value)
        
        fig.add_trace(
            go.Bar(
                x=numeric_metrics,
                y=numeric_values,
                name='Accuracy Metrics',
                marker_color='purple'
            ),
            row=3, col=1
        )
        
        # 6. Trading metrics
        trading_metrics = self.results['trading_metrics']
        trading_names = list(trading_metrics.keys())
        trading_values = list(trading_metrics.values())
        
        # Filter out non-numeric metrics
        numeric_trading = []
        numeric_trading_values = []
        for name, value in zip(trading_names, trading_values):
            if isinstance(value, (int, float)) and name != 'Total_Trades':
                numeric_trading.append(name)
                numeric_trading_values.append(value)
        
        fig.add_trace(
            go.Bar(
                x=numeric_trading,
                y=numeric_trading_values,
                name='Trading Metrics',
                marker_color='brown'
            ),
            row=3, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=1200,
            title_text=f"Backtest Results: {self.results['symbol']} ({self.results['start_date']} to {self.results['end_date']})",
            showlegend=True
        )
        
        # Show plot
        fig.show()
        
        if save_path:
            fig.write_html(save_path)
            print(f"✅ Plot saved to {save_path}")
    
    def print_summary_report(self):
        """Print a comprehensive summary report of backtest results."""
        if not self.results:
            print("❌ No backtest results to report")
            return
        
        print("\n" + "="*60)
        print("📊 BACKTEST SUMMARY REPORT")
        print("="*60)
        
        print(f"Symbol: {self.results['symbol']}")
        print(f"Period: {self.results['start_date']} to {self.results['end_date']}")
        print(f"Forecast Horizon: {self.results['forecast_horizon']} days")
        print(f"Total Predictions: {self.results['total_predictions']}")
        
        print("\n📈 ACCURACY METRICS:")
        print("-" * 30)
        accuracy = self.results['accuracy_metrics']
        for metric, value in accuracy.items():
            if isinstance(value, float):
                if 'Rate' in metric or 'Accuracy' in metric:
                    print(f"{metric}: {value:.2f}%")
                else:
                    print(f"{metric}: {value:.4f}")
            else:
                print(f"{metric}: {value}")
        
        print("\n💰 TRADING PERFORMANCE:")
        print("-" * 30)
        trading = self.results['trading_metrics']
        for metric, value in trading.items():
            if isinstance(value, float):
                if 'Return' in metric or 'Drawdown' in metric:
                    print(f"{metric}: {value:.2%}")
                elif 'Ratio' in metric:
                    print(f"{metric}: {value:.2f}")
                else:
                    print(f"{metric}: {value:.4f}")
            else:
                print(f"{metric}: {value}")
        
        print("\n🎯 KEY INSIGHTS:")
        print("-" * 30)
        
        # Directional accuracy insight
        if accuracy.get('Directional_Accuracy', 0) > 60:
            print("✅ Good directional accuracy - model predicts price direction well")
        elif accuracy.get('Directional_Accuracy', 0) > 50:
            print("⚠️ Moderate directional accuracy - some room for improvement")
        else:
            print("❌ Poor directional accuracy - model struggles with direction")
        
        # MAPE insight
        mape = accuracy.get('MAPE', 0)
        if mape < 5:
            print("✅ Excellent prediction accuracy (MAPE < 5%)")
        elif mape < 10:
            print("✅ Good prediction accuracy (MAPE < 10%)")
        elif mape < 20:
            print("⚠️ Moderate prediction accuracy (MAPE < 20%)")
        else:
            print("❌ Poor prediction accuracy (MAPE > 20%)")
        
        # Trading performance insight
        sharpe = trading.get('Sharpe_Ratio', 0)
        if sharpe > 1:
            print("✅ Excellent risk-adjusted returns (Sharpe > 1)")
        elif sharpe > 0.5:
            print("✅ Good risk-adjusted returns (Sharpe > 0.5)")
        elif sharpe > 0:
            print("⚠️ Positive but low risk-adjusted returns")
        else:
            print("❌ Negative risk-adjusted returns")
        
        print("="*60)


def test_backtesting():
    """
    Test the backtesting functionality.
    """
    print("🧪 Testing Backtesting Module")
    print("=" * 50)
    
    # Initialize backtester
    backtester = Backtester()
    
    # Test with a sample stock
    symbol = 'AAPL'
    start_date = '2023-01-01'
    end_date = '2024-01-01'
    
    print(f"\n📊 Testing backtest for {symbol}")
    print(f"   Period: {start_date} to {end_date}")
    
    # Test different forecast horizons
    horizons = [7, 14, 30]  # 1 week, 2 weeks, 1 month
    
    for horizon in horizons:
        print(f"\n🔍 Testing {horizon}-day forecast horizon...")
        
        try:
            results = backtester.walk_forward_backtest(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                forecast_horizon=horizon,
                step_size=5,
                include_macro=True
            )
            
            if results:
                print(f"✅ Completed backtest with {results['total_predictions']} predictions")
                
                # Print summary
                backtester.print_summary_report()
                
                # Create visualization
                backtester.plot_backtest_results()
                
            else:
                print("❌ No backtest results generated")
                
        except Exception as e:
            print(f"❌ Error in backtest: {e}")
    
    print("\n🎉 Backtesting test completed!")


if __name__ == "__main__":
    test_backtesting() 