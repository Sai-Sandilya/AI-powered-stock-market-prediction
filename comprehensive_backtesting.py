import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

class ComprehensiveBacktester:
    def __init__(self, symbol='AAPL', start_date='2020-01-01', end_date=None):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
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
    
    def walk_forward_analysis(self, df, window_size=252, step_size=63):
        """Perform walk-forward analysis"""
        print("🔄 Performing walk-forward analysis...")
        
        results = []
        dates = df.index
        
        for i in range(window_size, len(df) - step_size, step_size):
            # Training period
            train_start = i - window_size
            train_end = i
            train_data = df.iloc[train_start:train_end]
            
            # Testing period
            test_start = i
            test_end = min(i + step_size, len(df))
            test_data = df.iloc[test_start:test_end]
            
            # Simple prediction (using last known value as baseline)
            baseline_pred = train_data['Close'].iloc[-1]
            actual_prices = test_data['Close'].values
            
            # Calculate metrics
            mse = mean_squared_error(actual_prices, [baseline_pred] * len(actual_prices))
            mae = mean_absolute_error(actual_prices, [baseline_pred] * len(actual_prices))
            rmse = np.sqrt(mse)
            mape = np.mean(np.abs((actual_prices - baseline_pred) / actual_prices)) * 100
            
            # Directional accuracy
            actual_direction = np.diff(actual_prices) > 0
            pred_direction = np.diff([baseline_pred] * len(actual_prices)) > 0
            directional_accuracy = np.mean(actual_direction == pred_direction) * 100
            
            results.append({
                'Period': f"{dates[train_start].strftime('%Y-%m-%d')} to {dates[test_end-1].strftime('%Y-%m-%d')}",
                'Train_Start': dates[train_start],
                'Train_End': dates[train_end-1],
                'Test_Start': dates[test_start],
                'Test_End': dates[test_end-1],
                'RMSE': rmse,
                'MAE': mae,
                'MAPE': mape,
                'Directional_Accuracy': directional_accuracy,
                'Actual_Return': (actual_prices[-1] - actual_prices[0]) / actual_prices[0],
                'Predicted_Return': 0  # Baseline prediction
            })
        
        return pd.DataFrame(results)
    
    def multi_timeframe_analysis(self, df):
        """Analyze performance across different timeframes"""
        print("⏰ Performing multi-timeframe analysis...")
        
        timeframes = {
            '1D': 1,
            '1W': 5,
            '1M': 21,
            '3M': 63,
            '6M': 126,
            '1Y': 252
        }
        
        results = {}
        
        for tf_name, tf_days in timeframes.items():
            if len(df) < tf_days * 2:
                continue
                
            # Calculate returns for different timeframes
            returns = df['Close'].pct_change(tf_days).dropna()
            
            # Calculate volatility
            volatility = returns.rolling(window=tf_days).std().dropna()
            
            # Calculate Sharpe ratio (assuming risk-free rate of 0)
            sharpe_ratio = returns.mean() / returns.std() if returns.std() > 0 else 0
            
            # Calculate maximum drawdown
            cumulative_returns = (1 + returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = drawdown.min()
            
            results[tf_name] = {
                'Mean_Return': returns.mean(),
                'Volatility': volatility.mean(),
                'Sharpe_Ratio': sharpe_ratio,
                'Max_Drawdown': max_drawdown,
                'Win_Rate': (returns > 0).mean(),
                'Best_Day': returns.max(),
                'Worst_Day': returns.min()
            }
        
        return pd.DataFrame(results).T
    
    def risk_metrics_analysis(self, df):
        """Calculate comprehensive risk metrics"""
        print("⚠️ Calculating risk metrics...")
        
        returns = df['Close'].pct_change().dropna()
        
        # Basic risk metrics
        mean_return = returns.mean()
        volatility = returns.std()
        sharpe_ratio = mean_return / volatility if volatility > 0 else 0
        
        # Value at Risk (VaR)
        var_95 = np.percentile(returns, 5)
        var_99 = np.percentile(returns, 1)
        
        # Expected Shortfall (Conditional VaR)
        es_95 = returns[returns <= var_95].mean()
        es_99 = returns[returns <= var_99].mean()
        
        # Maximum drawdown
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Skewness and Kurtosis
        skewness = returns.skew()
        kurtosis = returns.kurtosis()
        
        # Calmar ratio (annual return / max drawdown)
        annual_return = mean_return * 252
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Sortino ratio (using downside deviation)
        downside_returns = returns[returns < 0]
        downside_deviation = downside_returns.std()
        sortino_ratio = mean_return / downside_deviation if downside_deviation > 0 else 0
        
        return {
            'Mean_Return': mean_return,
            'Volatility': volatility,
            'Sharpe_Ratio': sharpe_ratio,
            'Sortino_Ratio': sortino_ratio,
            'Calmar_Ratio': calmar_ratio,
            'VaR_95': var_95,
            'VaR_99': var_99,
            'ES_95': es_95,
            'ES_99': es_99,
            'Max_Drawdown': max_drawdown,
            'Skewness': skewness,
            'Kurtosis': kurtosis,
            'Win_Rate': (returns > 0).mean(),
            'Best_Day': returns.max(),
            'Worst_Day': returns.min()
        }
    
    def trading_strategy_backtest(self, df, strategy='momentum'):
        """Backtest different trading strategies"""
        print(f"📈 Backtesting {strategy} strategy...")
        
        if strategy == 'momentum':
            # Momentum strategy: buy when price > SMA_20, sell when price < SMA_20
            df['Position'] = np.where(df['Close'] > df['SMA_20'], 1, -1)
        elif strategy == 'mean_reversion':
            # Mean reversion: buy when price < BB_Lower, sell when price > BB_Upper
            df['Position'] = np.where(df['Close'] < df['BB_Lower'], 1, 
                                    np.where(df['Close'] > df['BB_Upper'], -1, 0))
        elif strategy == 'rsi':
            # RSI strategy: buy when RSI < 30, sell when RSI > 70
            df['Position'] = np.where(df['RSI'] < 30, 1, 
                                    np.where(df['RSI'] > 70, -1, 0))
        else:
            # Buy and hold
            df['Position'] = 1
        
        # Calculate strategy returns
        df['Returns'] = df['Close'].pct_change()
        df['Strategy_Returns'] = df['Position'].shift(1) * df['Returns']
        
        # Calculate cumulative returns
        df['Cumulative_Returns'] = (1 + df['Returns']).cumprod()
        df['Strategy_Cumulative_Returns'] = (1 + df['Strategy_Returns']).cumprod()
        
        # Calculate metrics
        total_return = df['Strategy_Cumulative_Returns'].iloc[-1] - 1
        annual_return = (1 + total_return) ** (252 / len(df)) - 1
        volatility = df['Strategy_Returns'].std() * np.sqrt(252)
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0
        
        # Maximum drawdown
        running_max = df['Strategy_Cumulative_Returns'].expanding().max()
        drawdown = (df['Strategy_Cumulative_Returns'] - running_max) / running_max
        max_drawdown = drawdown.min()
        
        return {
            'Strategy': strategy,
            'Total_Return': total_return,
            'Annual_Return': annual_return,
            'Volatility': volatility,
            'Sharpe_Ratio': sharpe_ratio,
            'Max_Drawdown': max_drawdown,
            'Win_Rate': (df['Strategy_Returns'] > 0).mean(),
            'Returns_Data': df[['Strategy_Returns', 'Strategy_Cumulative_Returns']]
        }
    
    def comprehensive_analysis(self):
        """Run comprehensive backtesting analysis"""
        print("🚀 Starting comprehensive backtesting analysis...")
        
        # Fetch data
        df = self.fetch_data()
        
        # Walk-forward analysis
        walk_forward_results = self.walk_forward_analysis(df)
        
        # Multi-timeframe analysis
        multi_tf_results = self.multi_timeframe_analysis(df)
        
        # Risk metrics
        risk_metrics = self.risk_metrics_analysis(df)
        
        # Trading strategies
        strategies = ['momentum', 'mean_reversion', 'rsi', 'buy_hold']
        strategy_results = {}
        
        for strategy in strategies:
            strategy_results[strategy] = self.trading_strategy_backtest(df, strategy)
        
        # Store all results
        self.results = {
            'walk_forward': walk_forward_results,
            'multi_timeframe': multi_tf_results,
            'risk_metrics': risk_metrics,
            'strategies': strategy_results,
            'data': df
        }
        
        return self.results
    
    def print_comprehensive_results(self):
        """Print comprehensive backtesting results"""
        print("\n" + "="*100)
        print(f"📊 COMPREHENSIVE BACKTESTING RESULTS FOR {self.symbol}")
        print("="*100)
        
        # Risk Metrics
        print("\n⚠️ RISK METRICS:")
        print("-" * 50)
        risk_metrics = self.results['risk_metrics']
        print(f"   📈 Annual Return: {risk_metrics['Mean_Return'] * 252:.2%}")
        print(f"   📊 Volatility: {risk_metrics['Volatility'] * np.sqrt(252):.2%}")
        print(f"   🎯 Sharpe Ratio: {risk_metrics['Sharpe_Ratio']:.2f}")
        print(f"   📉 Sortino Ratio: {risk_metrics['Sortino_Ratio']:.2f}")
        print(f"   📊 Calmar Ratio: {risk_metrics['Calmar_Ratio']:.2f}")
        print(f"   ⚠️ Max Drawdown: {risk_metrics['Max_Drawdown']:.2%}")
        print(f"   📊 VaR (95%): {risk_metrics['VaR_95']:.2%}")
        print(f"   📊 VaR (99%): {risk_metrics['VaR_99']:.2%}")
        print(f"   🎯 Win Rate: {risk_metrics['Win_Rate']:.2%}")
        
        # Multi-timeframe Analysis
        print("\n⏰ MULTI-TIMEFRAME ANALYSIS:")
        print("-" * 50)
        multi_tf = self.results['multi_timeframe']
        for tf in multi_tf.index:
            print(f"   {tf}: Return={multi_tf.loc[tf, 'Mean_Return']:.2%}, "
                  f"Vol={multi_tf.loc[tf, 'Volatility']:.2%}, "
                  f"Sharpe={multi_tf.loc[tf, 'Sharpe_Ratio']:.2f}")
        
        # Trading Strategies
        print("\n📈 TRADING STRATEGIES COMPARISON:")
        print("-" * 50)
        strategies = self.results['strategies']
        for strategy_name, strategy_data in strategies.items():
            print(f"   {strategy_name.upper()}:")
            print(f"     📈 Total Return: {strategy_data['Total_Return']:.2%}")
            print(f"     📊 Annual Return: {strategy_data['Annual_Return']:.2%}")
            print(f"     🎯 Sharpe Ratio: {strategy_data['Sharpe_Ratio']:.2f}")
            print(f"     📉 Max Drawdown: {strategy_data['Max_Drawdown']:.2%}")
            print(f"     🎯 Win Rate: {strategy_data['Win_Rate']:.2%}")
        
        # Walk-forward Analysis Summary
        print("\n🔄 WALK-FORWARD ANALYSIS SUMMARY:")
        print("-" * 50)
        wf_results = self.results['walk_forward']
        print(f"   📊 Average RMSE: ${wf_results['RMSE'].mean():.2f}")
        print(f"   📏 Average MAE: ${wf_results['MAE'].mean():.2f}")
        print(f"   📊 Average MAPE: {wf_results['MAPE'].mean():.2f}%")
        print(f"   🎯 Average Directional Accuracy: {wf_results['Directional_Accuracy'].mean():.2f}%")
        
        # Find best strategy
        best_strategy = max(strategies.keys(), 
                          key=lambda x: strategies[x]['Sharpe_Ratio'])
        print(f"\n🏆 BEST STRATEGY: {best_strategy.upper()}")
        print(f"   Sharpe Ratio: {strategies[best_strategy]['Sharpe_Ratio']:.2f}")
        print(f"   Total Return: {strategies[best_strategy]['Total_Return']:.2%}")

def run_backtesting_for_multiple_stocks():
    """Run comprehensive backtesting for multiple stocks"""
    stocks = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA', 'AMZN', 'META', 'NFLX']
    
    all_results = {}
    
    for stock in stocks:
        print(f"\n{'='*80}")
        print(f"🎯 PROCESSING {stock}")
        print(f"{'='*80}")
        
        try:
            backtester = ComprehensiveBacktester(symbol=stock)
            results = backtester.comprehensive_analysis()
            backtester.print_comprehensive_results()
            
            all_results[stock] = results
            
            # Save results
            os.makedirs('results', exist_ok=True)
            joblib.dump(results, f'results/{stock}_comprehensive_backtest.pkl')
            
        except Exception as e:
            print(f"❌ Error processing {stock}: {e}")
            continue
    
    return all_results

if __name__ == "__main__":
    # Run comprehensive backtesting
    results = run_backtesting_for_multiple_stocks()
    
    print("\n" + "="*100)
    print("🎉 COMPREHENSIVE BACKTESTING COMPLETED!")
    print("="*100)
    print("📁 Results saved in 'results/' directory")
    print("📊 Check individual stock files for detailed analysis") 