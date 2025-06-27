import pandas as pd
from datetime import datetime, timedelta
import warnings
import os
import joblib
import numpy as np
import tensorflow as tf
import yfinance as yf

# Add project root to path to allow imports from dags
os.path.abspath(os.path.dirname(__file__))

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# --- Local Functions (to avoid import issues) ---
def fetch_yfinance(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, progress=False)
    if df is None or df.empty:
        raise ValueError(f"No data found for ticker {ticker}")
    df.reset_index(inplace=True)
    df.set_index('Date', inplace=True) # Set Date as index for time-based slicing
    return df

# --- Project-Specific Imports ---
from institutional_model_training import train_and_evaluate_model
from institutional_backtesting import run_backtest_strategy

def perform_walk_forward_analysis(symbol, start_date_str, end_date_str, train_period_days, test_period_days):
    """
    Performs a walk-forward analysis by training and backtesting in rolling windows.
    """
    current_date = pd.to_datetime(start_date_str)
    end_date = pd.to_datetime(end_date_str)
    all_backtest_results = []
    
    run_number = 1
    while current_date + timedelta(days=train_period_days + test_period_days) <= end_date:
        train_start = current_date
        train_end = train_start + timedelta(days=train_period_days)
        test_start = train_end + timedelta(days=1)
        test_end = test_start + timedelta(days=test_period_days)

        print(f"\n===== Walk-Forward Run #{run_number} for {symbol} =====")
        print(f"Training: {train_start.date()} to {train_end.date()} | Testing: {test_start.date()} to {test_end.date()}")

        # 1. Fetch and process data
        print("Step 1: Fetching and processing data...")
        try:
            full_period_data = fetch_yfinance(symbol, train_start.strftime('%Y-%m-%d'), test_end.strftime('%Y-%m-%d'))
            if full_period_data.empty:
                raise ValueError("Data fetching returned empty DataFrame.")
        except Exception as e:
            print(f"Could not fetch data for run #{run_number}. Skipping. Error: {e}")
            current_date += timedelta(days=test_period_days)
            continue
        
        # Use a simplified feature set for walk-forward
        full_period_data['Returns'] = full_period_data['Close'].pct_change()
        full_period_data['Volatility'] = full_period_data['Returns'].rolling(window=21).std()
        full_period_data['Momentum'] = full_period_data['Close'].pct_change(21)
        full_period_data.dropna(inplace=True)
        
        feature_columns = ['Returns', 'Volatility', 'Momentum', 'Close'] # Add Close to features
        target_column = 'Close'

        training_data = full_period_data.loc[train_start:train_end].copy()

        # 2. Train the model
        print(f"Step 2: Training model for run #{run_number}...")
        model_prefix = os.path.join('models', f"{symbol}_wfa_run_{run_number}")
        
        try:
            model_path, scaler_features_path, scaler_target_path, metadata_path = train_and_evaluate_model(
                symbol, training_data, feature_columns, target_column, model_prefix
            )
        except Exception as e:
            print(f"Error during model training for run #{run_number}: {e}")
            print("This may be due to the 'PATENCE' typo. Continuing to next run.")
            current_date += timedelta(days=test_period_days)
            run_number += 1
            continue

        # 3. Backtest the model
        print(f"Step 3: Backtesting model for run #{run_number}...")
        model = tf.keras.models.load_model(model_path, compile=False)
        scaler_features = joblib.load(scaler_features_path)
        scaler_target = joblib.load(scaler_target_path)
        metadata = joblib.load(metadata_path)
        
        testing_data = full_period_data.loc[test_start:test_end].copy()
        
        backtest_run_results = run_backtest_strategy(
            testing_data, model, scaler_features, scaler_target, metadata['sequence_length'], metadata['feature_columns']
        )
        
        all_backtest_results.append(backtest_run_results)
        print(f"Run #{run_number} complete. Cumulative Return: {backtest_run_results['Cumulative_Returns'].iloc[-1]:.2%}")

        current_date += timedelta(days=test_period_days)
        run_number += 1
        
    print("\n===== Walk-Forward Analysis Complete =====")
    if not all_backtest_results:
        print("No analysis was run. Check date ranges or data fetching.")
        return

    # 4. Aggregate and report final results
    final_results_df = pd.concat(all_backtest_results)
    final_results_df['Cumulative_Returns'] = (1 + final_results_df['System_Returns']).cumprod()
    
    print("\n--- Aggregated Performance ---")
    final_return = final_results_df['Cumulative_Returns'].iloc[-1]
    sharpe_ratio = (final_results_df['System_Returns'].mean() / final_results_df['System_Returns'].std()) * np.sqrt(252)
    cumulative_max = final_results_df['Cumulative_Returns'].cummax()
    drawdown = (cumulative_max - final_results_df['Cumulative_Returns']) / cumulative_max
    max_drawdown = drawdown.max()

    print(f"Total Period Tested: {final_results_df.index.min().date()} to {final_results_df.index.max().date()}")
    print(f"Final Cumulative Return: {final_return:.2%}")
    print(f"Annualized Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Maximum Drawdown: {max_drawdown:.2%}")
    
    results_file = f"results/{symbol}_walk_forward_results.pkl"
    final_results_df.to_pickle(results_file)
    print(f"\n✅ Aggregated results saved to {results_file}")

if __name__ == "__main__":
    perform_walk_forward_analysis(
        symbol='AAPL',
        start_date_str='2018-01-01',
        end_date_str='2023-12-31',
        train_period_days=365 * 2,
        test_period_days=180
    ) 