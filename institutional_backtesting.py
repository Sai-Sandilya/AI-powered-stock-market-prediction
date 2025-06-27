import pandas as pd
import numpy as np
import os
import joblib
import tensorflow as tf

# --- Configuration ---
SYMBOL = 'AAPL'
FEATURE_DATA_PATH = 'data/featured_data'
MODEL_PATH = 'models'
RESULTS_PATH = 'results'

# Ensure results directory exists
os.makedirs(RESULTS_PATH, exist_ok=True)

def create_sequences(features, sequence_length):
    """Creates sequences from the feature data."""
    xs = []
    for i in range(len(features) - sequence_length):
        xs.append(features[i:i + sequence_length])
    return np.array(xs)

def run_backtest_strategy(backtest_df, model, scaler_features, scaler_target, sequence_length, feature_columns):
    """
    Runs the enhanced trading strategy on the provided data.
    This function is designed to be reusable for walk-forward analysis.
    """
    # 1. Prepare data and generate predictions
    features_scaled = scaler_features.transform(backtest_df[feature_columns])
    X_sequences = create_sequences(features_scaled, sequence_length)
    
    # The backtest starts after the first full sequence
    backtest_df = backtest_df.iloc[sequence_length:].copy()

    predicted_scaled = model.predict(X_sequences)
    predicted_prices = scaler_target.inverse_transform(predicted_scaled)
    backtest_df['Predicted_Close'] = predicted_prices

    # 2. Simulate Enhanced Trading Strategy
    CONFIDENCE_THRESHOLD = 0.005 # 0.5% price change
    STOP_LOSS_PCT = 0.02 # 2% stop-loss
    
    # Ensure predicted_prices is a 1D array
    predicted_prices_1d = predicted_prices.flatten()

    predicted_pct_change = (predicted_prices_1d / backtest_df['Close'].values) - 1
    
    # Use np.where for robust conditional assignment
    long_signals = predicted_pct_change > CONFIDENCE_THRESHOLD
    short_signals = predicted_pct_change < -CONFIDENCE_THRESHOLD
    
    backtest_df['Signal'] = np.where(long_signals, 1, np.where(short_signals, -1, 0))
    
    backtest_df['Signal'] = backtest_df['Signal'].replace(0, np.nan).ffill().fillna(0)
    backtest_df['System_Returns'] = backtest_df['Close'].pct_change() * backtest_df['Signal'].shift(1)

    # Apply stop-loss logic
    position = 0
    returns = []
    for i in range(len(backtest_df)):
        ret = backtest_df['System_Returns'].iloc[i]
        if position != 0 and ret < -STOP_LOSS_PCT:
            ret = -STOP_LOSS_PCT
            position = 0
        returns.append(ret)
        position = backtest_df['Signal'].iloc[i]
        
    backtest_df['System_Returns'] = returns
    backtest_df.fillna(0, inplace=True)
    backtest_df['Cumulative_Returns'] = (1 + backtest_df['System_Returns']).cumprod()
    
    return backtest_df

if __name__ == '__main__':
    print("🚀 Starting Institutional Backtest (Standalone Run)...")

    # 1. Load Model and Artifacts
    print("📂 Loading model and artifacts...")
    try:
        model = tf.keras.models.load_model(os.path.join(MODEL_PATH, f"{SYMBOL}_institutional_model.keras"), compile=False)
        scaler_features = joblib.load(os.path.join(MODEL_PATH, f"{SYMBOL}_institutional_scaler_features.pkl"))
        scaler_target = joblib.load(os.path.join(MODEL_PATH, f"{SYMBOL}_institutional_scaler_target.pkl"))
        metadata = joblib.load(os.path.join(MODEL_PATH, f"{SYMBOL}_institutional_metadata.pkl"))
        sequence_length = metadata['sequence_length']
        feature_columns = metadata['feature_columns']
    except FileNotFoundError:
        print(f"❌ Error: Model artifacts for {SYMBOL} not found. Run training first.")
        exit()

    # 2. Load and Prepare Data
    print("📊 Preparing data...")
    input_file = os.path.join(FEATURE_DATA_PATH, f"{SYMBOL}_featured_data.parquet")
    df = pd.read_parquet(input_file)
    
    # Use the same 80/20 split as in training for validation
    train_size = int(len(df) * 0.8)
    validation_df = df.iloc[train_size:].copy()

    # 3. Run the backtest
    print("📈 Running backtest simulation...")
    backtest_results = run_backtest_strategy(
        validation_df, model, scaler_features, scaler_target, sequence_length, feature_columns
    )
    
    # 4. Calculate and Print Performance Metrics
    print("📊 Calculating performance metrics...")
    final_return = backtest_results['Cumulative_Returns'].iloc[-1]
    sharpe_ratio = (backtest_results['System_Returns'].mean() / backtest_results['System_Returns'].std()) * np.sqrt(252)
    cumulative_max = backtest_results['Cumulative_Returns'].cummax()
    drawdown = (cumulative_max - backtest_results['Cumulative_Returns']) / cumulative_max
    max_drawdown = drawdown.max()
    
    print("\n--- Backtest Results ---")
    print(f"✅ Final Cumulative Return: {final_return:.2%}")
    print(f"✅ Annualized Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"✅ Maximum Drawdown: {max_drawdown:.2%}")

    # 5. Save Results
    results_file = os.path.join(RESULTS_PATH, f"{SYMBOL}_institutional_backtest.pkl")
    backtest_results.to_pickle(results_file)
    print(f"\n💾 Saved detailed backtest results to {results_file}") 