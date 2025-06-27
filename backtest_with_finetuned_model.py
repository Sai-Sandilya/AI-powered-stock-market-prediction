import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from keras.models import load_model
import joblib
import os
import warnings

from advanced_feature_engineering_v2 import AdvancedFeatureEngineer

warnings.filterwarnings('ignore')

class FinetunedModelBacktester:
    def __init__(self, symbol='AAPL', start_date='2020-01-01', end_date=None):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        
        # Load the fine-tuned model, scaler, and feature columns
        self.model = load_model(f"models/{self.symbol}_finetuned_best.h5")
        self.scaler = joblib.load(f"models/{self.symbol}_finetuned_scaler.pkl")
        self.feature_columns = joblib.load(f"models/{self.symbol}_feature_columns.pkl")
        
        self.feature_engineer = AdvancedFeatureEngineer()
        self.results = {}

    def fetch_and_prepare_data(self):
        """Fetch data and prepare it for backtesting."""
        print(f"📊 Fetching and preparing data for {self.symbol}...")
        stock = yf.Ticker(self.symbol)
        df = stock.history(start=self.start_date, end=self.end_date)
        
        if df.empty:
            raise ValueError(f"No data found for {self.symbol}")
        
        # Apply the same feature engineering
        df = self.feature_engineer.create_all_features(df)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)
        
        # One-hot encode categorical columns, aligning with training
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        
        # Align columns with the trained model
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0 # Or some other default value
        df = df[self.feature_columns]
        
        return df

    def generate_predictions(self, df, sequence_length=30):
        """Generate predictions using the fine-tuned model."""
        print("🤖 Generating predictions...")
        
        scaled_data = self.scaler.transform(df[self.feature_columns])
        
        X = []
        for i in range(sequence_length, len(scaled_data)):
            X.append(scaled_data[i-sequence_length:i])
        X = np.array(X)
        
        predictions_scaled = self.model.predict(X)
        
        # We need to inverse transform the predictions
        # Create a dummy array with the same shape as the scaler expects
        dummy_array = np.zeros((predictions_scaled.shape[0], len(self.feature_columns)))
        
        # Get the index of the 'Close' column
        close_idx = self.feature_columns.index('Close')
        dummy_array[:, close_idx] = predictions_scaled.flatten()
        
        # Inverse transform
        inversed_predictions = self.scaler.inverse_transform(dummy_array)[:, close_idx]
        
        # Align predictions with the original dataframe index
        pred_df = pd.DataFrame(inversed_predictions, 
                               index=df.index[sequence_length:], 
                               columns=['Predicted_Close'])
        return pred_df

    def run_backtest(self):
        """Run the backtest and evaluate performance."""
        df = self.fetch_and_prepare_data()
        predictions = self.generate_predictions(df)
        
        # Merge predictions with actuals
        results_df = df.join(predictions, how='inner')
        results_df.dropna(inplace=True)

        # Calculate metrics
        actual = results_df['Close']
        predicted = results_df['Predicted_Close']
        
        # Directional Accuracy
        actual_direction = (actual.diff().dropna() > 0).astype(int)
        predicted_direction = (predicted.diff().dropna() > 0).astype(int)
        
        directional_accuracy = np.mean(actual_direction == predicted_direction) * 100
        
        print("\n" + "="*50)
        print(f"      BACKTEST RESULTS FOR {self.symbol}")
        print("="*50 + "\n")
        print(f"  - Directional Accuracy: {directional_accuracy:.2f}%")
        
        # You can add more metrics here (e.g., trading strategy simulation)
        
        return directional_accuracy

def run_backtesting_for_stock(symbol):
    """Convenience function to run the backtesting for a single stock."""
    backtester = FinetunedModelBacktester(symbol=symbol)
    backtester.run_backtest()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, default="AAPL", help="Stock symbol to backtest.")
    args = parser.parse_args()
    
    run_backtesting_for_stock(args.symbol) 