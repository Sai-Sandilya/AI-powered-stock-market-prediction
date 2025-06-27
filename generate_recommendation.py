import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from keras.models import load_model
import joblib
import os
import warnings

from advanced_feature_engineering_v2 import AdvancedFeatureEngineer

warnings.filterwarnings('ignore')

class RecommendationGenerator:
    def __init__(self, symbol='AAPL'):
        self.symbol = symbol
        
        # Load the fine-tuned model, scaler, and feature columns
        self.model = load_model(f"models/{self.symbol}_finetuned_best.h5")
        self.scaler = joblib.load(f"models/{self.symbol}_finetuned_scaler.pkl")
        self.feature_columns = joblib.load(f"models/{self.symbol}_feature_columns.pkl")
        
        self.feature_engineer = AdvancedFeatureEngineer()

    def generate(self, hold_threshold=0.005):
        """Fetches latest data, predicts, and generates a trading recommendation."""
        print(f"📈 Generating recommendation for {self.symbol}...")
        
        # Fetch data for the last ~100 days to have enough for sequence generation
        end_date = datetime.now()
        start_date = end_date - timedelta(days=150)
        
        stock = yf.Ticker(self.symbol)
        df = stock.history(start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
        
        if df.empty:
            raise ValueError(f"No data found for {self.symbol}")
            
        # Get the most recent closing price
        last_close = df['Close'].iloc[-1]
        
        # Engineer features
        df_featured = self.feature_engineer.create_all_features(df.copy())
        df_featured.replace([np.inf, -np.inf], np.nan, inplace=True)
        df_featured.dropna(inplace=True)

        # One-hot encode and align columns
        categorical_cols = df_featured.select_dtypes(include=['object', 'category']).columns
        df_featured = pd.get_dummies(df_featured, columns=categorical_cols, drop_first=True)
        
        for col in self.feature_columns:
            if col not in df_featured.columns:
                df_featured[col] = 0
        df_featured = df_featured[self.feature_columns]

        # Prepare the last sequence for prediction
        sequence_length = self.model.input_shape[1]
        last_sequence = df_featured.iloc[-sequence_length:].values
        last_sequence_scaled = self.scaler.transform(last_sequence)
        
        X = np.array([last_sequence_scaled])
        
        # Predict
        prediction_scaled = self.model.predict(X)[0][0]
        
        # Inverse transform the prediction
        dummy_array = np.zeros((1, len(self.feature_columns)))
        close_idx = self.feature_columns.index('Close')
        dummy_array[0, close_idx] = prediction_scaled
        predicted_price = self.scaler.inverse_transform(dummy_array)[0, close_idx]

        # Generate recommendation
        price_change_pct = (predicted_price - last_close) / last_close
        
        if price_change_pct > hold_threshold:
            recommendation = "Buy"
        elif price_change_pct < -hold_threshold:
            recommendation = "Sell"
        else:
            recommendation = "Hold"
            
        print("\n" + "="*50)
        print(f"      RECOMMENDATION FOR {self.symbol}")
        print("="*50 + "\n")
        print(f"  Current Price:      ${last_close:.2f}")
        print(f"  Predicted Next Day:   ${predicted_price:.2f}")
        print(f"  Predicted Change:   {price_change_pct:+.2%}")
        print(f"  Recommendation:       {recommendation}")
        print("\n" + "="*50)
        
        return recommendation

def get_recommendation_for_stock(symbol):
    """Convenience function to generate a recommendation for a single stock."""
    generator = RecommendationGenerator(symbol=symbol)
    generator.generate()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, default="AAPL", help="Stock symbol for recommendation.")
    args = parser.parse_args()
    
    get_recommendation_for_stock(args.symbol) 