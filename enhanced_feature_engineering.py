"""
Enhanced Feature Engineering Module
Adds advanced features including sentiment analysis, market microstructure,
and sophisticated technical indicators for improved stock prediction.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from textblob import TextBlob
import requests
from typing import Dict, List, Optional, Tuple
import warnings

warnings.filterwarnings('ignore')

class EnhancedFeatureEngineer:
    def __init__(self):
        self.news_api_key = None  # Set your News API key here
        self.sentiment_cache = {}
    
    def add_sentiment_features(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Add sentiment features from news and social media.
        """
        print("📰 Adding sentiment features...")
        
        # Simulate sentiment data (in production, use real APIs)
        dates = df['Date'].tolist()
        sentiment_scores = []
        news_volume = []
        
        for date in dates:
            # Simulate sentiment based on price movement
            idx = df[df['Date'] == date].index[0]
            if idx > 0:
                price_change = (df.iloc[idx]['Close'] - df.iloc[idx-1]['Close']) / df.iloc[idx-1]['Close']
                # Sentiment correlates with price movement
                sentiment = np.tanh(price_change * 10) + np.random.normal(0, 0.1)
                sentiment_scores.append(sentiment)
                news_volume.append(np.random.poisson(5) + abs(price_change) * 100)
            else:
                sentiment_scores.append(0)
                news_volume.append(5)
        
        df['sentiment_score'] = sentiment_scores
        df['news_volume'] = news_volume
        df['sentiment_momentum'] = df['sentiment_score'].rolling(5).mean()
        df['sentiment_volatility'] = df['sentiment_score'].rolling(10).std()
        
        return df
    
    def add_market_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add market microstructure features.
        """
        print("🔬 Adding market microstructure features...")
        
        # Volume-based features
        df['volume_ma_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
        df['volume_price_trend'] = df['Volume'] * df['Close'].pct_change()
        df['volume_weighted_price'] = (df['Volume'] * df['Close']).rolling(5).sum() / df['Volume'].rolling(5).sum()
        
        # Price efficiency features
        df['price_efficiency'] = abs(df['Close'].pct_change()) / df['Close'].pct_change().rolling(20).std()
        df['price_momentum'] = df['Close'].pct_change(5)
        df['price_reversal'] = -df['Close'].pct_change(5)  # Mean reversion signal
        
        # Volatility features
        df['realized_volatility'] = df['Close'].pct_change().rolling(20).std() * np.sqrt(252)
        df['volatility_ratio'] = df['realized_volatility'] / df['realized_volatility'].rolling(60).mean()
        
        # Liquidity features
        df['amihud_illiquidity'] = abs(df['Close'].pct_change()) / (df['Volume'] / 1000000)
        df['turnover_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
        
        return df
    
    def add_advanced_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add advanced technical indicators.
        """
        print("📊 Adding advanced technical indicators...")
        
        # Advanced moving averages
        df['ema_12'] = df['Close'].ewm(span=12).mean()
        df['ema_26'] = df['Close'].ewm(span=26).mean()
        df['ema_cross'] = (df['ema_12'] - df['ema_26']) / df['ema_26']
        
        # MACD with signal line
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # RSI with multiple timeframes
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        df['rsi_ma'] = df['rsi'].rolling(10).mean()
        df['rsi_divergence'] = df['rsi'] - df['rsi_ma']
        
        # Bollinger Bands
        df['bb_middle'] = df['Close'].rolling(20).mean()
        bb_std = df['Close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        df['bb_squeeze'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        # Stochastic Oscillator
        low_min = df['Low'].rolling(14).min()
        high_max = df['High'].rolling(14).max()
        df['stoch_k'] = 100 * ((df['Close'] - low_min) / (high_max - low_min))
        df['stoch_d'] = df['stoch_k'].rolling(3).mean()
        df['stoch_signal'] = df['stoch_d'].rolling(3).mean()
        
        # Williams %R
        df['williams_r'] = -100 * ((high_max - df['Close']) / (high_max - low_min))
        
        # Average True Range (ATR)
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        df['atr'] = true_range.rolling(14).mean()
        df['atr_ratio'] = df['atr'] / df['Close']
        
        # Commodity Channel Index (CCI)
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        sma_tp = typical_price.rolling(20).mean()
        mad = typical_price.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())))
        df['cci'] = (typical_price - sma_tp) / (0.015 * mad)
        
        # Money Flow Index (MFI)
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        money_flow = typical_price * df['Volume']
        
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(14).sum()
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(14).sum()
        
        mfi_ratio = positive_flow / negative_flow
        df['mfi'] = 100 - (100 / (1 + mfi_ratio))
        
        # On-Balance Volume (OBV)
        df['obv'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
        df['obv_ma'] = df['obv'].rolling(20).mean()
        df['obv_signal'] = (df['obv'] - df['obv_ma']) / df['obv_ma']
        
        return df
    
    def add_pattern_recognition(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add pattern recognition features.
        """
        print("🔍 Adding pattern recognition features...")
        
        # Candlestick patterns
        df['body_size'] = abs(df['Close'] - df['Open'])
        df['upper_shadow'] = df['High'] - np.maximum(df['Open'], df['Close'])
        df['lower_shadow'] = np.minimum(df['Open'], df['Close']) - df['Low']
        df['body_ratio'] = df['body_size'] / (df['High'] - df['Low'])
        
        # Doji pattern
        df['is_doji'] = (df['body_ratio'] < 0.1).astype(int)
        
        # Hammer pattern
        df['is_hammer'] = ((df['lower_shadow'] > 2 * df['body_size']) & 
                          (df['upper_shadow'] < df['body_size'])).astype(int)
        
        # Shooting star pattern
        df['is_shooting_star'] = ((df['upper_shadow'] > 2 * df['body_size']) & 
                                 (df['lower_shadow'] < df['body_size'])).astype(int)
        
        # Engulfing patterns
        df['bullish_engulfing'] = ((df['Close'] > df['Open']) & 
                                  (df['Close'].shift(1) < df['Open'].shift(1)) &
                                  (df['Close'] > df['Open'].shift(1)) &
                                  (df['Open'] < df['Close'].shift(1))).astype(int)
        
        df['bearish_engulfing'] = ((df['Close'] < df['Open']) & 
                                  (df['Close'].shift(1) > df['Open'].shift(1)) &
                                  (df['Close'] < df['Open'].shift(1)) &
                                  (df['Open'] > df['Close'].shift(1))).astype(int)
        
        # Support and resistance levels
        df['support_level'] = df['Low'].rolling(20).min()
        df['resistance_level'] = df['High'].rolling(20).max()
        df['support_distance'] = (df['Close'] - df['support_level']) / df['Close']
        df['resistance_distance'] = (df['resistance_level'] - df['Close']) / df['Close']
        
        return df
    
    def add_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add market regime features.
        """
        print("🏛️ Adding market regime features...")
        
        # Volatility regime
        df['volatility_regime'] = pd.cut(df['realized_volatility'], 
                                       bins=[0, 0.15, 0.25, 0.35, 1.0], 
                                       labels=['low', 'medium', 'high', 'extreme'])
        
        # Trend regime
        df['trend_strength'] = abs(df['Close'].pct_change(20).rolling(10).mean())
        df['trend_regime'] = pd.cut(df['trend_strength'], 
                                  bins=[0, 0.01, 0.02, 0.05, 1.0], 
                                  labels=['sideways', 'weak_trend', 'strong_trend', 'extreme_trend'])
        
        # Momentum regime
        df['momentum_regime'] = pd.cut(df['rsi'], 
                                     bins=[0, 30, 40, 60, 70, 100], 
                                     labels=['oversold', 'weak_bear', 'neutral', 'weak_bull', 'overbought'])
        
        # Convert categorical to numeric
        df['volatility_regime_num'] = df['volatility_regime'].cat.codes
        df['trend_regime_num'] = df['trend_regime'].cat.codes
        df['momentum_regime_num'] = df['momentum_regime'].cat.codes
        
        return df
    
    def create_enhanced_features(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Create comprehensive enhanced features.
        """
        print(f"🚀 Creating enhanced features for {symbol}...")
        
        # Add all feature categories
        df = self.add_sentiment_features(df, symbol)
        df = self.add_market_microstructure_features(df)
        df = self.add_advanced_technical_indicators(df)
        df = self.add_pattern_recognition(df)
        df = self.add_regime_features(df)
        
        # Create interaction features
        df['volume_price_interaction'] = df['volume_ma_ratio'] * df['price_momentum']
        df['sentiment_volume_interaction'] = df['sentiment_score'] * df['volume_ma_ratio']
        df['rsi_volume_interaction'] = df['rsi'] * df['volume_ma_ratio']
        
        # Lag features for time series
        for lag in [1, 2, 3, 5]:
            df[f'close_lag_{lag}'] = df['Close'].shift(lag)
            df[f'volume_lag_{lag}'] = df['Volume'].shift(lag)
            df[f'rsi_lag_{lag}'] = df['rsi'].shift(lag)
        
        # Rolling statistics
        for window in [5, 10, 20]:
            df[f'close_ma_{window}'] = df['Close'].rolling(window).mean()
            df[f'volume_ma_{window}'] = df['Volume'].rolling(window).mean()
            df[f'rsi_ma_{window}'] = df['rsi'].rolling(window).mean()
            df[f'volatility_{window}'] = df['Close'].pct_change().rolling(window).std()
        
        print(f"✅ Created {len(df.columns)} enhanced features")
        
        return df

def test_enhanced_features():
    """Test the enhanced feature engineering."""
    print("🧪 Testing enhanced feature engineering...")
    
    # Fetch sample data
    symbol = 'AAPL'
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    df = yf.download(symbol, start=start_date, end=end_date, progress=False)
    df = df.reset_index()
    df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    
    # Create enhanced features
    engineer = EnhancedFeatureEngineer()
    enhanced_df = engineer.create_enhanced_features(df, symbol)
    
    print(f"📊 Original features: {len(df.columns)}")
    print(f"📊 Enhanced features: {len(enhanced_df.columns)}")
    print(f"📊 New features added: {len(enhanced_df.columns) - len(df.columns)}")
    
    # Show some key features
    key_features = ['sentiment_score', 'volume_ma_ratio', 'rsi', 'macd', 
                   'bb_position', 'atr_ratio', 'cci', 'mfi', 'obv_signal',
                   'is_hammer', 'support_distance', 'volatility_regime_num']
    
    print("\n🔍 Sample enhanced features:")
    for feature in key_features:
        if feature in enhanced_df.columns:
            value = enhanced_df[feature].iloc[-1]
            print(f"   {feature}: {value:.4f}")
    
    return enhanced_df

if __name__ == "__main__":
    test_enhanced_features() 