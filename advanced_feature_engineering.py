"""
Advanced Feature Engineering Module
Includes sentiment analysis, market microstructure features, and sophisticated
technical indicators for superior stock prediction performance.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from textblob import TextBlob
import requests
from typing import Dict, List, Optional, Tuple
import warnings
import talib
from scipy import stats
from scipy.signal import savgol_filter

warnings.filterwarnings('ignore')

class AdvancedFeatureEngineer:
    def __init__(self):
        self.news_api_key = None  # Set your News API key here
        self.sentiment_cache = {}
        self.feature_cache = {}
        
    def add_sentiment_features(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Add sentiment features from news and social media.
        """
        print("📰 Adding sentiment features...")
        
        # Simulate sentiment data (in production, use real APIs)
        dates = df['Date'].tolist()
        sentiment_scores = []
        news_volume = []
        social_sentiment = []
        
        for i, date in enumerate(dates):
            # Simulate sentiment based on price movement and volatility
            if i > 0:
                price_change = (df.iloc[i]['Close'] - df.iloc[i-1]['Close']) / df.iloc[i-1]['Close']
                volatility = abs(price_change)
                
                # Sentiment correlates with price movement and volatility
                base_sentiment = np.tanh(price_change * 10)
                volatility_factor = np.tanh(volatility * 20)
                sentiment = base_sentiment + volatility_factor + np.random.normal(0, 0.1)
                
                sentiment_scores.append(sentiment)
                news_volume.append(np.random.poisson(5) + abs(price_change) * 100)
                social_sentiment.append(sentiment + np.random.normal(0, 0.2))
            else:
                sentiment_scores.append(0)
                news_volume.append(5)
                social_sentiment.append(0)
        
        df['sentiment_score'] = sentiment_scores
        df['news_volume'] = news_volume
        df['social_sentiment'] = social_sentiment
        
        # Sentiment derivatives
        df['sentiment_momentum'] = df['sentiment_score'].rolling(5).mean()
        df['sentiment_volatility'] = df['sentiment_score'].rolling(10).std()
        df['sentiment_acceleration'] = df['sentiment_score'].diff().rolling(3).mean()
        df['sentiment_divergence'] = df['sentiment_score'] - df['sentiment_momentum']
        
        # News sentiment features
        df['news_sentiment_ratio'] = df['news_volume'] / df['news_volume'].rolling(20).mean()
        df['news_sentiment_trend'] = df['news_volume'].rolling(5).mean() - df['news_volume'].rolling(20).mean()
        
        return df
    
    def add_market_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add advanced market microstructure features.
        """
        print("🔬 Adding market microstructure features...")
        
        # Volume-based features
        df['volume_ma_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
        df['volume_price_trend'] = df['Volume'] * df['Close'].pct_change()
        df['volume_weighted_price'] = (df['Volume'] * df['Close']).rolling(5).sum() / df['Volume'].rolling(5).sum()
        df['volume_price_correlation'] = df['Volume'].rolling(10).corr(df['Close'])
        
        # Price efficiency features
        df['price_efficiency'] = abs(df['Close'].pct_change()) / df['Close'].pct_change().rolling(20).std()
        df['price_momentum'] = df['Close'].pct_change(5)
        df['price_reversal'] = -df['Close'].pct_change(5)  # Mean reversion signal
        df['price_acceleration'] = df['Close'].pct_change().diff().rolling(3).mean()
        
        # Volatility features
        df['realized_volatility'] = df['Close'].pct_change().rolling(20).std() * np.sqrt(252)
        df['volatility_ratio'] = df['realized_volatility'] / df['realized_volatility'].rolling(60).mean()
        df['volatility_of_volatility'] = df['realized_volatility'].rolling(10).std()
        df['volatility_regime'] = pd.cut(df['realized_volatility'], 
                                       bins=[0, 0.15, 0.25, 0.35, 1.0], 
                                       labels=[0, 1, 2, 3])
        
        # Liquidity features
        df['amihud_illiquidity'] = abs(df['Close'].pct_change()) / (df['Volume'] / 1000000)
        df['turnover_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
        df['bid_ask_spread'] = (df['High'] - df['Low']) / df['Close']  # Proxy for spread
        df['liquidity_ratio'] = df['Volume'] / df['amihud_illiquidity']
        
        # Market impact features
        df['price_impact'] = df['Volume'] * df['Close'].pct_change().abs()
        df['market_impact_ratio'] = df['price_impact'] / df['Volume'].rolling(20).mean()
        
        return df
    
    def add_advanced_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add sophisticated technical indicators using TA-Lib.
        """
        print("📊 Adding advanced technical indicators...")
        
        # Moving averages
        df['sma_5'] = talib.SMA(df['Close'], timeperiod=5)
        df['sma_10'] = talib.SMA(df['Close'], timeperiod=10)
        df['sma_20'] = talib.SMA(df['Close'], timeperiod=20)
        df['sma_50'] = talib.SMA(df['Close'], timeperiod=50)
        df['ema_12'] = talib.EMA(df['Close'], timeperiod=12)
        df['ema_26'] = talib.EMA(df['Close'], timeperiod=26)
        
        # MACD
        macd, macd_signal, macd_hist = talib.MACD(df['Close'])
        df['macd'] = macd
        df['macd_signal'] = macd_signal
        df['macd_histogram'] = macd_hist
        df['macd_divergence'] = macd - macd_signal
        
        # RSI with multiple timeframes
        df['rsi_14'] = talib.RSI(df['Close'], timeperiod=14)
        df['rsi_21'] = talib.RSI(df['Close'], timeperiod=21)
        df['rsi_divergence'] = df['rsi_14'] - df['rsi_21']
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = talib.BBANDS(df['Close'])
        df['bb_upper'] = bb_upper
        df['bb_middle'] = bb_middle
        df['bb_lower'] = bb_lower
        df['bb_position'] = (df['Close'] - bb_lower) / (bb_upper - bb_lower)
        df['bb_squeeze'] = (bb_upper - bb_lower) / bb_middle
        
        # Stochastic Oscillator
        stoch_k, stoch_d = talib.STOCH(df['High'], df['Low'], df['Close'])
        df['stoch_k'] = stoch_k
        df['stoch_d'] = stoch_d
        df['stoch_signal'] = stoch_k - stoch_d
        
        # Williams %R
        df['williams_r'] = talib.WILLR(df['High'], df['Low'], df['Close'])
        
        # Average True Range (ATR)
        df['atr'] = talib.ATR(df['High'], df['Low'], df['Close'])
        df['atr_ratio'] = df['atr'] / df['Close']
        
        # Commodity Channel Index (CCI)
        df['cci'] = talib.CCI(df['High'], df['Low'], df['Close'])
        
        # Money Flow Index (MFI)
        df['mfi'] = talib.MFI(df['High'], df['Low'], df['Close'], df['Volume'])
        
        # On-Balance Volume (OBV)
        df['obv'] = talib.OBV(df['Close'], df['Volume'])
        df['obv_ma'] = df['obv'].rolling(20).mean()
        df['obv_signal'] = (df['obv'] - df['obv_ma']) / df['obv_ma']
        
        # Parabolic SAR
        df['sar'] = talib.SAR(df['High'], df['Low'])
        df['sar_signal'] = np.where(df['Close'] > df['sar'], 1, -1)
        
        # ADX (Average Directional Index)
        df['adx'] = talib.ADX(df['High'], df['Low'], df['Close'])
        
        # Aroon Oscillator
        aroon_down, aroon_up = talib.AROON(df['High'], df['Low'])
        df['aroon_oscillator'] = aroon_up - aroon_down
        
        # Chaikin Money Flow
        df['cmf'] = talib.ADOSC(df['High'], df['Low'], df['Close'], df['Volume'])
        
        return df
    
    def add_pattern_recognition(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add advanced pattern recognition features.
        """
        print("🔍 Adding pattern recognition features...")
        
        # Candlestick patterns
        df['doji'] = talib.CDLDOJI(df['Open'], df['High'], df['Low'], df['Close'])
        df['hammer'] = talib.CDLHAMMER(df['Open'], df['High'], df['Low'], df['Close'])
        df['shooting_star'] = talib.CDLSHOOTINGSTAR(df['Open'], df['High'], df['Low'], df['Close'])
        df['engulfing'] = talib.CDLENGULFING(df['Open'], df['High'], df['Low'], df['Close'])
        df['morning_star'] = talib.CDLMORNINGSTAR(df['Open'], df['High'], df['Low'], df['Close'])
        df['evening_star'] = talib.CDLEVENINGSTAR(df['Open'], df['High'], df['Low'], df['Close'])
        
        # Price patterns
        df['body_size'] = abs(df['Close'] - df['Open'])
        df['upper_shadow'] = df['High'] - np.maximum(df['Open'], df['Close'])
        df['lower_shadow'] = np.minimum(df['Open'], df['Close']) - df['Low']
        df['body_ratio'] = df['body_size'] / (df['High'] - df['Low'])
        
        # Support and resistance levels
        df['support_level'] = df['Low'].rolling(20).min()
        df['resistance_level'] = df['High'].rolling(20).max()
        df['support_distance'] = (df['Close'] - df['support_level']) / df['Close']
        df['resistance_distance'] = (df['resistance_level'] - df['Close']) / df['Close']
        
        # Trend strength
        df['trend_strength'] = abs(df['Close'].pct_change(20).rolling(10).mean())
        df['trend_consistency'] = df['Close'].pct_change().rolling(10).apply(
            lambda x: np.sum(np.sign(x) == np.sign(x.iloc[0])) / len(x)
        )
        
        return df
    
    def add_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add statistical and mathematical features.
        """
        print("📈 Adding statistical features...")
        
        # Rolling statistics
        for window in [5, 10, 20]:
            df[f'close_mean_{window}'] = df['Close'].rolling(window).mean()
            df[f'close_std_{window}'] = df['Close'].rolling(window).std()
            df[f'close_skew_{window}'] = df['Close'].rolling(window).skew()
            df[f'close_kurt_{window}'] = df['Close'].rolling(window).kurt()
            
            # Z-score
            df[f'close_zscore_{window}'] = (df['Close'] - df[f'close_mean_{window}']) / df[f'close_std_{window}']
            
            # Percentile rank
            df[f'close_percentile_{window}'] = df['Close'].rolling(window).rank(pct=True)
        
        # Momentum indicators
        df['momentum_5'] = df['Close'] / df['Close'].shift(5) - 1
        df['momentum_10'] = df['Close'] / df['Close'].shift(10) - 1
        df['momentum_20'] = df['Close'] / df['Close'].shift(20) - 1
        
        # Rate of change
        df['roc_5'] = talib.ROC(df['Close'], timeperiod=5)
        df['roc_10'] = talib.ROC(df['Close'], timeperiod=10)
        df['roc_20'] = talib.ROC(df['Close'], timeperiod=20)
        
        # Smoothing
        df['close_smooth'] = savgol_filter(df['Close'], window_length=5, polyorder=2)
        df['volume_smooth'] = savgol_filter(df['Volume'], window_length=5, polyorder=2)
        
        return df
    
    def add_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add market regime and regime change features.
        """
        print("🏛️ Adding regime features...")
        
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
        df['momentum_regime'] = pd.cut(df['rsi_14'], 
                                     bins=[0, 30, 40, 60, 70, 100], 
                                     labels=['oversold', 'weak_bear', 'neutral', 'weak_bull', 'overbought'])
        
        # Market regime change detection
        df['regime_change'] = (
            (df['volatility_regime'] != df['volatility_regime'].shift(1)) |
            (df['trend_regime'] != df['trend_regime'].shift(1)) |
            (df['momentum_regime'] != df['momentum_regime'].shift(1))
        ).astype(int)
        
        # Convert categorical to numeric
        df['volatility_regime_num'] = df['volatility_regime'].cat.codes
        df['trend_regime_num'] = df['trend_regime'].cat.codes
        df['momentum_regime_num'] = df['momentum_regime'].cat.codes
        
        return df
    
    def add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add interaction features between different indicators.
        """
        print("🔗 Adding interaction features...")
        
        # Volume-price interactions
        df['volume_price_interaction'] = df['volume_ma_ratio'] * df['price_momentum']
        df['volume_rsi_interaction'] = df['volume_ma_ratio'] * df['rsi_14']
        df['volume_macd_interaction'] = df['volume_ma_ratio'] * df['macd']
        
        # Sentiment interactions
        df['sentiment_volume_interaction'] = df['sentiment_score'] * df['volume_ma_ratio']
        df['sentiment_price_interaction'] = df['sentiment_score'] * df['price_momentum']
        df['sentiment_rsi_interaction'] = df['sentiment_score'] * df['rsi_14']
        
        # Technical indicator interactions
        df['rsi_macd_interaction'] = df['rsi_14'] * df['macd']
        df['bb_rsi_interaction'] = df['bb_position'] * df['rsi_14']
        df['stoch_rsi_interaction'] = df['stoch_k'] * df['rsi_14']
        
        # Volatility interactions
        df['volatility_price_interaction'] = df['realized_volatility'] * df['price_momentum']
        df['volatility_volume_interaction'] = df['realized_volatility'] * df['volume_ma_ratio']
        
        return df
    
    def add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add lagged features for time series analysis.
        """
        print("⏰ Adding lag features...")
        
        # Price lags
        for lag in [1, 2, 3, 5, 10]:
            df[f'close_lag_{lag}'] = df['Close'].shift(lag)
            df[f'volume_lag_{lag}'] = df['Volume'].shift(lag)
            df[f'rsi_lag_{lag}'] = df['rsi_14'].shift(lag)
            df[f'macd_lag_{lag}'] = df['macd'].shift(lag)
            df[f'sentiment_lag_{lag}'] = df['sentiment_score'].shift(lag)
        
        # Rolling statistics lags
        for window in [5, 10, 20]:
            df[f'close_ma_lag_{window}'] = df[f'close_mean_{window}'].shift(1)
            df[f'volume_ma_lag_{window}'] = df['Volume'].rolling(window).mean().shift(1)
        
        return df
    
    def create_advanced_features(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Create comprehensive advanced features.
        """
        print(f"🚀 Creating advanced features for {symbol}...")
        
        # Add all feature categories
        df = self.add_sentiment_features(df, symbol)
        df = self.add_market_microstructure_features(df)
        df = self.add_advanced_technical_indicators(df)
        df = self.add_pattern_recognition(df)
        df = self.add_statistical_features(df)
        df = self.add_regime_features(df)
        df = self.add_interaction_features(df)
        df = self.add_lag_features(df)
        
        # Remove any remaining NaN values
        df = df.dropna()
        
        print(f"✅ Created {len(df.columns)} advanced features")
        
        return df

def test_advanced_features():
    """Test the advanced feature engineering."""
    print("🧪 Testing advanced feature engineering...")
    
    # Fetch sample data
    symbol = 'AAPL'
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    df = yf.download(symbol, start=start_date, end=end_date, progress=False)
    df = df.reset_index()
    df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    
    # Create advanced features
    engineer = AdvancedFeatureEngineer()
    advanced_df = engineer.create_advanced_features(df, symbol)
    
    print(f"📊 Original features: {len(df.columns)}")
    print(f"📊 Advanced features: {len(advanced_df.columns)}")
    print(f"📊 New features added: {len(advanced_df.columns) - len(df.columns)}")
    
    # Show some key features
    key_features = ['sentiment_score', 'volume_ma_ratio', 'rsi_14', 'macd', 
                   'bb_position', 'atr_ratio', 'cci', 'mfi', 'obv_signal',
                   'volatility_regime_num', 'trend_regime_num', 'momentum_regime_num']
    
    print("\n🔍 Sample advanced features:")
    for feature in key_features:
        if feature in advanced_df.columns:
            value = advanced_df[feature].iloc[-1]
            print(f"   {feature}: {value:.4f}")
    
    return advanced_df

if __name__ == "__main__":
    test_advanced_features() 