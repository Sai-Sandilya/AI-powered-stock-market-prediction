"""
Advanced Feature Engineering v2.0
Enhanced features for better stock prediction accuracy
"""

import pandas as pd
import numpy as np
import talib
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class AdvancedFeatureEngineer:
    def __init__(self):
        self.feature_names = []
        
    def add_market_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add market microstructure features for better prediction.
        """
        print("🔬 Adding market microstructure features...")
        
        # Volume-weighted average price (VWAP)
        df['VWAP'] = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()
        
        # Volume profile features
        df['Volume_MA_5'] = df['Volume'].rolling(5).mean()
        df['Volume_MA_20'] = df['Volume'].rolling(20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA_20']
        df['Volume_Spike'] = (df['Volume'] > df['Volume_MA_20'] * 2).astype(int)
        
        # Price-volume relationship
        df['Price_Volume_Trend'] = ((df['Close'] - df['Close'].shift(1)) * df['Volume']).cumsum()
        df['On_Balance_Volume'] = (np.sign(df['Close'].diff()) * df['Volume']).cumsum()
        
        # Money flow indicators
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        money_flow = typical_price * df['Volume']
        df['Money_Flow_Index'] = talib.MFI(df['High'], df['Low'], df['Close'], df['Volume'], timeperiod=14)
        
        # Accumulation/Distribution Line
        df['AD_Line'] = talib.AD(df['High'], df['Low'], df['Close'], df['Volume'])
        
        # Chaikin Money Flow
        df['CMF'] = talib.ADOSC(df['High'], df['Low'], df['Close'], df['Volume'], fastperiod=3, slowperiod=10)
        
        # Volume Rate of Change
        df['Volume_ROC'] = talib.ROC(df['Volume'], timeperiod=10)
        
        # Volume Weighted RSI
        df['Volume_Weighted_RSI'] = self.calculate_volume_weighted_rsi(df)
        
        # Volume Price Trend
        df['VPT'] = self.calculate_vpt(df)
        
        # Ease of Movement
        df['EOM'] = talib.ADX(df['High'], df['Low'], df['Close'], timeperiod=14)
        
        return df
    
    def add_advanced_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add advanced technical indicators for better pattern recognition.
        """
        print("📊 Adding advanced technical indicators...")
        
        # Elliott Wave-like features
        df['Price_Acceleration'] = df['Close'].diff().diff()
        df['Price_Momentum'] = df['Close'].diff(5)
        df['Price_Acceleration_MA'] = df['Price_Acceleration'].rolling(10).mean()
        
        # Harmonic patterns detection
        df['Harmonic_Retracement'] = self.calculate_harmonic_retracement(df)
        df['Fibonacci_Levels'] = self.calculate_fibonacci_levels(df)
        
        # Support and Resistance
        df['Support_Level'] = self.find_support_level(df)
        df['Resistance_Level'] = self.find_resistance_level(df)
        df['Price_vs_Support'] = (df['Close'] - df['Support_Level']) / df['Close']
        df['Price_vs_Resistance'] = (df['Resistance_Level'] - df['Close']) / df['Close']
        
        # Advanced oscillators
        df['Ultimate_Oscillator'] = talib.ULTOSC(df['High'], df['Low'], df['Close'], timeperiod1=7, timeperiod2=14, timeperiod3=28)
        df['Williams_%R'] = talib.WILLR(df['High'], df['Low'], df['Close'], timeperiod=14)
        df['CCI'] = talib.CCI(df['High'], df['Low'], df['Close'], timeperiod=14)
        
        # Parabolic SAR
        df['SAR'] = talib.SAR(df['High'], df['Low'], acceleration=0.02, maximum=0.2)
        df['SAR_Direction'] = np.where(df['Close'] > df['SAR'], 1, -1)
        
        # Average Directional Index
        df['ADX'] = talib.ADX(df['High'], df['Low'], df['Close'], timeperiod=14)
        df['Plus_DI'] = talib.PLUS_DI(df['High'], df['Low'], df['Close'], timeperiod=14)
        df['Minus_DI'] = talib.MINUS_DI(df['High'], df['Low'], df['Close'], timeperiod=14)
        
        # Aroon indicators
        df['Aroon_Up'] = talib.AROON(df['High'], timeperiod=14)[0]
        df['Aroon_Down'] = talib.AROON(df['Low'], timeperiod=14)[1]
        df['Aroon_Oscillator'] = talib.AROONOSC(df['High'], df['Low'], timeperiod=14)
        
        # Rate of Change variations
        df['ROC_5'] = talib.ROC(df['Close'], timeperiod=5)
        df['ROC_10'] = talib.ROC(df['Close'], timeperiod=10)
        df['ROC_20'] = talib.ROC(df['Close'], timeperiod=20)
        
        # Momentum indicators
        df['MOM'] = talib.MOM(df['Close'], timeperiod=10)
        df['TSF'] = talib.TSF(df['Close'], timeperiod=14)
        
        return df
    
    def add_pattern_recognition_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add pattern recognition features for technical analysis.
        """
        print("🔍 Adding pattern recognition features...")
        
        # Candlestick patterns
        df['Doji'] = talib.CDLDOJI(df['Open'], df['High'], df['Low'], df['Close'])
        df['Hammer'] = talib.CDLHAMMER(df['Open'], df['High'], df['Low'], df['Close'])
        df['Shooting_Star'] = talib.CDLSHOOTINGSTAR(df['Open'], df['High'], df['Low'], df['Close'])
        df['Engulfing'] = talib.CDLENGULFING(df['Open'], df['High'], df['Low'], df['Close'])
        df['Morning_Star'] = talib.CDLMORNINGSTAR(df['Open'], df['High'], df['Low'], df['Close'])
        df['Evening_Star'] = talib.CDLEVENINGSTAR(df['Open'], df['High'], df['Low'], df['Close'])
        
        # Price patterns
        df['Higher_High'] = (df['High'] > df['High'].shift(1)).astype(int)
        df['Lower_Low'] = (df['Low'] < df['Low'].shift(1)).astype(int)
        df['Higher_Low'] = (df['Low'] > df['Low'].shift(1)).astype(int)
        df['Lower_High'] = (df['High'] < df['High'].shift(1)).astype(int)
        
        # Trend strength
        df['Trend_Strength'] = self.calculate_trend_strength(df)
        df['Breakout_Strength'] = self.calculate_breakout_strength(df)
        
        # Gap analysis
        df['Gap_Up'] = (df['Open'] > df['Close'].shift(1)).astype(int)
        df['Gap_Down'] = (df['Open'] < df['Close'].shift(1)).astype(int)
        df['Gap_Size'] = abs(df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
        
        return df
    
    def add_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add statistical features for better prediction.
        """
        print("📈 Adding statistical features...")
        
        # Rolling statistics
        for window in [5, 10, 20, 50]:
            df[f'Rolling_Mean_{window}'] = df['Close'].rolling(window).mean()
            df[f'Rolling_Std_{window}'] = df['Close'].rolling(window).std()
            df[f'Rolling_Skew_{window}'] = df['Close'].rolling(window).skew()
            df[f'Rolling_Kurt_{window}'] = df['Close'].rolling(window).kurt()
            df[f'Z_Score_{window}'] = (df['Close'] - df[f'Rolling_Mean_{window}']) / df[f'Rolling_Std_{window}']
        
        # Volatility measures
        df['Realized_Volatility'] = df['Close'].pct_change().rolling(20).std() * np.sqrt(252)
        df['Parkinson_Volatility'] = np.sqrt((1/(4*np.log(2))) * ((np.log(df['High']/df['Low'])**2).rolling(20).mean()))
        df['Garman_Klass_Volatility'] = np.sqrt(((0.5 * (np.log(df['High']/df['Low'])**2)) - ((2*np.log(2)-1) * (np.log(df['Close']/df['Open'])**2))).rolling(20).mean())
        
        # Autocorrelation features
        df['Price_Autocorr_1'] = df['Close'].autocorr(lag=1)
        df['Price_Autocorr_5'] = df['Close'].autocorr(lag=5)
        df['Volume_Autocorr_1'] = df['Volume'].autocorr(lag=1)
        
        # Mean reversion indicators
        df['Mean_Reversion_5'] = (df['Close'] - df['Close'].rolling(5).mean()) / df['Close'].rolling(5).std()
        df['Mean_Reversion_20'] = (df['Close'] - df['Close'].rolling(20).mean()) / df['Close'].rolling(20).std()
        
        return df
    
    def add_sentiment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add sentiment-based features (simulated for now).
        """
        print("📰 Adding sentiment features...")
        
        # Simulate news sentiment based on price movement and volume
        price_change = df['Close'].pct_change()
        volume_spike = df['Volume'] / df['Volume'].rolling(20).mean()
        
        # Sentiment correlates with price movement and volume
        sentiment_base = np.tanh(price_change * 10) + np.tanh(volume_spike * 0.5)
        noise = np.random.normal(0, 0.1, len(df))
        df['News_Sentiment'] = np.clip(sentiment_base + noise, -1, 1)
        
        # Social media sentiment (simulated)
        df['Social_Sentiment'] = np.random.normal(0, 0.2, len(df))
        
        # Earnings sentiment (simulated)
        df['Earnings_Sentiment'] = np.random.normal(0, 0.3, len(df))
        
        # Combined sentiment score
        df['Combined_Sentiment'] = (df['News_Sentiment'] * 0.4 + 
                                   df['Social_Sentiment'] * 0.3 + 
                                   df['Earnings_Sentiment'] * 0.3)
        
        # Sentiment momentum
        df['Sentiment_Momentum'] = df['Combined_Sentiment'].diff(5)
        df['Sentiment_MA'] = df['Combined_Sentiment'].rolling(10).mean()
        
        return df
    
    def add_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add market regime detection features.
        """
        print("🌍 Adding market regime features...")
        
        # Volatility regime
        df['Volatility_Regime'] = np.where(df['Realized_Volatility'] > df['Realized_Volatility'].rolling(50).quantile(0.75), 'High',
                                          np.where(df['Realized_Volatility'] < df['Realized_Volatility'].rolling(50).quantile(0.25), 'Low', 'Medium'))
        
        # Trend regime
        df['Trend_Regime'] = np.where(df['Close'] > df['Close'].rolling(50).mean() * 1.02, 'Bull',
                                     np.where(df['Close'] < df['Close'].rolling(50).mean() * 0.98, 'Bear', 'Sideways'))
        
        # Volume regime
        df['Volume_Regime'] = np.where(df['Volume'] > df['Volume'].rolling(20).quantile(0.75), 'High',
                                      np.where(df['Volume'] < df['Volume'].rolling(20).quantile(0.25), 'Low', 'Normal'))
        
        # Market efficiency
        df['Market_Efficiency'] = self.calculate_market_efficiency(df)
        
        # Regime change indicators
        df['Regime_Change'] = ((df['Volatility_Regime'] != df['Volatility_Regime'].shift(1)) | 
                              (df['Trend_Regime'] != df['Trend_Regime'].shift(1))).astype(int)
        
        return df
    
    def add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add interaction features between different indicators.
        """
        print("🔗 Adding interaction features...")
        
        # Price-Volume interactions
        df['Price_Volume_Correlation'] = df['Close'].rolling(20).corr(df['Volume'])
        df['Price_Volume_Trend_Strength'] = df['Price_Volume_Trend'] / df['Volume'].rolling(20).sum()
        
        # Technical indicator interactions
        df['RSI_MACD_Interaction'] = df['RSI_14'] * df['MACD']
        df['Bollinger_RSI_Interaction'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower']) * df['RSI_14']
        
        # Momentum interactions
        df['Momentum_Acceleration'] = df['Price_Momentum'].diff()
        df['Volume_Momentum'] = df['Volume_ROC'] * df['Price_Momentum']
        
        # Sentiment interactions
        df['Sentiment_Price_Interaction'] = df['Combined_Sentiment'] * df['Close'].pct_change()
        df['Sentiment_Volume_Interaction'] = df['Combined_Sentiment'] * df['Volume_Ratio']
        
        return df
    
    def add_lag_features(self, df: pd.DataFrame, max_lag: int = 10) -> pd.DataFrame:
        """
        Add lagged features for time series prediction.
        """
        print(f"⏰ Adding lag features (max_lag={max_lag})...")
        
        # Price lags
        for lag in range(1, max_lag + 1):
            df[f'Price_Lag_{lag}'] = df['Close'].shift(lag)
            df[f'Volume_Lag_{lag}'] = df['Volume'].shift(lag)
            df[f'Returns_Lag_{lag}'] = df['Close'].pct_change().shift(lag)
        
        # Technical indicator lags
        for lag in [1, 2, 3, 5]:
            df[f'RSI_Lag_{lag}'] = df['RSI_14'].shift(lag)
            df[f'MACD_Lag_{lag}'] = df['MACD'].shift(lag)
            df[f'BB_Position_Lag_{lag}'] = ((df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])).shift(lag)
        
        return df
    
    def calculate_volume_weighted_rsi(self, df: pd.DataFrame) -> pd.Series:
        """Calculate volume-weighted RSI."""
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0) * df['Volume']).rolling(14).sum()
        loss = (-delta.where(delta < 0, 0) * df['Volume']).rolling(14).sum()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def calculate_vpt(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Volume Price Trend."""
        return ((df['Close'] - df['Close'].shift(1)) / df['Close'].shift(1) * df['Volume']).cumsum()
    
    def calculate_harmonic_retracement(self, df: pd.DataFrame) -> pd.Series:
        """Calculate harmonic retracement levels."""
        high = df['High'].rolling(20).max()
        low = df['Low'].rolling(20).min()
        return (df['Close'] - low) / (high - low)
    
    def calculate_fibonacci_levels(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Fibonacci retracement levels."""
        high = df['High'].rolling(20).max()
        low = df['Low'].rolling(20).min()
        range_size = high - low
        return (df['Close'] - low) / range_size
    
    def find_support_level(self, df: pd.DataFrame) -> pd.Series:
        """Find dynamic support levels."""
        return df['Low'].rolling(20).min()
    
    def find_resistance_level(self, df: pd.DataFrame) -> pd.Series:
        """Find dynamic resistance levels."""
        return df['High'].rolling(20).max()
    
    def calculate_trend_strength(self, df: pd.DataFrame) -> pd.Series:
        """Calculate trend strength indicator."""
        price_change = df['Close'].diff(20)
        volatility = df['Close'].rolling(20).std()
        return price_change / volatility
    
    def calculate_breakout_strength(self, df: pd.DataFrame) -> pd.Series:
        """Calculate breakout strength."""
        bb_position = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        return np.where(bb_position > 1, bb_position - 1, 
                       np.where(bb_position < 0, -bb_position, 0))
    
    def calculate_market_efficiency(self, df: pd.DataFrame) -> pd.Series:
        """Calculate market efficiency ratio."""
        price_change = df['Close'].diff()
        path_length = price_change.abs().rolling(20).sum()
        net_change = price_change.rolling(20).sum().abs()
        return net_change / path_length
    
    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all advanced features for the dataset.
        """
        print("🚀 Creating comprehensive feature set...")
        
        # Add all feature categories
        df = self.add_market_microstructure_features(df)
        df = self.add_advanced_technical_indicators(df)
        df = self.add_pattern_recognition_features(df)
        df = self.add_statistical_features(df)
        df = self.add_sentiment_features(df)
        df = self.add_regime_features(df)
        df = self.add_interaction_features(df)
        df = self.add_lag_features(df)
        
        # Clean up NaN values
        df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # Store feature names
        self.feature_names = [col for col in df.columns if col not in ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        
        print(f"✅ Created {len(self.feature_names)} advanced features")
        return df
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names."""
        return self.feature_names

def create_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience function to create all advanced features.
    """
    engineer = AdvancedFeatureEngineer()
    return engineer.create_all_features(df) 