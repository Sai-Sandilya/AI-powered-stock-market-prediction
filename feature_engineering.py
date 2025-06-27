"""
Enhanced feature engineering with advanced technical indicators and macroeconomic data:
- Moving Averages (Simple, Exponential)
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- Stochastic Oscillator
- Average True Range (ATR)
- Williams %R
- Volatility measures
- Macroeconomic indicators (GDP, Inflation, Interest Rates, etc.)
- Market regime features
- Sector rotation signals
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Optional, Dict, List
import warnings
warnings.filterwarnings('ignore')

# Import macro indicators
try:
    from macro_indicators import MacroIndicators
    MACRO_AVAILABLE = True
except ImportError:
    MACRO_AVAILABLE = False
    print("Warning: MacroIndicators not available. Install macro_indicators.py for full functionality.")

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.rolling(window=period, min_periods=period).mean()
    ma_down = down.rolling(window=period, min_periods=period).mean()
    rs = ma_up / ma_down
    return 100 - (100 / (1 + rs))

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add comprehensive technical indicators to the dataframe.
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        DataFrame with technical indicators added
    """
    if df.empty:
        return df
    
    # Ensure we have the required columns
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Copy dataframe to avoid modifying original
    df = df.copy()
    
    # Basic Moving Averages
    df['MA_20'] = df['Close'].rolling(window=20).mean()
    df['MA_50'] = df['Close'].rolling(window=50).mean()
    df['MA_200'] = df['Close'].rolling(window=200).mean()
    
    # Exponential Moving Averages
    df['EMA_12'] = df['Close'].ewm(span=12).mean()
    df['EMA_26'] = df['Close'].ewm(span=26).mean()
    
    # RSI (Relative Strength Index)
    df['RSI_14'] = calculate_rsi(df['Close'].astype(float), window=14)
    
    # MACD (Moving Average Convergence Divergence)
    df['MACD'], df['MACD_Signal'], df['MACD_Histogram'] = calculate_macd(df['Close'].astype(float))
    
    # Bollinger Bands
    df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = calculate_bollinger_bands(df['Close'].astype(float))
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
    df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
    
    # Stochastic Oscillator
    df['Stoch_K'], df['Stoch_D'] = calculate_stochastic(
        df['High'].astype(float), 
        df['Low'].astype(float), 
        df['Close'].astype(float)
    )
    
    # Average True Range (ATR)
    df['ATR_14'] = calculate_atr(
        df['High'].astype(float), 
        df['Low'].astype(float), 
        df['Close'].astype(float), 
        window=14
    )
    
    # Williams %R
    df['Williams_R'] = calculate_williams_r(
        df['High'].astype(float), 
        df['Low'].astype(float), 
        df['Close'].astype(float)
    )
    
    # Volatility measures
    df['Volatility'] = df['Close'].pct_change().rolling(window=20).std()
    df['Price_Range'] = (df['High'] - df['Low']) / df['Close']
    
    # Volume indicators
    df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
    
    # Price momentum
    df['Momentum'] = df['Close'] - df['Close'].shift(10)
    df['ROC'] = ((df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10)) * 100
    
    # Support and Resistance levels (simplified)
    df['Support_Level'] = df['Low'].rolling(window=20).min()
    df['Resistance_Level'] = df['High'].rolling(window=20).max()
    
    # Fill NaN values with backward fill
    return df.bfill()

def calculate_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    """Calculate RSI (Relative Strength Index)."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
    """Calculate MACD, Signal line, and Histogram."""
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram

def calculate_bollinger_bands(prices: pd.Series, window: int = 20, num_std: float = 2) -> tuple:
    """Calculate Bollinger Bands (Upper, Middle, Lower)."""
    middle = prices.rolling(window=window).mean()
    std = prices.rolling(window=window).std()
    upper = middle + (std * num_std)
    lower = middle - (std * num_std)
    return upper, middle, lower

def calculate_stochastic(high: pd.Series, low: pd.Series, close: pd.Series, 
                        k_window: int = 14, d_window: int = 3) -> tuple:
    """Calculate Stochastic Oscillator (%K and %D)."""
    lowest_low = low.rolling(window=k_window).min()
    highest_high = high.rolling(window=k_window).max()
    k_percent = ((close - lowest_low) / (highest_high - lowest_low)) * 100
    d_percent = k_percent.rolling(window=d_window).mean()
    return k_percent, d_percent

def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Calculate Average True Range."""
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=window).mean()
    return atr

def calculate_williams_r(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Calculate Williams %R."""
    highest_high = high.rolling(window=window).max()
    lowest_low = low.rolling(window=window).min()
    williams_r = ((highest_high - close) / (highest_high - lowest_low)) * -100
    return williams_r

def get_trading_signals(df: pd.DataFrame) -> dict:
    """
    Generate trading signals based on technical indicators.
    
    Returns:
        Dictionary with trading signals and confidence levels
    """
    if df.empty:
        return {}
    
    signals = {}
    
    # RSI signals
    current_rsi = df['RSI_14'].iloc[-1]
    if current_rsi < 30:
        signals['RSI'] = {'signal': 'BUY', 'confidence': 0.8, 'value': current_rsi}
    elif current_rsi > 70:
        signals['RSI'] = {'signal': 'SELL', 'confidence': 0.8, 'value': current_rsi}
    else:
        signals['RSI'] = {'signal': 'NEUTRAL', 'confidence': 0.5, 'value': current_rsi}
    
    # MACD signals
    current_macd = df['MACD'].iloc[-1]
    current_signal = df['MACD_Signal'].iloc[-1]
    if current_macd > current_signal:
        signals['MACD'] = {'signal': 'BUY', 'confidence': 0.7, 'value': current_macd}
    else:
        signals['MACD'] = {'signal': 'SELL', 'confidence': 0.7, 'value': current_macd}
    
    # Bollinger Bands signals
    current_price = df['Close'].iloc[-1]
    bb_position = df['BB_Position'].iloc[-1]
    if bb_position < 0.2:
        signals['Bollinger_Bands'] = {'signal': 'BUY', 'confidence': 0.6, 'value': bb_position}
    elif bb_position > 0.8:
        signals['Bollinger_Bands'] = {'signal': 'SELL', 'confidence': 0.6, 'value': bb_position}
    else:
        signals['Bollinger_Bands'] = {'signal': 'NEUTRAL', 'confidence': 0.4, 'value': bb_position}
    
    # Moving Average signals
    current_price = df['Close'].iloc[-1]
    ma_20 = df['MA_20'].iloc[-1]
    ma_50 = df['MA_50'].iloc[-1]
    
    if current_price > ma_20 > ma_50:
        signals['Moving_Averages'] = {'signal': 'BUY', 'confidence': 0.6, 'value': current_price}
    elif current_price < ma_20 < ma_50:
        signals['Moving_Averages'] = {'signal': 'SELL', 'confidence': 0.6, 'value': current_price}
    else:
        signals['Moving_Averages'] = {'signal': 'NEUTRAL', 'confidence': 0.4, 'value': current_price}
    
    # Overall signal aggregation
    buy_signals = sum(1 for signal in signals.values() if signal['signal'] == 'BUY')
    sell_signals = sum(1 for signal in signals.values() if signal['signal'] == 'SELL')
    
    if buy_signals > sell_signals:
        overall_signal = 'BUY'
        confidence = min(0.9, 0.5 + (buy_signals * 0.1))
    elif sell_signals > buy_signals:
        overall_signal = 'SELL'
        confidence = min(0.9, 0.5 + (sell_signals * 0.1))
    else:
        overall_signal = 'NEUTRAL'
        confidence = 0.5
    
    signals['Overall'] = {'signal': overall_signal, 'confidence': confidence, 'value': None}
    
    return signals

def scale_features(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    return df

def add_macro_indicators(df: pd.DataFrame, macro_data: Optional[Dict[str, pd.DataFrame]] = None) -> pd.DataFrame:
    """
    Add macroeconomic indicators to the dataframe.
    
    Args:
        df: DataFrame with stock data
        macro_data: Dictionary of macroeconomic DataFrames (optional)
        
    Returns:
        DataFrame with macro indicators added
    """
    if not MACRO_AVAILABLE:
        print("Warning: Macro indicators not available")
        return df
    
    if df.empty:
        return df
    
    # Copy dataframe to avoid modifying original
    df = df.copy()
    
    # Initialize macro indicators
    macro = MacroIndicators()
    
    # If macro_data is not provided, try to load from saved files
    if macro_data is None:
        try:
            macro_data = macro.load_macro_data('data')
        except:
            print("No saved macro data found. Creating sample macro data...")
            macro_data = create_sample_macro_data(df.index)
    
    if macro_data:
        # Calculate macro features
        macro_features = macro.calculate_macro_features(macro_data)
        regime_features = macro.get_market_regime_features(macro_data)
        rotation_signals = macro.get_sector_rotation_signals(macro_data)
        
        # Merge macro features with stock data
        if not macro_features.empty:
            df = df.merge(macro_features, on='Date', how='left')
        
        if not regime_features.empty:
            df = df.merge(regime_features, on='Date', how='left')
        
        if not rotation_signals.empty:
            df = df.merge(rotation_signals, on='Date', how='left')
        
        # Create macro-enhanced features
        df = create_macro_enhanced_features(df)
    
    return df

def create_sample_macro_data(dates: pd.DatetimeIndex) -> Dict[str, pd.DataFrame]:
    """
    Create sample macroeconomic data for testing.
    
    Args:
        dates: DatetimeIndex for the stock data
        
    Returns:
        Dictionary of sample macro DataFrames
    """
    sample_data = {}
    
    # Interest rates (trending up)
    sample_data['interest_rate'] = pd.DataFrame({
        'Date': dates,
        'FEDFUNDS': np.random.normal(3.0, 1.0, len(dates)) + np.linspace(0, 5, len(dates))
    })
    
    # Inflation (cyclical)
    sample_data['inflation'] = pd.DataFrame({
        'Date': dates,
        'CPIAUCSL': np.random.normal(2.5, 0.5, len(dates)) + np.sin(np.linspace(0, 4*np.pi, len(dates)))
    })
    
    # GDP (trending up)
    sample_data['gdp'] = pd.DataFrame({
        'Date': dates,
        'GDP': np.random.normal(100, 5, len(dates)) + np.linspace(0, 10, len(dates))
    })
    
    # Unemployment (cyclical)
    sample_data['unemployment'] = pd.DataFrame({
        'Date': dates,
        'UNRATE': np.random.normal(4.0, 0.5, len(dates)) + np.sin(np.linspace(0, 2*np.pi, len(dates)))
    })
    
    # VIX (volatile)
    sample_data['vix'] = pd.DataFrame({
        'Date': dates,
        'VIXCLS': np.random.normal(20, 5, len(dates)) + np.abs(np.sin(np.linspace(0, 6*np.pi, len(dates)))) * 10
    })
    
    return sample_data

def create_macro_enhanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create enhanced features that combine technical and macro indicators.
    
    Args:
        df: DataFrame with both technical and macro indicators
        
    Returns:
        DataFrame with enhanced features
    """
    # Interest rate adjusted returns
    if 'interest_rate_change' in df.columns:
        df['rate_adjusted_returns'] = df['Close'].pct_change() - df['interest_rate_change']
    
    # Inflation adjusted volatility
    if 'inflation_CPIAUCSL' in df.columns:
        df['inflation_adjusted_volatility'] = df['Volatility'] * (1 + df['inflation_CPIAUCSL'] / 100)
    
    # Economic stress indicator
    stress_indicators = []
    if 'high_rate_environment' in df.columns:
        stress_indicators.append(df['high_rate_environment'])
    if 'high_inflation' in df.columns:
        stress_indicators.append(df['high_inflation'])
    if 'high_volatility' in df.columns:
        stress_indicators.append(df['high_volatility'])
    
    if stress_indicators:
        df['economic_stress'] = pd.concat(stress_indicators, axis=1).mean(axis=1)
    
    # Macro-adjusted RSI
    if 'RSI_14' in df.columns and 'inflation_CPIAUCSL' in df.columns:
        df['macro_adjusted_rsi'] = df['RSI_14'] * (1 + df['inflation_CPIAUCSL'] / 100)
    
    # Interest rate impact on momentum
    if 'Momentum' in df.columns and 'interest_rate_FEDFUNDS' in df.columns:
        df['rate_adjusted_momentum'] = df['Momentum'] / (1 + df['interest_rate_FEDFUNDS'] / 100)
    
    # Economic cycle adjusted volume
    if 'Volume_Ratio' in df.columns and 'economic_expansion' in df.columns:
        df['cycle_adjusted_volume'] = df['Volume_Ratio'] * (1 + df['economic_expansion'] * 0.2)
    
    return df

def get_comprehensive_features(df: pd.DataFrame, include_macro: bool = True) -> pd.DataFrame:
    """
    Get comprehensive features including both technical and macro indicators.
    
    Args:
        df: DataFrame with OHLCV data
        include_macro: Whether to include macroeconomic indicators
        
    Returns:
        DataFrame with comprehensive features
    """
    # Add technical indicators
    df = add_technical_indicators(df)
    
    # Add macro indicators if requested and available
    if include_macro and MACRO_AVAILABLE:
        df = add_macro_indicators(df)
    
    return df

def get_macro_trading_signals(df: pd.DataFrame) -> dict:
    """
    Generate trading signals based on macroeconomic conditions.
    
    Args:
        df: DataFrame with macro indicators
        
    Returns:
        Dictionary with macro-based trading signals
    """
    if df.empty or not MACRO_AVAILABLE:
        return {}
    
    signals = {}
    
    # Interest rate environment signals
    if 'high_rate_environment' in df.columns:
        current_rate_env = df['high_rate_environment'].iloc[-1]
        if current_rate_env == 1:
            signals['Interest_Rate'] = {
                'signal': 'DEFENSIVE', 
                'confidence': 0.7, 
                'reason': 'High interest rate environment - favor defensive stocks'
            }
        else:
            signals['Interest_Rate'] = {
                'signal': 'GROWTH', 
                'confidence': 0.6, 
                'reason': 'Low interest rate environment - favor growth stocks'
            }
    
    # Inflation signals
    if 'high_inflation' in df.columns:
        current_inflation = df['high_inflation'].iloc[-1]
        if current_inflation == 1:
            signals['Inflation'] = {
                'signal': 'INFLATION_HEDGE', 
                'confidence': 0.8, 
                'reason': 'High inflation - consider inflation-hedging assets'
            }
        else:
            signals['Inflation'] = {
                'signal': 'NORMAL', 
                'confidence': 0.5, 
                'reason': 'Normal inflation environment'
            }
    
    # Economic cycle signals
    if 'economic_expansion' in df.columns and 'economic_contraction' in df.columns:
        expansion = df['economic_expansion'].iloc[-1]
        contraction = df['economic_contraction'].iloc[-1]
        
        if expansion == 1:
            signals['Economic_Cycle'] = {
                'signal': 'CYCLICAL', 
                'confidence': 0.7, 
                'reason': 'Economic expansion - favor cyclical stocks'
            }
        elif contraction == 1:
            signals['Economic_Cycle'] = {
                'signal': 'DEFENSIVE', 
                'confidence': 0.8, 
                'reason': 'Economic contraction - favor defensive stocks'
            }
        else:
            signals['Economic_Cycle'] = {
                'signal': 'NEUTRAL', 
                'confidence': 0.5, 
                'reason': 'Mixed economic signals'
            }
    
    # Sector rotation signals
    if 'value_growth_signal' in df.columns:
        value_signal = df['value_growth_signal'].iloc[-1]
        if value_signal == 1:
            signals['Sector_Rotation'] = {
                'signal': 'VALUE', 
                'confidence': 0.6, 
                'reason': 'Favor value stocks over growth stocks'
            }
    
    if 'growth_tech_signal' in df.columns:
        tech_signal = df['growth_tech_signal'].iloc[-1]
        if tech_signal == 1:
            signals['Sector_Rotation'] = {
                'signal': 'GROWTH_TECH', 
                'confidence': 0.6, 
                'reason': 'Favor growth and technology stocks'
            }
    
    return signals
