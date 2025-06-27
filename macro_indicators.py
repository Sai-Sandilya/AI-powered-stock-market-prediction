"""
Macroeconomic Indicators Module
Fetches and processes macroeconomic data to enhance stock prediction models.
"""

import pandas as pd
import numpy as np
import requests
import yfinance as yf
from datetime import datetime, timedelta
import time
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class MacroIndicators:
    """
    Fetches and processes macroeconomic indicators from various sources.
    """
    
    def __init__(self, fred_api_key: Optional[str] = None):
        """
        Initialize the macro indicators fetcher.
        
        Args:
            fred_api_key: API key for FRED (Federal Reserve Economic Data)
        """
        self.fred_api_key = fred_api_key
        self.base_urls = {
            'fred': 'https://api.stlouisfed.org/fred/series/observations',
            'world_bank': 'https://api.worldbank.org/v2/country',
            'yahoo_finance': 'https://query1.finance.yahoo.com/v8/finance/chart'
        }
        
        # Common macroeconomic indicators
        self.indicators = {
            'us': {
                'gdp': 'GDP',  # Gross Domestic Product
                'inflation': 'CPIAUCSL',  # Consumer Price Index
                'unemployment': 'UNRATE',  # Unemployment Rate
                'interest_rate': 'FEDFUNDS',  # Federal Funds Rate
                'money_supply': 'M2SL',  # M2 Money Supply
                'consumer_confidence': 'UMCSENT',  # Consumer Sentiment
                'manufacturing_pmi': 'NAPM',  # ISM Manufacturing PMI
                'housing_starts': 'HOUST',  # Housing Starts
                'retail_sales': 'RSAFS',  # Retail Sales
                'industrial_production': 'INDPRO',  # Industrial Production
                'vix': 'VIXCLS',  # VIX Volatility Index
                'dollar_index': 'DTWEXBGS',  # Trade Weighted US Dollar Index
                'oil_prices': 'DCOILWTICO',  # WTI Crude Oil Prices
                'gold_prices': 'GOLDPMGBD228NLBM',  # Gold Prices
                'bond_yield_10y': 'GS10',  # 10-Year Treasury Constant Maturity Rate
                'bond_yield_2y': 'GS2',  # 2-Year Treasury Constant Maturity Rate
                'bond_spread': 'T10Y2Y',  # 10-Year minus 2-Year Treasury Constant Maturity Rate
            },
            'global': {
                'usd_index': 'DTWEXBGS',
                'euro_index': 'DEXUSEU',
                'yen_index': 'DEXJPUS',
                'pound_index': 'DEXUSUK',
                'emerging_markets': 'DGS10',
            }
        }
    
    def fetch_fred_data(self, series_id: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch data from FRED API.
        
        Args:
            series_id: FRED series ID
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame with date and value columns
        """
        if not self.fred_api_key:
            print(f"Warning: No FRED API key provided. Cannot fetch {series_id}")
            return pd.DataFrame()
        
        params = {
            'series_id': series_id,
            'api_key': self.fred_api_key,
            'file_type': 'json',
            'sort_order': 'asc'
        }
        
        if start_date:
            params['observation_start'] = start_date
        if end_date:
            params['observation_end'] = end_date
        
        try:
            response = requests.get(self.base_urls['fred'], params=params)
            response.raise_for_status()
            data = response.json()
            
            if 'observations' in data:
                df = pd.DataFrame(data['observations'])
                df['date'] = pd.to_datetime(df['date'])
                df['value'] = pd.to_numeric(df['value'], errors='coerce')
                df = df[['date', 'value']].dropna()
                df.columns = ['Date', series_id]
                return df
            else:
                print(f"No observations found for {series_id}")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"Error fetching {series_id}: {e}")
            return pd.DataFrame()
    
    def fetch_yahoo_macro_data(self, symbol: str, period: str = '2y') -> pd.DataFrame:
        """
        Fetch macroeconomic data from Yahoo Finance.
        
        Args:
            symbol: Yahoo Finance symbol
            period: Time period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period)
            if df.empty:
                return pd.DataFrame()
            df.reset_index(inplace=True)
            df['Date'] = pd.to_datetime(df['Date'])
            return df
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
            return pd.DataFrame()
    
    def get_macro_indicators(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Fetch all macroeconomic indicators.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            Dictionary of DataFrames for each indicator
        """
        if not start_date:
            start_date = (datetime.now() - timedelta(days=365*2)).strftime('%Y-%m-%d')
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        macro_data = {}
        
        # Fetch FRED data
        for indicator_name, series_id in self.indicators['us'].items():
            print(f"Fetching {indicator_name} ({series_id})...")
            df = self.fetch_fred_data(series_id, start_date, end_date)
            if not df.empty:
                macro_data[indicator_name] = df
            time.sleep(0.1)  # Rate limiting
        
        # Fetch Yahoo Finance data for commodities and currencies
        yahoo_symbols = {
            'oil_futures': 'CL=F',
            'gold_futures': 'GC=F',
            'silver_futures': 'SI=F',
            'copper_futures': 'HG=F',
            'natural_gas': 'NG=F',
            'usd_eur': 'EURUSD=X',
            'usd_jpy': 'USDJPY=X',
            'usd_gbp': 'GBPUSD=X',
            'usd_cny': 'USDCNY=X',
            'bitcoin': 'BTC-USD',
            'ethereum': 'ETH-USD',
        }
        
        for indicator_name, symbol in yahoo_symbols.items():
            print(f"Fetching {indicator_name} ({symbol})...")
            df = self.fetch_yahoo_macro_data(symbol, '2y')
            if not df.empty:
                macro_data[indicator_name] = df
            time.sleep(0.1)
        
        return macro_data
    
    def calculate_macro_features(self, macro_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Calculate derived macroeconomic features.
        
        Args:
            macro_data: Dictionary of macroeconomic DataFrames
            
        Returns:
            DataFrame with calculated features
        """
        features = []
        
        for indicator_name, df in macro_data.items():
            if df.empty:
                continue
                
            # Ensure we have a 'Date' column
            if 'Date' not in df.columns:
                continue
            
            # Calculate various features based on indicator type
            if 'Close' in df.columns:  # Yahoo Finance data
                # Price-based features
                close_series = df['Close']
                df[f'{indicator_name}_returns'] = close_series.pct_change()
                df[f'{indicator_name}_volatility'] = close_series.rolling(window=20).std()
                df[f'{indicator_name}_sma_20'] = close_series.rolling(window=20).mean()
                df[f'{indicator_name}_sma_50'] = close_series.rolling(window=50).mean()
                df[f'{indicator_name}_rsi'] = self.calculate_rsi(close_series)
                
                # Select relevant columns
                feature_cols = ['Date', 'Close', f'{indicator_name}_returns', 
                              f'{indicator_name}_volatility', f'{indicator_name}_rsi']
                df_features = df[feature_cols].copy()
                
            else:  # FRED data
                # Economic indicator features
                value_col = df.columns[1]  # Second column is the value
                value_series = df[value_col]
                df[f'{indicator_name}_change'] = value_series.pct_change()
                df[f'{indicator_name}_ma_12'] = value_series.rolling(window=12).mean()
                df[f'{indicator_name}_ma_24'] = value_series.rolling(window=24).mean()
                df[f'{indicator_name}_std_12'] = value_series.rolling(window=12).std()
                
                # Select relevant columns
                feature_cols = ['Date', value_col, f'{indicator_name}_change', 
                              f'{indicator_name}_ma_12', f'{indicator_name}_std_12']
                df_features = df[feature_cols].copy()
            
            features.append(df_features)
        
        # Merge all features on date
        if features:
            merged_features = features[0]
            for df in features[1:]:
                merged_features = merged_features.merge(df, on='Date', how='outer')
            
            # Forward fill missing values
            merged_features = merged_features.sort_values('Date').fillna(method='ffill')
            
            return merged_features
        else:
            return pd.DataFrame()
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index.
        
        Args:
            prices: Price series
            period: RSI period
            
        Returns:
            RSI series
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def get_market_regime_features(self, macro_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Calculate market regime indicators.
        
        Args:
            macro_data: Dictionary of macroeconomic DataFrames
            
        Returns:
            DataFrame with market regime features
        """
        regime_features = []
        
        # Interest rate environment
        if 'interest_rate' in macro_data:
            rates = macro_data['interest_rate']
            if not rates.empty:
                value_col = rates.columns[1]
                rate_series = rates[value_col]
                rate_trend = pd.Series(np.where(rate_series.diff() > 0, 1, -1), index=rate_series.index)
                regime_features.append(pd.DataFrame({
                    'Date': rates['Date'],
                    'high_rate_environment': (rate_series > 5.0).astype(int),
                    'low_rate_environment': (rate_series < 2.0).astype(int),
                    'rate_trend': rate_trend
                }))
        
        # Inflation environment
        if 'inflation' in macro_data:
            inflation = macro_data['inflation']
            if not inflation.empty:
                value_col = inflation.columns[1]
                infl_series = inflation[value_col]
                regime_features.append(pd.DataFrame({
                    'Date': inflation['Date'],
                    'high_inflation': (infl_series > 3.0).astype(int),
                    'deflation_risk': (infl_series < 1.0).astype(int)
                }))
        
        # Volatility regime
        if 'vix' in macro_data:
            vix = macro_data['vix']
            if not vix.empty:
                value_col = vix.columns[1]
                vix_series = vix[value_col]
                regime_features.append(pd.DataFrame({
                    'Date': vix['Date'],
                    'high_volatility': (vix_series > 25).astype(int),
                    'low_volatility': (vix_series < 15).astype(int)
                }))
        
        # Economic cycle indicators
        if 'unemployment' in macro_data and 'gdp' in macro_data:
            unemployment = macro_data['unemployment']
            gdp = macro_data['gdp']
            if not unemployment.empty and not gdp.empty:
                # Simple economic cycle indicator
                unemp_col = unemployment.columns[1]
                gdp_col = gdp.columns[1]
                
                # Align dates
                merged = unemployment.merge(gdp, on='Date', how='inner')
                if not merged.empty:
                    regime_features.append(pd.DataFrame({
                        'Date': merged['Date'],
                        'economic_expansion': ((merged[gdp_col].pct_change() > 0) & 
                                             (merged[unemp_col] < 5.0)).astype(int),
                        'economic_contraction': ((merged[gdp_col].pct_change() < 0) & 
                                               (merged[unemp_col] > 6.0)).astype(int)
                    }))
        
        # Combine all regime features
        if regime_features:
            combined = regime_features[0]
            for features in regime_features[1:]:
                combined = combined.merge(features, on='Date', how='outer')
            
            combined = combined.sort_values('Date').fillna(method='ffill')
            return combined
        else:
            return pd.DataFrame()
    
    def get_sector_rotation_signals(self, macro_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Generate sector rotation signals based on macroeconomic conditions.
        
        Args:
            macro_data: Dictionary of macroeconomic DataFrames
            
        Returns:
            DataFrame with sector rotation signals
        """
        signals = []
        
        if 'interest_rate' in macro_data and 'inflation' in macro_data:
            rates = macro_data['interest_rate']
            inflation = macro_data['inflation']
            
            if not rates.empty and not inflation.empty:
                # Align dates
                merged = rates.merge(inflation, on='Date', how='inner')
                if not merged.empty:
                    rate_col = rates.columns[1]
                    infl_col = inflation.columns[1]
                    
                    signals.append(pd.DataFrame({
                        'Date': merged['Date'],
                        'value_growth_signal': ((merged[rate_col] > 3.0) & 
                                               (merged[infl_col] < 2.5)).astype(int),
                        'growth_tech_signal': ((merged[rate_col] < 2.0) & 
                                               (merged[infl_col] < 2.0)).astype(int),
                        'defensive_signal': ((merged[rate_col] > 4.0) | 
                                            (merged[infl_col] > 3.5)).astype(int),
                        'cyclical_signal': ((merged[rate_col] < 3.0) & 
                                           (merged[infl_col] > 2.0)).astype(int)
                    }))
        
        if signals:
            combined = signals[0]
            for df in signals[1:]:
                combined = combined.merge(df, on='Date', how='outer')
            combined = combined.sort_values('Date').fillna(method='ffill')
            return combined
        else:
            return pd.DataFrame()
    
    def save_macro_data(self, macro_data: Dict[str, pd.DataFrame], output_dir: str = 'data'):
        """
        Save macroeconomic data to CSV files.
        
        Args:
            macro_data: Dictionary of macroeconomic DataFrames
            output_dir: Output directory
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        for indicator_name, df in macro_data.items():
            if not df.empty:
                filename = f"{output_dir}/macro_{indicator_name}.csv"
                df.to_csv(filename, index=False)
                print(f"Saved {filename}")
    
    def load_macro_data(self, input_dir: str = 'data') -> Dict[str, pd.DataFrame]:
        """
        Load macroeconomic data from CSV files.
        
        Args:
            input_dir: Input directory
            
        Returns:
            Dictionary of macroeconomic DataFrames
        """
        import os
        import glob
        
        macro_data = {}
        pattern = f"{input_dir}/macro_*.csv"
        
        for filename in glob.glob(pattern):
            indicator_name = filename.split('macro_')[1].split('.csv')[0]
            df = pd.read_csv(filename)
            df['Date'] = pd.to_datetime(df['Date'])
            macro_data[indicator_name] = df
            print(f"Loaded {filename}")
        
        return macro_data


def test_macro_indicators():
    """
    Test the macroeconomic indicators module.
    """
    print("🧪 Testing Macroeconomic Indicators Module")
    print("=" * 50)
    
    # Initialize the module
    macro = MacroIndicators()
    
    # Test with sample data (without API key)
    print("\n1. Testing with sample data...")
    
    # Create sample macroeconomic data
    dates = pd.date_range(start='2022-01-01', end='2024-01-01', freq='D')
    sample_data = {
        'gdp': pd.DataFrame({
            'Date': dates,
            'GDP': np.random.normal(100, 5, len(dates))
        }),
        'inflation': pd.DataFrame({
            'Date': dates,
            'CPIAUCSL': np.random.normal(2.5, 0.5, len(dates))
        }),
        'interest_rate': pd.DataFrame({
            'Date': dates,
            'FEDFUNDS': np.random.normal(3.0, 1.0, len(dates))
        }),
        'oil_futures': pd.DataFrame({
            'Date': dates,
            'Close': np.random.normal(80, 10, len(dates)),
            'High': np.random.normal(82, 10, len(dates)),
            'Low': np.random.normal(78, 10, len(dates)),
            'Volume': np.random.randint(1000, 10000, len(dates))
        })
    }
    
    print("✅ Sample data created")
    
    # Test feature calculation
    print("\n2. Testing feature calculation...")
    features = macro.calculate_macro_features(sample_data)
    print(f"✅ Calculated {len(features.columns)} features")
    print(f"Features: {list(features.columns)}")
    
    # Test market regime features
    print("\n3. Testing market regime features...")
    regime_features = macro.get_market_regime_features(sample_data)
    print(f"✅ Calculated {len(regime_features.columns)} regime features")
    if not regime_features.empty:
        print(f"Regime features: {list(regime_features.columns)}")
    
    # Test sector rotation signals
    print("\n4. Testing sector rotation signals...")
    rotation_signals = macro.get_sector_rotation_signals(sample_data)
    print(f"✅ Calculated {len(rotation_signals.columns)} rotation signals")
    if not rotation_signals.empty:
        print(f"Rotation signals: {list(rotation_signals.columns)}")
    
    # Test data saving and loading
    print("\n5. Testing data persistence...")
    macro.save_macro_data(sample_data, 'test_data')
    loaded_data = macro.load_macro_data('test_data')
    print(f"✅ Saved and loaded {len(loaded_data)} datasets")
    
    # Clean up test data
    import shutil
    shutil.rmtree('test_data', ignore_errors=True)
    
    print("\n🎉 All tests completed successfully!")
    print("\n📊 Sample feature statistics:")
    if not features.empty:
        print(features.describe())
    
    return features, regime_features, rotation_signals


if __name__ == "__main__":
    test_macro_indicators() 