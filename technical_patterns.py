import pandas as pd

def detect_bullish_breakout(df: pd.DataFrame) -> bool:
    return df['Close'].iloc[-1] > df['High'].rolling(20).max().iloc[-2]
