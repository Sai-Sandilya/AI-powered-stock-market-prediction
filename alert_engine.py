from technical_patterns import detect_bullish_breakout
from insider_signals import fetch_insider_transactions
from datetime import datetime

def evaluate_alerts(symbol: str, market_df, news_articles):
    alerts = []
    if detect_bullish_breakout(market_df):
        alerts.append((symbol, 'Bullish Breakout Detected', datetime.now()))
    insiders = fetch_insider_transactions('API_KEY', symbol)
    for tx in insiders:
        alerts.append((symbol, f"Insider {tx['type']} of {tx['shares']} shares", tx['date']))
    return alerts
