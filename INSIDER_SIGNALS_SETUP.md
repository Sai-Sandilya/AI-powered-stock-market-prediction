# 👥 Insider Trading Signals Setup Guide

## 🎯 Overview

The enhanced insider trading signals feature provides:
- **Real-time insider transaction data** from Alpha Vantage
- **Comprehensive pattern analysis** of insider behavior
- **Trading signal generation** based on insider activity
- **Volume and timing analysis** for strategic insights
- **Top insider tracking** by transaction volume
- **Fallback sample data** when API keys are not available

## 🚀 Quick Start

### 1. Install Dependencies
```bash
source airflow_env/bin/activate
pip install -r requirements.txt
```

### 2. Test Current Setup
```bash
python test_insider_signals.py
```

## 🔑 API Key Setup

### Alpha Vantage (Required for Real Data)
1. Visit: https://www.alphavantage.co/
2. Sign up for a free account
3. Get your API key
4. Set environment variable:
```bash
export ALPHA_VANTAGE_KEY=your_api_key_here
```

## 📊 Features

### ✅ What's Working Now
- **Pattern Analysis**: Buy/sell ratio analysis
- **Volume Analysis**: Transaction size and frequency
- **Signal Generation**: Buy/sell signals with confidence levels
- **Sample Data**: Realistic insider transaction examples
- **Top Insiders**: Ranking by transaction volume
- **Recent Activity**: Last 7 days analysis

### 🔄 What You Get with API Keys
- **Real Insider Data**: Actual SEC filings and transactions
- **Historical Patterns**: Long-term insider behavior analysis
- **Multiple Companies**: Track insiders across different stocks
- **Enhanced Accuracy**: Real market insider activity

## 🧪 Testing

### Run the Test Script
```bash
python test_insider_signals.py
```

### Expected Output (without API keys)
```
👥 Testing Insider Trading Signals
==================================================

2. Fetching Insider Transactions:
   📊 Found 3 insider transactions for AAPL
   1. 🟢 John Smith (CEO)
      Type: BUY
      Shares: 10,000
      Price: $150.00
      Total Value: $1,500,000.00

3. Insider Pattern Analysis:
   📈 Summary:
      Total Transactions: 3
      Buys: 2 ($2,716,000.00)
      Sells: 1 ($740,000.00)
      Net Activity: $1,976,000.00
   
   🎯 Trading Signals:
      Buy Signal: 🟢 BUY
      Sell Signal: ⚪ NEUTRAL
      Confidence: 65.0%
```

## 🔧 Integration with Main Application

### In Your Code
```python
from insider_signals import InsiderTradingAnalyzer
from config import Config

# Initialize analyzer
analyzer = InsiderTradingAnalyzer()

# Fetch insider transactions
transactions = analyzer.fetch_insider_transactions("AAPL")

# Analyze patterns
analysis = analyzer.analyze_insider_patterns(transactions)

# Get trading signals
buy_signal = analysis['signals']['buy_signal']
confidence = analysis['signals']['confidence']
print(f"Insider Signal: {buy_signal} (Confidence: {confidence:.1%})")
```

### API Endpoint Enhancement
You can enhance the inference API to include insider signals:

```python
@app.get('/predict_with_insider_signals')
def predict_with_insider_signals(symbol: str, days: int = 180):
    # Get price prediction
    price_forecast = predict(symbol, days)
    
    # Get insider signals
    analyzer = InsiderTradingAnalyzer()
    transactions = analyzer.fetch_insider_transactions(symbol)
    insider_analysis = analyzer.analyze_insider_patterns(transactions)
    
    return {
        'symbol': symbol,
        'price_forecast': price_forecast['forecast'],
        'insider_signals': insider_analysis['signals'],
        'insider_summary': insider_analysis['summary'],
        'top_insiders': insider_analysis['top_insiders']
    }
```

## 📈 Signal Analysis Details

### Signal Types
- **Strong Buy**: 🟢🟢 Multiple recent buys, no sells
- **Buy**: 🟢 More buys than sells, higher buy value
- **Weak Buy**: 🟡 Slightly more buys than sells
- **Neutral**: ⚪ Balanced activity or no data
- **Weak Sell**: 🟡 Slightly more sells than buys
- **Sell**: 🔴 More sells than buys, higher sell value
- **Strong Sell**: 🔴🔴 Multiple recent sells, no buys

### Confidence Calculation
- **Volume Factor**: Based on total transaction value
- **Frequency Factor**: Based on number of transactions
- **Range**: 0% to 100%
- **Higher Confidence**: More transactions and larger values

### Key Metrics
- **Net Activity**: Total buy value - Total sell value
- **Buy/Sell Ratio**: Number of buys vs sells
- **Average Transaction Size**: Mean shares per transaction
- **Recent Activity**: Last 7 days analysis

## 🛠️ Troubleshooting

### Common Issues

1. **"ALPHA_VANTAGE_KEY not found"**
   - Set environment variable: `export ALPHA_VANTAGE_KEY=your_key`
   - Or create `.env` file in project root

2. **"429 Too Many Requests"**
   - Alpha Vantage free tier: 5 requests/minute
   - Consider upgrading to paid plans
   - Implement request caching

3. **"No insider data found"**
   - Some stocks may have limited insider activity
   - Try larger, more actively traded stocks
   - Check if company has recent insider filings

4. **"API Error"**
   - Verify API key is correct
   - Check Alpha Vantage service status
   - Ensure stock symbol is valid

### Debug Mode
```python
import logging
logging.basicConfig(level=logging.DEBUG)

analyzer = InsiderTradingAnalyzer()
transactions = analyzer.fetch_insider_transactions("AAPL")
```

## 🎯 Trading Strategy Integration

### Combining Signals
```python
def get_comprehensive_signals(symbol: str):
    # Technical analysis signals
    technical_signal = get_technical_signals(symbol)
    
    # Insider trading signals
    analyzer = InsiderTradingAnalyzer()
    transactions = analyzer.fetch_insider_transactions(symbol)
    insider_analysis = analyzer.analyze_insider_patterns(transactions)
    insider_signal = insider_analysis['signals']['buy_signal']
    
    # News sentiment signals
    news_analyzer = NewsAnalyzer()
    news = news_analyzer.fetch_newsapi_headlines(symbol)
    sentiment_summary = news_analyzer.get_sentiment_summary(news)
    
    return {
        'technical': technical_signal,
        'insider': insider_signal,
        'sentiment': sentiment_summary['avg_sentiment'],
        'confidence': insider_analysis['signals']['confidence']
    }
```

### Alert System
```python
def check_insider_alerts(symbol: str):
    analyzer = InsiderTradingAnalyzer()
    transactions = analyzer.fetch_insider_transactions(symbol)
    analysis = analyzer.analyze_insider_patterns(transactions)
    
    signals = analysis['signals']
    
    if signals['buy_signal'] == 'strong_buy' and signals['confidence'] > 0.7:
        send_alert(f"🚨 STRONG INSIDER BUY SIGNAL for {symbol}")
    
    if signals['sell_signal'] == 'strong_sell' and signals['confidence'] > 0.7:
        send_alert(f"🚨 STRONG INSIDER SELL SIGNAL for {symbol}")
```

## 🎯 Next Steps

1. **Get API Key**: Sign up for free Alpha Vantage account
2. **Test Integration**: Run the test script
3. **Enhance UI**: Add insider signals to Streamlit interface
4. **Model Integration**: Use insider signals in price predictions
5. **Alerts**: Set up insider-based trading alerts
6. **Portfolio Tracking**: Monitor insider activity across watchlist

## 📚 Resources

- [Alpha Vantage Documentation](https://www.alphavantage.co/documentation/)
- [SEC Insider Trading Data](https://www.sec.gov/data/insider-trading)
- [Insider Trading Analysis Guide](https://www.investopedia.com/terms/i/insidertrading.asp)

## ⚠️ Important Notes

- **Legal Compliance**: Insider trading data is public information
- **Timing**: Insider filings have reporting delays
- **Context**: Always combine with other analysis methods
- **Limitations**: Not all insider activity is predictive

---

**🎉 Your insider trading signals are now ready to use!** 