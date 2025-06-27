# 📰 News Sentiment Analysis Setup Guide

## 🎯 Overview

The enhanced news sentiment analysis feature provides:
- **Real-time news fetching** from NewsAPI
- **RSS feed scraping** for financial news
- **Automatic sentiment analysis** using TextBlob
- **Sentiment scoring** and classification
- **Fallback sample data** when API keys are not available

## 🚀 Quick Start

### 1. Install Dependencies
```bash
source airflow_env/bin/activate
pip install -r requirements.txt
```

### 2. Test Current Setup
```bash
python test_news_sentiment.py
```

## 🔑 API Key Setup

### NewsAPI (Recommended)
1. Visit: https://newsapi.org/
2. Sign up for a free account
3. Get your API key
4. Set environment variable:
```bash
export NEWSAPI_KEY=your_api_key_here
```

### Alpha Vantage (Optional)
1. Visit: https://www.alphavantage.co/
2. Sign up for a free account
3. Get your API key
4. Set environment variable:
```bash
export ALPHA_VANTAGE_KEY=your_api_key_here
```

## 📊 Features

### ✅ What's Working Now
- **Sentiment Analysis**: TextBlob-based sentiment scoring
- **Sample Data**: Fallback data when APIs are unavailable
- **RSS Scraping**: Financial news from RSS feeds
- **Sentiment Classification**: Positive/Negative/Neutral labels
- **Summary Statistics**: Sentiment distribution analysis

### 🔄 What You Get with API Keys
- **Real-time News**: Live financial news from NewsAPI
- **Historical Data**: News from the past 7 days
- **Multiple Sources**: Various financial news outlets
- **Enhanced Accuracy**: Real market sentiment data

## 🧪 Testing

### Run the Test Script
```bash
python test_news_sentiment.py
```

### Expected Output (without API keys)
```
🧪 Testing News Sentiment Analysis
==================================================

1. API Key Status:
   ❌ newsapi: Not configured
   ❌ alpha_vantage: Not configured

2. Fetching News with Sentiment Analysis:
   📰 Found 3 articles for query: 'stock market'
   1. 😊 Stock Market Shows Strong Recovery
      Sentiment: 0.300 (positive)
      Source: Financial Times

3. Sentiment Summary:
   📊 Total Articles: 3
   😊 Positive: 1
   😐 Neutral: 2
   📈 Average Sentiment: 0.067
```

## 🔧 Integration with Main Application

### In Your Code
```python
from news_ingestion import NewsAnalyzer
from config import Config

# Initialize analyzer
analyzer = NewsAnalyzer()

# Fetch news with sentiment
articles = analyzer.fetch_newsapi_headlines("AAPL", page_size=10)

# Get sentiment summary
summary = analyzer.get_sentiment_summary(articles)
print(f"Average sentiment: {summary['avg_sentiment']}")
```

### API Endpoint Enhancement
You can enhance the inference API to include news sentiment:

```python
@app.get('/predict_with_sentiment')
def predict_with_sentiment(symbol: str, days: int = 180):
    # Get price prediction
    price_forecast = predict(symbol, days)
    
    # Get news sentiment
    analyzer = NewsAnalyzer()
    news = analyzer.fetch_newsapi_headlines(symbol, page_size=5)
    sentiment_summary = analyzer.get_sentiment_summary(news)
    
    return {
        'symbol': symbol,
        'price_forecast': price_forecast['forecast'],
        'news_sentiment': sentiment_summary,
        'recent_news': news[:3]  # Top 3 articles
    }
```

## 📈 Sentiment Analysis Details

### Sentiment Scoring
- **Range**: -1.0 to +1.0
- **Positive**: > 0.1 (😊)
- **Neutral**: -0.1 to 0.1 (😐)
- **Negative**: < -0.1 (😞)

### TextBlob Features
- **Language Detection**: Automatic language identification
- **Part-of-Speech Tagging**: Advanced text analysis
- **Subjectivity Analysis**: Objective vs subjective content
- **No API Key Required**: Works offline

## 🛠️ Troubleshooting

### Common Issues

1. **"TextBlob not installed"**
   ```bash
   pip install textblob
   ```

2. **"NEWSAPI_KEY not found"**
   - Set environment variable: `export NEWSAPI_KEY=your_key`
   - Or create `.env` file in project root

3. **"429 Too Many Requests"**
   - NewsAPI free tier: 1,000 requests/day
   - Alpha Vantage free tier: 5 requests/minute
   - Consider upgrading to paid plans

4. **RSS Feed Errors**
   - Some RSS feeds may be rate-limited
   - Try different financial news RSS feeds
   - Check feed URL validity

### Debug Mode
```python
import logging
logging.basicConfig(level=logging.DEBUG)

analyzer = NewsAnalyzer()
articles = analyzer.fetch_newsapi_headlines("AAPL")
```

## 🎯 Next Steps

1. **Get API Keys**: Sign up for free NewsAPI account
2. **Test Integration**: Run the test script
3. **Enhance UI**: Add sentiment to Streamlit interface
4. **Model Integration**: Use sentiment in price predictions
5. **Alerts**: Set up sentiment-based trading alerts

## 📚 Resources

- [NewsAPI Documentation](https://newsapi.org/docs)
- [TextBlob Documentation](https://textblob.readthedocs.io/)
- [Alpha Vantage Documentation](https://www.alphavantage.co/documentation/)

---

**🎉 Your news sentiment analysis is now ready to use!** 