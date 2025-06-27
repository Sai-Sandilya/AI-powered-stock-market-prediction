#!/usr/bin/env python3
"""
Test script for news sentiment analysis functionality
"""
from news_ingestion import NewsAnalyzer
from config import Config
import json

def test_news_sentiment():
    """Test the news sentiment analysis functionality"""
    print("🧪 Testing News Sentiment Analysis")
    print("=" * 50)
    
    # Initialize news analyzer
    analyzer = NewsAnalyzer()
    
    # Test 1: Check API key status
    print("\n1. API Key Status:")
    api_status = Config.validate_api_keys()
    for key, status in api_status.items():
        status_icon = "✅" if status else "❌"
        print(f"   {status_icon} {key}: {'Configured' if status else 'Not configured'}")
    
    # Test 2: Fetch news with sentiment analysis
    print("\n2. Fetching News with Sentiment Analysis:")
    query = "stock market"
    articles = analyzer.fetch_newsapi_headlines(query, page_size=5)
    
    print(f"   📰 Found {len(articles)} articles for query: '{query}'")
    
    # Display articles with sentiment
    for i, article in enumerate(articles, 1):
        title = article.get('title', 'No title')[:60] + "..." if len(article.get('title', '')) > 60 else article.get('title', 'No title')
        sentiment = article.get('sentiment', 0.0)
        label = article.get('sentiment_label', 'neutral')
        
        # Sentiment emoji
        if label == 'positive':
            emoji = "😊"
        elif label == 'negative':
            emoji = "😞"
        else:
            emoji = "😐"
            
        print(f"   {i}. {emoji} {title}")
        print(f"      Sentiment: {sentiment:.3f} ({label})")
        print(f"      Source: {article.get('source', {}).get('name', 'Unknown')}")
        print()
    
    # Test 3: Generate sentiment summary
    print("3. Sentiment Summary:")
    summary = analyzer.get_sentiment_summary(articles)
    print(f"   📊 Total Articles: {summary['total_articles']}")
    print(f"   😊 Positive: {summary['positive']}")
    print(f"   😐 Neutral: {summary['neutral']}")
    print(f"   😞 Negative: {summary['negative']}")
    print(f"   📈 Average Sentiment: {summary['avg_sentiment']:.3f}")
    
    # Test 4: Test RSS feed scraping (if available)
    print("\n4. RSS Feed Test:")
    rss_url = "https://feeds.finance.yahoo.com/rss/2.0/headline"
    try:
        rss_articles = analyzer.scrape_rss_feed(rss_url)
        print(f"   📡 RSS Feed: {len(rss_articles)} articles scraped")
        if rss_articles:
            rss_summary = analyzer.get_sentiment_summary(rss_articles)
            print(f"   📊 RSS Sentiment: {rss_summary['avg_sentiment']:.3f}")
    except Exception as e:
        print(f"   ❌ RSS Feed Error: {e}")
    
    # Test 5: Configuration instructions
    print("\n5. Configuration Instructions:")
    missing_keys = Config.get_missing_keys()
    if missing_keys:
        print("   ⚠️  Missing API keys:")
        for key in missing_keys:
            print(f"      - {key}")
        print("\n   📝 To get API keys:")
        print("      - NewsAPI: https://newsapi.org/ (free tier available)")
        print("      - Alpha Vantage: https://www.alphavantage.co/ (free tier available)")
        print("\n   🔧 Set environment variables:")
        print("      export NEWSAPI_KEY=your_key_here")
        print("      export ALPHA_VANTAGE_KEY=your_key_here")
    else:
        print("   ✅ All API keys are configured!")
    
    print("\n" + "=" * 50)
    print("✅ News sentiment analysis test completed!")

if __name__ == "__main__":
    test_news_sentiment() 