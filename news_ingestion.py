"""
Responsible for gathering news-related inputs:
- Queries NewsAPI or scrapes RSS feeds for relevant headlines and articles
- Normalizes publication timestamps and sources
- Performs sentiment analysis on news content
- Outputs structured lists of news items with sentiment scores
"""
import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime, timedelta
import os
from typing import List, Dict, Optional
import json

# Sentiment analysis using TextBlob (no API key required)
try:
    from textblob import TextBlob
    SENTIMENT_AVAILABLE = True
except ImportError:
    SENTIMENT_AVAILABLE = False
    print("TextBlob not installed. Install with: pip install textblob")

class NewsAnalyzer:
    def __init__(self, newsapi_key: Optional[str] = None):
        self.newsapi_key = newsapi_key or os.getenv('NEWSAPI_KEY')
        self.sentiment_available = SENTIMENT_AVAILABLE
        
    def fetch_newsapi_headlines(self, query: str, page_size: int = 100, days_back: int = 7) -> List[Dict]:
        """Fetch news from NewsAPI with sentiment analysis"""
        if not self.newsapi_key:
            print("Warning: NEWSAPI_KEY not found. Using sample data.")
            return self._get_sample_news_data()
            
        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            url = ('https://newsapi.org/v2/everything?' 
                   f'q={query}&pageSize={page_size}&sortBy=publishedAt&'
                   f'from={start_date.strftime("%Y-%m-%d")}&'
                   f'to={end_date.strftime("%Y-%m-%d")}&'
                   f'apiKey={self.newsapi_key}')
            
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            
            articles = data.get('articles', [])
            return self._add_sentiment_to_articles(articles)
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching news from NewsAPI: {e}")
            return self._get_sample_news_data()
        except Exception as e:
            print(f"Unexpected error: {e}")
            return self._get_sample_news_data()
    
    def scrape_rss_feed(self, rss_url: str) -> List[Dict]:
        """Scrape RSS feed with sentiment analysis"""
        try:
            resp = requests.get(rss_url, timeout=10)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.content, 'xml')
            
            items = []
            for item in soup.find_all('item'):
                article = {
                    'title': item.title.text if item.title else '',
                    'link': item.link.text if item.link else '',
                    'pubDate': item.pubDate.text if item.pubDate else '',
                    'description': item.description.text if item.description else '',
                    'source': {'name': 'RSS Feed'}
                }
                items.append(article)
            
            return self._add_sentiment_to_articles(items)
            
        except Exception as e:
            print(f"Error scraping RSS feed: {e}")
            return []
    
    def _add_sentiment_to_articles(self, articles: List[Dict]) -> List[Dict]:
        """Add sentiment scores to articles"""
        if not self.sentiment_available:
            for article in articles:
                article['sentiment'] = 0.0
                article['sentiment_label'] = 'neutral'
            return articles
            
        for article in articles:
            # Combine title and description for sentiment analysis
            text = f"{article.get('title', '')} {article.get('description', '')}"
            sentiment_score = self._analyze_sentiment(text)
            
            article['sentiment'] = sentiment_score
            article['sentiment_label'] = self._get_sentiment_label(sentiment_score)
            
        return articles
    
    def _analyze_sentiment(self, text: str) -> float:
        """Analyze sentiment of text using TextBlob"""
        try:
            blob = TextBlob(text)
            return blob.sentiment.polarity
        except:
            return 0.0
    
    def _get_sentiment_label(self, score: float) -> str:
        """Convert sentiment score to label"""
        if score > 0.1:
            return 'positive'
        elif score < -0.1:
            return 'negative'
        else:
            return 'neutral'
    
    def _get_sample_news_data(self) -> List[Dict]:
        """Return sample news data for testing"""
        sample_articles = [
            {
                'title': 'Stock Market Shows Strong Recovery',
                'description': 'Major indices gain as investors show confidence in economic recovery',
                'source': {'name': 'Financial Times'},
                'publishedAt': datetime.now().isoformat(),
                'sentiment': 0.3,
                'sentiment_label': 'positive'
            },
            {
                'title': 'Tech Stocks Face Volatility',
                'description': 'Technology sector experiences mixed trading session',
                'source': {'name': 'Reuters'},
                'publishedAt': datetime.now().isoformat(),
                'sentiment': -0.1,
                'sentiment_label': 'neutral'
            },
            {
                'title': 'Federal Reserve Announces Policy Changes',
                'description': 'Central bank signals potential interest rate adjustments',
                'source': {'name': 'Bloomberg'},
                'publishedAt': datetime.now().isoformat(),
                'sentiment': 0.0,
                'sentiment_label': 'neutral'
            }
        ]
        return sample_articles
    
    def get_sentiment_summary(self, articles: List[Dict]) -> Dict:
        """Generate sentiment summary from articles"""
        if not articles:
            return {'positive': 0, 'negative': 0, 'neutral': 0, 'avg_sentiment': 0.0}
        
        sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
        total_sentiment = 0.0
        
        for article in articles:
            label = article.get('sentiment_label', 'neutral')
            sentiment_counts[label] += 1
            total_sentiment += article.get('sentiment', 0.0)
        
        return {
            **sentiment_counts,
            'avg_sentiment': total_sentiment / len(articles),
            'total_articles': len(articles)
        }

# Legacy functions for backward compatibility
def fetch_newsapi_headlines(api_key: str, query: str, page_size: int = 100):
    analyzer = NewsAnalyzer(api_key)
    return analyzer.fetch_newsapi_headlines(query, page_size)

def scrape_rss_feed(rss_url: str):
    analyzer = NewsAnalyzer()
    return analyzer.scrape_rss_feed(rss_url)
