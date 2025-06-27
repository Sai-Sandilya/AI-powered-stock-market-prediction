#!/usr/bin/env python3
"""
News Sentiment Analysis Module for Stock Prediction
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import time
from textblob import TextBlob
import re

# API Keys
NEWSAPI_KEY = "3f4e25357c374ad1b38416e5a3174433"
ALPHA_VANTAGE_KEY = "ddef003d559869eb7aa1fd8dca6b085f"

class NewsSentimentAnalyzer:
    def __init__(self):
        self.newsapi_key = NEWSAPI_KEY
        self.alpha_vantage_key = ALPHA_VANTAGE_KEY
        self.base_url_news = "https://newsapi.org/v2"
        self.base_url_av = "https://www.alphavantage.co/query"
        
        self.positive_keywords = {
            'strong', 'growth', 'profit', 'gains', 'bullish', 'optimistic', 
            'upgrade', 'beat', 'outperform', 'surge', 'rally', 'boom',
            'increase', 'rise', 'earnings', 'revenue', 'breakthrough'
        }
        
        self.negative_keywords = {
            'loss', 'decline', 'bearish', 'pessimistic', 'downgrade',
            'miss', 'underperform', 'plunge', 'crash', 'recession',
            'lawsuit', 'investigation', 'warning', 'cut', 'layoffs'
        }
        
        # Company name mapping for better news search
        self.company_mapping = {
            # Indian Stocks
            "TRENT.NS": "Trent Limited Westside",
            "RELIANCE.NS": "Reliance Industries",
            "TCS.NS": "Tata Consultancy Services TCS",
            "HDFCBANK.NS": "HDFC Bank",
            "INFY.NS": "Infosys",
            "ICICIBANK.NS": "ICICI Bank",
            "HINDUNILVR.NS": "Hindustan Unilever HUL",
            "ITC.NS": "ITC Limited",
            "SBIN.NS": "State Bank India SBI",
            "BHARTIARTL.NS": "Bharti Airtel",
            "AXISBANK.NS": "Axis Bank",
            "KOTAKBANK.NS": "Kotak Mahindra Bank",
            "ASIANPAINT.NS": "Asian Paints",
            "MARUTI.NS": "Maruti Suzuki",
            "SUNPHARMA.NS": "Sun Pharmaceutical",
            "TATAMOTORS.NS": "Tata Motors",
            "WIPRO.NS": "Wipro",
            "ULTRACEMCO.NS": "UltraTech Cement",
            "TITAN.NS": "Titan Company",
            "BAJFINANCE.NS": "Bajaj Finance",
            "NESTLEIND.NS": "Nestle India",
            "POWERGRID.NS": "Power Grid Corporation",
            "TECHM.NS": "Tech Mahindra",
            "BAJAJFINSV.NS": "Bajaj Finserv",
            "NTPC.NS": "NTPC Limited",
            "HCLTECH.NS": "HCL Technologies",
            "ONGC.NS": "Oil Natural Gas Corporation ONGC",
            "JSWSTEEL.NS": "JSW Steel",
            "TATACONSUM.NS": "Tata Consumer Products",
            "ADANIENT.NS": "Adani Enterprises",
            "COALINDIA.NS": "Coal India",
            "HINDALCO.NS": "Hindalco Industries",
            "TATASTEEL.NS": "Tata Steel",
            "BRITANNIA.NS": "Britannia Industries",
            "GRASIM.NS": "Grasim Industries",
            "INDUSINDBK.NS": "IndusInd Bank",
            "M&M.NS": "Mahindra Mahindra",
            "BAJAJ-AUTO.NS": "Bajaj Auto",
            "VEDL.NS": "Vedanta Limited",
            "UPL.NS": "UPL Limited",
            "BPCL.NS": "Bharat Petroleum",
            "SBILIFE.NS": "SBI Life Insurance",
            "HDFCLIFE.NS": "HDFC Life Insurance",
            "DIVISLAB.NS": "Divi's Laboratories",
            "CIPLA.NS": "Cipla",
            "EICHERMOT.NS": "Eicher Motors",
            "HEROMOTOCO.NS": "Hero MotoCorp",
            "SHREECEM.NS": "Shree Cement",
            "ADANIPORTS.NS": "Adani Ports",
            "DRREDDY.NS": "Dr Reddy's Laboratories",
            "APOLLOHOSP.NS": "Apollo Hospitals",
            
            # US Stocks
            "AAPL": "Apple Inc",
            "MSFT": "Microsoft Corporation",
            "GOOGL": "Alphabet Google",
            "TSLA": "Tesla",
            "AMZN": "Amazon",
            "META": "Meta Facebook",
            "NVDA": "Nvidia",
            "NFLX": "Netflix",
            "ORCL": "Oracle",
            "ADBE": "Adobe",
            "CRM": "Salesforce",
            "INTC": "Intel",
            "AMD": "Advanced Micro Devices",
            "QCOM": "Qualcomm",
            "IBM": "International Business Machines",
            "PYPL": "PayPal",
            "DIS": "Walt Disney",
            "UBER": "Uber Technologies",
            "LYFT": "Lyft",
            "ZOOM": "Zoom Video Communications"
        }
    
    def get_company_info(self, symbol: str) -> tuple:
        """Get company name and search terms for better news discovery."""
        # Remove exchange suffix for mapping lookup
        base_symbol = symbol.replace('.NS', '').replace('.BO', '')
        
        # Try exact match first
        if symbol in self.company_mapping:
            company_name = self.company_mapping[symbol]
        elif base_symbol in self.company_mapping:
            company_name = self.company_mapping[base_symbol]
        else:
            # Fallback to symbol
            company_name = base_symbol
        
        # Create multiple search terms for better coverage
        search_terms = [symbol, base_symbol]
        if company_name != base_symbol:
            search_terms.extend(company_name.split())
        
        return company_name, search_terms
    
    def get_company_news(self, symbol: str, company_name: str = None, days_back: int = 7) -> List[Dict]:
        """Fetch news for a specific company with enhanced search."""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            # Get company info for better search
            if not company_name:
                company_name, search_terms = self.get_company_info(symbol)
            else:
                search_terms = [symbol, company_name]
            
            all_articles = []
            
            # Try multiple search strategies
            search_queries = [
                f'"{company_name}"',  # Exact company name
                symbol,  # Stock symbol
                symbol.replace('.NS', '').replace('.BO', ''),  # Base symbol without exchange
            ]
            
            # Add individual company name words for broader search
            if company_name and ' ' in company_name:
                company_words = company_name.split()
                if len(company_words) >= 2:
                    search_queries.append(' '.join(company_words[:2]))
            
            for query in search_queries[:3]:  # Limit to first 3 to avoid rate limits
                try:
                    url = f"{self.base_url_news}/everything"
                    params = {
                        'q': query,
                        'from': start_date.strftime('%Y-%m-%d'),
                        'to': end_date.strftime('%Y-%m-%d'),
                        'sortBy': 'relevancy',
                        'pageSize': 20,  # Reduced to avoid overwhelming results
                        'language': 'en',
                        'apiKey': self.newsapi_key
                    }
                    
                    response = requests.get(url, params=params, timeout=10)
                    
                    if response.status_code == 200:
                        data = response.json()
                        articles = data.get('articles', [])
                        for article in articles:
                            if article.get('title') and self._is_relevant_article(article, symbol, company_name):
                                processed_article = self._process_article(article, symbol)
                                if processed_article not in all_articles:  # Avoid duplicates
                                    all_articles.append(processed_article)
                    
                    # Rate limiting
                    time.sleep(0.5)
                    
                except Exception as e:
                    print(f"Error with query '{query}': {e}")
                    continue
            
            return all_articles[:30]  # Limit final results
                
        except Exception as e:
            print(f"Error fetching news: {e}")
            return []
    
    def _is_relevant_article(self, article: Dict, symbol: str, company_name: str) -> bool:
        """Check if article is relevant to the company."""
        # Handle None values safely
        title = article.get('title') or ''
        description = article.get('description') or ''
        
        # Ensure we have strings before calling .lower()
        title = str(title).lower() if title else ''
        description = str(description).lower() if description else ''
        content = f"{title} {description}"
        
        # Check for symbol or company name presence
        base_symbol = symbol.replace('.NS', '').replace('.BO', '').lower()
        company_words = company_name.lower().split() if company_name else []
        
        # Must contain at least the base symbol or main company words
        if base_symbol in content:
            return True
        
        if company_words and len(company_words) >= 2:
            # Check if at least 2 company name words are present
            word_matches = sum(1 for word in company_words if word in content and len(word) > 3)
            if word_matches >= 2:
                return True
        
        return False
    
    def get_alpha_vantage_news(self, symbol: str) -> List[Dict]:
        """Fetch news from Alpha Vantage with enhanced symbol handling."""
        try:
            # Alpha Vantage typically works better with base symbols
            base_symbol = symbol.replace('.NS', '').replace('.BO', '')
            
            url = self.base_url_av
            params = {
                'function': 'NEWS_SENTIMENT',
                'tickers': base_symbol,
                'apikey': self.alpha_vantage_key,
                'limit': 50
            }
            
            response = requests.get(url, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                if 'feed' in data:
                    return [self._process_av_article(item, symbol) for item in data['feed'][:20]]
            
            return []
            
        except Exception as e:
            print(f"Alpha Vantage error: {e}")
            return []
    
    def analyze_sentiment(self, articles: List[Dict]) -> Dict:
        """Analyze sentiment of articles."""
        if not articles:
            return self._empty_sentiment_result()
        
        sentiments = []
        for article in articles:
            sentiment = self._calculate_sentiment(article)
            article['sentiment_score'] = sentiment
            sentiments.append(sentiment)
        
        overall_sentiment = np.mean(sentiments)
        
        # Count sentiment distribution
        positive = len([s for s in sentiments if s > 0.1])
        negative = len([s for s in sentiments if s < -0.1])
        neutral = len(sentiments) - positive - negative
        
        return {
            'overall_sentiment': overall_sentiment,
            'total_articles': len(articles),
            'sentiment_distribution': {
                'positive': positive,
                'negative': negative,
                'neutral': neutral,
                'positive_pct': positive / len(articles) * 100,
                'negative_pct': negative / len(articles) * 100,
                'neutral_pct': neutral / len(articles) * 100
            },
            'articles_with_sentiment': articles
        }
    
    def get_sentiment_summary(self, symbol: str, company_name: str = None, days_back: int = 7) -> Dict:
        """Get comprehensive sentiment summary with enhanced search."""
        print(f"Fetching news sentiment for {symbol}...")
        
        # Get company information
        if not company_name:
            company_name, _ = self.get_company_info(symbol)
        
        print(f"Searching for: {company_name}")
        
        # Fetch from multiple sources
        newsapi_articles = self.get_company_news(symbol, company_name, days_back)
        av_articles = self.get_alpha_vantage_news(symbol)
        
        all_articles = newsapi_articles + av_articles
        
        # Remove duplicates based on title similarity
        unique_articles = self._remove_duplicate_articles(all_articles)
        
        print(f"Found {len(unique_articles)} unique articles")
        
        sentiment_analysis = self.analyze_sentiment(unique_articles)
        sentiment_label = self._interpret_sentiment(sentiment_analysis['overall_sentiment'])
        
        return {
            'symbol': symbol,
            'company_name': company_name,
            'sentiment_score': sentiment_analysis['overall_sentiment'],
            'sentiment_label': sentiment_label,
            'analysis': sentiment_analysis,
            'analysis_date': datetime.now().isoformat(),
            'search_info': {
                'newsapi_articles': len(newsapi_articles),
                'alpha_vantage_articles': len(av_articles),
                'total_unique_articles': len(unique_articles)
            }
        }
    
    def _remove_duplicate_articles(self, articles: List[Dict]) -> List[Dict]:
        """Remove duplicate articles based on title similarity."""
        unique_articles = []
        seen_titles = set()
        
        for article in articles:
            title = article.get('title', '').lower().strip()
            # Simple deduplication based on title
            title_key = ''.join(title.split()[:5])  # First 5 words as key
            
            if title_key not in seen_titles and len(title) > 10:
                seen_titles.add(title_key)
                unique_articles.append(article)
        
        return unique_articles
    
    def _process_article(self, article: Dict, symbol: str) -> Dict:
        """Process article data safely."""
        pub_date = article.get('publishedAt') or ''
        try:
            if pub_date:
                published_date = datetime.fromisoformat(str(pub_date).replace('Z', '+00:00')).date()
            else:
                published_date = datetime.now().date()
        except:
            published_date = datetime.now().date()
        
        # Safely handle all fields
        source_info = article.get('source') or {}
        
        return {
            'title': str(article.get('title') or ''),
            'description': str(article.get('description') or ''),
            'url': str(article.get('url') or ''),
            'source': str(source_info.get('name') or 'Unknown'),
            'published_date': published_date,
            'symbol': symbol
        }
    
    def _process_av_article(self, item: Dict, symbol: str) -> Dict:
        """Process Alpha Vantage article safely."""
        time_published = item.get('time_published') or ''
        try:
            if time_published:
                pub_datetime = datetime.strptime(str(time_published), '%Y%m%dT%H%M%S')
                published_date = pub_datetime.date()
            else:
                published_date = datetime.now().date()
        except:
            published_date = datetime.now().date()
        
        return {
            'title': str(item.get('title') or ''),
            'description': str(item.get('summary') or ''),
            'url': str(item.get('url') or ''),
            'source': 'Alpha Vantage',
            'published_date': published_date,
            'symbol': symbol
        }
    
    def _calculate_sentiment(self, article: Dict) -> float:
        """Calculate sentiment score for an article."""
        # Handle None values safely
        title = article.get('title') or ''
        description = article.get('description') or ''
        
        # Ensure we have strings before processing
        title = str(title) if title else ''
        description = str(description) if description else ''
        text = f"{title} {description}".lower()
        
        # TextBlob sentiment
        try:
            blob = TextBlob(text)
            textblob_score = blob.sentiment.polarity
        except:
            textblob_score = 0.0
        
        # Keyword-based sentiment
        positive_count = sum(1 for word in self.positive_keywords if word in text)
        negative_count = sum(1 for word in self.negative_keywords if word in text)
        
        if positive_count + negative_count == 0:
            keyword_score = 0.0
        else:
            keyword_score = (positive_count - negative_count) / (positive_count + negative_count)
        
        # Combined score
        return (textblob_score + keyword_score) / 2
    
    def _interpret_sentiment(self, sentiment: float) -> Dict:
        """Interpret sentiment score."""
        if sentiment >= 0.3:
            return {'label': 'Very Positive', 'emoji': '🚀', 'color': '#00C851'}
        elif sentiment >= 0.1:
            return {'label': 'Positive', 'emoji': '📈', 'color': '#4CAF50'}
        elif sentiment >= -0.1:
            return {'label': 'Neutral', 'emoji': '➖', 'color': '#FFC107'}
        elif sentiment >= -0.3:
            return {'label': 'Negative', 'emoji': '📉', 'color': '#FF5722'}
        else:
            return {'label': 'Very Negative', 'emoji': '🔻', 'color': '#F44336'}
    
    def _empty_sentiment_result(self) -> Dict:
        """Return empty sentiment result."""
        return {
            'overall_sentiment': 0.0,
            'total_articles': 0,
            'sentiment_distribution': {
                'positive': 0, 'negative': 0, 'neutral': 0,
                'positive_pct': 0, 'negative_pct': 0, 'neutral_pct': 0
            },
            'articles_with_sentiment': []
        } 