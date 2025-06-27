import os
import pandas as pd
import sys
from datetime import datetime, timedelta

try:
    from newsapi import NewsApiClient
except ImportError:
    NewsApiClient = None

# Add the 'dags' directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config_manager import load_config, get_api_key

def _generate_sample_news_data(symbols: list, start_date: str, end_date: str) -> pd.DataFrame:
    """Generates sample news data if the NewsAPI key is not available."""
    print("⚠️ NewsAPI key not found. Generating sample news data.")
    
    all_articles = []
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    for symbol in symbols:
        # Create a few dummy articles for each symbol
        for date in date_range[::30]: # One article every 30 days
            all_articles.append({
                'symbol': symbol,
                'Date': date,
                'title': f"Positive Outlook for {symbol} as Market Rallies",
                'description': f"Analysts are bullish on {symbol} following recent performance.",
                'source': 'Sample News Inc.'
            })
    
    if not all_articles:
        return pd.DataFrame()

    return pd.DataFrame(all_articles)

def fetch_news_data(symbols: list, start_date: str, end_date: str):
    """
    Fetches news headlines from NewsAPI, with a fallback to sample data.
    """
    print(f"Fetching NewsAPI data for {len(symbols)} symbols...")
    
    try:
        api_key = get_api_key('newsapi')
        if not NewsApiClient:
            raise ImportError("newsapi-python is not installed.")
        
        newsapi = NewsApiClient(api_key=api_key)
        all_articles = []
        for symbol in symbols:
            start_fetch_date = (datetime.now() - timedelta(days=29)).strftime('%Y-%m-%d')
            articles = newsapi.get_everything(
                q=symbol, from_param=start_fetch_date, to=end_date, language='en', sort_by='relevancy', page_size=100
            )
            for article in articles['articles']:
                all_articles.append({
                    'symbol': symbol,
                    'publishedAt': article['publishedAt'],
                    'title': article['title'],
                    'description': article.get('description', ''),
                    'source': article['source']['name']
                })
        
        if not all_articles:
            print("No articles found from NewsAPI.")
            return _generate_sample_news_data(symbols, start_date, end_date)

        df = pd.DataFrame(all_articles)
        df['publishedAt'] = pd.to_datetime(df['publishedAt'])
        df.rename(columns={'publishedAt': 'Date'}, inplace=True)
        
        print(f"✅ Successfully fetched {len(df)} articles.")
        return df
        
    except (ValueError, ImportError):
        return _generate_sample_news_data(symbols, start_date, end_date)
    except Exception as e:
        print(f"An unexpected error occurred with NewsAPI: {e}")
        return _generate_sample_news_data(symbols, start_date, end_date)

def save_raw_data(df: pd.DataFrame, raw_data_path: str):
    """Saves the raw DataFrame to a Parquet file."""
    if not df.empty:
        output_path = os.path.join(raw_data_path, "news_sentiment_data.parquet")
        df.to_parquet(output_path)
        print(f"💾 Saved raw news data to {output_path}")

if __name__ == '__main__':
    config = load_config()
    
    symbols_to_test = config['stocks']
    start = config['data_fetching']['start_date']
    end = config['data_fetching']['end_date']
    raw_path = config['storage']['raw_path']
    
    os.makedirs(raw_path, exist_ok=True)
    
    df = fetch_news_data(symbols_to_test, start, end)
    
    if not df.empty:
        save_raw_data(df, raw_path)
        print("\n--- Sample Data ---")
        print(df.head())
        print("\n--- Data Info ---")
        df.info() 