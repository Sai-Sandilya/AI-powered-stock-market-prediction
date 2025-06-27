from __future__ import annotations

import pendulum

from airflow.models.dag import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.dummy import DummyOperator
from datetime import datetime
import os
import pandas as pd
import sys

# Add the 'dags' directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__)))
from config_manager import load_config
from tasks.fetch_polygon_data import fetch_polygon_data, save_raw_data as save_polygon
from tasks.fetch_fred_data import fetch_fred_data, save_raw_data as save_fred
from tasks.fetch_news_data import fetch_news_data, save_raw_data as save_news
from tasks.fetch_insider_transactions import fetch_insider_transactions, save_raw_data as save_insider

# Load configuration
config = load_config()
default_args = config['airflow']['default_args']

# --- Task Functions ---

def run_fetch_polygon(symbol, **kwargs):
    start = config['data_fetching']['start_date']
    end = config['data_fetching']['end_date']
    raw_path = config['storage']['raw_path']
    os.makedirs(raw_path, exist_ok=True)
    df = fetch_polygon_data(symbol, start, end)
    save_polygon(df, symbol, raw_path)

def run_fetch_fred(**kwargs):
    series = config['data_fetching']['fred_series']
    start = config['data_fetching']['start_date']
    end = config['data_fetching']['end_date']
    raw_path = config['storage']['raw_path']
    os.makedirs(raw_path, exist_ok=True)
    df = fetch_fred_data(series, start, end)
    save_fred(df, raw_path)

def run_fetch_news(**kwargs):
    symbols = config['stocks']
    start = config['data_fetching']['start_date']
    end = config['data_fetching']['end_date']
    raw_path = config['storage']['raw_path']
    os.makedirs(raw_path, exist_ok=True)
    df = fetch_news_data(symbols, start, end)
    save_news(df, raw_path)

def run_fetch_insider(**kwargs):
    symbols = config['stocks']
    raw_path = config['storage']['raw_path']
    os.makedirs(raw_path, exist_ok=True)
    df = fetch_insider_transactions(symbols)
    save_insider(df, raw_path)

def run_process_data(**kwargs):
    """
    Merges all raw data into a single, processed file per stock.
    """
    print("🚀 Starting data processing and merging...")
    raw_path = config['storage']['raw_path']
    processed_path = config['storage']['processed_path']
    os.makedirs(processed_path, exist_ok=True)
    
    def standardize_df(df: pd.DataFrame, name: str) -> pd.DataFrame:
        """Helper to ensure a DataFrame has a UTC DatetimeIndex."""
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], utc=True)
            df.set_index('Date', inplace=True)
        elif isinstance(df.index, pd.DatetimeIndex):
             if df.index.tz is None:
                df.index = df.index.tz_localize('UTC')
             else:
                df.index = df.index.tz_convert('UTC')
        else:
            print(f"Warning: Cannot find a date for {name}. It may not be merged correctly.")
        return df

    # Load and standardize all common datasets
    try:
        macro_df = standardize_df(pd.read_parquet(os.path.join(raw_path, 'macroeconomic_data.parquet')), 'Macro')
        news_df = standardize_df(pd.read_parquet(os.path.join(raw_path, 'news_sentiment_data.parquet')), 'News')
        insider_df = standardize_df(pd.read_parquet(os.path.join(raw_path, 'insider_transactions_data.parquet')), 'Insider')
    except FileNotFoundError as e:
        print(f"Error: A raw data file is missing. Please run the full pipeline. Details: {e}")
        return

    for symbol in config['stocks']:
        print(f"Processing data for {symbol}...")
        price_df_path = os.path.join(raw_path, f"{symbol}_price_data.parquet")
        if not os.path.exists(price_df_path):
            print(f"Warning: Price data for {symbol} not found. Skipping.")
            continue
            
        price_df = pd.read_parquet(price_df_path).reset_index()
        merged_df = standardize_df(price_df, f"Price-{symbol}")

        # Merge price data with macroeconomic data
        merged_df = pd.merge(merged_df, macro_df, left_index=True, right_index=True, how='left')
        merged_df.ffill(inplace=True)

        # Aggregate and merge news data
        if 'symbol' in news_df.columns:
            symbol_news = news_df[news_df['symbol'] == symbol].resample('D').size().rename('news_volume')
            merged_df = pd.merge(merged_df, symbol_news, left_index=True, right_index=True, how='left')

        # Aggregate and merge insider data
        if 'symbol' in insider_df.columns:
            symbol_insider = insider_df[insider_df['symbol'] == symbol].resample('D').size().rename('insider_trades')
            merged_df = pd.merge(merged_df, symbol_insider, left_index=True, right_index=True, how='left')

        merged_df.fillna(0, inplace=True)

        # Save processed file
        output_path = os.path.join(processed_path, f"{symbol}_processed_data.parquet")
        merged_df.to_parquet(output_path)
        print(f"✅ Saved processed data for {symbol} to {output_path}")

# --- DAG Definition ---

with DAG(
    dag_id=config['airflow']['dag_id'],
    default_args=default_args,
    description='ETL pipeline for fetching and processing stock market data',
    schedule=config['airflow']['schedule_interval'],
    start_date=pendulum.datetime(2023, 1, 1, tz="UTC"),
    catchup=config['airflow']['catchup']
) as dag:

    start_pipeline = DummyOperator(task_id='start_pipeline')

    # Parallel tasks for fetching data from different sources
    fetch_fred_task = PythonOperator(task_id='fetch_fred_data', python_callable=run_fetch_fred)
    fetch_news_task = PythonOperator(task_id='fetch_news_data', python_callable=run_fetch_news)
    fetch_insider_task = PythonOperator(task_id='fetch_insider_data', python_callable=run_fetch_insider)

    # Parallel tasks for fetching each stock's price data
    fetch_price_tasks = []
    for stock_symbol in config['stocks']:
        task = PythonOperator(
            task_id=f'fetch_polygon_{stock_symbol}',
            python_callable=run_fetch_polygon,
            op_kwargs={'symbol': stock_symbol}
        )
        fetch_price_tasks.append(task)
    
    # Synchronization point after all data is fetched
    data_fetched_sync = DummyOperator(task_id='data_fetched_synchronization')

    # Final task to process and merge all the data
    process_data_task = PythonOperator(task_id='process_and_merge_data', python_callable=run_process_data)
    
    end_pipeline = DummyOperator(task_id='end_pipeline')

    # --- Define Task Dependencies ---
    
    start_pipeline >> [fetch_fred_task, fetch_news_task, fetch_insider_task]
    start_pipeline >> fetch_price_tasks

    (fetch_price_tasks + [fetch_fred_task, fetch_news_task, fetch_insider_task]) >> data_fetched_sync
    
    data_fetched_sync >> process_data_task >> end_pipeline 