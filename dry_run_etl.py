import os
import shutil
from dags.config_manager import load_config
from dags.tasks.fetch_polygon_data import fetch_polygon_data, save_raw_data as save_polygon
from dags.tasks.fetch_fred_data import fetch_fred_data, save_raw_data as save_fred
from dags.tasks.fetch_news_data import fetch_news_data, save_raw_data as save_news
from dags.tasks.fetch_insider_transactions import fetch_insider_transactions, save_raw_data as save_insider
from dags.institutional_etl_dag import run_process_data

def dry_run_etl_pipeline():
    """
    Performs a dry run of the entire ETL pipeline, executing each task
    in the correct sequence.
    """
    print("🚀 Starting ETL Pipeline Dry Run...")
    
    config = load_config()
    raw_path = config['storage']['raw_path']
    processed_path = config['storage']['processed_path']
    stocks = config['stocks']
    
    # Clean up previous runs
    print("🧹 Cleaning up old data...")
    if os.path.exists(raw_path):
        shutil.rmtree(raw_path)
    if os.path.exists(processed_path):
        shutil.rmtree(processed_path)
    os.makedirs(raw_path, exist_ok=True)
    os.makedirs(processed_path, exist_ok=True)
    
    # --- Fetching Tasks ---
    print("\n--- 1. Fetching Data ---")
    
    # Fetch common data
    run_fetch_fred()
    run_fetch_news()
    run_fetch_insider()
    
    # Fetch price data for each stock
    for symbol in stocks:
        run_fetch_polygon(symbol)
        
    print("\n--- 2. Processing and Merging Data ---")
    run_process_data()
    
    print("\n✅ ETL Pipeline Dry Run Completed Successfully!")
    
    # --- Verification ---
    print("\n--- 3. Verifying Output ---")
    print("Checking for processed files:")
    all_files_found = True
    for symbol in stocks:
        expected_file = os.path.join(processed_path, f"{symbol}_processed_data.parquet")
        if os.path.exists(expected_file):
            print(f"  ✅ Found: {expected_file}")
        else:
            print(f"  ❌ Missing: {expected_file}")
            all_files_found = False
            
    if all_files_found:
        print("\nAll processed files were created successfully.")
    else:
        print("\nSome processed files are missing. Please check the logs.")

if __name__ == '__main__':
    # A simple way to run the functions from the DAG file
    def run_fetch_polygon_main(symbol):
        start = load_config()['data_fetching']['start_date']
        end = load_config()['data_fetching']['end_date']
        raw_path = load_config()['storage']['raw_path']
        df = fetch_polygon_data(symbol, start, end)
        save_polygon(df, symbol, raw_path)

    def run_fetch_fred_main():
        config = load_config()
        series = config['data_fetching']['fred_series']
        start = config['data_fetching']['start_date']
        end = config['data_fetching']['end_date']
        raw_path = config['storage']['raw_path']
        df = fetch_fred_data(series, start, end)
        save_fred(df, raw_path)
        
    def run_fetch_news_main():
        config = load_config()
        symbols = config['stocks']
        start = config['data_fetching']['start_date']
        end = config['data_fetching']['end_date']
        raw_path = config['storage']['raw_path']
        df = fetch_news_data(symbols, start, end)
        save_news(df, raw_path)

    def run_fetch_insider_main():
        config = load_config()
        symbols = config['stocks']
        raw_path = config['storage']['raw_path']
        df = fetch_insider_transactions(symbols)
        save_insider(df, raw_path)

    # Overwrite the original functions with these test-runners
    run_fetch_polygon = run_fetch_polygon_main
    run_fetch_fred = run_fetch_fred_main
    run_fetch_news = run_fetch_news_main
    run_fetch_insider = run_fetch_insider_main

    dry_run_etl_pipeline() 