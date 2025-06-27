from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from data_ingestion import fetch_yfinance
from model_training import train_model
import os

def job():
    df = fetch_yfinance('BAJAJ-AUTO.NS', '2020-01-01', datetime.today().strftime('%Y-%m-%d'))
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/BAJAJ-AUTO.NS.csv', index=False)
    os.makedirs('models', exist_ok=True)
    train_model('data/BAJAJ-AUTO.NS.csv', ['MA_20','MA_50','RSI_14','Volatility'])

with DAG('stock_pipeline', start_date=datetime(2025,6,20), schedule_interval='@daily') as dag:
    task = PythonOperator(task_id='run_job', python_callable=job)
