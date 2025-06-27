"""
Configuration file for API keys and settings
"""
import os
from typing import Optional

class Config:
    """Configuration class for the stock predictor application"""
    
    # News API Configuration
    NEWSAPI_KEY: Optional[str] = os.getenv('NEWSAPI_KEY')
    
    # Alpha Vantage API Configuration
    ALPHA_VANTAGE_KEY: Optional[str] = os.getenv('ALPHA_VANTAGE_KEY')
    
    # Model Configuration
    MODEL_PATH: str = 'models/lstm_model.pth'
    DATA_PATH: str = 'data'
    
    # API Configuration
    API_HOST: str = '0.0.0.0'
    API_PORT: int = 8000
    
    # Streamlit Configuration
    STREAMLIT_PORT: int = 8501
    
    # Feature Configuration
    FEATURE_COLUMNS: list = ['MA_20', 'MA_50', 'RSI_14', 'Volatility']
    SEQUENCE_LENGTH: int = 60
    
    # Training Configuration
    EPOCHS: int = 10
    LEARNING_RATE: float = 1e-3
    BATCH_SIZE: int = 32
    
    # News Configuration
    NEWS_DAYS_BACK: int = 7
    NEWS_PAGE_SIZE: int = 100
    
    @classmethod
    def validate_api_keys(cls) -> dict:
        """Validate and return status of API keys"""
        status = {
            'newsapi': bool(cls.NEWSAPI_KEY),
            'alpha_vantage': bool(cls.ALPHA_VANTAGE_KEY),
            'all_configured': bool(cls.NEWSAPI_KEY and cls.ALPHA_VANTAGE_KEY)
        }
        return status
    
    @classmethod
    def get_missing_keys(cls) -> list:
        """Get list of missing API keys"""
        missing = []
        if not cls.NEWSAPI_KEY:
            missing.append('NEWSAPI_KEY')
        if not cls.ALPHA_VANTAGE_KEY:
            missing.append('ALPHA_VANTAGE_KEY')
        return missing

# Environment variable setup instructions
ENV_SETUP_INSTRUCTIONS = """
To set up API keys, create a .env file in the project root with:

NEWSAPI_KEY=your_newsapi_key_here
ALPHA_VANTAGE_KEY=your_alpha_vantage_key_here

Or set environment variables:

export NEWSAPI_KEY=your_newsapi_key_here
export ALPHA_VANTAGE_KEY=your_alpha_vantage_key_here

Get API keys from:
- NewsAPI: https://newsapi.org/
- Alpha Vantage: https://www.alphavantage.co/
""" 