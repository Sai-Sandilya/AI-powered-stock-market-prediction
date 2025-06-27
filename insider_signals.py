"""
Insider Trading Signals Module

Provides comprehensive insider trading detection and analysis:
- Fetches insider trading data from multiple sources
- Analyzes transaction patterns and volumes
- Identifies significant insider activities
- Generates trading signals based on insider behavior
- Tracks historical insider patterns
"""
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import os
from dataclasses import dataclass
import json

@dataclass
class InsiderTransaction:
    """Data class for insider trading transactions"""
    symbol: str
    insider_name: str
    title: str
    transaction_type: str  # 'buy', 'sell', 'option_exercise', 'gift'
    shares: int
    price_per_share: float
    total_value: float
    transaction_date: datetime
    filing_date: datetime
    source: str

class InsiderTradingAnalyzer:
    """Comprehensive insider trading analysis"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('ALPHA_VANTAGE_KEY')
        self.base_url = "https://www.alphavantage.co/query"
        
    def fetch_insider_transactions(self, symbol: str, days_back: int = 30) -> List[InsiderTransaction]:
        """Fetch insider trading data from Alpha Vantage"""
        if not self.api_key:
            print("Warning: ALPHA_VANTAGE_KEY not found. Using sample data.")
            return self._get_sample_insider_data(symbol)
            
        try:
            params = {
                'function': 'INSIDER_TRANSACTIONS',
                'symbol': symbol,
                'apikey': self.api_key
            }
            
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if 'Error Message' in data:
                print(f"API Error: {data['Error Message']}")
                return self._get_sample_insider_data(symbol)
                
            transactions = data.get('insiderTransactions', [])
            return self._parse_insider_transactions(transactions, symbol)
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching insider data: {e}")
            return self._get_sample_insider_data(symbol)
        except Exception as e:
            print(f"Unexpected error: {e}")
            return self._get_sample_insider_data(symbol)
    
    def _parse_insider_transactions(self, transactions: List[Dict], symbol: str) -> List[InsiderTransaction]:
        """Parse raw transaction data into structured format"""
        parsed_transactions = []
        
        for tx in transactions:
            try:
                # Parse transaction type
                tx_type = self._parse_transaction_type(tx.get('transactionType', ''))
                
                # Parse dates
                tx_date = datetime.strptime(tx.get('transactionDate', ''), '%Y-%m-%d')
                filing_date = datetime.strptime(tx.get('filingDate', ''), '%Y-%m-%d')
                
                # Parse numeric values
                shares = int(tx.get('sharesTraded', 0))
                price = float(tx.get('pricePerShare', 0))
                total_value = shares * price
                
                transaction = InsiderTransaction(
                    symbol=symbol,
                    insider_name=tx.get('insiderName', 'Unknown'),
                    title=tx.get('insiderTitle', 'Unknown'),
                    transaction_type=tx_type,
                    shares=shares,
                    price_per_share=price,
                    total_value=total_value,
                    transaction_date=tx_date,
                    filing_date=filing_date,
                    source='Alpha Vantage'
                )
                parsed_transactions.append(transaction)
                
            except (ValueError, KeyError) as e:
                print(f"Error parsing transaction: {e}")
                continue
                
        return parsed_transactions
    
    def _parse_transaction_type(self, raw_type: str) -> str:
        """Parse and standardize transaction types"""
        raw_type = raw_type.lower()
        
        if any(word in raw_type for word in ['buy', 'purchase', 'acquisition']):
            return 'buy'
        elif any(word in raw_type for word in ['sell', 'disposition', 'sale']):
            return 'sell'
        elif any(word in raw_type for word in ['option', 'exercise']):
            return 'option_exercise'
        elif any(word in raw_type for word in ['gift', 'donation']):
            return 'gift'
        else:
            return 'other'
    
    def analyze_insider_patterns(self, transactions: List[InsiderTransaction]) -> Dict:
        """Analyze insider trading patterns"""
        if not transactions:
            return self._get_empty_analysis()
        
        df = pd.DataFrame([vars(tx) for tx in transactions])
        
        # Basic statistics
        total_buys = len(df[df['transaction_type'] == 'buy'])
        total_sells = len(df[df['transaction_type'] == 'sell'])
        total_value_buys = df[df['transaction_type'] == 'buy']['total_value'].sum()
        total_value_sells = df[df['transaction_type'] == 'sell']['total_value'].sum()
        
        # Volume analysis
        avg_buy_size = df[df['transaction_type'] == 'buy']['shares'].mean() if total_buys > 0 else 0
        avg_sell_size = df[df['transaction_type'] == 'sell']['shares'].mean() if total_sells > 0 else 0
        
        # Recent activity (last 7 days)
        recent_date = datetime.now() - timedelta(days=7)
        recent_transactions = df[df['transaction_date'] >= recent_date]
        recent_buys = len(recent_transactions[recent_transactions['transaction_type'] == 'buy'])
        recent_sells = len(recent_transactions[recent_transactions['transaction_type'] == 'sell'])
        
        # Signal generation
        buy_signal = self._generate_buy_signal(total_buys, total_sells, total_value_buys, total_value_sells, recent_buys, recent_sells)
        sell_signal = self._generate_sell_signal(total_buys, total_sells, total_value_buys, total_value_sells, recent_buys, recent_sells)
        
        return {
            'summary': {
                'total_transactions': len(transactions),
                'total_buys': total_buys,
                'total_sells': total_sells,
                'total_value_buys': total_value_buys,
                'total_value_sells': total_value_sells,
                'net_insider_activity': total_value_buys - total_value_sells
            },
            'volume_analysis': {
                'avg_buy_size': avg_buy_size,
                'avg_sell_size': avg_sell_size,
                'largest_buy': df[df['transaction_type'] == 'buy']['shares'].max() if total_buys > 0 else 0,
                'largest_sell': df[df['transaction_type'] == 'sell']['shares'].max() if total_sells > 0 else 0
            },
            'recent_activity': {
                'recent_buys': recent_buys,
                'recent_sells': recent_sells,
                'days_analyzed': 7
            },
            'signals': {
                'buy_signal': buy_signal,
                'sell_signal': sell_signal,
                'confidence': self._calculate_confidence(total_buys, total_sells, total_value_buys, total_value_sells)
            },
            'top_insiders': self._get_top_insiders(df)
        }
    
    def _generate_buy_signal(self, buys: int, sells: int, buy_value: float, sell_value: float, recent_buys: int, recent_sells: int) -> str:
        """Generate buy signal based on insider activity"""
        if buys == 0 and sells == 0:
            return 'neutral'
        
        # Strong buy signals
        if recent_buys >= 3 and recent_sells == 0:
            return 'strong_buy'
        if buy_value > sell_value * 2 and buys > sells:
            return 'buy'
        if recent_buys > recent_sells * 2:
            return 'buy'
        
        # Weak buy signals
        if buys > sells and buy_value > sell_value:
            return 'weak_buy'
        
        return 'neutral'
    
    def _generate_sell_signal(self, buys: int, sells: int, buy_value: float, sell_value: float, recent_buys: int, recent_sells: int) -> str:
        """Generate sell signal based on insider activity"""
        if buys == 0 and sells == 0:
            return 'neutral'
        
        # Strong sell signals
        if recent_sells >= 3 and recent_buys == 0:
            return 'strong_sell'
        if sell_value > buy_value * 2 and sells > buys:
            return 'sell'
        if recent_sells > recent_buys * 2:
            return 'sell'
        
        # Weak sell signals
        if sells > buys and sell_value > buy_value:
            return 'weak_sell'
        
        return 'neutral'
    
    def _calculate_confidence(self, buys: int, sells: int, buy_value: float, sell_value: float) -> float:
        """Calculate confidence level of signals (0-1)"""
        total_transactions = buys + sells
        if total_transactions == 0:
            return 0.0
        
        # Higher confidence with more transactions and larger values
        volume_factor = min((buy_value + sell_value) / 1000000, 1.0)  # Normalize to 1M
        frequency_factor = min(total_transactions / 10, 1.0)  # Normalize to 10 transactions
        
        return (volume_factor + frequency_factor) / 2
    
    def _get_top_insiders(self, df: pd.DataFrame) -> List[Dict]:
        """Get top insiders by transaction volume"""
        if df.empty:
            return []
        
        insider_stats = df.groupby('insider_name').agg({
            'total_value': 'sum',
            'shares': 'sum',
            'transaction_type': 'count'
        }).reset_index()
        
        insider_stats = insider_stats.sort_values('total_value', ascending=False)
        
        return [
            {
                'name': row['insider_name'],
                'total_value': row['total_value'],
                'total_shares': row['shares'],
                'transaction_count': row['transaction_type']
            }
            for _, row in insider_stats.head(5).iterrows()
        ]
    
    def _get_empty_analysis(self) -> Dict:
        """Return empty analysis structure"""
        return {
            'summary': {
                'total_transactions': 0,
                'total_buys': 0,
                'total_sells': 0,
                'total_value_buys': 0,
                'total_value_sells': 0,
                'net_insider_activity': 0
            },
            'volume_analysis': {
                'avg_buy_size': 0,
                'avg_sell_size': 0,
                'largest_buy': 0,
                'largest_sell': 0
            },
            'recent_activity': {
                'recent_buys': 0,
                'recent_sells': 0,
                'days_analyzed': 0
            },
            'signals': {
                'buy_signal': 'neutral',
                'sell_signal': 'neutral',
                'confidence': 0.0
            },
            'top_insiders': []
        }
    
    def _get_sample_insider_data(self, symbol: str) -> List[InsiderTransaction]:
        """Return sample insider trading data for testing"""
        sample_transactions = [
            InsiderTransaction(
                symbol=symbol,
                insider_name="John Smith",
                title="CEO",
                transaction_type="buy",
                shares=10000,
                price_per_share=150.0,
                total_value=1500000.0,
                transaction_date=datetime.now() - timedelta(days=2),
                filing_date=datetime.now() - timedelta(days=1),
                source="Sample Data"
            ),
            InsiderTransaction(
                symbol=symbol,
                insider_name="Jane Doe",
                title="CFO",
                transaction_type="sell",
                shares=5000,
                price_per_share=148.0,
                total_value=740000.0,
                transaction_date=datetime.now() - timedelta(days=5),
                filing_date=datetime.now() - timedelta(days=4),
                source="Sample Data"
            ),
            InsiderTransaction(
                symbol=symbol,
                insider_name="Mike Johnson",
                title="CTO",
                transaction_type="buy",
                shares=8000,
                price_per_share=152.0,
                total_value=1216000.0,
                transaction_date=datetime.now() - timedelta(days=1),
                filing_date=datetime.now(),
                source="Sample Data"
            )
        ]
        return sample_transactions

# Legacy function for backward compatibility
def fetch_insider_transactions(api_key: str, symbol: str) -> list:
    analyzer = InsiderTradingAnalyzer(api_key)
    transactions = analyzer.fetch_insider_transactions(symbol)
    return [vars(tx) for tx in transactions]
