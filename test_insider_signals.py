#!/usr/bin/env python3
"""
Test script for insider trading signals functionality
"""
from insider_signals import InsiderTradingAnalyzer
from config import Config
import json

def test_insider_signals():
    """Test the insider trading signals functionality"""
    print("👥 Testing Insider Trading Signals")
    print("=" * 50)
    
    # Initialize insider analyzer
    analyzer = InsiderTradingAnalyzer()
    
    # Test 1: Check API key status
    print("\n1. API Key Status:")
    api_status = Config.validate_api_keys()
    for key, status in api_status.items():
        status_icon = "✅" if status else "❌"
        print(f"   {status_icon} {key}: {'Configured' if status else 'Not configured'}")
    
    # Test 2: Fetch insider transactions
    print("\n2. Fetching Insider Transactions:")
    symbol = "AAPL"
    transactions = analyzer.fetch_insider_transactions(symbol)
    
    print(f"   📊 Found {len(transactions)} insider transactions for {symbol}")
    
    # Display transactions
    for i, tx in enumerate(transactions, 1):
        tx_type_emoji = "🟢" if tx.transaction_type == "buy" else "🔴" if tx.transaction_type == "sell" else "🟡"
        print(f"   {i}. {tx_type_emoji} {tx.insider_name} ({tx.title})")
        print(f"      Type: {tx.transaction_type.upper()}")
        print(f"      Shares: {tx.shares:,}")
        print(f"      Price: ${tx.price_per_share:.2f}")
        print(f"      Total Value: ${tx.total_value:,.2f}")
        print(f"      Date: {tx.transaction_date.strftime('%Y-%m-%d')}")
        print()
    
    # Test 3: Analyze insider patterns
    print("3. Insider Pattern Analysis:")
    analysis = analyzer.analyze_insider_patterns(transactions)
    
    # Summary
    summary = analysis['summary']
    print(f"   📈 Summary:")
    print(f"      Total Transactions: {summary['total_transactions']}")
    print(f"      Buys: {summary['total_buys']} (${summary['total_value_buys']:,.2f})")
    print(f"      Sells: {summary['total_sells']} (${summary['total_value_sells']:,.2f})")
    print(f"      Net Activity: ${summary['net_insider_activity']:,.2f}")
    
    # Volume analysis
    volume = analysis['volume_analysis']
    print(f"   📊 Volume Analysis:")
    print(f"      Avg Buy Size: {volume['avg_buy_size']:,.0f} shares")
    print(f"      Avg Sell Size: {volume['avg_sell_size']:,.0f} shares")
    print(f"      Largest Buy: {volume['largest_buy']:,.0f} shares")
    print(f"      Largest Sell: {volume['largest_sell']:,.0f} shares")
    
    # Recent activity
    recent = analysis['recent_activity']
    print(f"   ⏰ Recent Activity (Last {recent['days_analyzed']} days):")
    print(f"      Recent Buys: {recent['recent_buys']}")
    print(f"      Recent Sells: {recent['recent_sells']}")
    
    # Signals
    signals = analysis['signals']
    print(f"   🎯 Trading Signals:")
    
    buy_signal = signals['buy_signal']
    buy_emoji = {
        'strong_buy': '🟢🟢',
        'buy': '🟢',
        'weak_buy': '🟡',
        'neutral': '⚪',
        'weak_sell': '🟡',
        'sell': '🔴',
        'strong_sell': '🔴🔴'
    }.get(buy_signal, '⚪')
    
    sell_signal = signals['sell_signal']
    sell_emoji = {
        'strong_sell': '🔴🔴',
        'sell': '🔴',
        'weak_sell': '🟡',
        'neutral': '⚪',
        'weak_buy': '🟡',
        'buy': '🟢',
        'strong_buy': '🟢🟢'
    }.get(sell_signal, '⚪')
    
    print(f"      Buy Signal: {buy_emoji} {buy_signal.upper()}")
    print(f"      Sell Signal: {sell_emoji} {sell_signal.upper()}")
    print(f"      Confidence: {signals['confidence']:.1%}")
    
    # Top insiders
    top_insiders = analysis['top_insiders']
    if top_insiders:
        print(f"   👑 Top Insiders by Volume:")
        for i, insider in enumerate(top_insiders, 1):
            print(f"      {i}. {insider['name']}")
            print(f"         Total Value: ${insider['total_value']:,.2f}")
            print(f"         Total Shares: {insider['total_shares']:,.0f}")
            print(f"         Transactions: {insider['transaction_count']}")
    
    # Test 4: Signal interpretation
    print("\n4. Signal Interpretation:")
    net_activity = summary['net_insider_activity']
    if net_activity > 0:
        print(f"   🟢 Net insider activity is POSITIVE (${net_activity:,.2f})")
        print("      → Insiders are buying more than selling")
        print("      → Generally bullish signal")
    elif net_activity < 0:
        print(f"   🔴 Net insider activity is NEGATIVE (${abs(net_activity):,.2f})")
        print("      → Insiders are selling more than buying")
        print("      → Generally bearish signal")
    else:
        print("   ⚪ No net insider activity")
        print("      → Balanced buying and selling")
    
    # Test 5: Configuration instructions
    print("\n5. Configuration Instructions:")
    missing_keys = Config.get_missing_keys()
    if 'ALPHA_VANTAGE_KEY' in missing_keys:
        print("   ⚠️  Missing Alpha Vantage API key for real insider data")
        print("\n   📝 To get API key:")
        print("      - Visit: https://www.alphavantage.co/")
        print("      - Sign up for free account")
        print("      - Get your API key")
        print("\n   🔧 Set environment variable:")
        print("      export ALPHA_VANTAGE_KEY=your_key_here")
    else:
        print("   ✅ Alpha Vantage API key is configured!")
    
    print("\n" + "=" * 50)
    print("✅ Insider trading signals test completed!")

if __name__ == "__main__":
    test_insider_signals() 