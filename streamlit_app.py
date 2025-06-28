import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from data_ingestion import fetch_yfinance
from feature_engineering import (
    add_technical_indicators, 
    get_trading_signals, 
    get_comprehensive_features,
    get_macro_trading_signals
)
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from future_forecasting import FutureForecaster
import joblib
import os

# Optional imports - only import if available (silent during build)
try:
    from train_enhanced_system import EnhancedTrainingSystem
    ENHANCED_TRAINING_AVAILABLE = True
except ImportError:
    ENHANCED_TRAINING_AVAILABLE = False

try:
    from backtesting import Backtester
    BACKTESTING_AVAILABLE = True
except ImportError:
    BACKTESTING_AVAILABLE = False

try:
    from macro_indicators import MacroIndicators
    MACRO_INDICATORS_AVAILABLE = True
except ImportError:
    MACRO_INDICATORS_AVAILABLE = False

# Check for TensorFlow availability
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

# Check for additional ML libraries
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

st.set_page_config(page_title="Stock Predictor Enhanced Dashboard", layout="wide")
st.title("📈 Advanced Stock Predictor Dashboard with Macro Analysis")

# Display system status in the app (only show if user wants to see it)
with st.expander("🔧 System Status & Available Features", expanded=False):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Core Features")
        st.success("✅ Stock Data (yfinance)")
        st.success("✅ Technical Analysis")
        st.success("✅ Interactive Charts (Plotly)")
        st.success("✅ Price Forecasting")
        st.success("✅ Trading Signals")
        
    with col2:
        st.subheader("🤖 Advanced Features")
        if TENSORFLOW_AVAILABLE:
            st.success("✅ LSTM Neural Networks (TensorFlow)")
        else:
            st.info("ℹ️ Using Advanced Scikit-Learn Models (TensorFlow-free deployment)")
            
        if ENHANCED_TRAINING_AVAILABLE:
            st.success("✅ Enhanced Training System")
        else:
            st.info("ℹ️ Enhanced Training System (Optional module)")
            
        if BACKTESTING_AVAILABLE:
            st.success("✅ Advanced Backtesting")
        else:
            st.info("ℹ️ Advanced Backtesting (Optional module)")
            
        if MACRO_INDICATORS_AVAILABLE:
            st.success("✅ Macro Economic Indicators")
        else:
            st.info("ℹ️ Macro Indicators (Optional module)")
            
        if OPTUNA_AVAILABLE:
            st.success("✅ Hyperparameter Optimization")
        else:
            st.info("ℹ️ Hyperparameter Tuning (Optuna not available)")
            
        if MATPLOTLIB_AVAILABLE:
            st.success("✅ Advanced Visualizations")
        else:
            st.info("ℹ️ Some Advanced Charts (Matplotlib not available)")
    
    st.info("💡 All core functionality works regardless of optional features!")

# Helper functions for macro overlay charts
def create_macro_overlay_chart(stock_data, macro_data, selected_indicators, normalize_data, symbol):
    """Create an overlay chart with stock price and macro indicators on dual y-axes."""
    fig = go.Figure()
    
    # Add stock price (primary y-axis)
    fig.add_trace(go.Scatter(
        x=stock_data['Date'],
        y=stock_data['Close'],
        name=f'{symbol} Price',
        yaxis='y',
        line=dict(color='#1f77b4', width=2)
    ))
    
    # Add macro indicators (secondary y-axis)
    colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    
    for i, indicator in enumerate(selected_indicators):
        if indicator in macro_data:
            data = macro_data[indicator]
            if not data.empty:
                y_data = data.iloc[:, 1]  # Assuming second column is the value
                
                if normalize_data:
                    # Normalize to 0-1 range
                    y_data = (y_data - y_data.min()) / (y_data.max() - y_data.min())
                    # Scale to stock price range
                    stock_range = stock_data['Close'].max() - stock_data['Close'].min()
                    y_data = y_data * stock_range + stock_data['Close'].min()
                
                fig.add_trace(go.Scatter(
                    x=data.iloc[:, 0],  # Date column
                    y=y_data,
                    name=indicator,
                    yaxis='y2',
                    line=dict(color=colors[i % len(colors)], width=1, dash='dot'),
                    opacity=0.7
                ))
    
    # Update layout
    fig.update_layout(
        title=f'{symbol} Price vs Macro Indicators (Overlay)',
        xaxis_title='Date',
        yaxis=dict(title=f'{symbol} Price', side='left'),
        yaxis2=dict(title='Macro Indicators', side='right', overlaying='y'),
        hovermode='x unified',
        height=600,
        showlegend=True
    )
    
    return fig

def create_macro_subplot_chart(stock_data, macro_data, selected_indicators, normalize_data, symbol):
    """Create a subplot chart with stock price and macro indicators."""
    n_indicators = len(selected_indicators)
    fig = make_subplots(
        rows=n_indicators + 1, cols=1,
        subplot_titles=[f'{symbol} Price'] + selected_indicators,
        vertical_spacing=0.05
    )
    
    # Add stock price
    fig.add_trace(
        go.Scatter(x=stock_data['Date'], y=stock_data['Close'], name=f'{symbol} Price'),
        row=1, col=1
    )
    
    # Add macro indicators
    for i, indicator in enumerate(selected_indicators):
        if indicator in macro_data:
            data = macro_data[indicator]
            if not data.empty:
                fig.add_trace(
                    go.Scatter(x=data.iloc[:, 0], y=data.iloc[:, 1], name=indicator),
                    row=i+2, col=1
                )
    
    fig.update_layout(height=200 * (n_indicators + 1), showlegend=False)
    return fig

def create_correlation_heatmap(stock_data, macro_data, selected_indicators, symbol):
    """Create a correlation heatmap between stock price and macro indicators."""
    # Prepare data for correlation
    correlation_data = {}
    correlation_data[f'{symbol}_Price'] = stock_data['Close']
    
    for indicator in selected_indicators:
        if indicator in macro_data:
            data = macro_data[indicator]
            if not data.empty:
                # Align dates and interpolate if necessary
                aligned_data = data.set_index(data.iloc[:, 0])[data.iloc[:, 1]]
                aligned_data = aligned_data.reindex(stock_data['Date']).interpolate()
                correlation_data[indicator] = aligned_data
    
    # Calculate correlation matrix
    corr_df = pd.DataFrame(correlation_data).corr()
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=corr_df.values,
        x=corr_df.columns,
        y=corr_df.index,
        colorscale='RdBu',
        zmid=0,
        text=corr_df.round(3).values,
        texttemplate="%{text}",
        textfont={"size": 10}
    ))
    
    fig.update_layout(
        title=f'Correlation Heatmap: {symbol} vs Macro Indicators',
        height=500
    )
    
    return fig

def calculate_macro_correlations(stock_data, macro_data, selected_indicators):
    """Calculate correlation coefficients between stock price and macro indicators."""
    correlations = []
    
    for indicator in selected_indicators:
        if indicator in macro_data:
            data = macro_data[indicator]
            if not data.empty:
                # Align dates and interpolate
                aligned_data = data.set_index(data.iloc[:, 0])[data.iloc[:, 1]]
                aligned_data = aligned_data.reindex(stock_data['Date']).interpolate()
                
                # Calculate correlation
                correlation = stock_data['Close'].corr(aligned_data)
                correlations.append({
                    'Indicator': indicator,
                    'Correlation': correlation,
                    'Strength': 'Strong' if abs(correlation) > 0.7 else 'Moderate' if abs(correlation) > 0.3 else 'Weak',
                    'Direction': 'Positive' if correlation > 0 else 'Negative'
                })
    
    return pd.DataFrame(correlations).sort_values('Correlation', key=abs, ascending=False)

def generate_correlation_insights(correlation_df, symbol):
    """Generate insights from correlation analysis."""
    insights = []
    
    if not correlation_df.empty:
        # Strongest positive correlation
        strongest_pos = correlation_df[correlation_df['Correlation'] > 0].iloc[0] if len(correlation_df[correlation_df['Correlation'] > 0]) > 0 else None
        if strongest_pos is not None:
            insights.append(f"**{strongest_pos['Indicator']}** shows the strongest positive correlation ({strongest_pos['Correlation']:.3f}) with {symbol} price.")
        
        # Strongest negative correlation
        strongest_neg = correlation_df[correlation_df['Correlation'] < 0].iloc[0] if len(correlation_df[correlation_df['Correlation'] < 0]) > 0 else None
        if strongest_neg is not None:
            insights.append(f"**{strongest_neg['Indicator']}** shows the strongest negative correlation ({strongest_neg['Correlation']:.3f}) with {symbol} price.")
        
        # Overall correlation strength
        strong_correlations = len(correlation_df[correlation_df['Strength'] == 'Strong'])
        if strong_correlations > 0:
            insights.append(f"{strong_correlations} indicator(s) show strong correlation with {symbol} price.")
        
        # Trading implications
        if strongest_pos is not None and strongest_pos['Correlation'] > 0.7:
            insights.append(f"Consider monitoring **{strongest_pos['Indicator']}** as a leading indicator for {symbol} price movements.")
    
    return insights

def is_indian_stock(symbol: str) -> bool:
    """
    Check if the stock symbol is for an Indian stock.
    Indian stocks typically end with .NS (NSE) or .BO (BSE)
    """
    return symbol.upper().endswith(('.NS', '.BO', '.NSE', '.BSE'))

def get_currency_symbol(symbol: str) -> str:
    """
    Get the appropriate currency symbol based on the stock.
    """
    if is_indian_stock(symbol):
        return "₹"
    else:
        return "$"

def format_currency(value: float, symbol: str) -> str:
    """
    Format currency value with appropriate symbol and formatting.
    """
    if symbol == "₹":
        # Indian Rupee formatting
        if value >= 10000000:  # 1 crore
            return f"₹{value/10000000:.2f}Cr"
        elif value >= 100000:  # 1 lakh
            return f"₹{value/100000:.2f}L"
        else:
            return f"₹{value:,.2f}"
    else:
        # US Dollar formatting
        return f"${value:,.2f}"

# Sidebar controls
st.sidebar.header("Stock Selection")

# Popular stocks dropdown
popular_stocks = {
    "US Stocks": ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN", "META", "NVDA", "NFLX"],
    "Indian Stocks": [
        # Nifty 50 - Complete List
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS", 
        "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "AXISBANK.NS",
        "KOTAKBANK.NS", "ASIANPAINT.NS", "MARUTI.NS", "SUNPHARMA.NS", "TATAMOTORS.NS",
        "WIPRO.NS", "ULTRACEMCO.NS", "TITAN.NS", "BAJFINANCE.NS", "NESTLEIND.NS",
        "POWERGRID.NS", "TECHM.NS", "BAJAJFINSV.NS", "NTPC.NS", "HCLTECH.NS",
        "ONGC.NS", "JSWSTEEL.NS", "TATACONSUM.NS", "ADANIENT.NS", "COALINDIA.NS",
        "HINDALCO.NS", "TATASTEEL.NS", "BRITANNIA.NS", "GRASIM.NS", "INDUSINDBK.NS",
        "M&M.NS", "BAJAJ-AUTO.NS", "VEDL.NS", "UPL.NS", "BPCL.NS",
        "SBILIFE.NS", "HDFCLIFE.NS", "DIVISLAB.NS", "CIPLA.NS", "EICHERMOT.NS",
        "HEROMOTOCO.NS", "SHREECEM.NS", "ADANIPORTS.NS", "DRREDDY.NS", "APOLLOHOSP.NS",
        "TATACONSUM.NS", "BAJFINANCE.NS", "HINDUNILVR.NS", "NESTLEIND.NS",
        
        # Nifty Next 50 - Complete List
        "ADANIENT.NS", "ADANIPORTS.NS", "APOLLOHOSP.NS", "BAJAJFINSV.NS", "BAJFINANCE.NS",
        "BHARTIARTL.NS", "BRITANNIA.NS", "CIPLA.NS", "COALINDIA.NS", "DIVISLAB.NS",
        "DRREDDY.NS", "EICHERMOT.NS", "GRASIM.NS", "HCLTECH.NS", "HDFCLIFE.NS",
        "HEROMOTOCO.NS", "HINDALCO.NS", "HINDUNILVR.NS", "ICICIBANK.NS", "INDUSINDBK.NS",
        "INFY.NS", "ITC.NS", "JSWSTEEL.NS", "KOTAKBANK.NS", "M&M.NS",
        "MARUTI.NS", "NESTLEIND.NS", "NTPC.NS", "ONGC.NS", "POWERGRID.NS",
        "RELIANCE.NS", "SBILIFE.NS", "SBIN.NS", "SHREECEM.NS", "SUNPHARMA.NS",
        "TATACONSUM.NS", "TATAMOTORS.NS", "TATASTEEL.NS", "TCS.NS", "TECHM.NS",
        "TITAN.NS", "ULTRACEMCO.NS", "UPL.NS", "VEDL.NS", "WIPRO.NS",
        
        # Banking & Financial Services
        "HDFC.NS", "KOTAKBANK.NS", "AXISBANK.NS", "ICICIBANK.NS", "SBIN.NS",
        "INDUSINDBK.NS", "FEDERALBNK.NS", "IDFCFIRSTB.NS", "BANDHANBNK.NS", "PNB.NS",
        "CANBK.NS", "UNIONBANK.NS", "BANKBARODA.NS", "IOB.NS", "UCOBANK.NS",
        "CENTRALBK.NS", "MAHABANK.NS", "INDIANB.NS", "PSB.NS", "ALLAHABAD.NS",
        "HDFCAMC.NS", "ICICIPRULI.NS", "SBICARD.NS", "BAJFINANCE.NS", "BAJAJFINSV.NS",
        "CHOLAFIN.NS", "MUTHOOTFIN.NS", "PEL.NS", "RECLTD.NS", "PFC.NS",
        
        # IT & Technology
        "TCS.NS", "INFY.NS", "WIPRO.NS", "HCLTECH.NS", "TECHM.NS",
        "MINDTREE.NS", "LTI.NS", "MPHASIS.NS", "PERSISTENT.NS", "COFORGE.NS",
        "L&TINFOTECH.NS", "HEXAWARE.NS", "NIITTECH.NS", "CYIENT.NS", "KPITTECH.NS",
        "SONATSOFTW.NS", "RAMCOSYS.NS", "INTELLECT.NS", "QUESS.NS", "TEAMLEASE.NS",
        "APLLTD.NS", "DATAPATTERNS.NS", "MAPMYINDIA.NS", "TATAELXSI.NS", "ZENSARTECH.NS",
        
        # Auto & Manufacturing
        "TATAMOTORS.NS", "MARUTI.NS", "EICHERMOT.NS", "HEROMOTOCO.NS", "BAJAJ-AUTO.NS",
        "M&M.NS", "ASHOKLEY.NS", "TVSMOTOR.NS", "ESCORTS.NS", "MRF.NS",
        "CEAT.NS", "APOLLOTYRE.NS", "JKTYRE.NS", "BALKRISIND.NS", "AMARAJABAT.NS",
        "EXIDEIND.NS", "SUNDARMFIN.NS", "BAJAJELEC.NS", "CROMPTON.NS", "HAVELLS.NS",
        "VOLTAS.NS", "BLUESTARCO.NS", "WHIRLPOOL.NS", "GODREJCP.NS", "MARICO.NS",
        
        # Pharma & Healthcare
        "SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS", "BIOCON.NS",
        "APOLLOHOSP.NS", "FORTIS.NS", "ALKEM.NS", "TORNTPHARM.NS", "CADILAHC.NS",
        "LUPIN.NS", "AUROPHARMA.NS", "GLENMARK.NS", "NATCOPHARM.NS", "AJANTPHARM.NS",
        "LAURUSLABS.NS", "GRANULES.NS", "IPCA.NS", "PFIZER.NS", "SANOFI.NS",
        "ABBOTINDIA.NS", "ALKEM.NS", "TORNTPHARM.NS", "CADILAHC.NS",
        
        # Consumer Goods & FMCG
        "HINDUNILVR.NS", "ITC.NS", "NESTLEIND.NS", "BRITANNIA.NS", "MARICO.NS",
        "DABUR.NS", "COLPAL.NS", "GODREJCP.NS", "EMAMILTD.NS", "VBL.NS",
        "UBL.NS", "RADICO.NS", "UNILEVER.NS", "GILLETTE.NS", "MARICO.NS",
        "DABUR.NS", "COLPAL.NS", "GODREJCP.NS", "EMAMILTD.NS", "VBL.NS",
        "UBL.NS", "RADICO.NS", "UNILEVER.NS", "GILLETTE.NS", "MARICO.NS",
        
        # Energy & Oil
        "RELIANCE.NS", "ONGC.NS", "COALINDIA.NS", "NTPC.NS", "POWERGRID.NS",
        "BPCL.NS", "IOC.NS", "HPCL.NS", "ADANIGREEN.NS", "TATAPOWER.NS",
        "ADANITRANS.NS", "ADANIGAS.NS", "ADANIPOWER.NS", "ADANIENT.NS", "ADANIPORTS.NS",
        "GAIL.NS", "PETRONET.NS", "OIL.NS", "CONCOR.NS", "CONTAINER.NS",
        
        # Metals & Mining
        "TATASTEEL.NS", "JSWSTEEL.NS", "HINDALCO.NS", "VEDL.NS", "HINDCOPPER.NS",
        "NATIONALUM.NS", "WELCORP.NS", "JINDALSTEL.NS", "SAIL.NS", "NMDC.NS",
        "HINDZINC.NS", "VEDL.NS", "HINDALCO.NS", "TATASTEEL.NS", "JSWSTEEL.NS",
        "SAIL.NS", "NMDC.NS", "HINDZINC.NS", "VEDL.NS", "HINDALCO.NS",
        
        # Real Estate & Construction
        "DLF.NS", "GODREJPROP.NS", "SUNTV.NS", "PRESTIGE.NS", "BRIGADE.NS",
        "OBEROIRLTY.NS", "PHOENIXLTD.NS", "SOBHA.NS", "GODREJIND.NS", "KOLTEPATIL.NS",
        "LODHA.NS", "MACROTECH.NS", "GODREJPROP.NS", "DLF.NS", "SUNTV.NS",
        "PRESTIGE.NS", "BRIGADE.NS", "OBEROIRLTY.NS", "PHOENIXLTD.NS", "SOBHA.NS",
        
        # Telecom & Media
        "BHARTIARTL.NS", "IDEA.NS", "VODAFONE.NS", "MTNL.NS", "BSNL.NS",
        "SUNTV.NS", "ZEEL.NS", "PVR.NS", "INOXLEISURE.NS", "PVR.NS",
        "INOXLEISURE.NS", "SUNTV.NS", "ZEEL.NS", "PVR.NS", "INOXLEISURE.NS",
        
        # Cement & Construction
        "ULTRACEMCO.NS", "SHREECEM.NS", "ACC.NS", "AMBUJACEM.NS", "RAMCOCEM.NS",
        "HEIDELBERG.NS", "BIRLACORPN.NS", "JKLAKSHMI.NS", "ORIENTCEM.NS", "MANGALAM.NS",
        "ULTRACEMCO.NS", "SHREECEM.NS", "ACC.NS", "AMBUJACEM.NS", "RAMCOCEM.NS",
        
        # Chemicals & Fertilizers
        "UPL.NS", "COROMANDEL.NS", "CHAMBLFERT.NS", "GSFC.NS", "RCF.NS",
        "NATIONALUM.NS", "HINDALCO.NS", "VEDL.NS", "HINDCOPPER.NS", "NATIONALUM.NS",
        "UPL.NS", "COROMANDEL.NS", "CHAMBLFERT.NS", "GSFC.NS", "RCF.NS",
        
        # Aviation & Logistics
        "INDIGO.NS", "SPICEJET.NS", "JETAIRWAYS.NS", "AIRINDIA.NS", "VISTARA.NS",
        "CONCOR.NS", "CONTAINER.NS", "ADANIPORTS.NS", "ADANITRANS.NS", "ADANIGAS.NS",
        
        # E-commerce & Digital
        "FLIPKART.NS", "AMAZON.NS", "SNAPDEAL.NS", "PAYTM.NS", "ZOMATO.NS",
        "NYKAA.NS", "DELHIVERY.NS", "CARTRADE.NS", "EASEMYTRIP.NS", "MAPMYINDIA.NS",
        
        # Small Cap Gems
        "TATACOMM.NS", "TATACONSUM.NS", "TATAMOTORS.NS", "TATAPOWER.NS", "TATASTEEL.NS",
        "TATACONSUM.NS", "TATAMOTORS.NS", "TATAPOWER.NS", "TATASTEEL.NS", "TATACOMM.NS",
        
        # PSU Banks
        "SBIN.NS", "PNB.NS", "CANBK.NS", "UNIONBANK.NS", "BANKBARODA.NS",
        "IOB.NS", "UCOBANK.NS", "CENTRALBK.NS", "MAHABANK.NS", "INDIANB.NS",
        
        # Private Banks
        "HDFCBANK.NS", "ICICIBANK.NS", "AXISBANK.NS", "KOTAKBANK.NS", "INDUSINDBK.NS",
        "FEDERALBNK.NS", "IDFCFIRSTB.NS", "BANDHANBNK.NS", "RBLBANK.NS", "YESBANK.NS",
        
        # NBFCs & Financial Services
        "BAJFINANCE.NS", "BAJAJFINSV.NS", "HDFC.NS", "HDFCAMC.NS", "ICICIPRULI.NS",
        "SBICARD.NS", "CHOLAFIN.NS", "MUTHOOTFIN.NS", "PEL.NS", "RECLTD.NS",
        
        # Insurance
        "SBILIFE.NS", "HDFCLIFE.NS", "ICICIPRULI.NS", "MAXLIFE.NS", "BAJAJALLIANZ.NS",
        "SHRIRAM.NS", "CHOLAMANDALAM.NS", "BAJAJFINSV.NS", "HDFCAMC.NS", "ICICIPRULI.NS"
    ]
}

stock_category = st.sidebar.selectbox(
    "Select Stock Category",
    ["US Stocks", "Indian Stocks", "Custom"]
)

if stock_category == "Custom":
    symbol = st.sidebar.text_input("Enter Stock Symbol (e.g., AAPL, MSFT, TSLA, RELIANCE.NS)", value="AAPL")
else:
    if stock_category == "Indian Stocks":
        # Add search functionality for Indian stocks
        search_term = st.sidebar.text_input("🔍 Search Indian Stocks", placeholder="Type to search...")
        
        if search_term:
            # Filter stocks based on search term
            filtered_stocks = [stock for stock in popular_stocks[stock_category] 
                             if search_term.upper() in stock.upper()]
            if filtered_stocks:
                symbol = st.sidebar.selectbox(
                    f"Select {stock_category} (Filtered)",
                    filtered_stocks,
                    index=0
                )
            else:
                st.sidebar.warning("No stocks found matching your search.")
                symbol = st.sidebar.selectbox(
                    f"Select {stock_category}",
                    popular_stocks[stock_category],
                    index=0
                )
        else:
            symbol = st.sidebar.selectbox(
                f"Select {stock_category}",
                popular_stocks[stock_category],
                index=0
            )
    else:
        symbol = st.sidebar.selectbox(
            f"Select {stock_category}",
            popular_stocks[stock_category],
            index=0
        )

end_date = st.sidebar.date_input("End Date", value=datetime.now().date(), key="main_end_date")
start_date = st.sidebar.date_input("Start Date", value=(datetime.now() - timedelta(days=365)).date(), key="main_start_date")

# Show currency information
if symbol:
    currency_symbol = get_currency_symbol(symbol)
    is_indian = is_indian_stock(symbol)
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("💰 Currency Information")
    
    if is_indian:
        st.sidebar.success(f"🇮🇳 Indian Stock - Currency: {currency_symbol} (Rupees)")
        st.sidebar.info("💡 Indian stocks use Rupee (₹) currency. Large amounts are shown in Lakhs (L) and Crores (Cr).")
        
        # Show stock category information for Indian stocks
        stock_name = symbol.replace('.NS', '')
        
        # Nifty 50 stocks
        nifty50_stocks = ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK', 'HINDUNILVR', 'ITC', 'SBIN', 
                         'BHARTIARTL', 'AXISBANK', 'KOTAKBANK', 'ASIANPAINT', 'MARUTI', 'SUNPHARMA', 'TATAMOTORS',
                         'WIPRO', 'ULTRACEMCO', 'TITAN', 'BAJFINANCE', 'NESTLEIND', 'POWERGRID', 'TECHM', 
                         'BAJAJFINSV', 'NTPC', 'HCLTECH', 'ONGC', 'JSWSTEEL', 'TATACONSUM', 'ADANIENT', 'COALINDIA',
                         'HINDALCO', 'TATASTEEL', 'BRITANNIA', 'GRASIM', 'INDUSINDBK', 'M&M', 'BAJAJ-AUTO', 
                         'VEDL', 'UPL', 'BPCL', 'SBILIFE', 'HDFCLIFE', 'DIVISLAB', 'CIPLA', 'EICHERMOT', 
                         'HEROMOTOCO', 'SHREECEM', 'ADANIPORTS', 'DRREDDY', 'APOLLOHOSP']
        
        # Banking & Financial
        banking_stocks = ['HDFC', 'KOTAKBANK', 'AXISBANK', 'ICICIBANK', 'SBIN', 'INDUSINDBK', 'FEDERALBNK', 
                         'IDFCFIRSTB', 'BANDHANBNK', 'PNB', 'CANBK', 'UNIONBANK', 'BANKBARODA', 'IOB', 
                         'UCOBANK', 'CENTRALBK', 'MAHABANK', 'INDIANB', 'PSB', 'ALLAHABAD', 'HDFCAMC', 
                         'ICICIPRULI', 'SBICARD', 'BAJFINANCE', 'BAJAJFINSV', 'CHOLAFIN', 'MUTHOOTFIN', 
                         'PEL', 'RECLTD', 'PFC']
        
        # IT & Technology
        it_stocks = ['TCS', 'INFY', 'WIPRO', 'HCLTECH', 'TECHM', 'MINDTREE', 'LTI', 'MPHASIS', 'PERSISTENT', 
                    'COFORGE', 'L&TINFOTECH', 'HEXAWARE', 'NIITTECH', 'CYIENT', 'KPITTECH', 'SONATSOFTW', 
                    'RAMCOSYS', 'INTELLECT', 'QUESS', 'TEAMLEASE', 'APLLTD', 'DATAPATTERNS', 'MAPMYINDIA', 
                    'TATAELXSI', 'ZENSARTECH']
        
        # Auto & Manufacturing
        auto_stocks = ['TATAMOTORS', 'MARUTI', 'EICHERMOT', 'HEROMOTOCO', 'BAJAJ-AUTO', 'M&M', 'ASHOKLEY', 
                      'TVSMOTOR', 'ESCORTS', 'MRF', 'CEAT', 'APOLLOTYRE', 'JKTYRE', 'BALKRISIND', 'AMARAJABAT', 
                      'EXIDEIND', 'SUNDARMFIN', 'BAJAJELEC', 'CROMPTON', 'HAVELLS', 'VOLTAS', 'BLUESTARCO', 
                      'WHIRLPOOL', 'GODREJCP', 'MARICO']
        
        # Pharma & Healthcare
        pharma_stocks = ['SUNPHARMA', 'DRREDDY', 'CIPLA', 'DIVISLAB', 'BIOCON', 'APOLLOHOSP', 'FORTIS', 
                        'ALKEM', 'TORNTPHARM', 'CADILAHC', 'LUPIN', 'AUROPHARMA', 'GLENMARK', 'NATCOPHARM', 
                        'AJANTPHARM', 'LAURUSLABS', 'GRANULES', 'IPCA', 'PFIZER', 'SANOFI', 'ABBOTINDIA']
        
        # Consumer Goods & FMCG
        fmcg_stocks = ['HINDUNILVR', 'ITC', 'NESTLEIND', 'BRITANNIA', 'MARICO', 'DABUR', 'COLPAL', 'GODREJCP', 
                      'EMAMILTD', 'VBL', 'UBL', 'RADICO', 'UNILEVER', 'GILLETTE']
        
        # Energy & Oil
        energy_stocks = ['RELIANCE', 'ONGC', 'COALINDIA', 'NTPC', 'POWERGRID', 'BPCL', 'IOC', 'HPCL', 
                        'ADANIGREEN', 'TATAPOWER', 'ADANITRANS', 'ADANIGAS', 'ADANIPOWER', 'ADANIENT', 
                        'ADANIPORTS', 'GAIL', 'PETRONET', 'OIL', 'CONCOR', 'CONTAINER']
        
        # Metals & Mining
        metals_stocks = ['TATASTEEL', 'JSWSTEEL', 'HINDALCO', 'VEDL', 'HINDCOPPER', 'NATIONALUM', 'WELCORP', 
                        'JINDALSTEL', 'SAIL', 'NMDC', 'HINDZINC']
        
        # Real Estate & Construction
        realty_stocks = ['DLF', 'GODREJPROP', 'SUNTV', 'PRESTIGE', 'BRIGADE', 'OBEROIRLTY', 'PHOENIXLTD', 
                        'SOBHA', 'GODREJIND', 'KOLTEPATIL', 'LODHA', 'MACROTECH']
        
        # Cement & Construction
        cement_stocks = ['ULTRACEMCO', 'SHREECEM', 'ACC', 'AMBUJACEM', 'RAMCOCEM', 'HEIDELBERG', 'BIRLACORPN', 
                        'JKLAKSHMI', 'ORIENTCEM', 'MANGALAM']
        
        # Telecom & Media
        telecom_stocks = ['BHARTIARTL', 'IDEA', 'VODAFONE', 'MTNL', 'BSNL', 'SUNTV', 'ZEEL', 'PVR', 'INOXLEISURE']
        
        # E-commerce & Digital
        digital_stocks = ['FLIPKART', 'AMAZON', 'SNAPDEAL', 'PAYTM', 'ZOMATO', 'NYKAA', 'DELHIVERY', 
                         'CARTRADE', 'EASEMYTRIP', 'MAPMYINDIA']
        
        # Insurance
        insurance_stocks = ['SBILIFE', 'HDFCLIFE', 'ICICIPRULI', 'MAXLIFE', 'BAJAJALLIANZ', 'SHRIRAM', 
                           'CHOLAMANDALAM']
        
        if stock_name in nifty50_stocks:
            st.sidebar.info("🏆 **Nifty 50** - Large Cap Stock")
        elif stock_name in banking_stocks:
            st.sidebar.info("🏦 **Banking & Financial** Sector")
        elif stock_name in it_stocks:
            st.sidebar.info("💻 **IT & Technology** Sector")
        elif stock_name in auto_stocks:
            st.sidebar.info("🚗 **Auto & Manufacturing** Sector")
        elif stock_name in pharma_stocks:
            st.sidebar.info("💊 **Pharma & Healthcare** Sector")
        elif stock_name in fmcg_stocks:
            st.sidebar.info("🛒 **Consumer Goods & FMCG** Sector")
        elif stock_name in energy_stocks:
            st.sidebar.info("⚡ **Energy & Oil** Sector")
        elif stock_name in metals_stocks:
            st.sidebar.info("🏭 **Metals & Mining** Sector")
        elif stock_name in realty_stocks:
            st.sidebar.info("🏢 **Real Estate & Construction** Sector")
        elif stock_name in cement_stocks:
            st.sidebar.info("🏗️ **Cement & Construction** Sector")
        elif stock_name in telecom_stocks:
            st.sidebar.info("📡 **Telecom & Media** Sector")
        elif stock_name in digital_stocks:
            st.sidebar.info("🌐 **E-commerce & Digital** Sector")
        elif stock_name in insurance_stocks:
            st.sidebar.info("🛡️ **Insurance** Sector")
        else:
            st.sidebar.info("📈 **Indian Stock** - Other Sector")
    else:
        st.sidebar.success(f"🇺🇸 US Stock - Currency: {currency_symbol} (Dollars)")
        st.sidebar.info("💡 US stocks use Dollar ($) currency with standard formatting.")

# Analysis options
st.sidebar.header("Analysis Options")
include_macro = st.sidebar.checkbox("Include Macroeconomic Analysis", value=True, key="main_include_macro")
show_advanced = st.sidebar.checkbox("Show Advanced Features", value=True, key="main_show_advanced")

if st.sidebar.button("Analyze") or symbol:
    try:
        # Data ingestion
        df = fetch_yfinance(symbol, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        
        # Add comprehensive features
        if include_macro:
            df = get_comprehensive_features(df, include_macro=True)
        else:
            df = add_technical_indicators(df)
            
        st.success(f"Loaded {len(df)} records for {symbol}")

        # Get currency symbol for this stock
        currency_symbol = get_currency_symbol(symbol)
        is_indian = is_indian_stock(symbol)

        # Create tabs for different analyses
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
            "📊 Stock Analysis", 
            "🔮 Future Forecasting",
            "🤖 Enhanced Training",
            "📈 Technical Indicators", 
            "📰 News Sentiment", 
            "💰 Insider Trading",
            "🌍 Macro Analysis",
            "⚙️ Settings"
        ])

        with tab1:
            # Price chart
            st.subheader(f"Price Chart for {symbol}")
            st.line_chart(df.set_index('Date')['Close'])

            # Show latest technical indicators
            st.subheader("Latest Technical Indicators")
            latest = df.iloc[-1]
            cols = st.columns(3)
            with cols[0]:
                st.metric("RSI (14)", f"{latest['RSI_14']:.2f}")
                st.metric("MACD", f"{latest['MACD']:.2f}")
                st.metric("MACD Signal", f"{latest['MACD_Signal']:.2f}")
            with cols[1]:
                st.metric("Bollinger Upper", format_currency(latest['BB_Upper'], currency_symbol))
                st.metric("Bollinger Lower", format_currency(latest['BB_Lower'], currency_symbol))
                st.metric("Stochastic K", f"{latest['Stoch_K']:.2f}")
            with cols[2]:
                st.metric("ATR (14)", format_currency(latest['ATR_14'], currency_symbol))
                st.metric("Williams %R", f"{latest['Williams_R']:.2f}")
                st.metric("Volatility", f"{latest['Volatility']:.4f}")

            # Current price and market info
            st.subheader("📈 Current Market Information")
            current_price = latest['Close']
            price_change = df['Close'].iloc[-1] - df['Close'].iloc[-2] if len(df) > 1 else 0
            price_change_pct = (price_change / df['Close'].iloc[-2]) * 100 if len(df) > 1 else 0
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric(
                    "Current Price",
                    format_currency(current_price, currency_symbol),
                    f"{price_change_pct:+.2f}%"
                )
            with col2:
                st.metric(
                    "Day High",
                    format_currency(latest['High'], currency_symbol)
                )
            with col3:
                st.metric(
                    "Day Low",
                    format_currency(latest['Low'], currency_symbol)
                )
            with col4:
                st.metric(
                    "Volume",
                    f"{latest['Volume']:,.0f}"
                )

        with tab2:
            st.header("🔮 Future Price Forecasting")
            st.markdown("Forecast stock prices up to 2 years into the future using advanced ML models and macroeconomic indicators.")
            
            col1, col2, col3 = st.columns([2, 2, 1])
            
            with col1:
                forecast_symbol = st.selectbox(
                    "Select Stock for Forecasting",
                    options=[symbol],
                    index=0,
                    key="forecast_symbol"
                )
            
            with col2:
                forecast_horizon = st.selectbox(
                    "Forecast Horizon",
                    options=[
                        ("30 days", 30),
                        ("3 months", 90),
                        ("6 months", 180),
                        ("1 year", 365),
                        ("2 years", 504)
                    ],
                    format_func=lambda x: x[0],
                    index=0,  # Default to 30 days for more realistic results
                    key="forecast_horizon"
                )
            forecast_days = forecast_horizon[1]
            
            with col3:
                include_macro_forecast = st.checkbox(
                    "Include Macro Features",
                    value=True,
                    help="Include macroeconomic indicators in forecasting",
                    key="forecast_include_macro"
                )
                
                include_news_sentiment = st.checkbox(
                    "📰 Include News Sentiment",
                    value=False,
                    help="Analyze news sentiment and influence forecast based on sentiment score",
                    key="forecast_include_news"
                )
            
            # Information about forecasting
            st.info("""
            **🔮 Forecasting Information:**
            • Uses advanced algorithmic simulation for realistic price movements
            • Includes technical indicators, market cycles, and sentiment analysis
            • Generates consistent results for reliable analysis
            """.strip())
            
            # Forecast button
            if st.button("🚀 Generate Forecast", type="primary", key="generate_forecast"):
                with st.spinner("🔮 Generating future forecast..."):
                    try:
                        # Initialize forecaster
                        forecaster = FutureForecaster()
                        
                        # Initialize sentiment variables
                        sentiment_score = 0.0
                        sentiment_info = None
                        
                        # Analyze news sentiment if requested
                        if include_news_sentiment:
                            st.info("📰 Analyzing news sentiment...")
                            
                            try:
                                from news_sentiment import NewsSentimentAnalyzer
                                analyzer = NewsSentimentAnalyzer()
                                
                                # First try with 7 days
                                sentiment_summary = analyzer.get_sentiment_summary(
                                    symbol=forecast_symbol,
                                    days_back=7
                                )
                                
                                # If no articles found, try 30 days
                                if sentiment_summary['analysis']['total_articles'] == 0:
                                    st.warning("⚠️ No news found in 7 days, searching last 30 days...")
                                    sentiment_summary = analyzer.get_sentiment_summary(
                                        symbol=forecast_symbol,
                                        days_back=30
                                    )
                                
                                sentiment_score = sentiment_summary['sentiment_score']
                                total_articles = sentiment_summary['analysis']['total_articles']
                                sentiment_info = sentiment_summary
                                
                                if total_articles > 0:
                                    sentiment_label = sentiment_summary['sentiment_label']['label']
                                    st.success(f"✅ Found {total_articles} articles. Sentiment: {sentiment_label} (Score: {sentiment_score:.3f})")
                                else:
                                    st.warning("⚠️ No news articles found even in 30 days. Using neutral sentiment for forecast.")
                                    sentiment_score = 0.0  # Neutral sentiment when no news
                                
                            except Exception as e:
                                st.error(f"❌ Error analyzing sentiment: {str(e)}")
                                sentiment_score = 0.0
                        
                        # Generate forecast
                        forecast_df = forecaster.forecast_future(
                            symbol=forecast_symbol,
                            forecast_days=forecast_days,
                            include_macro=include_macro_forecast,
                            sentiment_score=sentiment_score if include_news_sentiment else None
                        )
                        
                        if not forecast_df.empty:
                            success_msg = f"✅ Generated {len(forecast_df)} predictions for {forecast_symbol}"
                            if include_news_sentiment and sentiment_info:
                                sentiment_label = sentiment_info['sentiment_label']['label']
                                success_msg += f" (with {sentiment_label} sentiment influence)"
                            st.success(success_msg)
                            
                            # Display sentiment analysis results if available
                            if include_news_sentiment and sentiment_info:
                                total_articles = sentiment_info['analysis']['total_articles']
                                if total_articles > 0:
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.info(f"📰 **News Articles**: {total_articles}")
                                    with col2:
                                        sentiment_label = sentiment_info['sentiment_label']['label']
                                        sentiment_emoji = sentiment_info['sentiment_label']['emoji']
                                        st.info(f"{sentiment_emoji} **Sentiment**: {sentiment_label}")
                                    with col3:
                                        st.info(f"📊 **Score**: {sentiment_score:.3f}")
                                else:
                                    st.info("📰 **No news found** - Using neutral sentiment for forecast")
                            
                            # Get historical data for comparison
                            end_date = datetime.now()
                            start_date = end_date - timedelta(days=365)
                            historical_df = fetch_yfinance(forecast_symbol, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
                            
                            # Get forecast summary
                            summary = forecaster.get_forecast_summary(forecast_df, historical_df)
                            
                            # Display summary metrics
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.metric(
                                    "Current Price",
                                    format_currency(summary.get('current_price', 0), currency_symbol)
                                )
                            
                            with col2:
                                st.metric(
                                    "Final Price",
                                    format_currency(summary.get('final_price', 0), currency_symbol),
                                    f"{summary.get('total_change_pct', 0):+.2f}%"
                                )
                            
                            with col3:
                                st.metric(
                                    "Forecast Volatility",
                                    f"{summary.get('forecast_volatility', 0):.2%}"
                                )
                            
                            with col4:
                                st.metric(
                                    "Max Drawdown",
                                    f"{summary.get('max_drawdown', 0):.2%}"
                                )
                            
                            # Create combined chart
                            st.subheader("📈 Historical vs Forecasted Prices")
                            
                            # Clean, direct approach to chart creation
                            with st.expander("🔍 Raw Data Validation", expanded=False):
                                st.write(f"**Historical Data:** {historical_df.shape[0]} records")
                                if not historical_df.empty:
                                    st.write(f"Price range: ${historical_df['Close'].min():.2f} - ${historical_df['Close'].max():.2f}")
                                
                                st.write(f"**Forecast Data:** {forecast_df.shape[0]} predictions")
                                if not forecast_df.empty:
                                    st.write(f"Price range: ${forecast_df['Predicted_Close'].min():.2f} - ${forecast_df['Predicted_Close'].max():.2f}")
                            
                            # Simple data preparation for charting
                            st.subheader("📈 Historical vs Forecasted Prices")
                            
                            # Historical data preparation
                            if not historical_df.empty and 'Close' in historical_df.columns and 'Date' in historical_df.columns:
                                hist_clean = historical_df[['Date', 'Close']].copy()
                                hist_clean['Date'] = pd.to_datetime(hist_clean['Date'])
                                hist_clean = hist_clean.dropna().sort_values('Date')
                                hist_clean['Type'] = 'Historical'
                                hist_clean = hist_clean.rename(columns={'Close': 'Price'})
                            else:
                                st.error("❌ Invalid historical data")
                                hist_clean = pd.DataFrame()
                            
                            # Forecast data preparation
                            if not forecast_df.empty and 'Predicted_Close' in forecast_df.columns and 'Date' in forecast_df.columns:
                                fore_clean = forecast_df[['Date', 'Predicted_Close']].copy()
                                fore_clean['Date'] = pd.to_datetime(fore_clean['Date'])
                                fore_clean = fore_clean.dropna().sort_values('Date')
                                fore_clean['Type'] = 'Forecast'
                                fore_clean = fore_clean.rename(columns={'Predicted_Close': 'Price'})
                            else:
                                st.error("❌ Invalid forecast data")
                                fore_clean = pd.DataFrame()
                            
                            # 3. Create chart using Plotly (matplotlib alternative)
                            if not hist_clean.empty and not fore_clean.empty:
                                try:
                                    # Try matplotlib first (if available)
                                    import matplotlib.pyplot as plt
                                    import matplotlib.dates as mdates
                                    
                                    # Create matplotlib figure
                                    fig, ax = plt.subplots(figsize=(14, 8))
                                    
                                    # Plot historical data
                                    ax.plot(hist_clean['Date'], hist_clean['Price'], 
                                           label='Historical', color='#1f77b4', linewidth=2, alpha=0.8)
                                    
                                    # Plot forecast data
                                    ax.plot(fore_clean['Date'], fore_clean['Price'], 
                                           label='Forecast', color='#ff7f0e', linewidth=2, alpha=0.8)
                                    
                                    # Formatting
                                    ax.set_xlabel('Date', fontsize=12)
                                    ax.set_ylabel('Price ($)', fontsize=12)
                                    ax.set_title(f"{forecast_symbol} Price Forecast ({forecast_horizon[0]})", fontsize=14, fontweight='bold')
                                    ax.legend(fontsize=12)
                                    ax.grid(True, alpha=0.3)
                                    
                                    # Format x-axis dates
                                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                                    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
                                    plt.xticks(rotation=45)
                                    
                                    # Adjust layout to prevent label cutoff
                                    plt.tight_layout()
                                    
                                    # Display the matplotlib chart in Streamlit
                                    st.pyplot(fig)
                                    
                                except ImportError:
                                    # Fallback to Plotly if matplotlib not available
                                    st.info("📊 Using Plotly charts (matplotlib not available)")
                                    
                                    # Create Plotly figure
                                    fig = go.Figure()
                                    
                                    # Add historical data
                                    fig.add_trace(go.Scatter(
                                        x=hist_clean['Date'],
                                        y=hist_clean['Price'],
                                        mode='lines',
                                        name='Historical',
                                        line=dict(color='#1f77b4', width=2),
                                        opacity=0.8
                                    ))
                                    
                                    # Add forecast data
                                    fig.add_trace(go.Scatter(
                                        x=fore_clean['Date'],
                                        y=fore_clean['Price'],
                                        mode='lines',
                                        name='Forecast',
                                        line=dict(color='#ff7f0e', width=2),
                                        opacity=0.8
                                    ))
                                    
                                    # Update layout
                                    fig.update_layout(
                                        title=f"{forecast_symbol} Price Forecast ({forecast_horizon[0]})",
                                        xaxis_title="Date",
                                        yaxis_title="Price ($)",
                                        hovermode='x unified',
                                        height=500,
                                        showlegend=True
                                    )
                                    
                                    # Display the Plotly chart
                                    st.plotly_chart(fig, use_container_width=True)
                                
                                # Debug: Show what we're actually plotting
                                st.write(f"**📊 Chart Data Summary:**")
                                st.write(f"Historical: {len(hist_clean)} points from {hist_clean['Date'].min().date()} to {hist_clean['Date'].max().date()}")
                                st.write(f"Forecast: {len(fore_clean)} points from {fore_clean['Date'].min().date()} to {fore_clean['Date'].max().date()}")
                                
                                # Calculate price statistics
                                all_prices = pd.concat([hist_clean['Price'], fore_clean['Price']])
                                st.write(f"Price range: ${all_prices.min():.2f} to ${all_prices.max():.2f}")
                                
                                # Show sample data to verify it's not linear
                                with st.expander("🔍 View Sample Data", expanded=False):
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.write("**Historical Sample:**")
                                        st.dataframe(hist_clean.head(10))
                                    with col2:
                                        st.write("**Forecast Sample:**")
                                        st.dataframe(fore_clean.head(10))
                                
                            else:
                                st.error("❌ No valid data for charting")
                            
                            # Forecast details
                            col1, col2 = st.columns([2, 1])
                            
                            with col1:
                                st.subheader("📊 Forecast Details")
                                
                                # Price trend analysis
                                trend_slope = summary.get('trend_slope', 0)
                                if trend_slope > 0:
                                    trend_direction = "📈 Bullish"
                                    trend_color = "green"
                                else:
                                    trend_direction = "📉 Bearish"
                                    trend_color = "red"
                                
                                st.markdown(f"""
                                **Trend Analysis:**
                                - Direction: {trend_direction}
                                - Slope: {trend_slope:.4f}
                                - Confidence: {summary.get('confidence_level', 'Medium')}
                                """)
                                
                                # Key milestones
                                st.subheader("🎯 Key Milestones")
                                
                                milestones = [30, 90, 180, 365]
                                if forecast_days > 365:
                                    milestones.append(504)
                                
                                milestone_data = []
                                for milestone in milestones:
                                    if milestone <= len(forecast_df):
                                        milestone_price = forecast_df.iloc[milestone-1]['Predicted_Close']
                                        milestone_date = forecast_df.iloc[milestone-1]['Date']
                                        milestone_data.append({
                                            'Days': milestone,
                                            'Date': milestone_date.strftime('%Y-%m-%d'),
                                            'Price': format_currency(milestone_price, currency_symbol)
                                        })
                                
                                if milestone_data:
                                    milestone_df = pd.DataFrame(milestone_data)
                                    st.dataframe(milestone_df, use_container_width=True)
                            
                            with col2:
                                st.subheader("📋 Forecast Summary")
                                
                                st.markdown(f"""
                                **Forecast Period:**
                                - Start: {forecast_df['Date'].iloc[0].strftime('%Y-%m-%d')}
                                - End: {forecast_df['Date'].iloc[-1].strftime('%Y-%m-%d')}
                                - Duration: {len(forecast_df)} business days
                                
                                **Price Statistics:**
                                - Min: {format_currency(float(forecast_df['Predicted_Close'].min()), currency_symbol)}
                                - Max: {format_currency(float(forecast_df['Predicted_Close'].max()), currency_symbol)}
                                - Mean: {format_currency(float(forecast_df['Predicted_Close'].mean()), currency_symbol)}
                                - Std: {format_currency(float(forecast_df['Predicted_Close'].std()), currency_symbol)}
                                """)
                                
                            # Download forecast data
                            st.subheader("💾 Download Forecast Data")
                            
                            csv_data = forecast_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Forecast CSV",
                                data=csv_data,
                                file_name=f"{forecast_symbol}_forecast_{forecast_days}days.csv",
                                mime="text/csv"
                            )
                            
                        else:
                            st.error("❌ Failed to generate forecast. Please try again.")
                            
                    except Exception as e:
                        st.error(f"❌ Error generating forecast: {str(e)}")
                        st.exception(e)

        with tab3:
            st.header("🤖 Enhanced Model Training")
            st.markdown("Train advanced LSTM models with comprehensive features, hyperparameter optimization, and backtesting.")
            
            # Training configuration
            col1, col2, col3 = st.columns([2, 2, 1])
            
            with col1:
                training_symbol = st.selectbox(
                    "Select Stock for Training",
                    options=[symbol],
                    index=0,
                    key="training_symbol"
                )
                
                training_start_date = st.date_input(
                    "Training Start Date",
                    value=(datetime.now() - timedelta(days=730)).date(),
                    key="training_start"
                )
                
            with col2:
                training_end_date = st.date_input(
                    "Training End Date",
                    value=datetime.now().date(),
                    key="training_end"
                )
                
                initial_capital = st.number_input(
                    "Initial Capital ($)",
                    value=100000,
                    min_value=1000,
                    step=10000,
                    key="initial_capital"
                )
                
            with col3:
                run_optimization = st.checkbox(
                    "Run Hyperparameter Optimization",
                    value=True,
                    help="Use Bayesian optimization to find best parameters",
                    key="training_run_optimization"
                )
                
                optimization_trials = st.number_input(
                    "Optimization Trials",
                    value=10,
                    min_value=5,
                    max_value=50,
                    help="Number of optimization trials"
                )
            
            # Training options
            st.subheader("🎯 Training Options")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                include_advanced_features = st.checkbox(
                    "Advanced Features",
                    value=True,
                    help="Include sentiment, microstructure, and advanced technical indicators",
                    key="training_advanced_features"
                )
                
                include_macro_features = st.checkbox(
                    "Macro Features",
                    value=True,
                    help="Include macroeconomic indicators",
                    key="training_macro_features"
                )
                
            with col2:
                run_backtest = st.checkbox(
                    "Run Backtesting",
                    value=True,
                    help="Perform walk-forward backtesting after training",
                    key="training_run_backtest"
                )
                
                save_model = st.checkbox(
                    "Save Model",
                    value=True,
                    help="Save the trained model for future use",
                    key="training_save_model"
                )
                
            with col3:
                show_training_plots = st.checkbox(
                    "Show Training Plots",
                    value=True,
                    help="Display training progress and metrics",
                    key="training_show_plots"
                )
                
                export_results = st.checkbox(
                    "Export Results",
                    value=True,
                    help="Export training results and backtesting data",
                    key="training_export_results"
                )
            
            # Start training button
            if st.button("🚀 Start Enhanced Training", type="primary", key="start_training"):
                with st.spinner("🤖 Training enhanced model..."):
                    try:
                        # Initialize enhanced training system
                        if not ENHANCED_TRAINING_AVAILABLE:
                            st.error("❌ Enhanced training system not available. Please ensure all dependencies are installed.")
                        else:
                            training_system = EnhancedTrainingSystem(
                                symbol=training_symbol,
                                start_date=training_start_date.strftime('%Y-%m-%d'),
                                end_date=training_end_date.strftime('%Y-%m-%d'),
                                initial_capital=initial_capital
                            )
                            
                            # Create progress bar
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            # Step 1: Fetch and prepare data
                            status_text.text("📊 Fetching and preparing data...")
                            df = training_system.fetch_and_prepare_data()
                            progress_bar.progress(20)
                            
                            # Step 2: Create enhanced features
                            status_text.text("🔧 Creating enhanced features...")
                            featured_df = training_system.create_enhanced_features(df)
                            progress_bar.progress(40)
                            
                            # Step 3: Optimize hyperparameters (if enabled)
                            if run_optimization:
                                status_text.text("🎯 Running hyperparameter optimization...")
                                optimization_results = training_system.optimize_hyperparameters(featured_df)
                                progress_bar.progress(60)
                                
                                # Display optimization results
                                st.subheader("📊 Optimization Results")
                                if optimization_results:
                                    best_params = optimization_results.get('best_params', {})
                                    best_value = optimization_results.get('best_value', 0)
                                    
                                    st.success(f"✅ Best MAPE: {best_value:.2f}%")
                                    st.write("**Best Parameters:**")
                                    for param, value in best_params.items():
                                        st.write(f"- {param}: {value}")
                            
                            # Step 4: Train model
                            status_text.text("🤖 Training enhanced LSTM model...")
                            model, scaler = training_system.train_enhanced_model(featured_df)
                            progress_bar.progress(80)
                            
                            # Step 5: Evaluate performance
                            status_text.text("📊 Evaluating system performance...")
                            performance = training_system.evaluate_system_performance(featured_df)
                            progress_bar.progress(100)
                            
                            # Display results
                            st.success("✅ Enhanced training completed successfully!")
                            
                            # Training metrics
                            st.subheader("📈 Training Results")
                            
                            if performance and 'backtest_results' in performance:
                                backtest_results = performance['backtest_results']
                                
                                col1, col2, col3, col4 = st.columns(4)
                                
                                with col1:
                                    st.metric(
                                        "MAPE",
                                        f"{backtest_results.get('mape', 0):.2f}%"
                                    )
                                
                                with col2:
                                    st.metric(
                                        "Directional Accuracy",
                                        f"{backtest_results.get('directional_accuracy', 0):.2f}%"
                                    )
                                
                                with col3:
                                    st.metric(
                                        "Cumulative Return",
                                        f"{backtest_results.get('cumulative_return', 0):.2f}%"
                                    )
                                
                                with col4:
                                    st.metric(
                                        "Sharpe Ratio",
                                        f"{backtest_results.get('sharpe_ratio', 0):.2f}"
                                    )
                            
                            # Model information
                            st.subheader("🤖 Model Information")
                            
                            if performance and 'model_info' in performance:
                                model_info = performance['model_info']
                                
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.write(f"**Model Type:** {model_info.get('model_type', 'Enhanced LSTM')}")
                                    st.write(f"**Features Used:** {model_info.get('features_used', 0)}")
                                    st.write(f"**Data Points:** {model_info.get('data_points', 0)}")
                                
                                with col2:
                                    st.write(f"**Training Period:** {training_start_date} to {training_end_date}")
                                    st.write(f"**Symbol:** {training_symbol}")
                                    st.write(f"**Initial Capital:** ${initial_capital:,}")
                            
                            # Save system state
                            if save_model:
                                training_system.save_system_state()
                                st.success("💾 Model and system state saved successfully!")
                            
                            # Export results
                            if export_results and performance:
                                st.subheader("💾 Export Results")
                                
                                # Create results summary
                                results_summary = {
                                    'symbol': training_symbol,
                                    'training_period': f"{training_start_date} to {training_end_date}",
                                    'performance_metrics': performance.get('backtest_results', {}),
                                    'model_info': performance.get('model_info', {}),
                                    'training_date': datetime.now().isoformat()
                                }
                                
                                # Export as JSON
                                import json
                                json_data = json.dumps(results_summary, indent=2)
                                st.download_button(
                                    label="📥 Download Results JSON",
                                    data=json_data,
                                    file_name=f"{training_symbol}_enhanced_training_results.json",
                                    mime="application/json"
                                )
                        
                    except Exception as e:
                        st.error(f"❌ Error during enhanced training: {str(e)}")
                        st.exception(e)

        with tab4:
            st.subheader("Trading Signals")
            
            # Technical signals
            st.write("**Technical Analysis Signals**")
            signals = get_trading_signals(df)
            for indicator, signal_data in signals.items():
                if indicator != 'Overall':
                    emoji = "🟢" if signal_data['signal'] == 'BUY' else "🔴" if signal_data['signal'] == 'SELL' else "⚪"
                    st.write(f"{emoji} **{indicator}**: {signal_data['signal']} (Confidence: {signal_data['confidence']:.0%})")
            
            if 'Overall' in signals:
                st.markdown(f"### 🎯 **Overall Technical Signal:** {signals['Overall']['signal']} (Confidence: {signals['Overall']['confidence']:.0%})")

            # Macro signals
            if include_macro:
                st.write("**Macroeconomic Signals**")
                macro_signals = get_macro_trading_signals(df)
                if macro_signals:
                    for indicator, signal_data in macro_signals.items():
                        emoji = "🟢" if signal_data['signal'] in ['GROWTH', 'CYCLICAL', 'GROWTH_TECH'] else "🔴" if signal_data['signal'] in ['DEFENSIVE', 'INFLATION_HEDGE'] else "⚪"
                        st.write(f"{emoji} **{indicator}**: {signal_data['signal']} (Confidence: {signal_data['confidence']:.0%})")
                        st.write(f"   *{signal_data['reason']}*")
                else:
                    st.info("No macro signals available")

        with tab5:
            st.header("📰 News Sentiment Analysis")
            st.markdown("Analyze news sentiment for the selected stock using advanced NLP and multiple data sources.")
            
            # Import news sentiment analyzer
            try:
                from news_sentiment import NewsSentimentAnalyzer
                
                # Configuration options
                col1, col2, col3 = st.columns([2, 2, 1])
                
                with col1:
                    sentiment_symbol = st.selectbox(
                        "Select Stock for Sentiment Analysis",
                        options=[symbol],
                        index=0,
                        key="sentiment_symbol"
                    )
                    
                    # Try to get company name mapping
                    company_names = {
                        'AAPL': 'Apple Inc',
                        'MSFT': 'Microsoft Corporation',
                        'GOOGL': 'Alphabet Inc',
                        'AMZN': 'Amazon.com Inc',
                        'TSLA': 'Tesla Inc',
                        'NVDA': 'NVIDIA Corporation',
                        'META': 'Meta Platforms Inc',
                        'NFLX': 'Netflix Inc',
                        'ICICIBANK.NS': 'ICICI Bank',
                        'RELIANCE.NS': 'Reliance Industries',
                        'TCS.NS': 'Tata Consultancy Services',
                        'INFY.NS': 'Infosys Limited'
                    }
                    
                    company_name = company_names.get(sentiment_symbol.upper(), None)
                    
                with col2:
                    days_back = st.selectbox(
                        "Analysis Period",
                        options=[3, 7, 14, 30],
                        index=1,
                        format_func=lambda x: f"{x} days",
                        key="sentiment_days"
                    )
                    
                    news_sources = st.multiselect(
                        "News Sources",
                        ["NewsAPI", "Alpha Vantage", "Both"],
                        default=["Both"],
                        key="news_sources"
                    )
                
                with col3:
                    auto_refresh = st.checkbox(
                        "Auto Refresh",
                        value=False,
                        help="Automatically refresh sentiment data",
                        key="auto_refresh"
                    )
                
                # Analyze sentiment button
                if st.button("🔍 Analyze News Sentiment", type="primary", key="analyze_sentiment"):
                    with st.spinner("📰 Fetching and analyzing news sentiment..."):
                        try:
                            # Initialize sentiment analyzer
                            analyzer = NewsSentimentAnalyzer()
                            
                            # Show search information
                            company_name_display, search_terms = analyzer.get_company_info(sentiment_symbol)
                            
                            st.info(f"🔍 **Searching for:** {company_name_display}")
                            st.info(f"📋 **Search terms:** {', '.join(search_terms[:3])}")
                            
                            # Get sentiment summary
                            sentiment_summary = analyzer.get_sentiment_summary(
                                symbol=sentiment_symbol,
                                company_name=company_name,
                                days_back=days_back
                            )
                            
                            if sentiment_summary:
                                # Show search statistics
                                search_info = sentiment_summary.get('search_info', {})
                                if search_info:
                                    st.success(f"✅ **Search Results:** Found {search_info.get('newsapi_articles', 0)} NewsAPI articles + {search_info.get('alpha_vantage_articles', 0)} Alpha Vantage articles = {search_info.get('total_unique_articles', 0)} unique articles")
                                
                                # Display overall sentiment
                                st.subheader("📊 Overall Sentiment")
                                
                                sentiment_label = sentiment_summary['sentiment_label']
                                sentiment_score = sentiment_summary['sentiment_score']
                                
                                col1, col2, col3, col4 = st.columns(4)
                                
                                with col1:
                                    st.metric(
                                        "Sentiment Score",
                                        f"{sentiment_score:.3f}",
                                        help="Range: -1 (Very Negative) to +1 (Very Positive)"
                                    )
                                
                                with col2:
                                    st.markdown(f"""
                                    <div style="text-align: center; padding: 10px; border-radius: 10px; background-color: {sentiment_label['color']}20; border: 2px solid {sentiment_label['color']};">
                                        <h3 style="color: {sentiment_label['color']}; margin: 0;">
                                            {sentiment_label['emoji']} {sentiment_label['label']}
                                        </h3>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                with col3:
                                    total_articles = sentiment_summary['analysis']['total_articles']
                                    st.metric("Articles Analyzed", total_articles)
                                
                                with col4:
                                    analysis_date = sentiment_summary['analysis_date']
                                    formatted_date = pd.to_datetime(analysis_date).strftime('%Y-%m-%d %H:%M')
                                    st.metric("Last Updated", formatted_date)
                                
                                # Sentiment distribution
                                if total_articles > 0:
                                    st.subheader("📈 Sentiment Distribution")
                                    
                                    dist = sentiment_summary['analysis']['sentiment_distribution']
                                    
                                    col1, col2 = st.columns([2, 1])
                                    
                                    with col1:
                                        # Create pie chart
                                        import plotly.express as px
                                        
                                        sentiment_data = pd.DataFrame({
                                            'Sentiment': ['Positive', 'Neutral', 'Negative'],
                                            'Count': [dist['positive'], dist['neutral'], dist['negative']],
                                            'Percentage': [dist['positive_pct'], dist['neutral_pct'], dist['negative_pct']]
                                        })
                                        
                                        fig_pie = px.pie(
                                            sentiment_data, 
                                            values='Count', 
                                            names='Sentiment',
                                            color_discrete_map={
                                                'Positive': '#4CAF50',
                                                'Neutral': '#FFC107', 
                                                'Negative': '#F44336'
                                            },
                                            title="News Sentiment Distribution"
                                        )
                                        
                                        st.plotly_chart(fig_pie, use_container_width=True)
                                    
                                    with col2:
                                        st.write("**Breakdown:**")
                                        st.write(f"🟢 Positive: {dist['positive']} ({dist['positive_pct']:.1f}%)")
                                        st.write(f"⚪ Neutral: {dist['neutral']} ({dist['neutral_pct']:.1f}%)")
                                        st.write(f"🔴 Negative: {dist['negative']} ({dist['negative_pct']:.1f}%)")
                                        
                                        # Sentiment strength indicator
                                        if dist['positive_pct'] > 60:
                                            strength = "🚀 Very Bullish"
                                        elif dist['positive_pct'] > 40:
                                            strength = "📈 Bullish"
                                        elif dist['negative_pct'] > 60:
                                            strength = "📉 Very Bearish"
                                        elif dist['negative_pct'] > 40:
                                            strength = "🔻 Bearish"
                                        else:
                                            strength = "➖ Mixed"
                                        
                                        st.write(f"**Market Sentiment:** {strength}")
                                
                                # Recent news articles
                                st.subheader("📰 Recent News Articles")
                                
                                articles = sentiment_summary['analysis']['articles_with_sentiment']
                                
                                if articles:
                                    # Sort by sentiment score and date
                                    articles_sorted = sorted(articles, key=lambda x: (x.get('sentiment_score', 0), x.get('published_date', datetime.now().date())), reverse=True)
                                    
                                    # Display top articles
                                    for i, article in enumerate(articles_sorted[:10], 1):
                                        sentiment_score = article.get('sentiment_score', 0)
                                        
                                        # Sentiment color and emoji
                                        if sentiment_score >= 0.1:
                                            sentiment_color = '#4CAF50'
                                            sentiment_emoji = '📈'
                                        elif sentiment_score <= -0.1:
                                            sentiment_color = '#F44336'
                                            sentiment_emoji = '📉'
                                        else:
                                            sentiment_color = '#FFC107'
                                            sentiment_emoji = '➖'
                                        
                                        with st.expander(f"{sentiment_emoji} {article.get('title', 'No title')[:80]}...", expanded=False):
                                            col1, col2 = st.columns([3, 1])
                                            
                                            with col1:
                                                st.write(f"**Description:** {article.get('description', 'No description available')}")
                                                st.write(f"**Source:** {article.get('source', 'Unknown')}")
                                                st.write(f"**Published:** {article.get('published_date', 'Unknown')}")
                                                
                                                if article.get('url'):
                                                    st.markdown(f"[🔗 Read Full Article]({article['url']})")
                                            
                                            with col2:
                                                st.metric(
                                                    "Sentiment Score", 
                                                    f"{sentiment_score:.3f}",
                                                    help="Article sentiment contribution"
                                                )
                                                
                                                st.markdown(f"""
                                                <div style="text-align: center; padding: 5px; border-radius: 5px; background-color: {sentiment_color}20; border: 1px solid {sentiment_color};">
                                                    <small style="color: {sentiment_color};">{sentiment_emoji}</small>
                                                </div>
                                                """, unsafe_allow_html=True)
                                else:
                                    st.warning("⚠️ No recent news articles found for this stock.")
                                    
                                    # Provide explanations for why no news was found
                                    with st.expander("🤔 Why might there be no news articles?", expanded=True):
                                        st.markdown(f"""
                                        **Possible reasons for {sentiment_symbol}:**
                                        
                                        1. **📈 Market Coverage**: Smaller or mid-cap stocks may have limited news coverage
                                        2. **🌐 Regional Limitations**: International stocks (especially Indian stocks) may have less English news coverage
                                        3. **⏰ Time Period**: Try increasing the analysis period (7-30 days) for better coverage
                                        4. **📰 News Sources**: Some stocks are covered more by financial press vs general news
                                        5. **🔍 Search Terms**: The company might be referenced by different names in news articles
                                        
                                        **Suggestions:**
                                        - Try a longer time period (14-30 days)
                                        - Check if this is a subsidiary or part of a larger company
                                        - Look for news under the full company name: **{company_name_display}**
                                        - Some stocks have seasonal news patterns
                                        """)
                                        
                                        # Show what was actually searched for
                                        st.info(f"**Searched for:** {company_name_display}")
                                        st.info(f"**Search terms used:** {', '.join(search_terms)}")
                                        
                                        # Suggest alternatives
                                        if '.NS' in sentiment_symbol:
                                            st.info("💡 **Tip for Indian stocks:** News coverage may be limited in English. Consider checking local financial news sources.")
                                        elif sentiment_symbol in ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN']:
                                            st.info("💡 **Tip for major US stocks:** These should have plenty of news. Try increasing the time period.")
                                        else:
                                            st.info("💡 **Tip:** Try searching for recent earnings reports, press releases, or regulatory filings for this company.")
                                
                                # Trading implications
                                st.subheader("💡 Trading Implications")
                                
                                if sentiment_score >= 0.3:
                                    implication = "🚀 **Strong Positive Sentiment**: Consider bullish positions. High positive sentiment may indicate upcoming price appreciation."
                                    signal_color = "#4CAF50"
                                elif sentiment_score >= 0.1:
                                    implication = "📈 **Moderate Positive Sentiment**: Cautiously optimistic outlook. Consider accumulating on dips."
                                    signal_color = "#8BC34A"
                                elif sentiment_score <= -0.3:
                                    implication = "🔻 **Strong Negative Sentiment**: Consider bearish positions or risk management. Negative sentiment may lead to price decline."
                                    signal_color = "#F44336"
                                elif sentiment_score <= -0.1:
                                    implication = "📉 **Moderate Negative Sentiment**: Exercise caution. Consider reducing exposure or hedging."
                                    signal_color = "#FF5722"
                                else:
                                    implication = "➖ **Neutral Sentiment**: Mixed signals. Wait for clearer directional bias before making significant moves."
                                    signal_color = "#FFC107"
                                
                                st.markdown(f"""
                                <div style="padding: 15px; border-radius: 10px; background-color: {signal_color}20; border-left: 5px solid {signal_color};">
                                    {implication}
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Download sentiment data
                                st.subheader("💾 Export Sentiment Data")
                                
                                # Create downloadable data
                                export_data = {
                                    'symbol': sentiment_symbol,
                                    'analysis_date': sentiment_summary['analysis_date'],
                                    'sentiment_score': sentiment_score,
                                    'sentiment_label': sentiment_label['label'],
                                    'total_articles': total_articles,
                                    'sentiment_distribution': dist,
                                    'articles': articles
                                }
                                
                                import json
                                json_data = json.dumps(export_data, indent=2, default=str)
                                
                                st.download_button(
                                    label="📥 Download Sentiment Analysis",
                                    data=json_data,
                                    file_name=f"{sentiment_symbol}_sentiment_analysis.json",
                                    mime="application/json"
                                )
                            
                            else:
                                st.error("❌ No sentiment data available.")
                                
                                with st.expander("🔧 Troubleshooting Guide", expanded=True):
                                    st.markdown(f"""
                                    **For {sentiment_symbol}, try these solutions:**
                                    
                                    **1. Extend Time Period**
                                    - Current: {days_back} days
                                    - Try: 14-30 days for better coverage
                                    
                                    **2. Check Stock Symbol**
                                    - Verify the ticker symbol is correct
                                    - For Indian stocks, ensure it ends with .NS or .BO
                                    - Example: TRENT should be TRENT.NS
                                    
                                    **3. API Status**
                                    - Check internet connection
                                    - NewsAPI and Alpha Vantage may have rate limits
                                    - Try again in a few minutes
                                    
                                    **4. Alternative Stocks to Test**
                                    - **US:** AAPL, MSFT, GOOGL, TSLA (high news coverage)
                                    - **Indian:** RELIANCE.NS, TCS.NS, INFY.NS (major stocks)
                                    
                                    **5. Company Information**
                                    - Searched for: **{company_name_display if 'company_name_display' in locals() else 'Unknown'}**
                                    - This company might have limited news coverage
                                    """)
                                    
                                    if '.NS' in sentiment_symbol:
                                        st.info("🇮🇳 **Indian Stock Note**: English news coverage for Indian stocks varies significantly. Major companies like Reliance, TCS, Infosys have better coverage than smaller companies.")
                                    
                                    st.warning("💡 **Developer Note**: This system searches multiple news sources. If no articles are found, the company may genuinely have limited recent news coverage.")
                        
                        except Exception as e:
                            st.error(f"❌ Error analyzing sentiment: {str(e)}")
                            st.write("Please check your internet connection and API keys.")
                
                # Information about sentiment analysis
                with st.expander("ℹ️ About News Sentiment Analysis", expanded=False):
                    st.markdown("""
                    **How it works:**
                    
                    1. **Data Sources**: Fetches news from NewsAPI and Alpha Vantage
                    2. **NLP Analysis**: Uses TextBlob and keyword-based sentiment analysis
                    3. **Financial Context**: Applies financial-specific sentiment scoring
                    4. **Aggregation**: Combines multiple sentiment signals for robust analysis
                    
                    **Sentiment Score Interpretation:**
                    - **+0.3 to +1.0**: Very Positive (Strong bullish sentiment)
                    - **+0.1 to +0.3**: Positive (Moderate bullish sentiment)
                    - **-0.1 to +0.1**: Neutral (Mixed or no clear sentiment)
                    - **-0.3 to -0.1**: Negative (Moderate bearish sentiment)
                    - **-1.0 to -0.3**: Very Negative (Strong bearish sentiment)
                    
                    **Trading Applications:**
                    - Sentiment can be a leading indicator of price movements
                    - Combine with technical analysis for better signals
                    - Extreme sentiment levels may indicate contrarian opportunities
                    - Monitor sentiment changes over time for trend confirmation
                    """)
            
            except ImportError:
                st.error("❌ News sentiment module not available. Please ensure all dependencies are installed.")
            except Exception as e:
                st.error(f"❌ Error loading sentiment analysis: {str(e)}")

        with tab6:
            st.header("💰 Insider Trading Analysis")
            st.markdown("Analyze insider trading patterns and corporate actions for investment insights.")
            
            # Placeholder for insider trading functionality
            st.info("🚧 Insider Trading analysis is under development. This will include:")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("""
                **Features Coming Soon:**
                - SEC Form 4 filings analysis
                - Insider buying/selling patterns
                - Corporate executive transactions
                - Institutional ownership changes
                - Share buyback programs
                """)
            
            with col2:
                st.write("""
                **Analysis Capabilities:**
                - Insider sentiment scoring
                - Transaction volume analysis
                - Price correlation with insider activity
                - Unusual insider activity alerts
                - Historical insider performance
                """)
            
            # Mock data for demonstration
            st.subheader("📊 Sample Insider Activity")
            
            # Create sample insider data
            insider_data = pd.DataFrame({
                'Date': pd.date_range(start='2024-01-01', periods=10, freq='W'),
                'Insider': ['CEO John Doe', 'CFO Jane Smith', 'Director Bob Wilson', 'CEO John Doe', 'VP Sales Mike Brown',
                           'Director Alice Johnson', 'CFO Jane Smith', 'CTO David Lee', 'CEO John Doe', 'VP Ops Sarah Davis'],
                'Action': ['Buy', 'Sell', 'Buy', 'Buy', 'Sell', 'Buy', 'Sell', 'Buy', 'Buy', 'Sell'],
                'Shares': [10000, 5000, 15000, 8000, 12000, 6000, 3000, 20000, 7500, 9000],
                'Price': [150.25, 152.80, 148.90, 155.30, 157.60, 153.40, 159.20, 151.70, 156.80, 158.90],
                'Value': [1502500, 764000, 2233500, 1242400, 1891200, 920400, 477600, 3034000, 1176000, 1430100]
            })
            
            # Display the data
            st.dataframe(insider_data, use_container_width=True)
            
            # Simple analysis
            buy_actions = len(insider_data[insider_data['Action'] == 'Buy'])
            sell_actions = len(insider_data[insider_data['Action'] == 'Sell'])
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Buy Transactions", buy_actions)
            
            with col2:
                st.metric("Sell Transactions", sell_actions)
            
            with col3:
                sentiment = "Bullish" if buy_actions > sell_actions else "Bearish" if sell_actions > buy_actions else "Neutral"
                st.metric("Insider Sentiment", sentiment)

        with tab7:
            st.header("🌍 Macroeconomic Analysis")
            st.markdown("Analyze macroeconomic indicators and their relationship with stock prices.")
            
            # Macro superimposition section
            st.subheader("📊 Macro Indicator Overlay")
            st.markdown("Superimpose macroeconomic indicators on stock price charts to visualize correlations and trends.")
            
            # Macro indicator selection
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Available macro indicators
                macro_indicators = {
                    "Interest Rates": ["Federal Funds Rate", "10-Year Treasury Yield", "2-Year Treasury Yield", "Yield Curve Spread"],
                    "Inflation": ["Consumer Price Index", "Core CPI", "Producer Price Index", "Inflation Expectations"],
                    "Economic Growth": ["GDP Growth", "Industrial Production", "Retail Sales", "Manufacturing PMI"],
                    "Employment": ["Unemployment Rate", "Non-Farm Payrolls", "Job Openings", "Wage Growth"],
                    "Commodities": ["Oil Prices (WTI)", "Gold Prices", "Copper Prices", "Natural Gas"],
                    "Currencies": ["US Dollar Index", "EUR/USD", "USD/JPY", "USD/CNY"],
                    "Market Sentiment": ["VIX Volatility", "Consumer Confidence", "Business Confidence", "Housing Market Index"],
                    "Money Supply": ["M2 Money Supply", "M1 Money Supply", "Bank Reserves", "Credit Growth"]
                }
                
                selected_category = st.selectbox(
                    "Select Macro Category",
                    list(macro_indicators.keys()),
                    key="macro_category"
                )
                
                selected_indicators = st.multiselect(
                    "Select Indicators to Overlay",
                    macro_indicators[selected_category],
                    default=macro_indicators[selected_category][:2],
                    key="macro_indicators"
                )
            
            with col2:
                # Chart options
                st.subheader("📈 Chart Options")
                
                chart_type = st.selectbox(
                    "Chart Type",
                    ["Overlay (Dual Y-axis)", "Subplot", "Correlation Heatmap"],
                    key="chart_type"
                )
                
                date_range = st.selectbox(
                    "Date Range",
                    ["1 Year", "2 Years", "5 Years", "10 Years"],
                    index=1,
                    key="date_range"
                )
                
                normalize_data = st.checkbox(
                    "Normalize Data",
                    value=True,
                    help="Normalize indicators to same scale for better comparison",
                    key="macro_normalize_data"
                )
                
                show_correlation = st.checkbox(
                    "Show Correlation Analysis",
                    value=True,
                    help="Display correlation coefficients between stock and macro indicators",
                    key="macro_show_correlation"
                )
            
            # Generate macro overlay chart
            if st.button("🚀 Generate Macro Overlay Chart", type="primary"):
                with st.spinner("📊 Fetching macro data and generating overlay chart..."):
                    try:
                        # Initialize macro indicators fetcher
                        if not MACRO_INDICATORS_AVAILABLE:
                            st.error("❌ Macro indicators module not available. Please ensure all dependencies are installed.")
                        else:
                            macro_fetcher = MacroIndicators()
                            
                            # Calculate date range
                            years_map = {"1 Year": 1, "2 Years": 2, "5 Years": 5, "10 Years": 10}
                            years = years_map[date_range]
                            start_date_macro = (datetime.now() - timedelta(days=365*years)).strftime('%Y-%m-%d')
                            end_date_macro = datetime.now().strftime('%Y-%m-%d')
                            
                            # Fetch macro data
                            macro_data = macro_fetcher.get_macro_indicators(start_date_macro, end_date_macro)
                        
                        if macro_data:
                            # Prepare stock data for the same period
                            stock_data = fetch_yfinance(symbol, start_date_macro, end_date_macro)
                            
                            if not stock_data.empty:
                                # Create the overlay chart based on selected type
                                if chart_type == "Overlay (Dual Y-axis)":
                                    fig = create_macro_overlay_chart(
                                        stock_data, macro_data, selected_indicators, 
                                        normalize_data, symbol
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                elif chart_type == "Subplot":
                                    fig = create_macro_subplot_chart(
                                        stock_data, macro_data, selected_indicators, 
                                        normalize_data, symbol
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                elif chart_type == "Correlation Heatmap":
                                    fig = create_correlation_heatmap(
                                        stock_data, macro_data, selected_indicators, symbol
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                
                                # Show correlation analysis
                                if show_correlation:
                                    st.subheader("📊 Correlation Analysis")
                                    correlation_df = calculate_macro_correlations(
                                        stock_data, macro_data, selected_indicators
                                    )
                                    st.dataframe(correlation_df, use_container_width=True)
                                    
                                    # Correlation insights
                                    st.subheader("💡 Correlation Insights")
                                    insights = generate_correlation_insights(correlation_df, symbol)
                                    for insight in insights:
                                        st.write(f"• {insight}")
                                
                                # Macro trading signals
                                st.subheader("🎯 Macro Trading Signals")
                                macro_signals = get_macro_trading_signals(df)
                                if macro_signals:
                                    for indicator, signal_data in macro_signals.items():
                                        emoji = "🟢" if signal_data['signal'] in ['GROWTH', 'CYCLICAL', 'GROWTH_TECH'] else "🔴" if signal_data['signal'] in ['DEFENSIVE', 'INFLATION_HEDGE'] else "⚪"
                                        st.write(f"{emoji} **{indicator}**: {signal_data['signal']} (Confidence: {signal_data['confidence']:.0%})")
                                        st.write(f"   *{signal_data['reason']}*")
                                else:
                                    st.info("No macro signals available")
                                
                            else:
                                st.error("❌ Failed to fetch stock data for the selected period.")
                        else:
                            st.error("❌ Failed to fetch macroeconomic data. Please check your internet connection.")
                            
                    except Exception as e:
                        st.error(f"❌ Error generating macro overlay chart: {str(e)}")
                        st.exception(e)
            
            # Macro risk factors
            if include_macro and show_advanced:
                st.subheader("Macro Risk Factors")
                macro_risk_cols = st.columns(2)
                with macro_risk_cols[0]:
                    if 'economic_stress' in latest:
                        stress = latest['economic_stress']
                        stress_level = "🔴 High Risk" if stress > 0.6 else "🟡 Medium Risk" if stress > 0.3 else "🟢 Low Risk"
                        st.metric("Economic Stress Risk", stress_level, f"{stress:.3f}")
                    
                    if 'inflation_adjusted_volatility' in latest:
                        adj_vol = latest['inflation_adjusted_volatility']
                        vol_risk = "🔴 High" if adj_vol > 0.05 else "🟡 Medium" if adj_vol > 0.02 else "🟢 Low"
                        st.metric("Inflation Adjusted Vol Risk", vol_risk, f"{adj_vol:.4f}")
                
                with macro_risk_cols[1]:
                    if 'rate_adjusted_returns' in latest:
                        rate_adj_ret = latest['rate_adjusted_returns']
                        ret_signal = "🟢 Positive" if rate_adj_ret > 0 else "🔴 Negative"
                        st.metric("Rate Adjusted Returns", ret_signal, f"{rate_adj_ret:.4f}")

        with tab8:
            st.subheader("Settings")
            
            # Stock selection
            symbol = st.text_input("Enter Stock Symbol (e.g., AAPL, MSFT, TSLA)", value=symbol)
            
            # Date selection
            end_date = st.date_input("End Date", value=end_date, key="settings_end_date")
            start_date = st.date_input("Start Date", value=start_date, key="settings_start_date")
            
            # Analysis options
            include_macro = st.checkbox("Include Macroeconomic Analysis", value=include_macro, key="settings_include_macro")
            show_advanced = st.checkbox("Show Advanced Features", value=show_advanced, key="settings_show_advanced")

    except Exception as e:
        st.error(f"Error: {e}")
        st.error("Please check your internet connection and try again.")
