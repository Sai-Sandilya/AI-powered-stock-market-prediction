import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# Page config
st.set_page_config(page_title="Stock Predictor - Minimal", layout="wide")
st.title("📈 Stock Predictor - Minimal Version")

# Sidebar
st.sidebar.header("Stock Selection")
symbol = st.sidebar.text_input("Enter Stock Symbol", value="AAPL")
days = st.sidebar.slider("Days of Historical Data", 30, 365, 180)

if symbol:
    try:
        # Fetch data
        st.write(f"Fetching data for {symbol}...")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Use yfinance directly without custom functions
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date)
        
        if not df.empty:
            st.success(f"✅ Fetched {len(df)} records for {symbol}")
            
            # Reset index to make Date a column
            df.reset_index(inplace=True)
            
            # Basic chart
            fig = make_subplots(rows=2, cols=1, 
                              subplot_titles=(f'{symbol} Price', 'Volume'),
                              vertical_spacing=0.1,
                              row_heights=[0.7, 0.3])
            
            # Price chart
            fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], 
                                   name='Close Price', line=dict(color='blue')),
                         row=1, col=1)
            
            # Volume chart
            fig.add_trace(go.Bar(x=df['Date'], y=df['Volume'], 
                               name='Volume', marker=dict(color='orange')),
                         row=2, col=1)
            
            fig.update_layout(height=600, showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
            
            # Basic stats
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Current Price", f"${df['Close'].iloc[-1]:.2f}")
            
            with col2:
                change = df['Close'].iloc[-1] - df['Close'].iloc[-2]
                st.metric("Daily Change", f"${change:.2f}")
            
            with col3:
                pct_change = (change / df['Close'].iloc[-2]) * 100
                st.metric("Daily Change %", f"{pct_change:.2f}%")
            
            with col4:
                st.metric("Volume", f"{df['Volume'].iloc[-1]:,.0f}")
            
            # Simple forecast (without complex algorithms)
            st.subheader("📊 Simple Price Forecast")
            
            forecast_days = st.slider("Forecast Days", 1, 30, 10)
            
            if st.button("Generate Simple Forecast"):
                # Very simple forecast using moving average trend
                recent_prices = df['Close'].tail(10)
                trend = (recent_prices.iloc[-1] - recent_prices.iloc[0]) / len(recent_prices)
                
                forecast_data = []
                last_price = df['Close'].iloc[-1]
                last_date = df['Date'].iloc[-1]
                
                for i in range(forecast_days):
                    # Simple linear trend with small random variation
                    next_price = last_price + trend + np.random.normal(0, last_price * 0.01)
                    next_date = last_date + timedelta(days=i+1)
                    
                    forecast_data.append({
                        'Date': next_date,
                        'Predicted_Price': next_price
                    })
                    last_price = next_price
                
                forecast_df = pd.DataFrame(forecast_data)
                
                # Display forecast
                st.write("**Forecast Results:**")
                st.dataframe(forecast_df.round(2))
                
                # Simple forecast chart
                fig_forecast = go.Figure()
                
                # Historical prices
                fig_forecast.add_trace(go.Scatter(
                    x=df['Date'].tail(20), 
                    y=df['Close'].tail(20),
                    name='Historical', 
                    line=dict(color='blue')
                ))
                
                # Forecast prices
                fig_forecast.add_trace(go.Scatter(
                    x=forecast_df['Date'], 
                    y=forecast_df['Predicted_Price'],
                    name='Forecast', 
                    line=dict(color='red', dash='dash')
                ))
                
                fig_forecast.update_layout(
                    title=f'{symbol} Price Forecast',
                    xaxis_title='Date',
                    yaxis_title='Price ($)',
                    height=400
                )
                
                st.plotly_chart(fig_forecast, use_container_width=True)
            
            # Raw data
            if st.checkbox("Show Raw Data"):
                st.subheader("Raw Data")
                st.dataframe(df)
                
        else:
            st.error(f"❌ No data found for {symbol}")
            
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.write("Try a different stock symbol or check your internet connection.")

# Footer
st.markdown("---")
st.markdown("**Note**: This is a minimal version for testing deployment. Full features available in main app.") 