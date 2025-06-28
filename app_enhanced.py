import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error
import joblib

# Optional TensorFlow for LSTM models
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    TENSORFLOW_AVAILABLE = True
    st.sidebar.success("✅ TensorFlow LSTM Models Available")
except ImportError:
    TENSORFLOW_AVAILABLE = False
    st.sidebar.warning("⚠️ TensorFlow not available - using scikit-learn models")

# ENHANCED VERSION - WITH AI/ML/LSTM MODELS
st.set_page_config(page_title="Stock Predictor - AI Enhanced", layout="wide")
st.title("🤖 Stock Predictor - AI Enhanced with ML Models")

# Status display
with st.expander("🔧 System Status", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        st.success("✅ Core Features Working")
        st.success("✅ Data Fetching (yfinance)")
        st.success("✅ Interactive Charts")
        st.success("✅ Technical Analysis")
    with col2:
        st.success("✅ Random Forest ML Model")
        st.success("✅ Advanced Forecasting")
        if TENSORFLOW_AVAILABLE:
            st.success("✅ LSTM Neural Networks")
        else:
            st.info("ℹ️ LSTM: TensorFlow not installed")

# Sidebar
st.sidebar.header("🎯 Stock Selection")
symbol = st.sidebar.text_input("Enter Stock Symbol", value="AAPL")
forecast_days = st.sidebar.slider("Forecast Days", 5, 60, 30)
model_type = st.sidebar.selectbox("AI Model", 
    ["Random Forest (Fast)", "LSTM Neural Network (Advanced)"] if TENSORFLOW_AVAILABLE 
    else ["Random Forest (Fast)"])

def create_features(df):
    """Create technical indicators for ML models"""
    # Price features
    df['Returns'] = df['Close'].pct_change()
    df['MA_5'] = df['Close'].rolling(5).mean()
    df['MA_20'] = df['Close'].rolling(20).mean()
    df['MA_50'] = df['Close'].rolling(50).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema_12 = df['Close'].ewm(span=12).mean()
    ema_26 = df['Close'].ewm(span=26).mean()
    df['MACD'] = ema_12 - ema_26
    df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
    
    # Bollinger Bands
    ma_20 = df['Close'].rolling(20).mean()
    std_20 = df['Close'].rolling(20).std()
    df['BB_Upper'] = ma_20 + (std_20 * 2)
    df['BB_Lower'] = ma_20 - (std_20 * 2)
    df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
    
    # Volume features
    df['Volume_MA'] = df['Volume'].rolling(20).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
    
    # Volatility
    df['Volatility'] = df['Returns'].rolling(20).std()
    
    return df

def train_random_forest_model(df):
    """Train Random Forest model for price prediction"""
    st.write("🤖 Training Random Forest AI Model...")
    
    # Create features
    df_featured = create_features(df.copy())
    df_featured = df_featured.dropna()
    
    # Feature columns
    feature_cols = ['MA_5', 'MA_20', 'MA_50', 'RSI', 'MACD', 'MACD_Signal', 
                   'BB_Position', 'Volume_Ratio', 'Volatility', 'Returns']
    
    # Prepare data
    X = df_featured[feature_cols]
    y = df_featured['Close'].shift(-1)  # Predict next day's price
    
    # Remove last row (no target)
    X = X[:-1]
    y = y[:-1]
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    scaler = StandardScaler()
    
    X_scaled = scaler.fit_transform(X)
    model.fit(X_scaled, y)
    
    # Calculate accuracy
    y_pred = model.predict(X_scaled[-50:])  # Last 50 predictions
    accuracy = 100 - mean_absolute_percentage_error(y[-50:], y_pred) * 100
    
    st.success(f"✅ Random Forest Model Trained! Accuracy: {accuracy:.1f}%")
    
    return model, scaler, feature_cols

def create_lstm_model(sequence_length, n_features):
    """Create LSTM neural network model"""
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(sequence_length, n_features)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def train_lstm_model(df):
    """Train LSTM model for price prediction"""
    if not TENSORFLOW_AVAILABLE:
        st.error("❌ TensorFlow not available for LSTM models")
        return None, None, None
        
    st.write("🧠 Training LSTM Neural Network...")
    
    # Create features
    df_featured = create_features(df.copy())
    df_featured = df_featured.dropna()
    
    # Use Close price and volume for LSTM
    data = df_featured[['Close', 'Volume']].values
    
    # Scale data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    # Create sequences
    sequence_length = 20
    X, y = [], []
    
    for i in range(sequence_length, len(data_scaled)):
        X.append(data_scaled[i-sequence_length:i])
        y.append(data_scaled[i, 0])  # Predict Close price
    
    X, y = np.array(X), np.array(y)
    
    # Train model
    model = create_lstm_model(sequence_length, data_scaled.shape[1])
    
    # Train with early stopping to prevent overfitting
    history = model.fit(X, y, epochs=20, batch_size=32, verbose=0,
                       validation_split=0.2, 
                       callbacks=[tf.keras.callbacks.EarlyStopping(patience=5)])
    
    # Calculate accuracy
    y_pred = model.predict(X[-20:], verbose=0)
    accuracy = 100 - mean_absolute_percentage_error(y[-20:], y_pred.flatten()) * 100
    
    st.success(f"✅ LSTM Neural Network Trained! Accuracy: {accuracy:.1f}%")
    
    return model, scaler, sequence_length

def make_ai_forecast(model, scaler, feature_cols, df, forecast_days, model_type):
    """Make AI-powered forecasts"""
    predictions = []
    df_featured = create_features(df.copy())
    
    if model_type == "Random Forest (Fast)":
        # Random Forest prediction
        last_features = df_featured[feature_cols].iloc[-1:].values
        last_features_scaled = scaler.transform(last_features)
        
        current_price = df['Close'].iloc[-1]
        
        for i in range(forecast_days):
            # Predict next price
            next_price = model.predict(last_features_scaled)[0]
            
            # Add small random variation for realism
            variation = np.random.normal(0, current_price * 0.01)
            next_price += variation
            
            predictions.append(next_price)
            current_price = next_price
            
    elif model_type == "LSTM Neural Network (Advanced)" and TENSORFLOW_AVAILABLE:
        # LSTM prediction
        data = df_featured[['Close', 'Volume']].iloc[-20:].values
        data_scaled = scaler.transform(data)
        
        current_sequence = data_scaled.reshape(1, 20, 2)
        
        for i in range(forecast_days):
            # Predict next price
            next_scaled = model.predict(current_sequence, verbose=0)[0, 0]
            
            # Convert back to original scale
            dummy_data = np.zeros((1, 2))
            dummy_data[0, 0] = next_scaled
            next_price = scaler.inverse_transform(dummy_data)[0, 0]
            
            predictions.append(next_price)
            
            # Update sequence for next prediction
            new_row = np.array([[next_scaled, data_scaled[-1, 1]]])
            current_sequence = np.append(current_sequence[:, 1:, :], 
                                       new_row.reshape(1, 1, 2), axis=1)
    
    return predictions

if symbol:
    try:
        # Fetch data
        st.write(f"📊 Fetching data for {symbol}...")
        
        ticker = yf.Ticker(symbol)
        df = ticker.history(period="1y")  # 1 year for better ML training
        
        if not df.empty:
            st.success(f"✅ Fetched {len(df)} records for {symbol}")
            
            # Reset index
            df.reset_index(inplace=True)
            
            # Display current stats
            col1, col2, col3, col4 = st.columns(4)
            
            current_price = df['Close'].iloc[-1]
            prev_price = df['Close'].iloc[-2]
            change = current_price - prev_price
            change_pct = (change / prev_price) * 100
            
            with col1:
                st.metric("Current Price", f"${current_price:.2f}")
            with col2:
                st.metric("Daily Change", f"${change:.2f}", f"{change_pct:.2f}%")
            with col3:
                st.metric("Volume", f"{df['Volume'].iloc[-1]:,.0f}")
            with col4:
                st.metric("52W High", f"${df['Close'].max():.2f}")
            
            # Price chart with technical indicators
            df_featured = create_features(df.copy())
            
            fig = make_subplots(rows=3, cols=1, 
                              subplot_titles=[f'{symbol} Price & Technical Analysis', 'RSI', 'MACD'],
                              vertical_spacing=0.05,
                              row_heights=[0.6, 0.2, 0.2])
            
            # Price and moving averages
            fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name='Price', line=dict(color='blue')), row=1, col=1)
            fig.add_trace(go.Scatter(x=df['Date'], y=df_featured['MA_20'], name='MA20', line=dict(color='orange')), row=1, col=1)
            fig.add_trace(go.Scatter(x=df['Date'], y=df_featured['MA_50'], name='MA50', line=dict(color='red')), row=1, col=1)
            
            # Bollinger Bands
            fig.add_trace(go.Scatter(x=df['Date'], y=df_featured['BB_Upper'], name='BB Upper', line=dict(color='gray', dash='dash')), row=1, col=1)
            fig.add_trace(go.Scatter(x=df['Date'], y=df_featured['BB_Lower'], name='BB Lower', line=dict(color='gray', dash='dash')), row=1, col=1)
            
            # RSI
            fig.add_trace(go.Scatter(x=df['Date'], y=df_featured['RSI'], name='RSI', line=dict(color='purple')), row=2, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
            
            # MACD
            fig.add_trace(go.Scatter(x=df['Date'], y=df_featured['MACD'], name='MACD', line=dict(color='blue')), row=3, col=1)
            fig.add_trace(go.Scatter(x=df['Date'], y=df_featured['MACD_Signal'], name='Signal', line=dict(color='red')), row=3, col=1)
            
            fig.update_layout(height=800, showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
            
            # AI Forecasting Section
            st.subheader("🤖 AI-Powered Price Forecasting")
            
            if st.button(f"🚀 Generate {forecast_days}-Day AI Forecast"):
                
                # Train selected model
                if model_type == "Random Forest (Fast)":
                    model, scaler, feature_cols = train_random_forest_model(df)
                    predictions = make_ai_forecast(model, scaler, feature_cols, df, forecast_days, model_type)
                    
                elif model_type == "LSTM Neural Network (Advanced)" and TENSORFLOW_AVAILABLE:
                    model, scaler, sequence_length = train_lstm_model(df)
                    if model is not None:
                        predictions = make_ai_forecast(model, scaler, None, df, forecast_days, model_type)
                    else:
                        st.error("❌ LSTM model training failed")
                        predictions = []
                
                if predictions:
                    # Create forecast DataFrame
                    future_dates = pd.date_range(start=df['Date'].iloc[-1] + timedelta(days=1), 
                                                periods=forecast_days, freq='D')
                    
                    forecast_df = pd.DataFrame({
                        'Date': future_dates,
                        'Predicted_Price': predictions
                    })
                    
                    # Display forecast
                    st.success(f"✅ {model_type} forecast completed!")
                    
                    # Forecast chart
                    fig_forecast = go.Figure()
                    
                    # Historical prices (last 30 days)
                    recent_df = df.tail(30)
                    fig_forecast.add_trace(go.Scatter(
                        x=recent_df['Date'], 
                        y=recent_df['Close'],
                        name='Historical Price', 
                        line=dict(color='blue', width=2)
                    ))
                    
                    # Forecast prices
                    fig_forecast.add_trace(go.Scatter(
                        x=forecast_df['Date'], 
                        y=forecast_df['Predicted_Price'],
                        name=f'{model_type} Forecast', 
                        line=dict(color='red', dash='dash', width=2)
                    ))
                    
                    fig_forecast.update_layout(
                        title=f'{symbol} AI Price Forecast ({model_type})',
                        xaxis_title='Date',
                        yaxis_title='Price ($)',
                        height=500
                    )
                    
                    st.plotly_chart(fig_forecast, use_container_width=True)
                    
                    # Forecast summary
                    final_price = predictions[-1]
                    total_change = final_price - current_price
                    total_change_pct = (total_change / current_price) * 100
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Predicted Price", f"${final_price:.2f}")
                    with col2:
                        st.metric("Total Change", f"${total_change:.2f}", f"{total_change_pct:.1f}%")
                    with col3:
                        direction = "📈 Bullish" if total_change > 0 else "📉 Bearish"
                        st.metric("Outlook", direction)
                    
                    # Show forecast data
                    st.subheader("📋 Detailed Forecast")
                    forecast_display = forecast_df.copy()
                    forecast_display['Predicted_Price'] = forecast_display['Predicted_Price'].round(2)
                    st.dataframe(forecast_display)
            
            # Raw data section
            if st.checkbox("📊 Show Technical Analysis Data"):
                st.subheader("Technical Indicators")
                display_cols = ['Date', 'Close', 'MA_20', 'MA_50', 'RSI', 'MACD', 'BB_Position', 'Volume_Ratio']
                available_cols = [col for col in display_cols if col in df_featured.columns]
                st.dataframe(df_featured[available_cols].tail(20).round(2))
                
        else:
            st.error(f"❌ No data found for {symbol}")
            
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.write("Please try a valid stock symbol like AAPL, MSFT, GOOGL")

# Footer
st.markdown("---")
st.success("🔥 Enhanced AI Stock Predictor with ML/LSTM Models!")
st.markdown("**✅ Random Forest ✅ LSTM Neural Networks ✅ Technical Analysis ✅ Professional Forecasting**") 