import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA

# Configure page
st.set_page_config(page_title="Time Series & Portfolio Dashboard", layout="wide")
st.title("📈 Time Series Forecasting & Portfolio Optimization")
st.markdown("A capstone project dashboard for analyzing Indian Stocks (NSE).")

# --- SIDEBAR CONFIGURATION ---
st.sidebar.header("Configuration")
stocks_dict = {
    "TCS.NS": "IT", "HDFCBANK.NS": "Banking", "SUNPHARMA.NS": "Pharma",
    "HINDUNILVR.NS": "FMCG", "MARUTI.NS": "Auto", "RELIANCE.NS": "Energy"
}
selected_stock = st.sidebar.selectbox("Select a Stock to Analyze", list(stocks_dict.keys()))
start_date = st.sidebar.date_input("Start Date", datetime(2021, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.today())
forecast_days = st.sidebar.slider("Forecast Horizon (Days)", 7, 90, 30)

# --- FETCH DATA ---
@st.cache_data
def load_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end, progress=False)
    # Fix potential multi-index issues with newer yfinance versions
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.reset_index(inplace=True)
    return df

with st.spinner(f"Loading data for {selected_stock}..."):
    data = load_data(selected_stock, start_date, end_date)

st.subheader(f"Raw Data: {selected_stock} ({stocks_dict[selected_stock]})")
st.dataframe(data.tail())

# --- PLOT HISTORICAL DATA ---
st.subheader("Historical Close Price")
fig = go.Figure()
fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], mode='lines', name='Close Price'))
fig.update_layout(xaxis_title="Date", yaxis_title="Price (INR)", margin=dict(l=0, r=0, t=30, b=0))
st.plotly_chart(fig, use_container_width=True)

# --- FORECASTING (ARIMA) ---
st.subheader("ARIMA Forecast")
if st.button("Run ARIMA Forecast"):
    with st.spinner("Training ARIMA model..."):
        # Fit ARIMA Model
        train_data = data['Close'].values
        model = ARIMA(train_data, order=(5, 1, 0)) # Simple baseline parameters
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=forecast_days)
        
        # Create future dates
        last_date = data['Date'].iloc[-1]
        future_dates = [last_date + timedelta(days=i) for i in range(1, forecast_days + 1)]
        
        # Plot
        fig_forecast = go.Figure()
        fig_forecast.add_trace(go.Scatter(x=data['Date'][-100:], y=data['Close'][-100:], mode='lines', name='Historical (Last 100d)'))
        fig_forecast.add_trace(go.Scatter(x=future_dates, y=forecast, mode='lines', name='Forecast', line=dict(dash='dash', color='red')))
        fig_forecast.update_layout(title=f"{forecast_days}-Day Forecast for {selected_stock}", xaxis_title="Date", yaxis_title="Price")
        st.plotly_chart(fig_forecast, use_container_width=True)
        
        predicted_return = ((forecast[-1] - data['Close'].iloc[-1]) / data['Close'].iloc[-1]) * 100
        st.metric(label="Expected Return", value=f"{predicted_return:.2f}%")

# --- PORTFOLIO CORRELATION ---
st.subheader("Portfolio Correlation Heatmap")
if st.button("Generate Portfolio Analytics"):
    with st.spinner("Fetching portfolio data..."):
        portfolio_data = pd.DataFrame()
        for ticker in stocks_dict.keys():
            temp_df = load_data(ticker, start_date, end_date)
            portfolio_data[ticker] = temp_df.set_index('Date')['Close']
            
        returns = portfolio_data.pct_change().dropna()
        corr = returns.corr()
        
        fig_corr = go.Figure(data=go.Heatmap(
                   z=corr.values,
                   x=corr.index.values,
                   y=corr.columns.values,
                   colorscale='Viridis'))
        fig_corr.update_layout(title="Asset Correlation (Daily Returns)")
        st.plotly_chart(fig_corr, use_container_width=True)