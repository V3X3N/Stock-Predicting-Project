# Import necessary libraries
import streamlit as st
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.graph_objects as go
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from textblob import TextBlob

# Set Seaborn style
sns.set_theme(style="whitegrid")

# Define start date and today's date
START = "2015-01-01"
TODAY = pd.to_datetime("today").strftime("%Y-%m-%d")

# Streamlit app title
st.title("Stock Prediction Project")

# Text input for the user to enter the stock symbol
selected_stocks = st.text_input("Enter stock symbol (e.g., GOOG, AAPL, NFLX)")

# Check if the user has entered a valid stock symbol
if not selected_stocks:
    st.warning("Please enter a valid stock symbol.")
    st.stop()

# Caching function to load stock data using yfinance
@st.cache_data()
def load_data(ticker):
    try:
        # Download stock data using yfinance
        data = yf.download(ticker, START, TODAY)

        # Check if data is empty
        if data.empty:
            st.warning("No data available for the selected stock symbol.")
            return None

        # Reset index to make Date a regular column
        data.reset_index(inplace=True)

        # Drop 'Adj Close' column if exists
        if 'Adj Close' in data.columns:
            data = data.drop(columns=['Adj Close'])

        return data
    except ValueError as e:
        st.warning(f"Error loading data: {e}")
        return None

# Display loading progress and load selected stock data
with st.spinner('Loading data...'):
    data = load_data(selected_stocks)

# Check if data is loaded successfully
if data is None:
    st.error("Failed to load data. Please check the entered stock symbol.")
    st.stop()

st.success('Loading complete!')

# Sidebar for additional controls
st.sidebar.header("Settings")
selected_start_date = st.sidebar.date_input("Select start date", pd.to_datetime("2015-01-01"))
selected_end_date = st.sidebar.date_input("Select end date", pd.to_datetime(TODAY))

# Convert date to datetime64[ns]
selected_start_date = pd.to_datetime(selected_start_date)
selected_end_date = pd.to_datetime(selected_end_date)

# Filter data based on the selected date range
filtered_data = data[(data['Date'] >= selected_start_date) & (data['Date'] <= selected_end_date)]

# Display raw data
st.subheader('Raw data')
st.write(filtered_data.tail())

# Dynamic layout with two columns
col1, col2 = st.columns(2)

# Column 1: Time Series Data and Moving Average Plot
with col1:
    st.subheader('Time Series Data')
    # Plot time series data
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(x=filtered_data['Date'], y=filtered_data['Close'], ax=ax)
    plt.title("Time Series Data")
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # Moving Average Plot
    st.subheader('Moving Average Plot')
    # Choose moving average window size using a slider
    moving_avg_window = st.slider("Select moving average window", 1, 30, 7)
    moving_average = filtered_data['Close'].rolling(window=moving_avg_window).mean()

    fig_ma, ax_ma = plt.subplots(figsize=(10, 5))
    sns.lineplot(x=filtered_data['Date'], y=filtered_data['Close'], label='Close Price', ax=ax_ma)
    sns.lineplot(x=filtered_data['Date'], y=moving_average, label=f'Moving Avg ({moving_avg_window} days)', ax=ax_ma)
    plt.title("Moving Average Plot")
    plt.xticks(rotation=45)
    plt.legend()
    st.pyplot(fig_ma)

# Column 2: Company Information and Sentiment Analysis
with col2:
    # Company Information
    st.subheader('Company Information')
    # Fetch and display company information using yfinance
    company_info = yf.Ticker(selected_stocks).info
    st.write(f"**Company Name:** {company_info.get('longName', 'N/A')}")
    
    # Check if 'sector' key exists in company_info
    if 'sector' in company_info:
        st.write(f"**Sector:** {company_info['sector']}")
    else:
        st.write("**Sector:** Information not available")

    st.write(f"**Industry:** {company_info.get('industry', 'N/A')}")
    st.write(f"**Website:** {company_info.get('website', 'N/A')}")

    # Sentiment Analysis
    st.subheader('Sentiment Analysis')
    # Perform sentiment analysis on the company description
    company_description = company_info['longBusinessSummary']
    blob = TextBlob(company_description)
    sentiment_score = blob.sentiment.polarity
    sentiment_label = "Positive" if sentiment_score > 0 else ("Negative" if sentiment_score < 0 else "Neutral")

    st.write(f"**Sentiment Score:** {sentiment_score:.2f}")
    st.write(f"**Sentiment Label:** {sentiment_label}")

# Forecasting section
# Extract relevant columns for forecasting
df_train = filtered_data[['Date', 'Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

# Slider to choose the number of prediction months
n_months = st.slider("Predicted months:", 1, 60)
period = n_months * 30

# Create and train Prophet model
m = Prophet()
m.fit(df_train)

# Generate future dataframe for prediction
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# Display forecast data
st.subheader('Forecast data')
st.write(forecast.tail())

# Plot interactive forecast chart using Plotly
st.subheader('Forecast Chart')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)
