# Import necessary libraries
import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import pandas as pd

# Define start date and today's date
START = "2015-01-01"
TODAY = pd.to_datetime("today").strftime("%Y-%m-%d")

# Streamlit app title
st.title("Stock Prediction Project")

# Text input for user to enter the stock symbol
selected_stocks = st.text_input("Enter stock symbol (e.g., BTC-USD, GOOG, AAPL, NFLX)")

# Check if user has entered a valid stock symbol
if not selected_stocks:
    st.warning("Please enter a valid stock symbol.")
    st.stop()

# Caching function to load stock data using yfinance
@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    if 'Adj Close' in data.columns:
        data = data.drop(columns=['Adj Close'])
    return data

# Display loading progress and load selected stock data
with st.spinner('Loading data...'):
    data = load_data(selected_stocks)
st.success('Loading complete!')

# Display raw data
st.subheader('Raw data')
st.write(data.tail())

# Plot raw data using Plotly
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close']))
    fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

# Call function to plot raw data
plot_raw_data()

# Forecasting section
# Extract relevant columns for forecasting
df_train = data[['Date', 'Close']]
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

# Plot interactive forecast chart
st.write('Forecast data')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)