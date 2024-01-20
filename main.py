# Import necessary libraries
import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

# Define start date and today's date
START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

# Streamlit app title
st.title("Stock Prediction Project")

# List of stock symbols
stocks = ("BTC-USD", "GOOG", "AAPL", "NFLX")
# Dropdown to select a stock
selected_stocks = st.selectbox("Select company", stocks)

# Caching function to load stock data using yfinance
@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

# Display loading progress and load selected stock data
data_load_progress = st.text("Loading...")
data = load_data(selected_stocks)
data_load_progress.text("Loading complete!")

# Display raw data
st.subheader('Raw data')
st.write(data.tail())

# Plot raw data using Plotly
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='stock_open'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='stock_closed'))
    fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

# Call function to plot raw data
plot_raw_data()

# Forecasting section
# Extract relevant columns for forecasting
df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

# Slider to choose the number of prediction years
n_years = st.slider("Prediction years:", 1, 5)
period = n_years * 365

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

# Display forecast components (trend, weekly, and yearly)
st.write('Forecast components')
fig2 = m.plot_components(forecast)
st.write(fig2)
