import streamlit as st
from datetime import date
import yfinance as yf
import Cython
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as graph_objs
from plotly import graph_objs as go


START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title("Stock Predition Project")

stocks = ("BTC-USD", "GOOG", "AAPL", "NFLX")
selected_stocks = st.selectbox("Select company", stocks)

@st.cache_data
def load_data(thicker):
    data = yf.download(thicker, START, TODAY)
    data.reset_index(inplace=True)
    return data

data_load_progress = st.text("Loading...")
data = load_data(selected_stocks)
data_load_progress.text("Loading complete!")

st.subheader('Raw data')
st.write(data.tail())

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='stock_open'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='stock_closed'))
    fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()

#Forecast
df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

n_years = st.slider("Predition years:", 1, 5)
period = n_years * 365

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

st.subheader('Forecast data')
st.write(forecast.tail())

st.write('forecast data')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write('forecast components')
fig2 = m.plot_components(forecast)
st.write(fig2)