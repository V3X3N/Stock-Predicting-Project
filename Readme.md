# Stock Prediction Project

## Overview
This is a Streamlit web application for stock prediction. The application allows users to input a stock symbol (e.g., BTC-USD, GOOG, AAPL, NFLX) and provides a time series analysis, raw data visualization, and a forecast for the selected stock.

## Libraries Used
- **Streamlit**: Used for creating the interactive web application.
- **yfinance**: Used for downloading historical stock data.
- **Prophet**: Used for time series forecasting.
- **Plotly**: Used for creating interactive and visually appealing plots.

## How to Use
1. Enter a valid stock symbol in the text input field.
2. The application will load historical stock data using yfinance.
3. Raw data will be displayed in a table, and a time series plot will show the closing prices over time.
4. Use the slider to choose the number of months for the forecast.
5. The application uses the Prophet library to generate a forecast for the selected stock.
6. The forecast data and an interactive plot will be displayed.

## Getting Started
1. Clone the repository to your local machine.
   ```bash
   git clone https://github.com/your-username/stock-prediction.git
   cd stock-prediction
   ```

2. Install the required dependencies.
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app.
   ```bash
   streamlit run app.py
   ```

4. You will see the localhost address in the terminal

## Contributors
- [MKawa](https://github.com/V3X3N)