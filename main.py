import streamlit as st
from datetime import date

import yfinance as yf
from sklearn.linear_model import LinearRegression
from plotly import graph_objects as go

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")
st.title("Stock Prediction App")

stocks = ("AAPL", "GOOG", "MSFT", "GME")
selected_stock = st.text_input("Type a stock ticker for prediction")
n_years = st.slider("Years of prediction: ", 1, 4)
period = n_years * 365

def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)#Makes the date be the first column
    return data
data_load_state = st.text("Load data...")
if len(selected_stock) > 0:
    data = load_data(selected_stock)
    data_load_state.text("Loading data...done!")


    st.subheader('Raw data')
    st.write(data.tail())

    def plot_raw_data():
        fig = go.Figure(data=[go.Candlestick(x=data['Date'],
                                             open=data['Open'],
                                             high=data['High'],
                                             low=data['Low'],
                                             close=data['Close'])])
        st.plotly_chart(fig)

    plot_raw_data()