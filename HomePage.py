import streamlit as st
import pandas as pd
from datetime import date
import yfinance as yf
from plotly import graph_objects as go
from LinearRegressionAlgo import get_pred_table

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")


class HomePage:

    def __init__(self, webapp):
        self.st = webapp

    def load_data(self, ticker):
        dataframe = yf.download(ticker, START, TODAY)
        dataframe.drop(["Volume"], axis=1, inplace=True)
        dataframe.reset_index(inplace=True)  # Makes the date be the first column
        return dataframe

    def plot_raw_data(self, data, predTable):
        CandleStickFig = go.Figure(data=[go.Candlestick(x=data['Date'],
                                                        open=data['Open'],
                                                        high=data['High'],
                                                        low=data['Low'],
                                                        close=data['Close'])])
        CandleStickFig.layout.update(title_text='Candle Stick Chart')
        predFig = go.Figure()
        predFig.add_trace(go.Scatter(x=data['Date'], y=data['Adj Close'], name='Stock_Adj_close'))
        predFig.add_trace(go.Scatter(x=predTable['Date'], y=predTable['Prediction'], name='Prediction'))
        predFig.layout.update(title_text='Prediction Closing Price')
        predFig.update_xaxes(title_text='Date')
        predFig.update_yaxes(title_text='Price')
        predFig.update_layout(yaxis_tickprefix='$', yaxis_tickformat=',.2f')
        CandleStickFig.update_layout(yaxis_tickprefix='$', yaxis_tickformat=',.2f')
        st.plotly_chart(CandleStickFig)
        st.plotly_chart(predFig)

    def display(self):
        self.st.title("Stock Prediction App")
        selected_stock = st.text_input("Type a stock ticker for prediction")
        n_days = self.st.slider("Days of prediction: ", 1, 30)

        if not selected_stock:
            self.st.warning('Please enter a ticker')
            return

        with self.st.spinner("Loading..."):
            data = self.load_data(selected_stock)
        if data.empty:
            self.st.error('No such stock')
            return

        self.st.success("Done!")
        self.st.subheader('Raw data')
        tail = data.tail()
        print(tail)
        tail['Date'] = tail['Date'].dt.strftime('%d/%m/%Y')
        tail = tail.reindex(['Date', 'Open', 'Close', 'Low', 'High'], axis=1)  # Reaarange Columns
        tail.rename(columns={'Open': 'Open- $', 'Close': 'Close- $', 'Low': 'Low- $', 'High': 'High- $'},
                    inplace=True, errors='raise')  # Renaming columns
        style = tail.style.hide_index()
        self.st.write(style.to_html(), unsafe_allow_html=True)  # Show the table without the index column
        self.st.write("")
        self.st.write("")
        self.st.subheader('Prediction table: ')
        self.st.caption('Linear regression prediction')
        predTable = get_pred_table(n_days, data)
        dispTable = (predTable.tail(predTable.shape[0] - 1)).iloc[:, ::-1]
        dispTable.rename(columns={'Prediction': 'Prediction- $'}, inplace=True, errors='raise')
        dispTable['Date'] = dispTable['Date'].dt.strftime('%d/%m/%Y')
        style = dispTable.style.hide_index()
        self.st.write(style.to_html(), unsafe_allow_html=True)  # Show the table without the index column
        self.st.text("")
        self.plot_raw_data(data, predTable)

