import streamlit as st
from config import stockList, wallet
import yfinance as yf


def get_amount_to_pay(amountOfStock, ticker):
    stock = yf.Ticker(ticker)
    price = stock.info['regularMarketPrice'] * amountOfStock
    return price


class StockPurchaseBot:
    def __init__(self, st):
        self.st = st

    def display(self):
        self.st.title("Stock Trading Bot")
        self.st.write('Your current balance is ', wallet.getBalance(), '$')
        ticker = self.st.text_input('Please enter a stock ticker: ')
        if ticker:
            with st.spinner('Loading...'):
                self.st.write('The Current Price is: ', get_amount_to_pay(1, ticker), '$')
            with self.st.form('Stock Watchlist Form'):
                amount = self.st.number_input('Please enter the amount of  you would like to deposit: ',
                                                     min_value=0.0)
                submitted = self.st.form_submit_button("Submit")
                if submitted:
                    if amount > wallet.getBalance():
                        self.st.error('Not Enough Money')
                    else:
                        self.st.balloons()
                        self.st.success('Success...')
                        wallet.withdraw(amount)
                        if ticker in stockList:
                            stockList[ticker] += amount
                        else:
                            stockList[ticker] = amount

        with self.st.expander('Watch watchlist'):
            i = 0
            cols = self.st.columns(3)
            for stock in stockList:
                if i >= 3:
                    i = 0
                cols[i].metric(stock, stockList[stock])
                i = i + 1
