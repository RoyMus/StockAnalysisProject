import streamlit as st
from config import stockList,wallet


class StockPurchaseBot:
    def __init__(self, st):
        self.st = st

    def display(self):
        self.st.title("Stock Trading Bot")
        with self.st.form('Stock Watchlist Form'):
            ticker = self.st.text_input('Please enter a stock ticker: ')
            amount = self.st.number_input('Please enter the amount of stocks you would like to purchase: ', min_value=1, step=1 )
            submitted = self.st.form_submit_button("Submit")
            if submitted:
                if amount > wallet.getBalance():
                    self.st.error('Not Enough Money')
                wallet.withdraw(amount)
                stockList[ticker] = amount

        with self.st.expander('Watch watchlist'):
            i = 0
            cols = self.st.columns(3)
            for stock in stockList:
                if i >= 3:
                    i = 0
                cols[i].metric(stock, stockList[stock])
                i = i+1
