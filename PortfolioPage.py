import streamlit as st
from config import stockList, portfolioStocks
from LinearRegressionAlgo import get_lr_prediction
from HomePage import load_data
import yfinance as yf


def get_market_price(ticker):
    return yf.Ticker(ticker).info['regularMarketPrice']


def get_amount(stock, amountofStock):
    return get_market_price(stock) * amountofStock


def get_portfolioValue():
    sum = 0
    for stock in portfolioStocks.keys():
        sum += get_amount(stock, portfolioStocks[stock][0])
    return sum


def BuyOrHold():
    for stock in stockList.copy():
        if get_market_price(stock) < get_lr_prediction(1, load_data(stock)):
            amount = stockList.pop(stock)
            if stock in portfolioStocks:
                portfolioStocks[stock][0] += get_shares(amount, stock)
                portfolioStocks[stock][1] += amount
            else:
                portfolioStocks[stock] = [get_shares(amount, stock), amount]


def get_shares(amount, ticker):
    return amount / get_market_price(ticker)


class PortfolioPage:
    def __init__(self, st):
        self.st = st

    def display(self):
        self.st.title("Portfolio")
        with self.st.spinner('Loading...'):
            if stockList:
                BuyOrHold()
            self.st.write('Your Portfolio Value: ', round(get_portfolioValue(), 2), '$')
            i = 0
            cols = self.st.columns(3)
            for stock in portfolioStocks:
                if i >= 3:
                    i = 0
                cols[i].metric(stock, round(portfolioStocks[stock][0], 2), round(((get_amount(stock, portfolioStocks[stock][0]) - portfolioStocks[stock][1])/portfolioStocks[stock][1])*100, 2))
                i = i + 1
