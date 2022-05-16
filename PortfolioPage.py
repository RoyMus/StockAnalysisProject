import streamlit as st
from config import stockList, portfolioStocks, wallet
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


def BuySellHold():
    if stockList:
        for stock in stockList.copy():
            if get_market_price(stock) < get_lr_prediction(1, load_data(stock)):
                amount = stockList.pop(stock)
                if stock in portfolioStocks:
                    portfolioStocks[stock][0] += get_shares(amount, stock)
                    portfolioStocks[stock][1] += amount
                else:
                    portfolioStocks[stock] = [get_shares(amount, stock), amount]
    if portfolioStocks:
        for stock in portfolioStocks.copy():
            if get_market_price(stock) > get_lr_prediction(1, load_data(stock)) or calcChange(stock,
                                                                                              portfolioStocks) < 0:
                amount = portfolioStocks.pop(stock)[0]
                wallet.deposit(amount)


def calcChange(stock, DictOfStocks):
    return round(((get_amount(stock, DictOfStocks[stock][0]) - DictOfStocks[stock][1]) / DictOfStocks[stock][1]) * 100,
                 2)


def get_shares(amount, ticker):
    return round(amount / get_market_price(ticker))


class PortfolioPage:
    def __init__(self, st):
        self.st = st

    def display(self):
        self.st.title("Portfolio")
        with self.st.spinner('Loading...'):
            if stockList:
                BuySellHold()
            self.st.write('Your Portfolio Value: ', round(get_portfolioValue(), 2), '$')
            i = 0
            cols = self.st.columns(3)
            for stock in portfolioStocks:
                if i >= 3:
                    i = 0
                cols[i].metric(stock, round(portfolioStocks[stock][0], 2), calcChange(stock, portfolioStocks))
                i = i + 1
