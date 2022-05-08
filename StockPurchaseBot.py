import streamlit as st


class StockPurchaseBot:
    def __init__(self, st):
        self.st = st

    def display(self):
        self.st.write('Please choose a stock: ')
        # TODO: Add a stock
