import streamlit as st
from config import stockList
from LinearRegressionAlgo import get_pred_table

class PortfolioPage:
    def __init__(self, st):
        self.st = st

    def display(self):
        self.st.title("Portfolio")
