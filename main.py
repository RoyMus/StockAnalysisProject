import streamlit as st
from WalletPage import WalletPage
from HomePage import HomePage
from StockPurchaseBot import StockPurchaseBot
from streamlit_option_menu import option_menu
from PortfolioPage import PortfolioPage



with st.sidebar:
    selected = option_menu("Main Menu", ["Home", 'Wallet', "Stock Trading Bot", "Portfolio"],
                           icons=['house', 'wallet2', 'robot','clipboard-data'], menu_icon="bar-chart", default_index=0,
                           styles={
        "nav-link-selected": {"background-color": "DarkCyan"}})
page = None

if selected == 'Home':
    page = HomePage(st)
    page.display()
if selected == 'Wallet':
    page = WalletPage(st)
    page.display()
if selected == 'Stock Trading Bot':
    page = StockPurchaseBot(st)
    page.display()
if selected == 'Portfolio':
    page = PortfolioPage(st)
    page.display()
