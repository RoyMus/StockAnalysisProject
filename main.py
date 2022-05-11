import streamlit as st
from WalletPage import WalletPage
from HomePage import HomePage
from StockPurchaseBot import StockPurchaseBot
from streamlit_option_menu import option_menu
import config
with st.sidebar:
    selected = option_menu("Main Menu", ["Home", 'Wallet', "Stock Trading Bot", "Portfolio"],
                           icons=['house', 'wallet2', 'robot','clipboard-data'], menu_icon="bar-chart", default_index=1,
                           styles={
        "nav-link-selected": {"background-color": "DarkCyan"}})

if selected == 'Home':
    HomePage = HomePage(st)
    HomePage.display()
if selected == 'Wallet':
    WalletPage = WalletPage(st)
    WalletPage.display()
if selected == 'Stock Trading Bot':
    StockPurchaseBot = StockPurchaseBot(st)
    StockPurchaseBot.display()
if selected == 'Portfolio':
    StockPurchaseBot = StockPurchaseBot(st)
    StockPurchaseBot.display()
