import streamlit as st
from WalletPage import WalletPage
from HomePage import HomePage
from StockPurchaseBot import StockPurchaseBot
Page = st.sidebar.radio(
    "Welcome!",
    ("Home", "Wallet", "Stock Trading Bot"))


if Page == 'Home':
    HomePage = HomePage(st)
    HomePage.display()
if Page == 'Wallet':
    WalletPage = WalletPage(st)
    WalletPage.display()
if Page == 'Stock Trading Bot':
    StockPurchaseBot = StockPurchaseBot(st)
    StockPurchaseBot.display()

