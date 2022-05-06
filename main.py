import streamlit as st
from datetime import date
from WalletPage import WalletPage
from HomePage import HomePage
from WalletBalance import WalletBalance
Page = st.sidebar.radio(
    "Welcome!",
    ("Home", "Wallet", "Stock Trading Bot"))

if Page == 'Home':
    HomePage = HomePage(st)
    HomePage.display()
if Page == 'Wallet':
    wallet = WalletBalance(0)
    WalletPage = WalletPage(st, wallet)
    WalletPage.display()
