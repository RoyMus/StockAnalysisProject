import streamlit as st
from config import wallet


class WalletPage:
    def __init__(self, webapp):
        self.st = webapp

    def display(self):
        self.st.title("Wallet")
        option = self.st.selectbox('Would you like to deposit money or withdraw? ', ('Deposit', 'Withdraw'))
        self.st.write('')
        self.st.write('')
        self.st.write('')
        if option == 'Deposit':
            deposit = self.st.number_input('Insert the amount of cash you would like to deposit in $:')
            wallet.deposit(deposit)
        else:
            withdraw = self.st.number_input('Insert the amount of cash you would like to withdraw in $: ')
            wallet.withdraw(withdraw)
        self.st.write('Your current balance is ', wallet.getBalance(), '$')
