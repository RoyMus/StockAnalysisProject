import streamlit as st


class WalletPage:
    def __init__(self, webapp, wallet):
        self.st = webapp
        self.wallet = wallet
    def display(self):
        option = self.st.selectbox('Would you like to deposit money or withdraw? ', ('Deposit', 'Withdraw'))
        self.st.write('')
        self.st.write('')
        self.st.write('')
        if option == 'Deposit':
            deposit = self.st.number_input('Insert the amount of cash you would like to deposit: ')
            self.wallet.deposit(deposit)
        else:
            withdraw = self.st.number_input('Insert the amount of cash you would like to withdraw: ')
            self.wallet.withdraw(withdraw)
        self.st.write(self.wallet.getBalance())

