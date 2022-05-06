class WalletBalance:
    def __init__(self, amountToDeposit):
        self.amount = amountToDeposit

    def buyStock(self, stockPrice):
        self.amount = -stockPrice

    def sellStock(self, stockPrice):
        self.amount = +stockPrice

    def deposit(self, amountToDeposit):
        self.amount = +amountToDeposit

    def withdraw(self, amountToWithdraw):
        self.amount = -amountToWithdraw

    def getBalance(self):
        return self.amount
