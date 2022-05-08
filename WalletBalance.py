class WalletBalance:
    def __init__(self, amountToDeposit):
        self.amount = amountToDeposit

    def deposit(self, amountToDeposit):
        self.amount = self.amount + amountToDeposit

    def withdraw(self, amountToWithdraw):
        self.amount = self.amount - amountToWithdraw

    def getBalance(self):
        return self.amount
