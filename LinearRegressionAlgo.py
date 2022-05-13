import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from datetime import timedelta


def get_pred_table(n_days, data):
    lastPrice = data['Adj Close'].iloc[-1]
    lr_prediction_array = np.array([lastPrice])  # Adding the last value to the prediction array
    lastDate = data['Date'].iloc[-1]
    predTable = pd.DataFrame()
    predDates = [lastDate]  # Creating a list with the last date
    i = 0
    days = 0
    while days < n_days:
        nextDate = lastDate + timedelta(days=i + 1)
        i += 1
        if 2 <= nextDate.weekday() <= 6:  # Checking that the date added is in the stock market open days
            lr_prediction = get_lr_prediction((days+1), data)  # Get the predicted price for the number of days selected
            lr_prediction_array = np.append(lr_prediction_array, lr_prediction)
            predDates.append(nextDate)
            days += 1

    predTable['Prediction'] = lr_prediction_array
    predTable['Date'] = np.array(predDates)
    return predTable


def get_lr_prediction(n_days, data):
    df = data[['Adj Close']]  # choose a data frame of adj close
    # A variable for predicting 1 day out into the future
    forecast_out = n_days
    # Create a prediction column shifted 1 day up
    df['Prediction'] = df[['Adj Close']].shift(periods=-forecast_out)
    # Creating a data set to train the Linear Regression model
    X = np.array(df.drop(['Prediction'], axis=1))
    # Get all of the X values except the last  row
    X = X[:-forecast_out]
    # Create the dependent data set (y)
    # Convert the dataframe to a numpy array
    y = np.array(df['Prediction'])
    # Get all of the y values except the last row
    y = y[:-forecast_out]
    # Split the data into 80% training and 20% testing
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # Create the Linear Regression  Model
    lr = LinearRegression()
    # Train the model
    lr.fit(x_train, y_train)
    # Testing Model: Score returns the coefficient of determination R^2 of the prediction.
    # The best possible score is 1.0
    lr_confidence = lr.score(x_test, y_test)
    # Set x_forecast equal to the last 'n' rows of the original data set from Adj. Close column
    x_forecast = np.array(df.drop(['Prediction'], axis=1))[-1:]
    # Print linear regression model predictions for the next  day
    lr_prediction = lr.predict(x_forecast)
    return lr_prediction
