import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from datetime import timedelta


def get_pred_table(n_days, data):
    df = data[['Adj Close']]  # choose a data frame of adj close
    # A variable for predicting 'n' days out into the future
    forecast_out = n_days
    # Create a prediction column shifted 'n' days up
    df['Prediction'] = df[['Adj Close']].shift(periods=-forecast_out)
    # Creating a data set to train the Linear Regression model
    X = np.array(df.drop(['Prediction'], axis=1))
    # et all of the X values except the last 'n' rows
    X = X[:-forecast_out]
    # Create the dependent data set (y)
    # Convert the dataframe to a numpy array
    y = np.array(df['Prediction'])
    # Get all of the y values except the last 'n' rows
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
    # Set x_forecast equal to the last 30 rows of the original data set from Adj. Close column
    x_forecast = np.array(df.drop(['Prediction'], axis=1))[-forecast_out:]
    # Print linear regression model predictions for the next '30' days
    lastPrice = data['Adj Close'].iloc[-1]
    lr_prediction = lr.predict(x_forecast)
    lr_prediction = np.append([lastPrice], lr_prediction)  # Adding the last value to the prediction array
    lastDate = data['Date'].iloc[-1]
    predTable = pd.DataFrame()
    predTable['Prediction'] = lr_prediction
    predDates = [lastDate]  # Creating a list with the last date
    counter = 0
    i = 0
    while counter < n_days:
        nextDate = lastDate + timedelta(days=i + 1)
        i += 1
        if 2 <= nextDate.weekday() <= 6:  # Checking that the date added is in the stock market open days
            predDates.append(nextDate)
            counter += 1

    predTable['Date'] = np.array(predDates)
    return predTable
