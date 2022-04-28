import streamlit as st
from datetime import date
import time
import numpy as np
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from plotly import graph_objects as go

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

Page = st.sidebar.radio(
    "Welcome!",
    ("Home", "Wallet", "Stock Trading Bot"))


def load_data(ticker):
    dataframe = yf.download(ticker, START, TODAY)
    dataframe.drop(["Adj Close", "Volume"], axis=1, inplace=True)
    dataframe.reset_index(inplace=True)  # Makes the date be the first column
    return dataframe


if Page == 'Home':
    st.title("Stock Prediction App")

    selected_stock = st.text_input("Type a stock ticker for prediction")
    n_days = st.slider("Days of prediction: ", 1, 25)

    if len(selected_stock) > 0:
        data = load_data(selected_stock)
        # df = data['Close Price']
        # Create a variable to predict 'x' days out into the future
        # future_days = n_days
        # Create a new column(target) shifted 'x' days up
        # df['Prediction'] = df[['Close Price']].shift(-future_days)
        # Create the feature data set (X) and convert it to numpy array and remove last 'x' rows/days
        # X = np.array(df.drop(['Prediction'],1))[:-future_days]
        # Create the target data set(y) and convert it to numpy array and get all of the target values except the last 'x'
        # y = np.array(df['Prediction'])[:-future_days]
        # Split the data to 25% testing and 75% training
        # x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.25)
        # lr = LinearRegression().fit(x_train, y_train)
        # x_future = df.drop(['Prediction'], 1)[:-fu]
        with st.spinner("Loading..."):
            time.sleep(3)
        st.success("Done!")
        st.subheader('Raw data')
        st.write(data.tail())


        def plot_raw_data():
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='Stock_open'))
            fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='Stock_close'))
            fig.layout.update(title_text='Time Series Data')
            fig.update_xaxes(title_text='Date')
            fig.update_yaxes(title_text='Price')
            st.plotly_chart(fig)


        plot_raw_data()
    else:
        st.warning('Please enter a vaild ticker')
