# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from prophet import Prophet
import warnings
warnings.filterwarnings('ignore')


# Função para carregar dados
@st.cache
def load_data():
    symbol = 'BZ=F'
    start_date = '1987-05-20'
    end_date = '2024-05-20'
    df = yf.download(symbol, start=start_date, end=end_date)
    df['Date'] = pd.to_datetime(df.index)
    return df.reset_index(drop=True)

# Função para plotar série temporal
def plot_time_series(data, title="Time Series"):
    print(data.columns)  # Verifica as colunas presentes no DataFrame
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.plot(data['Date'], data['Close'], label='Close')
    ax.set_title(title)
    ax.set_xlabel('Date')
    ax.set_ylabel('Close Price')



# Função para decompor série temporal
def plot_decomposition(data):
    result = seasonal_decompose(data['Close'], model='multiplicative', period=7)
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(15, 8))
    result.observed.plot(ax=ax1)
    result.trend.plot(ax=ax2)
    result.seasonal.plot(ax=ax3)
    result.resid.plot(ax=ax4)
    plt.tight_layout()
    st.pyplot(fig)

# Função para prever usando ARIMA
def arima_forecast(data):
    model = ARIMA(data['Close'], order=(2, 1, 2))
    results = model.fit()
    data['Forecast_ARIMA'] = results.fittedvalues
    return data

# Função para prever usando Prophet
def prophet_forecast(train_data, periods=365):
    model = Prophet(daily_seasonality=True)
    train_data = train_data.rename(columns={"Date": "ds", "Close": "y"})
    model.fit(train_data)
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return forecast

# Função para prever usando LSTM
def lstm_forecast(data, look_back=10):
    date_column = data['Date']
    data = data[['Close']]
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)
    X, y = [], []
    for i in range(look_back, len(data_scaled)):
        X.append(data_scaled[i-look_back:i, 0])
        y.append(data_scaled[i, 0])
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=1, batch_size=1, verbose=0)
    
    predictions = model.predict(X)
    predictions = scaler.inverse_transform(predictions)
    
    data['Forecast_LSTM'] = np.nan
    data['Forecast_LSTM'].iloc[look_back:] = predictions.flatten()
    
    data['Date'] = date_column  # Reinsira a coluna 'Date'
    return data

# Função principal da aplicação Streamlit
def main():
    st.title("Predição de Séries Temporais")

    menu = ["Carregar Dados", "Visualizar Dados", "Decomposição", "Previsão ARIMA", "Previsão Prophet", "Previsão LSTM"]
    choice = st.sidebar.selectbox("Menu", menu)

    data = load_data()

    if choice == "Carregar Dados":
        st.subheader("Carregar Dados")
        st.write(data)

    elif choice == "Visualizar Dados":
        st.subheader("Visualizar Dados")
        plot_time_series(data)

    elif choice == "Decomposição":
        st.subheader("Decomposição")
        plot_decomposition(data)

    elif choice == "Previsão ARIMA":
        st.subheader("Previsão ARIMA")
        forecast_data = arima_forecast(data)
        plot_time_series(forecast_data, title="Previsão ARIMA")
    elif choice == "Previsão Prophet":
        st.subheader("Previsão Prophet")
        forecast = prophet_forecast(data)
        fig = plt.figure(figsize=(15, 10))
        plt.plot(data['Date'], data['Close'], label='Close')
        plt.plot(forecast['ds'], forecast['yhat'], label='Forecast')
        plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], alpha=0.2)
        plt.xlabel('Date')
        plt.ylabel('Close Price')
        plt.legend()
        st.pyplot(fig)

    elif choice == "Previsão LSTM":
        st.subheader("Previsão LSTM")
        forecast_data = lstm_forecast(data)
        plot_time_series(forecast_data, title="Previsão LSTM")

if __name__ == "__main__":
    main()
