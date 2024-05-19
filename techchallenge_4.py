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

# Função para carregar dados
@st.cache
def load_data():
    symbol = 'BZ=F'
    start_date = '1987-05-20'
    end_date = '2024-05-20'
    df = yf.download(symbol, start=start_date, end=end_date)
    df['Date'] = pd.to_datetime(df.index)
    # Clonar o DataFrame para evitar mutação
    cloned_df = df.copy()
    return cloned_df.reset_index(drop=True)

# Função para plotar série temporal
def plot_time_series(data, title="Time Series"):
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.plot(data['Date'], data['Close'], label='Close')
    ax.set_title(title)
    ax.set_xlabel('Date')
    ax.set_ylabel('Close Price')
    st.pyplot(fig)

# Função para decompor série temporal

def plot_decomposition(data):
    result = seasonal_decompose(data['Close'], model='multiplicative', period=7)
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(15, 8))
    
    ax1.plot(result.observed)
    ax1.set_title('Série Real')
    
    ax2.plot(result.trend)
    ax2.set_title('Tendência')
    
    ax3.plot(result.seasonal)
    ax3.set_title('Sazonalidade')
    
    ax4.plot(result.resid)
    ax4.set_title('Resíduos')

    plt.tight_layout()
    st.markdown("### Decomposição da série temporal")
    st.pyplot(fig)


# Função para prever usando ARIMA
def arima_forecast(data):
    try:
        model = ARIMA(data['Close'], order=(2, 1, 2))
        results = model.fit()
        data['Forecast_ARIMA'] = results.fittedvalues

        # Plotando os dados originais e as previsões no mesmo gráfico
        plt.figure(figsize=(10, 6))
        plt.plot(data['Date'], data['Close'], label='Original Data')
        plt.plot(data['Date'], data['Forecast_ARIMA'], label='ARIMA Forecast', color='red')
        plt.title('ARIMA Forecast')
        plt.xlabel('Date')
        plt.ylabel('Close Price')
        plt.legend()
        st.pyplot(plt.gcf())  # Usando plt.gcf() para obter a figura atual
        return data

    except Exception as e:
        st.error(f"Erro ao executar previsão ARIMA: {e}")
        return None

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
    st.title("Análise de Preço do Petróleo Brent")

    menu = ["Challenge", "Carregar Dados", "Visualizar Dados", "Decomposição", "Previsão ARIMA", "Previsão Prophet", "Previsão LSTM"]
    choice = st.sidebar.selectbox("Menu", menu)

    data = load_data()

    if choice == "Carregar Dados":
        st.subheader("Carregar Dados")
        st.markdown(
            """
            Carregando os dados utilizando a Biblioteca Yahoo Finance. Os dados obtidos utilizando esta biblioteca entregam mais colunas que necessário. Neste caso, estaremos analisando as colunas de 'CLOSE' e 'DATE'.
            """
        )
        st.write(data)

    elif choice == "Visualizar Dados":
        st.subheader("Visualizar Dados")
        st.markdown(
            """
            Visualização histórica dos dados com a variação de preço no decorrer do tempo. Os dados começam em 30 de Junho de 2007.
            """
        )
        plot_time_series(data)

    elif choice == "Decomposição":
        st.subheader("Decomposição")
        st.markdown(
            """
            Notas da análise de decomposição da série
            
            - model='additive': Este parâmetro especifica o tipo de modelo utilizado na decomposição.
            - Os dois tipos principais são:
            - "additive" (modelo apropriado quando a magnitude da sazonalidade não varia com a tendência)
            - "multiplicative" (modelo é mais apropriado quando a magnitude da sazonalidade varia com a tendência).
            - period: Este é o período da sazonalidade. Ele especifica o número de observações em um ciclo sazonal.
            """
        )
        plot_decomposition(data)

    elif choice == "Previsão ARIMA":
        st.subheader("Previsão ARIMA")
        forecast_data = arima_forecast(data)
        if forecast_data is not None:
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
        
    elif choice == "Challenge":
        st.subheader("Challenge")
        st.markdown(
            """
            Você foi contratado para uma consultoria, e seu trabalho envolve analisar os dados de preço do petróleo Brent, que pode ser encontrado no site do ipea. Essa base de dados histórica envolve duas colunas: data e preço (em dólares).

            Um grande cliente do segmento pediu para que a consultoria desenvolvesse um dashboard interativo e que gere insights relevantes para tomada de decisão. Além disso, solicitaram que fosse desenvolvido um modelo de Machine Learning para fazer o forecasting do preço do petróleo.

            Seu objetivo é:

            - Criar um dashboard interativo com ferramentas à sua escolha.
            - Seu dashboard deve fazer parte de um storytelling que traga insights relevantes sobre a variação do preço do petróleo, como situações geopolíticas, crises econômicas, demanda global por energia, etc. Isso pode te ajudar com seu modelo. É obrigatório que você traga pelo menos 4 insights neste desafio.
            - Criar um modelo de Machine Learning que faça a previsão do preço do petróleo diariamente (lembre-se de time series). Esse modelo deve estar contemplado em seu storytelling e deve conter o código que você trabalhou, analisando as performances do modelo.
            - Criar um plano para fazer o deploy em produção do modelo, com as ferramentas que são necessárias.
            - Faça um MVP do seu modelo em produção usando o Streamlit.
            """
        )


if __name__ == "__main__":
    main()
