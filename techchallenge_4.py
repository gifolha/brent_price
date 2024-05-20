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
from datetime import timedelta

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

        # Plotando os dados originais e as previsões
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

# Função para prever o modelo ARIMA para os próximos 30 dias
def arima_forecast_future(data, future_dates):
    # Convertendo a coluna 'Date' para datetime
    data['Date'] = pd.to_datetime(data['Date'])

    # Definindo a coluna 'Date' como índice
    data.set_index('Date', inplace=True)

    # Ajustando o modelo ARIMA
    model = ARIMA(data['Close'], order=(5,1,0))
    fit_model = model.fit()

    # Realizando previsão para os próximos 30 dias
    forecast = fit_model.forecast(steps=30)

    # Criando dataframe com as datas futuras e as previsões
    forecast_dates = [data.index[-1] + timedelta(days=i) for i in range(1, 31)]
    forecast_df = pd.DataFrame({'Date': forecast_dates, 'Forecast_ARIMA': forecast})

    return forecast_df

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

    menu = ["Navegação", "Challenge", "Narrativa Temporal - Histórico Geopolítico", "Insights", "Dashboard", "Carregar Dados", "Visualizar Dados", "Decomposição", "Previsão ARIMA", "Previsão Prophet", "Previsão LSTM", "DashBoard Interativa", "Previsões e Modelo de Machine Learning", "Conclusão", "Informações Importantes"]
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
        st.markdown(
            """
            Fazer previsão usando o modelo ARIMA (AutoRegressive Integrated Moving Average) envolve prever valores futuros com base em padrões observados em dados históricos de uma série temporal. Para utilizar essa ferramenta:
            
            - Precisamos entender a série temporal: Antes de aplicar o modelo ARIMA, é importante entender a série temporal que será analizada. Isso inclui entender sua tendência, sazonalidade e qualquer padrão cíclico.
            - Dividir os dados: Separar os dados em conjuntos de treinamento e teste. O conjunto de treinamento será usado para treinar o modelo, enquanto o conjunto de teste será usado para avaliar o desempenho da previsão.
            - Identificar os parâmetros ARIMA: Determinar os parâmetros do modelo ARIMA, que incluem a ordem dos termos autoregressivos (p), a ordem de diferenciação (d) e a ordem dos termos de média móvel (q).
            - Ajustar o modelo ARIMA: Aplique o modelo ARIMA aos dados de treinamento. Isso envolve estimar os parâmetros do modelo com os dados de treinamento.
            - Validar o modelo: Avalie o desempenho do modelo usando os dados de teste. Isso pode ser feito comparando as previsões do modelo com os valores reais e calculando métricas de erro, como RMSE (Root Mean Squared Error) ou MAE (Mean Absolute Error).
            - Fazer previsões: Use o modelo ajustado para fazer previsões sobre os valores futuros da série temporal.
            
            O ARIMA é um modelo popular para previsão de séries temporais devido à sua simplicidade e eficácia em capturar padrões temporais em dados históricos. 
            """
        )
        forecast_data = arima_forecast(data)
        if forecast_data is not None:
            plot_time_series(forecast_data, title="Previsão ARIMA")
            
    elif choice == "Previsão Prophet":
        st.subheader("Previsão Prophet")
        st.markdown(
            """
            O Prophet é especialmente útil para previsões em séries temporais que exibem tendências sazonais e feriados, pois ele pode modelar automaticamente esses padrões sem a necessidade de muita intervenção manual. Ele também lida bem com dados ausentes e outliers.
            """
        )
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
        st.markdown(
            """
            A LSTM (Long Short-Term Memory) é um tipo de rede neural recorrente (RNN) muito utilizada em problemas de previsão em séries temporais, especialmente quando se trata de dados sequenciais com dependências temporais de longo prazo. Ela é capaz de lembrar informações por longos períodos de tempo, permitindo assim capturar padrões complexos em séries temporais.
            """
        )
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
    elif choice == "Narrativa Temporal - Histórico Geopolítico":
        st.subheader("Narrativa Temporal - Histórico Geopolítico")
        container = st.container(border=True)
        container.write(
            """
            * *2008, o ano em que o petróleo enlouqueceu o mercado:*

            31/12/08 - 12h33 - Atualizado em 31/12/08 - 12h35
            
            O mercado do petróleo viveu em 2008 um drama em dois atos, marcado pela superação da barreira dos 100 dólares o barril e uma disparada meteórica dos preços até 147,50 dólares.
            Depois disso, os preços começaram a despencar mais rapidamente do que haviam subido, ficando a 39,35 dólares em Londres, no início de dezembro.
            No primeiro semestre, uma soma de fatores levou os preços às alturas: tensões geopolíticas, do Irã à Nigéria passando pelo Paquistão; o equilíbrio tenso entre uma oferta limitada e uma demanda puxada pelos países emergentes; a conscientização de que as reservas são limitadas e de acesso cada vez mais difícil; uma febre dos fundos de investimento por matérias-primas.
            Depois da falência do banco americano Lehman Brothers em setembro, esta lógica se inverte. Temendo a deflação, os investidores abandonam o petróleo, porque precisam urgentemente de liquidez.
            
            
            Fonte: https://g1.globo.com/Noticias/Economia_Negocios/0,,MUL940136-9356,00-O+ANO+EM+QUE+O+PETROLEO+ENLOUQUECEU+O+MERCADO.html

            """
        )
        container.write(
            """
            * *Brent cai abaixo de US$90; tem menor nível desde 2010:*

            Publicado em 10 de outubro de 2014 às 10h13.
            
            Arábia Saudita disse que elevou a produção no mês passado, aumentando especulação sobre uma guerra de preços na Organização dos Países Exportadores de Petróleo.
            O petróleo Brent diminuiu para o menor patamar desde 2010, operando abaixo de 90 dólares o barril, após a Arábia Saudita dizer que elevou a produção, aumentando especulação sobre uma guerra de preços na Organização dos Países Exportadores de Petróleo (Opep), levantando dúvidas se o maior exportador do mundo estará preparado para tomar medidas unilaterais.
            O Brent para entrega em novembro caía 0,71 dólar, a 89,34 dólares o barril, por volta das 8h50 (horário de Brasília), depois de cair mais cedo para 88,11 dólares, seu nível mais baixo desde dezembro de 2010.
            O petróleo nos EUA recuava no mesmo horário quase 1 dólar, a 84,79 dólares o barril. Também conhecido como West Texas Intermediate (WTI), a commodity norte-americana atingiu uma mínima de 83,59 dólares, seu nível mais baixo desde julho de 2012.
            A produção de petróleo também tem aumentado em outros países membros da Opep, como Iraque e Líbia, disse o grupo em seu relatório mensal de mercado nesta sexta-feira, apesar da violência e da instabilidade em ambos os países.
            
            
            Fonte: https://exame.com/economia/brent-cai-abaixo-de-us-90-tem-menor-nivel-desde-2010/

            """
        )
        container.write(
            """
            * *Petróleo Brent cai para cerca de US$ 37 o barril, perto de mínima de 11 anos* 
            
            28/12/2015 10h53 - Atualizado em 28/12/2015 10h59
            
            Petróleo operava em baixa, com o Brent sendo negociado a cerca de USS 37 por barril, perto de uma mínima de 11 anos, sob pressão do excesso de oferta que reduziu em mais da metade o valor da commodity desde meados de 2014.
            O petróleo nos EUA (WTI) era negociado próximo da paridade com o Brent, tendo registrado um prêmio no início em dezembro ante a referencial global pela primeira vez em cerca de um ano.
            O petróleo Brent recuava USS 0,87, ou 2,3%, a USS 37,02 por barril, às 10:29 (horário de Brasília). O petróleo dos Estados Unidos caía US$ 1,18, ou 3,1%, a USS 36,92 por barril.
            O Brent registrou uma mínima de 11 anos de USS 35,98 na última terça-feira (22/12/2015).


            Fonte: https://g1.globo.com/economia/mercados/noticia/2015/12/petroleo-brent-cai-para-cerca-de-us37-o-barril-perto-de-minima-de-11-anos.html
            
            """
        )
        container.write(
            """
            * *Petróleo fecha no nível mais alto em quatro anos*
            
            24/09/2018 07h52  Atualizado há 5 anos
            
            O preço do petróleo Brent alcançou seu nível mais alto em quatro anos, após a decisão da Opep e de seus sócios de não aumentar a produção apesar das pressões de Donald Trump. A cotação do Brent do Mar do Norte subiu USS 2,40, ou 3,1%, a USS 81,20 o barril, depois de chegar na máxima a USS 81,39, segundo a agência Reuters.
            O petróleo dos EUA (WTI) teve alta de US$ 1,30, ou 1,8%, a USS 72,08 o barril.
            Apesar da elevação, o preço do barril do Brent ainda está longe de sua máxima histórica. Em julho de 2008, o barril chegou a ser negociado a USS 145,61.
            
            O barril do petróleo Brent chegou a USS 80 neste mês, levando Trump a reiterar seu pedido para que a Opep baixasse os preços. A alta das cotações foi contida principalmente por uma queda nas exportações do Irã, membro da Opep, devido ao restabelecimento de sanções dos EUA.
            "Nós protegemos os países do Oriente Médio, eles não estariam seguros por muito tempo sem nós e, ainda assim, eles continuam incentivando preços cada vez mais altos para o petróleo. Nós lembraremos disso. O monopólio da Opep deve baixar os preços agora!", escreveu Trump em sua conta no Twitter.
            O ministro da Energia saudita, Khalid al-Falih disse que a Arábia Saudita tinha capacidade para aumentar a produção de petróleo, mas que a medida não seria necessária no momento. "A minha informação é que os mercados estão sendo adequadamente abastecidos. Não sei de nenhuma refinaria no mundo que esteja precisando de petróleo e não esteja conseguindo", afirmou.
            A alta do petróleo também tem sido sustentada pela perspectiva de menores exportações do Irã, terceiro maior produtor da Opep, devido a sanções dos EUA.
            Segundo tradings de commodities, os preços do petróleo podem subir para 100 dólares o barril ao final do ano ou no início de 2019 com o impacto de sanções ao Irã.
            
            
            Fonte: https://g1.globo.com/economia/noticia/2018/09/24/petroleo-sobe-para-quase-us-81-nivel-mais-alto-em-4-anos.ghtml

            """
        )
        container.write(
            """
            * *Pandemia faz preço do barril de petróleo fechar ano 20% mais barato*

            31.dez.2020 (quinta-feira) – 18h53
            Atualizado: 7.jan.2021 (quinta-feira) – 22h41

            O barril do petróleo encerrou 2020 mais barato do que começou. Os contratos futuros do petróleo Brent fecharam dia 31 dez 2020 em 51.80 dólares, uma queda de 21,5% em comparação ao preço de 2019 (66 dólares). A desvalorização do insumo eh uma das marcas de um ano difícil para o setor petroleiro. Levantamento divulgado pelo The Wall Street Jornal mostra que o segmento sofreu depreciação de aproximadamente 145 bilhões nos 3 primeiros trimestres de 2020.
            A redução pela demanda em consequência do isolamento social – principal medida para evitar a disseminação do novo coronavírus – e seus impactos econômicos foram sentidos, principalmente, em fevereiro na Europa e em marco nos Estados Unidos. Na comparação dos fechamentos anuais, porém, observasse que a tendencia não eh de alta equivalente a do inicio da década.
            Segundo o presidente da Inter B consultoria, isso se explica em parte pelo processo de transição energética. O que significa uma crescente busca por fontes de energia limpas, em especial a 


            Fonte: https://www.poder360.com.br/economia/pandemia-faz-preco-do-barril-de-petroleo-fechar-ano-20-mais-barato/#:~:text=O%20barril%20do%20petr%C3%B3leo%20encerrou,US%24%2066%2C00
            
            """
        )
        container.write(
            """
            * *Petróleo encerra o ano em alta superior a 21%, no maior valor em quase 10 anos*

            30/12/2022 17h51  Atualizado há um ano
            
            Os preços do petróleo fecharam o último pregão do ano em alta em um movimento de reposicionamento em um dia de baixa liquidez, que sempre acaba por provocar movimentos mais fortes. As cotações também se recuperam depois dos recuos registrados ontem depois que estoques americanos se mostraram maiores que o esperado. No ano, o petróleo terminou acumulando ganhos expressivos: o Brent (referência mundial) saltou 21,13% enquanto o WTI (referência norte-americana) subiu 12,80% em 2022.
            Considerando os preços de fechamento anuais, este é o maior valor para o Brent e para o WTI desde 2013, quando o barril alcançou USS 109,95 e USS 98,17, respectivamente, numa sequência de altas anuais em função das ondas revolucionárias no Oriente Médio (conhecida como a Primavera Árabe), já que a guerra civil paralisou a atividade econômica - inclusive as exportações de petróleo - nesses países.
            
            No período de 2010 a 2013, o mundo também atravessou o aumento da demanda mundial pela commodity com o reaquecimento das economias após a crise de 2008.
            Nesta sexta-feira (30), os futuros do Brent para março subiram 2,93% a US$ 85,91 enquanto os futuros do WTI para fevereiro fecharam em alta de 2,4% a USS80,26. Apesar dos ganhos significativos no ano, os preços terminam 2022 bem longe dos mais de USS 120 por barril registrados durante o pico da crise da guerra da Ucrânia.
            “Embora os fundamentos para o petróleo permaneçam fortes, a percepção de destruição de demanda provocada pelo aumento dos casos de covid na China e pela alta dos juros dos EUA deixarão muitos investidores de lado no início do ano”, afirmaram os analistas do BOK Financial, em nota.
            “Enquanto os produtores finalmente atingiram a demanda pós-pandêmica, outros riscos permanecem no próximo ano, principalmente em relação à produção russa em meio ao novo teto de preço e suas ameaças de cortar a produção e não fornecer a nenhum país que o cumpra. Isso não é um problema agora, mas se os preços começarem a subir, isso pode acelerar o movimento rapidamente”, disse o analista da Oanda, Craig Erlam.
            
            
            Fonte: https://valorinveste.globo.com/mercados/internacional-e-commodities/noticia/2022/12/30/petroleo-encerra-o-ano-em-alta-superior-a-21percent-no-maior-valor-em-quase-10-anos.ghtml

            """
        )
        container.write(
            """
            * *Petróleo cai ao menor nível desde 2021*

            15.mar.2023 (quarta-feira) - 13h19

            O preço do barril de petróleo tipo Brent cai 5,72 nesta 4ª feira (15 mar 2023) puxado pela expectativa de menor demanda pelo consumo do mundo. Há duvidas sobre o rumo do setor financeiro global depois das falências dos bancos Silicon Valley Bank e Signature Bank nos Estados Unidos.
            
            FALÊNCIA DE BANCOS NOS EUA 
            O Departamento de Proteção Financeira e Inovação da Califórnia anunciou na 6ª feira (10.mar) o fechamento do SVB (Silicon Valley Bank). No domingo (12.mar.2023), o Signature Bank encerrou as atividades. O SVB informou na 4ª feira (8.mar) que havia liquidado USS 21 bilhões em títulos, com USS 1,8 bilhão em prejuízo no 1º trimestre de 2023. Também estudava a venda de USS 1,7 bilhão em ações. 
            Houve uma corrida de clientes para sacar os depósitos do banco. Esses recursos retirados estavam investidos em outros ativos, de menor liquidez. Ou seja, a instituição não conseguiu atender aos pedidos de saques. O Fed (Federal Reserve, o Banco Central dos EUA) disse que vai restituir os clientes com depósitos no banco. Reservou USS 25 bilhões para os pagamentos através de um fundo de garantia do Tesouro.

            Fonte: https://www.poder360.com.br/economia/petroleo-cai-ao-menor-nivel-desde-2021/

            """
        )
        container.write(
            """
            * *Petróleo sobe e bolsas da Ásia têm queda após explosão no Irã; veja os impactos econômicos*

            Publicado em 19/04/2024 às 09h39
            
            Em meio a tensão com Israel, relatos de explosão no Irã na noite desta quinta-feira, 18, já começaram a movimentar o mercado internacional. Os impactos econômicos do ataque próximo ao aeroporto da cidade iraniana de Isfahan, na região central do país, menos de uma semana depois de Teerã lançar um ataque em direção ao território israelense afetaram o mercado petroleiro, a cotação do dólar frente ao iene, juros dos treasuries e as bolsas da Ásia.
            Autoridades militares israelenses disseram ao The New York Times que o país atingiu o Irã, mas nem o Exército, nem o governo se pronunciaram oficialmente.
            A crise escala desde que a representação diplomática de Teerã em Damasco, na Síria, foi alvo do ataque que matou comandantes da Guarda Revolucionária Iraniana, no começo do mês. O Irã culpou Israel e retaliou com mais de 300 mísseis e drones no último fim de semana, elevando o risco de uma guerra aberta no Oriente Médio.
            
            Os contratos futuros de petróleo operam em forte alta neste fim de noite, impulsionados por relatos de explosão no Irã. Na Intercontinental Exchange, o barril do petróleo Brent para junho subia às 23h37 (de Brasília) 4,10%, a USS 90,68. Na New York Mercantile Exchange (Nymex), o barril do WTI para o mesmo mês saltava 4,205, a USS 85,55.
            As explosões também alimentam uma busca por ativos de segurança no mercado financeiro internacional. As cotações do iene, dos títulos longos do Tesouro dos Estados Unidos e do ouro operam em alta forte.
            
            
            Fonte: https://www.udop.com.br/noticia/2024/04/19/petroleo-sobe-e-bolsas-da-asia-tem-queda-apos-explosao-no-ira-veja-os-impactos-econ-ocirc-micos.html

            """
        )

    elif choice == "Insights":
        st.subheader("Insights")
        st.image('resumo.png', caption='Insights Baseados No Histórico Geopolítico')
        st.write(
            """
            * Crise de 2008: 
            Tensões geopolíticas, do Irã à Nigéria passando pelo Paquistão; o equilíbrio tenso entre uma oferta limitada e uma demanda puxada pelos países emergentes; a conscientização de que as reservas são limitadas e de acesso cada vez mais difícil; uma febre dos fundos de investimento por matérias-primas.
            Depois da falência do banco americano Lehman Brothers em setembro, esta lógica se inverte. Temendo a deflação, os investidores abandonam o petróleo, porque precisam urgentemente de liquidez.

            * Menores preços em 2010 devido a Arábia Saudita ter aumentado a produção:
            O petróleo Brent diminuiu para o menor patamar desde 2010, operando abaixo de 90 dólares o barril, após a Arábia Saudita dizer que elevou a produção, aumentando especulação sobre uma guerra de preços na Organização dos Países Exportadores de Petróleo (Opep), levantando dúvidas se o maior exportador do mundo estará preparado para tomar medidas unilaterais.
            A produção de petróleo também tem aumentado em outros países membros da Opep, como Iraque e Líbia, disse o grupo em seu relatório mensal de mercado nesta sexta-feira, apesar da violência e da instabilidade em ambos os países.

            * 2010-2013 período que aumentou a demanda mundial pela commodity, reaquecimento do mercado após 2008:
            No período de 2010 a 2013, o mundo atravessou o aumento da demanda mundial pela commodity com o reaquecimento das economias após a crise de 2008.
           
            * Menor preço do Brent em 10 anos (2015) - excesso de oferta que reduziu mais da metade do preço da commodity desde meados de 2014:
            Petróleo operava em baixa, com o Brent sendo negociado a cerca de USS 37 por barril, perto de uma mínima de 11 anos, sob pressão do excesso de oferta que reduziu em mais da metade o valor da commodity desde meados de 2014.

            * Brent chega ao maior preço em 5 anos (2018) Opep não quis aumentar da pressão do governo americano:
            O preço do petróleo Brent alcançou seu nível mais alto em quatro anos, após a decisão da Opep e de seus sócios de não aumentar a produção apesar das pressões de Donald Trump. 

            * Brent diminui por causa da pandemia:
            O barril do petróleo encerrou 2020 mais barato do que começou. Os contratos futuros do petróleo Brent fecharam dia 31 dez 2020 em 51.80 dólares, uma queda de 21,5% em comparação ao preço de 2019 (66 dólares). A desvalorização do insumo eh uma das marcas de um ano difícil para o setor petroleiro. Levantamento divulgado pelo The Wall Street Jornal mostra que o segmento sofreu depreciação de aproximadamente 145 bilhões nos 3 primeiros trimestres de 2020.
            A redução pela demanda em consequência do isolamento social – principal medida para evitar a disseminação do novo coronavírus – e seus impactos econômicos foram sentidos, principalmente, em fevereiro na Europa e em marco nos Estados Unidos. Na comparação dos fechamentos anuais, porém, observasse que a tendencia não eh de alta equivalente a do inicio da década.
            Segundo o presidente da Inter B consultoria, isso se explica em parte pelo processo de transição energética. O que significa uma crescente busca por fontes de energia limpas, em especial a 

            * Petróleo em Alta, maior valor em 10 anos (2022):
            No período de 2010 a 2013, o mundo também atravessou o aumento da demanda mundial pela commodity com o reaquecimento das economias após a crise de 2008.
            Nesta sexta-feira (30), os futuros do Brent para março subiram 2,93% a US$ 85,91 enquanto os futuros do WTI para fevereiro fecharam em alta de 2,4% a USS80,26. Apesar dos ganhos significativos no ano, os preços terminam 2022 bem longe dos mais de USS 120 por barril registrados durante o pico da crise da guerra da Ucrânia.
            “Embora os fundamentos para o petróleo permaneçam fortes, a percepção de destruição de demanda provocada pelo aumento dos casos de covid na China e pela alta dos juros dos EUA deixarão muitos investidores de lado no início do ano”, afirmaram os analistas do BOK Financial, em nota.
            “Enquanto os produtores finalmente atingiram a demanda pós-pandêmica, outros riscos permanecem no próximo ano, principalmente em relação à produção russa em meio ao novo teto de preço e suas ameaças de cortar a produção e não fornecer a nenhum país que o cumpra. Isso não é um problema agora, mas se os preços começarem a subir, isso pode acelerar o movimento rapidamente”, disse o analista da Oanda, Craig Erlam.

            * 2024 - momento incerto. Quedas no preço por causa das explosões do Irã - instabilidade econômica e política:
            Em meio a tensão com Israel, relatos de explosão no Irã já começaram a movimentar o mercado internacional. Os impactos econômicos do ataque próximo ao aeroporto da cidade iraniana de Isfahan, na região central do país, menos de uma semana depois de Teerã lançar um ataque em direção ao território israelense afetaram o mercado petroleiro, a cotação do dólar frente ao iene, juros dos treasuries e as bolsas da Ásia.
            A crise escala desde que a representação diplomática de Teerã em Damasco, na Síria, foi alvo do ataque que matou comandantes da Guarda Revolucionária Iraniana, no começo do mês. O Irã culpou Israel e retaliou com mais de 300 mísseis e drones no último fim de semana, elevando o risco de uma guerra aberta no Oriente Médio.
            Os contratos futuros de petróleo operam em forte alta neste fim de noite, impulsionados por relatos de explosão no Irã. Na Intercontinental Exchange, o barril do petróleo Brent para junho subia às 23h37 (de Brasília) 4,10%, a USS 90,68. Na New York Mercantile Exchange (Nymex), o barril do WTI para o mesmo mês saltava 4,205, a USS 85,55.
            As explosões também alimentam uma busca por ativos de segurança no mercado financeiro internacional. As cotações do iene, dos títulos longos do Tesouro dos Estados Unidos e do ouro operam em alta forte.
            
            """
        )

    elif choice == "Previsões e Modelo de Machine Learning":
        st.subheader("Previsões e Modelo de Machine Learning")

    # Plotando a série temporal original
        fig, ax = plt.subplots(figsize=(15, 8))
        ax.plot(data['Date'], data['Close'], label='Close Price (Original)', color='black')

    # Previsões ARIMA
        arima_data = arima_forecast(data)
        if arima_data is not None:
            ax.plot(arima_data['Date'], arima_data['Forecast_ARIMA'], label='ARIMA Forecast', color='red')
    
        # Previsões Prophet
        prophet_data = prophet_forecast(data)
        ax.plot(prophet_data['ds'], prophet_data['yhat'], label='Prophet Forecast', color='blue')
    
        # Previsões LSTM
        lstm_data = lstm_forecast(data)
        ax.plot(lstm_data['Date'], lstm_data['Forecast_LSTM'], label='LSTM Forecast', color='green')
    
        ax.set_xlabel('Date')
        ax.set_ylabel('Close Price')
        ax.set_title('Comparação das Previsões ARIMA, Prophet e LSTM')
        ax.legend()
        st.pyplot(fig)


        # Criando a tabela com os dados reais e as previsões
        st.subheader("Tabela de Previsões")
        if arima_data is not None:  # Verificando se arima_data não é None antes de usá-lo
            predictions_table = pd.DataFrame({
                'Data': data['Date'],
                'Close': data['Close'],
                'ARIMA': arima_data['Forecast_ARIMA'],  # Usando a coluna correta do arima_data
                'Erro ARIMA': data['Close'] - arima_data['Forecast_ARIMA'],
                'LSTM': lstm_data['Forecast_LSTM'],
                'Erro LSTM': data['Close'] - lstm_data['Forecast_LSTM'],
                'Prophet': prophet_data['yhat'],
                'Erro Prophet': data['Close'] - prophet_data['yhat']
            })
    
            # Ordenando o DataFrame pela coluna de datas em ordem decrescente
            predictions_table = predictions_table.sort_values(by='Data', ascending=False)
    
            # Aplicando estilos condicionais para destacar as colunas desejadas
            def highlight_close(val):
                color = 'lightgreen' if val == data['Close'].max() else 'white'
                return f'background-color: {color}'
    
            def highlight_forecast(val):
                color = 'lightcoral' if val != data['Close'].max() else 'white'
                return f'background-color: {color}'
    
            predictions_table_styled = predictions_table.style.applymap(highlight_close, subset=['Close']).applymap(highlight_forecast, subset=['ARIMA', 'LSTM', 'Prophet'])
    
            st.write(predictions_table_styled)

    elif choice == "Conclusão":
        st.subheader("Conclusão")
        st.write(
            """
            Os modelos Arima e LSTM se mostraram muito eficientes para prever os dados com base nos dados históricos da base do Brent.
            O ARIMA é um modelo popular para previsão de séries temporais devido à sua simplicidade e eficácia em capturar padrões temporais em dados históricos.
            A LSTM (Long Short-Term Memory) é um tipo de rede neural recorrente (RNN) muito utilizada em problemas de previsão em séries temporais, especialmente quando se trata de dados sequenciais com dependências temporais de longo prazo. Ela é capaz de lembrar informações por longos períodos de tempo, permitindo assim capturar padrões complexos em séries temporais.

            Analisar os dados de uma base como a base histórica do preço em dólar do Brent é uma tarefa complicada, pois existem muitos fatores e variáveis que podem influenciar na queda e aumento do preço.
            Como podemos analisar geopoliticamente, o preço do Brent pode aumentar ou diminuir devido a tensões políticas, guerras, eleições, alta ou baixa demanda e pouca ou muita produção.

            Neste caso, o modelo ARIMA se mostrou um pouco melhor do que o LSTM, porém os dois modelos se sairam muito bem em prever os dados.

            O Prophet é especialmente útil para previsões em séries temporais que exibem tendências sazonais e feriados, pois ele pode modelar automaticamente esses padrões sem a necessidade de muita intervenção manual. Ele também lida bem com dados ausentes e outliers.
            No modelo Prophet, conseguimos enxergar que os dados foram previstos acima do que os dados reais de fechamento, o que pode mostrar uma tendência de que o preço do Brent pode vir a subir, se comparado com a tendência histórica. 
            Os dados gerados pelo Prophet são bons para ser usados em um contexto geral de predição futura analisando apenas se a tendência é de queda ou crescimento do preço.

            No caso do ARIMA e LSTM, conseguiriamos ter uma resposta imediata (predição mais curta) mais acertiva.

            Concluindo que neste projeto, o modelo ARIMA foi o que se comportou sendo o mais acertivo.
            """
        )

    elif choice == "Informações Importantes":
        st.subheader("Informações Importantes")
        st.write(
            """
            Data Analytics - 2DTAT

            FASE 4 - DATA VIZ AND PRODUCTION MODELS

            Tech Challenge #4 - Consultoria IPEA (analisar os dados de preço do petróleo Brent)
            
            Grupo 46 - Giovanna Folha Carlomagno, Henrique Estevão

            Código Colab - dataviz, data manipulation, model training, complementary information:
            https://colab.research.google.com/drive/1xHyhS7xYwSHL1e_bUYWCJEwRyTxTdgQi#scrollTo=aZlPmBbQcd4j

            Link direto para o projeto no GitHub:
            https://github.com/gifolha/brent_price

            Link para esse app do Streamlit:
            https://brentprice-tech4.streamlit.app/

            Link para a Dashboard:
            https://lookerstudio.google.com/u/0/reporting/aaa5dd57-d85c-491a-bc87-a7b85e56e405/page/kurzD
            
            Link para o Colab do Dashboard:
            https://drive.google.com/file/u/0/d/1dNt9kT8-fPNL_Q-ZHF_jEw6PcIt_5O68/edit
            """
        )

    elif choice == "Dashboard":
        st.subheader("DashBoard Interativa")
        st.write("""
        - Dashboard link: https://lookerstudio.google.com/u/0/reporting/aaa5dd57-d85c-491a-bc87-a7b85e56e405/page/kurzD
        
        - Dashboard colab: https://drive.google.com/file/u/0/d/1dNt9kT8-fPNL_Q-ZHF_jEw6PcIt_5O68/edit

        Essa é a dashboard interativa do projeto, desenvolvida para fornecer uma visão detalhada e dinâmica de dois indicadores econômicos fundamentais: o histórico do preço do Brent e o histórico do valor do dólar.
        
        O gráfico do histórico do preço do Brent permite acompanhar as variações no valor do barril de petróleo ao longo do tempo, facilitando a análise de tendências e a compreensão dos fatores que influenciam os preços no mercado global de energia.
        
        Já o gráfico do histórico do dólar oferece uma perspectiva sobre as flutuações cambiais da moeda norte-americana, essencial para entender seu impacto em diversas áreas da economia, incluindo importações, exportações e investimentos.
        
        Utilize essa ferramenta para explorar os dados de forma interativa e obter insights valiosos para suas análises e tomadas de decisão.
        """)
        st.image('dash.jpeg', caption='Dashboard Brent x Dólar')

        #aguardando confirmação se será necessário inserir dashboard ou não
    
    elif choice == "Navegação":
        st.subheader("Navegação")
        st.write(
            """
            Projeto #4 - Tech Challenge DATA VIZ AND PRODUCTION MODELS
            """
        )
        container = st.container(border=True)
        container.write(
                """
                Como navegar nesse Streamlit:

                Você está na aba "Navegação"!
            
                * Challenge: aqui você vai encontrar as regras do challenge, todos os requisitos. Basicamente a "regra do negócio".
                * Narrativa Temporal - Histórico Geopolítico: aqui você vai encontrar o histórico dos dados, ocorrências geopolíticas e o cenário histórico das flutuações do preço do Brent.
                * Insights - aqui você vai encontrar os insights gerados com base na pesquisa geopolítica e com os dados históricos retirados da base do yfinance.
                * Dashboard - aqui você vai encontrar a dashboard.
                * Carregar Dados - aqui você vai encontrar o primeiro contato com os dados, a tabela com as informações brutas retiradas do YFINANCE para o Brent.
                * Visualizar Dados - aqui você vai encontrar a primeira visualização dos dados, com o fechamento do preço do Brent e dados. Apenas um snapshot do contexto geral da base.
                * Decomposição - aqui você vai encontrar a decomposição dos dados.
                * Previsão ARIMA - aqui você vai encontrar a primeira previsão dos dados, utilizando o modelo de Machine Learning ARIMA.
                * Previsão Prophet - aqui você vai encontrar a segunda previsão dos dados, utilizando o modelo de Machine Learning Prophet.
                * Previsão LSTM - aqui você vai encontrar a terceira previsão dos dados, utilizando o modelo de Machine Learning LSTM.
                * Previsões e Modelo de Machine Learning - aqui você vai encontrar alguns dados e comparações entre os modelos ARIMA, PROPHET e LSTM.
                * Conclusão - aqui você vai encontrar a conclusão deste projeto.
                * Informações Importantes - seção para os professores da pós tech encontrarem links do github e colab com o restante do projeto! :)
                """
            )


if __name__ == "__main__":
    main()
