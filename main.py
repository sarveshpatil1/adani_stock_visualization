
import streamlit as st
from datetime import date

import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

START = "2017-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title('Stocks Forecast ')
st.write('Choose your favroite stock snd number of years for future prediction')
stocks = ('ADANIENT.NS', 'ADANIPORTS.NS', 'MSFT')
selected_stock = st.selectbox('Select stock ', stocks)

n_years = st.slider('Select the number of years:', 1, 4)
period = n_years * 365

@st.cache_data
def load_data(ticker):
    data=yf.download(ticker,START,TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state=st.text("Load data...")
data=load_data(selected_stock)
data_load_state.text("loading data... done! ")

st.subheader('Raw data')
st.write(data.tail())

def plot_rawdata():
    figure=go.Figure()
    figure.add_trace(go.Scatter(x=data['Date'],y=data['Open'],name='stock_open'))
    figure.add_trace(go.Scatter(x=data['Date'],y=data['Close'],name='stock_close'))
    figure.layout.update(title_text="Time Series data visualized",xaxis_rangeslider_visible=True,xaxis_title="Year", yaxis_title="Price")
    st.plotly_chart(figure)

plot_rawdata()

# Prediction
df_train=data[['Date','Close']]
df_train=df_train.rename(columns={"Date":"ds","Close":"y"})

m=Prophet()
m.fit(df_train)
future=m.make_future_dataframe(periods=period)
forecast=m.predict(future)

st.subheader('Forecast data')
st.write(forecast.tail())

st.subheader('Forecast Graph')
figure_forecast=plot_plotly(m,forecast)
figure_forecast.update_layout(autosize=False,width=700,height=500, xaxis_title="Year", yaxis_title="Price")
st.plotly_chart(figure_forecast)

st.subheader('Forecast Component')
figure_forecast_component=m.plot_components(forecast)
st.write(figure_forecast_component)