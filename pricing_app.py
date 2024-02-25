import streamlit as st
import pandas as pd
import numpy as np

import plotly.express as px 
import matplotlib.pyplot as plt

from fetcher import Fetcher
from models import BSM, PricingMethod

# from matplotlib.backends.backend_agg import RendererAgg
# _lock = RendererAgg.lock

st.title("Option Pricing via Fourier Transform")

@st.cache_data
def get_historical_data(symbol: str):
    fetcher = Fetcher(symbol)
    return fetcher.get_stock_option_data()

def get_method(method: str, *args, **kwargs) -> PricingMethod:
    if method == "Black-Scholes-Merton":
        return BSM()
    elif method == "Fourier Transform":
        return BSM()
    else:
        raise Exception("Wrong method")

def get_volatility(data: pd.Series):
    daily_returns = data.pct_change().dropna()
    volatility = daily_returns.rolling(window=20).std().dropna()
    annaualized_volatility = volatility * np.sqrt(252)
    return annaualized_volatility

# Sidebar
st.sidebar.header("About")
st.sidebar.markdown(
    "Indipendent project developed by [Erik Pillon](https://ErikPillon.github.io) for the pricing evaluation of put-call options with different numerical and analytical methods."
)

option_types = ["Call Option", "Put Option"]
option = st.sidebar.selectbox("Select option type", option_types)

methods = ["Black-Scholes-Merton", "Fourier Transform"]
method = st.sidebar.selectbox("Select method", methods)

tickers = ["AAPL", "AMZN", "GOOG", "TSLA"]
ticker = st.sidebar.selectbox("Select ticker", tickers)

st.sidebar.header("Additional Resources")
st.sidebar.markdown(
    """
- [Option Pricing Specialization On Coursera]()
- [FFT visual explanation]()
- [FFT introduction]()

Inspiration for this work came from the wonderful work performed by [XYZ](https://github.com/)
"""
)


st.subheader('Time Series Performance')

data = get_historical_data(ticker)["Close"]
fig1 = px.line(data)

st.plotly_chart(fig1)

st.subheader('Option Price Evaluation')
number = st.number_input("Insert the strike value", value=data[-1]*1.1, placeholder="Type a number...")
st.write("currently selected: ", number)

st.markdown(f"Evaluating the {option} using the {method} method. Last price: {data[-1]} and selected strike price: {number}")

method = get_method(method)

volatility = get_volatility(data=data)
print(volatility)

st.write("The following values will be used for the evaluation:")
st.write("Today's price:", data[-1])
st.write("Strike price:", number)
st.write("Maturity:", 5)
st.write("Risk free rate:", 0)
st.write("Volatility:", volatility[-1])

price = method.price(option_type="call", S=data[-1], K=number, T=5, r=0.02, sigma=volatility[-1])

st.markdown(f"Option price: {price}")
