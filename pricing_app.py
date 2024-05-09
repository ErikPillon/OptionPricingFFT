import streamlit as st
import pandas as pd
import numpy as np

import plotly.express as px

from fetcher import Fetcher
from methods.base_method import BruteForce, Fourier, FFT, Exact, Method
from pricing_models.base_model import PricingModel
from pricing_models.models import BSM, VG, Heston

# from matplotlib.backends.backend_agg import RendererAgg
# _lock = RendererAgg.lock

st.title("Option Pricing via Transform Techniques")


@st.cache_data
def get_historical_data(symbol: str):
    fetcher = Fetcher(symbol)
    return fetcher.get_stock_option_data()


def get_model(model: str, **kwargs) -> PricingModel:
    if model == "Black-Scholes-Merton":
        return BSM()
    elif model == "Heston":
        return Heston()
    elif model == "Variance-Gamma":
        return VG()
    else:
        raise Exception("Wrong method")


def get_method(method: str) -> Method:
    # selector for method
    if method == "exact":
        return Exact
    elif method == "fft":
        return FFT
    elif method == "fourier":
        return Fourier
    elif method == "brute-force":
        return BruteForce
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

st.sidebar.header("Model Parameters")
st.sidebar.markdown(
    "Use the sidebar selectors below to adjust the parameters of the model."
)

option_types = ["Call Option", "Put Option"]
option = st.sidebar.selectbox("Select option type", option_types)

models = ["Black-Scholes-Merton", "Heston", "Variance-Gamma"]
model = st.sidebar.selectbox("Select model", models)

methods = ["exact", "brute-force", "fourier", "fft"]
method = st.sidebar.selectbox("Select method", methods)

tickers = [
    "AAPL",
    "AMZN",
    "GOOG",
    "TSLA",
    "MSFT",
    "FB",
    "NVDA",
    "JPM",
    "JNJ",
    "UNH",
    "MA",
    "PFE",
    "NFLX",
    "BABA",
    "PG",
    "V",
    "MA",
    "HD",
    "DIS",
    "PFE",
    "BAC",
    "T",
]

ticker = st.sidebar.selectbox("Select ticker", tickers)

maturity = st.sidebar.slider("Maturity (years)", 0.0, 20.0, 5.0)  # min, max, default

st.sidebar.header("Additional Resources")
st.sidebar.markdown(
    """
- [Option Pricing Specialization On Coursera](https://www.coursera.org/learn/financial-engineering-computationalmethods)
- [FFT visual explanation]()
- [FFT introduction]()

Inspiration for this work came from the wonderful work performed by [Nikola Krivacevic](https://github.com/mcf-long-short/option-pricing-fourier-transform/tree/main)
"""
)


st.subheader("Time Series Performance")

data = get_historical_data(ticker)["Close"]
fig1 = px.line(data, title=f"Time Series Performance ({ticker})")

st.plotly_chart(fig1)

st.subheader("Option Price Evaluation")
number = st.number_input(
    "Insert the strike value", value=data[-1] * 1.1, placeholder="Type a number..."
)
st.write("currently selected: ", number)

st.markdown(
    f"Evaluating the {option} using the {method} method. Last price: {data[-1]} and selected strike price: {number}"
)

calc_method = get_method(method)

calc_model = get_model(model)

volatility = get_volatility(data=data)

st.write("The following values will be used for the evaluation:")
st.write("Today's price:", data[-1])
st.write("Strike price:", number)
st.write("Maturity:", maturity)
st.write("Risk free rate:", 0.02)
st.write("Volatility:", volatility[-1])

with st.expander("Notes about the volatility calculation"):
    st.markdown("""
* Volatility is calculated using the historical data of the asset.
* The annualized volatility of the last 20 returns of the historical data is used for the evaluation.

See also:
 * [Historical Volatility vs. Implied Volatility](https://www.investopedia.com/ask/answers/060115/how-implied-volatility-used-blackscholes-formula.asp#:~:text=Plugging%20the%20option%27s%20price%20into,implied%20by%20the%20option%20price.)
""")


def get_price():
    calculation_method = calc_method(
        model=calc_model,
        S0=data[-1],
        K=number,
        T=maturity,
        r=0.02,
        sig=volatility[-1],
    )
    if option == "Call Option":
        price = calculation_method.get_call()
    else:
        price = calculation_method.get_put()
    return price


price = get_price()
st.header("Results")

print(method, calc_model, number, maturity, volatility[-1])
st.subheader(f"Option price: {price}")

if method == "Black-Scholes-Merton":
    st.write("**Some notes on the Black-Scholes-Merton model**")
    st.write(
        "The Black-Scholes-Merton (BSM) model formula for a European call option is given by:"
    )
    st.latex(r"""
    \begin{equation*}
    C = S_0 \cdot N(d_1) - X \cdot e^{-rT} \cdot N(d_2)
    \end{equation*}
    """)
    st.write("where:")
    st.latex(r"""
    \begin{align*}
    S_0 &\text{ is the current stock price,} \\
    X &\text{ is the strike price,} \\
    T &\text{ is the time to maturity,} \\
    r &\text{ is the risk-free interest rate,}
    \end{align*} 
    """)
    st.write("and:")
    st.latex(r"""
    N(d_1) \text{ and } N(d_2) \text{ are the cumulative distribution functions of the standard normal distribution, where}
    \\
    d_1 = \frac{\ln\left(\frac{S_0}{X}\right) + \left(r + \frac{\sigma^2}{2}\right)T}{\sigma\sqrt{T}} \\
    d_2 = d_1 - \sigma\sqrt{T}
    """)
