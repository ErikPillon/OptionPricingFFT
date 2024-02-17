import streamlit as st
import pandas as pd
import numpy as np

st.title("Option Pricing via Fourier Transform")



# Sidebar
st.sidebar.header("About")
st.sidebar.markdown(
    "Indipendent project developed by [Erik Pillon](https://ErikPillon.github.io) for the pricing evaluation of put-call options with different numerical and analytical methods."
)

st.sidebar.header("Furhter Resources")
st.sidebar.markdown(
    """
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Cheat sheet](https://docs.streamlit.io/library/cheatsheet)
- [Book](https://www.amazon.com/dp/180056550X) (Getting Started with Streamlit for Data Science)
- [Blog](https://blog.streamlit.io/how-to-master-streamlit-for-data-science/) (How to master Streamlit for data science)
"""
)
