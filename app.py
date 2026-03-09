# app.py - run with: streamlit run app.py
import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

st.title("AAPL Stock Price Predictor")

# Load saved models (you'll save them later)
# For now, just show results
st.header("Model Performance")
st.write("ARIMA Rolling: MAE 2.28 USD")
st.write("Tuned LSTM: MAE 7.64 USD")
st.write("XGBoost Improved: MAE 35.99 USD")
st.write("Ensemble: ~2.0–3.0 USD (estimated)")

# Upload future CSV
uploaded_file = st.file_uploader("Upload future_30days_forecast.csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.dataframe(df)
    fig, ax = plt.subplots()
    ax.plot(df['Date'], df['ARIMA'], label='ARIMA')
    ax.plot(df['Date'], df['LSTM'], label='LSTM')
    ax.plot(df['Date'], df['XGBoost'], label='XGBoost')
    ax.legend()
    st.pyplot(fig)