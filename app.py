import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta

# ────────────────────────────────────────────────
# Page config & title
# ────────────────────────────────────────────────
st.set_page_config(page_title="AAPL Stock Prediction", layout="wide")
st.title("AAPL Stock Price Prediction Dashboard")
st.markdown("Built with ARIMA, LSTM, and XGBoost | Portfolio demo by Fridah")

# ────────────────────────────────────────────────
# Sidebar - Controls & Info
# ────────────────────────────────────────────────
with st.sidebar:
    st.header("Project Info")
    st.write("Train period: 2010–2022")
    st.write("Test period: 2022–2026")
    st.write("Models:")
    st.write("- ARIMA (rolling): Best short-term")
    st.write("- Tuned LSTM: Deep learning")
    st.write("- XGBoost: Tree-based")
    
    st.header("Performance Summary")
    st.metric("ARIMA MAE", "2.28 USD")
    st.metric("LSTM MAE", "7.64 USD")
    st.metric("XGBoost MAE", "35.99 USD")
    
    st.info("Free tier may sleep after inactivity — first load takes 30–60s.")

# ────────────────────────────────────────────────
# Simulated / Placeholder Results (replace with your real data)
# You can load from CSVs if saved
# ────────────────────────────────────────────────
st.header("Model Performance Comparison")

col1, col2, col3 = st.columns(3)
with col1:
    st.subheader("ARIMA")
    st.metric("MAE", "2.28 USD")
    st.metric("RMSE", "3.32 USD")
with col2:
    st.subheader("LSTM")
    st.metric("MAE", "7.64 USD")
    st.metric("RMSE", "9.61 USD")
with col3:
    st.subheader("XGBoost")
    st.metric("MAE", "35.99 USD")
    st.metric("RMSE", "47.82 USD")

# ────────────────────────────────────────────────
# Future Forecast Table (paste your last run's table or simulate)
# ────────────────────────────────────────────────
st.header("30-Day Future Forecast (Post March 2026)")

# Example data from your last run — replace with real CSV load if needed
future_data = {
    "Date": ["2026-03-09", "2026-03-10", "2026-03-11", "2026-03-12", "2026-03-13",
             "2026-03-16", "2026-03-17", "2026-03-18", "2026-03-19", "2026-03-20"],
    "ARIMA": [257.25, 257.33, 257.30, 257.31, 257.31, 257.31, 257.31, 257.31, 257.31, 257.31],
    "LSTM": [251.79, 250.16, 248.27, 246.24, 244.16, 242.07, 239.99, 237.94, 235.93, 233.96],
    "XGBoost": [174.07, 173.41, 163.83, 155.49, 139.68, 119.19, 101.68, 85.53, 81.85, 62.38]
}

future_df = pd.DataFrame(future_data)
st.dataframe(future_df.style.format({"ARIMA": "{:.2f}", "LSTM": "{:.2f}", "XGBoost": "{:.2f}"}))
st.caption("ARIMA is flat (conservative), LSTM declining, XGBoost very pessimistic — ARIMA most reliable here.")

# ────────────────────────────────────────────────
# Simple Chart Example (replace with your real plot)
# ────────────────────────────────────────────────
st.header("Forecast Trend Preview")
dates = pd.to_datetime(future_data["Date"])
fig, ax = plt.subplots()
ax.plot(dates, future_data["ARIMA"], label="ARIMA", color="orange", linestyle="--")
ax.plot(dates, future_data["LSTM"], label="LSTM", color="green")
ax.plot(dates, future_data["XGBoost"], label="XGBoost", color="purple", linestyle="-.")
ax.set_xlabel("Date")
ax.set_ylabel("Predicted Close Price (USD)")
ax.legend()
ax.grid(True, alpha=0.3)
st.pyplot(fig)

# ────────────────────────────────────────────────
# Footer
# ────────────────────────────────────────────────
st.markdown("---")
st.caption("Project by Fridah | GitHub: [AAPL-Stock-Prediction-ML](https://github.com/macfeighbitange1-dot/AAPL-Stock-Prediction-ML)")
st.caption("Built for learning financial ML — open to fintech & banking roles!")