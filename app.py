import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime

# ────────────────────────────────────────────────
# Page config
# ────────────────────────────────────────────────
st.set_page_config(
    page_title="AAPL Stock Prediction Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ────────────────────────────────────────────────
# Title & Intro
# ────────────────────────────────────────────────
st.title("📈 AAPL Stock Price Prediction Dashboard")
st.markdown("""
Built with **ARIMA**, **LSTM**, and **XGBoost** | Portfolio demo by Fridah  
Train: 2010–2022 | Test: 2022–2026 | Future forecast beyond March 2026
""")

# ────────────────────────────────────────────────
# Sidebar
# ────────────────────────────────────────────────
with st.sidebar:
    st.header("Project Overview")
    st.markdown("**Models compared:**")
    st.write("- ARIMA (rolling): Best short-term accuracy")
    st.write("- Tuned LSTM: Deep learning sequence model")
    st.write("- XGBoost: Tree-based with lagged features")
    
    st.header("Performance Highlights")
    st.metric("ARIMA MAE", "2.28 USD", delta="Best")
    st.metric("LSTM MAE", "7.64 USD")
    st.metric("XGBoost MAE", "35.99 USD")
    
    st.info("Free hosting (Streamlit Cloud) may take 30–60s on first load after sleep.")
    
    st.markdown("---")
    st.caption("GitHub: [AAPL-Stock-Prediction-ML](https://github.com/macfeighbitange1-dot/AAPL-Stock-Prediction-ML)")
    st.caption("Built for learning financial ML — open to fintech & banking roles!")

# ────────────────────────────────────────────────
# Load real data from CSVs if available
# ────────────────────────────────────────────────
@st.cache_data
def load_future_forecast():
    try:
        df = pd.read_csv('future_30days_forecast.csv')
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    except FileNotFoundError:
        return None

future_df = load_future_forecast()

# ────────────────────────────────────────────────
# Model Performance Section
# ────────────────────────────────────────────────
st.header("Model Performance on Test Set")

col1, col2, col3 = st.columns(3)
with col1:
    st.subheader("ARIMA (rolling)")
    st.metric("MAE", "2.28 USD", delta="Best short-term")
    st.metric("RMSE", "3.32 USD")
with col2:
    st.subheader("Tuned LSTM")
    st.metric("MAE", "7.64 USD")
    st.metric("RMSE", "9.61 USD")
with col3:
    st.subheader("XGBoost")
    st.metric("MAE", "35.99 USD")
    st.metric("RMSE", "47.82 USD")

st.caption("ARIMA excels in short-term rolling forecasts. LSTM captures patterns but lags long-term. XGBoost struggles with trend extrapolation.")

# ────────────────────────────────────────────────
# Future Forecast Section
# ────────────────────────────────────────────────
st.header("30-Day Future Forecast (After March 6, 2026)")

if future_df is not None and not future_df.empty:
    st.dataframe(
        future_df.style.format({
            "ARIMA": "{:.2f}",
            "LSTM": "{:.2f}",
            "XGBoost": "{:.2f}"
        }),
        use_container_width=True
    )
    
    # Download button
    csv = future_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Future Forecast CSV",
        data=csv,
        file_name="aapl_future_30days_forecast.csv",
        mime="text/csv"
    )
    
    # Interactive chart
    st.subheader("Forecast Trend Visualization")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(future_df["Date"], future_df["ARIMA"], label="ARIMA", color="orange", linestyle="--", linewidth=2)
    ax.plot(future_df["Date"], future_df["LSTM"], label="LSTM", color="green", linewidth=2)
    ax.plot(future_df["Date"], future_df["XGBoost"], label="XGBoost", color="purple", linestyle="-.", linewidth=2)
    ax.set_xlabel("Date")
    ax.set_ylabel("Predicted Close Price (USD)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    st.pyplot(fig)
    
    st.caption("**Interpretation**: ARIMA remains stable/conservative. LSTM shows gradual decline. XGBoost is highly pessimistic. In real trading, ARIMA would be most reliable for short horizon here.")
else:
    st.warning("Future forecast CSV not found. Run stock_predict.py locally to generate 'future_30days_forecast.csv'.")
    st.info("You can manually upload the CSV using the file uploader below.")
    uploaded_file = st.file_uploader("Upload future_30days_forecast.csv", type="csv")
    if uploaded_file:
        future_df = pd.read_csv(uploaded_file)
        st.success("CSV uploaded! Scroll up to see the table and chart.")

# ────────────────────────────────────────────────
# Footer
# ────────────────────────────────────────────────
st.markdown("---")
st.caption("Project by Fridah | GitHub: [AAPL-Stock-Prediction-ML](https://github.com/macfeighbitange1-dot/AAPL-Stock-Prediction-ML)")
st.caption("Built for learning financial ML — open to fintech & banking roles!")