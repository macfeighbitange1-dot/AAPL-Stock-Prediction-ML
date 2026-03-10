import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
import os
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

print("=== Stock Price Prediction Project (Enhanced Version) ===")
print("We will use manual CSV first to avoid yfinance bugs.\n")

# ────────────────────────────────────────────────
# 1. Load Data
# ────────────────────────────────────────────────
csv_file = 'aapl_data.csv'
data = None
if os.path.exists(csv_file):
    try:
        data = pd.read_csv(csv_file, index_col='Date', parse_dates=True)
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"CSV missing columns: {missing_cols}")
        
        data = data.sort_index(ascending=True)
        print(f"→ SUCCESS: Loaded {len(data):,} rows")
        print(f"Date range: {data.index.min().date()} → {data.index.max().date()}")
    except Exception as e:
        print(f"CSV invalid: {e}")
        data = None

if data is None:
    print("Trying yfinance...")
    try:
        today = datetime.now().strftime('%Y-%m-%d')
        ticker = yf.Ticker('AAPL')
        data = ticker.history(start='2010-01-01', end=today)
        if data.empty:
            raise ValueError("Empty data")
        print(f"→ yfinance success: {len(data):,} rows")
    except Exception as e:
        print(f"yfinance failed: {e}")
        exit(1)

# 2. Preprocessing
data = data.dropna(subset=['Close'])
data['Return'] = np.log(data['Close'] / data['Close'].shift(1)).fillna(0)
print(f"After preprocessing → shape: {data.shape}")
data.to_csv('aapl_data_clean.csv', index=True)

# 3. Train/Test Split
train_size = int(len(data) * 0.80)
train_data = data.iloc[:train_size]
test_data = data.iloc[train_size:]
train_close = train_data['Close']
test_close = test_data['Close']
print(f"Train: {train_data.index[0].date()} → {train_data.index[-1].date()}")
print(f" Test: {test_data.index[0].date()} → {test_data.index[-1].date()}")

# ────────────────────────────────────────────────
# ARIMA (rolling forecast - kept as is)
# ────────────────────────────────────────────────
print("\n=== ARIMA (rolling) ===")
arima_order = (1, 1, 1)
history = list(train_close)
arima_predictions = []
for i in range(len(test_close)):
    try:
        model = ARIMA(history, order=arima_order)
        model_fit = model.fit()
        yhat = model_fit.forecast(steps=1)[0]
        arima_predictions.append(yhat)
        history.append(test_close.iloc[i])
    except:
        yhat = history[-1]
        arima_predictions.append(yhat)
        history.append(yhat)
arima_forecast_series = pd.Series(arima_predictions, index=test_close.index)
mae_arima = mean_absolute_error(test_close, arima_forecast_series)
rmse_arima = np.sqrt(mean_squared_error(test_close, arima_forecast_series))
print(f"MAE: {mae_arima:,.2f} | RMSE: {rmse_arima:,.2f}")

# ────────────────────────────────────────────────
# Tuned LSTM
# ────────────────────────────────────────────────
print("\n=== Tuned LSTM ===")
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
def create_sequences(data, seq_length=60):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length), 0])
        y.append(data[i + seq_length, 0])
    return np.array(X), np.array(y)
seq_length = 60
X, y = create_sequences(scaled_data, seq_length)
train_seq_end = train_size - seq_length
X_train = X[:train_seq_end]
y_train = y[:train_seq_end]
X_test = X[train_seq_end:]
y_test = y[train_seq_end:]
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(seq_length, 1)))
model.add(Dropout(0.2))
model.add(LSTM(50))
model.add(Dropout(0.2))
model.add(Dense(25))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')
early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
print("Training tuned LSTM (up to 100 epochs)...")
model.fit(X_train, y_train, batch_size=32, epochs=100, verbose=1, callbacks=[early_stop])
lstm_pred_scaled = model.predict(X_test)
lstm_pred = scaler.inverse_transform(lstm_pred_scaled).flatten()
lstm_dates = test_close.index[-len(lstm_pred):]
lstm_forecast_series = pd.Series(lstm_pred, index=lstm_dates)
aligned_actual = test_close[lstm_dates]
mae_lstm = mean_absolute_error(aligned_actual, lstm_forecast_series)
rmse_lstm = np.sqrt(mean_squared_error(aligned_actual, lstm_forecast_series))
print(f"LSTM Tuned MAE: {mae_lstm:,.2f} | RMSE: {rmse_lstm:,.2f}")

# ────────────────────────────────────────────────
# Improved Gradient Boosting (XGBoost + more lags)
# ────────────────────────────────────────────────
print("\n=== Improved Gradient Boosting (XGBoost) ===")
def create_lagged_features(df, lags=20):
    df = df.copy()
    for i in range(1, lags + 1):
        df[f'Close_lag_{i}'] = df['Close'].shift(i)
        df[f'Return_lag_{i}'] = df['Return'].shift(i)
    return df.dropna()
lagged_data = create_lagged_features(data, lags=20)
train_lagged = lagged_data.iloc[:train_size]
test_lagged = lagged_data.iloc[train_size:]
features = [f'Close_lag_{i}' for i in range(1, 21)] + [f'Return_lag_{i}' for i in range(1, 21)]
X_train_gb = train_lagged[features]
y_train_gb = train_lagged['Close']
X_test_gb = test_lagged[features]
y_test_gb = test_lagged['Close']
gb_model = XGBRegressor(n_estimators=500, learning_rate=0.03, max_depth=5, subsample=0.8, random_state=42)
gb_model.fit(X_train_gb, y_train_gb)
gb_pred = gb_model.predict(X_test_gb)
gb_forecast_series = pd.Series(gb_pred, index=y_test_gb.index)
mae_gb = mean_absolute_error(y_test_gb, gb_forecast_series)
rmse_gb = np.sqrt(mean_squared_error(y_test_gb, gb_forecast_series))
print(f"XGBoost MAE: {mae_gb:,.2f} | RMSE: {rmse_gb:,.2f}")

# ────────────────────────────────────────────────
# Future Prediction - Next 30 Days
# ────────────────────────────────────────────────
print("\n=== Future Prediction - Next 30 Days ===")
future_days = 30

# Safety check for pandas
try:
    future_dates = pd.date_range(start=test_close.index[-1] + timedelta(days=1), periods=future_days, freq='B')
except NameError as e:
    print("Error: pandas not available. Make sure 'import pandas as pd' is at the top.")
    exit(1)

# ARIMA future (rolling from recent history)
arima_future = []
arima_hist = list(test_close[-100:])  # Use recent history for stability
for _ in range(future_days):
    try:
        arima_temp_model = ARIMA(arima_hist, order=(1,1,1))
        arima_fit = arima_temp_model.fit()
        pred = arima_fit.forecast(steps=1)[0]
        arima_future.append(pred)
        arima_hist.append(pred)
    except Exception as e:
        print(f"ARIMA future step failed: {e}. Using last value.")
        pred = arima_hist[-1]
        arima_future.append(pred)
        arima_hist.append(pred)

arima_future_series = pd.Series(arima_future, index=future_dates)

# LSTM future (recursive)
lstm_future = []
current_seq = scaled_data[-seq_length:].reshape(1, seq_length, 1)
for _ in range(future_days):
    pred_scaled = model.predict(current_seq, verbose=0)
    pred = scaler.inverse_transform(pred_scaled)[0][0]
    lstm_future.append(pred)
    current_seq = np.roll(current_seq, -1, axis=1)
    current_seq[0, -1, 0] = scaler.transform([[pred]])[0][0]

lstm_future_series = pd.Series(lstm_future, index=future_dates)

# XGBoost future (recursive)
gb_future = []
current_features = lagged_data.iloc[-1][features].values.reshape(1, -1)
for _ in range(future_days):
    pred = gb_model.predict(current_features)[0]
    gb_future.append(pred)
    # Shift lags (approximate update)
    current_features = np.roll(current_features, -2, axis=1)
    current_features[0, 0] = pred
    if current_features[0, 1] != 0:
        current_features[0, 10] = np.log(pred / current_features[0, 1])
    else:
        current_features[0, 10] = 0  # avoid div by zero

gb_future_series = pd.Series(gb_future, index=future_dates)

# Print future table
print("\nNext 30 Days Forecast:")
future_df = pd.DataFrame({
    'Date': future_dates.date,
    'ARIMA': arima_future_series.round(2),
    'LSTM': lstm_future_series.round(2),
    'XGBoost': gb_future_series.round(2)
})
print(future_df.to_string(index=False))

future_df.to_csv('future_30days_forecast.csv', index=False)
print("Future forecast saved: 'future_30days_forecast.csv' (open in Excel)")

# Plot recent + future
plt.figure(figsize=(15, 8))
plt.plot(test_close.index[-100:], test_close[-100:], label='Recent Actual', color='blue')
plt.plot(future_dates, arima_future_series, label='ARIMA Future', color='orange', linestyle='--')
plt.plot(future_dates, lstm_future_series, label='LSTM Future', color='green')
plt.plot(future_dates, gb_future_series, label='XGBoost Future', color='purple', linestyle='-.')
plt.title('30-Day Future Forecast (All Models)')
plt.xlabel('Date')
plt.ylabel('Close Price (USD)')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('future_30days_plot.png', dpi=150)
plt.close()
print("→ Future plot saved: 'future_30days_plot.png'")
print("   Open it — zoom right to see the 3 model predictions after March 6, 2026")

print("\nProject fully complete with future forecasts!")
print("Reply with 'portfolio tips' for GitHub/resume help to use this for bank jobs, or 'analyze future' to interpret the next 30 days predictions.\n")