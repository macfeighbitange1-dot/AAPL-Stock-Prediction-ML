# Create README.md
echo # AAPL Stock Price Prediction Project > README.md
echo. >> README.md
echo Machine learning project to forecast Apple (AAPL) stock prices using three models: >> README.md
echo - ARIMA (rolling forecast) → MAE 2.28 USD >> README.md
echo - Tuned LSTM → MAE 7.64 USD >> README.md
echo - XGBoost (improved with lags) → MAE 35.99 USD >> README.md
echo. >> README.md
echo Includes future 30-day forecasts and model comparison plots. >> README.md
echo. >> README.md
echo ## How to Run >> README.md
echo 1. Clone the repo: `git clone https://github.com/macfeighbitange1-dot/AAPL-Stock-Prediction-ML.git` >> README.md
echo 2. Install dependencies: `pip install -r requirements.txt` >> README.md
echo 3. Run: `python stock_predict.py` >> README.md
echo. >> README.md
echo ## Skills Demonstrated >> README.md
echo - Time series forecasting (ARIMA) >> README.md
echo - Deep learning (LSTM with dropout & early stopping) >> README.md
echo - Gradient boosting (XGBoost) >> README.md
echo - Data preprocessing & visualization (pandas, matplotlib) >> README.md
echo - Future prediction & model ensemble concepts >> README.md
echo. >> README.md
echo Built for learning financial ML — open to fintech/banking roles! >> README.md

# Create requirements.txt (list of packages)
pip freeze > requirements.txt

# Add & push
git add README.md requirements.txt
git commit -m "Add professional README and requirements.txt"
git push
