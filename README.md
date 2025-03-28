# Stock Price Prediction using Time Series Models

This project implements and compares different time series forecasting models to predict Apple Inc. (AAPL) stock prices. The models include LSTM, GRU (deep learning approaches), ARIMA, and Prophet (traditional statistical methods).

## Features

- **Data Collection**: Fetches historical AAPL stock data from Yahoo Finance (2015-2025)
- **Feature Engineering**:
  - Technical indicators (RSI, EMA)
  - Target variable creation (next day's closing price)
  - Data normalization (MinMax scaling)
- **Model Implementations**:
  - LSTM with 150 units
  - GRU with 150 units
  - ARIMA(3,1,3)
  - Prophet with logistic growth
- **Hybrid Approach**: Combines best-performing models
- **Evaluation**: RMSE comparison across all models

## Results Summary

| Model          | RMSE   |
|----------------|--------|
| ARIMA          | 4.03   |
| GRU            | 4.74   |
| LSTM           | 7.62   |
| Hybrid Prophet | 10.73  |
| Prophet        | 68.03  |

## Key Findings

1. **ARIMA performed best** with lowest RMSE (4.03), suggesting strong autocorrelations in the data
2. **Deep learning models (GRU/LSTM)** showed competitive performance but required more tuning
3. **Prophet struggled** with default settings but improved with weekly aggregation and logistic growth
4. **Feature importance**: Technical indicators (RSI, EMAs) contributed significantly to predictions

## Requirements

- Python 3.8+
- Libraries:
pandas,
numpy,
yfinance,
pandas_ta,
scikit-learn,
tensorflow/keras,
statsmodels,
prophet,
matplotlib


## Usage

1. Clone the repository
2. Install requirements: `pip install -r requirements.txt`
3. Run the Jupyter notebook: `jupyter notebook stock_prediction.ipynb`


