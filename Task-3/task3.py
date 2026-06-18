import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

# Load and Prepare Data
df = pd.read_csv(
    "airline-passengers.csv",
    parse_dates=['Month'],
    index_col='Month'
)

print("Data Head:\n", df.head())

# Resample to Quarterly
quarterly = df.resample('Q').mean()

# Trend and Seasonality Decomposition
decomposition = seasonal_decompose(
    df,
    model='multiplicative'
)

fig = decomposition.plot()
fig.set_size_inches(12, 8)
plt.show()

# Moving Averages
df['MA_6'] = df['Passengers'].rolling(window=6).mean()
df['MA_12'] = df['Passengers'].rolling(window=12).mean()

plt.figure(figsize=(12, 6))
plt.plot(df['Passengers'], label='Actual')
plt.plot(df['MA_6'], label='6-Month MA', linestyle='--')
plt.plot(df['MA_12'], label='12-Month MA', color='red')

plt.legend()
plt.title('Moving Averages Smoothing')
plt.show()

# ARIMA Forecasting
train = df.iloc[:-12]
test = df.iloc[-12:]

model = ARIMA(
    train,
    order=(2, 1, 1),
    seasonal_order=(1, 1, 1, 12)
)

result = model.fit()

forecast = result.forecast(steps=12)

rmse = np.sqrt(
    mean_squared_error(test, forecast)
)

print(f"RMSE: {rmse:.1f} passengers")

# Forecast vs Actual Plot
plt.figure(figsize=(12, 6))

plt.plot(
    train.index,
    train,
    label='Training Data'
)

plt.plot(
    test.index,
    test,
    label='Actual',
    color='blue'
)

plt.plot(
    test.index,
    forecast,
    label='Forecast',
    color='red',
    linestyle='--'
)

plt.fill_between(
    test.index,
    forecast * 0.8,
    forecast * 1.2,
    alpha=0.2
)

plt.title(
    f"ARIMA Forecast (RMSE={rmse:.1f})"
)

plt.legend()
plt.show()

print("Time Series Analysis Completed Successfully!")
