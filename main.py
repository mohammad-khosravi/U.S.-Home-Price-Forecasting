import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("enigma.csv")

# Filter for national-level median home prices
df_filtered = df[
    (df["geo_code"] == "US") &
    (df["dt_code"] == "MEDIAN") &
    (df["val"].notnull())
].copy()

# Convert date column to datetime
df_filtered["date"] = pd.to_datetime(df_filtered["per_name"])

# Keep only date and value
df_ts = df_filtered[["date", "val"]].sort_values("date")

# Plot the time series
plt.figure(figsize=(14, 6))
plt.plot(df_ts["date"], df_ts["val"], linewidth=2)
plt.title("U.S. Median Home Prices Over Time", fontsize=16)
plt.xlabel("Date")
plt.ylabel("Median Price (USD)")
plt.grid(True)
plt.tight_layout()
plt.show()



window = 12  # 12-month window for rolling statistics

# Calculate rolling mean and std
rolling_mean = df_ts["val"].rolling(window=window).mean()
rolling_std = df_ts["val"].rolling(window=window).std()

# Plot
plt.figure(figsize=(14, 6))
plt.plot(df_ts["date"], df_ts["val"], label="Original", color="blue")
plt.plot(df_ts["date"], rolling_mean, label="Rolling Mean (12 mo)", color="red")
plt.plot(df_ts["date"], rolling_std, label="Rolling Std Dev (12 mo)", color="green")
plt.title("Rolling Mean and Standard Deviation")
plt.xlabel("Date")
plt.ylabel("Median Price (USD)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# First-order differencing
df_ts["val_diff"] = df_ts["val"].diff()

# Plot differenced series
plt.figure(figsize=(14, 6))
plt.plot(df_ts["date"], df_ts["val_diff"], color="purple", linewidth=1.5)
plt.title("Differenced Time Series (1st Order)")
plt.xlabel("Date")
plt.ylabel("Price Change (USD)")
plt.grid(True)
plt.tight_layout()
plt.show()

from statsmodels.tsa.seasonal import seasonal_decompose

# Drop NA values from differenced series for decomposition
df_clean = df_ts.dropna(subset=["val"])

# Set date as index
df_clean.set_index("date", inplace=True)

# Decompose the original series (not differenced)
decomposition = seasonal_decompose(df_clean["val"], model="additive", period=12)

# Plot decomposition
fig = decomposition.plot()
fig.set_size_inches(14, 10)
plt.tight_layout()
plt.show()

from statsmodels.tsa.stattools import adfuller

# Drop NA values from differenced series
diff_series = df_ts["val_diff"].dropna()

# Run ADF test
adf_result = adfuller(diff_series)

# Print the results
print("ADF Statistic:", adf_result[0])
print("p-value:", adf_result[1])
for key, value in adf_result[4].items():
    print(f"Critical Value ({key}): {value}")

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# ACF and PACF plots
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plot_acf(diff_series, ax=plt.gca(), lags=40)
plt.title("Autocorrelation Function (ACF)")

plt.subplot(1, 2, 2)
plot_pacf(diff_series, ax=plt.gca(), lags=40, method="ywm")
plt.title("Partial Autocorrelation Function (PACF)")

plt.tight_layout()
plt.show()


from statsmodels.tsa.arima.model import ARIMA

# Fit ARIMA(1,1,1) model
model = ARIMA(df_clean["val"], order=(1, 1, 1))
model_fit = model.fit()

# Forecast next 24 months
forecast = model_fit.get_forecast(steps=24)
forecast_df = forecast.summary_frame()

# Plot observed data and forecast
plt.figure(figsize=(14, 6))
plt.plot(df_clean.index, df_clean["val"], label="Observed", linewidth=2)
plt.plot(forecast_df.index, forecast_df["mean"], label="Forecast", linestyle="--", linewidth=2)
plt.fill_between(forecast_df.index, forecast_df["mean_ci_lower"], forecast_df["mean_ci_upper"], color='gray', alpha=0.3)
plt.title("ARIMA(1,1,1) Forecast of U.S. Median Home Prices")
plt.xlabel("Date")
plt.ylabel("Median Price (USD)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Split data: last 24 months for testing
train = df_clean.iloc[:-24]
test = df_clean.iloc[-24:]

# Fit model on training data
model_train = ARIMA(train["val"], order=(1, 1, 1))
model_fit_train = model_train.fit()

# Forecast on test period
forecast_test = model_fit_train.forecast(steps=24)

# Calculate error metrics
mae = mean_absolute_error(test["val"], forecast_test)
rmse = np.sqrt(mean_squared_error(test["val"], forecast_test))

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

from sklearn.metrics import r2_score

# R-squared calculation
r2 = r2_score(test["val"], forecast_test)
print(f"R-squared: {r2:.4f}")

from statsmodels.tsa.statespace.sarimax import SARIMAX

# Fit SARIMA(1,1,1)(1,1,1,12) model on training data
sarima_model = SARIMAX(train["val"], order=(1,1,1), seasonal_order=(1,1,1,12))
sarima_fit = sarima_model.fit(disp=False)

# Forecast the test period
sarima_forecast = sarima_fit.forecast(steps=24)

# Evaluate performance
sarima_mae = mean_absolute_error(test["val"], sarima_forecast)
sarima_rmse = np.sqrt(mean_squared_error(test["val"], sarima_forecast))
sarima_r2 = r2_score(test["val"], sarima_forecast)

sarima_mae, sarima_rmse, sarima_r2

import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# Prepare data
df_prophet = df_clean.reset_index()[["date", "val"]].rename(columns={"date": "ds", "val": "y"})

# Initialize and fit model
model = Prophet(yearly_seasonality=True)
model.fit(df_prophet)

# Forecast next 24 months
future = model.make_future_dataframe(periods=24, freq="MS")
forecast = model.predict(future)

# Plot forecast
fig = model.plot(forecast)
plt.title("Facebook Prophet Forecast of U.S. Median Home Prices")
plt.xlabel("Date")
plt.ylabel("Median Price (USD)")
plt.tight_layout()
plt.show()
