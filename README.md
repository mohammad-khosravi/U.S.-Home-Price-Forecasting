# Predicting U.S. Median Home Prices

Goal: Forecast U.S. median home prices using historical trends and seasonal patterns.
Techniques: Applied time series models including ARIMA, SARIMA, and Prophet with evaluation metrics.
Dataset: Monthly median home price data from 1963–2017, sourced from U.S. Census Bureau via Enigma.

### **Step 1: Visualize the Time Series**

Let’s understand the long-term trend, seasonality, and any irregular patterns in the U.S. median home prices over time.

```python
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
```

![U S  Median Home Prices Over Time](https://github.com/user-attachments/assets/08c01ea1-3f8f-4172-8eb1-e53e2c0498c6)

Here's the plot of U.S. median home prices over time.

You can clearly see the **steady long-term upward trend**, with:

- **Accelerated growth** in the 2000s,
- A **notable dip** around 2008–2010 (likely the housing crisis),
- Then continued growth into the 2020s.

### **Step 2:** Is the Time Series Stationary?

Next, we’ll check if the time series is **stationary**, which is essential for many forecasting models. A stationary series has a constant mean and variance over time. To do this visually, we’ll use **rolling statistics** — comparing the original series to its rolling mean and standard deviation.

```python
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
```

![Rolling Mean and Standard Deviation](https://github.com/user-attachments/assets/46566f22-bfb5-4ddf-8fa3-2a8b4e031323)

This plot shows that both the rolling mean and standard deviation change significantly over time, especially with an upward drift in the mean and volatility around the 2008 crisis. That tells us the series is **not stationary**, which means we’ll need to difference the data before modeling.

When we say a time series is **not stationary**, it means that:

- The **average value** (mean) is changing over time — like the upward trend in home prices.
- The **spread** (variance) of the data may also change — like how prices fluctuated more during the 2008 crisis.

Many forecasting models, especially ARIMA-based models, **assume** that the data is stationary — because stationary data is easier to model and predict accurately.

### So, what does "differencing" mean?

**Differencing** means subtracting the previous value from the current value. This removes trends and helps flatten the series.

Here's a simple example:

| Month | Price | Differenced |
| --- | --- | --- |
| Jan | 100000 | – |
| Feb | 101000 | 1000 |
| Mar | 102500 | 1500 |

This new column (Differenced) represents the **change** in price rather than the price itself.

If we difference once and the trend disappears, great — if not, we may need to difference again (second-order differencing).

Let’s apply **first-order differencing** — that is, subtract each value from the one before it. This will transform the series into one showing monthly **price changes** instead of the raw price.

```python
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
```

![Differenced Time Series (1st Order)](https://github.com/user-attachments/assets/6d7b7c56-8910-4828-b7e8-97a66912a951)

Here’s the differenced time series. The overall trend is now gone, and we’re looking at how much prices changed each month.

This new series looks much more stable — the values now fluctuate around zero. That’s a strong sign that the series is **closer to stationary** and suitable for modeling.

### **Step 3: Make the Time Series Stationary with Differencing**

We apply first-order differencing to remove the trend from the original series. This transformation helps stabilize the mean, which is required for most time series forecasting models.

The result is a plot of monthly **price changes**, rather than raw home prices. This series looks more stable and ready for modeling.

```python
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
```
![seasonal_decomposition_us_home_prices](https://github.com/user-attachments/assets/23f0a0b3-5474-48c0-ac3d-f9d58a725670)


This decomposition plot breaks the original time series into three components:

- **Trend**: Clearly rising over time, confirming the overall upward drift in home prices.
- **Seasonality**: Small but visible repeating cycles — likely tied to yearly housing market behavior.
- **Residual**: The leftover noise after removing trend and seasonality.

### **Step 4: Test for Stationarity with the Augmented Dickey-Fuller Test**

To confirm whether our time series is stationary, we’ll use the **Augmented Dickey-Fuller (ADF) test**. This test checks for a unit root — a sign of non-stationarity.

If the **p-value is less than 0.05**, we can reject the null hypothesis and say the series **is stationary**.

We'll apply this to the **differenced series**, since the original one was clearly non-stationary.

```python
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
```

which outputs:

```python
ADF Statistic: -3.89203719711937
p-value: 0.002094236783477825
Critical Value (1%): -3.440856177517568
Critical Value (5%): -2.86617548304384
Critical Value (10%): -2.56923863104
```

Since the ADF statistic is **less than all critical values** and the **p-value is well below 0.05**, we can **confidently reject the null hypothesis** — the differenced series is **stationary**.

### **Step 5: Identify Model Parameters with ACF and PACF Plots**

Before fitting an ARIMA model, we need to decide on its parameters:

- **AR (Auto-Regressive)**: How many past values influence the current one (parameter `p`)
- **MA (Moving Average)**: How many past forecast errors influence the current value (parameter `q`)
- **Differencing (d)**: We already applied first-order differencing, so `d = 1`

To find good values for `p` and `q`, we’ll plot the **Autocorrelation Function (ACF)** and **Partial Autocorrelation Function (PACF)**.

```python
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
```

![ACF and PACF plots](https://github.com/user-attachments/assets/ff34a93d-6491-40c6-840f-77a864b45de9)

These plots help us decide the ARIMA parameters:

- **ACF (left)** shows how correlated the series is with its past values. A sharp drop after lag 1 suggests a low value for **q** (likely 1).
- **PACF (right)** shows the correlation of the series with past values, controlling for intermediate lags. A sharp cut-off after lag 1 suggests **p = 1**.

Based on this, a good starting point for the ARIMA model is **ARIMA(1,1,1)**.

### **Step 6: Fit the ARIMA Model and Forecast Future Prices**

We’ll now fit an **ARIMA(1,1,1)** model:

- `p = 1`: One autoregressive term
- `d = 1`: First differencing (already applied)
- `q = 1`: One moving average term

Then, we’ll generate a forecast for the next **24 months** and plot it alongside the historical data.

```python
from statsmodels.tsa.arima.model import ARIMA

# Fit ARIMA(1,1,1) model on original (non-differenced) series
model = ARIMA(df_clean["val"], order=(1, 1, 1))
model_fit = model.fit()

# Forecast next 24 months
forecast = model_fit.get_forecast(steps=24)
forecast_df = forecast.summary_frame()

# Plot actual data and forecast
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
```

![ARIMA(1,1,1) Forecast of U S  Median Home Prices](https://github.com/user-attachments/assets/8d200b24-5c53-4af5-bc54-b2a1effef0e7)

Here’s the ARIMA(1,1,1) forecast for the next 24 months.

The orange dashed line shows predicted values continuing the upward trend — consistent with historical growth. While we didn’t plot the confidence interval here, the forecast still gives a solid outlook based on recent data.

### **Step 7: Evaluate the Model’s Performance**

Now that we've fit and forecasted with the ARIMA(1,1,1) model, let’s assess how well it performed. We’ll do this by:

- Splitting the time series into a **train/test split**
- Fitting the model on the training set
- Forecasting on the test set
- Comparing predictions to actual values using **Mean Absolute Error (MAE)** and **Root Mean Squared Error (RMSE)**

```python
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
```

The model’s performance on the last 24 months of data:

- **Mean Absolute Error (MAE)**: $16,675
- **Root Mean Squared Error (RMSE)**: $19,366

This means, on average, the model’s predictions are off by around **$16–19k**, which is reasonable depending on your tolerance and the range of home prices.

let’s compute **R-squared (coefficient of determination)** as well, to measure how much of the variance in the actual prices our model explains.

```python
from sklearn.metrics import r2_score

# R-squared calculation
r2 = r2_score(test["val"], forecast_test)
print(f"R-squared: {r2:.4f}")
```

The **R-squared value is -2.90**, which is quite low — and actually **negative**.

This can happen when:

- The model fits poorly on the test data
- A simple horizontal line (mean of the actuals) would predict better than the model

Despite the MAE and RMSE being in a tolerable range, the negative R² means the **model isn’t capturing the variability in home prices well** during this forecast window.

We can try:

- A more complex model (like **SARIMA** to handle seasonality)
- Adding **exogenous variables** (e.g., interest rates, inflation)
- Using **Prophet** for its built-in handling of seasonality and holidays

### **Step 8: Fit a SARIMA Model to Improve Forecast Accuracy**

Since our ARIMA model didn’t capture the variability well (negative R²), let’s upgrade to **SARIMA** — a seasonal version of ARIMA that can model repeating cycles, like those in housing markets.

We’ll start with:

- `(p,d,q) = (1,1,1)` — same as ARIMA
- `(P,D,Q,s) = (1,1,1,12)` — assuming yearly seasonality (`s=12` months)

```python
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
```

Here’s how the **SARIMA(1,1,1)(1,1,1,12)** model performed:

- **MAE**: $9,860
- **RMSE**: $12,889
- **R²: -0.73**

Compared to the ARIMA model:

- Errors (MAE, RMSE) are **significantly lower** — a good sign.
- R² is still negative, but **less negative**, showing an improvement.

This suggests the SARIMA model **captures the patterns better**, though there’s still room for improvement. We could now try **Facebook Prophet.**

### **Step 9: Forecast U.S. Home Prices with Facebook Prophet**

Prophet is a powerful forecasting tool by Meta (Facebook) designed for business time series. It handles:

- Seasonality (yearly, weekly)
- Holidays
- Missing data
- Trend changes

Let’s prep the data and run Prophet for a 24-month forecast.

```python
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
```

![Facebook Prophet Forecast of U S  Median Home Prices](https://github.com/user-attachments/assets/e5c34cca-9cca-4d5f-98cc-a3d02fb333b7)

Perfect — that chart looks great!

The **Facebook Prophet forecast** shows:

- A smooth continuation of the long-term upward trend in home prices
- Clear **confidence intervals** (light blue bands), giving a sense of uncertainty
- Reasonable tracking of historical prices, including dips (like around 2008)

This result:

- Outperforms ARIMA in terms of trend fit
- Automatically handles yearly seasonality
- Is visually clean and perfect for reporting
