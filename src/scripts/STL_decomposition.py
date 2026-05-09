"""
Script: STL_decomposition.py
Description: Perform STL decomposition on hourly temperature data from a specific weather station. This
is used in the thesis to analyze the seasonal, trend, and residual components of the temperature time series.
"""
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL

# 1. Load data
df = pd.read_csv('./data/Turku/Turku_preprocessed.csv')

# Print min and max datetime to check the range of the data
print("Min datetime:", df['datetime'].min())
print("Max datetime:", df['datetime'].max())

# 2. Filter on a specific station
df_station = df[df['station_name'] == 'Betel'].copy()

# Create time series with datetime index
series = pd.Series(
    data=df_station['temp_dry_avg_2m'].values,
    index=pd.to_datetime(df_station['datetime'])
).asfreq('1h').dropna()


# 3. Resample to hourly frequency (average temperature per hour)
df_hourly = series.resample('ME').mean()

# 4. STL Decomposition
# period=24 because there are 24 hours in a day
res = STL(df_hourly, period=24, robust=True).fit()

fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)

axes[0].plot(df_hourly.index, df_hourly, linewidth=1)
axes[0].set_title("Originele maandelijkse temperatuurdata")

axes[1].plot(df_hourly.index, res.trend, linewidth=2)
axes[1].set_title("Trend")

axes[2].plot(df_hourly.index, res.seasonal, linewidth=0.8)
axes[2].set_title("Seizoensgebondenheid")

axes[3].plot(df_hourly.index, res.resid, linewidth=0.8)
axes[3].set_title("Resterende component (remainder)")

fig.supylabel("Temperatuur (°C)")
plt.tight_layout()
plt.savefig('plots/STL_time_series_decomposition.png')
