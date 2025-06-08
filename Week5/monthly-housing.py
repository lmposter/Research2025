import pandas as pd
from pipe import analyze_time_series

df_thrips = pd.read_csv('monthly-housing.csv')

ts = df_thrips['hpi']
results = analyze_time_series(ts, seasonal_lag=12)

for key, val in results.items():
    print(f"{key}: {val}")

ts = df_thrips['numsold']
results = analyze_time_series(ts, seasonal_lag=12)

for key, val in results.items():
    print(f"{key}: {val}")