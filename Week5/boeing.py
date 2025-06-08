import pandas as pd
from pipe import analyze_time_series

df_boeing = pd.read_csv('Boeing.csv')
ts_boeing = pd.Series(df_boeing['price'])

results_boeing = analyze_time_series(ts_boeing, seasonal_lag=12)

print("Analysis Results on 'Boeing.csv' (using 'price' as the series):")
for key, value in results_boeing.items():
    print(f"{key}: {value}")