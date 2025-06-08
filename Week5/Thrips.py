import pandas as pd
from pipe import analyze_time_series

df_thrips = pd.read_csv('Thrips.csv')

ts = df_thrips['logThrips']
results = analyze_time_series(ts, seasonal_lag=12)

print("\nAnalysis Results on 'Thrips.csv':")
for key, val in results.items():
    print(f"{key}: {val}")