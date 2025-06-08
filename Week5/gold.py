import pandas as pd
from pipe import analyze_time_series

df_gold = pd.read_csv('Gold.csv')

ts = df_gold['VALUE']

results = analyze_time_series(ts, seasonal_lag=12)

print("\nAnalysis Results on 'Gold.csv':")
for k, v in results.items():
    print(f"{k}: {v}")