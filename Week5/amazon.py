import pandas as pd
from pipe import analyze_time_series

df = pd.read_csv('Amazon.csv')
ts = pd.Series(df['rt'])

results = analyze_time_series(ts, seasonal_lag=12)

print("Analysis Results on 'Amazon.csv':")
for key, value in results.items():
    print(f"{key}: {value}")