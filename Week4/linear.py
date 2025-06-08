import numpy as np
import pandas as pd
from scipy.stats import linregress, theilslopes, binomtest
import pymannkendall as mk
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import STL

np.random.seed(41)

def generate_series(n, slope, noise_std):
    x = np.arange(n)
    y = slope * x + np.random.normal(0, noise_std, n)
    return x, y

def label_from_slope(s):
    return ('increasing' if s > 0.1
            else 'decreasing' if s < -0.1
            else 'no trend')

def linear_regression_trend(x, y):
    return label_from_slope(linregress(x, y).slope)

def mann_kendall_trend(_, y):
    t = mk.original_test(y).trend
    return {'increasing':'increasing', 'decreasing':'decreasing'}.get(t, 'no trend')

def theil_sen_trend(x, y):
    return label_from_slope(theilslopes(y, x=x)[0])

def cox_stuart_trend(_, y, alpha=0.05):
    half = len(y)//2
    pos = np.sum(y[-half:] > y[:half])
    neg = np.sum(y[-half:] < y[:half])
    n = pos + neg
    if n == 0 or binomtest(min(pos,neg), n, .5).pvalue >= alpha:
        return 'no trend'
    return 'increasing' if pos > neg else 'decreasing'

def adf_lr_trend(x, y, alpha=0.05):
    p = adfuller(y, autolag='AIC')[1]
    return 'no trend' if p < alpha else label_from_slope(linregress(x, y).slope)

def stl_lr_trend(x, y):
    trend = STL(y, period=max(7, len(y)//10), robust=True).fit().trend
    return label_from_slope(linregress(x, trend).slope)

METHODS = {
    'Mann-Kendall'     : mann_kendall_trend,
    'Linear Regression': linear_regression_trend,
    'Theil–Sen'        : theil_sen_trend,
    'Cox–Stuart'       : cox_stuart_trend,
    'ADF + Slope'      : adf_lr_trend,
    'STL + Slope'      : stl_lr_trend,
}

def run(trials=1000, n=100, noise_std=1.0):
    slopes  = [ 2.0,  0.3, 0.0, -0.3, -2.0]
    truths  = ['increasing','increasing','no trend','decreasing','decreasing']
    counts  = {m:0 for m in METHODS}
    total   = trials * len(slopes)

    for _ in range(trials):
        for s, truth in zip(slopes, truths):
            x, y = generate_series(n, s, noise_std)
            for name, fn in METHODS.items():
                if fn(x, y) == truth:
                    counts[name] += 1

    acc = {k: v/total for k,v in counts.items()}
    df  = pd.DataFrame([acc]).T.rename(columns={0:'accuracy'}).sort_values('accuracy', ascending=False)
    best_name = df.index[0]
    best_acc  = df.iloc[0,0]
    print(df.to_string(float_format='{:.4f}'.format))
    print(f"\nBest performer ⇒ {best_name}  (accuracy = {best_acc:.4f})")

if __name__ == '__main__':
    run(trials=1000, n=100, noise_std=1.0)
