import numpy as np
import pandas as pd
from scipy.stats import linregress
from scipy.optimize import curve_fit
import pymannkendall as mk

np.random.seed(42)

def gen_exp_series(n=100, k=0.0, sigma=0.1, A=1.0):
    t = np.arange(n)
    log_y = np.log(A) + k * t + np.random.normal(0, sigma, n)
    return t, np.exp(log_y)

THRESH = 0.01

def label_from_slope(k):
    return ('increasing' if k > THRESH
            else 'decreasing' if k < -THRESH
            else 'no trend')

def log_lr_trend(t, y):
    slope = linregress(t, np.log(y)).slope
    return label_from_slope(slope)

def log_mk_trend(_, y):
    trend = mk.original_test(np.log(y)).trend
    return {'increasing':'increasing', 'decreasing':'decreasing'}.get(trend, 'no trend')

def exp_func(t, A, k):
    return A * np.exp(k * t)

def nls_exp_trend(t, y):
    try:
        (_, k), _ = curve_fit(exp_func, t, y, p0=(y[0], 0.01), maxfev=5000)
    except RuntimeError:
        return 'no trend'
    return label_from_slope(k)

def cagr_trend(_, y):
    r = (y[-1] / y[0])**(1/(len(y)-1)) - 1
    return label_from_slope(r)

METHODS = {
    'Log-Linear Reg' : log_lr_trend,
    'Log-Mann-Kendall': log_mk_trend,
    'Exp Curve-Fit'  : nls_exp_trend,
    'CAGR'           : cagr_trend,
}

def run(trials=1000, n=100, sigma=0.1):
    ks     = [0.10, 0.02, 0.00, -0.02, -0.10]
    truths = ['increasing','increasing','no trend',
              'decreasing','decreasing']

    hits   = {m: 0 for m in METHODS}
    total  = trials * len(ks)

    for _ in range(trials):
        for k, truth in zip(ks, truths):
            t, y = gen_exp_series(n=n, k=k, sigma=sigma)
            for name, fn in METHODS.items():
                if fn(t, y) == truth:
                    hits[name] += 1

    acc = {k: v/total for k, v in hits.items()}
    df  = pd.DataFrame(acc.values(), index=acc.keys(),
                       columns=['accuracy']).sort_values('accuracy',
                                                          ascending=False)
    best = df.index[0]
    best_acc = df.iloc[0,0]

    print(df.to_string(float_format='{:.4f}'.format))
    print(f"\nBest performer ⇒ {best}  (accuracy = {best_acc:.4f})")

if __name__ == '__main__':
    run(trials=1000, n=100, sigma=0.1)
