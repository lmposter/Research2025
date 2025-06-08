import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss
from arch.unitroot import PhillipsPerron, VarianceRatio

np.random.seed(42)

def ar1_series(n=200, phi=0.5):
    e = np.random.normal(size=n)
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = phi * y[t-1] + e[t]
    return y

def random_walk(n=200):
    return np.cumsum(np.random.normal(size=n))

def trend_plus_noise(n=200, slope=0.05):
    t = np.arange(n)
    return slope * t + np.random.normal(size=n)

ALPHA = 0.05

def adf_test(y):
    p = adfuller(y, autolag='AIC')[1]
    return 'stationary' if p < ALPHA else 'non-stationary'

def kpss_test(y):
    p = kpss(y, regression='c', nlags='auto')[1]
    return 'non-stationary' if p < ALPHA else 'stationary'

def pp_test(y):
    p = PhillipsPerron(y).pvalue
    return 'stationary' if p < ALPHA else 'non-stationary'

def vr_test(y):
    p = VarianceRatio(y).pvalue
    return 'stationary' if p < ALPHA else 'non-stationary'

METHODS = {
    'ADF'             : adf_test,
    'KPSS'            : kpss_test,
    'Phillips-Perron' : pp_test,
    'Variance Ratio'  : vr_test,
}

def run(trials=1_000, n=200):
    cases = [
        ('Stationary AR(1)', ar1_series,        'stationary'),
        ('Random Walk',       random_walk,      'non-stationary'),
        ('Trend + Noise',     trend_plus_noise, 'non-stationary'),
    ]

    correct = {m: 0 for m in METHODS}
    total   = trials * len(cases)

    for _ in range(trials):
        for _, gen, truth in cases:
            y = gen(n)
            for name, fn in METHODS.items():
                if fn(y) == truth:
                    correct[name] += 1

    acc = {k: v / total for k, v in correct.items()}
    df  = pd.DataFrame(acc.values(), index=acc.keys(), columns=['accuracy']).sort_values('accuracy', ascending=False)
    best_name, best_acc = df.index[0], df.iloc[0, 0]

    print(df.to_string(float_format='{:.4f}'.format))
    print(f"\nBest method ⇒ {best_name}  (accuracy = {best_acc:.4f})")

if __name__ == '__main__':
    run(trials=1000, n=200)
