import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss
from arch.unitroot import PhillipsPerron, VarianceRatio

np.random.seed(42)

def random_walk(n=200, drift=0.0, sigma=1.0):
    e = np.random.normal(0, sigma, n)
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = y[t-1] + drift + e[t]
    return y

def ar1_series(n=200, phi=0.5, sigma=1.0):
    e = np.random.normal(0, sigma, n)
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = phi * y[t-1] + e[t]
    return y

ALPHA = 0.05

def adf_detect(y):
    """ADF: H₀ = unit root (random walk)."""
    p = adfuller(y, autolag='AIC')[1]
    return 'random walk' if p >= ALPHA else 'non random walk'

def pp_detect(y):
    """Phillips–Perron: H₀ = unit root."""
    p = PhillipsPerron(y).pvalue
    return 'random walk' if p >= ALPHA else 'non random walk'

def vr_detect(y):
    """Variance Ratio: H₀ = random walk."""
    p = VarianceRatio(y).pvalue
    return 'random walk' if p >= ALPHA else 'non random walk'

def kpss_detect(y):
    """KPSS: H₀ = stationary (non-RW)."""
    p = kpss(y, regression='c', nlags='auto')[1]
    return 'random walk' if p < ALPHA else 'non random walk'

METHODS = {
    'ADF'            : adf_detect,
    'Phillips–Perron': pp_detect,
    'Variance Ratio' : vr_detect,
    'KPSS'           : kpss_detect,
}

def run(trials=1000, n=200, sigma=1.0):
    generators = [
        ('Pure RW',          lambda: random_walk(n, 0.0, sigma), 'random walk'),
        ('RW with drift',    lambda: random_walk(n, 0.05, sigma), 'random walk'),
        ('Stationary AR(1)', lambda: ar1_series(n, 0.5, sigma),   'non random walk')
    ]

    hits  = {m: 0 for m in METHODS}
    total = trials * len(generators)

    for _ in range(trials):
        for _, gen, truth in generators:
            y = gen()
            for name, fn in METHODS.items():
                if fn(y) == truth:
                    hits[name] += 1

    acc = {k: v / total for k, v in hits.items()}
    df  = (pd.DataFrame(acc.values(), index=acc.keys(), columns=['accuracy'])
             .sort_values('accuracy', ascending=False))

    best_name = df.index[0]
    best_acc  = df.iloc[0, 0]

    print(df.to_string(float_format='{:.4f}'.format))
    print(f"\nBest performer ⇒ {best_name}  (accuracy = {best_acc:.4f})")

if __name__ == '__main__':
    run(trials=1000, n=200, sigma=1.0)
