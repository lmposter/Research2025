import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from arch.unitroot import VarianceRatio
import statsmodels.api as sm
from scipy.stats import t

np.random.seed(42)

def ar1_series(n=200, phi=0.5, sigma=1.0):
    e   = np.random.normal(scale=sigma, size=n)
    y   = np.zeros(n)
    for t in range(1, n):
        y[t] = phi * y[t-1] + e[t]
    return y

ALPHA = 0.05

def adf_detect(y):
    p = adfuller(y, autolag='AIC')[1]
    return 'mean reverting' if p < ALPHA else 'non mean reverting'

def vr_detect(y):
    p = VarianceRatio(y).pvalue
    return 'mean reverting' if p < ALPHA else 'non mean reverting'

def hurst_rs(y):
    """Return Hurst exponent (0 < H < 1) via rescaled-range."""
    y = np.asarray(y)
    N = len(y)
    max_k = int(np.floor(N / 2))
    lags  = np.logspace(2, np.log10(max_k), num=20, dtype=int)
    rs = []
    for k in np.unique(lags):
        segs = N // k
        if segs < 2:
            continue
        for i in range(segs):
            chunk = y[i*k:(i+1)*k]
            Z = chunk - np.mean(chunk)
            R = np.max(np.cumsum(Z)) - np.min(np.cumsum(Z))
            S = np.std(chunk, ddof=1)
            if S > 0:
                rs.append(R / S)
    rs = np.array(rs)
    H  = np.polyfit(np.log(range(1, len(rs)+1)), np.log(np.sort(rs)), 1)[0]
    return H

def hurst_detect(y, thresh=0.48):
    H = hurst_rs(y)
    return 'mean reverting' if H < thresh else 'non mean reverting'

def ou_detect(y):
    y_lag = y[:-1]
    y_now = y[1:]
    X     = sm.add_constant(y_lag)
    model = sm.OLS(y_now, X).fit()
    phi   = model.params[1]
    se    = model.bse[1]
    tval  = (phi - 1) / se
    df    = model.df_resid
    p_one_sided = t.cdf(tval, df=df)
    decision = (p_one_sided < ALPHA)
    return 'mean reverting' if (decision and phi < 1) else 'non mean reverting'

METHODS = {
    'ADF'            : adf_detect,
    'Variance Ratio' : vr_detect,
    'Hurst < 0.5'    : hurst_detect,
    'OU Regression'  : ou_detect,
}

def run(trials=1_000, n=200, sigma=1.0):
    phis   = [0.5, 0.9, 1.0]
    truths = ['mean reverting', 'mean reverting', 'non mean reverting']

    hits  = {m: 0 for m in METHODS}
    total = trials * len(phis)

    for _ in range(trials):
        for phi, truth in zip(phis, truths):
            y = ar1_series(n=n, phi=phi, sigma=sigma)
            for name, fn in METHODS.items():
                if fn(y) == truth:
                    hits[name] += 1

    acc = {k: v / total for k, v in hits.items()}
    df  = pd.DataFrame(acc.values(), index=acc.keys(),
                       columns=['accuracy']).sort_values('accuracy',
                                                          ascending=False)
    best = df.index[0]
    best_acc = df.iloc[0, 0]

    print(df.to_string(float_format='{:.4f}'.format))
    print(f"\nBest performer ⇒ {best}  (accuracy = {best_acc:.4f})")

if __name__ == '__main__':
    run(trials=1000, n=200, sigma=1.0)
