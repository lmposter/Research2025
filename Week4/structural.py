import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.diagnostic import breaks_cusumolsresid
import ruptures as rpt

np.random.seed(42)

def gen_series(n=200, shift=0.0, sigma=1.0, break_loc=100):
    """Return 1-D numpy array with or w/o mean shift at break_loc."""
    y = np.random.normal(0, sigma, n)
    if shift != 0:
        y[break_loc:] += shift
    return y

ALPHA = 0.05

def cusum_detect(y):
    x = np.ones_like(y)
    model = sm.OLS(y, x).fit()
    pval = breaks_cusumolsresid(model.resid, ddof=0)[1]
    return 'break' if pval < ALPHA else 'stable'

def chow_scan_detect(y, min_seg=30):
    n = len(y)
    x = np.ones_like(y)
    sse_full = np.sum((y - y.mean())**2)
    best_p = 1.0
    for τ in range(min_seg, n - min_seg):
        sse1 = np.sum((y[:τ] - y[:τ].mean())**2)
        sse2 = np.sum((y[τ:] - y[τ:].mean())**2)
        k = 1
        fnum = (sse_full - (sse1 + sse2)) / k
        fden = (sse1 + sse2) / (n - 2*k)
        F = fnum / fden
        p = 1 - stats.f.cdf(F, k, n - 2*k)
        best_p = min(best_p, p)
    return 'break' if best_p < ALPHA else 'stable'

def pelt_detect(y):
    algo = rpt.Pelt(model="l2").fit(y)
    brk = algo.predict(pen=3)
    return 'break' if len(brk) > 1 else 'stable'

def binseg_detect(y):
    brk = rpt.Binseg(model="l2").fit(y).predict(n_bkps=1)
    return 'break' if brk[0] < len(y) else 'stable'

METHODS = {
    'CUSUM'       : cusum_detect,
    'Chow Scan'   : chow_scan_detect,
    'Bai–Perron'  : pelt_detect,
    'Binseg'      : binseg_detect,
}

def run(trials=1000, n=200, sigma=1.0, break_loc=100):
    shifts = [3.0, 0.7, 0.0]
    truths = ['break', 'break', 'stable']
    hits   = {m: 0 for m in METHODS}
    total  = trials * len(shifts)

    for _ in range(trials):
        for shift, truth in zip(shifts, truths):
            y = gen_series(n=n, shift=shift, sigma=sigma, break_loc=break_loc)
            for name, fn in METHODS.items():
                if fn(y) == truth:
                    hits[name] += 1

    acc = {k: v/total for k, v in hits.items()}
    df  = pd.DataFrame(acc.values(), index=acc.keys(),
                       columns=['accuracy']).sort_values('accuracy',
                                                          ascending=False)
    best = df.index[0]
    best_acc = df.iloc[0, 0]

    print(df.to_string(float_format='{:.4f}'.format))
    print(f"\nBest performer ⇒ {best}  (accuracy = {best_acc:.4f})")

if __name__ == '__main__':
    run(trials=1000, n=200, sigma=1.0, break_loc=100)
