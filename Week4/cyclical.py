import numpy as np
import pandas as pd
from scipy.fft import rfft, rfftfreq
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.filters.hp_filter import hpfilter
from statsmodels.tsa.filters.bk_filter import bkfilter

np.random.seed(0)

def gen_cycle_series(n=200, amp=0.0, sigma=1.0, period=50):
    t = np.arange(n)
    return amp * np.sin(2 * np.pi * t / period) + np.random.normal(0, sigma, n)

VAR_THRESH  = 0.20
ACF_THRESH  = 0.30
PEAK_RATIO  = 5.0

def hp_cycle(y, lamb=1600):
    cycle, _ = hpfilter(y, lamb=lamb)
    return 'cyclical' if np.nanvar(cycle) / np.var(y) > VAR_THRESH else 'no cycle'

def bk_cycle(y, low=15, high=80, K=12):
    cycle = bkfilter(y, low=low, high=high, K=K)
    cyc = cycle[~np.isnan(cycle)]
    return 'cyclical' if np.var(cyc) / np.var(y) > VAR_THRESH else 'no cycle'

def fft_cycle(y, period=50):
    n = len(y)
    f  = rfftfreq(n, d=1)
    p  = np.abs(rfft(y - np.mean(y)))**2
    idx = np.argmin(np.abs(f - 1/period))
    return 'cyclical' if p[idx] / np.mean(p) > PEAK_RATIO else 'no cycle'

def acf_cycle(y, period=50):
    a = acf(y, nlags=period, fft=True)
    return 'cyclical' if abs(a[period]) > ACF_THRESH else 'no cycle'

METHODS = {
    'HP Filter'   : hp_cycle,
    'Baxter–King' : bk_cycle,
    'FFT Peak'    : fft_cycle,
    'ACF Lag'     : acf_cycle,
}

def run(trials=1000, n=200, sigma=1.0, period=50):
    amps   = [2.0, 0.5, 0.0]
    truths = ['cyclical', 'cyclical', 'no cycle']
    hits   = {m: 0 for m in METHODS}
    total  = trials * len(amps)

    for _ in range(trials):
        for amp, truth in zip(amps, truths):
            y = gen_cycle_series(n=n, amp=amp, sigma=sigma, period=period)
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
    run(trials=1000, n=200, sigma=1.0, period=50)
