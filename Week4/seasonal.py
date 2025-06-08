import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf

def generate_series(n, amp, noise_std, period):
    x = np.arange(n)
    return amp * np.sin(2 * np.pi * x / period) + np.random.normal(0, noise_std, n)

def stl_detect(y, period):
    r = STL(y, period=period, robust=True).fit()
    return 'seasonal' if np.var(r.seasonal) > 1.5 * np.var(r.resid) else 'no season'

def acf_detect(y, period):
    return 'seasonal' if abs(acf(y, nlags=period, fft=True)[period]) > 0.2 else 'no season'

def fft_detect(y, period):
    n = len(y)
    f = np.fft.rfftfreq(n, d=1)
    p = np.abs(np.fft.rfft(y - np.mean(y)))**2
    i = np.argmin(np.abs(f - 1 / period))
    return 'seasonal' if p[i] / np.mean(p) > 5 else 'no season'

def snaive_detect(y, period):
    n = len(y)
    if n <= period:
        return 'no season'
    ps = np.concatenate([np.full(period, np.nan), y[:-period]])
    pn = np.concatenate([[np.nan], y[:-1]])
    return 'seasonal' if np.nansum((y - ps) ** 2) < 0.9 * np.nansum((y - pn) ** 2) else 'no season'

def seasonal_benchmark(trials=1000, n=120, noise_std=1.0, period=12):
    methods = {
        'STL': lambda y: stl_detect(y, period),
        'ACF': lambda y: acf_detect(y, period),
        'FFT': lambda y: fft_detect(y, period),
        'Snaive': lambda y: snaive_detect(y, period)
    }
    amps = [5.0, 1.0, 0.0]
    truths = ['seasonal', 'seasonal', 'no season']
    counts = {m: 0 for m in methods}
    total = trials * len(amps)
    for _ in range(trials):
        for a, t in zip(amps, truths):
            y = generate_series(n, a, noise_std, period)
            for name, fn in methods.items():
                if fn(y) == t:
                    counts[name] += 1
    acc = {k: v / total for k, v in counts.items()}
    df = pd.DataFrame(acc.values(), index=acc.keys(), columns=['accuracy']).sort_values('accuracy', ascending=False)
    best = df.index[0]
    best_acc = df.iloc[0, 0]
    print(df.to_string(float_format='{:.4f}'.format))
    print(f'\nBest method: {best} (accuracy {best_acc:.4f})')

if __name__ == '__main__':
    seasonal_benchmark(trials=1000, n=120, noise_std=1.0, period=12)
