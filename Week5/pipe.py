import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from statsmodels.tsa.stattools import adfuller

def analyze_time_series(ts, seasonal_lag):
    ts = ts.dropna()
    n = len(ts)
    X = np.arange(n).reshape(-1, 1)
    y = ts.values.reshape(-1, 1)
    lr = LinearRegression().fit(X, y)
    y_pred = lr.predict(X)
    r2_linear = r2_score(y, y_pred)
    log_ts = np.log(ts[ts > 0])
    n_log = len(log_ts)
    X_log = np.arange(n_log).reshape(-1, 1)
    y_log = log_ts.values.reshape(-1, 1)
    lr_log = LinearRegression().fit(X_log, y_log)
    y_log_pred = lr_log.predict(X_log)
    r2_exponential = r2_score(y_log, y_log_pred)
    if seasonal_lag < n:
        autocorr = ts.autocorr(lag=seasonal_lag)
    else:
        autocorr = np.nan
    adf_result = adfuller(ts)
    p_value = adf_result[1]
    linear_present = r2_linear > 0.5
    seasonal_present = ~np.isnan(autocorr) and abs(autocorr) > 0.5
    stationary_present = p_value < 0.05
    exponential_present = r2_exponential > 0.5
    return {
        'r2_linear': r2_linear,
        'autocorr_seasonal_lag': autocorr,
        'adf_p_value': p_value,
        'r2_exponential': r2_exponential,
        'linear_trend_present': linear_present,
        'seasonal_trend_present': seasonal_present,
        'stationary_pattern_present': stationary_present,
        'exponential_trend_present': exponential_present
    }
