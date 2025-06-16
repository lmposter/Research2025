import pandas as pd
import numpy as np
from scipy.stats import spearmanr, skew
from scipy.fft import fft

def process(x, y):
    spearman_corr, spearman_p = spearmanr(x, y)
    monotonic_present = abs(spearman_corr) > 0.7 and spearman_p < 0.05

    y_detrended = y.to_numpy() - np.mean(y.to_numpy())
    fft_values = fft(y_detrended)
    frequencies = np.fft.fftfreq(len(y_detrended))
    fft_amplitude = np.abs(fft_values)
    fft_amplitude[0] = 0
    dominant_freq = frequencies[np.argmax(fft_amplitude)]
    periodic_present = np.max(fft_amplitude) > np.mean(fft_amplitude) + 2 * np.std(fft_amplitude)

    q1 = np.percentile(y, 25)
    q3 = np.percentile(y, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = y[(y < lower_bound) | (y > upper_bound)]
    outliers_present = not outliers.empty

    skewness_value = skew(y)
    skewness_present = abs(skewness_value) > 0.5

    print("Monotonic Trend:")
    print("  Present:", monotonic_present)
    print("  Spearman Correlation:", spearman_corr)

    print("\nPeriodic Pattern:")
    print("  Present:", periodic_present)
    print("  Dominant Frequency:", dominant_freq)

    print("\nOutliers:")
    print("  Present:", outliers_present)
    print("  Outlier Values:", outliers.tolist())

    print("\nSkewness:")
    print("  Present:", skewness_present)
    print("  Skewness Value:", skewness_value)
