import numpy as np
import pandas as pd
from scipy.special import binom
from statsmodels.tsa.stattools import adfuller

def frac_diff(series, d, thresh=1e-5):
    """
    Grünwald‑Letnikov fractional differentiation of a time series.
    Returns fractionally differenced series.
    """
    # Compute weights: w_k = binom(d, k) * (-1)^k
    weights = [1.0]
    for k in range(1, len(series)):
        w = -weights[-1] * (d - k + 1) / k
        if abs(w) < thresh:
            break
        weights.append(w)
    weights = np.array(weights)
    # Apply convolution
    diff = np.convolve(series, weights, mode='full')[:len(series)]
    return diff

def optimal_d(series, d_min=0.0, d_max=1.0, step=0.05, p_thresh=0.05):
    """
    Find smallest d such that the fractionally differenced series is stationary (ADF p < 0.05).
    Returns optimal d and the differenced series.
    """
    best_d = d_min
    best_series = series
    # We search from low to high d (increasing stationarity)
    for d in np.arange(d_min, d_max + step, step):
        diffed = frac_diff(series, d)
        # Drop initial NaNs (if any)
        diffed = diffed[~np.isnan(diffed)]
        if len(diffed) < 10:
            continue
        p_val = adfuller(diffed, autolag='AIC')[1]
        if p_val < p_thresh:
            return d, diffed
    # If none stationary, return highest d
    return d_max, frac_diff(series, d_max)

def compute_frac_features(returns_df, etf, window, d_opt=None):
    """
    For a single ETF, compute fractionally differenced series over the last `window` days.
    If d_opt is None, compute optimal d on that window.
    Returns (frac_series, optimal_d).
    """
    series = returns_df[etf].iloc[-window:].dropna().values
    if d_opt is None:
        d_opt, frac_series = optimal_d(series, config.D_MIN, config.D_MAX, config.D_STEP)
    else:
        frac_series = frac_diff(series, d_opt)
    return frac_series, d_opt
