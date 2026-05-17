import numpy as np
import pandas as pd
from scipy.special import binom
from statsmodels.tsa.stattools import adfuller
import config   # <-- added

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
    for d in np.arange(d_min, d_max + step, step):
        diffed = frac_diff(series, d)
        diffed = diffed[~np.isnan(diffed)]
        if len(diffed) < 10:
            continue
        try:
            p_val = adfuller(diffed, autolag='AIC')[1]
        except:
            continue
        if p_val < p_thresh:
            return d, diffed
    # If none stationary, return highest d
    return d_max, frac_diff(series, d_max)

def compute_frac_features(returns_df, etf, window, d_opt=None):
    series = returns_df[etf].iloc[-window:].dropna().values
    if d_opt is None:
        d_opt, frac_series = optimal_d(series, config.D_MIN, config.D_MAX, config.D_STEP)
    else:
        frac_series = frac_diff(series, d_opt)
    return frac_series, d_opt
