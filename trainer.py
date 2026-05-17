import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import config
import data_manager
from fractional_deriv import compute_frac_features

def main():
    if not config.HF_TOKEN:
        print("HF_TOKEN not set")
        return

    df = data_manager.load_master_data()
    all_results = {}
    today = datetime.now().strftime("%Y-%m-%d")

    for universe_name, tickers in config.UNIVERSES.items():
        print(f"\n=== Universe: {universe_name} (Fractional Calculus) ===")
        returns = data_manager.prepare_returns_matrix(df, tickers)
        if returns.empty or len(returns) < config.OPT_D_WINDOW + config.PREDICTION_WINDOW + 10:
            print("  Insufficient data")
            all_results[universe_name] = {"top_etfs": []}
            continue

        # For each ETF, find optimal d on the last OPT_D_WINDOW days, then predict next day
        predictions = {}
        full_scores = {}
        for etf in tickers:
            if etf not in returns.columns:
                continue
            # Use last OPT_D_WINDOW days to find optimal d (if needed)
            frac_series, d_opt = compute_frac_features(returns, etf, config.OPT_D_WINDOW, d_opt=None)
            # Now we have fractionally differenced series; we need to predict next day's return.
            # We'll use the most recent PREDICTION_WINDOW points of the fractionally differenced series
            # as features for a ridge regression. But we need training data.
            # Build training dataset from the same window: use sliding window of size PREDICTION_WINDOW
            # to predict the next raw return.
            # However, we only have one time series. We need a supervised dataset.
            # We'll create features X = last PREDICTION_WINDOW fractional differences, y = next day raw return.
            # We'll slide over the last OPT_D_WINDOW days.
            # First, align indices
            raw_returns = returns[etf].iloc[-config.OPT_D_WINDOW:].values
            if len(frac_series) != len(raw_returns):
                # Trim to same length
                min_len = min(len(frac_series), len(raw_returns))
                frac_series = frac_series[:min_len]
                raw_returns = raw_returns[:min_len]
            # Build sliding window
            X = []
            y = []
            for i in range(config.PREDICTION_WINDOW, len(frac_series)-1):
                X.append(frac_series[i-config.PREDICTION_WINDOW:i])
                y.append(raw_returns[i+1])   # next day's raw return
            X = np.array(X)
            y = np.array(y)
            if len(X) < 20:
                continue
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            model = Ridge(alpha=config.RIDGE_ALPHA)
            model.fit(X_scaled, y)
            # Predict for the most recent window (last PREDICTION_WINDOW points of frac_series)
            if len(frac_series) < config.PREDICTION_WINDOW:
                continue
            last_X = frac_series[-config.PREDICTION_WINDOW:].reshape(1, -1)
            last_X_scaled = scaler.transform(last_X)
            pred = model.predict(last_X_scaled)[0]
            predictions[etf] = pred
            full_scores[etf] = pred

        if not predictions:
            print("  No predictions")
            all_results[universe_name] = {"top_etfs": []}
            continue

        sorted_etfs = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        top_etfs = []
        for ticker, pred in sorted_etfs[:config.TOP_N]:
            top_etfs.append({"ticker": ticker, "pred_return": float(pred)})
        print(f"  Top 3 ETFs by predicted return: {[e['ticker'] for e in top_etfs]}")
        all_results[universe_name] = {
            "top_etfs": top_etfs,
            "full_scores": full_scores,
            "run_date": today
        }

    Path("results").mkdir(exist_ok=True)
    local_path = Path(f"results/frac_calc_{today}.json")
    with open(local_path, "w") as f:
        json.dump({"run_date": today, "universes": all_results}, f, indent=2)

    import push_results
    push_results.push_daily_result(local_path)
    print("\n=== Fractional Calculus Engine complete ===")

if __name__ == "__main__":
    main()
