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
        if returns.empty or len(returns) < max(config.WINDOWS) + config.PRED_WINDOW + 10:
            print("  Insufficient data")
            all_results[universe_name] = {"top_etfs": []}
            continue

        best_per_etf = {}   # ticker -> (best_pred, best_window)
        window_results = {} # win -> dict of predictions

        for win in config.WINDOWS:
            if len(returns) < win + config.PRED_WINDOW + 10:
                print(f"  Skipping window {win}d (insufficient data)")
                continue
            print(f"  Processing window {win}d...")
            etf_pred = {}
            for etf in tickers:
                if etf not in returns.columns:
                    continue
                # Compute fractional features on the last `win` days
                frac_series, d_opt = compute_frac_features(returns, etf, win, d_opt=None)
                raw_returns = returns[etf].iloc[-win:].values
                # Trim to same length
                min_len = min(len(frac_series), len(raw_returns))
                frac_series = frac_series[:min_len]
                raw_returns = raw_returns[:min_len]
                if len(frac_series) < config.PRED_WINDOW + 5:
                    continue
                # Build sliding window training data
                X = []
                y = []
                for i in range(config.PRED_WINDOW, len(frac_series)-1):
                    X.append(frac_series[i-config.PRED_WINDOW:i])
                    y.append(raw_returns[i+1])
                X = np.array(X)
                y = np.array(y)
                if len(X) < 20:
                    continue
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                model = Ridge(alpha=config.RIDGE_ALPHA)
                model.fit(X_scaled, y)
                # Predict for most recent window
                last_X = frac_series[-config.PRED_WINDOW:].reshape(1, -1)
                last_X_scaled = scaler.transform(last_X)
                pred = model.predict(last_X_scaled)[0]
                etf_pred[etf] = pred
            window_results[win] = etf_pred
            for etf, pred in etf_pred.items():
                if etf not in best_per_etf or pred > best_per_etf[etf][0]:
                    best_per_etf[etf] = (pred, win)

        if not best_per_etf:
            print("  No valid predictions")
            all_results[universe_name] = {"top_etfs": []}
            continue

        sorted_etfs = sorted(best_per_etf.items(), key=lambda x: x[1][0], reverse=True)
        top_etfs = []
        full_scores = {}
        for ticker, (pred, win) in sorted_etfs[:config.TOP_N]:
            top_etfs.append({"ticker": ticker, "pred_return": float(pred), "best_window": win})
            full_scores[ticker] = {"score": float(pred), "best_window": win}
        print(f"  Top 3 ETFs by best window: {[(e['ticker'], e['pred_return'], e['best_window']) for e in top_etfs]}")
        all_results[universe_name] = {
            "top_etfs": top_etfs,
            "full_scores": full_scores,
            "window_results": window_results,
            "run_date": today
        }

    Path("results").mkdir(exist_ok=True)
    local_path = Path(f"results/frac_calc_{today}.json")
    with open(local_path, "w") as f:
        json.dump({"run_date": today, "universes": all_results}, f, indent=2)

    import push_results
    push_results.push_daily_result(local_path)
    print("\n=== Fractional Calculus Engine (multi‑window) complete ===")

if __name__ == "__main__":
    main()
