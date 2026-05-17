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
        print(f"\n=== Universe: {university_name} (Fractional Calculus) ===")
        returns = data_manager.prepare_returns_matrix(df, tickers)
        if returns.empty or len(returns) < max(config.WINDOWS) + config.PREDICTION_WINDOW + 10:
            print("  Insufficient data")
            all_results[universe_name] = {"top_etfs": []}
            continue

        best_per_etf = {}
        window_results = {}

        for win in config.WINDOWS:
            if len(returns) < win + config.PREDICTION_WINDOW + 1:
                print(f"  Skipping window {win}d (insufficient data)")
                continue
            print(f"  Processing window {win}d...")
            etf_pred = {}
            for etf in tickers:
                if etf not in returns.columns:
                    continue
                # Compute optimal d and fractionally differenced series on the window
                series = returns[etf].iloc[-win:].dropna().values
                if len(series) < config.PREDICTION_WINDOW + 1:
                    continue
                d_opt, frac_series = optimal_d(series, config.D_MIN, config.D_MAX, config.D_STEP)
                # Build supervised dataset
                X, y = [], []
                for i in range(config.PREDICTION_WINDOW, len(frac_series)-1):
                    X.append(frac_series[i-config.PREDICTION_WINDOW:i])
                    y.append(series[i+1])
                X = np.array(X); y = np.array(y)
                if len(X) < 10:
                    continue
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                model = Ridge(alpha=config.RIDGE_ALPHA)
                model.fit(X_scaled, y)
                last_X = frac_series[-config.PREDICTION_WINDOW:].reshape(1,-1)
                last_scaled = scaler.transform(last_X)
                pred = model.predict(last_scaled)[0]
                etf_pred[etf] = pred
            window_results[win] = etf_pred
            for etf, pred in etf_pred.items():
                if etf not in best_per_etf or pred > best_per_etf[etf][0]:
                    best_per_etf[etf] = (pred, win)

        if not best_per_etf:
            print("  No predictions")
            all_results[universe_name] = {"top_etfs": []}
            continue

        full_scores = {ticker: {"score": score, "best_window": win} for ticker, (score, win) in best_per_etf.items()}
        sorted_etfs = sorted(best_per_etf.items(), key=lambda x: x[1][0], reverse=True)
        top_etfs = [{"ticker": ticker, "pred_return": float(score), "best_window": win} for ticker, (score, win) in sorted_etfs[:config.TOP_N]]

        print(f"  Top 3 ETFs: {[e['ticker'] for e in top_etfs]}")
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
    print("\n=== Fractional Calculus Engine complete ===")

if __name__ == "__main__":
    main()
