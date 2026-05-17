# Fractional Calculus Engine

Applies Grünwald‑Letnikov fractional differentiation (order d between 0 and 1) to ETF return series. Optimal d is chosen to achieve stationarity (ADF test). The fractionally differenced series preserves long memory while being stationary. A ridge regression model uses recent fractional differences to predict next‑day returns.

- **Fractional order search:** 0.0 to 1.0, step 0.05
- **Stationarity test:** Augmented Dickey‑Fuller (p < 0.05)
- **Prediction window:** 20 days of fractional features
- **Output:** top 3 ETFs per universe by predicted return

Runs daily on GitHub Actions.

## Local execution

```bash
pip install -r requirements.txt
export HF_TOKEN=<your_token>
python trainer.py
streamlit run streamlit_app.py
