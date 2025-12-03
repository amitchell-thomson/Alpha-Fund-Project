import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from scipy.optimize import minimize
from typing import Dict, Any

steps_since_retrain = {}
models = {}


def fit_cascade_model_and_returns(
    df: pd.DataFrame,
    min_samples: int = 10,
    lookback_train: int = 10000,
    retrain_interval: int = 25,
) -> pd.DataFrame:
    """Train separate cascade model for EACH ticker, retraining only every `retrain_interval` steps."""
    global steps_since_retrain, models

    drop_like = ["fwd_"]
    drop_exact = ["label_cascade", "timestamp", "ticker"]
    cand = [c for c in df.columns if c not in drop_exact and not any(dl in c for dl in drop_like)]
    features = df[cand].select_dtypes(include=[np.number]).columns.tolist()

    forward_looking = [f for f in features if "fwd_" in f or f == "label_cascade"]
    if forward_looking:
        raise ValueError(f"Forward-looking features detected: {forward_looking}")

    all_predictions = []

    for ticker in df["ticker"].unique():
        ticker_data = df[df["ticker"] == ticker].copy()

        mask = ticker_data[features].notna().all(axis=1) & ticker_data["label_cascade"].notna()
        data = ticker_data.loc[mask].copy()

        prob_cascade = np.zeros(len(data))
        expected_return = np.zeros(len(data))

        if len(data) > 0:
            # Initialize counter if needed
            if ticker not in steps_since_retrain:
                steps_since_retrain[ticker] = retrain_interval  # Force initial train

            # Check if we should retrain
            should_retrain = steps_since_retrain[ticker] >= retrain_interval

            if should_retrain:
                train_data = data.tail(lookback_train).copy()

                if len(train_data) >= min_samples and train_data["label_cascade"].nunique() >= 2:
                    X_train = train_data[features].values
                    y_train = train_data["label_cascade"].values

                    try:
                        model = make_pipeline(
                            StandardScaler(),
                            LogisticRegression(penalty="l1", solver="liblinear", C=0.1, max_iter=500)
                        )
                        model.fit(X_train, y_train)
                        models[ticker] = model
                        steps_since_retrain[ticker] = 0
                    except Exception as e:
                        print(f"⚠️ Model train failed for {ticker}: {e}")

            # Use existing model if available
            if ticker in models:
                try:
                    X_full = data[features].values
                    prob_cascade = models[ticker].predict_proba(X_full)[:, 1]
                    historical_cascade_data = data[data["label_cascade"] == 1]
                    avg_cascade_magnitude = historical_cascade_data["fwd_dd_log_H"].abs().mean()

                    # Use this average for all predictions
                    expected_return = -prob_cascade * avg_cascade_magnitude
                except Exception as e:
                    print(f"⚠️ Prediction failed for {ticker}: {e}")

            steps_since_retrain[ticker] += 1

        ticker_predictions = pd.DataFrame({
            "timestamp": data["timestamp"].values,
            "ticker": ticker,
            "prob_cascade": prob_cascade,
            "expected_return": expected_return,
            "rv_20": data["rv_20"].values,
            "logret": data["logret"].values,
            "close": data["close"].values,
        })

        all_predictions.append(ticker_predictions)

    if not all_predictions:
        return pd.DataFrame(columns=[
            "timestamp", "ticker", "prob_cascade", "expected_return",
            "rv_20", "logret", "close"
        ])

    result = pd.concat(all_predictions, ignore_index=True)
    return result.sort_values(["timestamp", "ticker"]).reset_index(drop=True)


def build_simple_cascade_portfolio(
    df: pd.DataFrame,
    lookback_bars: int = 8,
    risk_col: str = "rv_20",
    top_n_tickers: int = 6,
    min_risk: float = 1e-6,
    max_weight: float = 0.3,
    min_prob_cascade: float = 0.1,  # CHANGED: filter by cascade probability
    target_gross_exposure: float = 1.0,
) -> pd.DataFrame:
    """
    SHORT-ONLY CASCADE PORTFOLIO.

    Only shorts stocks with high downward cascade probability.
    Weights are negative and sum to -target_gross_exposure.
    """
    df = df.sort_values(["ticker", "timestamp"])

    # STEP 1: Get LAST observation for each ticker
    latest = df.groupby("ticker").tail(1).reset_index(drop=True)

    # STEP 2: For each ticker, get max cascade prob from RECENT history
    recent_max = []
    for ticker in latest["ticker"]:
        ticker_history = df[df["ticker"] == ticker].tail(lookback_bars)
        max_prob = ticker_history["prob_cascade"].max()
        recent_max.append(max_prob)

    latest["max_prob_cascade"] = recent_max

    # STEP 3: Filter by cascade probability (not expected_return)
    latest = latest[
        latest["max_prob_cascade"] >= min_prob_cascade
    ].copy()

    if latest.empty:
        return pd.DataFrame(columns=["ticker", "weight"])

    # STEP 4: Select top N by cascade probability
    selected = latest.nlargest(top_n_tickers, "max_prob_cascade").copy()

    # STEP 5: Base weights (inverse volatility)
    volatility = selected[risk_col].clip(lower=min_risk)
    inv_vol = 1.0 / volatility
    base_weights = inv_vol / inv_vol.sum()

    # STEP 6: Apply position limits
    base_weights = np.minimum(base_weights, max_weight)
    base_weights = base_weights / base_weights.sum()

    # STEP 7: Make all weights NEGATIVE (short-only)
    short_weights = -base_weights * target_gross_exposure

    return pd.DataFrame({
        "ticker": selected["ticker"].values,
        "weight": short_weights,
        "prob_cascade": selected["max_prob_cascade"].values,
        "expected_return": selected["expected_return"].values,
        "volatility": volatility.values,
    })


from scipy.optimize import minimize


def build_mvo_short_portfolio(
        df: pd.DataFrame,
        lookback_bars: int = 8,
        risk_col: str = "rv_20",
        top_n_tickers: int = 6,
        min_prob_cascade: float = 0.1,
        target_gross_exposure: float = 1.0,
        max_weight: float = 0.3,
        min_weight: float = 0.05,
        lookback_cov: int = 50,
    ) -> pd.DataFrame:
        """
        Build SHORT-ONLY portfolio with SPY hedge for market neutrality.
        Optimizes with positive weights, then negates for short positions.
        """
        df = df.sort_values(["ticker", "timestamp"])

        df_stocks = df[df["ticker"] != "SPY"].copy()
        df_spy = df[df["ticker"] == "SPY"].copy()

        if df_spy.empty:
            raise ValueError("SPY required for beta hedging")

        # 1. Filter by minimum cascade probability
        latest = df_stocks.groupby("ticker").tail(1).reset_index(drop=True)

        recent_max = []
        for ticker in latest["ticker"]:
            ticker_history = df_stocks[df_stocks["ticker"] == ticker].tail(lookback_bars)
            max_prob = ticker_history["prob_cascade"].max()
            recent_max.append(max_prob)

        latest["max_prob_cascade"] = recent_max
        latest = latest[latest["max_prob_cascade"] >= min_prob_cascade].copy()

        if latest.empty or len(latest) < 2:
            return pd.DataFrame(columns=["ticker", "weight", "beta_to_spy"])

        # 2. Select top N stocks
        selected = latest.nlargest(top_n_tickers, "max_prob_cascade").copy()
        tickers = selected["ticker"].values
        n = len(tickers)

        # 3. Build covariance matrix
        returns_list = []
        for ticker in tickers:
            ticker_data = df_stocks[df_stocks["ticker"] == ticker].tail(lookback_cov)
            if "logret" in ticker_data.columns:
                returns_list.append(ticker_data["logret"].dropna().values[-lookback_cov:])
            else:
                prices = ticker_data["close"].values
                log_rets = np.log(prices[1:] / prices[:-1])
                returns_list.append(log_rets[-lookback_cov:])

        min_len = min(len(r) for r in returns_list)
        returns_matrix = np.column_stack([r[-min_len:] for r in returns_list])
        cov_matrix = np.cov(returns_matrix, rowvar=False) + np.eye(n) * 1e-6

        # 4. Calculate betas to SPY
        spy_data = df_spy.tail(lookback_cov + 1)
        if "logret" in spy_data.columns:
            spy_returns = spy_data["logret"].dropna().values[-min_len:]
        else:
            spy_prices = spy_data["close"].values
            spy_returns = np.log(spy_prices[1:] / spy_prices[:-1])[-min_len:]

        betas = []
        for i in range(n):
            cov_with_spy = np.cov(returns_matrix[:, i], spy_returns)[0, 1]
            var_spy = np.var(spy_returns)
            beta = cov_with_spy / var_spy if var_spy > 0 else 0.0
            betas.append(beta)

        betas = np.array(betas)
        expected_returns = selected["expected_return"].values

        # 5. Optimization with POSITIVE weights
        # Objective: maximize (expected_return - variance)
        # Since we'll negate weights, we want to MAXIMIZE expected_return (which is negative)
        def objective(w):
            portfolio_return = np.dot(w, expected_returns)
            portfolio_variance = w @ cov_matrix @ w
            return portfolio_return + 0.5 * portfolio_variance

        # 6. Constraints
        constraints = [
            {"type": "eq", "fun": lambda w: w.sum() - target_gross_exposure},
            # Beta neutrality AFTER negation: sum(-w_i * beta_i) + w_spy * 1.0 = 0
            # => w_spy = sum(w_i * beta_i)
            # Since we'll set w_spy = -sum(-w_i * beta_i) = sum(w_i * beta_i)
            # This constraint ensures: -sum(w_i * beta_i) + sum(w_i * beta_i) = 0
            # So we just need to track it, no explicit constraint needed here
            # because SPY weight will be calculated as: w_spy = sum(w_i * beta_i)
        ]

        bounds = [(min_weight, max_weight) for _ in range(n)]

        # 7. Initial guess
        volatility = selected[risk_col].clip(lower=1e-6).values
        inv_vol = 1.0 / volatility
        x0 = inv_vol / inv_vol.sum() * target_gross_exposure
        x0 = np.clip(x0, min_weight, max_weight)
        x0 = x0 / x0.sum() * target_gross_exposure

        # 8. Optimize
        result = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 2000, "ftol": 1e-9}
        )

        if not result.success:
            print(f"⚠️  Optimization failed: {result.message}, using fallback")
            optimized_positive_weights = x0
        else:
            optimized_positive_weights = result.x

        # 9. Negate weights for short positions
        stock_weights = -optimized_positive_weights

        # 10. Calculate SPY hedge
        # After negation: portfolio beta = sum(-w_i * beta_i)
        # SPY weight needed: w_spy = -portfolio_beta = sum(w_i * beta_i)
        portfolio_beta_from_stocks = np.dot(stock_weights, betas)
        spy_weight = -portfolio_beta_from_stocks  # This equals sum(optimized_positive_weights * betas)

        # 11. Build final portfolio
        stock_portfolio = pd.DataFrame({
            "ticker": tickers,
            "weight": stock_weights,
            "prob_cascade": selected["max_prob_cascade"].values,
            "expected_return": expected_returns,
            "beta_to_spy": betas,
        })

        spy_portfolio = pd.DataFrame({
            "ticker": ["SPY"],
            "weight": [spy_weight],
            "prob_cascade": [0.0],
            "expected_return": [0.0],
            "beta_to_spy": [1.0],
        })

        final_portfolio = pd.concat([stock_portfolio, spy_portfolio], ignore_index=True)

        # Validate beta neutrality
        final_beta = (final_portfolio["weight"] * final_portfolio["beta_to_spy"]).sum()
        if abs(final_beta) > 0.2:
            print(f"⚠️  Final portfolio beta = {final_beta:.3f}")

        return final_portfolio

