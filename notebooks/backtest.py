import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from typing import Dict, Any
from get_portfolio import build_mvo_short_portfolio, build_simple_cascade_portfolio, fit_cascade_model_and_returns


def _compute_performance_stats(
    port_ret: pd.Series,
    bars_per_year: int = 252 * 78  # 5min bars in US RTH
) -> Dict[str, float]:
    """Compute simple performance statistics from a return series."""
    port_ret = port_ret.dropna()
    if port_ret.empty:
        return {
            "cagr": 0.0,
            "ann_vol": 0.0,
            "sharpe": 0.0,
            "max_dd": 0.0,
        }

    # cumulative equity using LOG returns (not simple returns)
    equity = np.exp(port_ret.cumsum())
    start_date = port_ret.index[0]
    end_date = port_ret.index[-1]
    time_diff = end_date - start_date
    years = time_diff.days / 365.25

    final_equity = equity.iloc[-1]
    cagr = final_equity ** (1.0 / years) - 1.0 if years > 0 else 0.0

    ann_vol = port_ret.std() * np.sqrt(bars_per_year)

    sharpe = cagr / ann_vol if ann_vol > 0 else 0.0

    running_max = equity.cummax()
    dd = equity / running_max - 1.0
    max_dd = dd.min()

    return {
        "cagr": float(cagr),
        "ann_vol": float(ann_vol),
        "sharpe": float(sharpe),
        "max_dd": float(max_dd),
    }


def run_backtest_mvo(
    df: pd.DataFrame,
    risk_col: str = "rv_20",
    top_n_tickers: int = 6,
    lookback_bars: int = 8,
    lookback_cov: int = 50,
    min_prob_cascade: float = 0.3,
    target_gross_exposure: float = 1.0,
    max_weight: float = 0.3,
    min_weight: float = 0.05,
    start_timestamp_index: int = 100,
    max_timestamps: int = -1,
) -> Dict[str, Any]:
    """
    BACKTEST WITH MEAN-VARIANCE OPTIMIZATION (SHORT-ONLY, MARKET NEUTRAL)

    For each timestamp:
    1. Train SEPARATE cascade models for each ticker on historical data
    2. Generate prob_cascade predictions (including SPY for beta calculation)
    3. Build SHORT-ONLY MVO portfolio:
       - Filter stocks by min cascade probability
       - Minimize portfolio variance subject to:
         * Zero beta to SPY (market neutral)
         * Position size limits
         * Target gross exposure
       - All weights are negative (short-only)
    4. Calculate portfolio returns

    Parameters
    ----------
    df : pd.DataFrame
        Must contain SPY and stock data with required columns
    risk_col : str
        Volatility column for fallback weighting
    top_n_tickers : int
        Maximum stocks in portfolio
    lookback_bars : int
        Periods for max cascade probability calculation
    lookback_cov : int
        Periods for covariance matrix estimation
    min_prob_cascade : float
        Minimum cascade probability threshold
    target_gross_exposure : float
        Target sum of absolute weights
    max_weight : float
        Maximum absolute position size
    min_weight : float
        Minimum absolute position size
    start_timestamp_index : int
        First timestamp index to process
    max_timestamps : int
        Maximum timestamps to process (-1 = all)

    Returns
    -------
    Dict with keys:
        - equity_curve : pd.Series, cumulative returns
        - returns : pd.Series, period returns
        - stats : Dict, performance metrics
        - weights : pd.DataFrame, position history
        - betas : pd.DataFrame, portfolio beta to SPY history
    """

    print("=" * 70)
    print("🚀 BACKTEST WITH MEAN-VARIANCE OPTIMIZATION (SHORT-ONLY, MARKET NEUTRAL)")
    print("=" * 70)

    # Verify SPY exists in data
    if "SPY" not in df["ticker"].unique():
        raise ValueError("SPY must be included in input data for market neutrality")

    # Get unique timestamps
    timestamps = sorted(df["timestamp"].unique())

    # Limit to requested range
    if max_timestamps > 0:
        timestamps = timestamps[start_timestamp_index:start_timestamp_index + max_timestamps]
    else:
        timestamps = timestamps[start_timestamp_index:]

    print(f"Timestamps to process: {len(timestamps)}")
    print(f"Date range: {timestamps[0]} to {timestamps[-1]}")
    print(f"Strategy: SHORT-ONLY MVO with SPY beta=0")
    print(f"Top {top_n_tickers} stocks, cascade prob >= {min_prob_cascade}")
    print(f"Covariance lookback: {lookback_cov} bars")
    print(f"ML Models: Separate L1 Logistic per ticker")
    print("=" * 70 + "\n")

    portfolio_returns = []
    all_weights = []
    portfolio_betas = []

    for i, current_ts in enumerate(timestamps):

        if i % 50 == 0:
            print(f"📊 Processing timestamp {i}/{len(timestamps)}: {current_ts}")

        # STEP 1: Get all data BEFORE current timestamp (for training)
        historical_data = df[df["timestamp"] < current_ts].copy()

        if len(historical_data) < max(lookback_bars, lookback_cov):
            continue  # Not enough history

        # STEP 2: Train per-ticker models on historical data
        try:
            model_predictions = fit_cascade_model_and_returns(historical_data)

            if model_predictions.empty:
                print(f"⚠️  No predictions at {current_ts}")
                continue

            # Verify SPY is in predictions
            if "SPY" not in model_predictions["ticker"].unique():
                print(f"⚠️  SPY missing from predictions at {current_ts}")
                continue

        except Exception as e:
            print(f"⚠️  Model training failed at {current_ts}: {e}")
            continue

        # STEP 3: Build MVO portfolio (short-only, market neutral)
        try:
            current_weights = build_mvo_short_portfolio(
                df=model_predictions,
                lookback_bars=lookback_bars,
                risk_col=risk_col,
                top_n_tickers=top_n_tickers,
                min_prob_cascade=min_prob_cascade,
                target_gross_exposure=target_gross_exposure,
                max_weight=max_weight,
                min_weight=min_weight,
                lookback_cov=lookback_cov,
            )

            if current_weights.empty:
                continue

        except Exception as e:
            print(f"⚠️  MVO failed at {current_ts}: {e}")
            continue

        # STEP 4: Get next timestamp data for return calculation
        next_ts_idx = timestamps.index(current_ts) + 1
        if next_ts_idx >= len(timestamps):
            break

        next_ts = timestamps[next_ts_idx]
        next_data = df[df["timestamp"] == next_ts].copy()

        if next_data.empty:
            continue

        # STEP 5: Calculate portfolio return (short-only)
        merged = current_weights.merge(
            next_data[["ticker", "logret"]],
            on="ticker",
            how="inner"
        )

        if merged.empty:
            continue

        # Portfolio return = sum of (weight * return)
        # All weights are negative (short positions)
        portfolio_return = (merged["weight"] * merged["logret"]).sum()

        # Calculate portfolio beta from current_weights (already has beta_to_spy)
        portfolio_beta = (current_weights["weight"] * current_weights["beta_to_spy"]).sum()

        # Store weight snapshot
        current_weights["timestamp"] = current_ts

        portfolio_returns.append({
            "timestamp": current_ts,
            "return": portfolio_return,
            "gross_exposure": current_weights["weight"].abs().sum(),
            "net_exposure": current_weights["weight"].sum(),
            "num_positions": len(current_weights),
            "portfolio_beta": portfolio_beta,
        })

        all_weights.append(current_weights)
        portfolio_betas.append({
            "timestamp": current_ts,
            "portfolio_beta": portfolio_beta,
        })

    # Final summary
    print(f"\n{'='*70}")
    print(f"✅ Backtest Complete!")
    print(f"{'='*70}")
    print(f"Total timestamps processed: {len(portfolio_returns)}")
    print(f"Total weight records: {sum(len(w) for w in all_weights)}")
    print(f"{'='*70}\n")

    if not portfolio_returns:
        print("⚠️  No portfolio returns generated!")
        return {
            "equity_curve": pd.Series(dtype=float),
            "returns": pd.Series(dtype=float),
            "stats": _compute_performance_stats(pd.Series(dtype=float)),
            "weights": pd.DataFrame(),
            "betas": pd.DataFrame(),
        }

    # Convert to DataFrames
    returns_df = pd.DataFrame(portfolio_returns)
    betas_df = pd.DataFrame(portfolio_betas)

    returns_series = pd.Series(
        data=returns_df["return"].values,
        index=returns_df["timestamp"].values,
        name="portfolio_return"
    )

    # Build equity curve using LOG returns
    equity_curve = np.exp(returns_series.cumsum())

    # Calculate performance stats
    stats = _compute_performance_stats(returns_series)

    # Add portfolio-specific stats
    stats["avg_gross_exposure"] = returns_df["gross_exposure"].mean()
    stats["avg_net_exposure"] = returns_df["net_exposure"].mean()
    stats["avg_num_positions"] = returns_df["num_positions"].mean()
    stats["avg_portfolio_beta"] = returns_df["portfolio_beta"].mean()
    stats["max_abs_beta"] = returns_df["portfolio_beta"].abs().max()

    # Combine all weights
    weights_combined = pd.concat(all_weights, ignore_index=True) if all_weights else pd.DataFrame()

    # Plot equity curve and beta
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Equity curve
    equity_curve.plot(ax=ax1, linewidth=2, color='darkblue')
    ax1.set_title("SHORT-ONLY MVO Equity Curve (Market Neutral)", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Cumulative Return")
    ax1.grid(True, alpha=0.3)

    # Portfolio beta over time
    betas_df.set_index("timestamp")["portfolio_beta"].plot(ax=ax2, linewidth=2, color='darkred')
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.set_title("Portfolio Beta to SPY", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Beta")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Print stats
    print("\n📈 PERFORMANCE STATISTICS:")
    print("=" * 70)
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"{k:25s}: {v:>12.4f}")
        else:
            print(f"{k:25s}: {v}")
    print("=" * 70)

    return {
        "equity_curve": equity_curve,
        "returns": returns_series,
        "stats": stats,
        "weights": weights_combined,
        "betas": betas_df,
    }

def run_backtest_risk_parity(
    df: pd.DataFrame,
    risk_col: str = "rv_20",
    top_n_tickers: int = 10,
    lookback_bars: int = 8,
    lookback_cov: int = 50,
    target_gross_exposure: float = 1.0,
    max_weight: float = 0.2,
    min_weight: float = 0.02,
    start_timestamp_index: int = 100,
    max_timestamps: int = -1,
) -> Dict[str, Any]:
    """
    BACKTEST WITH RISK PARITY (LONG-ONLY, LOW CASCADE STOCKS)

    For each timestamp:
    1. Train cascade models on historical data
    2. Select stocks with LOWEST cascade probability (most stable)
    3. Build risk parity portfolio (equal risk contribution)
    4. Calculate portfolio returns

    Parameters
    ----------
    df : pd.DataFrame
        Stock data with required columns
    risk_col : str
        Volatility column for weighting
    top_n_tickers : int
        Maximum stocks in portfolio
    lookback_bars : int
        Periods for cascade probability calculation
    lookback_cov : int
        Periods for covariance matrix estimation
    target_gross_exposure : float
        Target sum of weights (should be 1.0)
    max_weight : float
        Maximum position size
    min_weight : float
        Minimum position size
    start_timestamp_index : int
        First timestamp index to process
    max_timestamps : int
        Maximum timestamps to process (-1 = all)

    Returns
    -------
    Dict with keys:
        - equity_curve : pd.Series
        - returns : pd.Series
        - stats : Dict
        - weights : pd.DataFrame
    """

    print("=" * 70)
    print("🚀 BACKTEST WITH RISK PARITY (LONG-ONLY, LOW CASCADE)")
    print("=" * 70)

    timestamps = sorted(df["timestamp"].unique())

    if max_timestamps > 0:
        timestamps = timestamps[start_timestamp_index:start_timestamp_index + max_timestamps]
    else:
        timestamps = timestamps[start_timestamp_index:]

    print(f"Timestamps to process: {len(timestamps)}")
    print(f"Date range: {timestamps[0]} to {timestamps[-1]}")
    print(f"Strategy: LONG-ONLY Risk Parity")
    print(f"Top {top_n_tickers} STABLE stocks (lowest cascade prob)")
    print(f"Covariance lookback: {lookback_cov} bars")
    print("=" * 70 + "\n")

    portfolio_returns = []
    all_weights = []

    for i, current_ts in enumerate(timestamps):

        if i % 50 == 0:
            print(f"📊 Processing timestamp {i}/{len(timestamps)}: {current_ts}")

        historical_data = df[df["timestamp"] < current_ts].copy()

        if len(historical_data) < max(lookback_bars, lookback_cov):
            continue

        try:
            model_predictions = fit_cascade_model_and_returns(historical_data)
            if model_predictions.empty:
                continue
        except Exception as e:
            print(f"⚠️  Model training failed at {current_ts}: {e}")
            continue

        try:

            current_weights = build_simple_cascade_portfolio(
                df=model_predictions,
                lookback_bars=lookback_bars,
                risk_col=risk_col,
                top_n_tickers=top_n_tickers,
                target_gross_exposure=target_gross_exposure,
                max_weight=max_weight,
            )

            if current_weights.empty:
                continue

        except Exception as e:
            print(f"⚠️  Risk parity failed at {current_ts}: {e}")
            continue

        next_ts_idx = timestamps.index(current_ts) + 1
        if next_ts_idx >= len(timestamps):
            break

        next_ts = timestamps[next_ts_idx]
        next_data = df[df["timestamp"] == next_ts].copy()

        if next_data.empty:
            continue

        merged = current_weights.merge(
            next_data[["ticker", "logret"]],
            on="ticker",
            how="inner"
        )

        if merged.empty:
            continue

        portfolio_return = (merged["weight"] * merged["logret"]).sum()

        current_weights["timestamp"] = current_ts

        portfolio_returns.append({
            "timestamp": current_ts,
            "return": portfolio_return,
            "gross_exposure": current_weights["weight"].abs().sum(),
            "net_exposure": current_weights["weight"].sum(),
            "num_positions": len(current_weights),
        })

        all_weights.append(current_weights)

    print(f"\n{'='*70}")
    print(f"✅ Backtest Complete!")
    print(f"{'='*70}")
    print(f"Total timestamps processed: {len(portfolio_returns)}")
    print(f"{'='*70}\n")

    if not portfolio_returns:
        return {
            "equity_curve": pd.Series(dtype=float),
            "returns": pd.Series(dtype=float),
            "stats": _compute_performance_stats(pd.Series(dtype=float)),
            "weights": pd.DataFrame(),
        }

    returns_df = pd.DataFrame(portfolio_returns)

    returns_series = pd.Series(
        data=returns_df["return"].values,
        index=returns_df["timestamp"].values,
        name="portfolio_return"
    )

    equity_curve = np.exp(returns_series.cumsum())
    stats = _compute_performance_stats(returns_series)

    stats["avg_gross_exposure"] = returns_df["gross_exposure"].mean()
    stats["avg_net_exposure"] = returns_df["net_exposure"].mean()
    stats["avg_num_positions"] = returns_df["num_positions"].mean()

    weights_combined = pd.concat(all_weights, ignore_index=True) if all_weights else pd.DataFrame()

    fig, ax = plt.subplots(figsize=(12, 6))
    equity_curve.plot(ax=ax, linewidth=2, color='darkgreen')
    ax.set_title("LONG-ONLY Risk Parity Equity Curve", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    print("\n📈 PERFORMANCE STATISTICS:")
    print("=" * 70)
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"{k:25s}: {v:>12.4f}")
        else:
            print(f"{k:25s}: {v}")
    print("=" * 70)

    return {
        "equity_curve": equity_curve,
        "returns": returns_series,
        "stats": stats,
        "weights": weights_combined,
    }