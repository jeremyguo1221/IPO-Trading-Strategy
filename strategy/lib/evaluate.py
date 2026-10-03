"""Performance metrics for one backtest run, plus the Deflated Sharpe Ratio for multiple-trial correction."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from lib import zeus_metrics as zm


def metrics(res: dict) -> dict:
    tr, eq = res["trades"], res["equity"]
    if tr is None or tr.empty or eq.empty:
        return {"trades": 0}
    eq_full = pd.concat([pd.Series({eq.index[0] - pd.Timedelta(days=1): 1.0}), eq])
    daily = eq_full.pct_change().dropna()
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    total = eq.iloc[-1] - 1
    cagr = eq.iloc[-1] ** (1 / years) - 1 if years > 0.25 and eq.iloc[-1] > 0 else np.nan
    by_year = eq_full.groupby(eq_full.index.year).last()
    prev = by_year.shift(1).fillna(1.0)
    yearly = (by_year / prev - 1)
    ledger = res["ledger"]
    boot = zm.stationary_bootstrap_p_value(daily, n_resamples=2000, seed=11)
    r = tr.net_ret
    return {
        "trades": int(len(tr)),
        "start": str(eq.index[0].date()), "end": str(eq.index[-1].date()),
        "total_return": round(total * 100, 2), "cagr": round(cagr * 100, 2) if np.isfinite(cagr) else None,
        "sharpe": round(zm.sharpe_ratio(daily), 3), "sortino": round(min(zm.sortino_ratio(daily), 99), 3),
        "max_dd": round(zm.max_drawdown(eq_full) * 100, 2), "calmar": round(zm.calmar_ratio(daily), 3),
        "win_rate": round(zm.win_rate(r) * 100, 1), "profit_factor": round(min(zm.profit_factor(tr.net_pnl), 99), 3),
        "mean_trade": round(r.mean() * 100, 2), "median_trade": round(r.median() * 100, 2),
        "best_trade": round(r.max() * 100, 1), "worst_trade": round(r.min() * 100, 1),
        "trade_t": round(r.mean() / (r.std(ddof=1) / np.sqrt(len(r))), 2) if len(r) > 2 and r.std() > 0 else None,
        "avg_gross": round(ledger.gross_short.mean() * 100, 1), "avg_positions": round(ledger.open_positions.mean(), 2),
        "pct_stopped": round((tr.exit_reason == "stop").mean() * 100, 1),
        "yearly": {int(y): round(v * 100, 2) for y, v in yearly.items()},
        "pct_years_pos": round((yearly > 0).mean() * 100, 1),
        "boot_p": round(boot["p_value"], 4), "n_days": int(len(daily)),
        "daily_sr": float(daily.mean() / daily.std(ddof=1)) if daily.std() > 0 else 0.0,
        "skew": float(stats.skew(daily)), "kurt": float(stats.kurtosis(daily, fisher=False)),
    }


def deflated_sharpe(sr: float, n_obs: int, skew: float, kurt: float, trial_srs: list[float]) -> float:
    """Bailey & Lopez de Prado (2014) Deflated Sharpe Ratio: probability that the true (per-period) Sharpe
    exceeds the maximum expected from `len(trial_srs)` independent trials under the null.
    All Sharpe values in per-observation (daily) units."""
    n = len(trial_srs)
    if n < 2 or n_obs < 3:
        return float("nan")
    var = float(np.var(trial_srs, ddof=1))
    g = 0.5772156649
    sr0 = np.sqrt(var) * ((1 - g) * stats.norm.ppf(1 - 1 / n) + g * stats.norm.ppf(1 - 1 / (n * np.e)))
    denom = np.sqrt(max(1e-12, 1 - skew * sr + (kurt - 1) / 4 * sr ** 2))
    return float(stats.norm.cdf((sr - sr0) * np.sqrt(n_obs - 1) / denom))
