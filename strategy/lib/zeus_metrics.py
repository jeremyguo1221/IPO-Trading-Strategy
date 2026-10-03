r"""Verbatim copy of selected pure functions from the Zeus Trading Agent repo.

Source: D:\Coding Projects\Zeus Trading Agent\zeus\backtesting\metrics.py
Copied: 2026-09-28 (functions extracted unmodified via ast.get_source_segment)
Functions: optimal_block_length, _stationary_bootstrap_resample, stationary_bootstrap_p_value, sharpe_ratio, max_drawdown, sortino_ratio, calmar_ratio, win_rate, profit_factor
Copied (not imported) so this project runs without any dependency on the Zeus repo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def optimal_block_length(series: pd.Series) -> float:
    """Politis & White (2004), corrected by Patton, Politis & White (2009):
    automatic mean block length for the stationary bootstrap, via a
    flat-top lag window applied to the sample autocovariances.

    Reimplemented here rather than importing `arch.bootstrap` (Addendum
    A.4) -- this keeps the Docker image's dependency set, and therefore the
    feature pipeline's transitive numpy/pandas resolution, untouched by a
    change that is purely about inference. Algorithm (matches the
    documented procedure behind `arch.bootstrap.optimal_block_length`):

    ```
    1. rho_k = gamma_k / gamma_0 for k = 1..m_max
    2. m = first k where k_n consecutive rho's all fall inside
       +/- 2*sqrt(log10(n)/n); capped at ceil(sqrt(n)) + k_n,
       k_n = max(5, log10(n))
    3. flat-top triangular window h(x) = min(1, 2*(1-|x|)) over lags -m..m:
       g = sum h(k/m)*|k|*gamma_k     sigma2 = sum h(k/m)*gamma_k
    4. b_stationary = (2*g^2 / D_SB * n) ** (1/3),  D_SB = 2*sigma2^2
       (the 2009 correction's D_SB -- Politis & White's 2004 original used
       an erroneous constant, corrected in Patton, Politis & White 2009)
    ```

    Returns the mean block length (float, not rounded -- `1/b` is used
    directly as the stationary bootstrap's per-step restart probability).
    """
    clean = pd.Series(series).dropna().to_numpy(dtype=float)
    n = len(clean)
    if n < 10:
        return 1.0
    x = clean - clean.mean()
    gamma0 = float(np.dot(x, x) / n)
    if gamma0 == 0:
        return 1.0

    def gamma(k: int) -> float:
        return float(np.dot(x[k:], x[:-k]) / n) if k > 0 else gamma0

    k_n = max(5, int(np.log10(n)))
    m_max = min(int(np.ceil(np.sqrt(n))) + k_n, n - 1)
    rho = np.array([gamma(k) / gamma0 for k in range(1, m_max + 1)])
    band = 2 * np.sqrt(np.log10(n) / n)
    inside = np.abs(rho) < band
    m = m_max
    for start in range(len(inside) - k_n + 1):
        if inside[start : start + k_n].all():
            m = start + 1
            break
    m = max(m, 1)

    g = 0.0
    sigma2 = 0.0
    for k in range(-m, m + 1):
        w = max(0.0, min(1.0, 2.0 * (1.0 - abs(k / m))))
        gk = gamma(abs(k))
        g += w * abs(k) * gk
        sigma2 += w * gk

    if sigma2 <= 0:
        return 1.0
    d_sb = 2.0 * sigma2**2
    b_opt = (2.0 * g**2 / d_sb * n) ** (1 / 3)
    if not np.isfinite(b_opt):
        return 1.0
    return float(max(1.0, min(b_opt, n / 2)))


def _stationary_bootstrap_resample(
    x: np.ndarray, mean_block_length: float, rng: np.random.Generator
) -> np.ndarray:
    """One stationary-bootstrap (Politis & Romano 1994) resample of `x`:
    geometric-length blocks (mean `mean_block_length`), each starting at a
    uniformly random position and wrapping circularly, concatenated to
    length `len(x)`. Block-length-vectorized rather than a per-element
    restart-coin-flip loop -- the same algorithm, ~n/mean_block draws
    instead of n.
    """
    n = len(x)
    p_restart = 1.0 / max(mean_block_length, 1.0)
    idx = np.empty(n, dtype=np.int64)
    filled = 0
    pos = int(rng.integers(0, n))
    while filled < n:
        run = int(rng.geometric(p_restart))
        run = min(run, n - filled)
        idx[filled : filled + run] = (pos + np.arange(run)) % n
        filled += run
        pos = int(rng.integers(0, n))
    return x[idx]


def stationary_bootstrap_p_value(
    diff_series: pd.Series,
    *,
    block_length: float | None = None,
    n_resamples: int = 2000,
    seed: int = 42,
) -> dict:
    """Politis & Romano (1994) stationary block bootstrap, confirmatory
    p-value for H0: mean(diff_series) == 0.

    N_eff ~ 120 for a canonical h=20 comparison is not large enough to lean
    on asymptotic normality alone (PHASE6_LITERATURE_REVIEW.md §1.4). This
    resamples the *difference* series in geometric-length blocks rather than
    i.i.d. rows, so the resampled series' autocorrelation structure matches
    the original's instead of being destroyed by row-level resampling. If
    `block_length` is not given, it is derived from this series via
    `optimal_block_length` (Addendum A.4: derive each comparison's own
    block length from its own two-series difference, do not reuse a value
    computed for a different comparison).

    p-value is the standard nonparametric bootstrap test for a zero-mean
    null: `2 * min(P(boot_mean <= 0), P(boot_mean >= 0))` -- how much of the
    *unshifted* bootstrap distribution of the mean sits on the side of zero
    opposite the observed sign. Equivalent to inverting a percentile
    confidence interval.

    Returns a dict: `p_value`, `block_length` (as used), `boot_mean`,
    `boot_se`, `n_resamples`, `observed_mean`.
    """
    clean = pd.Series(diff_series).dropna().to_numpy(dtype=float)
    n = len(clean)
    if n < 10:
        return {
            "p_value": 1.0,
            "block_length": float("nan"),
            "boot_mean": 0.0,
            "boot_se": float("nan"),
            "n_resamples": n_resamples,
            "observed_mean": 0.0,
        }

    b = block_length if block_length is not None else optimal_block_length(clean)
    b = max(1.0, float(b))

    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        boot_means[i] = _stationary_bootstrap_resample(clean, b, rng).mean()

    observed = float(clean.mean())
    frac_le = float((boot_means <= 0).mean())
    frac_ge = float((boot_means >= 0).mean())
    p_value = float(min(1.0, 2 * min(frac_le, frac_ge)))
    return {
        "p_value": p_value,
        "block_length": b,
        "boot_mean": float(boot_means.mean()),
        "boot_se": float(boot_means.std(ddof=1)) if n_resamples > 1 else float("nan"),
        "n_resamples": n_resamples,
        "observed_mean": observed,
    }


def sharpe_ratio(returns: pd.Series, freq: int = 252) -> float:
    """Annualised Sharpe ratio."""
    clean = returns.dropna()
    if len(clean) < 2 or clean.std() == 0:
        return 0.0
    return float(clean.mean() / clean.std() * np.sqrt(freq))


def max_drawdown(equity: pd.Series) -> float:
    """Maximum drawdown as a negative fraction."""
    clean = equity.dropna()
    if clean.empty:
        return 0.0
    peak = clean.cummax()
    dd = (clean - peak) / peak
    return float(dd.min())


def sortino_ratio(returns: pd.Series, freq: int = 252) -> float:
    """Annualised Sortino ratio (downside deviation denominator)."""
    clean = returns.dropna()
    if len(clean) < 2:
        return 0.0
    downside = clean[clean < 0]
    if len(downside) == 0:
        return float("inf")
    dd_std = downside.std()
    if dd_std == 0:
        return 0.0
    return float(clean.mean() / dd_std * np.sqrt(freq))


def calmar_ratio(returns: pd.Series) -> float:
    """Annualised return / absolute max drawdown."""
    clean = returns.dropna()
    if clean.empty:
        return 0.0
    equity = (1 + clean).cumprod()
    mdd = max_drawdown(equity)
    if mdd == 0:
        return 0.0
    annual_ret = clean.mean() * 252
    return float(annual_ret / abs(mdd))


def win_rate(trade_pnls: pd.Series) -> float:
    """Fraction of trades with positive PnL."""
    clean = trade_pnls.dropna()
    if clean.empty:
        return 0.0
    return float((clean > 0).mean())


def profit_factor(trade_pnls: pd.Series) -> float:
    """Gross profits / gross losses (absolute)."""
    clean = trade_pnls.dropna()
    gross_profit = clean[clean > 0].sum()
    gross_loss = abs(clean[clean < 0].sum())
    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else 0.0
    return float(gross_profit / gross_loss)
