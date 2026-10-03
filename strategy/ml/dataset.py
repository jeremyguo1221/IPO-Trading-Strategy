"""Phase 2 candidate/label dataset.

Each clean IPO is scored at checkpoint signal days CHECKPOINTS (entry at the next open) if it is tradable then
(same filter as the engine: offer >= $5, close >= $5, median dollar volume >= $5M, a full 21-day hold fits).
Label = 21-day net SPY-hedged short return per $ of notional, using the engine's exact cost formula
(verified against the engine in test_ml.py).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from lib.engine import IPO, Config, load_ipos, pre_entry_beta
from lib.features import ALL_FEATURES, RegimeContext, compute

HERE = Path(__file__).resolve().parents[1]
FULL = HERE / "data" / "full"
CHECKPOINTS = [2, 5, 10, 15, 20]
BASE = Config("ml_base", hold=21, hedge=True)


def rd(name, **kw):
    return pd.read_csv(FULL / name, keep_default_na=False, na_values=[""], **kw)


def load_full():
    meta = rd("ipo_meta.csv", parse_dates=["first_date", "priced_date"])
    panel = rd("panel.csv", parse_dates=["date"])
    uni = rd("ipo_universe.csv", parse_dates=["priced_date"])
    spy = rd("spy.csv", parse_dates=["date"])
    ipos = load_ipos(meta, panel)
    ctx = RegimeContext(uni, meta, panel, spy)
    return ipos, ctx, pd.DatetimeIndex(spy.date), meta, panel


def label(ipo: IPO, t: int, cfg: Config = BASE) -> tuple[float, float, int]:
    """Net hedged short return for signal day t (entry open of day t+1, exit close of day t+hold)."""
    e = t          # 0-based index of entry day (day t+1)
    x = t + cfg.hold - 1
    o_e, c_x = ipo.o[e], ipo.c[x]
    gross = 1 - c_x / o_e
    slip = cfg.slippage * (1 + c_x / o_e)
    borrow = cfg.borrow / 252 * (1 + ipo.c[e:x].sum() / o_e)
    net = gross - slip - borrow
    beta = np.nan
    if cfg.hedge:
        ho, hc = ipo.hx[cfg.hedge_symbol]
        beta = pre_entry_beta(ipo, t, cfg.hedge_symbol)
        net += beta * (hc[x] / ho[e] - 1) - cfg.spy_slippage * beta * (1 + hc[x] / ho[e])
    return float(net), float(beta), x


def build(ipos, ctx, meta: pd.DataFrame, cfg: Config = BASE) -> pd.DataFrame:
    m = meta.set_index("symbol")
    uni_cols = {}
    rows = []
    for sym, ipo in ipos.items():
        if ipo.offer < cfg.min_offer:
            continue
        extra = {"shares": m.loc[sym, "shares"] if "shares" in m.columns else np.nan,
                 "deal_size": m.loc[sym, "deal_size"], "exchange": m.loc[sym, "exchange"]}
        for t in CHECKPOINTS:
            if t + cfg.hold > ipo.n:
                break
            v = ipo.view(t)
            if v.c[-1] < cfg.min_price or np.median(v.c * v.v) < cfg.min_dollar_vol:
                continue
            f = compute(v, extra, ctx)
            y, beta, x = label(ipo, t, cfg)
            rows.append({"symbol": sym, "split": ipo.split, "signal_day": t,
                         "signal_date": pd.Timestamp(ipo.dates[t - 1]), "entry_date": pd.Timestamp(ipo.dates[t]),
                         "exit_date": pd.Timestamp(ipo.dates[x]), **f, "y": y, "beta": beta})
    df = pd.DataFrame(rows)
    df["signal_day"] = df.signal_day.astype(int)  # the float feature of the same name must not change the key type
    df["year"] = df.entry_date.dt.year
    return df


if __name__ == "__main__":
    import sys
    ipos, ctx, cal, meta, panel = load_full()
    uni = rd("ipo_universe.csv")
    meta = meta.merge(uni[["symbol", "priced_date", "shares"]].assign(priced_date=lambda d: pd.to_datetime(d.priced_date)),
                      on=["symbol", "priced_date"], how="left")
    df = build(ipos, ctx, meta)
    out = HERE / "ml" / "candidates.csv"
    df.to_csv(out, index=False)
    print(f"{len(df)} candidates from {df.symbol.nunique()} IPOs -> {out}")
    # never print label statistics for the fresh holdout (it must stay unseen until final_eval_ml.py)
    s = df.groupby("split").agg(n=("y", "size"), ipos=("symbol", "nunique"), mean_y=("y", "mean")).round(4)
    s.loc[s.index == "fresh_holdout", "mean_y"] = np.nan
    print(s.to_string())
    print("\nfeature coverage (non-missing share):")
    print(df[ALL_FEATURES].notna().mean().round(3).to_string())
