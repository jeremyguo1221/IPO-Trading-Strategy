"""Placebo / null-distribution tools.

random_established_shorts: short random NON-IPO stocks on the same entry dates and holding periods as a
    strategy's trades. A correct engine with no edge should return about -(stock drift) - costs.
shuffled_signal_null: keep the strategy's trade count, holding rule and entry-day distribution, but pick the
    IPOs and days at random. The real strategy must beat the 95th percentile of this distribution.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from lib.engine import IPO, Config, simulate
from lib.evaluate import metrics


def _pseudo_ipo(sym: str, g: pd.DataFrame, spy: pd.DataFrame, entry_date: pd.Timestamp, need: int) -> tuple[IPO, int] | None:
    g = g.reset_index(drop=True)
    pos = g.index[g.date == entry_date]
    if len(pos) == 0:
        return None
    k = int(pos[0])
    lo = max(0, k - 25)
    w = g.iloc[lo: k + need + 1]
    if len(w) < (k - lo) + need:
        return None
    w = w.join(spy.set_index("date")[["open", "close"]].rename(columns={"open": "spy_open", "close": "spy_close"}), on="date")
    if w.spy_close.isna().any():
        return None
    ipo = IPO(f"{sym}@{entry_date.date()}", "placebo", float(w.open.iloc[0]), pd.Timestamp(w.date.iloc[0]), w)
    return ipo, (k - lo) + 1  # 1-based entry day inside the pseudo series


def random_established_shorts(cfg: Config, trades: pd.DataFrame, placebo_bars: pd.DataFrame, spy: pd.DataFrame,
                              calendar: pd.DatetimeIndex, seed: int = 0, draws: int = 3) -> pd.DataFrame:
    """For each real trade, short `draws` random established stocks at the same entry date with the same
    holding length (no stop/target, so only drift + costs remain). Unlimited capacity, tiny size."""
    rng = np.random.default_rng(seed)
    groups = {s: g for s, g in placebo_bars.groupby("symbol")}
    syms = np.array(sorted(groups))
    ipos, entries = {}, []
    for t in trades.itertuples():
        for _ in range(draws):
            for _try in range(10):
                s = syms[rng.integers(len(syms))]
                made = _pseudo_ipo(s, groups[s], spy, pd.Timestamp(t.entry_date), int(t.days_held))
                if made:
                    ipo, e = made
                    if ipo.symbol in ipos:
                        continue
                    ipos[ipo.symbol] = ipo
                    entries.append({"symbol": ipo.symbol, "split": "placebo", "signal_day": e - 1, "entry_day": e,
                                    "entry_date": pd.Timestamp(t.entry_date), "strength": 0.0, "hold": int(t.days_held)})
                    break
    ent = pd.DataFrame(entries)
    out = []
    for hold, grp in ent.groupby("hold"):
        c = Config(cfg.name + "_placebo", hold=int(hold), stop=None, target=None, size=0.001, max_positions=10**6,
                   gross_cap=10**6, hedge=False, borrow=cfg.borrow, slippage=cfg.slippage)
        res = simulate(c, {s: ipos[s] for s in grp.symbol}, grp.drop(columns="hold"), calendar)
        out.append(res["trades"])
    tr = pd.concat(out, ignore_index=True)
    return tr


def shuffled_signal_null(cfg: Config, ipos: dict, strategy_entries: pd.DataFrame, calendar: pd.DatetimeIndex,
                         n: int = 200, seed: int = 0) -> pd.DataFrame:
    """Null distribution: same number of trades, entry days drawn from the strategy's own signal-day
    distribution, IPOs drawn at random (without replacement) from the same split's eligible IPOs."""
    rng = np.random.default_rng(seed)
    eligible = [s for s, i in ipos.items() if i.offer >= cfg.min_offer]
    days = strategy_entries.signal_day.to_numpy()
    k = len(strategy_entries)
    rows = []
    for i in range(n):
        picks = rng.choice(eligible, size=min(k, len(eligible)), replace=False)
        ent = []
        for s in picks:
            ipo = ipos[s]
            for _ in range(20):
                t = int(rng.choice(days))
                if t + cfg.hold <= ipo.n:
                    ent.append({"symbol": s, "split": ipo.split, "signal_day": t, "entry_day": t + 1,
                                "entry_date": pd.Timestamp(ipo.dates[t]), "strength": 0.0})
                    break
        res = simulate(cfg, {e["symbol"]: ipos[e["symbol"]] for e in ent}, pd.DataFrame(ent), calendar)
        m = metrics(res)
        rows.append({"i": i, "sharpe": m.get("sharpe"), "total_return": m.get("total_return"), "cagr": m.get("cagr"),
                     "mean_trade": m.get("mean_trade"), "trades": m.get("trades")})
    return pd.DataFrame(rows)
