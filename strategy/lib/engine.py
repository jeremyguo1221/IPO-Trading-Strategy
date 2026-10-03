"""Event-driven backtest engine for shorting recently listed IPOs with daily bars.

Timing contract (enforced, tested in test_engine.py):
  * A signal for an IPO is evaluated on trading day t using ONLY that IPO's bars for days 1..t
    (the signal function receives a truncated view; it cannot see day t+1 or later) plus a market
    context object that answers only with information dated strictly before day t+1's open.
  * The order fills at the OPEN of day t+1. Entries are never earlier than day `min_entry_day`
    (default 3: shares are normally not borrowable before settlement).
  * Exits: the stop (price rises to entry*(1+stop)) triggers when the day's HIGH reaches it and fills
    at max(stop price, that day's open) - a gap through the stop is charged in full. A profit target
    (price falls to entry*(1-target)) fills at min(target price, open). Otherwise the position is
    covered at the CLOSE of its `hold`-th trading day (hold <= 21).
  * Costs: `slippage` per side on the stock (sell lower / buy back higher); borrow fee `borrow` per
    year accrued every trading day on the position's prior-close market value (rate / 252); no interest
    is credited on short proceeds. Hedge (long SPY) pays `spy_slippage` per side.
  * Portfolio: equity marked to market daily at closes; each new trade's notional = size * equity at
    the previous close (optionally volatility-scaled), subject to `max_positions` and gross short
    exposure <= `gross_cap` * equity. Same-day candidates are taken in a deterministic order
    (signal strength if provided, else symbol) and skipped when capacity is full.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Callable

import numpy as np
import pandas as pd


@dataclass
class Config:
    name: str
    signal: str = "fixed_day"          # key into SIGNALS (lib/signals.py)
    params: dict = field(default_factory=dict)
    min_entry_day: int = 3
    last_signal_day: int = 42          # latest trading day a signal may fire on
    hold: int = 21                     # max trading days held (<= 21: one-month cap)
    stop: float | None = None          # e.g. 0.25 = cover if price rises 25% above entry
    target: float | None = None        # e.g. 0.20 = cover if price falls 20% below entry
    trail: float | None = None         # cover if price rises this far above the lowest low since entry (known before today)
    thesis_stop: bool = False          # cover if price retakes the pre-entry post-IPO high (the top we shorted from)
    exit_signal: str | None = None     # close-based exit, filled at the NEXT open: "ma5" (close > 5-day avg close) |
                                       # "prev_high" (close > previous day high)
    size: float = 0.07                 # fraction of equity per trade
    vol_target: float | None = None    # daily vol target for size scaling (None = fixed size)
    range_target: float | None = None  # size *= clip(range_target / day-1 (high-low)/open, 0.5, 1.5)
    max_positions: int = 15
    gross_cap: float = 1.0
    hedge: bool = False
    hedge_symbol: str = "SPY"         # SPY (from the Zeus DB) or IWM (Nasdaq API, see data/iwm.csv)
    borrow: float = 0.30
    slippage: float = 0.0025
    spy_slippage: float = 0.0002
    min_price: float = 5.0
    min_dollar_vol: float = 5e6
    min_offer: float = 5.0

    def as_dict(self) -> dict:
        return asdict(self)


# ------------------------------------------------------------------ data containers
class IPO:
    """Arrays for one IPO, indexed 0..n-1 for trading days 1..n."""
    __slots__ = ("symbol", "split", "offer", "first_date", "dates", "o", "h", "l", "c", "v", "so", "sc", "n", "hx")

    def __init__(self, symbol, split, offer, first_date, g: pd.DataFrame):
        self.symbol, self.split, self.offer, self.first_date = symbol, split, float(offer), first_date
        self.dates = g.date.to_numpy()
        self.o, self.h, self.l, self.c = (g[k].to_numpy(float) for k in ("open", "high", "low", "close"))
        self.v = g.volume.to_numpy(float)
        self.so, self.sc = g.spy_open.to_numpy(float), g.spy_close.to_numpy(float)
        self.n = len(g)
        self.hx = {"SPY": (self.so, self.sc)}  # hedge instrument open/close arrays aligned to this IPO's dates

    def add_hedge(self, name: str, px: pd.DataFrame) -> None:
        a = px.reindex(pd.DatetimeIndex(self.dates))
        if a[["open", "close"]].isna().any().any():
            raise ValueError(f"{name} prices missing on some {self.symbol} dates")
        self.hx[name] = (a.open.to_numpy(float), a.close.to_numpy(float))

    def view(self, t: int) -> "View":
        return View(self, t)


class View:
    """Read-only window on an IPO's bars for days 1..t. Indexing beyond t raises."""
    __slots__ = ("_ipo", "t", "symbol", "offer")

    def __init__(self, ipo: IPO, t: int):
        self._ipo, self.t, self.symbol, self.offer = ipo, t, ipo.symbol, ipo.offer

    def _cut(self, arr):
        return arr[: self.t]

    o = property(lambda s: s._cut(s._ipo.o))
    h = property(lambda s: s._cut(s._ipo.h))
    l = property(lambda s: s._cut(s._ipo.l))
    c = property(lambda s: s._cut(s._ipo.c))
    v = property(lambda s: s._cut(s._ipo.v))
    sc = property(lambda s: s._cut(s._ipo.sc))
    dates = property(lambda s: s._cut(s._ipo.dates))


def load_ipos(meta: pd.DataFrame, panel: pd.DataFrame, hedges: dict[str, pd.DataFrame] | None = None) -> dict[str, IPO]:
    out = {}
    m = meta.set_index("symbol")
    for sym, g in panel.groupby("symbol", sort=True):
        r = m.loc[sym]
        out[sym] = IPO(sym, r.split, r.offer_price, pd.Timestamp(r.first_date), g.sort_values("day"))
        for name, px in (hedges or {}).items():
            out[sym].add_hedge(name, px)
    return out


# ------------------------------------------------------------------ candidate generation
def find_entries(cfg: Config, ipos: dict[str, IPO], signal_fn: Callable, context) -> pd.DataFrame:
    """One candidate per IPO: the first day t in [min_entry_day-1, last_signal_day] where the signal fires
    and entry-time tradability holds. Entry = open of day t+1."""
    rows = []
    for sym, ipo in ipos.items():
        if ipo.offer < cfg.min_offer:
            continue
        t_lo, t_hi = cfg.min_entry_day - 1, min(cfg.last_signal_day, ipo.n - 1)
        for t in range(t_lo, t_hi + 1):
            if t + cfg.hold > ipo.n:  # a full hold must fit inside the available data
                break
            view = ipo.view(t)
            if view.c[-1] < cfg.min_price or np.median(view.c * view.v) < cfg.min_dollar_vol:
                continue
            res = signal_fn(view, cfg.params, context)
            if res is False or res is None:
                continue
            strength = float(res) if not isinstance(res, bool) else 0.0
            rows.append({"symbol": sym, "split": ipo.split, "signal_day": t, "entry_day": t + 1,
                         "entry_date": pd.Timestamp(ipo.dates[t]), "strength": strength})
            break
    return pd.DataFrame(rows, columns=["symbol", "split", "signal_day", "entry_day", "entry_date", "strength"])


def pre_entry_beta(ipo: IPO, t: int, hedge_symbol: str = "SPY") -> float:
    """Beta of daily returns vs the hedge instrument over days 2..t (known before entry). Default 1.5 if < 10 obs."""
    if t < 11:
        return 1.5
    r = np.diff(ipo.c[:t]) / ipo.c[: t - 1]
    hc = ipo.hx[hedge_symbol][1]
    m = np.diff(hc[:t]) / hc[: t - 1]
    if np.var(m) == 0:
        return 1.5
    return float(np.clip(np.cov(r, m)[0, 1] / np.var(m, ddof=1), 0.5, 3.0))


def pre_entry_vol(ipo: IPO, t: int) -> float:
    r = np.diff(ipo.c[:t]) / ipo.c[: t - 1]
    return float(np.std(r, ddof=1)) if len(r) >= 2 else np.nan


# ------------------------------------------------------------------ simulation
EXIT_FIELDS = ("hold", "target", "stop", "trail", "thesis_stop", "exit_signal")


def _exit_params(cfg: Config, r) -> dict:
    """Per-trade exit settings: an entry row may carry any of EXIT_FIELDS to override the Config (phase 5:
    sector-specific exits in one portfolio). Missing / NaN columns fall back to the Config."""
    out = {}
    for f in EXIT_FIELDS:
        v = getattr(r, f, None)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            v = getattr(cfg, f)
        elif f == "hold":
            v = int(v)
        elif f == "thesis_stop":
            v = bool(v)
        out[f] = v
    return out


def simulate(cfg: Config, ipos: dict[str, IPO], entries: pd.DataFrame, calendar: pd.DatetimeIndex) -> dict:
    """Run the daily portfolio simulation. Returns trades, daily equity, and an accounting ledger."""
    if entries.empty:
        return {"trades": pd.DataFrame(), "equity": pd.Series(dtype=float), "ledger": pd.DataFrame()}
    entries = entries.sort_values(["entry_date", "strength", "symbol"], ascending=[True, False, True]).reset_index(drop=True)
    start = entries.entry_date.min()
    by_date = {d: g for d, g in entries.groupby("entry_date")}
    dates = calendar[calendar >= start]
    equity = 1.0
    open_pos: list[dict] = []
    trades, eq_rows, ledger = [], [], []
    holds = entries["hold"].fillna(cfg.hold).astype(int) if "hold" in entries.columns else pd.Series(cfg.hold, index=entries.index)
    pending_last = max(pd.Timestamp(ipos[s].dates[min(e + h - 1, ipos[s].n - 1)])
                       for s, e, h in zip(entries.symbol, entries.entry_day - 1, holds))

    for d in dates:
        if d > pending_last and not open_pos:
            break
        eq_prev = equity
        day_pnl = day_slip = day_borrow = day_hedge = 0.0
        # 1) new entries at today's open (sized on previous close equity)
        for r in (by_date[d].itertuples() if d in by_date else []):
            ipo = ipos[r.symbol]
            k = r.entry_day - 1  # 0-based index of entry day
            gross_now = sum(p["shares"] * p["mark"] for p in open_pos)
            if len(open_pos) >= cfg.max_positions:
                continue
            size = cfg.size
            if cfg.vol_target is not None:
                vol = pre_entry_vol(ipo, r.signal_day)
                size = cfg.size * float(np.clip(cfg.vol_target / vol, 0.25, 2.0)) if vol and np.isfinite(vol) else cfg.size
            if cfg.range_target is not None:
                d1r = (ipo.h[0] - ipo.l[0]) / ipo.o[0]
                size = size * float(np.clip(cfg.range_target / d1r, 0.5, 1.5)) if d1r > 0 else size
            notional = size * eq_prev
            if gross_now + notional > cfg.gross_cap * eq_prev + 1e-12:
                continue
            shares = notional / ipo.o[k]
            slip = shares * ipo.o[k] * cfg.slippage
            xp = _exit_params(cfg, r)
            pos = {"symbol": r.symbol, "split": r.split, "signal_day": r.signal_day, "k0": k, "k": k,
                   "entry_date": d, "entry_open": ipo.o[k], "shares": shares, "mark": ipo.o[k],
                   "notional": notional, "pnl": -0.0, "slip": slip, "borrow": 0.0, "hedge_pnl": 0.0,
                   "xp": xp,
                   "stop_px": ipo.o[k] * (1 + xp["stop"]) if xp["stop"] else None,
                   "thesis_px": float(ipo.h[: r.signal_day].max()) if xp["thesis_stop"] else None,
                   "low_ref": ipo.o[k], "pending_exit": False,
                   "tgt_px": ipo.o[k] * (1 - xp["target"]) if xp["target"] else None}
            if cfg.hedge:
                ho = ipo.hx[cfg.hedge_symbol][0]
                beta = pre_entry_beta(ipo, r.signal_day, cfg.hedge_symbol)
                pos["beta"] = beta
                pos["spy_units"] = beta * notional / ho[k]
                pos["spy_mark"] = ho[k]
                pos["slip"] += pos["spy_units"] * ho[k] * cfg.spy_slippage
            day_slip += pos["slip"]
            open_pos.append(pos)
        # 2) mark / exit every open position on today's bar
        still = []
        for p in open_pos:
            ipo = ipos[p["symbol"]]
            k = p["k"]
            bar_date = pd.Timestamp(ipo.dates[k])
            if bar_date > d:  # this stock has no bar today (rare missing day): carry the position unchanged
                still.append(p)
                continue
            assert bar_date == d, f"calendar mismatch {p['symbol']} {bar_date} vs {d}"
            # borrow accrues on the prior mark for each day held (entry day accrues on entry value)
            b = cfg.borrow / 252 * p["shares"] * (p["mark"] if k > p["k0"] else p["entry_open"])
            p["borrow"] += b
            day_borrow += b
            exit_px, reason = None, None
            # protective levels are all fixed before today's trading (trail uses lows up to yesterday)
            levels = {"stop": p["stop_px"], "thesis": p["thesis_px"],
                      "trail": p["low_ref"] * (1 + p["xp"]["trail"]) if p["xp"]["trail"] else None}
            levels = {n: v for n, v in levels.items() if v is not None}
            stop_name = min(levels, key=levels.get) if levels else None
            if p["pending_exit"]:
                exit_px, reason = ipo.o[k], "signal"
            elif stop_name is not None and ipo.h[k] >= levels[stop_name]:
                lvl = levels[stop_name]
                exit_px, reason = max(lvl, ipo.o[k] if k > p["k0"] else lvl), stop_name
            elif p["tgt_px"] is not None and ipo.l[k] <= p["tgt_px"]:
                exit_px, reason = min(p["tgt_px"], ipo.o[k] if k > p["k0"] else p["tgt_px"]), "target"
            elif k - p["k0"] + 1 >= p["xp"]["hold"]:
                exit_px, reason = ipo.c[k], "time"
            mark_to = exit_px if exit_px is not None else ipo.c[k]
            pnl = p["shares"] * (p["mark"] - mark_to)
            p["pnl"] += pnl
            day_pnl += pnl
            p["mark"] = mark_to
            if cfg.hedge:
                hc = ipo.hx[cfg.hedge_symbol][1]
                spy_to = hc[k]
                hp = p["spy_units"] * (spy_to - p["spy_mark"])
                p["hedge_pnl"] += hp
                day_hedge += hp
                p["spy_mark"] = spy_to
                if exit_px is not None:
                    sl = p["spy_units"] * hc[k] * cfg.spy_slippage
                    p["slip"] += sl
                    day_slip += sl
            if exit_px is not None:
                sl = p["shares"] * exit_px * cfg.slippage
                p["slip"] += sl
                day_slip += sl
                net = p["pnl"] - p["slip"] - p["borrow"] + p["hedge_pnl"]
                trades.append({
                    "symbol": p["symbol"], "split": p["split"], "signal_day": p["signal_day"],
                    "entry_day": p["k0"] + 1, "exit_day": k + 1, "entry_date": p["entry_date"], "exit_date": d,
                    "entry_open": p["entry_open"], "exit_px": exit_px, "exit_reason": reason,
                    "days_held": k - p["k0"] + 1, "notional": p["notional"], "shares": p["shares"],
                    "gross_ret": 1 - exit_px / p["entry_open"],
                    "price_pnl": p["pnl"], "slippage": p["slip"], "borrow": p["borrow"], "hedge_pnl": p["hedge_pnl"],
                    "beta": p.get("beta", np.nan), "net_pnl": net, "net_ret": net / p["notional"]})
            else:
                p["low_ref"] = min(p["low_ref"], ipo.l[k])
                if p["xp"]["exit_signal"] == "ma5" and k >= 4 and ipo.c[k] > ipo.c[k - 4: k + 1].mean():
                    p["pending_exit"] = True
                elif p["xp"]["exit_signal"] == "prev_high" and k > p["k0"] and ipo.c[k] > ipo.h[k - 1]:
                    p["pending_exit"] = True
                p["k"] = k + 1
                still.append(p)
        open_pos = still
        equity = eq_prev + day_pnl + day_hedge - day_slip - day_borrow
        gross = sum(p["shares"] * p["mark"] for p in open_pos)
        eq_rows.append((d, equity))
        ledger.append({"date": d, "equity_prev": eq_prev, "price_pnl": day_pnl, "hedge_pnl": day_hedge,
                       "slippage": day_slip, "borrow": day_borrow, "equity": equity,
                       "open_positions": len(open_pos), "gross_short": gross / equity if equity else np.nan})
        if equity <= 0:
            break
    eq = pd.Series(dict(eq_rows), name="equity")
    return {"trades": pd.DataFrame(trades), "equity": eq, "ledger": pd.DataFrame(ledger)}


def run(cfg: Config, ipos: dict[str, IPO], signal_fn: Callable, context, calendar: pd.DatetimeIndex,
        split: str | None = None) -> dict:
    sub = {s: i for s, i in ipos.items() if split is None or i.split == split}
    entries = find_entries(cfg, sub, signal_fn, context)
    out = simulate(cfg, sub, entries, calendar)
    out["entries"] = entries
    return out
