"""Phase 2 features. Every feature for an IPO at signal day t uses only data dated on or before day t's close.

IPO-specific features come from the View (days 1..t only). Regime features come from RegimeContext, which answers
only from SPY bars <= the query date and from other IPOs' bars <= the query date.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from lib.context import Context

IPO_FEATURES = ["pop", "d1_oc", "d1_range", "d1_turnover", "log_deal", "log_offer", "exch_nyse", "ext", "run",
                "ret3", "ret5", "rvol", "dd_from_high", "dist_d1low", "vol_ratio", "signal_day", "below_offer",
                "below_d1low", "log_dollar_vol"]
REGIME_FEATURES = ["spy_ret20", "spy_ret60", "spy_above_ma50", "spy_vol20", "hot_count", "hot_pop",
                   "cohort_mom", "cohort_below_offer", "cohort_n"]
ALL_FEATURES = IPO_FEATURES + REGIME_FEATURES


class RegimeContext(Context):
    """Phase-1 Context (hot_count / hot_pop) plus SPY trend/vol and IPO-cohort momentum."""

    def __init__(self, universe: pd.DataFrame, meta: pd.DataFrame, panel: pd.DataFrame, spy: pd.DataFrame):
        super().__init__(universe, meta, panel)
        s = spy.sort_values("date")
        self.spy_dates = pd.to_datetime(s.date).to_numpy()
        self.spy_close = s.close.to_numpy(float)
        offer = meta.set_index("symbol").offer_price
        first = meta.set_index("symbol").first_date
        self.coh = []  # (first_date, offer, day1 close, dates array, closes array)
        for sym, g in panel.groupby("symbol"):
            g = g.sort_values("date")
            self.coh.append((np.datetime64(pd.Timestamp(first[sym])), float(offer[sym]), float(g.close.iloc[0]),
                             pd.to_datetime(g.date).to_numpy(), g.close.to_numpy(float)))
        self.coh.sort(key=lambda x: x[0])
        self.coh_first = np.array([c[0] for c in self.coh])
        self._rc: dict = {}

    def spy_feats(self, date) -> dict:
        d = np.datetime64(pd.Timestamp(date))
        i = int(np.searchsorted(self.spy_dates, d, side="right")) - 1  # last SPY bar on/before date
        c = self.spy_close
        out = {"spy_ret20": np.nan, "spy_ret60": np.nan, "spy_above_ma50": np.nan, "spy_vol20": np.nan}
        if i >= 20:
            out["spy_ret20"] = c[i] / c[i - 20] - 1
            r = np.diff(c[i - 20: i + 1]) / c[i - 20: i]
            out["spy_vol20"] = float(np.std(r, ddof=1))
        if i >= 60:
            out["spy_ret60"] = c[i] / c[i - 60] - 1
        if i >= 49:
            out["spy_above_ma50"] = float(c[i] > c[i - 49: i + 1].mean())
        return out

    def cohort_feats(self, date) -> dict:
        key = pd.Timestamp(date)
        if key in self._rc:
            return self._rc[key]
        d = np.datetime64(key)
        lo = np.searchsorted(self.coh_first, d - np.timedelta64(90, "D"), side="left")
        hi = np.searchsorted(self.coh_first, d, side="left")  # listed strictly before date
        rets, below = [], []
        for first, offer, c1, dates, closes in self.coh[lo:hi]:
            j = int(np.searchsorted(dates, d, side="right")) - 1  # that IPO's last close on/before date
            if j < 0:
                continue
            rets.append(closes[j] / c1 - 1)
            below.append(closes[j] < offer)
        out = {"cohort_mom": float(np.median(rets)) if len(rets) >= 5 else np.nan,
               "cohort_below_offer": float(np.mean(below)) if len(below) >= 5 else np.nan,
               "cohort_n": float(len(rets))}
        self._rc[key] = out
        return out


def compute(v, extra: dict, ctx: RegimeContext) -> dict:
    """v: View over days 1..t. extra: static IPO info from the calendar (shares, deal_size, exchange)."""
    t = v.t
    c, o, h, l, vol = v.c, v.o, v.h, v.l, v.v
    r = np.diff(c) / c[:-1] if t >= 2 else np.array([])
    f = {
        "pop": c[0] / v.offer - 1,
        "d1_oc": c[0] / o[0] - 1,
        "d1_range": (h[0] - l[0]) / o[0],
        "d1_turnover": vol[0] / extra["shares"] if extra.get("shares") and extra["shares"] > 0 else np.nan,
        "log_deal": np.log(extra["deal_size"]) if extra.get("deal_size") and extra["deal_size"] > 0 else np.nan,
        "log_offer": np.log(v.offer),
        "exch_nyse": float("NYSE" in str(extra.get("exchange", "")).upper()),
        "ext": c[-1] / v.offer - 1,
        "run": c[-1] / c[0] - 1,
        "ret3": c[-1] / c[-4] - 1 if t >= 4 else np.nan,
        "ret5": c[-1] / c[-6] - 1 if t >= 6 else np.nan,
        "rvol": float(np.std(r, ddof=1)) if len(r) >= 2 else np.nan,
        "dd_from_high": c[-1] / h.max() - 1,
        "dist_d1low": c[-1] / l[0] - 1,
        "vol_ratio": vol[-3:].mean() / vol[0] if vol[0] > 0 else np.nan,
        "signal_day": float(t),
        "below_offer": float(c[-1] < v.offer),
        "below_d1low": float(t >= 2 and c[-1] < l[0]),
        "log_dollar_vol": float(np.log(np.median(c * vol))),
    }
    date = v.dates[-1]
    f.update(ctx.spy_feats(date))
    f["hot_count"] = float(ctx.ipo_count(date))
    hp = ctx.median_pop(date)
    f["hot_pop"] = np.nan if hp is None else hp
    f.update(ctx.cohort_feats(date))
    return f
