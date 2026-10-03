"""Market context known at a given close: how hot the IPO market is.

Both measures use only information dated strictly before the query date:
  ipo_count(date)  - operating-company IPOs priced in the 60 calendar days before `date` (full universe,
                     including names later missing from the price database, since pricings were public)
  median_pop(date) - median day-1 close/offer - 1 of IPOs whose first trading day was in the 60 calendar
                     days before `date` (only IPOs with clean price data)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

WINDOW = pd.Timedelta(days=60)


class Context:
    def __init__(self, universe: pd.DataFrame, meta: pd.DataFrame, panel: pd.DataFrame):
        self.priced = np.sort(pd.to_datetime(universe.priced_date).to_numpy())
        d1 = panel[panel.day == 1].merge(meta[["symbol", "offer_price"]], on="symbol")
        d1 = d1.assign(pop=d1.close / d1.offer_price - 1).sort_values("date")
        self.pop_dates = pd.to_datetime(d1.date).to_numpy()
        self.pops = d1["pop"].to_numpy()
        self._cache: dict = {}

    def ipo_count(self, date) -> int:
        date = np.datetime64(pd.Timestamp(date))
        lo = np.searchsorted(self.priced, date - np.timedelta64(60, "D"), side="left")
        hi = np.searchsorted(self.priced, date, side="left")  # strictly before `date`
        return int(hi - lo)

    def median_pop(self, date) -> float | None:
        key = pd.Timestamp(date)
        if key in self._cache:
            return self._cache[key]
        d = np.datetime64(key)
        lo = np.searchsorted(self.pop_dates, d - np.timedelta64(60, "D"), side="left")
        hi = np.searchsorted(self.pop_dates, d, side="left")
        vals = self.pops[lo:hi]
        out = float(np.median(vals)) if len(vals) >= 5 else None
        self._cache[key] = out
        return out
