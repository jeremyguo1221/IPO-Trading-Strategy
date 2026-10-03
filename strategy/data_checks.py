"""Automated data checks: turn the raw export into a clean per-IPO panel.

Policy: an IPO failing any check is DROPPED (never repaired) and logged to data/exclusions.csv
(stage = "data_check") with the reason and the measured value.

Outputs:
  data/ipo_meta.csv   one row per IPO that passed, with its split (holdout / train / validation)
  data/panel.csv      daily bars for passing IPOs, numbered by trading day (day 1 = first trade),
                      joined with SPY open/close on the same date
  data/coverage.csv   by year: IPOs in the universe, found in the price database, passing all checks
"""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
DATA = HERE / "data"
DATA_END = pd.Timestamp("2026-07-31")
SPLITS = [("fresh_holdout", "2015-01-01", "2018-12-31"), ("holdout", "2019-01-01", "2021-09-30"), ("train", "2021-10-01", "2024-06-30"),
          ("validation", "2024-07-01", "2026-06-30")]
CHECK_DAYS = 60  # bar-quality checks cover the first 60 trading days (the most any strategy uses is 43)


def read_csv(name: str, **kw) -> pd.DataFrame:
    """All CSV reads go through here: tickers like NA must not become NaN."""
    return pd.read_csv(DATA / name, keep_default_na=False, na_values=[""], **kw)


def set_data_dir(path) -> None:
    global DATA
    DATA = Path(path)


def split_of(d: pd.Timestamp) -> str | None:
    for name, a, b in SPLITS:
        if pd.Timestamp(a) <= d <= pd.Timestamp(b):
            return name
    return None


def main() -> None:
    uni = read_csv("ipo_universe.csv", parse_dates=["priced_date"])
    bars = read_csv("ipo_bars.csv", parse_dates=["date"])
    spy = read_csv("spy.csv", parse_dates=["date"]).set_index("date")
    cal = spy.index  # trading calendar
    excl, metas, panels = [], [], []

    def drop(row, reason, value=None):
        excl.append({"priced_date": row.priced_date, "symbol": row.symbol, "name": row.name,
                     "offer_price": row.offer_price, "drop_reason": reason, "value": value, "stage": "data_check"})

    by_sym = {s: g.sort_values("date") for s, g in bars.groupby("symbol")}
    for row in uni.itertuples(index=False):
        g = by_sym.get(row.symbol)
        if g is None or g.empty:
            drop(row, "no price data in database (delisted, renamed, or never listed)")
            continue
        g = g[g.date >= row.priced_date - pd.Timedelta(days=30)]
        pre = g[g.date < row.priced_date - pd.Timedelta(days=3)]
        if len(pre):
            drop(row, "bars exist before the IPO (ticker reused or not a new listing)", str(pre.date.iloc[0].date()))
            continue
        g = g[g.date >= row.priced_date].reset_index(drop=True)
        if g.empty:
            drop(row, "no bars on/after pricing date")
            continue
        first = g.date.iloc[0]
        gap = (first - row.priced_date).days
        if gap > 5:
            drop(row, "first bar more than 5 days after pricing", gap)
            continue
        split = split_of(first)
        if split is None:
            drop(row, "first trade after 2026-06-30 (no room for a full hold before data ends)", str(first.date()))
            continue
        ratio = g.open.iloc[0] / row.offer_price
        if not 0.3 <= ratio <= 5:
            drop(row, "day-1 open / offer price outside 0.3-5x (split-adjusted or bad data)", round(ratio, 3))
            continue
        if len(g) >= 2 and g.volume.iloc[0] < 0.10 * g.volume.iloc[1]:
            drop(row, "stub first bar (day-1 volume < 10% of day 2)", int(g.volume.iloc[0]))
            continue
        w = g.head(CHECK_DAYS)
        if (w.volume <= 0).sum() > 0:
            drop(row, "zero-volume day in first 60 bars", int((w.volume <= 0).sum()))
            continue
        exp_days = cal[(cal >= first) & (cal <= w.date.iloc[-1])]
        missing = len(exp_days) - len(w)
        if missing > 2:
            drop(row, "more than 2 missing trading days in first 60 bars", missing)
            continue
        jump = (w.close / w.close.shift(1)).dropna()
        if ((jump > 4) | (jump < 0.25)).any():
            drop(row, "close-to-close jump > 4x or < 0.25x in first 60 bars (possible split artifact)",
                 round(float(jump[(jump > 4) | (jump < 0.25)].iloc[0]), 3))
            continue
        bad_ohlc = (w.high < w[["open", "close"]].max(axis=1) - 1e-6) | (w.low > w[["open", "close"]].min(axis=1) + 1e-6)
        if bad_ohlc.any():
            drop(row, "inconsistent OHLC (high below open/close or low above)", int(bad_ohlc.sum()))
            continue
        last_needed = cal[cal >= first][:CHECK_DAYS]
        if g.date.iloc[-1] < min(last_needed[-1], DATA_END) and g.date.iloc[-1] < DATA_END - pd.Timedelta(days=5):
            drop(row, "stops trading within first 60 trading days", str(g.date.iloc[-1].date()))
            continue
        g = g.copy()
        g["day"] = np.arange(1, len(g) + 1)
        g = g.join(spy[["open", "close"]].rename(columns={"open": "spy_open", "close": "spy_close"}), on="date")
        if g.spy_close.isna().any():
            drop(row, "dates missing from SPY calendar", int(g.spy_close.isna().sum()))
            continue
        panels.append(g[["symbol", "day", "date", "open", "high", "low", "close", "volume", "spy_open", "spy_close"]])
        metas.append({"symbol": row.symbol, "name": row.name, "priced_date": row.priced_date, "first_date": first,
                      "offer_price": row.offer_price, "deal_size": row.deal_size, "exchange": row.exchange,
                      "split": split, "bars": len(g)})

    meta = pd.DataFrame(metas)
    panel = pd.concat(panels, ignore_index=True)
    ex = pd.DataFrame(excl)
    prev = read_csv("exclusions.csv")
    pd.concat([prev[prev.stage != "data_check"], ex], ignore_index=True).to_csv(DATA / "exclusions.csv", index=False)
    meta.to_csv(DATA / "ipo_meta.csv", index=False)
    panel.to_csv(DATA / "panel.csv", index=False)

    yr = uni.priced_date.dt.year
    found = uni.symbol.isin(bars.symbol.unique())
    cov = pd.DataFrame({"universe": uni.groupby(yr).size(), "in_price_db": found.groupby(yr).sum(),
                        "passed_checks": meta.groupby(meta.priced_date.dt.year).size()}).fillna(0).astype(int)
    cov["pct_in_db"] = (cov.in_price_db / cov.universe * 100).round(1)
    cov["pct_passed"] = (cov.passed_checks / cov.universe * 100).round(1)
    cov.index.name = "year"
    cov.to_csv(DATA / "coverage.csv")
    print(f"passed {len(meta)} of {len(uni)} IPOs; panel rows {len(panel):,}")
    print("\ndrop reasons:\n" + ex.drop_reason.value_counts().to_string())
    print("\ncoverage:\n" + cov.to_string())
    print("\nby split:\n" + meta.split.value_counts().to_string())


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        set_data_dir(sys.argv[1])  # phase 2: data/full
    main()
