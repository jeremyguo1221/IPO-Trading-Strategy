"""Event study: what do IPO stocks do around the day their initial lock-up expires?

Event day (tau = 0) = first trading day on or after the ACTUAL lock-up expiration date from Nasdaq's deal data.
Window tau = -70 .. +40 trading days; each IPO needs bars across the whole window (else excluded, with reason).
Returns are close-to-close. "Abnormal" = stock return minus SPY return over the same days (not beta-adjusted).

Windows (all anchored on closes):
  pre   : tau -21 -> -1   (the 20 days before expiry)
  event : tau  -1 ->  0   (expiry day)
  post5 : tau   0 -> +5
  post20: tau  +5 -> +20
  post40: tau +20 -> +40
  around: tau  -1 -> +20  (expiry day and the month after)
Placebo: identical windows anchored 45 trading days EARLIER (tau = -45), i.e. on non-event dates in the same stocks.
Cross-sections of 'around': unlocked supply (overhang terciles), price vs offer going in, run-up into expiry,
lock-up length, era, deal size.

Output: lockup/results.json, lockup/event_paths.csv, lockup/per_ipo.csv, lockup/exclusions.csv
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
FULL = HERE.parent / "data" / "full"
PRE, POST = 70, 40  # 70 bars before so every window has a placebo 45 days earlier
WINDOWS = {"pre": (-21, -1), "event": (-1, 0), "post5": (0, 5), "post20": (5, 20), "post40": (20, 40), "around": (-1, 20)}
RNG = np.random.default_rng(17)


def rd(n, **k):
    return pd.read_csv(FULL / n, keep_default_na=False, na_values=[""], **k)


def summarize(x: pd.Series) -> dict:
    x = x.dropna()
    if len(x) < 10:
        return {"n": len(x)}
    boots = [RNG.choice(x.values, len(x)).mean() for _ in range(3000)]
    return {"n": int(len(x)), "mean_%": round(x.mean() * 100, 2), "median_%": round(x.median() * 100, 2),
            "pct_negative": round((x < 0).mean() * 100, 1), "ci_lo_%": round(np.percentile(boots, 2.5) * 100, 2),
            "ci_hi_%": round(np.percentile(boots, 97.5) * 100, 2), "wilcoxon_p": round(float(stats.wilcoxon(x).pvalue), 4)}


def main():
    panel = rd("panel.csv", parse_dates=["date"])
    meta = rd("ipo_meta.csv", parse_dates=["first_date", "priced_date"]).set_index("symbol")
    lk = rd("lockups.csv", parse_dates=["lockup_exp", "first_date"]).set_index("symbol")
    by = {s: g.sort_values("day").reset_index(drop=True) for s, g in panel.groupby("symbol")}
    rows, paths, excl = [], [], []
    for sym, g in by.items():
        if sym not in lk.index or pd.isna(lk.at[sym, "lockup_exp"]):
            excl.append({"symbol": sym, "reason": "no lock-up date in Nasdaq deal data"}); continue
        exp = lk.at[sym, "lockup_exp"]
        after = g.index[g.date >= exp]
        if len(after) == 0:
            excl.append({"symbol": sym, "reason": "lock-up expiry beyond available price data"}); continue
        L = int(after[0])
        if (g.date.iloc[L] - exp).days > 5:
            excl.append({"symbol": sym, "reason": "gap in data around expiry"}); continue
        if L - PRE < 0 or L + POST >= len(g):
            excl.append({"symbol": sym, "reason": "not enough bars before/after expiry"}); continue
        w = g.iloc[L - PRE: L + POST + 1].reset_index(drop=True)
        w["tau"] = np.arange(-PRE, POST + 1)
        c, s = w.set_index("tau").close, w.set_index("tau").spy_close
        v = w.set_index("tau").volume
        r = {"symbol": sym, "event_date": g.date.iloc[L].date(), "year": g.date.iloc[L].year, "event_trading_day": L + 1,
             "lockup_days": lk.at[sym, "lockup_days"], "overhang": lk.at[sym, "overhang"],
             "offer": meta.at[sym, "offer_price"], "deal_size": meta.at[sym, "deal_size"]}
        for name, (a, b) in WINDOWS.items():
            raw = c[b] / c[a] - 1
            r[f"{name}_raw"] = raw
            r[f"{name}_abn"] = raw - (s[b] / s[a] - 1)
            # placebo: same window 45 trading days earlier
            if a - 45 >= -PRE:
                r[f"{name}_placebo_abn"] = (c[b - 45] / c[a - 45] - 1) - (s[b - 45] / s[a - 45] - 1)
        base_vol = v.loc[-40:-21].mean()  # volume baseline: tau -40..-21
        r["vol_event_ratio"] = v.loc[0] / base_vol if base_vol > 0 else np.nan
        r["vol_post5_ratio"] = v.loc[0:5].mean() / base_vol if base_vol > 0 else np.nan
        r["price_vs_offer_before"] = c[-1] / meta.at[sym, "offer_price"] - 1
        # measured BEFORE every event and placebo window (tau -70), so grouping on it can't build in the result
        r["price_vs_offer_early"] = c[-70] / meta.at[sym, "offer_price"] - 1
        r["runup_since_day1"] = c[-1] / g.close.iloc[0] - 1
        rows.append(r)
        cum = (c / c[-21] - 1) - (s / s[-21] - 1)
        vr = v / base_vol if base_vol > 0 else v * np.nan
        paths.append(pd.DataFrame({"symbol": sym, "tau": cum.index, "car": cum.values, "vol_ratio": vr.values}))
    d = pd.DataFrame(rows)
    P = pd.concat(paths)
    pd.DataFrame(excl).to_csv(HERE / "exclusions.csv", index=False)
    d.to_csv(HERE / "per_ipo.csv", index=False)
    agg = P.groupby("tau").agg(car_mean=("car", "mean"), car_median=("car", "median"),
                               car_p25=("car", lambda x: x.quantile(.25)), car_p75=("car", lambda x: x.quantile(.75)),
                               vol_ratio_median=("vol_ratio", "median"), n=("car", "size")).reset_index()
    agg.to_csv(HERE / "event_paths.csv", index=False)

    res = {"n_events": len(d), "excluded": pd.DataFrame(excl).reason.value_counts().to_dict(),
           "lockup_days": d.lockup_days.value_counts().head(6).to_dict(),
           "median_event_trading_day": float(d.event_trading_day.median()),
           "volume": {"expiry_day_median_ratio": round(float(d.vol_event_ratio.median()), 2),
                      "first_week_median_ratio": round(float(d.vol_post5_ratio.median()), 2)},
           "windows": {k: {"abnormal": summarize(d[f"{k}_abn"]), "raw": summarize(d[f"{k}_raw"]),
                           "placebo_abnormal": summarize(d.get(f"{k}_placebo_abn", pd.Series(dtype=float)))}
                       for k in WINDOWS}}
    y = "around_abn"
    def split(col, q=None, bins=None, labels=None):
        x = d[col]
        cut = pd.qcut(x.rank(method="first"), q, labels=labels) if q else pd.cut(x, bins, labels=labels)
        out = {}
        for k in (labels or cut.cat.categories):
            m = cut == k
            out[str(k)] = {**summarize(d.loc[m, y]), "placebo": summarize(d.loc[m, "around_placebo_abn"]),
                           "pre": summarize(d.loc[m, "pre_abn"]), "pre_placebo": summarize(d.loc[m, "pre_placebo_abn"])}
        return out
    res["by"] = {
        "unlocked_supply (overhang tercile)": split("overhang", q=3, labels=["low", "mid", "high"]),
        "price vs offer going in": split("price_vs_offer_before", bins=[-10, 0, 0.5, 100], labels=["below offer", "0–50% above", "50%+ above"]),
        "run-up since day-1 close": split("runup_since_day1", q=3, labels=["worst third", "middle", "best third"]),
        "era": split("year", bins=[2014, 2018, 2021, 2023, 2027], labels=["2015–18", "2019–21", "2022–23", "2024–26"]),
        "deal size": split("deal_size", q=3, labels=["small", "mid", "large"]),

    }
    (HERE / "results.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    print(f"events: {len(d)} | excluded: {res['excluded']}")
    print(f"lock-up lengths: {res['lockup_days']} | median event trading day {res['median_event_trading_day']}")
    print(f"volume: {res['volume']}")
    for k, v in res["windows"].items():
        a, pl = v["abnormal"], v["placebo_abnormal"]
        print(f"{k:7s} abn mean {a.get('mean_%')}% median {a.get('median_%')}% neg {a.get('pct_negative')}% "
              f"CI[{a.get('ci_lo_%')},{a.get('ci_hi_%')}] p={a.get('wilcoxon_p')} | placebo mean {pl.get('mean_%')}% p={pl.get('wilcoxon_p')}")
    for k, v in res["by"].items():
        print(f"\n{k}:")
        for g_, s_ in v.items():
            print(f"   {g_:14s} n={s_.get('n')} around: mean {s_.get('mean_%')}% median {s_.get('median_%')}% p={s_.get('wilcoxon_p')} "
                  f"(placebo mean {s_['placebo'].get('mean_%')}%) | pre: mean {s_['pre'].get('mean_%')}% (placebo {s_['pre_placebo'].get('mean_%')}%)")


if __name__ == "__main__":
    main()
