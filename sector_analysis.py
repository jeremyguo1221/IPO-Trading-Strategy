"""Sector-level analysis of IPO price behaviour over the first 121 trading days.

Same inputs and conventions as analyze.py (close-to-close returns, excess = stock minus SPY over the
same dates, a stock counts in a window only if it has data through the window end).
Adds: per-sector day-1 anatomy, window returns, typical path, volatility, beta to SPY, drawdowns,
and a listing-year-adjusted comparison (sector and listing year are heavily confounded).
Outputs: analysis/sector/*.csv and merges a "sector" block into analysis/report_data.json.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).parent
OUT = HERE / "analysis" / "sector"
OUT.mkdir(parents=True, exist_ok=True)
END = 121
BAD_DAY1 = {"KLAR", "IBTA", "CRBG", "INIO"}
EXCLUDE = {"PSUS", "SKHY"}
MIN_N = 3  # minimum stocks for a sector statistic to be reported

d = pd.read_csv(HERE / "ipo_first_6m_daily.csv", parse_dates=["date"])
s = pd.read_csv(HERE / "ipo_first_6m_summary.csv", parse_dates=["first_trade_date"])
spy = pd.read_csv(HERE / "spy_daily.csv", parse_dates=["date"]).set_index("date")
d, s = d[~d.symbol.isin(EXCLUDE)], s[~s.symbol.isin(EXCLUDE)]
meta = s.set_index("symbol")
d = d.join(spy.close.rename("spy"), on="date")

C = d.pivot(index="trading_day", columns="symbol", values="close").loc[:END]
O = d.pivot(index="trading_day", columns="symbol", values="open").loc[:END]
H = d.pivot(index="trading_day", columns="symbol", values="high").loc[:END]
L = d.pivot(index="trading_day", columns="symbol", values="low").loc[:END]
V = d.pivot(index="trading_day", columns="symbol", values="volume").astype(float).loc[:END]
S = d.pivot(index="trading_day", columns="symbol", values="spy").loc[:END]
offer, sector, year = meta.offer_price, meta.sector, meta.first_trade_date.dt.year
syms = list(C.columns)
SECTORS = sorted(sector[syms].unique())


def ex_ret(a, b):
    ok = C.loc[b].notna()
    return ((C.loc[b] / C.loc[a] - 1) - (S.loc[b] / S.loc[a] - 1))[ok]


# ---------------------------------------------------------------- per-stock metrics
ret, sret = C.pct_change(), S.pct_change()
per = pd.DataFrame(index=syms)
per["sector"], per["year"] = sector[syms], year[syms]
per["days"] = C.notna().sum()
per["pop"] = [C.loc[1, x] / offer[x] - 1 if x not in BAD_DAY1 and pd.notna(offer.get(x)) else np.nan for x in syms]
per["offer_to_open"] = [O.loc[1, x] / offer[x] - 1 if x not in BAD_DAY1 and pd.notna(offer.get(x)) else np.nan for x in syms]
per["open_to_close"] = [C.loc[1, x] / O.loc[1, x] - 1 if x not in BAD_DAY1 else np.nan for x in syms]
per["ex_2_21"] = ex_ret(1, 21)
per["ex_22_63"] = ex_ret(21, 63)
per["ex_64_121"] = ex_ret(63, END)
per["ex_6m"] = ex_ret(1, END)
per["vs_offer_6m"] = C.loc[END] / offer - 1
for x in syms:
    r = ret[x].loc[2:].dropna()
    per.loc[x, "abs_move_wk1"] = r.loc[2:5].abs().median()
    per.loc[x, "abs_move_after"] = r.loc[22:].abs().median() if len(r.loc[22:]) >= 20 else np.nan
    per.loc[x, "ann_vol"] = r.std() * np.sqrt(252) if len(r) >= 20 else np.nan
    per.loc[x, "pct_days_gt5"] = (r.abs() > .05).mean() if len(r) >= 20 else np.nan
    per.loc[x, "hl_range"] = ((H[x] - L[x]) / C[x].shift(1)).loc[2:].median()
    m = sret[x].loc[r.index]
    if len(r) >= 40:
        slope, _, rv, _, _ = stats.linregress(m.values, r.values)
        per.loc[x, "beta"], per.loc[x, "corr"] = slope, rv
    c = C[x].dropna()
    if len(c) >= END:
        per.loc[x, "max_dd"] = (c / c.cummax() - 1).min()
        per.loc[x, "peak_day"], per.loc[x, "trough_day"] = c.idxmax(), c.idxmin()
        per.loc[x, "ever_below_offer"] = float((c < offer[x]).any()) if pd.notna(offer.get(x)) else np.nan
    per.loc[x, "vol_day1_mult"] = V.loc[1, x] / V.loc[2:21, x].mean()
# listing-year adjustment: 6m excess relative to the median of IPOs from the same listing year
per["ex_6m_yr_adj"] = per.ex_6m - per.groupby("year").ex_6m.transform("median")
per.round(4).to_csv(OUT / "per_stock_sector_metrics.csv")


def med(col, g, pct=True):
    v = g[col].dropna()
    return (round(v.median() * (100 if pct else 1), 1 if pct else 2), len(v)) if len(v) >= MIN_N else (None, len(v))


# ---------------------------------------------------------------- sector table
rows = []
for sec in SECTORS:
    g = per[per.sector == sec]
    row = {"sector": sec, "n": len(g), "n_full": int(g.ex_6m.notna().sum())}
    for col in ["pop", "offer_to_open", "open_to_close", "ex_2_21", "ex_22_63", "ex_64_121", "ex_6m", "vs_offer_6m",
                "ex_6m_yr_adj", "abs_move_wk1", "abs_move_after", "ann_vol", "pct_days_gt5", "hl_range", "max_dd"]:
        row[col], row[col + "_n"] = med(col, g)
    for col in ["beta", "corr", "vol_day1_mult", "trough_day", "peak_day"]:
        row[col], row[col + "_n"] = med(col, g, pct=False)
    b = g.ex_6m.dropna()
    row["pct_beat_spy"] = round((b > 0).mean() * 100) if len(b) >= MIN_N else None
    row["ex_6m_p25"] = round(b.quantile(.25) * 100, 1) if len(b) >= MIN_N else None
    row["ex_6m_p75"] = round(b.quantile(.75) * 100, 1) if len(b) >= MIN_N else None
    eb = g.ever_below_offer.dropna()
    row["pct_below_offer"] = round(eb.mean() * 100) if len(eb) >= MIN_N else None
    p1 = g["pop"].dropna()
    row["pct_broke_issue"] = round((p1 <= 0).mean() * 100) if len(p1) >= MIN_N else None
    row["best"] = g.ex_6m.idxmax() if len(b) else None
    row["worst"] = g.ex_6m.idxmin() if len(b) else None
    row["years"] = ", ".join(f"{y}×{c}" for y, c in g.year.value_counts().sort_index().items())
    rows.append(row)
sec_tab = pd.DataFrame(rows)
sec_tab.to_csv(OUT / "sector_summary.csv", index=False)

# ---------------------------------------------------------------- are sector differences real? (Kruskal-Wallis)
def kw(col, min_n=MIN_N):
    groups = [per.loc[per.sector == sec, col].dropna().values for sec in SECTORS]
    groups = [g for g in groups if len(g) >= min_n]
    h, p = stats.kruskal(*groups)
    return dict(metric=col, groups=len(groups), H=round(h, 2), p=round(p, 4))


tests = pd.DataFrame([kw(c) for c in ["pop", "ex_2_21", "ex_22_63", "ex_64_121", "ex_6m", "ex_6m_yr_adj",
                                       "ann_vol", "abs_move_after", "beta", "max_dd"]])
tests.to_csv(OUT / "sector_difference_tests.csv", index=False)
yr_test = stats.kruskal(*[g.dropna().values for _, g in per.groupby("year").ex_6m if g.notna().sum() >= MIN_N])

# ---------------------------------------------------------------- typical path per sector (excess vs SPY from day-1 close)
paths = {}
days = list(C.index)
allpath = []
for t in days:
    e = ((C.loc[t] / C.loc[1] - 1) - (S.loc[t] / S.loc[1] - 1)).dropna()
    allpath.append(round(e.median() * 100, 2) if len(e) >= 20 else None)
for sec in SECTORS:
    cols = [x for x in syms if sector[x] == sec]
    vals, ns = [], []
    for t in days:
        e = ((C.loc[t, cols] / C.loc[1, cols] - 1) - (S.loc[t, cols] / S.loc[1, cols] - 1)).dropna()
        vals.append(round(e.median() * 100, 2) if len(e) >= MIN_N else None)
        ns.append(len(e))
    paths[sec] = dict(values=vals, n=ns)
pd.DataFrame({sec: p["values"] for sec, p in paths.items()} | {"All IPOs": allpath}, index=days).rename_axis("trading_day").to_csv(OUT / "sector_paths_excess_vs_spy.csv")

# ---------------------------------------------------------------- sector x year mix
mix = pd.crosstab(per.sector, per.year)
mix.to_csv(OUT / "sector_by_year_counts.csv")

stock_rows = per.reset_index(names="symbol")[["symbol", "sector", "year", "pop", "ex_6m", "vs_offer_6m", "ann_vol", "beta", "max_dd"]]
stock_rows = stock_rows.assign(company=lambda v: v.symbol.map(meta.company))

block = dict(
    table=sec_tab.astype(object).where(sec_tab.notna(), None).to_dict("records"), tests=tests.to_dict("records"),
    year_test=dict(H=round(yr_test.statistic, 2), p=round(yr_test.pvalue, 4)),
    paths=paths, all_path=allpath, days=days,
    mix=dict(sectors=list(mix.index), years=[int(y) for y in mix.columns], counts=mix.values.tolist()),
    stocks=[{k: (None if isinstance(v, float) and np.isnan(v) else (round(v * 100, 1) if k in ("pop", "ex_6m", "vs_offer_6m", "ann_vol", "max_dd") and v is not None else (round(v, 2) if k == "beta" else v)))
             for k, v in r.items()} for r in stock_rows.to_dict("records")],
)
rp = HERE / "analysis" / "report_data.json"
R = json.loads(rp.read_text(encoding="utf-8"))
R["sector"] = block
rp.write_text(json.dumps(R, default=lambda o: None if pd.isna(o) else o.item(), indent=1, ensure_ascii=False), encoding="utf-8")

pd.set_option("display.width", 250)
show = ["sector", "n", "n_full", "pop", "ex_2_21", "ex_22_63", "ex_64_121", "ex_6m", "ex_6m_yr_adj", "pct_beat_spy",
        "vs_offer_6m", "ann_vol", "abs_move_after", "pct_days_gt5", "beta", "corr", "max_dd", "pct_below_offer", "trough_day", "best", "worst"]
print(sec_tab[show].to_string(index=False))
print("\nKruskal-Wallis across sectors:\n", tests.to_string(index=False))
print("\nKruskal-Wallis across listing years (ex_6m):", block["year_test"])
print("\nSector x year:\n", mix.to_string())
print("\nyears per sector:\n", sec_tab[["sector", "years"]].to_string(index=False))
