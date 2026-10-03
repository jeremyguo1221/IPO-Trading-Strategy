"""Trend analysis of IPO stocks over their first ~6 months (trading days 1–121).

Inputs (same folder): ipo_first_6m_daily.csv, ipo_first_6m_summary.csv, spy_daily.csv
Outputs: analysis/*.csv tables and analysis/report_data.json (feeds the HTML report).

Conventions
- Returns are close-to-close. "Day 1" = first trading day. Offer-based returns use the IPO offer price.
- Excess return = stock return minus SPY return over the same dates (simple difference).
- A stock is included in a window only if it has data through the end of that window.
- Significance: Wilcoxon signed-rank test on the per-stock values (H0: median = 0),
  plus a bootstrap 95% CI for the median.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).parent
OUT = HERE / "analysis"
OUT.mkdir(exist_ok=True)
RNG = np.random.default_rng(7)
END = 121  # 6-month endpoint: every complete 6-month window has >= 121 trading days

# valuation at IPO ($B) and base deal size ($B), from the IPO ledger
DEAL = {
 "WRBY":(6.8,0),"GTLB":(11.0,0.80),"PTLO":(1.4,0.41),"RENT":(1.7,0.36),"GFS":(25.0,2.6),"FLNC":(4.8,0.94),
 "BIRD":(2.2,0.30),"NRDS":(1.4,0.12),"ARHS":(1.7,0.36),"RIVN":(66.5,11.9),"BRZE":(6.0,0.52),"SG":(3.1,0.36),
 "NU":(41.5,2.6),"IOT":(11.5,0.81),"TPG":(9.0,1.0),"CRDO":(1.5,0.20),"BLCO":(6.3,0.63),"CRBG":(13.6,1.68),
 "MBLY":(17.0,1.0),"NXT":(3.5,0.64),"KVUE":(41.0,3.8),"ATMU":(1.6,0.28),"CAVA":(2.5,0.32),"SVV":(2.9,0.40),
 "KGS":(1.2,0.26),"ODD":(2.0,0.42),"ARM":(54.5,4.87),"CART":(9.9,0.66),"KVYO":(9.2,0.58),"BIRK":(8.6,1.48),
 "AS":(6.3,1.37),"ALAB":(5.5,0.71),"RDDT":(6.4,0.75),"ULS":(5.6,0.95),"IBTA":(2.7,0.58),"RBRK":(5.6,0.75),
 "LOAR":(2.5,0.31),"VIK":(10.3,1.54),"WAY":(3.6,0.97),"TEM":(6.1,0.41),"WBTN":(2.7,0.32),"LINE":(18.0,4.44),
 "SARO":(8.0,1.44),"KLC":(2.8,0.58),"INGM":(5.0,0.24),"WRD":(4.4,0.44),"PONY":(5.3,0.26),"TTAN":(6.3,0.63),
 "VG":(60.0,1.75),"SFD":(7.9,0.52),"SAIL":(12.8,1.38),"CRWV":(23.0,1.5),"ETOR":(4.2,0.62),"HNGE":(2.6,0.44),
 "CRCL":(6.9,1.05),"OMDA":(1.1,0.15),"VOYG":(1.9,0.38),"CHYM":(11.6,0.86),"CAI":(5.3,0.49),"FIG":(18.8,1.21),
 "FLY":(6.3,0.87),"BLSH":(5.4,1.11),"KLAR":(15.1,1.37),"FIGR":(5.3,0.79),"GEMI":(3.3,0.43),"STUB":(8.6,0.80),
 "NTSK":(7.3,0.91),"FRMI":(12.5,0.79),"NAVN":(6.2,0.92),"BETA":(6.6,1.02),"MDLN":(38.0,6.27),"BTGO":(None,0.21),
 "EQPT":(6.0,0.75),"YSS":(None,0.63),"FPS":(8.2,1.51),"BOBS":(2.2,0.33),"MWH":(None,0.51),"MMED":(5.6,0.56),
 "PAYP":(12.4,0.88),"MAIR":(13.2,2.23),"ARXS":(11.0,1.13),"XE":(None,1.02),"PS":(None,5.0),"PSUS":(None,5.0),
 "HAWK":(None,0.42),"FRVO":(None,1.89),"CBRS":(56.4,5.55),"BXDC":(None,1.75),"PWRL":(None,1.51),"QNT":(None,1.68),
 "INIO":(25.5,2.43),"SPCX":(1745,75.0),"DPC":(None,0.92),"BSP":(19.5,1.68),"LIME":(1.8,0.18),"SKHY":(1086,26.5),
 "CSQR":(3.25,1.05),"JMKE":(7.3,1.0),
}
# first bar in the DB is not the true debut for these; excluded from day-1/offer-to-day-1 stats
BAD_DAY1 = {"KLAR", "IBTA", "CRBG", "INIO"}
# closed-end fund / already-public ADR: not a normal operating-company IPO; excluded from the main sample
EXCLUDE = {"PSUS", "SKHY"}

d = pd.read_csv(HERE / "ipo_first_6m_daily.csv", parse_dates=["date"])
s = pd.read_csv(HERE / "ipo_first_6m_summary.csv", parse_dates=["first_trade_date"])
spy = pd.read_csv(HERE / "spy_daily.csv", parse_dates=["date"]).set_index("date")
spy["prev_close"] = spy.close.shift(1)

d = d[~d.symbol.isin(EXCLUDE)].copy()
s = s[~s.symbol.isin(EXCLUDE)].copy()
meta = s.set_index("symbol")
d = d.join(spy[["close", "prev_close"]].rename(columns={"close": "spy", "prev_close": "spy_prev"}), on="date")
assert d.spy.notna().all(), "SPY missing on some IPO trading dates"

# wide matrices: rows = trading day (1..N), cols = symbol
C = d.pivot(index="trading_day", columns="symbol", values="close")
O = d.pivot(index="trading_day", columns="symbol", values="open")
H = d.pivot(index="trading_day", columns="symbol", values="high")
L = d.pivot(index="trading_day", columns="symbol", values="low")
V = d.pivot(index="trading_day", columns="symbol", values="volume").astype(float)
S = d.pivot(index="trading_day", columns="symbol", values="spy")
S0 = d[d.trading_day == 1].set_index("symbol").spy_prev  # SPY close the day before the debut
offer = meta.offer_price
MAXDAY = int(C.index.max())
syms = list(C.columns)
print(f"{len(syms)} IPOs in sample; max trading day {MAXDAY}")


def summarize(x):
    x = pd.Series(x).dropna()
    n = len(x)
    if n < 5:
        return dict(n=n, median=np.nan, mean=np.nan, pct_pos=np.nan, ci_lo=np.nan, ci_hi=np.nan, p=np.nan)
    boots = np.median(RNG.choice(x.values, (4000, n)), axis=1)
    p = stats.wilcoxon(x.values).pvalue if (x != 0).any() else 1.0
    return dict(n=n, median=round(x.median() * 100, 2), mean=round(x.mean() * 100, 2),
                pct_pos=round((x > 0).mean() * 100, 1),
                ci_lo=round(np.percentile(boots, 2.5) * 100, 2), ci_hi=round(np.percentile(boots, 97.5) * 100, 2),
                p=round(p, 4))


def win(a, b):
    """close(day a) -> close(day b) raw and excess returns for stocks with data through day b."""
    ok = C.loc[b].notna()
    raw = (C.loc[b] / C.loc[a] - 1)[ok]
    mkt = (S.loc[b] / S.loc[a] - 1)[ok]
    return raw, raw - mkt


R = {}  # everything that goes to the report

# ---------------------------------------------------------------- 1. the typical path
path_rows = []
for t in C.index:
    have = C.loc[t].notna()
    if have.sum() < 20 or t > END:
        break
    vs_offer = (C.loc[t] / offer - 1)[have].dropna()
    vs_d1 = (C.loc[t] / C.loc[1] - 1)[have]
    ex_d1 = vs_d1 - (S.loc[t] / S.loc[1] - 1)[have]
    path_rows.append(dict(day=t, n=int(have.sum()),
        offer_p25=vs_offer.quantile(.25)*100, offer_med=vs_offer.median()*100, offer_p75=vs_offer.quantile(.75)*100,
        d1_med=vs_d1.median()*100, d1_p25=vs_d1.quantile(.25)*100, d1_p75=vs_d1.quantile(.75)*100,
        ex_med=ex_d1.median()*100, ex_mean=ex_d1.mean()*100,
        pct_above_offer=(vs_offer > 0).mean()*100))
path = pd.DataFrame(path_rows).round(2)
path.to_csv(OUT / "path_by_trading_day.csv", index=False)
R["path"] = path.to_dict("list")

# ---------------------------------------------------------------- 2. day one anatomy
ok1 = [x for x in syms if x not in BAD_DAY1 and pd.notna(offer.get(x))]
d1 = pd.DataFrame({
    "offer_to_open": O.loc[1, ok1] / offer[ok1] - 1,
    "open_to_close": C.loc[1, ok1] / O.loc[1, ok1] - 1,
    "offer_to_close": C.loc[1, ok1] / offer[ok1] - 1,
    "intraday_range": (H.loc[1, ok1] - L.loc[1, ok1]) / O.loc[1, ok1],
})
day1_stats = {k: summarize(v) for k, v in d1.items()}
day1_stats["broke_issue_close"] = round((d1.offer_to_close < 0).mean() * 100, 1)
day1_stats["opened_below_offer"] = round((d1.offer_to_open < 0).mean() * 100, 1)
day1_stats["closed_below_open"] = round((d1.open_to_close < 0).mean() * 100, 1)
R["day1"] = day1_stats
R["day1_hist"] = (d1.offer_to_close * 100).round(1).sort_values().to_dict()

# ---------------------------------------------------------------- 3. window returns (period by period)
WINDOWS = [(1, 2, "Day 2"), (2, 5, "Days 3–5"), (5, 10, "Days 6–10"), (10, 21, "Days 11–21"),
           (21, 42, "Days 22–42"), (42, 63, "Days 43–63"), (63, 94, "Days 64–94"), (94, END, "Days 95–121")]
wrows = []
for a, b, lab in WINDOWS:
    raw, ex = win(a, b)
    r, e = summarize(raw), summarize(ex)
    wrows.append(dict(window=lab, start_day=a, end_day=b, n=r["n"],
                      raw_median=r["median"], raw_mean=r["mean"], raw_pct_pos=r["pct_pos"], raw_p=r["p"],
                      ex_median=e["median"], ex_mean=e["mean"], ex_pct_pos=e["pct_pos"],
                      ex_ci_lo=e["ci_lo"], ex_ci_hi=e["ci_hi"], ex_p=e["p"]))
wtab = pd.DataFrame(wrows)
wtab.to_csv(OUT / "window_returns.csv", index=False)
R["windows"] = wrows

# ---------------------------------------------------------------- 4. how much they move: volatility, range, volume decay
ret = C.pct_change()
absret = ret.abs()
rng_pct = (H - L) / C.shift(1).fillna(O)
spy_ret = S.pct_change()
BINS = [(2, 5), (6, 10), (11, 21), (22, 42), (43, 63), (64, 94), (95, END)]
vrows = []
for a, b in BINS:
    blk = ret.loc[a:b]
    have = blk.notna().sum() >= (b - a + 1) * 0.8
    ann = (blk.loc[:, have].std() * np.sqrt(252)).median()
    spy_ann = (spy_ret.loc[a:b, have].std() * np.sqrt(252)).median()
    vrows.append(dict(days=f"{a}–{b}", n=int(have.sum()),
                      median_abs_daily_move=round(absret.loc[a:b, have].stack().median() * 100, 2),
                      median_intraday_range=round(rng_pct.loc[a:b, have].stack().median() * 100, 2),
                      share_days_over_5pct=round((absret.loc[a:b, have].stack() > .05).mean() * 100, 1),
                      median_annualized_vol=round(ann * 100, 1), spy_annualized_vol=round(spy_ann * 100, 1)))
vtab = pd.DataFrame(vrows)
vtab.to_csv(OUT / "volatility_by_window.csv", index=False)
R["vol"] = vrows

# daily series for charts (days 2..END)
days = list(range(2, min(MAXDAY, END) + 1))
enough = lambda t: C.loc[t].notna().sum() >= 20
vol_daily = [dict(day=t, abs_move=round(absret.loc[t].median() * 100, 2),
                  range=round(rng_pct.loc[t].median() * 100, 2)) for t in days if enough(t)]
R["vol_daily"] = vol_daily
v_rel = V.div(V.loc[2:21].mean())  # volume relative to each stock's day 2–21 average
R["volume_daily"] = [dict(day=int(t), rel=round(v_rel.loc[t].median(), 3)) for t in V.index if enough(t) and t <= END]
R["volume_day1_vs_d2_21"] = round(v_rel.loc[1].median(), 2)

# ---------------------------------------------------------------- 5. drawdowns, peaks, breaking issue
full = [x for x in syms if C[x].notna().sum() >= 120]
dd_rows = []
for x in full:
    c = C[x].dropna().iloc[:END]
    off = offer.get(x)
    below = c[c < off] if pd.notna(off) else pd.Series(dtype=float)
    dd_rows.append(dict(symbol=x,
        peak_day=int(c.idxmax()), trough_day=int(c.idxmin()),
        max_drawdown=(c / c.cummax() - 1).min() * 100,
        max_gain_from_d1=(c.max() / c.iloc[0] - 1) * 100,
        ever_below_offer=bool(len(below)) if pd.notna(off) else None,
        first_day_below_offer=int(below.index[0]) if len(below) else None))
dd = pd.DataFrame(dd_rows)
dd.to_csv(OUT / "drawdown_peak_trough.csv", index=False)
R["drawdown"] = dict(
    n=len(dd), median_max_drawdown=round(dd.max_drawdown.median(), 1),
    pct_dd_over_30=round((dd.max_drawdown < -30).mean() * 100, 1),
    pct_dd_over_50=round((dd.max_drawdown < -50).mean() * 100, 1),
    pct_ever_below_offer=round(dd.ever_below_offer.dropna().astype(bool).mean() * 100, 1),
    median_first_day_below_offer=float(dd.first_day_below_offer.dropna().median()),
    peak_day_hist=np.histogram(dd.peak_day, bins=[1, 2, 6, 22, 43, 64, 95, END + 1])[0].tolist(),
    trough_day_hist=np.histogram(dd.trough_day, bins=[1, 2, 6, 22, 43, 64, 95, END + 1])[0].tolist(),
    hist_labels=["Day 1", "2–5", "6–21", "22–42", "43–63", "64–94", "95–121"],
)

# ---------------------------------------------------------------- 6. entry-day x holding-period grid
ENTRY = [1, 2, 5, 10, 21, 42, 63]
HOLD = [5, 21, 63]
grid = []
for e in ENTRY:
    for h in HOLD:
        if e + h > END:
            continue
        raw, ex = win(e, e + h)
        g = summarize(ex)
        grid.append(dict(entry_day=e, hold_days=h, n=g["n"], ex_median=g["median"], ex_mean=g["mean"],
                         ex_pct_pos=g["pct_pos"], ex_p=g["p"], raw_median=summarize(raw)["median"]))
gtab = pd.DataFrame(grid)
gtab.to_csv(OUT / "entry_hold_grid.csv", index=False)
R["grid"] = grid

# ---------------------------------------------------------------- 7. does the day-1 pop predict what comes next?
rows = []
for x in syms:
    if x in BAD_DAY1 or pd.isna(offer.get(x)) or pd.isna(C[x].get(END)):
        continue
    pop = C.loc[1, x] / offer[x] - 1
    after = C.loc[END, x] / C.loc[1, x] - 1
    after_ex = after - (S.loc[END, x] / S.loc[1, x] - 1)
    rows.append(dict(symbol=x, pop=pop, after=after, after_ex=after_ex, vs_offer=C.loc[END, x] / offer[x] - 1))
pp = pd.DataFrame(rows)
pp["bucket"] = pd.cut(pp["pop"], [-1, 0, .15, .40, 10], labels=["Broke issue (≤0%)", "0–15%", "15–40%", ">40%"])
pb = pp.groupby("bucket", observed=True).agg(
    n=("symbol", "size"), pop_median=("pop", "median"), after_median=("after", "median"),
    after_ex_median=("after_ex", "median"), vs_offer_median=("vs_offer", "median"),
    pct_after_pos=("after", lambda v: (v > 0).mean())).reset_index()
for c in ["pop_median", "after_median", "after_ex_median", "vs_offer_median", "pct_after_pos"]:
    pb[c] = (pb[c] * 100).round(1)
pb["bucket"] = pb.bucket.astype(str)
pb.to_csv(OUT / "day1_pop_buckets.csv", index=False)
rho, rho_p = stats.spearmanr(pp["pop"], pp.after_ex)
R["pop"] = dict(buckets=pb.to_dict("records"), spearman=round(rho, 3), spearman_p=round(rho_p, 4),
                points=pp[["symbol", "pop", "after_ex"]].assign(pop=lambda v: (v["pop"] * 100).round(1),
                                                               after_ex=lambda v: (v.after_ex * 100).round(1)).to_dict("records"))

# ---------------------------------------------------------------- 8. early momentum -> later returns
mom = []
for x in syms:
    if pd.isna(C[x].get(END)):
        continue
    early = C.loc[21, x] / C.loc[1, x] - 1 - (S.loc[21, x] / S.loc[1, x] - 1)
    late = C.loc[END, x] / C.loc[21, x] - 1 - (S.loc[END, x] / S.loc[21, x] - 1)
    mom.append(dict(symbol=x, early_ex=early, late_ex=late))
mom = pd.DataFrame(mom)
r_m, p_m = stats.spearmanr(mom.early_ex, mom.late_ex)
mom["early_half"] = np.where(mom.early_ex >= mom.early_ex.median(), "Top half days 2–21", "Bottom half days 2–21")
mh = mom.groupby("early_half").late_ex.agg(["size", "median", "mean"]).reset_index()
R["momentum"] = dict(spearman=round(r_m, 3), p=round(p_m, 4), n=len(mom),
                     halves=[dict(group=r.early_half, n=int(r["size"]), late_median=round(r["median"] * 100, 1),
                                  late_mean=round(r["mean"] * 100, 1)) for _, r in mh.iterrows()])

# ---------------------------------------------------------------- 9. research quiet-period expiry (25 calendar days)
qrows = []
for x in syms:
    dates = d[d.symbol == x].set_index("trading_day").date
    first = dates.iloc[0]
    end = first + pd.Timedelta(days=25)
    after = dates[dates >= end]
    if len(after) == 0:
        continue
    q = int(after.index[0])  # first trading day on/after quiet-period end
    if q - 3 < 1 or pd.isna(C[x].get(q + 5)):
        continue
    raw = C.loc[q + 5, x] / C.loc[q - 3, x] - 1
    mkt = S.loc[q + 5, x] / S.loc[q - 3, x] - 1
    qrows.append(dict(symbol=x, quiet_day=q, raw=raw, ex=raw - mkt))
qp = pd.DataFrame(qrows)
qs = summarize(qp.ex)
R["quiet"] = dict(typical_trading_day=int(qp.quiet_day.median()), **qs,
                  raw_median=round(qp.raw.median() * 100, 2))

# ---------------------------------------------------------------- 10. cross-sections: sector, year, size
cs = pd.DataFrame(index=[x for x in syms if pd.notna(C[x].get(END))])
cs["sector"] = meta.sector
cs["year"] = meta.first_trade_date.dt.year
cs["ret_vs_offer"] = [C.loc[END, x] / offer[x] - 1 if pd.notna(offer.get(x)) else np.nan for x in cs.index]
cs["ex_from_d1"] = [(C.loc[END, x] / C.loc[1, x] - 1) - (S.loc[END, x] / S.loc[1, x] - 1) for x in cs.index]
cs["val"] = [DEAL[x][0] for x in cs.index]
cs["size"] = pd.cut(cs.val, [0, 3, 8, 20, 1e5], labels=["<$3B", "$3–8B", "$8–20B", ">$20B"])


def grp(col):
    g = cs.groupby(col, observed=True).agg(n=("ex_from_d1", "size"),
        vs_offer_median=("ret_vs_offer", "median"), ex_median=("ex_from_d1", "median"),
        pct_beat_spy=("ex_from_d1", lambda v: (v > 0).mean()))
    for c in ["vs_offer_median", "ex_median", "pct_beat_spy"]:
        g[c] = (g[c] * 100).round(1)
    return g.reset_index().rename(columns={col: "group"}).assign(group=lambda v: v.group.astype(str))


for name in ["sector", "year", "size"]:
    t = grp(name)
    t.to_csv(OUT / f"by_{name}.csv", index=False)
    R[f"by_{name}"] = t.to_dict("records")

# ---------------------------------------------------------------- 11. headline 6-month numbers
six_vs_offer = cs.ret_vs_offer.dropna()
R["headline"] = dict(
    n_total=len(syms), n_full=len(cs),
    six_vs_offer=summarize(six_vs_offer), six_ex_from_d1=summarize(cs.ex_from_d1),
    six_ex_from_offer=summarize(pd.Series({x: (C.loc[END, x] / offer[x] - 1) - (S.loc[END, x] / S0[x] - 1)
                                           for x in cs.index if pd.notna(offer.get(x)) and x not in BAD_DAY1})),
)
R["excluded"] = sorted(EXCLUDE)
R["bad_day1"] = sorted(BAD_DAY1)

(OUT / "report_data.json").write_text(json.dumps(R, default=lambda o: None if pd.isna(o) else o.item(), indent=1, ensure_ascii=False), encoding="utf-8")
cs.round(4).to_csv(OUT / "per_stock_6m.csv")

# console digest
print("\nDAY 1:", {k: v for k, v in day1_stats.items()})
print("\nWINDOWS:\n", wtab[["window", "n", "raw_median", "ex_median", "ex_pct_pos", "ex_ci_lo", "ex_ci_hi", "ex_p"]].to_string(index=False))
print("\nVOL:\n", vtab.to_string(index=False))
print("\nVOLUME day1 vs d2-21 avg:", R["volume_day1_vs_d2_21"])
print("\nDRAWDOWN:", R["drawdown"])
print("\nGRID:\n", gtab.to_string(index=False))
print("\nPOP BUCKETS:\n", pb.to_string(index=False), "\nspearman", R["pop"]["spearman"], R["pop"]["spearman_p"])
print("\nMOMENTUM:", R["momentum"])
print("\nQUIET PERIOD:", R["quiet"])
for n in ["sector", "year", "size"]:
    print(f"\nBY {n.upper()}:\n", pd.DataFrame(R[f'by_{n}']).to_string(index=False))
print("\nHEADLINE:", R["headline"])
