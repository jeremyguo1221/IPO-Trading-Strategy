"""Phase 5: do sector-specific exits beat one global exit? (entry fixed = phase-3 rule)

Steps (search eras 2015-18, 2019-21, 2021-24 only, unless --validate):
  1. per-trade net return for each of 10 exits (unconstrained simulation, so portfolio capacity doesn't mix exits)
  2. sector buckets from data/full/sectors.csv; buckets with < 25 search trades merge into "Other"
  3. per-sector best exit by ERA-BALANCED mean net per trade (average of each era's mean)
  4. overfitting checks: leave-one-era-out (choose on 2 eras, test on the 3rd) and 500 shuffled-label runs
  5. --validate: 2024-26 ONCE (lock file; requires a completed dry run): full portfolio via the engine with
     per-trade exit overrides, sector-specific vs global 21-day vs global 5-day
Outputs: peak/sectors/cells.csv, loeo.csv, shuffle.json, summary.json (+ validation.json)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from lib.engine import Config, EXIT_FIELDS, simulate
from lib.evaluate import metrics
from peak.exits.search import ERA_LABEL, ERAS, default, load_bundle, to_cfg

HERE = Path(__file__).resolve().parent
FULL = HERE.parents[1] / "data" / "full"
LOCK = HERE / "VALIDATION_RUN.lock"
MIN_TRADES = 25
N_SHUFFLE = 500

D = default()
MENU = {
    "hold3": dict(D, hold=3), "hold5": dict(D, hold=5), "hold7": dict(D, hold=7), "hold10": dict(D, hold=10),
    "hold15": dict(D, hold=15), "hold21": dict(D), "target20": dict(D, target=0.20),
    "hold15+target30": dict(D, hold=15, target=0.30), "hold5+prev_high": dict(D, hold=5, sig="prev_high"),
    "thesis_stop": dict(D, thesis=True),
}


def per_trade_returns(ipos, entries, cal) -> pd.DataFrame:
    """rows = trades (symbol), columns = exits, values = net return per $ notional."""
    out = {}
    for name, x in MENU.items():
        cfg = to_cfg(x, size=0.0001, max_positions=10**6, gross_cap=10**6)
        tr = simulate(cfg, {s: ipos[s] for s in entries.symbol}, entries, cal)["trades"]
        out[name] = tr.set_index("symbol").net_ret
    return pd.DataFrame(out)


def era_balanced(R: pd.DataFrame, eras: pd.Series) -> pd.Series:
    """mean over eras of the per-era mean, for every column"""
    return R.groupby(eras).mean().mean()


def choose(R, eras, buckets):
    glob = era_balanced(R, eras).idxmax()
    per = {}
    for b in sorted(buckets.unique()):
        m = buckets == b
        per[b] = era_balanced(R[m], eras[m]).idxmax()
    return glob, per


def score_assignment(R, eras, buckets, per: dict) -> float:
    picked = pd.Series([R.at[i, per[b]] for i, b in zip(R.index, buckets)], index=R.index)
    return float(picked.groupby(eras).mean().mean())


def main():
    validate = "--validate" in sys.argv
    if validate and LOCK.exists():
        raise SystemExit(f"validation already run: {LOCK.read_text()}")
    if validate and not (HERE / "summary.json").exists():
        raise SystemExit("run the dry analysis (no flag) first")
    ipos, entries, cal = load_bundle()
    sec = pd.read_csv(FULL / "sectors.csv", keep_default_na=False, na_values=[""]).set_index("symbol")

    # 1-2. per-trade returns on the search eras, with sector buckets
    parts = []
    for era in ERAS:
        R = per_trade_returns(ipos, entries[era], cal)
        R["era"] = ERA_LABEL[era]
        parts.append(R)
    R = pd.concat(parts)
    R = R[~R.index.duplicated(keep="first")]
    eras = R.pop("era")
    raw_bucket = pd.Series([sec.bucket.get(s, "Unknown") for s in R.index], index=R.index)
    counts = raw_bucket.value_counts()
    small = [b for b, n in counts.items() if n < MIN_TRADES or b == "Unknown"]
    buckets = raw_bucket.where(~raw_bucket.isin(small), "Other")
    mapping = {"trades_per_raw_bucket": counts.to_dict(), "merged_into_Other": small,
               "trades_per_bucket": buckets.value_counts().to_dict()}
    print("bucket mapping:", json.dumps(mapping))

    # 3. per-sector cells
    rows = []
    for b in sorted(buckets.unique()):
        m = buckets == b
        eb = era_balanced(R[m], eras[m])
        for ex in MENU:
            rows.append({"bucket": b, "exit": ex, "trades": int(m.sum()), "era_balanced_mean_%": round(eb[ex] * 100, 2),
                         "pooled_mean_%": round(R.loc[m, ex].mean() * 100, 2), "win_rate_%": round((R.loc[m, ex] > 0).mean() * 100, 1)})
    cells = pd.DataFrame(rows)
    cells.to_csv(HERE / "cells.csv", index=False)
    glob, per = choose(R, eras, buckets)
    ins_sector = score_assignment(R, eras, buckets, per)
    ins_global = float(era_balanced(R, eras)[glob])
    print(f"global best exit: {glob} ({ins_global*100:.2f}%/trade) | per-sector: {per} ({ins_sector*100:.2f}%/trade in-sample)")

    # 4a. leave-one-era-out
    lo = []
    for e in eras.unique():
        tr_m, te_m = eras != e, eras == e
        g, p = choose(R[tr_m], eras[tr_m], buckets[tr_m])
        test_R, test_b = R[te_m], buckets[te_m]
        sect = np.mean([test_R.at[i, p.get(b, g)] for i, b in zip(test_R.index, test_b)])
        lo.append({"held_out_era": e, "global_exit": g, "sector_exits": json.dumps(p), "trades": int(te_m.sum()),
                   "global_mean_%": round(float(test_R[g].mean()) * 100, 2), "sector_mean_%": round(float(sect) * 100, 2)})
    loeo = pd.DataFrame(lo)
    loeo["sector_minus_global_%"] = loeo["sector_mean_%"] - loeo["global_mean_%"]
    loeo.to_csv(HERE / "loeo.csv", index=False)
    print("\nleave-one-era-out:\n" + loeo[["held_out_era", "global_exit", "global_mean_%", "sector_mean_%", "sector_minus_global_%"]].to_string(index=False))

    # 4b. shuffled labels: in-sample gain and leave-one-era-out gain under random sector labels
    rng = np.random.default_rng(21)
    def loeo_gain(bk):
        gains = []
        for e in eras.unique():
            tr_m, te_m = eras != e, eras == e
            g, p = choose(R[tr_m], eras[tr_m], bk[tr_m])
            tR, tb = R[te_m], bk[te_m]
            gains.append(np.mean([tR.at[i, p.get(b, g)] for i, b in zip(tR.index, tb)]) - tR[g].mean())
        return float(np.mean(gains))
    real_ins_gain = ins_sector - ins_global
    real_oos_gain = float(loeo["sector_minus_global_%"].mean() / 100)
    sh_ins, sh_oos = [], []
    for _ in range(N_SHUFFLE):
        bk = pd.Series(rng.permutation(buckets.to_numpy()), index=buckets.index)
        g, p = choose(R, eras, bk)
        sh_ins.append(score_assignment(R, eras, bk, p) - float(era_balanced(R, eras)[g]))
        sh_oos.append(loeo_gain(bk))
    shuffle = {"n": N_SHUFFLE, "real_in_sample_gain_%": round(real_ins_gain * 100, 3),
               "shuffled_in_sample_gain_p95_%": round(float(np.percentile(sh_ins, 95)) * 100, 3),
               "p_in_sample": round(float(np.mean(np.array(sh_ins) >= real_ins_gain)), 4),
               "real_loeo_gain_%": round(real_oos_gain * 100, 3),
               "shuffled_loeo_gain_median_%": round(float(np.median(sh_oos)) * 100, 3),
               "shuffled_loeo_gain_p95_%": round(float(np.percentile(sh_oos, 95)) * 100, 3),
               "p_loeo": round(float(np.mean(np.array(sh_oos) >= real_oos_gain)), 4)}
    (HERE / "shuffle.json").write_text(json.dumps(shuffle, indent=1), encoding="utf-8")
    print("\nshuffle test:", json.dumps(shuffle))
    passes_checks = bool(real_oos_gain > 0 and shuffle["p_loeo"] < 0.05)
    summary = {"mapping": mapping, "global_exit": glob, "sector_exits": per, "in_sample_%": {"global": round(ins_global * 100, 2),
               "sector": round(ins_sector * 100, 2)}, "loeo_mean_gain_%": round(real_oos_gain * 100, 3), "shuffle": shuffle,
               "passes_pre_validation_checks": passes_checks}
    if not validate:
        (HERE / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
        print("\npasses leave-one-era-out + shuffle checks:", passes_checks)
        return

    # 5. validation 2024-26, once: full portfolio via the engine with per-trade exit overrides
    LOCK.write_text(f"validation run {pd.Timestamp.now().isoformat(timespec='seconds')}")
    s0 = json.loads((HERE / "summary.json").read_text())
    ent = entries["validation"].copy()
    ent["bucket"] = [sec.bucket.get(s, "Unknown") for s in ent.symbol]
    ent["bucket"] = ent.bucket.where(~ent.bucket.isin(s0["mapping"]["merged_into_Other"]), "Other")
    field = {"hold": "hold", "target": "target", "stop": "stop", "trail": "trail", "thesis": "thesis_stop", "sig": "exit_signal"}
    for k, f in field.items():
        ent[f] = [MENU[s0["sector_exits"].get(b, s0["global_exit"])][k] for b in ent.bucket]
    val = {}
    sub = {s: ipos[s] for s in ent.symbol}
    for name, (e, x) in {"sector_specific": (ent, default()), "global_21d": (entries["validation"], MENU["hold21"]),
                         "global_5d": (entries["validation"], MENU["hold5"]),
                         "global_searched": (entries["validation"], MENU[s0["global_exit"]])}.items():
        res = simulate(to_cfg(x), sub, e, cal)
        m = metrics(res)
        val[name] = {k: m.get(k) for k in ["trades", "cagr", "sharpe", "max_dd", "profit_factor", "mean_trade", "total_return"]}
        if name == "sector_specific":
            tr = res["trades"].merge(ent[["symbol", "bucket"]], on="symbol")
            val[name]["by_bucket_mean_trade_%"] = (tr.groupby("bucket").net_ret.mean() * 100).round(2).to_dict()
            val[name]["exit_reasons"] = tr.exit_reason.value_counts().to_dict()
    (HERE / "validation.json").write_text(json.dumps(val, indent=1, default=str), encoding="utf-8")
    for k, v in val.items():
        print(f"{k:16s} n={v['trades']} cagr={v['cagr']} sharpe={v['sharpe']} dd={v['max_dd']} mean_trade={v['mean_trade']}")


if __name__ == "__main__":
    main()
