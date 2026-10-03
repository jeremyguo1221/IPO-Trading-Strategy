"""Phase 2 trials: 2 model families x 3 feature sets x 2 threshold modes = 12 trials, plus 2 baselines.
All use walk-forward out-of-sample decisions for 2020-2026 (development years only; the 2015-2018 fresh holdout
is never touched here) and run through the same engine, costs, hedge and sizing as the phase-1 final strategy.

Outputs: runs_ml/<run_id>/ (config.json, oos_predictions.csv, folds.csv, trades.csv, equity.csv, metrics.json)
         trials_ml.csv (one row per trial/baseline)
Selection rule (fixed in the plan before running): best WORST-YEAR out-of-sample Sharpe over 2020-2026.
"""
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from lib.engine import Config, find_entries, simulate
from lib.evaluate import metrics
from lib.signals import SIGNALS
from ml.dataset import load_full
from ml.walkforward import decision_table, walk_forward

HERE = Path(__file__).parent
RUNS = HERE / "runs_ml"
OOS_START = pd.Timestamp("2020-01-01")
SIM = dict(hedge=True, range_target=0.186, hold=21, last_signal_day=20)  # same trade mechanics as phase-1 final


def yearly_sharpe(eq: pd.Series) -> dict:
    full = pd.concat([pd.Series({eq.index[0] - pd.Timedelta(days=1): 1.0}), eq])
    r = full.pct_change().dropna()
    out = {}
    for y, g in r.groupby(r.index.year):
        out[int(y)] = round(float(g.mean() / g.std(ddof=1) * np.sqrt(252)), 3) if len(g) > 20 and g.std() > 0 else None
    return out


def simulate_entries(cfg, ipos, entries, cal):
    entries = entries[entries.entry_date >= OOS_START]
    res = simulate(cfg, {s: ipos[s] for s in entries.symbol.unique()}, entries, cal)
    res["entries"] = entries
    return res


def record(run_id, cfg, res, extra=None, preds=None, folds=None):
    d = RUNS / run_id
    d.mkdir(parents=True, exist_ok=True)
    c = cfg.as_dict()
    c["params"] = {k: (f"<{len(v)} decisions>" if k == "table" else v) for k, v in c["params"].items()}
    (d / "config.json").write_text(json.dumps({**c, **(extra or {})}, indent=1, default=str), encoding="utf-8")
    if preds is not None:
        preds.to_csv(d / "oos_predictions.csv", index=False)
    if folds is not None:
        folds.to_csv(d / "folds.csv", index=False)
    res["trades"].to_csv(d / "trades.csv", index=False)
    res["equity"].to_csv(d / "equity.csv", header=True)
    m = metrics(res)
    m["yearly_sharpe"] = yearly_sharpe(res["equity"])
    ys = [v for v in m["yearly_sharpe"].values() if v is not None]
    m["worst_year_sharpe"] = min(ys) if ys else None
    (d / "metrics.json").write_text(json.dumps(m, indent=1, default=str), encoding="utf-8")
    return m


def main():
    ipos, ctx, cal, meta, panel = load_full()
    dev_ipos = {s: i for s, i in ipos.items() if i.split != "fresh_holdout"}
    cands = pd.read_csv(HERE / "ml" / "candidates.csv", keep_default_na=False, na_values=[""],
                        parse_dates=["signal_date", "entry_date", "exit_date"])
    dev = cands[cands.split != "fresh_holdout"].reset_index(drop=True)
    rows = []

    def add(run_id, kind, m, extra=None):
        row = {"run_id": run_id, "kind": kind, **(extra or {})}
        for k in ["trades", "cagr", "sharpe", "max_dd", "profit_factor", "win_rate", "mean_trade", "boot_p",
                  "worst_year_sharpe", "daily_sr", "n_days", "skew", "kurt", "avg_gross"]:
            row[k] = m.get(k)
        row["yearly_sharpe"] = json.dumps(m.get("yearly_sharpe"))
        row["yearly_return"] = json.dumps(m.get("yearly"))
        rows.append(row)
        print(f"{run_id:28s} n={m.get('trades')} cagr={m.get('cagr')} sh={m.get('sharpe')} dd={m.get('max_dd')} "
              f"pf={m.get('profit_factor')} worst_yr_sh={m.get('worst_year_sharpe')} yearly={m.get('yearly_sharpe')}")

    # baselines
    table_all = decision_table(dev.assign(short_it=True))
    cfg = Config("B_all_candidates", signal="lookup", params={"table": table_all}, **SIM)
    res = simulate_entries(cfg, dev_ipos, find_entries(cfg, dev_ipos, SIGNALS["lookup"], ctx), cal)
    add("B_all_candidates", "baseline", record("B_all_candidates", cfg, res))
    lead = {"from_day": 2, "breakdown": ["d1_low", "offer"], "hot_count_min": 25}
    cfg = Config("B_phase1_R9", signal="rules", params=lead, hedge=True, range_target=0.186, hold=21)
    res = simulate_entries(cfg, dev_ipos, find_entries(cfg, dev_ipos, SIGNALS["rules"], ctx), cal)
    add("B_phase1_R9", "baseline", record("B_phase1_R9", cfg, res))

    # 12 trials
    for fam, fset, thr in itertools.product(["logit", "gbm"], ["ipo", "regime", "all"], ["fixed", "inner"]):
        run_id = f"M_{fam}_{fset}_{thr}"
        preds, folds = walk_forward(dev, fam, fset, thr)
        cfg = Config(run_id, signal="lookup", params={"table": decision_table(preds)}, **SIM)
        res = simulate_entries(cfg, dev_ipos, find_entries(cfg, dev_ipos, SIGNALS["lookup"], ctx), cal)
        extra = {"family": fam, "feature_set": fset, "threshold_mode": thr}
        m = record(run_id, cfg, res, extra, preds, folds)
        ic = preds.groupby("year").apply(lambda g: g.pred.corr(g.y, method="spearman"), include_groups=False)
        extra |= {"oos_ic_mean": round(float(ic.mean()), 4), "oos_ic_by_year": json.dumps(ic.round(3).to_dict()),
                  "candidates_taken_pct": round(float(preds["short_it"].mean() * 100), 1)}
        add(run_id, "trial", m, extra)

    t = pd.DataFrame(rows)
    t.to_csv(HERE / "trials_ml.csv", index=False)
    tr = t[t.kind == "trial"].sort_values("worst_year_sharpe", ascending=False)
    print("\nselection ranking (worst-year OOS Sharpe):")
    print(tr[["run_id", "worst_year_sharpe", "sharpe", "cagr", "max_dd", "trades", "oos_ic_mean"]].to_string(index=False))


if __name__ == "__main__":
    main()
