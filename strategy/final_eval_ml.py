"""Phase 2 final evaluation. Selection = best worst-year OOS Sharpe in trials_ml.csv (rule fixed in the plan).

  --dry-run : development only (2020-2026 walk-forward OOS): stress, shuffled null, feature diagnostics. No holdout.
  (no flag) : everything above + retrain on ALL development candidates, predict and trade the 2015-2018 fresh
              holdout ONCE. A lock file blocks reruns; the dry run must have completed first.
Outputs -> backtest_results_ml/ (or backtest_results_ml/dry_run/).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.inspection import permutation_importance

from lib.engine import Config, find_entries, simulate
from lib.evaluate import deflated_sharpe, metrics
from lib.features import ALL_FEATURES
from lib.placebo import random_established_shorts, shuffled_signal_null
from lib.signals import SIGNALS
from ml.dataset import FULL, load_full
from ml.walkforward import FEATURE_SETS, TEST_YEARS, decision_table, final_fit, fit_predict, make_model
from run_ml import SIM, yearly_sharpe

HERE = Path(__file__).parent
PASS_BAR = {"cagr": 15.0, "sharpe": 1.0, "max_dd": -25.0, "profit_factor": 1.3, "pct_years_pos": 75.0, "trades": 30,
            "boot_p": 0.05, "dsr": 0.9}


def scorecard(m, stress60, null_p95, dsr):
    c = {"CAGR >= 15%": (m.get("cagr"), m.get("cagr") is not None and m["cagr"] >= 15),
         "Sharpe >= 1.0": (m["sharpe"], m["sharpe"] >= 1.0),
         "Max drawdown <= 25%": (m["max_dd"], m["max_dd"] >= -25),
         "Profit factor >= 1.3": (m["profit_factor"], m["profit_factor"] >= 1.3),
         ">= 75% of years positive": (m["pct_years_pos"], m["pct_years_pos"] >= 75),
         ">= 30 trades": (m["trades"], m["trades"] >= 30),
         "Bootstrap p < 0.05": (m["boot_p"], m["boot_p"] < 0.05),
         "Profitable at 60%/yr borrow": (stress60["total_return"], (stress60["total_return"] or -1) > 0),
         "Sharpe > 95th pct of shuffled null": ((m["sharpe"], round(null_p95, 3)), m["sharpe"] > null_p95),
         "Deflated Sharpe prob > 0.9": (round(dsr, 3), dsr > 0.9)}
    return {k: {"value": v, "pass": bool(ok)} for k, (v, ok) in c.items()}


def evaluate(label, cfg, ipos, entries, cal, out, trial_srs, placebo=None):
    res = simulate(cfg, {s: ipos[s] for s in entries.symbol.unique()}, entries, cal)
    m = metrics(res)
    m["yearly_sharpe"] = yearly_sharpe(res["equity"])
    for k, v in {"trades": res["trades"], "equity": res["equity"], "ledger": res["ledger"]}.items():
        v.to_csv(out / f"{k}_{label}.csv", **({"index": False} if k != "equity" else {"header": True}))
    stress = {}
    for name, kw in [("borrow_60", {"borrow": 0.6}), ("borrow_100", {"borrow": 1.0}), ("slippage_50bps", {"slippage": 0.005}),
                     ("zero_cost", {"borrow": 0.0, "slippage": 0.0}), ("unhedged", {"hedge": False})]:
        c2 = Config(**{**cfg.as_dict(), **kw, "name": f"{cfg.name}_{name}"})
        ms = metrics(simulate(c2, {s: ipos[s] for s in entries.symbol.unique()}, entries, cal))
        stress[name] = {k: ms.get(k) for k in ["trades", "total_return", "cagr", "sharpe", "max_dd", "profit_factor"]}
    pool = {s: i for s, i in ipos.items() if i.dates[0] >= entries.entry_date.min() - pd.Timedelta(days=45)
            and i.dates[0] <= entries.entry_date.max()}
    null = shuffled_signal_null(cfg, pool, entries, cal, n=200, seed=7)
    null.to_csv(out / f"shuffled_null_{label}.csv", index=False)
    dsr = deflated_sharpe(m["daily_sr"], m["n_days"], m["skew"], m["kurt"], trial_srs)
    r = {"metrics": m, "stress": stress, "null_sharpe_p95": float(null.sharpe.quantile(0.95)),
         "null_sharpe_median": float(null.sharpe.median()), "null_pct_below_strategy": float((null.sharpe < m["sharpe"]).mean() * 100),
         "dsr": dsr}
    r["scorecard"] = scorecard(m, stress["borrow_60"], r["null_sharpe_p95"], dsr)
    r["passes_all"] = all(v["pass"] for v in r["scorecard"].values())
    if placebo is not None:
        pb, spy = placebo
        pl = random_established_shorts(cfg, res["trades"], pb, spy, cal, seed=13, draws=3)
        r["placebo"] = {"trades": int(len(pl)), "mean_gross_%": round(pl.gross_ret.mean() * 100, 3), "mean_net_%": round(pl.net_ret.mean() * 100, 3)}
    return r, res


def spot_audit(trades, cfg, n=5, seed=3):
    raw = pd.read_csv(FULL / "ipo_bars.csv", keep_default_na=False, na_values=[""], parse_dates=["date"])
    spy = pd.read_csv(FULL / "spy.csv", parse_dates=["date"]).set_index("date")
    out = []
    for t in trades.sample(min(n, len(trades)), random_state=seed).itertuples():
        b = raw[raw.symbol == t.symbol].set_index("date").sort_index()
        ed, xd = pd.Timestamp(t.entry_date), pd.Timestamp(t.exit_date)
        e, x = b.loc[ed, "open"], b.loc[xd, "close"]
        held = b.loc[ed:xd]
        so, sc = spy.loc[ed, "open"], spy.loc[xd, "close"]
        hand = (1 - x / e) - cfg.slippage * (1 + x / e) - cfg.borrow / 252 * (1 + held.close.iloc[:-1].sum() / e) \
            + t.beta * (sc / so - 1) - cfg.spy_slippage * t.beta * (1 + sc / so)
        out.append({"symbol": t.symbol, "entry_date": str(ed.date()), "exit_date": str(xd.date()), "entry_open_raw": round(e, 4),
                    "exit_close_raw": round(x, 4), "engine_net_ret": round(t.net_ret, 8), "hand_net_ret": round(hand, 8),
                    "match": abs(t.net_ret - hand) < 1e-6})
    return out


def feature_diagnostics(dev, family, fset, out):
    """Univariate OOS rank correlation of each feature with the label, by test year; and permutation importance of
    the selected model on each walk-forward test fold (refit on that fold's training window)."""
    ic = {f: [dev[dev.year == y][f].corr(dev[dev.year == y].y, method="spearman") for y in TEST_YEARS] for f in ALL_FEATURES}
    ict = pd.DataFrame(ic, index=TEST_YEARS).T
    ict["mean"] = ict.mean(axis=1)
    ict["years_same_sign_as_mean"] = (np.sign(ict[TEST_YEARS]) == np.sign(ict["mean"]).to_numpy()[:, None]).sum(axis=1)
    ict.round(3).to_csv(out / "feature_ic_by_year.csv")
    feats = FEATURE_SETS[fset]
    imps = []
    for y in TEST_YEARS:
        tr, te = dev[dev.exit_date < pd.Timestamp(f"{y}-01-01")], dev[dev.year == y]
        if len(tr) < 50 or len(te) < 30:
            continue
        m = make_model(family)
        if family == "logit":
            m.fit(tr[feats].to_numpy(float), (tr.y > 0).astype(int))
            pi = permutation_importance(m, te[feats].to_numpy(float), (te.y > 0).astype(int), scoring="roc_auc", n_repeats=10, random_state=0)
        else:
            m.fit(tr[feats].to_numpy(float), tr.y.clip(-0.5, 0.5))
            pi = permutation_importance(m, te[feats].to_numpy(float), te.y.clip(-0.5, 0.5), scoring="r2", n_repeats=10, random_state=0)
        imps.append(pd.Series(pi.importances_mean, index=feats, name=y))
    imp = pd.concat(imps, axis=1)
    imp["mean"] = imp.mean(axis=1)
    imp.sort_values("mean", ascending=False).round(4).to_csv(out / "permutation_importance_oos.csv")
    return ict, imp


def main(dry_run: bool):
    out = HERE / "backtest_results_ml" / ("dry_run" if dry_run else "")
    out.mkdir(parents=True, exist_ok=True)
    lock = HERE / "backtest_results_ml" / "FRESH_HOLDOUT_RUN.lock"
    if not dry_run:
        if lock.exists():
            raise SystemExit(f"fresh holdout already evaluated: {lock.read_text()}")
        if not (HERE / "backtest_results_ml" / "dry_run" / "scorecard.json").exists():
            raise SystemExit("run --dry-run first and check it")
    ipos, ctx, cal, meta, panel = load_full()
    cands = pd.read_csv(HERE / "ml" / "candidates.csv", keep_default_na=False, na_values=[""], parse_dates=["signal_date", "entry_date", "exit_date"])
    dev = cands[cands.split != "fresh_holdout"].reset_index(drop=True)
    trials = pd.read_csv(HERE / "trials_ml.csv")
    tr = trials[trials.kind == "trial"].sort_values("worst_year_sharpe", ascending=False)
    pick = tr.iloc[0]
    fam, fset, thr = pick.family, pick.feature_set, pick.threshold_mode
    print(f"selected {pick.run_id} (worst-year OOS Sharpe {pick.worst_year_sharpe})")
    p1 = pd.read_csv(HERE / "trials.csv")
    trial_srs = p1.validation_daily_sr.dropna().tolist() + trials[trials.kind == "trial"].daily_sr.dropna().tolist()

    rd = lambda n, **k: pd.read_csv(FULL / n, keep_default_na=False, na_values=[""], **k)
    pb, spy = rd("placebo_bars.csv", parse_dates=["date"]), rd("spy.csv", parse_dates=["date"])
    result = {"selected": pick.run_id, "family": fam, "feature_set": fset, "threshold_mode": thr,
              "n_trials_total": len(trial_srs), "sim": SIM}

    # development OOS (2020-2026): decisions straight from the recorded walk-forward predictions
    preds = pd.read_csv(HERE / "runs_ml" / pick.run_id / "oos_predictions.csv", keep_default_na=False, na_values=[""],
                        parse_dates=["signal_date", "entry_date", "exit_date"])
    dev_ipos = {s: i for s, i in ipos.items() if i.split != "fresh_holdout"}
    cfg = Config(pick.run_id, signal="lookup", params={"table": decision_table(preds)}, **SIM)
    ent = find_entries(cfg, dev_ipos, SIGNALS["lookup"], ctx)
    ent = ent[ent.entry_date >= pd.Timestamp("2020-01-01")]
    result["dev_oos"], _ = evaluate("dev_oos", cfg, dev_ipos, ent, cal, out, trial_srs)
    ict, imp = feature_diagnostics(dev, fam, fset, out)
    result["top_features_ic"] = ict.sort_values("mean", key=abs, ascending=False).head(10)["mean"].round(3).to_dict()
    result["top_features_importance"] = imp["mean"].sort_values(ascending=False).head(10).round(4).to_dict()

    if not dry_run:
        hold = cands[cands.split == "fresh_holdout"].reset_index(drop=True)
        hp = final_fit(dev, fam, fset, thr, hold)
        hp.to_csv(out / "holdout_predictions.csv", index=False)
        h_ipos = {s: i for s, i in ipos.items() if i.split == "fresh_holdout"}
        hcfg = Config(pick.run_id + "_fresh", signal="lookup", params={"table": decision_table(hp)}, **SIM)
        hent = find_entries(hcfg, h_ipos, SIGNALS["lookup"], ctx)
        lock.write_text(f"fresh holdout evaluated {pd.Timestamp.now().isoformat(timespec='seconds')} for {pick.run_id}")
        result["fresh_holdout"], hres = evaluate("fresh_holdout", hcfg, h_ipos, hent, cal, out, trial_srs, placebo=(pb, spy))
        result["fresh_holdout"]["threshold"] = float(hp.thr.iloc[0])
        result["fresh_holdout"]["candidates_shorted_pct"] = round(float(hp["short_it"].mean() * 100), 1)
        result["fresh_holdout"]["oos_ic"] = round(float(stats.spearmanr(hp.pred, hp.y).statistic), 4)
        result["spot_audit"] = spot_audit(hres["trades"], hcfg)
    (out / "scorecard.json").write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")

    for k in ["dev_oos"] + ([] if dry_run else ["fresh_holdout"]):
        r = result[k]
        m = r["metrics"]
        print(f"\n{k.upper()}: n={m['trades']} cagr={m['cagr']} sharpe={m['sharpe']} dd={m['max_dd']} pf={m['profit_factor']} "
              f"yearly_sharpe={m['yearly_sharpe']} passes_all={r['passes_all']}")
        for name, v in r["scorecard"].items():
            print(f"  {'PASS' if v['pass'] else 'FAIL'}  {name:38s} {v['value']}")
        print("  stress:", json.dumps(r["stress"]))
        if "placebo" in r:
            print("  placebo:", r["placebo"])
    print("\ntop features by |OOS IC|:", result["top_features_ic"])
    print("top permutation importance:", result["top_features_importance"])
    if not dry_run:
        print("spot audit:", [(a["symbol"], a["match"]) for a in result["spot_audit"]])


if __name__ == "__main__":
    main(dry_run="--dry-run" in sys.argv)
