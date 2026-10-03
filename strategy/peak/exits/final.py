"""Phase 4 final evaluation of the searched exit vs the 21-day hold.

  --dry-run : search eras only (2015-18, 2019-21, 2021-24): recompute chosen vs baseline, hindsight ceiling. No validation.
  (no flag) : the above + the 2024-26 validation set ONCE (lock file), cost stress, Deflated Sharpe, hand audit.
Chosen exit = top row of trials_exit.csv (the search's objective ranking). Baseline = 21-day hold.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from lib.engine import Config, simulate
from lib.evaluate import deflated_sharpe, metrics
from ml.dataset import FULL
from peak.exits.search import ERA_LABEL, ERAS, GRID, default, key, load_bundle, to_cfg

HERE = Path(__file__).resolve().parent
LOCK = HERE / "VALIDATION_RUN.lock"


def row_to_x(r) -> dict:
    return {k: (None if isinstance(r[k], float) and np.isnan(r[k]) else (int(r[k]) if k == "hold" else r[k])) for k in GRID}


def run_era(x, ipos, entries, cal, **over):
    cfg = to_cfg(x, **over)
    res = simulate(cfg, {s: ipos[s] for s in entries.symbol}, entries, cal)
    return cfg, res, metrics(res)


def ceiling(ipos, entries, trades) -> dict:
    """Best possible gross short return within the 21 days after entry (cover at the lowest low) vs achieved gross."""
    best = []
    for r in entries.itertuples():
        i = ipos[r.symbol]
        e = r.entry_day - 1
        best.append(1 - i.l[e: e + 21].min() / i.o[e])
    return {"ceiling_mean_gross_%": round(float(np.mean(best)) * 100, 2),
            "achieved_mean_gross_%": round(float(trades.gross_ret.mean()) * 100, 2),
            "capture_ratio_%": round(float(trades.gross_ret.mean() / np.mean(best)) * 100, 1)}


def hand_audit(trades, cfg, n=5):
    raw = pd.read_csv(FULL / "ipo_bars.csv", keep_default_na=False, na_values=[""], parse_dates=["date"])
    spy = pd.read_csv(FULL / "spy.csv", parse_dates=["date"]).set_index("date")
    out = []
    for t in trades.sample(min(n, len(trades)), random_state=4).itertuples():
        b = raw[raw.symbol == t.symbol].set_index("date").sort_index()
        ed, xd = pd.Timestamp(t.entry_date), pd.Timestamp(t.exit_date)
        e, x = b.loc[ed, "open"], b.loc[xd, "close"]
        held = b.loc[ed:xd]
        so, sc = spy.loc[ed, "open"], spy.loc[xd, "close"]
        hand = (1 - x / e) - cfg.slippage * (1 + x / e) - cfg.borrow / 252 * (1 + held.close.iloc[:-1].sum() / e) \
            + t.beta * (sc / so - 1) - cfg.spy_slippage * t.beta * (1 + sc / so)
        out.append({"symbol": t.symbol, "entry": str(ed.date()), "exit": str(xd.date()), "days": int(t.days_held),
                    "engine": round(t.net_ret, 8), "hand": round(hand, 8), "match": abs(t.net_ret - hand) < 1e-6})
    return out


def main(dry: bool):
    if not dry and LOCK.exists():
        raise SystemExit(f"validation already evaluated: {LOCK.read_text()}")
    if not dry and not (HERE / "final_dry_run.json").exists():
        raise SystemExit("run --dry-run first")
    ipos, entries, cal = load_bundle()
    t = pd.read_csv(HERE / "trials_exit.csv")
    chosen = row_to_x(t.iloc[0])
    base = default()
    out = {"chosen": key(chosen), "baseline": key(base), "n_configs": int(len(t)), "eras": {}}
    for era in ERAS + ([] if dry else ["validation"]):
        if era == "validation":
            LOCK.write_text(f"validation evaluated {pd.Timestamp.now().isoformat(timespec='seconds')} for {key(chosen)}")
        er = {}
        for name, x in [("chosen", chosen), ("baseline_21d", base)]:
            cfg, res, m = run_era(x, ipos, entries[era], cal)
            er[name] = {k: m.get(k) for k in ["trades", "cagr", "sharpe", "max_dd", "profit_factor", "mean_trade", "win_rate", "boot_p", "total_return"]}
            er[name]["avg_days"] = round(float(res["trades"].days_held.mean()), 2)
            er[name] |= ceiling(ipos, entries[era], res["trades"])
            if era == "validation":
                res["trades"].to_csv(HERE / f"validation_trades_{name}.csv", index=False)
                res["equity"].to_csv(HERE / f"validation_equity_{name}.csv", header=True)
                if name == "chosen":
                    er[name]["stress"] = {}
                    for lab, kw in [("borrow_60", {"borrow": 0.6}), ("borrow_100", {"borrow": 1.0}), ("slippage_50bps", {"slippage": 0.005})]:
                        _, _, ms = run_era(x, ipos, entries[era], cal, **kw)
                        er[name]["stress"][lab] = {k: ms.get(k) for k in ["total_return", "cagr", "sharpe", "max_dd"]}
                    srs = [np.nanmean([r[f"{e}_daily_sr"] for e in ERAS]) for _, r in t.iterrows()]
                    er[name]["dsr"] = round(deflated_sharpe(m["daily_sr"], m["n_days"], m["skew"], m["kurt"], srs), 4)
                    er[name]["hand_audit"] = hand_audit(res["trades"], cfg)
        out["eras"][ERA_LABEL[era]] = er
        c, b = er["chosen"], er["baseline_21d"]
        print(f"{ERA_LABEL[era]}: chosen  n={c['trades']} cagr={c['cagr']} sh={c['sharpe']} dd={c['max_dd']} mean_trade={c['mean_trade']} "
              f"days={c['avg_days']} capture={c['capture_ratio_%']}% of ceiling {c['ceiling_mean_gross_%']}%")
        print(f"{'':8s} 21-day  n={b['trades']} cagr={b['cagr']} sh={b['sharpe']} dd={b['max_dd']} mean_trade={b['mean_trade']} "
              f"days={b['avg_days']} capture={b['capture_ratio_%']}%")
    if not dry:
        v = out["eras"]["2024-26"]
        out["verdict_better_than_21d"] = bool(v["chosen"]["sharpe"] > v["baseline_21d"]["sharpe"])
        print("\nstress:", json.dumps(v["chosen"]["stress"]))
        print("DSR:", v["chosen"]["dsr"], "| better than 21-day on validation:", out["verdict_better_than_21d"])
        print("hand audit:", [(a["symbol"], a["match"]) for a in v["chosen"]["hand_audit"]])
    (HERE / ("final_dry_run.json" if dry else "final.json")).write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    main("--dry-run" in sys.argv)
