"""Stats for the best strategy found (reporting only - nothing is selected or tuned here).

Strategy: phase-3 entry (peak family A: ran >= 40% above offer, first close >= 5% below a high set within 10 days,
short next open), 21-day hold (phase 4/5: no exit beat it on 2024-26), SPY hedge, range-scaled sizing, base costs.

Runs: one continuous portfolio over ALL clean IPOs 2015-2026, plus each era separately, plus cost stress.
Output: peak/best_strategy.json, peak/best_equity_full.csv, peak/best_trades_full.csv
"""
import json
from pathlib import Path

import pandas as pd

from lib.engine import Config, find_entries, simulate
from lib.evaluate import metrics
from lib.signals import peak
from ml.dataset import load_full

HERE = Path(__file__).resolve().parent
ENTRY = {"family": "A", "m": 0.4, "p": 0.05, "k": 10}
CFG = dict(signal="peak", params=ENTRY, hold=21, hedge=True, range_target=0.186, last_signal_day=41)
ERAS = [("fresh_holdout", "2015–18"), ("holdout", "2019–21"), ("train", "2021–24"), ("validation", "2024–26")]
ROLE = {"fresh_holdout": "not used to pick it", "holdout": "not used to pick it",
        "train": "entry rule chosen here", "validation": "entry checked, 21‑day exit confirmed here"}
KEYS = ["trades", "total_return", "cagr", "sharpe", "sortino", "max_dd", "profit_factor", "win_rate", "mean_trade",
        "median_trade", "best_trade", "worst_trade", "avg_gross", "boot_p", "start", "end"]


def main():
    ipos, ctx, cal, meta, panel = load_full()
    cfg = Config("best", **CFG)
    ent = find_entries(cfg, ipos, peak, ctx)
    res = simulate(cfg, ipos, ent, cal)
    m = metrics(res)
    tr = res["trades"]
    res["equity"].to_csv(HERE / "best_equity_full.csv", header=True)
    tr.to_csv(HERE / "best_trades_full.csv", index=False)
    out = {"rules": CFG, "full": {k: m.get(k) for k in KEYS}, "yearly": m["yearly"],
           "avg_days_held": round(float(tr.days_held.mean()), 1),
           "avg_positions": m.get("avg_positions"), "eras": [], "stress": {}}
    out["full"]["trades_per_year"] = round(len(tr) / ((res["equity"].index[-1] - res["equity"].index[0]).days / 365.25), 1)
    for sp, lab in ERAS:
        sub = {s: i for s, i in ipos.items() if i.split == sp}
        e = find_entries(cfg, sub, peak, ctx)
        me = metrics(simulate(cfg, sub, e, cal))
        out["eras"].append({"era": lab, "role": ROLE[sp], **{k: me.get(k) for k in ["trades", "cagr", "sharpe", "max_dd", "profit_factor", "win_rate", "mean_trade"]}})
    for lab, kw in [("Borrow 60%/yr", {"borrow": 0.6}), ("Borrow 100%/yr", {"borrow": 1.0}), ("Slippage 50 bps", {"slippage": 0.005}),
                    ("No costs", {"borrow": 0.0, "slippage": 0.0}), ("Unhedged", {"hedge": False})]:
        c2 = Config("best_" + lab, **{**CFG, **kw})
        ms = metrics(simulate(c2, ipos, find_entries(c2, ipos, peak, ctx), cal))
        out["stress"][lab] = {k: ms.get(k) for k in ["cagr", "sharpe", "max_dd", "total_return"]}
    (HERE / "best_strategy.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    f = out["full"]
    print(f"FULL 2015-2026: trades={f['trades']} total={f['total_return']}% cagr={f['cagr']}% sharpe={f['sharpe']} maxdd={f['max_dd']}% "
          f"pf={f['profit_factor']} win={f['win_rate']}% mean_trade={f['mean_trade']}% days={out['avg_days_held']}")
    print("yearly:", out["yearly"])
    for e in out["eras"]:
        print(e)
    print("stress:", json.dumps(out["stress"]))


if __name__ == "__main__":
    main()
