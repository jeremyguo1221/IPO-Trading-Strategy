"""Phase 4: self-iterating exit search for the peak-timed IPO short (entry fixed: peak family A, m=0.4, p=0.05, k=10).

Search data (never the 2024-26 validation set): IPO eras 2015-18 ('fresh_holdout'), 2019-21 ('holdout'), 2021-24 ('train').
Objective: worst-era portfolio Sharpe (tie-break: mean across eras).
Loop: round 0 = seed set; each later round evaluates every unseen one-step neighbour of the top 3 so far.
Stops when the best objective improves < 0.05 for 2 consecutive rounds, the neighbourhood is exhausted, or 120
configurations have been evaluated. Entries are computed once (with a 21-day hold) so every exit sees the same trades.

Usage: .venv/Scripts/python -m peak.exits.search
Outputs: peak/exits/trials_exit.csv, peak/exits/rounds.csv, peak/exits/search_log.txt, peak/exits/entries_<era>.csv
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from lib.engine import Config, find_entries, simulate
from lib.evaluate import metrics
from lib.signals import peak
from ml.dataset import load_full

HERE = Path(__file__).resolve().parent
ENTRY = {"family": "A", "m": 0.4, "p": 0.05, "k": 10}
ERAS = ["fresh_holdout", "holdout", "train"]          # 2015-18, 2019-21, 2021-24
ERA_LABEL = {"fresh_holdout": "2015-18", "holdout": "2019-21", "train": "2021-24", "validation": "2024-26"}
GRID = {"hold": [3, 5, 7, 10, 15, 21], "target": [None, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40],
        "stop": [None, 0.10, 0.15, 0.25, 0.40], "trail": [None, 0.05, 0.10, 0.15, 0.20, 0.30],
        "sig": [None, "ma5", "prev_high"], "thesis": [False, True]}
BUDGET, PLATEAU, PATIENCE, TOP = 120, 0.05, 2, 3
BASE = dict(signal="peak", params=ENTRY, hedge=True, range_target=0.186, last_signal_day=41)


def key(x: dict) -> str:
    return "|".join(f"{k}={x[k]}" for k in GRID)


def to_cfg(x: dict, name: str = "exit", **over) -> Config:
    return Config(name, hold=x["hold"], target=x["target"], stop=x["stop"], trail=x["trail"], exit_signal=x["sig"],
                  thesis_stop=x["thesis"], **{**BASE, **over})


def default() -> dict:
    return {"hold": 21, "target": None, "stop": None, "trail": None, "sig": None, "thesis": False}


def seed() -> list[dict]:
    d = default
    s = [dict(d(), hold=h) for h in (3, 5, 10, 15, 21)]
    s += [dict(d(), target=t) for t in (0.10, 0.20, 0.30)]
    s += [dict(d(), stop=v) for v in (0.15, 0.25)]
    s += [dict(d(), thesis=True)]
    s += [dict(d(), trail=v) for v in (0.10, 0.15, 0.20)]
    s += [dict(d(), sig=v) for v in ("ma5", "prev_high")]
    s += [dict(d(), target=0.20, thesis=True), dict(d(), target=0.20, trail=0.15), dict(d(), thesis=True, sig="ma5"),
          dict(d(), hold=10, target=0.20), dict(d(), trail=0.15, thesis=True), dict(d(), target=0.30, stop=0.25)]
    return s


def neighbours(x: dict) -> list[dict]:
    """Every one-step move: a numeric parameter one notch up/down (None is the first notch, so this also adds/removes
    a component), set any active component to None, switch the exit signal, toggle the thesis stop."""
    out = []
    for p in ("hold", "target", "stop", "trail"):
        g = GRID[p]
        i = g.index(x[p])
        for j in (i - 1, i + 1):
            if 0 <= j < len(g):
                out.append({**x, p: g[j]})
        if p != "hold" and x[p] is not None:
            out.append({**x, p: None})
    for s in GRID["sig"]:
        if s != x["sig"]:
            out.append({**x, "sig": s})
    out.append({**x, "thesis": not x["thesis"]})
    seen, uniq = set(), []
    for o in out:
        k = key(o)
        if k not in seen and k != key(x):
            seen.add(k)
            uniq.append(o)
    return uniq


def evaluate(x: dict, ctx_bundle) -> dict:
    ipos, entries, cal = ctx_bundle
    row = {"id": key(x), **x}
    shs = []
    for era in ERAS:
        sub = {s: ipos[s] for s in entries[era].symbol}
        res = simulate(to_cfg(x), sub, entries[era], cal)
        m = metrics(res)
        tr = res["trades"]
        row |= {f"{era}_{k}": m.get(k) for k in ["trades", "sharpe", "cagr", "max_dd", "profit_factor", "mean_trade", "win_rate", "daily_sr"]}
        row[f"{era}_avg_days"] = round(float(tr.days_held.mean()), 2) if len(tr) else None
        row[f"{era}_reasons"] = json.dumps(tr.exit_reason.value_counts().to_dict()) if len(tr) else "{}"
        shs.append(m.get("sharpe", 0.0) or 0.0)
    row["objective"] = round(min(shs), 4)
    row["mean_sharpe"] = round(float(np.mean(shs)), 4)
    return row


def load_bundle():
    ipos, ctx, cal, meta, panel = load_full()
    base_cfg = to_cfg(default())
    entries = {}
    for era in ERAS + ["validation"]:
        sub = {s: i for s, i in ipos.items() if i.split == era}
        entries[era] = find_entries(base_cfg, sub, peak, ctx)
        entries[era].to_csv(HERE / f"entries_{era}.csv", index=False)
    return ipos, entries, cal


def search(bundle, budget=BUDGET, log=print) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    done: dict[str, dict] = {}
    rounds = []
    frontier = seed()
    best_hist, stalls, rnd, reason = [], 0, 0, ""
    while True:
        batch = [x for x in frontier if key(x) not in done][: max(0, budget - len(done))]
        for x in batch:
            r = evaluate(x, bundle)
            r["round"] = rnd
            done[r["id"]] = r
        t = pd.DataFrame(done.values()).sort_values(["objective", "mean_sharpe", "id"], ascending=[False, False, True])
        best = t.iloc[0]
        improve = best.objective - best_hist[-1] if best_hist else np.nan
        best_hist.append(best.objective)
        rounds.append({"round": rnd, "evaluated_this_round": len(batch), "evaluated_total": len(done),
                       "best_id": best.id, "best_objective": best.objective, "best_mean_sharpe": best.mean_sharpe,
                       "improvement": None if np.isnan(improve) else round(float(improve), 4)})
        log(f"round {rnd}: evaluated {len(batch)} (total {len(done)}) | best {best.id} objective={best.objective:.3f} "
            f"mean={best.mean_sharpe:.3f} | improvement {'n/a' if np.isnan(improve) else f'{improve:+.3f}'}")
        if len(done) >= budget:
            reason = f"budget of {budget} configurations reached"
            break
        if not np.isnan(improve):
            stalls = stalls + 1 if improve < PLATEAU else 0
            if stalls >= PATIENCE:
                reason = f"plateau: improvement < {PLATEAU} for {PATIENCE} consecutive rounds"
                break
        frontier = []
        for _, r in t.head(TOP).iterrows():
            frontier += neighbours({k: (None if (isinstance(r[k], float) and np.isnan(r[k])) else r[k]) for k in GRID})
        frontier = [x for x in frontier if key(x) not in done]
        if not frontier:
            reason = "neighbourhood exhausted"
            break
        rnd += 1
    return t, pd.DataFrame(rounds), reason


def main():
    lines = []
    def log(s):
        print(s); lines.append(s)
    bundle = load_bundle()
    log("entries per era: " + ", ".join(f"{ERA_LABEL[e]}={len(bundle[1][e])}" for e in ERAS) + " (validation entries saved, not used)")
    t, rounds, reason = search(bundle, log=log)
    log(f"stopped: {reason}")
    t.to_csv(HERE / "trials_exit.csv", index=False)
    rounds.to_csv(HERE / "rounds.csv", index=False)
    cols = ["id", "round", "objective", "mean_sharpe"] + [f"{e}_{m}" for e in ERAS for m in ("sharpe", "cagr", "max_dd", "avg_days")]
    log("\ntop 10:\n" + t[cols].head(10).to_string(index=False))
    base = t[t.id == key(default())].iloc[0]
    log(f"\n21-day hold baseline: objective={base.objective} mean={base.mean_sharpe} "
        + " ".join(f"{ERA_LABEL[e]} sh={base[f'{e}_sharpe']}" for e in ERAS))
    (HERE / "search_log.txt").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
