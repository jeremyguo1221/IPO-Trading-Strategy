"""Phase 3: find an entry rule that shorts as close as possible to the IPO's 63-day peak.

Metric (fixed in the plan before any rule was run):
  efficiency    = entry open / highest high of days 1..63 (1.00 = shorted at the exact top)
  paired gain   = rule efficiency - that SAME IPO's random-day efficiency (mean over entry days 3..42)
  selection     = highest mean paired gain on TRAIN, among rules firing on >= 40% of eligible IPOs
  "works" only if validation paired gain > 0 with bootstrap 95% CI above 0.
Eligible IPOs: offer >= $5, >= 63 bars, and tradable on some signal day 2..41 (same filter as the engine).

Usage: .venv/Scripts/python -m peak.peak_entry
Outputs: peak/trials_peak.csv, peak/runs_peak/<rule>/entries_<split>.csv, peak/top3_pnl.json
"""
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from lib.engine import Config, find_entries, simulate
from lib.evaluate import metrics
from lib.signals import peak
from ml.dataset import label, load_full

HERE = Path(__file__).resolve().parent
W = 63                   # peak window (trading days)
ENTRY_LAST = 42          # last allowed entry day (signal day <= 41)
RNG = np.random.default_rng(3)


def rules_grid() -> dict[str, dict]:
    g = {}
    for m, pb, k in itertools.product([0.2, 0.4, 0.6], [0.05, 0.10, 0.15], [5, 10]):
        g[f"A_m{int(m*100)}_p{int(pb*100)}_k{k}"] = {"family": "A", "m": m, "p": pb, "k": k}
    for vm in [None, 1.5, 2.0]:
        g[f"B_v{vm or 0}"] = {"family": "B", "vmult": vm}
    for r in [0.2, 0.4]:
        g[f"C_r{int(r*100)}"] = {"family": "C", "r": r}
    for n in [5, 10]:
        g[f"D_n{n}"] = {"family": "D", "n": n}
    return g


def cfg_for(params: dict) -> Config:
    return Config("peak", signal="peak", params=params, hold=21, hedge=True, range_target=0.186, last_signal_day=ENTRY_LAST - 1)


def evaluate(ipos: dict, entries: pd.DataFrame, eligible: set) -> tuple[dict, pd.DataFrame]:
    rows = []
    for r in entries.itertuples():
        if r.symbol not in eligible:
            continue
        i = ipos[r.symbol]
        peak_k = int(np.argmax(i.h[:W]))
        top = i.h[peak_k]
        e = r.entry_day - 1
        eff = i.o[e] / top
        rnd = float(np.mean(i.o[2:ENTRY_LAST] / top))
        net, _, _ = label(i, r.signal_day)
        rows.append({"symbol": r.symbol, "signal_day": r.signal_day, "entry_day": r.entry_day, "peak_day": peak_k + 1,
                     "eff": eff, "eff_random": rnd, "gain": eff - rnd, "fwd21_net": net})
    d = pd.DataFrame(rows)
    if d.empty:
        return {"fired": 0, "fire_rate": 0.0}, d
    boots = [RNG.choice(d.gain.to_numpy(), len(d)).mean() for _ in range(2000)]
    return {"fired": len(d), "fire_rate": round(len(d) / len(eligible) * 100, 1),
            "gain_mean": round(d.gain.mean() * 100, 2), "gain_ci_lo": round(np.percentile(boots, 2.5) * 100, 2),
            "gain_ci_hi": round(np.percentile(boots, 97.5) * 100, 2), "eff_median": round(d.eff.median() * 100, 1),
            "eff_mean": round(d.eff.mean() * 100, 1), "pct_within_10pct": round((d.eff >= 0.9).mean() * 100, 1),
            "days_from_peak_median": float((d.entry_day - d.peak_day).median()),
            "entry_day_median": float(d.entry_day.median()), "fwd21_net_mean": round(d.fwd21_net.mean() * 100, 2),
            "fwd21_net_median": round(d.fwd21_net.median() * 100, 2)}, d


def eligible_set(ipos, ctx, split):
    sub = {s: i for s, i in ipos.items() if i.split == split and i.n >= W}
    ent = find_entries(cfg_for({"family": "always"}), sub, peak, ctx)
    return sub, set(ent.symbol)


def main():
    ipos, ctx, cal, meta, panel = load_full()
    out_runs = HERE / "runs_peak"
    grid = rules_grid()
    rows = []
    elig = {sp: eligible_set(ipos, ctx, sp) for sp in ["train", "validation"]}
    print("eligible IPOs:", {sp: len(e[1]) for sp, e in elig.items()})
    for name, params in grid.items():
        row = {"rule": name, **{f"param_{k}": v for k, v in params.items()}}
        for sp in ["train", "validation"]:
            sub, el = elig[sp]
            ent = find_entries(cfg_for(params), {s: sub[s] for s in el}, peak, ctx)
            m, d = evaluate(ipos, ent, el)
            (out_runs / name).mkdir(parents=True, exist_ok=True)
            d.to_csv(out_runs / name / f"entries_{sp}.csv", index=False)
            row |= {f"{sp}_{k}": v for k, v in m.items()}
        rows.append(row)
        print(f"{name:18s} train fire={row.get('train_fire_rate')}% gain={row.get('train_gain_mean')} eff={row.get('train_eff_median')} "
              f"| val fire={row.get('validation_fire_rate')}% gain={row.get('validation_gain_mean')} "
              f"[{row.get('validation_gain_ci_lo')},{row.get('validation_gain_ci_hi')}] eff={row.get('validation_eff_median')}")
    t = pd.DataFrame(rows)
    # baselines (not trials): fixed days, computed the same way
    base = {}
    for sp in ["train", "validation"]:
        sub, el = elig[sp]
        for d in (3, 6, 11, 22):
            effs = [(sub[s].o[d - 1] / sub[s].h[:W].max()) - float(np.mean(sub[s].o[2:ENTRY_LAST] / sub[s].h[:W].max())) for s in el]
            base[f"{sp}_fixed_d{d}_gain"] = round(float(np.mean(effs)) * 100, 2)
        best = [sub[s].o[2:ENTRY_LAST].max() / sub[s].h[:W].max() for s in el]
        base[f"{sp}_oracle_eff_median"] = round(float(np.median(best)) * 100, 1)
    ok = t[t.train_fire_rate >= 40].sort_values(["train_gain_mean", "train_fire_rate"], ascending=False)
    t["selected"] = t.rule == ok.rule.iloc[0]
    t["train_rank"] = t.rule.map({r: i + 1 for i, r in enumerate(ok.rule)})
    t.to_csv(HERE / "trials_peak.csv", index=False)
    (HERE / "baselines.json").write_text(json.dumps(base, indent=1), encoding="utf-8")
    print("\nbaselines:", base)
    print("\ntop 5 on TRAIN (fire >= 40%):")
    cols = ["rule", "train_fire_rate", "train_gain_mean", "train_eff_median", "train_pct_within_10pct", "train_days_from_peak_median",
            "validation_fire_rate", "validation_gain_mean", "validation_gain_ci_lo", "validation_gain_ci_hi", "validation_eff_median",
            "train_fwd21_net_mean", "validation_fwd21_net_mean"]
    print(ok[cols].head(5).to_string(index=False))

    # P&L context for the top 3 train rules: 21-day hold, SPY hedge, base costs, through the engine
    pnl = {}
    for name in ok.rule.head(3):
        pnl[name] = {}
        for sp, note in [("train", ""), ("validation", ""), ("holdout", "previously used (phase 1)"), ("fresh_holdout", "previously used (phase 2)")]:
            sub = {s: i for s, i in ipos.items() if i.split == sp}
            cfg = cfg_for(grid[name])
            ent = find_entries(cfg, sub, peak, ctx)
            m = metrics(simulate(cfg, sub, ent, cal))
            pnl[name][sp] = {"note": note, **{k: m.get(k) for k in ["trades", "cagr", "sharpe", "max_dd", "profit_factor", "mean_trade", "win_rate"]}}
        print(f"\nP&L {name}: " + " | ".join(f"{sp}: n={v['trades']} cagr={v['cagr']} sh={v['sharpe']} dd={v['max_dd']}" for sp, v in pnl[name].items()))
    (HERE / "top3_pnl.json").write_text(json.dumps(pnl, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
