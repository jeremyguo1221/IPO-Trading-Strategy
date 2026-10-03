"""Run strategy rounds on TRAIN and VALIDATION only (the holdout is never touched here).

Usage:  .venv/Scripts/python run_iterations.py R1 [R2 ...]
Every configuration run is saved to runs/<run_id>/ (config.json, metrics.json, trades_*.csv, equity_*.csv)
and appended to trials.csv. Configurations are defined in rounds.py and are never deleted once run.
"""
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from lib.context import Context
from lib.engine import Config, load_ipos, run
from lib.evaluate import metrics
from lib.signals import SIGNALS
from rounds import ROUNDS

HERE = Path(__file__).parent
DATA, RUNS = HERE / "data", HERE / "runs"
TRIALS = HERE / "trials.csv"
BUDGET = 60


def load():
    rd = lambda n, **k: pd.read_csv(DATA / n, keep_default_na=False, na_values=[""], **k)
    meta = rd("ipo_meta.csv", parse_dates=["first_date", "priced_date"])
    panel = rd("panel.csv", parse_dates=["date"])
    uni = rd("ipo_universe.csv", parse_dates=["priced_date"])
    spy = rd("spy.csv", parse_dates=["date"])
    iwm = rd("iwm.csv", parse_dates=["date"]).set_index("date")
    return load_ipos(meta, panel, {"IWM": iwm}), Context(uni, meta, panel), pd.DatetimeIndex(spy.date)


def run_config(cfg: Config, rnd: str, ipos, ctx, cal, splits=("train", "validation")) -> dict:
    run_id = f"{rnd}_{cfg.name}"
    out_dir = RUNS / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(cfg.as_dict(), indent=1, default=str), encoding="utf-8")
    row = {"run_id": run_id, "round": rnd, "name": cfg.name, "hedge": cfg.hedge, "stop": cfg.stop, "target": cfg.target,
           "hold": cfg.hold, "size": cfg.size, "vol_target": cfg.vol_target, "params": json.dumps(cfg.params),
           "timestamp": datetime.now().isoformat(timespec="seconds")}
    all_m = {}
    for sp in splits:
        res = run(cfg, ipos, SIGNALS[cfg.signal], ctx, cal, split=sp)
        m = metrics(res)
        all_m[sp] = m
        if len(res["trades"]):
            res["trades"].to_csv(out_dir / f"trades_{sp}.csv", index=False)
            res["equity"].to_csv(out_dir / f"equity_{sp}.csv", header=True)
        for k in ["trades", "cagr", "sharpe", "max_dd", "profit_factor", "win_rate", "mean_trade", "worst_trade",
                  "pct_years_pos", "boot_p", "avg_gross", "pct_stopped", "daily_sr", "n_days", "skew", "kurt"]:
            row[f"{sp}_{k}"] = m.get(k)
        row[f"{sp}_yearly"] = json.dumps(m.get("yearly", {}))
    (out_dir / "metrics.json").write_text(json.dumps(all_m, indent=1, default=str), encoding="utf-8")
    return row


def main(rounds: list[str]) -> None:
    ipos, ctx, cal = load()
    done = pd.read_csv(TRIALS) if TRIALS.exists() else pd.DataFrame()
    for rnd in rounds:
        for cfg in ROUNDS[rnd]:
            run_id = f"{rnd}_{cfg.name}"
            if len(done) and run_id in set(done.run_id):
                print(f"skip (already run) {run_id}")
                continue
            n_distinct = (done.name.nunique() if len(done) else 0) + 1
            if n_distinct > BUDGET:
                raise SystemExit(f"trial budget of {BUDGET} configurations reached")
            row = run_config(cfg, rnd, ipos, ctx, cal)
            done = pd.concat([done, pd.DataFrame([row])], ignore_index=True)
            done.to_csv(TRIALS, index=False)
            print(f"{run_id:45s} train: n={row['train_trades']} cagr={row['train_cagr']} sh={row['train_sharpe']} "
                  f"dd={row['train_max_dd']} pf={row['train_profit_factor']} | val: n={row['validation_trades']} "
                  f"cagr={row['validation_cagr']} sh={row['validation_sharpe']} dd={row['validation_max_dd']} "
                  f"pf={row['validation_profit_factor']}")


if __name__ == "__main__":
    main(sys.argv[1:])
