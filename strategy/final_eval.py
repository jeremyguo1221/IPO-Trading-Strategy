"""Final evaluation of the selected strategy. Runs the HOLDOUT exactly once (refuses to rerun it).

Selection (fixed in research_log.md before R8): highest min(train Sharpe, validation Sharpe) across trials.csv.
Outputs -> backtest_results/: scorecard.json, holdout/validation/train trades & equity, stress tests,
shuffled-signal nulls, placebo, spot audit. Everything is also summarised to stdout.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from lib.engine import Config, run
from lib.evaluate import deflated_sharpe, metrics
from lib.placebo import random_established_shorts, shuffled_signal_null
from lib.signals import SIGNALS
from run_iterations import load

HERE = Path(__file__).parent
OUT = HERE / "backtest_results"
OUT.mkdir(exist_ok=True)
LOCK = OUT / "HOLDOUT_RUN.lock"

PASS_BAR = {"cagr": 15.0, "sharpe": 1.0, "max_dd": -25.0, "profit_factor": 1.3, "pct_years_pos": 75.0,
            "trades": 30, "boot_p": 0.05, "dsr": 0.9}


def select(trials: pd.DataFrame) -> str:
    t = trials.assign(robust=trials[["train_sharpe", "validation_sharpe"]].min(axis=1))
    t = t.sort_values("robust", ascending=False)
    t[["run_id", "train_sharpe", "validation_sharpe", "robust"]].head(10).to_csv(OUT / "selection_ranking.csv", index=False)
    return t.run_id.iloc[0]


def scorecard(m: dict, stress60: dict, null_p95: float, dsr: float) -> dict:
    checks = {
        "CAGR >= 15%": (m.get("cagr"), m.get("cagr") is not None and m["cagr"] >= PASS_BAR["cagr"]),
        "Sharpe >= 1.0": (m["sharpe"], m["sharpe"] >= PASS_BAR["sharpe"]),
        "Max drawdown <= 25%": (m["max_dd"], m["max_dd"] >= PASS_BAR["max_dd"]),
        "Profit factor >= 1.3": (m["profit_factor"], m["profit_factor"] >= PASS_BAR["profit_factor"]),
        ">= 75% of years positive": (m["pct_years_pos"], m["pct_years_pos"] >= PASS_BAR["pct_years_pos"]),
        ">= 30 trades": (m["trades"], m["trades"] >= PASS_BAR["trades"]),
        "Bootstrap p < 0.05": (m["boot_p"], m["boot_p"] < PASS_BAR["boot_p"]),
        "Profitable at 60%/yr borrow": (stress60["total_return"], stress60["total_return"] > 0),
        "Sharpe > 95th pct of shuffled null": ((m["sharpe"], round(null_p95, 3)), m["sharpe"] > null_p95),
        "Deflated Sharpe prob > 0.9": (round(dsr, 3), dsr > PASS_BAR["dsr"]),
    }
    return {k: {"value": v, "pass": bool(ok)} for k, (v, ok) in checks.items()}


def spot_audit(trades: pd.DataFrame, cfg: Config, n=5, seed=3) -> list[dict]:
    """Recompute trades straight from the raw exported bars (ipo_bars.csv / iwm / spy), not the engine's arrays."""
    raw = pd.read_csv(HERE / "data" / "ipo_bars.csv", keep_default_na=False, na_values=[""], parse_dates=["date"])
    spy = pd.read_csv(HERE / "data" / "spy.csv", parse_dates=["date"]).set_index("date")
    out = []
    for t in trades.sample(min(n, len(trades)), random_state=seed).itertuples():
        b = raw[raw.symbol == t.symbol].set_index("date").sort_index()
        b = b[b.index >= pd.Timestamp(t.entry_date) - pd.Timedelta(days=400)]
        e_open = b.loc[pd.Timestamp(t.entry_date), "open"]
        x_close = b.loc[pd.Timestamp(t.exit_date), "close"]
        held = b.loc[pd.Timestamp(t.entry_date): pd.Timestamp(t.exit_date)]
        gross = 1 - x_close / e_open
        borrow = cfg.borrow / 252 * (1 + held.close.iloc[:-1].sum() / e_open)
        slip = cfg.slippage * (1 + x_close / e_open)
        hedge = t.beta * (spy.loc[pd.Timestamp(t.exit_date), "close"] / spy.loc[pd.Timestamp(t.entry_date), "open"] - 1) \
            - cfg.spy_slippage * t.beta * (1 + spy.loc[pd.Timestamp(t.exit_date), "close"] / spy.loc[pd.Timestamp(t.entry_date), "open"])
        hand = gross - borrow - slip + (hedge if cfg.hedge else 0)
        out.append({"symbol": t.symbol, "entry_date": str(pd.Timestamp(t.entry_date).date()), "exit_date": str(pd.Timestamp(t.exit_date).date()),
                    "entry_open_raw": round(e_open, 4), "exit_close_raw": round(x_close, 4), "exit_reason": t.exit_reason,
                    "engine_net_ret": round(t.net_ret, 8), "hand_net_ret": round(hand, 8), "match": abs(t.net_ret - hand) < 1e-6})
    return out


def main() -> None:
    if LOCK.exists():
        raise SystemExit(f"Holdout already evaluated ({LOCK.read_text().strip()}). Refusing to rerun it.")
    ipos, ctx, cal = load()
    trials = pd.read_csv(HERE / "trials.csv")
    run_id = select(trials)
    cfg = Config(**json.loads((HERE / "runs" / run_id / "config.json").read_text()))
    sig = SIGNALS[cfg.signal]
    print(f"selected: {run_id}  ({len(trials)} configurations tried)")

    results = {}
    for sp in ["train", "validation", "holdout"]:
        res = run(cfg, ipos, sig, ctx, cal, split=sp)
        m = metrics(res)
        res["trades"].to_csv(OUT / f"trades_{sp}.csv", index=False)
        res["equity"].to_csv(OUT / f"equity_{sp}.csv", header=True)
        res["ledger"].to_csv(OUT / f"ledger_{sp}.csv", index=False)
        stress = {}
        for label, kw in [("borrow_60", {"borrow": 0.60}), ("borrow_100", {"borrow": 1.00}),
                          ("slippage_50bps", {"slippage": 0.005}), ("zero_cost", {"borrow": 0.0, "slippage": 0.0}),
                          ("unhedged", {"hedge": False})]:
            c2 = Config(**{**cfg.as_dict(), **kw, "name": cfg.name + "_" + label})
            ms = metrics(run(c2, ipos, sig, ctx, cal, split=sp))
            stress[label] = {k: ms.get(k) for k in ["trades", "total_return", "cagr", "sharpe", "max_dd", "profit_factor"]}
        null = shuffled_signal_null(cfg, {s: i for s, i in ipos.items() if i.split == sp}, res["entries"], cal, n=200, seed=5)
        null.to_csv(OUT / f"shuffled_null_{sp}.csv", index=False)
        results[sp] = {"metrics": m, "stress": stress, "null_sharpe_p95": float(null.sharpe.quantile(0.95)),
                       "null_sharpe_median": float(null.sharpe.median()),
                       "null_pct_below_strategy": float((null.sharpe < m["sharpe"]).mean() * 100)}
        if sp == "holdout":
            LOCK.write_text(f"holdout evaluated {pd.Timestamp.now().isoformat(timespec='seconds')} for {run_id}")
        print(f"{sp:10s} n={m['trades']} cagr={m['cagr']} sharpe={m['sharpe']} dd={m['max_dd']} pf={m['profit_factor']} "
              f"boot_p={m['boot_p']} null95={results[sp]['null_sharpe_p95']:.2f} yearly={m['yearly']}")

    # Deflated Sharpe: the selected strategy's daily SR vs the spread of daily SRs across ALL configurations tried
    for sp, col in [("validation", "validation_daily_sr"), ("holdout", "validation_daily_sr")]:
        m = results[sp]["metrics"]
        results[sp]["dsr"] = deflated_sharpe(m["daily_sr"], m["n_days"], m["skew"], m["kurt"], trials[col].dropna().tolist())
    results["train"]["dsr"] = deflated_sharpe(results["train"]["metrics"]["daily_sr"], results["train"]["metrics"]["n_days"],
                                             results["train"]["metrics"]["skew"], results["train"]["metrics"]["kurt"],
                                             trials["train_daily_sr"].dropna().tolist())
    for sp in results:
        results[sp]["scorecard"] = scorecard(results[sp]["metrics"], results[sp]["stress"]["borrow_60"],
                                             results[sp]["null_sharpe_p95"], results[sp]["dsr"])
        results[sp]["passes_all"] = all(v["pass"] for v in results[sp]["scorecard"].values())

    # engine sanity on the final trades: random established-stock placebo + hand audit
    all_tr = pd.concat([pd.read_csv(OUT / f"trades_{sp}.csv", parse_dates=["entry_date", "exit_date"]) for sp in results])
    pb = pd.read_csv(HERE / "data" / "placebo_bars.csv", keep_default_na=False, na_values=[""], parse_dates=["date"])
    spy = pd.read_csv(HERE / "data" / "spy.csv", parse_dates=["date"])
    pl = random_established_shorts(cfg, all_tr, pb, spy, pd.DatetimeIndex(spy.date), seed=9, draws=3)
    results["placebo_random_established"] = {"trades": int(len(pl)), "mean_gross_%": round(pl.gross_ret.mean() * 100, 3),
                                             "mean_net_%": round(pl.net_ret.mean() * 100, 3)}
    results["spot_audit"] = spot_audit(all_tr, cfg)
    results["selected_run_id"] = run_id
    results["config"] = cfg.as_dict()
    results["n_configurations_tried"] = int(len(trials))
    (OUT / "scorecard.json").write_text(json.dumps(results, indent=1, default=str), encoding="utf-8")

    for sp in ["train", "validation", "holdout"]:
        print(f"\n{sp.upper()} scorecard (passes all: {results[sp]['passes_all']})")
        for k, v in results[sp]["scorecard"].items():
            print(f"  {'PASS' if v['pass'] else 'FAIL'}  {k:38s} {v['value']}")
        print("  stress:", json.dumps(results[sp]["stress"]))
    print("\nplacebo:", results["placebo_random_established"])
    print("spot audit:", [(a["symbol"], a["match"]) for a in results["spot_audit"]])


if __name__ == "__main__":
    main()
