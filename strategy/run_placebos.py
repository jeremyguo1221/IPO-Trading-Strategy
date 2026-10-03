"""Engine sanity placebo: short random established (non-IPO) stocks on the same dates and holding lengths as
a strategy's trades. With no edge, the mean net return must be about -(mean stock return) - costs, and the
gross short return must not be significantly positive. Results saved to test_logs/.

Usage: .venv/Scripts/python run_placebos.py <run_id>     e.g. R1_base_d3
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from lib.engine import Config
from lib.placebo import random_established_shorts

HERE = Path(__file__).parent
DATA = HERE / "data"


def main(run_id: str) -> None:
    rd = lambda n, **k: pd.read_csv(DATA / n, keep_default_na=False, na_values=[""], **k)
    spy = rd("spy.csv", parse_dates=["date"])
    pb = rd("placebo_bars.csv", parse_dates=["date"])
    cfg = Config(**{k: v for k, v in json.loads((HERE / "runs" / run_id / "config.json").read_text()).items()})
    trades = pd.concat([pd.read_csv(f, parse_dates=["entry_date", "exit_date"]) for f in (HERE / "runs" / run_id).glob("trades_*.csv")])
    tr = random_established_shorts(cfg, trades, pb, spy, pd.DatetimeIndex(spy.date), seed=1, draws=3)
    g = tr.gross_ret
    cost_expect = cfg.slippage * (1 + (1 - g)) + cfg.borrow / 252 * tr.days_held  # approx: slippage both sides + borrow
    t_g = stats.ttest_1samp(g, 0)
    out = {
        "run_id": run_id, "placebo_trades": int(len(tr)), "distinct_stocks": int(tr.symbol.str.split("@").str[0].nunique()),
        "mean_gross_short_ret_%": round(g.mean() * 100, 3), "gross_t_stat": round(t_g.statistic, 2), "gross_p": round(t_g.pvalue, 4),
        "mean_net_ret_%": round(tr.net_ret.mean() * 100, 3),
        "mean_gross_minus_approx_costs_%": round((g - cost_expect).mean() * 100, 3),
        "net_minus_that_%": round((tr.net_ret - (g - cost_expect)).mean() * 100, 4),
        "win_rate_%": round((tr.net_ret > 0).mean() * 100, 1),
        "verdict": "PASS" if (t_g.pvalue > 0.01 or g.mean() < 0) and abs((tr.net_ret - (g - cost_expect)).mean()) < 0.002 else "CHECK",
    }
    txt = json.dumps(out, indent=1)
    print(txt)
    (HERE / "test_logs" / f"placebo_random_established_{run_id}.json").write_text(txt, encoding="utf-8")
    tr.to_csv(HERE / "test_logs" / f"placebo_random_established_{run_id}_trades.csv", index=False)


if __name__ == "__main__":
    main(sys.argv[1])
