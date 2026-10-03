"""Train-only diagnostic: which pre-entry features relate to the short's net return?
Uses unconstrained per-IPO trades (every eligible IPO, fixed entry day) so capacity doesn't hide anything.
Output: diagnostics/feature_scan_<entry_day>.txt"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.engine import Config, run
from lib.signals import rules
from run_iterations import load

ipos, ctx, cal = load()
meta = pd.read_csv(Path(__file__).resolve().parents[1] / "data" / "ipo_meta.csv", keep_default_na=False, na_values=[""]).set_index("symbol")
for ed in (3, 6, 11):
    cfg = Config("diag", signal="rules", params={"entry_day": ed}, size=0.001, max_positions=10**6, gross_cap=10**6)
    res = run(cfg, ipos, rules, ctx, cal, split="train")
    tr = res["trades"].copy()
    f = []
    for t in tr.itertuples():
        i = ipos[t.symbol]; s = t.signal_day
        c = i.c[:s]; d = pd.Timestamp(i.dates[s - 1])
        r = np.diff(c) / c[:-1]
        f.append({"pop": c[0] / i.offer - 1, "ext": c[-1] / i.offer - 1, "run": c[-1] / c[0] - 1,
                  "open_to_close_d1": c[0] / i.o[0] - 1, "vol": np.std(r) if len(r) > 1 else np.nan,
                  "log_dollar_vol": np.log(np.median(c * i.v[:s])), "log_deal": np.log(meta.loc[t.symbol, "deal_size"]),
                  "offer": i.offer, "hot_count": ctx.ipo_count(d), "hot_pop": ctx.median_pop(d),
                  "d1_range": (i.h[0] - i.l[0]) / i.o[0], "below_d1_low": float(c[-1] < i.l[0])})
    F = pd.DataFrame(f, index=tr.index)
    lines = [f"entry day {ed}: {len(tr)} train trades, mean net {tr.net_ret.mean()*100:.2f}%, median {tr.net_ret.median()*100:.2f}%"]
    for col in F:
        x = F[col].astype(float); ok = x.notna()
        if ok.sum() < 30 or x[ok].nunique() < 3:
            continue
        rho, p = stats.spearmanr(x[ok], tr.net_ret[ok])
        q = pd.qcut(x[ok].rank(method="first"), 3, labels=["low", "mid", "high"])
        by = tr.net_ret[ok].groupby(q, observed=True).mean() * 100
        lines.append(f"  {col:16s} rho={rho:+.3f} p={p:.3f} | mean net by tercile low/mid/high: " + " / ".join(f"{v:+.2f}%" for v in by))
    txt = "\n".join(lines)
    print(txt + "\n")
    (Path(__file__).parent / f"feature_scan_d{ed}.txt").write_text(txt, encoding="utf-8")
