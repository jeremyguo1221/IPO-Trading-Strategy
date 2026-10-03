"""Paired tests: is each window around lock-up expiry different from the SAME stock's placebo window 45 days earlier?
difference = event-window abnormal return - placebo-window abnormal return, per IPO; Wilcoxon signed-rank on it.
Output: lockup/contrasts.json"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
d = pd.read_csv(HERE / "per_ipo.csv")
rng = np.random.default_rng(5)


def paired(mask, win):
    x = (d.loc[mask, f"{win}_abn"] - d.loc[mask, f"{win}_placebo_abn"]).dropna()
    boots = [rng.choice(x.values, len(x)).mean() for _ in range(3000)]
    return {"n": int(len(x)), "event_mean_%": round(d.loc[mask, f"{win}_abn"].mean() * 100, 2),
            "placebo_mean_%": round(d.loc[mask, f"{win}_placebo_abn"].mean() * 100, 2),
            "diff_mean_%": round(x.mean() * 100, 2), "diff_median_%": round(x.median() * 100, 2),
            "ci_lo_%": round(np.percentile(boots, 2.5) * 100, 2), "ci_hi_%": round(np.percentile(boots, 97.5) * 100, 2),
            "p": round(float(stats.wilcoxon(x).pvalue), 4)}


allm = pd.Series(True, index=d.index)
ov = pd.qcut(d.overhang.rank(method="first"), 3, labels=["low", "mid", "high"])
ds = pd.qcut(d.deal_size.rank(method="first"), 3, labels=["small", "mid", "large"])
po = pd.cut(d.price_vs_offer_before, [-10, 0, 0.5, 100], labels=["below offer", "0-50% above", "50%+ above"])
out = {"all IPOs": {w: paired(allm, w) for w in ["pre", "event", "post5", "post20", "post40", "around"]}}
pe = pd.cut(d.price_vs_offer_early, [-10, 0, 0.5, 100], labels=["below offer", "0-50% above", "50%+ above"])
for name, grp in [("overhang", ov), ("deal size", ds), ("price vs offer going in", po), ("price vs offer 70 days before (clean)", pe)]:
    out[name] = {str(k): {w: paired(grp == k, w) for w in ["pre", "around", "post40"]} for k in grp.cat.categories}
(HERE / "contrasts.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
for w, r in out["all IPOs"].items():
    print(f"ALL {w:7s} event {r['event_mean_%']:+.2f}% vs placebo {r['placebo_mean_%']:+.2f}% -> diff {r['diff_mean_%']:+.2f}% "
          f"[{r['ci_lo_%']:+.2f},{r['ci_hi_%']:+.2f}] p={r['p']}")
for name in ["overhang", "deal size", "price vs offer going in", "price vs offer 70 days before (clean)"]:
    for k, ws in out[name].items():
        r, q = ws["around"], ws["pre"]
        print(f"{name:24s} {k:12s} around diff {r['diff_mean_%']:+.2f}% p={r['p']} | pre diff {q['diff_mean_%']:+.2f}% p={q['p']} "
              f"| post40 diff {ws['post40']['diff_mean_%']:+.2f}% p={ws['post40']['p']}")
