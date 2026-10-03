"""Signal rules. Each receives a View (the IPO's bars for days 1..t only), a params dict, and the market
context, and returns False/None (no signal) or True / a float strength (signal fires at day t's close;
the engine enters at day t+1's open).

`rules` is the single configurable signal used by every strategy round. Parameters (all optional):
  entry_day        int   fire only when t == entry_day - 1 (fixed-day entry)
  from_day         int   earliest trading day t a trigger may fire on
  pop_min/pop_max  float day-1 close vs offer price, e.g. 0.15 = +15%
  ext_min/ext_max  float day-t close vs offer price
  run_min/run_max  float day-t close vs day-1 close
  breakdown        str   'd1_low'   : close_t < day-1 low
                          'low5'     : close_t < min(low of days t-5..t-1)
                          'offer'    : close_t < offer price
                          'ma5'      : close_t < mean(close of days t-4..t) and close_t < close_{t-1}
                          [list]     : any of the listed breakdowns
  hot_count_min    int   >= this many IPOs priced in the 60 calendar days before day t
  hot_pop_min      float median day-1 pop of IPOs that first traded in the prior 60 days >= this
  hot_mode         str   'and' (default: every given hot test must pass) | 'or' (any one passes)
  d1_range_min     float day-1 (high - low) / open >= this
  vol_min          float std of daily close-to-close returns over days 1..t >= this (needs t >= 3)
  strength         str   'pop' | 'ext' | 'range' | None: ordering key when capacity is limited
"""
from __future__ import annotations

import numpy as np


def _pop(v) -> float:
    return v.c[0] / v.offer - 1


def rules(v, p: dict, ctx) -> bool | float:
    t = v.t
    if "entry_day" in p and t != p["entry_day"] - 1:
        return False
    if t < p.get("from_day", 0):
        return False
    pop = _pop(v)
    if "pop_min" in p and pop < p["pop_min"]:
        return False
    if "pop_max" in p and pop > p["pop_max"]:
        return False
    ext = v.c[-1] / v.offer - 1
    if "ext_min" in p and ext < p["ext_min"]:
        return False
    if "ext_max" in p and ext > p["ext_max"]:
        return False
    run = v.c[-1] / v.c[0] - 1
    if "run_min" in p and run < p["run_min"]:
        return False
    if "run_max" in p and run > p["run_max"]:
        return False
    bd = p.get("breakdown")
    if bd is not None:
        tests = {"d1_low": lambda: t >= 2 and v.c[-1] < v.l[0],
                 "low5": lambda: t >= 6 and v.c[-1] < v.l[-6:-1].min(),
                 "offer": lambda: v.c[-1] < v.offer,
                 "ma5": lambda: t >= 5 and v.c[-1] < v.c[-5:].mean() and v.c[-1] < v.c[-2]}
        # a list means ANY of the listed breakdowns qualifies; breakdown_all=True requires ALL of them
        agg = all if p.get("breakdown_all") else any
        if not agg(tests[b]() for b in (bd if isinstance(bd, list) else [bd])):
            return False
    if "d1_range_min" in p and (v.h[0] - v.l[0]) / v.o[0] < p["d1_range_min"]:
        return False
    if "vol_min" in p:
        if t < 3:
            return False
        r = np.diff(v.c) / v.c[:-1]
        if np.std(r, ddof=1) < p["vol_min"]:
            return False
    date = v.dates[-1]
    hot_checks = []
    if "hot_count_min" in p:
        hot_checks.append(ctx.ipo_count(date) >= p["hot_count_min"])
    if "hot_pop_min" in p:
        hp = ctx.median_pop(date)
        hot_checks.append(hp is not None and hp >= p["hot_pop_min"])
    if hot_checks and not (any(hot_checks) if p.get("hot_mode") == "or" else all(hot_checks)):
        return False
    s = p.get("strength")
    if s == "range":
        return float((v.h[0] - v.l[0]) / v.o[0])
    return float(pop) if s == "pop" else float(ext) if s == "ext" else True


def random_pick(v, p: dict, ctx) -> bool:
    """Placebo: fire on a pre-drawn random day per symbol (p['days'][symbol]); ignores prices."""
    return v.t == p["days"].get(v.symbol, -1)


def lookup(v, p: dict, ctx) -> bool:
    """Phase 2: fire where a (precomputed, out-of-sample) model decision says so. p['table'] = {'SYM|t': True}."""
    return bool(p["table"].get(f"{v.symbol}|{v.t}", False))


def peak(v, p: dict, ctx) -> bool:
    """Phase 3: entries aimed at the post-IPO top. Uses only days 1..t (the View). p['family']:
      A  pullback from a fresh high : running high >= (1+m)*offer, set within the last k days, close <= (1-p)*high
      B  reversal bar at the high   : today's high within 3% of the running high, close in bottom 30% of today's
                                      range and below today's open; optional volume >= vmult * prior 5-day average
      C  momentum exhaustion        : close_{t-1} / close_{t-6} - 1 >= r, and today closes down
      D  failed retest (lower high) : running high is >= n days old, a later high came within 10% of it, today
                                      closes below yesterday's low
      always                          : fires on the first tradable day (baseline)"""
    fam, t = p["family"], v.t
    h, l, c, o, vol = v.h, v.l, v.c, v.o, v.v
    if fam == "always":
        return True
    H = h.max()
    kH = int(np.argmax(h))          # 0-based day index of the running high (first occurrence)
    since = (t - 1) - kH            # trading days since the high was set (0 = today)
    if fam == "A":
        return bool(H >= (1 + p["m"]) * v.offer and since <= p["k"] and c[-1] <= (1 - p["p"]) * H)
    if fam == "B":
        rng = h[-1] - l[-1]
        if rng <= 0 or h[-1] < 0.97 * H or (c[-1] - l[-1]) / rng > 0.3 or c[-1] >= o[-1]:
            return False
        if p.get("vmult"):
            return bool(t >= 6 and vol[-1] >= p["vmult"] * vol[-6:-1].mean())
        return True
    if fam == "C":
        return bool(t >= 7 and c[-2] / c[-7] - 1 >= p["r"] and c[-1] < c[-2])
    if fam == "D":
        if since < p["n"] or t < 3:
            return False
        later = h[kH + 1: t - 1]    # highs after the peak, up to yesterday
        return bool(len(later) and later.max() >= 0.9 * H and c[-1] < l[-2])
    raise ValueError(fam)


SIGNALS = {"rules": rules, "random_pick": random_pick, "lookup": lookup, "peak": peak}
