"""Phase 4 tests: new exit types (trail, thesis stop, signal exits), priority, no look-ahead, and the search generator."""
import numpy as np
import pandas as pd
import pytest

from lib.engine import IPO, Config, find_entries, simulate
from lib.signals import peak
from ml.dataset import load_full
from test_engine import cal, make_ipo, one_entry

Z = dict(size=0.1, slippage=0, borrow=0)  # frictionless so prices are easy to check by hand


def trade(ipo, **kw):
    return simulate(Config("t", **{**Z, **kw}), {"TEST": ipo}, one_entry(ipo, 3), cal(ipo))["trades"].iloc[0]


def test_trailing_stop_uses_prior_lows_only():
    # entry day 3 @100. lows: d3 95, d4 90. Trail 10%: level on d4 = 95*1.1 = 104.5 (not hit, high 94);
    # level on d5 = 90*1.1 = 99 -> d5 high 100 hits, open 93 -> fill 99.
    ipo = make_ipo(opens=[100, 100, 100, 93, 93, 90], highs=[100, 100, 100, 94, 100, 90],
                   lows=[100, 100, 95, 90, 92, 90], closes=[100, 100, 96, 92, 95, 90])
    tr = trade(ipo, hold=4, trail=0.10)
    assert tr.exit_reason == "trail" and tr.exit_day == 5 and tr.exit_px == pytest.approx(99.0)
    assert tr.net_ret == pytest.approx(0.01)


def test_trailing_stop_gap_fills_at_open():
    ipo = make_ipo(opens=[100, 100, 100, 93, 105, 90], highs=[100, 100, 100, 94, 106, 90],
                   lows=[100, 100, 95, 90, 104, 90], closes=[100, 100, 96, 92, 105, 90])
    tr = trade(ipo, hold=4, trail=0.10)
    assert tr.exit_reason == "trail" and tr.exit_px == pytest.approx(105.0)


def test_thesis_stop_at_pre_entry_high():
    # pre-entry high = max(high day1, day2) = 120; day 4 high 121 -> cover at 120
    ipo = make_ipo(opens=[110, 108, 100, 101, 95], highs=[120, 110, 102, 121, 96], lows=[100, 100, 98, 99, 94],
                   closes=[105, 104, 100, 110, 95])
    tr = trade(ipo, hold=3, thesis_stop=True)
    assert tr.exit_reason == "thesis" and tr.exit_day == 4 and tr.exit_px == pytest.approx(120.0)
    assert tr.net_ret == pytest.approx(-0.20)


def test_ma5_signal_exits_next_open():
    # day 5 (k=4) close 99 > mean(100,100,100,95,99)=98.8 -> exit at day 6 open (97)
    ipo = make_ipo(opens=[100, 100, 100, 96, 96, 97, 90], closes=[100, 100, 100, 95, 99, 98, 90])
    tr = trade(ipo, hold=5, exit_signal="ma5")
    assert tr.exit_reason == "signal" and tr.exit_day == 6 and tr.exit_px == pytest.approx(97.0)


def test_prev_high_signal_exits_next_open():
    # day 4 close 99 > day 3 high 98? day3 high = max(open 100, close 97)=100 -> no. day 5 close 97 > day 4 high 96 -> exit day 6 open
    ipo = make_ipo(opens=[100, 100, 100, 95, 94, 96, 90], closes=[100, 100, 97, 96, 97, 95, 90])
    tr = trade(ipo, hold=5, exit_signal="prev_high")
    assert tr.exit_reason == "signal" and tr.exit_day == 6 and tr.exit_px == pytest.approx(96.0)


def test_priority_signal_before_stop_and_stop_before_target():
    # queued signal fires at the open even if the day also hits the stop
    ipo = make_ipo(opens=[100, 100, 100, 96, 96, 97, 90], highs=[100, 100, 100, 96, 99, 140, 90],
                   lows=[100, 100, 100, 95, 95, 60, 90], closes=[100, 100, 100, 95, 99, 98, 90])
    tr = trade(ipo, hold=5, exit_signal="ma5", stop=0.3, target=0.3)
    assert tr.exit_reason == "signal" and tr.exit_px == pytest.approx(97.0)
    # same day hits stop (130) and target (70): stop wins (conservative)
    ipo2 = make_ipo(opens=[100, 100, 100, 100], highs=[100, 100, 100, 135], lows=[100, 100, 100, 65], closes=[100, 100, 100, 100])
    tr2 = trade(ipo2, hold=2, stop=0.3, target=0.3)
    assert tr2.exit_reason == "stop" and tr2.exit_px == pytest.approx(130.0)


def test_tightest_protective_level_wins():
    # stop 40% (140), trail 10% from 95 -> 104.5, thesis 120: trail is tightest
    ipo = make_ipo(opens=[110, 108, 100, 100], highs=[120, 110, 100, 106], lows=[100, 100, 95, 99], closes=[105, 104, 97, 100])
    tr = trade(ipo, hold=3, stop=0.4, trail=0.10, thesis_stop=True)
    assert tr.exit_reason == "trail" and tr.exit_px == pytest.approx(104.5)


@pytest.fixture(scope="module")
def real():
    ipos, ctx, cal_, meta, panel = load_full()
    return ipos, ctx, cal_


def test_exit_decisions_ignore_bars_after_exit(real):
    ipos, ctx, cal_ = real
    params = {"family": "A", "m": 0.4, "p": 0.05, "k": 10}
    cfg = Config("t", signal="peak", params=params, hold=21, hedge=True, trail=0.15, thesis_stop=True, target=0.2,
                 exit_signal="ma5", size=0.001, max_positions=10**6, gross_cap=10**6, last_signal_day=41)
    sub = {s: i for s, i in ipos.items() if i.split in ("train", "holdout", "fresh_holdout")}
    ent = find_entries(cfg, sub, peak, ctx)
    tr = simulate(cfg, sub, ent, cal_)["trades"]
    assert len(tr) >= 100 and tr.exit_reason.nunique() >= 3, tr.exit_reason.value_counts()
    rng = np.random.default_rng(9)
    for t in tr.itertuples():
        ipo = sub[t.symbol]
        bad = IPO.__new__(IPO)
        for a in IPO.__slots__:
            v = getattr(ipo, a)
            setattr(bad, a, v.copy() if isinstance(v, np.ndarray) else v)
        cut = int(t.exit_day)  # keep days 1..exit_day, corrupt everything after
        for a in ("o", "h", "l", "c", "v"):
            arr = getattr(bad, a)
            arr[cut:] = rng.uniform(0.01, 1e4, len(arr) - cut)
        e2 = ent[ent.symbol == t.symbol]
        t2 = simulate(cfg, {t.symbol: bad}, e2, cal_)["trades"].iloc[0]
        assert (t2.exit_day, t2.exit_reason) == (t.exit_day, t.exit_reason) and t2.exit_px == pytest.approx(t.exit_px), t.symbol
    print(f"exit look-ahead check: {len(tr)} real trades, reasons {tr.exit_reason.value_counts().to_dict()}")


# ------------------------------------------------------------------ search generator
from peak.exits.search import GRID, default, key, neighbours, seed


def test_neighbours_are_valid_unique_and_deterministic():
    for x in seed():
        n1, n2 = neighbours(x), neighbours(x)
        assert [key(a) for a in n1] == [key(b) for b in n2]
        assert len({key(a) for a in n1}) == len(n1) and key(x) not in {key(a) for a in n1}
        for a in n1:
            assert all(a[k] in GRID[k] for k in GRID)
            diffs = [k for k in GRID if a[k] != x[k]]
            assert len(diffs) == 1  # exactly one parameter changed


def test_seed_contains_baseline_and_is_unique():
    s = seed()
    assert key(default()) in {key(x) for x in s}
    assert len({key(x) for x in s}) == len(s) == 22


def test_search_respects_budget_and_never_repeats():
    from peak.exits import search as S
    calls = []
    def fake_eval(x, bundle):
        calls.append(key(x))
        score = -abs((x["hold"] or 0) - 10) / 10 - (x["stop"] or 0)  # toy objective
        return {"id": key(x), **x, "objective": score, "mean_sharpe": score}
    orig = S.evaluate
    S.evaluate = fake_eval
    try:
        t, rounds, reason = S.search(None, budget=40, log=lambda s: None)
    finally:
        S.evaluate = orig
    assert len(calls) == len(set(calls)) <= 40
    assert reason


# ------------------------------------------------------------------ phase 5: per-trade exit overrides
def test_per_trade_exit_overrides_resolve_independently():
    a = make_ipo(opens=[100, 100, 100, 95, 90, 85, 80], closes=[100, 100, 100, 95, 90, 85, 80], symbol="AAA")
    b = make_ipo(opens=[100, 100, 100, 95, 90, 85, 80], closes=[100, 100, 100, 95, 90, 85, 80], symbol="BBB")
    ent = pd.concat([one_entry(a, 3).assign(hold=2), one_entry(b, 3).assign(hold=4, target=0.12)])
    tr = simulate(Config("t", hold=21, **Z), {"AAA": a, "BBB": b}, ent, cal(a))["trades"].set_index("symbol")
    assert tr.loc["AAA", "exit_reason"] == "time" and tr.loc["AAA", "exit_day"] == 4 and tr.loc["AAA", "exit_px"] == pytest.approx(95)
    # 12% target = 88; day 6 opens at 85, below the target, so the fill is the (better) open
    assert tr.loc["BBB", "exit_reason"] == "target" and tr.loc["BBB", "exit_day"] == 6 and tr.loc["BBB", "exit_px"] == pytest.approx(85)


def test_missing_override_columns_fall_back_to_config():
    a = make_ipo(opens=[100, 100, 100, 95, 90, 85, 80], closes=[100, 100, 100, 95, 90, 85, 80], symbol="AAA")
    ent = one_entry(a, 3).assign(hold=np.nan)
    tr = simulate(Config("t", hold=3, **Z), {"AAA": a}, ent, cal(a))["trades"].iloc[0]
    assert tr.exit_day == 5


def test_sector_profile_parsing_and_buckets():
    from fetch_sectors import BUCKET, parse
    js = {"data": {"Sector": {"value": "Health Care"}, "Industry": {"value": "Biotechnology: Pharmaceutical Preparations"}}}
    assert parse(js) == ("Health Care", "Biotechnology: Pharmaceutical Preparations")
    assert parse({"data": None}) == (None, None) and parse({"error": "x"}) == (None, None)
    assert parse({"data": {"Sector": {"value": "  "}}}) == (None, None)
    assert BUCKET["Telecommunications"] == "Technology" and BUCKET["Real Estate"] == "Finance"
    assert BUCKET.get(None, "Unknown") == "Unknown"
