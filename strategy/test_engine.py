"""Tests that the backtest engine is correct. Run: .venv/Scripts/python -m pytest -v
Every strategy result in this folder is only trusted if this whole file passes."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lib import zeus_metrics as zm
from lib.context import Context
from lib.engine import IPO, Config, find_entries, load_ipos, run, simulate
from lib.signals import SIGNALS, rules

HERE = Path(__file__).parent
DATA = HERE / "data"


# ------------------------------------------------------------------ helpers
def make_ipo(opens, highs=None, lows=None, closes=None, spy_o=None, spy_c=None, start="2023-01-02",
             symbol="TEST", offer=10.0, dates=None):
    n = len(opens)
    closes = closes if closes is not None else opens
    highs = highs if highs is not None else [max(o, c) for o, c in zip(opens, closes)]
    lows = lows if lows is not None else [min(o, c) for o, c in zip(opens, closes)]
    dates = dates if dates is not None else pd.bdate_range(start, periods=n)
    g = pd.DataFrame({"date": dates, "open": opens, "high": highs, "low": lows, "close": closes,
                      "volume": [1e7] * n, "spy_open": spy_o or [100.0] * n, "spy_close": spy_c or [100.0] * n})
    return IPO(symbol, "train", offer, pd.Timestamp(dates[0]), g)


def one_entry(ipo, entry_day, signal_day=None):
    return pd.DataFrame([{"symbol": ipo.symbol, "split": "train", "signal_day": signal_day or entry_day - 1,
                          "entry_day": entry_day, "entry_date": pd.Timestamp(ipo.dates[entry_day - 1]), "strength": 0.0}])


def cal(ipo):
    return pd.DatetimeIndex(ipo.dates)


# ------------------------------------------------------------------ 1. known answers
def test_known_answer_short_profit_with_exact_costs():
    #       day: 1    2    3    4
    ipo = make_ipo(opens=[10, 10, 100, 95], closes=[10, 10, 97, 90])
    cfg = Config("t", hold=2, size=0.10, slippage=0.0025, borrow=0.30)
    res = simulate(cfg, {"TEST": ipo}, one_entry(ipo, 3), cal(ipo))
    tr = res["trades"].iloc[0]
    shares = 0.10 / 100
    price_pnl = shares * (100 - 90)
    slip = shares * 100 * 0.0025 + shares * 90 * 0.0025
    borrow = 0.30 / 252 * shares * 100 + 0.30 / 252 * shares * 97
    assert tr.exit_reason == "time" and tr.exit_day == 4 and tr.days_held == 2
    assert tr.price_pnl == pytest.approx(price_pnl, abs=1e-12)
    assert tr.slippage == pytest.approx(slip, abs=1e-12)
    assert tr.borrow == pytest.approx(borrow, abs=1e-12)
    assert res["equity"].iloc[-1] == pytest.approx(1 + price_pnl - slip - borrow, abs=1e-12)
    assert tr.gross_ret == pytest.approx(0.10)


def test_short_loss_when_price_rises():
    ipo = make_ipo(opens=[10, 10, 100, 110], closes=[10, 10, 105, 120])
    res = simulate(Config("t", hold=2, size=0.1, slippage=0, borrow=0), {"TEST": ipo}, one_entry(ipo, 3), cal(ipo))
    assert res["trades"].iloc[0].net_ret == pytest.approx(-0.20)


def test_stop_intraday_fills_at_stop():
    ipo = make_ipo(opens=[10, 10, 100, 101, 100], highs=[10, 10, 101, 130, 100], lows=[10, 10, 99, 100, 99],
                   closes=[10, 10, 100, 102, 100])
    res = simulate(Config("t", hold=3, stop=0.25, size=0.1, slippage=0, borrow=0), {"TEST": ipo}, one_entry(ipo, 3), cal(ipo))
    tr = res["trades"].iloc[0]
    assert tr.exit_reason == "stop" and tr.exit_px == pytest.approx(125) and tr.exit_day == 4
    assert tr.net_ret == pytest.approx(-0.25)


def test_gap_through_stop_fills_at_open():
    ipo = make_ipo(opens=[10, 10, 100, 150], highs=[10, 10, 100, 160], lows=[10, 10, 100, 140], closes=[10, 10, 100, 155])
    res = simulate(Config("t", hold=3, stop=0.25, size=0.1, slippage=0, borrow=0), {"TEST": ipo}, one_entry(ipo, 3), cal(ipo))
    tr = res["trades"].iloc[0]
    assert tr.exit_reason == "stop" and tr.exit_px == pytest.approx(150)
    assert tr.net_ret == pytest.approx(-0.50)


def test_stop_on_entry_day():
    ipo = make_ipo(opens=[10, 10, 100, 100], highs=[10, 10, 140, 100], lows=[10, 10, 95, 100], closes=[10, 10, 110, 100])
    res = simulate(Config("t", hold=2, stop=0.25, size=0.1, slippage=0, borrow=0), {"TEST": ipo}, one_entry(ipo, 3), cal(ipo))
    tr = res["trades"].iloc[0]
    assert tr.exit_reason == "stop" and tr.exit_day == 3 and tr.exit_px == pytest.approx(125)


def test_profit_target_and_gap_down():
    ipo = make_ipo(opens=[10, 10, 100, 70], highs=[10, 10, 100, 72], lows=[10, 10, 90, 60], closes=[10, 10, 95, 65])
    res = simulate(Config("t", hold=3, target=0.20, size=0.1, slippage=0, borrow=0), {"TEST": ipo}, one_entry(ipo, 3), cal(ipo))
    tr = res["trades"].iloc[0]
    assert tr.exit_reason == "target" and tr.exit_px == pytest.approx(70)  # gapped below target: fill at open
    assert tr.net_ret == pytest.approx(0.30)


def test_hedge_leg_pnl():
    ipo = make_ipo(opens=[10, 10, 100, 100], closes=[10, 10, 100, 100], spy_o=[400, 400, 400, 404], spy_c=[400, 400, 402, 408])
    cfg = Config("t", hold=2, size=0.1, slippage=0, borrow=0, hedge=True, spy_slippage=0)
    res = simulate(cfg, {"TEST": ipo}, one_entry(ipo, 3), cal(ipo))
    tr = res["trades"].iloc[0]
    units = 1.5 * 0.1 / 400  # default beta 1.5 before 10 days of history
    assert tr.hedge_pnl == pytest.approx(units * (408 - 400))
    assert tr.beta == pytest.approx(1.5)


def test_hedge_uses_selected_instrument():
    ipo = make_ipo(opens=[10, 10, 100, 100], closes=[10, 10, 100, 100], spy_o=[400] * 4, spy_c=[400] * 4)
    iwm = pd.DataFrame({"open": [200, 200, 200, 210], "close": [200, 200, 205, 220]}, index=pd.DatetimeIndex(ipo.dates))
    ipo.add_hedge("IWM", iwm)
    cfg = Config("t", hold=2, size=0.1, slippage=0, borrow=0, hedge=True, hedge_symbol="IWM", spy_slippage=0)
    tr = simulate(cfg, {"TEST": ipo}, one_entry(ipo, 3), cal(ipo))["trades"].iloc[0]
    assert tr.hedge_pnl == pytest.approx(1.5 * 0.1 / 200 * (220 - 200))


def test_range_scaled_sizing():
    # day-1 range = (12-8)/10 = 40%; range_target 20% -> size scaled by 0.5
    ipo = make_ipo(opens=[10, 10, 100, 90], highs=[12, 10, 100, 90], lows=[8, 10, 100, 90], closes=[10, 10, 100, 90])
    tr = simulate(Config("t", hold=2, size=0.1, range_target=0.20, slippage=0, borrow=0), {"TEST": ipo}, one_entry(ipo, 3), cal(ipo))["trades"].iloc[0]
    assert tr.notional == pytest.approx(0.05)


def test_missing_bar_is_carried_not_crashed():
    dates = pd.DatetimeIndex(["2023-01-02", "2023-01-03", "2023-01-04", "2023-01-06", "2023-01-09"])  # 01-05 missing
    ipo = make_ipo(opens=[10, 10, 100, 95, 90], closes=[10, 10, 100, 95, 90], dates=dates)
    calendar = pd.bdate_range("2023-01-02", "2023-01-09")
    res = simulate(Config("t", hold=3, size=0.1, slippage=0, borrow=0), {"TEST": ipo}, one_entry(ipo, 3), calendar)
    tr = res["trades"].iloc[0]
    assert tr.exit_day == 5 and tr.days_held == 3 and tr.net_ret == pytest.approx(0.10)
    assert pd.Timestamp("2023-01-05") in res["equity"].index


def test_capacity_limits_respected():
    ipos = {f"S{i}": make_ipo(opens=[10, 10, 100, 100, 100], symbol=f"S{i}") for i in range(10)}
    entries = pd.concat([one_entry(ipo, 3) for ipo in ipos.values()])
    res = simulate(Config("t", hold=2, size=0.3, max_positions=5, gross_cap=1.0, slippage=0, borrow=0),
                   ipos, entries, cal(next(iter(ipos.values()))))
    assert len(res["trades"]) == 3  # gross cap 100% / 30% per trade => 3 fit
    assert res["ledger"].open_positions.max() <= 5


# ------------------------------------------------------------------ real-data fixtures
@pytest.fixture(scope="module")
def real():
    rd = lambda n, **k: pd.read_csv(DATA / n, keep_default_na=False, na_values=[""], **k)
    meta, panel = rd("ipo_meta.csv", parse_dates=["first_date", "priced_date"]), rd("panel.csv", parse_dates=["date"])
    uni = rd("ipo_universe.csv", parse_dates=["priced_date"])
    spy = rd("spy.csv", parse_dates=["date"])
    ipos = load_ipos(meta, panel)
    return {"ipos": ipos, "ctx": Context(uni, meta, panel), "cal": pd.DatetimeIndex(spy.date), "panel": panel, "meta": meta}


PARAM_SETS = [{"entry_day": 3}, {"from_day": 5, "breakdown": "low5"}, {"pop_min": 0.15, "entry_day": 6},
              {"from_day": 3, "breakdown": "d1_low", "hot_count_min": 20}, {"ext_min": 0.3, "from_day": 4, "hot_pop_min": 0.1}]


@pytest.mark.parametrize("params", PARAM_SETS)
def test_no_lookahead_future_corruption(real, params):
    """Destroying every bar AFTER an IPO's signal day must not change that IPO's signal or entry."""
    cfg = Config("t", signal="rules", params=params)
    syms = sorted(real["ipos"])[::7]
    sub = {s: real["ipos"][s] for s in syms}
    base = find_entries(cfg, sub, rules, real["ctx"]).set_index("symbol")
    assert len(base) >= 5, f"too few signals ({len(base)}) for this test to mean anything"
    print(f"lookahead check {params}: {len(base)} signals verified")
    rng = np.random.default_rng(0)
    for sym, r in base.iterrows():
        ipo = real["ipos"][sym]
        bad = IPO.__new__(IPO)
        for a in IPO.__slots__:
            setattr(bad, a, getattr(ipo, a).copy() if isinstance(getattr(ipo, a), np.ndarray) else getattr(ipo, a))
        cut = int(r.signal_day)  # days 1..cut stay intact; day cut+1 onward corrupted
        for a in ("o", "h", "l", "c", "v"):
            arr = getattr(bad, a)
            arr[cut:] = rng.uniform(0.01, 1e4, size=len(arr) - cut)
        again = find_entries(cfg, {sym: bad}, rules, real["ctx"])
        assert len(again) == 1 and int(again.signal_day.iloc[0]) == int(r.signal_day), sym


def test_view_cannot_see_future(real):
    ipo = next(iter(real["ipos"].values()))
    v = ipo.view(5)
    assert len(v.c) == 5 and len(v.h) == 5 and len(v.dates) == 5
    assert v.c[-1] == ipo.c[4]


def test_accounting_identity_real_run(real):
    cfg = Config("t", signal="rules", params={"entry_day": 3}, stop=0.3, hedge=True)
    res = run(cfg, real["ipos"], rules, real["ctx"], real["cal"], split="train")
    tr, led = res["trades"], res["ledger"]
    assert len(tr) > 30
    assert res["equity"].iloc[-1] - 1 == pytest.approx(tr.net_pnl.sum(), abs=1e-9)
    daily = led.equity_prev + led.price_pnl + led.hedge_pnl - led.slippage - led.borrow
    assert np.allclose(daily, led.equity, atol=1e-12)
    assert (led.gross_short <= cfg.gross_cap + 1e-9).all()
    assert (led.open_positions <= cfg.max_positions).all()
    for t_ in tr.itertuples():  # every fill is inside that day's real high-low range
        ipo = real["ipos"][t_.symbol]
        k = t_.exit_day - 1
        assert ipo.l[k] - 1e-9 <= t_.exit_px <= ipo.h[k] + 1e-9 or t_.exit_reason == "stop" and t_.exit_px <= ipo.h[k] + 1e-9


def test_independent_vectorized_cross_check(real):
    """Recompute each trade's net return from the raw panel with separate, vectorized code."""
    cfg = Config("t", signal="rules", params={"entry_day": 6}, hold=21, size=0.01, max_positions=10_000, gross_cap=100)
    res = run(cfg, real["ipos"], rules, real["ctx"], real["cal"], split="validation")
    tr = res["trades"]
    assert len(tr) >= 50, len(tr)
    print(f"cross-check: {len(tr)} trades recomputed independently")
    p = real["panel"].set_index(["symbol", "day"])
    s, borrow = cfg.slippage, cfg.borrow
    for t_ in tr.itertuples():
        e, x = t_.entry_day, t_.exit_day
        o_e = p.loc[(t_.symbol, e), "open"]
        c_x = p.loc[(t_.symbol, x), "close"]
        closes = p.loc[t_.symbol].loc[e:x - 1, "close"].to_numpy()
        expect = (1 - c_x / o_e) - s * (1 + c_x / o_e) - borrow / 252 * (1 + closes.sum() / o_e)
        assert t_.net_ret == pytest.approx(expect, abs=1e-9), t_.symbol
    assert len(tr) == len(res["entries"])  # unconstrained capacity: every signal became a trade


# ------------------------------------------------------------------ copied Zeus metrics
def test_zeus_metrics_known_values():
    assert zm.sharpe_ratio(pd.Series([0.0] * 10)) == 0.0
    # Known quirk of the copied Zeus function (left unmodified): a constant non-zero series has a float
    # std of ~1e-19 rather than exactly 0, so it returns a huge Sharpe instead of 0. Real returns never hit it.
    assert zm.sharpe_ratio(pd.Series([0.01] * 10)) > 1e6
    assert zm.max_drawdown(pd.Series([1.0, 1.2, 0.9, 1.3])) == pytest.approx(0.9 / 1.2 - 1)
    assert zm.profit_factor(pd.Series([3.0, -1.0, 2.0, -1.0])) == pytest.approx(2.5)
    assert zm.win_rate(pd.Series([1, -1, 1, 1])) == pytest.approx(0.75)
    rng = np.random.default_rng(1)
    strong = pd.Series(rng.normal(0.5, 1, 500))
    null = pd.Series(rng.normal(0, 1, 500))
    assert zm.stationary_bootstrap_p_value(strong, n_resamples=500)["p_value"] < 0.01
    assert zm.stationary_bootstrap_p_value(null - null.mean(), n_resamples=500)["p_value"] > 0.5
