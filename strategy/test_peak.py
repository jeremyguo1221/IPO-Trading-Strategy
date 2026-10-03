"""Phase 3 tests: peak-entry rules use no future data, fire exactly where hand-checked, and efficiency is right."""
import numpy as np
import pandas as pd
import pytest

from lib.engine import IPO, find_entries
from lib.signals import peak
from ml.dataset import load_full
from peak.peak_entry import ENTRY_LAST, W, cfg_for, evaluate, rules_grid
from test_engine import make_ipo


@pytest.fixture(scope="module")
def full():
    ipos, ctx, cal, meta, panel = load_full()
    return {"ipos": ipos, "ctx": ctx}


FAMILY_REPS = {"A": {"family": "A", "m": 0.2, "p": 0.05, "k": 10}, "B": {"family": "B", "vmult": None},
               "C": {"family": "C", "r": 0.2}, "D": {"family": "D", "n": 5}}


@pytest.mark.parametrize("fam", list(FAMILY_REPS))
def test_no_lookahead(full, fam):
    cfg = cfg_for(FAMILY_REPS[fam])
    sub = {s: i for s, i in full["ipos"].items() if i.split in ("train", "validation", "holdout")}
    base = find_entries(cfg, sub, peak, full["ctx"]).set_index("symbol")
    assert len(base) >= 50, f"{fam}: only {len(base)} signals"
    rng = np.random.default_rng(5)
    for sym, r in base.iterrows():
        ipo = sub[sym]
        bad = IPO.__new__(IPO)
        for a in IPO.__slots__:
            val = getattr(ipo, a)
            setattr(bad, a, val.copy() if isinstance(val, np.ndarray) else val)
        cut = int(r.signal_day)
        for a in ("o", "h", "l", "c", "v"):
            arr = getattr(bad, a)
            arr[cut:] = rng.uniform(0.01, 1e4, len(arr) - cut)
        again = find_entries(cfg, {sym: bad}, peak, full["ctx"])
        assert len(again) == 1 and int(again.signal_day.iloc[0]) == cut, (fam, sym)
    print(f"{fam}: {len(base)} signals unchanged under future corruption")


def _first_signal(ipo, params):
    ent = find_entries(cfg_for(params), {ipo.symbol: ipo}, peak, None)
    return None if ent.empty else int(ent.signal_day.iloc[0])


def _path(closes, highs=None, lows=None, opens=None, vols=None, n=70):
    c = list(closes) + [closes[-1]] * (n - len(closes))
    h = list(highs or [x * 1.01 for x in closes]) + [closes[-1] * 1.01] * (n - len(closes))
    l = list(lows or [x * 0.99 for x in closes]) + [closes[-1] * 0.99] * (n - len(closes))
    o = list(opens or closes) + [closes[-1]] * (n - len(closes))
    ipo = make_ipo(opens=o, highs=h, lows=l, closes=c, offer=10.0)
    if vols:
        ipo.v[: len(vols)] = vols
    return ipo


def test_family_A_by_hand():
    # runs to 13 (high 13.13 on day 4 = +31% vs offer), then closes 12.4 on day 6 (<= 95% of 13.13 = 12.47)
    ipo = _path([11, 12, 12.5, 13, 12.8, 12.4, 12.0])
    assert _first_signal(ipo, {"family": "A", "m": 0.2, "p": 0.05, "k": 10}) == 6
    assert _first_signal(ipo, {"family": "A", "m": 0.4, "p": 0.05, "k": 10}) is None  # never ran 40%


def test_family_B_by_hand():
    closes = [11, 12, 13, 14, 13.2, 13.0]
    highs = [11.1, 12.1, 13.1, 14.1, 14.2, 13.1]      # day 5 makes a new high 14.2
    lows = [10.9, 11.9, 12.9, 13.9, 13.0, 12.9]       # day 5 closes 13.2: (13.2-13.0)/(1.2) = 0.17 <= 0.3
    opens = [11, 12, 13, 14, 14.0, 13.1]              # day 5 close 13.2 < open 14.0
    ipo = _path(closes, highs, lows, opens)
    assert _first_signal(ipo, {"family": "B", "vmult": None}) == 5


def test_family_C_by_hand():
    closes = [10, 10, 10, 10.5, 11, 11.5, 12.5, 12.2]   # day 7 close 12.5 vs day 2 close 10 = +25%; day 8 closes down
    ipo = _path(closes)
    assert _first_signal(ipo, {"family": "C", "r": 0.2}) == 8
    assert _first_signal(ipo, {"family": "C", "r": 0.4}) is None


def test_family_D_by_hand():
    closes = [12, 11, 10.5, 10.4, 10.3, 10.2, 10.6, 11.3, 11.1, 10.5]
    highs = [13, 11.2, 10.7, 10.5, 10.4, 10.3, 10.8, 11.9, 11.3, 10.7]   # peak 13 on day 1; day 8 retest 11.9 >= 90% of 13
    lows = [11.8, 10.9, 10.4, 10.3, 10.2, 10.1, 10.4, 11.0, 10.9, 10.4]  # day 10 close 10.5 < day 9 low 10.9
    ipo = _path(closes, highs, lows)
    assert _first_signal(ipo, {"family": "D", "n": 5}) == 10


def test_efficiency_math():
    ipo = _path([11, 12, 13, 14, 12.0, 11.0])      # peak high = 14 * 1.01 = 14.14 on day 4
    ent = pd.DataFrame([{"symbol": "TEST", "split": "train", "signal_day": 4, "entry_day": 5,
                         "entry_date": pd.Timestamp(ipo.dates[4]), "strength": 0.0}])
    m, d = evaluate({"TEST": ipo}, ent, {"TEST"})
    top = 14 * 1.01
    assert d.eff.iloc[0] == pytest.approx(12.0 / top)
    assert d.eff_random.iloc[0] == pytest.approx(np.mean(ipo.o[2:ENTRY_LAST] / top))
    assert d.peak_day.iloc[0] == 4


def test_grid_size():
    assert len(rules_grid()) == 25
