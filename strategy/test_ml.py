"""Phase 2 tests: feature leakage, label correctness, walk-forward integrity, determinism, model signal.
Run: .venv/Scripts/python -m pytest -v test_ml.py"""
import numpy as np
import pandas as pd
import pytest

from lib.engine import IPO, Config, find_entries, simulate
from lib.features import ALL_FEATURES, IPO_FEATURES, REGIME_FEATURES, RegimeContext, compute
from lib.signals import lookup
from ml.dataset import BASE, CHECKPOINTS, label, load_full, rd
from ml.walkforward import decision_table, walk_forward


@pytest.fixture(scope="module")
def full():
    ipos, ctx, cal, meta, panel = load_full()
    uni = rd("ipo_universe.csv", parse_dates=["priced_date"])
    spy = rd("spy.csv", parse_dates=["date"])
    cands = pd.read_csv("ml/candidates.csv", keep_default_na=False, na_values=[""],
                        parse_dates=["signal_date", "entry_date", "exit_date"])
    return {"ipos": ipos, "ctx": ctx, "cal": cal, "meta": meta, "panel": panel, "uni": uni, "spy": spy, "cands": cands}


def _extra(meta_row):
    return {"shares": meta_row.get("shares", np.nan), "deal_size": meta_row["deal_size"], "exchange": meta_row["exchange"]}


def test_ipo_features_ignore_future_bars(full):
    rng = np.random.default_rng(1)
    meta = full["meta"].set_index("symbol")
    checked = 0
    for sym in sorted(full["ipos"])[::9]:
        ipo = full["ipos"][sym]
        for t in CHECKPOINTS:
            if t + 1 >= ipo.n:
                break
            base = compute(ipo.view(t), _extra(meta.loc[sym]), full["ctx"])
            bad = IPO.__new__(IPO)
            for a in IPO.__slots__:
                val = getattr(ipo, a)
                setattr(bad, a, val.copy() if isinstance(val, np.ndarray) else val)
            for a in ("o", "h", "l", "c", "v"):
                arr = getattr(bad, a)
                arr[t:] = rng.uniform(0.01, 1e4, len(arr) - t)
            again = compute(bad.view(t), _extra(meta.loc[sym]), full["ctx"])
            for k in IPO_FEATURES:
                assert (np.isnan(base[k]) and np.isnan(again[k])) or base[k] == again[k], (sym, t, k)
            checked += 1
    assert checked >= 200, checked
    print(f"IPO-feature leakage check: {checked} (IPO, day) pairs")


@pytest.mark.parametrize("date", ["2016-05-16", "2019-11-20", "2020-09-15", "2022-03-10", "2025-06-18"])
def test_regime_features_ignore_future_market_data(full, date):
    """Rebuild the context with EVERY bar (all IPOs and SPY) after `date` replaced by garbage: regime features at
    `date` must not change."""
    d = pd.Timestamp(date)
    rng = np.random.default_rng(2)
    panel = full["panel"].copy()
    fut = panel.date > d
    panel.loc[fut, ["open", "high", "low", "close"]] = rng.uniform(0.01, 1e4, (fut.sum(), 4))
    spy = full["spy"].copy()
    sf = spy.date > d
    spy.loc[sf, ["open", "high", "low", "close"]] = rng.uniform(1, 1e4, (sf.sum(), 4))
    ctx_bad = RegimeContext(full["uni"], full["meta"], panel, spy)
    a = {**full["ctx"].spy_feats(d), **full["ctx"].cohort_feats(d), "hot_count": full["ctx"].ipo_count(d), "hot_pop": full["ctx"].median_pop(d)}
    b = {**ctx_bad.spy_feats(d), **ctx_bad.cohort_feats(d), "hot_count": ctx_bad.ipo_count(d), "hot_pop": ctx_bad.median_pop(d)}
    for k in REGIME_FEATURES:
        va, vb = a[k], b[k]
        assert (va is None and vb is None) or (isinstance(va, float) and np.isnan(va) and np.isnan(vb)) or va == vb, (date, k, va, vb)
    assert a["cohort_n"] >= 3 or date == "2016-05-16"


def test_labels_match_engine(full):
    c = full["cands"].sample(80, random_state=4)
    checked = 0
    for r in c.itertuples():
        ipo = full["ipos"][r.symbol]
        ent = pd.DataFrame([{"symbol": r.symbol, "split": ipo.split, "signal_day": r.signal_day, "entry_day": r.signal_day + 1,
                             "entry_date": pd.Timestamp(ipo.dates[r.signal_day]), "strength": 0.0}])
        cfg = Config("t", hold=BASE.hold, hedge=True, size=0.001, max_positions=10**6, gross_cap=10**6)
        tr = simulate(cfg, {r.symbol: ipo}, ent, full["cal"])["trades"].iloc[0]
        assert tr.net_ret == pytest.approx(r.y, abs=1e-9), r.symbol
        assert tr.exit_reason == "time" and pd.Timestamp(tr.exit_date) == r.exit_date
        checked += 1
    assert checked == 80


def test_walkforward_integrity(full):
    dev = full["cands"][full["cands"].split != "fresh_holdout"]
    preds, folds = walk_forward(dev, "logit", "all", "inner")
    assert len(folds) >= 6
    for f in folds.itertuples():
        assert f.train_max_exit < pd.Timestamp(f"{f.year}-01-01"), f.year
        assert f.train_idx_overlap == 0
    assert (preds.year >= 2020).all()
    assert not preds.symbol.isin(full["cands"][full["cands"].split == "fresh_holdout"].symbol).any()


def test_walkforward_deterministic(full):
    dev = full["cands"][full["cands"].split != "fresh_holdout"]
    p1, _ = walk_forward(dev, "gbm", "all", "fixed")
    p2, _ = walk_forward(dev, "gbm", "all", "fixed")
    assert np.array_equal(p1.pred.to_numpy(), p2.pred.to_numpy())


def test_lookup_signal_takes_first_true_checkpoint(full):
    dev = full["cands"][full["cands"].split == "validation"]
    table = {f"{s}|{t}": True for s, t in zip(dev.symbol, dev.signal_day) if t in (5, 10)}
    cfg = Config("t", signal="lookup", params={"table": table}, hedge=True, last_signal_day=20)
    sub = {s: full["ipos"][s] for s in dev.symbol.unique()}
    ent = find_entries(cfg, sub, lookup, full["ctx"])
    assert len(ent) > 50
    expect = dev[dev.signal_day.isin([5, 10])].groupby("symbol").signal_day.min()
    got = ent.set_index("symbol").signal_day
    assert (got.reindex(expect.index) == expect).all()
