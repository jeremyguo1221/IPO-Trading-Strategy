"""Expanding-window, walk-forward model training on the development years (2019-2026).

For each test year Y (2020..2026): train only on candidates whose 21-day trade EXITED before Jan 1 of Y
(so no training label overlaps the test year), predict every candidate that ENTERS in Y.
Thresholds:
  fixed : the natural cut (P(net > 0) >= 0.5 for logistic; predicted net >= 0 for boosting)
  inner : chosen inside the training window only - fit on data exited before Jan 1 of Y-1, predict year Y-1,
          pick the cut (grid of prediction quantiles) with the highest mean net label among >= 30 candidates,
          then refit on the full training window. Falls back to `fixed` when the inner split is too small.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from lib.features import ALL_FEATURES, IPO_FEATURES, REGIME_FEATURES

FEATURE_SETS = {"ipo": IPO_FEATURES, "regime": REGIME_FEATURES, "all": ALL_FEATURES}
TEST_YEARS = list(range(2020, 2027))
LABEL_CLIP = 0.5  # boosting target clipped to +/-50% so a few squeezes don't dominate the fit


def make_model(family: str):
    if family == "logit":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=0.1, max_iter=5000))
    if family == "gbm":
        return HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=150, min_samples_leaf=40,
                                             l2_regularization=1.0, random_state=0)
    raise ValueError(family)


def fit_predict(family: str, feats: list[str], train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    m = make_model(family)
    X, Xt = train[feats].to_numpy(float), test[feats].to_numpy(float)
    if family == "logit":
        m.fit(X, (train.y > 0).astype(int))
        return m.predict_proba(Xt)[:, 1]
    m.fit(X, train.y.clip(-LABEL_CLIP, LABEL_CLIP))
    return m.predict(Xt)


def natural_threshold(family: str) -> float:
    return 0.5 if family == "logit" else 0.0


def inner_threshold(family: str, feats: list[str], train: pd.DataFrame, year: int) -> tuple[float, str]:
    inner_tr = train[train.exit_date < pd.Timestamp(f"{year - 1}-01-01")]
    inner_te = train[train.year == year - 1]
    if len(inner_tr) < 150 or len(inner_te) < 60:
        return natural_threshold(family), "fallback_fixed"
    p = fit_predict(family, feats, inner_tr, inner_te)
    best, best_mean = natural_threshold(family), -np.inf
    for q in np.arange(0.0, 0.91, 0.1):
        cut = float(np.quantile(p, q))
        sel = inner_te.y[p >= cut]
        if len(sel) >= 30 and sel.mean() > best_mean:
            best, best_mean = cut, sel.mean()
    return best, "inner"


def walk_forward(dev: pd.DataFrame, family: str, fset: str, thr_mode: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    feats = FEATURE_SETS[fset]
    preds, folds = [], []
    for Y in TEST_YEARS:
        train = dev[dev.exit_date < pd.Timestamp(f"{Y}-01-01")]
        test = dev[dev.year == Y]
        if test.empty or len(train) < 50:
            continue
        p = fit_predict(family, feats, train, test)
        thr, how = (natural_threshold(family), "fixed") if thr_mode == "fixed" else inner_threshold(family, feats, train, Y)
        out = test[["symbol", "signal_day", "signal_date", "entry_date", "exit_date", "year", "y"]].copy()
        out["pred"], out["thr"], out["short_it"] = p, thr, p >= thr
        preds.append(out)
        folds.append({"year": Y, "train_rows": len(train), "train_max_exit": train.exit_date.max(),
                      "test_rows": len(test), "threshold": thr, "threshold_source": how,
                      "taken": int((p >= thr).sum()), "train_idx_overlap": len(set(train.index) & set(test.index))})
    return pd.concat(preds), pd.DataFrame(folds)


def final_fit(dev: pd.DataFrame, family: str, fset: str, thr_mode: str, holdout: pd.DataFrame) -> pd.DataFrame:
    """Train on ALL development candidates and predict the fresh holdout (used once, by final_eval_ml.py)."""
    feats = FEATURE_SETS[fset]
    p = fit_predict(family, feats, dev, holdout)
    if thr_mode == "fixed":
        thr = natural_threshold(family)
    else:
        last = int(dev.year.max())
        thr, _ = inner_threshold(family, feats, dev, last + 1)  # inner split: predict the last dev year
    out = holdout[["symbol", "signal_day", "signal_date", "entry_date", "exit_date", "year", "y"]].copy()
    out["pred"], out["thr"], out["short_it"] = p, thr, p >= thr
    return out


def decision_table(preds: pd.DataFrame) -> dict:
    """{'SYM|t': True} for every candidate the model would short (engine takes the first per IPO)."""
    return {f"{s}|{int(t)}": True for s, t in zip(preds.symbol[preds["short_it"]], preds.signal_day[preds["short_it"]])}
