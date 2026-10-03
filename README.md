# IPO Trading Strategy

**Can you make money by shorting newly listed stocks in their first months of trading?**
This project tests that question with real price data on about 1,000 US IPOs from 2015 to 2026.
It started as a study of how IPOs trade in their first six months. It grew into six rounds of strategy
research with pre-registered rules, untouched holdout periods and a 54-test verification suite.

**The result:** a market-neutral IPO short strategy that times its entry near each IPO's post-listing
peak. From January 2023 to July 2026 it returned **+28.3%** (7.2% a year) with a **−11.5% maximum
drawdown** and a **63% win rate** over 105 trades, after slippage and borrow costs. In 2025–26 it
returned **+36.7% with a Sharpe ratio of 1.75**. The strategy is regime-dependent: it was built for
fading, post-boom IPO markets like 2022–26 and loses in IPO booms (see
[Earlier market regimes](#earlier-market-regimes)).

---

## Why this project is worth a look

| What it shows | Where |
|---|---|
| **Research discipline.** Train / validation / holdout splits fixed up front. Selection rules written down *before* the results. Each holdout run once, then locked against reruns. | [strategy/research_log.md](strategy/research_log.md), `*.lock` files |
| **Honesty about failure.** All 51+ configurations are logged, including the failures and two process mistakes I disclosed. | [strategy/trials.csv](strategy/trials.csv), [strategy/research_log.md](strategy/research_log.md) |
| **A backtester built from scratch.** Event-driven, with slippage, daily borrow fees, gap fills, position limits, beta-hedging and trade-level exit overrides. | [strategy/lib/](strategy/lib/) |
| **Testing that catches real bugs.** 54 pytest tests, including look-ahead tests that replace future prices with random garbage and confirm no signal changes. They caught 6+ bugs before any results were read. | [strategy/test_*.py](strategy/), [strategy/test_logs/](strategy/test_logs/) |
| **Statistics beyond the backtest.** Bootstrap confidence intervals, Deflated Sharpe ratio for multiple testing, shuffled-label nulls, random-stock placebos, Wilcoxon tests and leave-one-era-out validation. | [strategy/lib/evaluate.py](strategy/lib/evaluate.py), [strategy/lib/placebo.py](strategy/lib/placebo.py) |
| **Machine learning without leakage.** 28 engineered features and walk-forward logistic regression and gradient-boosted trees, where each year is predicted only by models trained on trades that had already closed. | [strategy/lib/features.py](strategy/lib/features.py), [strategy/ml/](strategy/ml/) |
| **Data engineering.** Read-only exports from a Postgres market-data warehouse, plus Nasdaq IPO calendars, company profiles and lock-up terms, with automated data-quality checks and checksums. | [strategy/export_prices.py](strategy/export_prices.py), [strategy/fetch_*.py](strategy/), [strategy/data_checks.py](strategy/data_checks.py) |
| **Presenting results.** Self-contained HTML reports with interactive charts. | [ipo_trends_report.html](ipo_trends_report.html), [strategy/ipo_short_lab.html](strategy/ipo_short_lab.html) |

**Tools:** Python, pandas, NumPy, SciPy, scikit-learn, pytest, PostgreSQL, Docker, HTML/JS for reporting.

---

## The research, step by step

### 1. How do IPOs trade in their first 6 months? (96 large, well-known IPOs, 2021–2026)
Scripts: [analyze.py](analyze.py), [sector_analysis.py](sector_analysis.py) → [ipo_trends_report.html](ipo_trends_report.html)

- **Day 1:** median pop of +17% from offer price to close. After the open, though, the median stock goes nowhere for the rest of the day.
- **After that, a steady fade.** Measured against the S&P 500, the median IPO lagged by 4–10% in each window from week 3 to month 6, which was statistically significant in most windows.
- **Big drawdowns are normal:** the median peak-to-trough fall was −54%, and 79% of IPOs traded below their offer price at some point.
- Volatility was 60–90% annualized, versus about 12% for the S&P 500.

That looks like an easy short. The rest of the project tests whether it really is.

### 2. Rule-based short strategies (39 configurations)
- Universe expanded to **every** US operating-company IPO from 2019 to 2026, excluding SPACs and funds.
- Realistic costs: 25 bps slippage per side and a 30%/yr borrow fee, with an optional SPY hedge.
- **Best rule:** short when an IPO closes below its day-1 low or its offer price during a "hot" IPO market. It had a Sharpe ratio of about 0.9–1.0 on both train (2021–24) and validation (2024–26).
- **Holdout (2019–21), run once:** Sharpe −0.98 and a **−51% drawdown**. During the 2020–21 IPO boom, broken IPOs bounced back hard.

### 3. A machine-learning model with 28 indicators
- I wanted to know whether a model could learn the market regime that broke the rules.
- Walk-forward logistic regression and gradient boosting on 3,076 candidate trades, then one test on fresh **2015–18** data.
- **Result:** no predictive power on unseen years (rank IC −0.03) and a statistically significant loss. The models couldn't learn to avoid an IPO frenzy because their training data contained none.

### 4. Timing the entry near the peak
- **What worked:** waiting for a fast run-up followed by a 5% pullback. That entered **about 4–5 percentage points closer to the peak** than a random day for the same IPO, and the 95% CI on validation excluded zero.
- This was the one component that held up out of sample.

### 5. Searching for the best exit
- I searched 53 exit configurations: stops, trailing stops, profit targets, signal exits and fixed holds.
- Every protective exit got shaken out by IPO volatility.
- **The best exit depends on the regime:** hold 21 days when IPOs are fading, and get out fast when they're running.
- Sector-specific exits did no better than randomly shuffled sector labels (p = 0.86).

### 6. Event study: lock-up expiry (925 events)
- The 180-day insider lock-up expiry is mostly **priced in**. Nothing unusual happens on the day, and volume barely rises.
- Stocks drift down about 4% a month *into* the expiry. The decline slows afterwards; it doesn't speed up.

---

## Strategy performance, 2023–2026

**Rules:** short a new IPO after it has run at least 40% above its offer price and then closed 5% below
a high set within the last 10 days. Size each position by the stock's day-1 volatility, hedge with a
beta-weighted SPY long, and cover after 21 trading days. All figures are net of 25 bps slippage per side
and a 30%/yr borrow fee.

| Jan 2023 – Jul 2026 | |
|---|---|
| Total return | **+28.3%** |
| Annualized return | **+7.2%** |
| Sharpe ratio | **0.78** |
| Maximum drawdown | **−11.5%** |
| Trades | 105 |
| Win rate | **63%** |
| Average / median trade | +3.8% / +5.1% |
| Profit factor | 1.71 |

| Year | Trades | Win rate | Avg trade | Strategy return |
|---|---|---|---|---|
| 2023 | 9 | 44% | −1.8% | −2.9% |
| 2024 | 25 | 52% | −3.3% | −3.4% |
| 2025 | 44 | **66%** | **+5.1%** | **+12.4%** |
| 2026 (to Jul) | 27 | **74%** | **+10.1%** | **+21.6%** |

**2025–26 on its own:** +36.7% total return, 21.9% a year, Sharpe 1.75, maximum drawdown −7.6%.

The strategy holds only about 20% gross short exposure on average, because it waits for specific
setups. A desk could run it alongside other strategies or size it up. Most of the 2023–26 period
was used to design and validate the rules (entry chosen on 2021–24, confirmed on 2024–26), so these
figures are in-sample, not a live track record.

*Source: [strategy/peak/best_equity_full.csv](strategy/peak/best_equity_full.csv),
[strategy/peak/best_trades_full.csv](strategy/peak/best_trades_full.csv).*

### Earlier market regimes
The same rules lose money in IPO booms. In 2015–18 the CAGR was −10.4%, and in 2019–21 it was −20.3%,
with a −54% drawdown during the 2020–21 frenzy, when broken IPOs rallied instead of fading. Those years
were never used to choose the rules, which is what makes them a useful stress test. The next step is
a regime filter that switches the strategy off in hot IPO markets.

**Takeaways**
1. In fading IPO markets, timing the short near the peak produced a reliable edge: a 63–74% win rate in 2025–26.
2. Testing across several market eras showed exactly when the edge exists, and when it doesn't.
3. Simple, interpretable rules beat the 28-feature ML models out of sample.

---

## Repository layout

```
├── analyze.py, summarize.py, sector_analysis.py   # Part 1: descriptive study of 96 IPOs
├── ipo_trends_report.html                         # Part 1 report (open in a browser)
├── analysis/                                      # Part 1 output tables
└── strategy/
    ├── lib/              # backtest engine, features, evaluation metrics, placebo tests
    ├── ml/               # walk-forward ML dataset and models
    ├── peak/             # peak-timed entry, exit search, sector exits
    ├── lockup/           # lock-up expiry event study
    ├── test_*.py         # 54 pytest tests (engine, ML, peak entry, exits)
    ├── test_logs/        # saved test runs, data-quality checks, audits
    ├── runs*/, backtest_results*/   # per-configuration metrics, trades and equity curves
    ├── research_log.md   # full chronological lab notebook: every trial, every decision
    └── ipo_short_lab.html  # strategy-research report (open in a browser)
```

## Running it

```bash
cd strategy
python -m venv .venv && .venv\Scripts\activate    # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
pytest                                             # needs strategy/data/ (see below)
```

The raw price files (`strategy/data/`, about 230 MB) aren't in the repo. They were exported from a private
PostgreSQL market-data database with `export_prices*.py`, and the backtests and many tests need them. The
saved results, test logs and HTML reports are all included, so you can follow the findings without
re-running anything. GitHub doesn't render the HTML
reports; download them and open them in a browser.

---

*This is research, not investment advice.*
