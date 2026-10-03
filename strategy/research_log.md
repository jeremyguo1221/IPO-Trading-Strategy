# IPO short strategy: research log

Everything tried is recorded here in order, including failures. Numbers come from `trials.csv` and
`runs/<run_id>/metrics.json`. The holdout (IPOs Jan 2019 – Sep 2021) is not touched until the final
strategy is chosen.

## Setup (2026-09-28)
- **Environment:** local `.venv` (see `requirements.txt`). The Zeus metric functions are copied verbatim into `lib/zeus_metrics.py`; nothing imports from the Zeus repo.
- **Universe:** Nasdaq IPO calendar, Jan 2019 – Sep 2026. 2,983 priced deals, of which 1,622 are operating companies after removing SPACs and funds. Cross‑check vs stockanalysis lists: 84–95% overlap by year; 2021 is lower only because that site caps each year at 500 rows.
- **Prices:** exported once, read‑only, from Zeus `ohlcv_daily`. 776 IPOs pass all automated checks: holdout 324, train 184, validation 268. See `test_logs/data_quality.txt`.
- **Known bias:** 342 sizeable IPOs have no price data (the database holds only stocks still trading), and 47 sizeable IPOs were dropped because of later reverse splits. These are mostly failed companies, so results are probably **conservative** for a short strategy.
- **Split:** train = first trade Oct 2021 – Jun 2024; validation = Jul 2024 – Jun 2026; holdout = Jan 2019 – Sep 2021.
- **Base costs:** 25 bps slippage per side; 30%/yr borrow accrued daily; no interest credited on short proceeds. No entry before day 3; each trade held at most 21 trading days. Each position is 7% of equity, at most 15 open, gross short exposure ≤ 100%.

## Engine verification
- **`pytest`: 18/18 pass** (`test_logs/pytest_engine_*.txt`):
  - hand‑computed costs are exact;
  - stop, gap‑through‑stop, entry‑day stop, profit target and gap‑down fills;
  - hedge P&L; a missing trading day; capacity limits;
  - no look‑ahead across 5 parameter sets (42–89 real signals each, future bars replaced with random garbage, identical entries);
  - daily accounting identity; every fill inside the day's high–low range;
  - 138 real trades recomputed independently, matching to 1e‑9;
  - the copied Zeus metrics.
- **Bugs found while writing and testing (all fixed before any results):**
  - slippage was double‑counted on both the stock and hedge legs;
  - a stock missing a trading day would have crashed the run;
  - tickers such as `NA` were read as missing values by pandas;
  - tests that could pass without testing anything (0 signals) now require a minimum count.
- **Zeus quirk (left unmodified, since the copy is verbatim):** `sharpe_ratio` on a constant non‑zero series returns about 1e17 instead of 0, because of float rounding in the standard deviation. Real return series are unaffected.
- **Random‑stock placebo** (`test_logs/placebo_random_established_R1_base_d3.json`): 819 shorts of random established stocks on the same dates and holding lengths as R1.
  - Gross short return −0.07% (t = −0.13), so no false edge.
  - Net −3.06%, matching modeled costs to within 0.01%. **PASS.**

## R1: baseline, short every tradable IPO at the day‑3 open, hold 21 days
| config | train CAGR | train Sharpe | train max DD | val CAGR | val Sharpe | val max DD |
|---|---|---|---|---|---|---|
| base_d3 (pure short) | −6.3% | −0.44 | −22.8% | +1.5% | 0.17 | −14.2% |
| base_d3_H (SPY‑hedged) | −0.7% | 0.01 | −19.2% | +13.7% | 0.78 | −17.1% |

**Diagnosis:** unconditional shorting roughly breaks even. Costs are about 3% per trade (0.5% slippage plus 2.5% borrow over 21 days), and they absorb the post‑IPO drift. Hedging helps, which says the market's rise was hurting the shorts. An edge has to come from selecting *which* IPOs to short and *when*.

## R2: entry timing (enter at the day 6 / 11 / 22 open, hold 21)
| config | train CAGR | train Sharpe | val CAGR | val Sharpe |
|---|---|---|---|---|
| entry_d6 | −2.8% | −0.23 | −8.4% | −0.37 |
| entry_d6_H | +0.2% | 0.07 | +0.8% | 0.13 |
| entry_d11 | −4.7% | −0.42 | +0.2% | 0.09 |
| entry_d11_H | −1.4% | −0.09 | +8.4% | 0.65 |
| entry_d22 | +2.8% | 0.34 | −1.5% | −0.02 |
| entry_d22_H | +3.0% | 0.40 | +1.2% | 0.16 |

**Diagnosis:** timing alone gives no edge. Day 22 helps train but not validation, so it's not kept.

**Train‑only feature scan** (`diagnostics/feature_scan_d{3,6,11}.txt`): unconstrained per‑IPO trades on train, 12 pre‑entry features. Two features were consistently positive at every entry day:
- hot IPO market (IPO count in prior 60 days): ρ = +0.15 / +0.25 / +0.35;
- day‑1 high–low range: ρ = +0.21 / +0.32 / +0.32.

Day‑1 pop, which the 96‑stock analysis flagged, shows no consistent effect across the broad universe. That's a caution about patterns found on large notable IPOs only. 12 features × 3 entry days is a multiple‑testing risk.

## R3: selection filters from the scan (thresholds = train terciles)
| config | train n | train Sharpe | train PF | val n | val Sharpe | val PF |
|---|---|---|---|---|---|---|
| hot31_d6 | 29 | −0.06 | 0.86 | 77 | −0.15 | 0.84 |
| hot31_d6_H | 30 | 0.22 | 1.33 | 77 | 0.20 | 1.11 |
| rng26_d6 | 16 | 0.97 | 3.36 | 24 | −0.43 | 0.44 |
| rng26_d6_H | 16 | 0.87 | 3.40 | 24 | −0.10 | 0.76 |
| hot25_rng19_d6 | 18 | 0.50 | 2.58 | 34 | −0.67 | 0.35 |
| hot25_rng19_d6_H | 18 | 0.54 | 3.22 | 34 | −0.28 | 0.62 |
| hot25_rng19_d11 | 14 | 0.81 | 5.60 | 29 | −0.19 | 0.78 |
| hot25_rng19_d11_H | 14 | 0.84 | 8.08 | 29 | 0.30 | 1.29 |

**Diagnosis:** a classic overfit.
- The day‑1‑range filter looked excellent on train (PF 3–8) and failed on validation.
- Train covers the 2022–24 IPO drought, only 14–30 trades per filter, so selection on it is mostly noise.
- Only the hot‑market filter with a hedge stayed (weakly) positive in both periods.
- Nothing from R3 is kept except the hot‑market idea as a candidate regime filter.

## R4: breakdown triggers (short on the first qualifying close in days 2–42)
| config | train n | train Sharpe | train PF | val n | val Sharpe | val PF | val max DD |
|---|---|---|---|---|---|---|---|
| bd_d1low | 65 | 0.26 | 1.17 | 105 | 0.30 | 1.16 | −11.6% |
| bd_d1low_H | 65 | 0.72 | 1.58 | 105 | 0.75 | 1.57 | −12.5% |
| bd_low5 | 68 | −0.15 | 0.87 | 106 | −0.52 | 0.68 | −31.5% |
| bd_low5_H | 68 | 0.02 | 0.98 | 107 | −0.05 | 0.93 | −22.2% |
| bd_offer | 66 | 0.30 | 1.22 | 81 | 0.61 | 1.47 | −7.3% |
| bd_offer_H | 66 | 0.77 | 1.79 | 81 | 0.72 | 1.55 | −6.4% |

**Diagnosis:** the first idea that holds in **both** periods. Shorting IPOs that close below their day‑1 low or below the offer price earns about +3–4% net per trade, win rate 57–64%.
- CAGR is low because capital sits idle: average gross exposure is only 12–29%.
- Still not significant: bootstrap p ≈ 0.2.
- A 5‑day‑low breakdown does *not* work, so the specific day‑1/offer levels matter.

## R5: combine breakdowns, hot‑market regime, shorter hold
| config | train n | train Sharpe | train PF | val n | val Sharpe | val PF | val max DD |
|---|---|---|---|---|---|---|---|
| bd_any | 83 | 0.14 | 1.06 | 121 | 0.12 | 1.02 | −11.5% |
| bd_any_H | 83 | 0.61 | 1.43 | 121 | 0.57 | 1.34 | −14.2% |
| bd_any_hot25 | 57 | 0.38 | 1.31 | 106 | 0.38 | 1.23 | −11.5% |
| **bd_any_hot25_H** | 57 | **0.86** | 1.88 | 106 | **0.85** | 1.62 | −14.1% |
| bd_any_h10 | 83 | 0.66 | 1.47 | 127 | −0.58 | 0.73 | −20.5% |
| bd_any_h10_H | 83 | 1.04 | 1.88 | 127 | −0.25 | 0.86 | −17.7% |

**Diagnosis:**
- The hot‑market filter now helps in both periods: **kept** (bd_any_hot25_H is the new leader).
- The 10‑day hold is another train‑only mirage (train 1.04, validation −0.25): not kept.

**Decision:** across all 14 idea pairs, the unhedged version never beat its hedged twin, in either period. From R6 on, only hedged versions are run, to save the trial budget (28 of 60 used).

## R6: risk management and sizing on the leader (hedged only)
| config | train n | train Sharpe | train max DD | val n | val CAGR | val Sharpe | val max DD |
|---|---|---|---|---|---|---|---|
| lead_stop25_H | 57 | 0.92 | −7.4% | 109 | +0.6% | 0.11 | −16.8% |
| lead_stop40_H | 57 | 0.71 | −8.0% | 107 | −0.1% | 0.06 | −19.1% |
| lead_size12_H (12%/trade, max 8) | 51 | 0.92 | −8.0% | 91 | +14.3% | 0.74 | −21.5% |
| bdo_hot25_H (offer breakdown only) | 48 | 0.37 | −10.2% | 73 | +9.7% | 0.84 | −5.8% |

**Diagnosis:**
- Stops improve train and wreck validation. IPO swings stop out trades that would have recovered, and the gap‑fill rule charges the full gap. **No stop.**
- Bigger sizing scales return and drawdown together without improving Sharpe. It's a leverage choice, left for the end.
- The leader stays bd_any_hot25_H (Sharpe ≈ 0.85 in both periods, bootstrap p ≈ 0.2, not yet significant).

## R7: signal quality
| config | train n | train Sharpe | train PF | val n | val CAGR | val Sharpe | val PF | val max DD |
|---|---|---|---|---|---|---|---|---|
| lead_both_H (both breakdowns) | 36 | 0.38 | 1.51 | 59 | +10.0% | 0.88 | 1.85 | −6.3% |
| lead_hotpop_H (heat = recent median pop ≥ 10%) | 44 | 0.82 | 2.05 | 59 | +15.3% | 1.07 | 1.72 | −7.9% |
| lead_late_H (ignore breakdowns before day 10) | 41 | 0.88 | 2.17 | 83 | +1.2% | 0.16 | 1.06 | −17.1% |

**Diagnosis:**
- lead_hotpop_H is the first to clear Sharpe 1.0 on validation. But its train Sharpe (0.82) is *below* the leader's (0.86), so under the pre‑registered keep rule (improve train, don't worsen validation) it does **not** replace the leader. Promoting it would be selecting on validation.
- lead_late_H is another train‑only effect.

**Final‑selection rule, fixed now before any further runs:** the final strategy is the configuration with the highest **min(train Sharpe, validation Sharpe)** among all trials. It rewards consistency, not one good period. Current standings: bd_any_hot25_H 0.85 · lead_hotpop_H 0.82 · bd_offer_H 0.72 · bd_d1low_H 0.72.

**Next (R8):** IPOs behave like small‑cap growth stocks, so hedging with IWM (Russell 2000) instead of SPY should remove more market noise. This is a risk‑model choice, not a data‑mined signal. Needs a one‑time read‑only export of IWM bars.

## R8: hedge instrument = IWM (Russell 2000) instead of SPY
IWM isn't in the Zeus database (ETFs other than SPY aren't ingested), and Stooq requires browser verification, which I didn't try to get around. IWM came from the Nasdaq historical API: 1,968 bars on exactly the SPY calendar (`data/iwm.csv`, raw `data/raw_iwm_nasdaq.json`). Its prices are not dividend‑adjusted, which slightly understates the long hedge (conservative). The engine refactor was regression‑checked: SPY‑hedged results are identical to before (`test_logs/regression_after_hedge_refactor.txt`).

| config | train Sharpe | train max DD | val CAGR | val Sharpe | val max DD |
|---|---|---|---|---|---|
| lead_iwm_H | 0.44 | −7.3% | +15.6% | 1.11 | −14.5% |
| hotpop_iwm_H | 0.49 | −8.7% | +16.6% | 1.16 | −6.7% |

**Diagnosis:** IWM helps validation but hurts train, because small caps badly lagged in 2022–24. It isn't consistent (min Sharpe 0.44–0.49), so it's not kept.

## R9: final planned round
| config | train n | train CAGR | train Sharpe | train max DD | val n | val CAGR | val Sharpe | val PF | val max DD |
|---|---|---|---|---|---|---|---|---|---|
| **lead_rangesize_H** | 57 | +8.3% | **0.88** | −8.3% | 105 | +13.7% | **0.98** | 1.69 | −11.0% |
| lead_hot_or_H | 68 | +6.8% | 0.72 | −8.6% | 113 | +7.7% | 0.57 | 1.35 | −14.1% |

**Diagnosis:** risk‑balanced sizing (smaller bets on IPOs with a wild day 1) improves **both** periods, so it's kept. The broader "or" hot definition dilutes the signal.

**Iteration stopped after 39 of 60 configurations.** Each further trial raises the multiple‑testing penalty, and the remaining ideas lacked a strong prior. By the pre‑fixed rule (highest min(train, val) Sharpe) the final strategy is **R9_lead_rangesize_H**.

### Final strategy rules (plain language)
1. Watch every newly listed US operating company (not SPACs or funds) with an offer price ≥ $5.
2. From day 2 to day 42 after listing, at each close check whether the stock closed **below its day‑1 low or below its IPO price**. Also require that the IPO market is hot: **≥ 25 IPOs priced in the previous 60 days**. It must be tradable too: price ≥ $5 and median daily dollar volume ≥ $5M.
3. On the first day all of that is true, **short at the next open**. Base size is 7% of equity, scaled by 18.6% ÷ (day‑1 high–low range), within 0.5–1.5×. Hedge with a long SPY position of β × the short's size, with β estimated from pre‑entry data.
4. **Cover at the close of the 21st trading day.** No stop‑loss, no profit target. At most 15 positions and 100% gross short exposure.

## Process note: holdout run started directly (2026-09-28)
I meant to dry‑run `final_eval.py` on train and validation first, so a crash after the holdout couldn't force a rerun. The patch adding `--dry-run` failed an assertion and **wrote nothing**. The follow‑on command then ran the original `final_eval.py`, which evaluates train, validation **and the holdout** in one pass.

- This is still the single holdout evaluation of the strategy selected by the rule fixed before R8. No selection happened after seeing any holdout number.
- It's recorded here for transparency.

## Final evaluation (`final_eval.py` → `backtest_results/scorecard.json`), run once
The process note above applies: the script ran all three periods in one pass and completed without errors (22m50s). `backtest_results/HOLDOUT_RUN.lock` now blocks any rerun.

| | Train (Oct 21 – Jun 24) | Validation (Jul 24 – Jun 26) | **Holdout (Jan 19 – Sep 21)** |
|---|---|---|---|
| trades | 57 | 105 | 166 |
| CAGR | +8.3% | +13.7% | **−18.2%** |
| Sharpe | 0.88 | 0.98 | **−0.98** |
| max drawdown | −8.3% | −11.0% | **−51.2%** |
| profit factor | 1.93 | 1.69 | 0.60 |
| yearly | 21 +7.7 / 22 +5.5 / 23 +2.5 / 24 +8.2 | 24 −3.6 / 25 +6.6 / 26 +25.5 | 19 −9.4 / 20 −19.6 / 21 −19.2 |
| bootstrap p | 0.13 | 0.14 | 0.11 (negative) |
| shuffled null: median / 95th pct Sharpe | −0.00 / 0.70 | 0.54 / 1.29 | −0.95 / −0.37 |
| Deflated Sharpe prob (39 trials) | 0.49 | 0.44 | 0.00 |
| borrow 60%/yr: total return | +12.9% | +5.3% | −58.6% |
| borrow 100%/yr: total return | −2.7% | −20.1% | −74.5% |

**Verdict: FAIL.** The strategy fails the pre‑registered bar in every period and loses heavily on the holdout.

### Engine checks on the final trades: all consistent
- Spot audit: 5/5 random trades (BRCB, ULCC, CVRX, NRDS, KMTS) recomputed from the raw exported bars match the engine to 1e‑6.
- Random‑established‑stock placebo on the final trade dates: gross −1.71% (those stocks rose over those windows), net −4.73%, i.e. gross minus costs, as expected.

### Post‑mortem (`diagnostics/holdout_postmortem.txt`; diagnostic only, nothing was tuned)
- **Random selection explains most of it.** In the holdout, random IPO shorts with the same trade count and timing had a median Sharpe of −0.95; the strategy scored −0.98. In validation, random IPO shorts had a median Sharpe of +0.54. The breakdown and hot‑market rules added little beyond the **IPO‑cycle regime**: shorting any IPO worked in 2022–26 and failed in 2019–21.
- **The hot‑market filter backfired in 2020–21.** The market was hot, and hot IPOs kept rising for months. Broken IPOs rebounded violently: INMD −80%, CRBU −78%, ZIM −66%, FTHM −64% per trade. The fade the analysis found came only after the 2021 peak.
- **Costs decide the sign.** With zero costs, validation Sharpe is 1.85 and holdout Sharpe is −0.01. At realistic IPO borrow rates (often 60–100%+ in the first month), even the good periods turn negative.
- **The train–validation agreement was partly luck.** Both periods sat inside the same 2022–26 post‑bubble regime. Only the older holdout could reveal that.

### Conclusion
No consistent high‑return one‑month IPO short strategy was found under rigorous testing. The best candidate looked good in 2021–2026 (Sharpe ≈ 0.9–1.0) because those years were a favourable regime for shorting IPOs in general, and it lost 51% peak to trough in 2019–2021. The holdout has now been used, so further tuning on these years would be fitting to known outcomes. A new cycle of research needs fresh untouched data, e.g. 2015–2018 IPOs, if the Nasdaq calendar and price data go back that far.

---
# Phase 2: multi‑variable, regime‑aware model (plan: `.claude/plans/using-the-analysis-you-jaunty-fern.md`)

**Question from the user:** would a more comprehensive strategy with many more indicators work better?

**Expectation stated up front:** adding variables alone mostly adds overfitting. The phase‑1 failure was the IPO‑market regime, so the hope rests on regime variables (IPO‑cohort momentum, share of recent IPOs below their offer price, SPY trend) inside a model trained only on past data.

## New untouched test: 2015–2018
- The Nasdaq calendar goes back to 2015, adding 685 more operating‑company IPOs (2015–18) to `data/full/ipo_universe.csv`. The 2019–26 counts are identical to phase 1.
- Read‑only export of 2015–18 bars, SPY from Dec 2014, and a new 300‑stock placebo set, all into `data/full/`. The container was stopped afterwards.
- Phase‑1 data files are unchanged: 12/12 checksums OK (`test_logs/phase1_data_checksums.txt`).
- Data checks (`test_logs/data_quality_phase2.txt`): **241 clean IPOs in the fresh holdout** (first trade 2015–2018). The 2019–26 split counts are identical to phase 1.

## Features and candidates
- **`lib/features.py`:** 28 features, 19 about the IPO and 9 about market regime, listed in the file.
- **`ml/dataset.py`:** 3,076 tradable candidates from 841 IPOs, one per IPO per checkpoint (signal days 2/5/10/15/20). Each is labeled with its 21‑day net SPY‑hedged short return, using the engine's cost formula.
- **Mistake disclosed:** the first dataset build printed the fresh holdout's **unconditional mean label (−4.95%)**. It's a base rate for shorting every tradable IPO; nothing in model selection uses it, and selection uses 2019–26 only. The print is now suppressed for the fresh holdout.
- Development base rates (mean net label): 2019–21 −4.34%, 2021–24 +0.55%, 2024–26 +1.62%.

## Tests (`test_ml.py`, 10 tests; with `test_engine.py`: 30/30 pass, `test_logs/pytest_phase2_*.txt`)
- **IPO features:** 565 (IPO, day) pairs unchanged when every later bar is replaced with garbage.
- **Regime features at 5 dates** (2016–2025) unchanged when **all** IPO and SPY bars after the date are replaced with garbage and the context is rebuilt.
- **Labels:** 80 random labels equal the engine's `net_ret` to 1e‑9.
- **Walk‑forward:** every training row exits before its test year; no train/test overlap; no fresh‑holdout rows; identical predictions on rerun.
- **Lookup signal:** takes the first qualifying checkpoint per IPO.
- **Bugs caught by these tests before any result:**
  - the float `signal_day` feature overwrote the integer key column;
  - `preds.take` resolved to the pandas method instead of the column (renamed to `short_it`).

## Walk‑forward trials (`run_ml.py` → `trials_ml.csv`, `runs_ml/`)
Out‑of‑sample 2020–2026: each year is predicted by a model trained only on trades that exited before that year. The trades use the phase‑1 final mechanics (SPY hedge, range sizing, 21‑day hold, same costs).

| run | trades | CAGR | Sharpe | max DD | worst‑year Sharpe | OOS rank IC |
|---|---|---|---|---|---|---|
| baseline: short every candidate | 426 | −8.1% | −0.37 | −61.6% | −1.34 | – |
| baseline: phase‑1 R9 rule | 292 | +1.1% | 0.15 | −41.4% | −1.15 | – |
| logit · IPO · fixed | 231 | −0.8% | 0.00 | −36.8% | −1.40 | 0.109 |
| logit · IPO · inner | 185 | +1.5% | 0.19 | −28.7% | −1.40 | 0.109 |
| logit · regime · fixed | 197 | −8.7% | −0.60 | −51.7% | −2.15 | 0.043 |
| logit · regime · inner | 237 | −6.2% | −0.38 | −45.7% | −1.28 | 0.043 |
| **logit · all · fixed** (selected) | 232 | −2.9% | −0.14 | −44.0% | **−1.09** | 0.139 |
| logit · all · inner | 222 | −4.2% | −0.27 | −45.8% | −1.76 | 0.139 |
| gbm · IPO · fixed | 249 | −1.5% | −0.05 | −38.3% | −1.91 | 0.088 |
| gbm · IPO · inner | 336 | −5.7% | −0.28 | −54.6% | −1.91 | 0.088 |
| gbm · regime · fixed | 191 | −5.6% | −0.45 | −42.4% | −1.23 | 0.045 |
| gbm · regime · inner | 227 | −4.5% | −0.32 | −38.0% | −1.28 | 0.045 |
| gbm · all · fixed | 287 | −8.7% | −0.55 | −59.0% | −1.76 | 0.051 |
| gbm · all · inner | 351 | −6.4% | −0.32 | −53.8% | −1.76 | 0.051 |

**Diagnosis:**
- **No model is profitable out of sample in a meaningful way.** The best overall is +1.5% CAGR (Sharpe 0.19, −29% drawdown).
- **The models do rank candidates.** Out‑of‑sample rank IC is positive for all 12 (0.04–0.14), and most beat "short everything" (−8.1% CAGR). But the ranking is too weak to overcome ~3% costs per trade and the regime swings.
- **The regime variables did not solve the regime problem; they were the worst sets.** When predicting 2020 and 2021, the models had only 2019–20 to learn from, which contains no example of an IPO frenzy. They can't learn to avoid a regime they have never seen. Every model loses in 2020 and 2021 (yearly Sharpe −0.1 to −2.1), and most are positive only in 2025–26.
- **Boosted trees are worse than logistic regression.** The extra flexibility fits noise with this few samples (~200–600 training candidates per fold).
- **Selected by the pre‑fixed rule** (best worst‑year Sharpe): `M_logit_all_fixed`. It already fails on development, and it still gets its single fresh‑holdout evaluation, as planned.

## Phase‑2 final evaluation (`final_eval_ml.py`)
- **Dry run first** (development only, `test_logs/final_eval_ml_dry_run.txt`): completed without errors. Only then was the real run started, fixing the phase‑1 process slip.
- **Selected model** on development OOS 2020–26: CAGR −2.9%, Sharpe −0.14, max DD −44.0%. Yearly Sharpe: 2020 −1.00 · 2021 −1.09 · 2022 −0.15 · 2023 +0.54 · 2024 −0.27 · 2025 +0.57 · 2026 +3.03. With zero costs, Sharpe 0.41.
- **Retrained on all 2019–26 candidates**, then run **once** on the fresh 2015–18 holdout (`test_logs/final_eval_ml_holdout_run.txt`, lock `backtest_results_ml/FRESH_HOLDOUT_RUN.lock`):

| fresh holdout 2015–18 | value |
|---|---|
| trades | 67 (the model shorted 19% of candidates) |
| CAGR / Sharpe / max DD | **−9.9% / −1.39 / −37.5%** |
| yearly Sharpe | 2015 0.02 · 2016 −1.58 · 2017 −0.75 · 2018 −2.73 · 2019 (trades exiting early 2019) 2.93 |
| profit factor | 0.38 |
| bootstrap p | 0.014, i.e. a *significant loss* |
| prediction vs outcome rank IC | **−0.03** (no predictive power) |
| random IPO shorts, same timing: median / 95th pct Sharpe | −1.12 / −0.46; the model beats only 28% of random runs |
| zero costs | −5.8% CAGR, Sharpe −0.80 |
| 60% / 100% borrow | −43% / −53% total |
| Deflated Sharpe prob (51 trials) | 0.00 |

- Placebo on holdout dates: gross +0.24%, net −2.76% = −costs. Spot audit: 5/5 trades (STNE, CTMX, VERI, ALRM, SNAP) match hand calculation from the raw bars.

**Verdict: FAIL.** The multi‑variable model has no predictive power on unseen years (IC −0.03) and loses money even before costs.

### Which variables mattered at all (development OOS, `backtest_results_ml/dry_run/feature_ic_by_year.csv`)
- The strongest univariate indicators were liquidity and first‑day trading intensity: dollar volume (IC +0.12), day‑1 turnover (+0.11), and volatility (+0.09). More liquid, heavily flipped IPOs made slightly better shorts.
- The model leaned most on the share of recent IPOs below their offer price and on price vs offer.
- All effects are small (|IC| ≤ 0.12) and inconsistent from year to year.

## Overall conclusion (phases 1 + 2, 51 configurations)
Across hand‑built rules (39) and 28‑indicator models (12), tested out of sample on three separate eras (2015–18, 2019–21, 2020–26), **no one‑month IPO short strategy shows a consistent, cost‑surviving edge.**
- Short‑horizon IPO returns are dominated by the IPO‑market cycle, which isn't predictable from the history available here.
- Realistic borrow costs (~3%/month at 30%/yr, often far more for new IPOs) consume what small ranking skill exists.
- More variables increased the ways to fit noise without adding out‑of‑sample signal.

**Scorecard correction (display only, no rerun):** the "Bootstrap p < 0.05" check in `final_eval.py` and `final_eval_ml.py` tested significance in *either* direction, so the fresh holdout's significant **loss** (p = 0.014, Sharpe −1.39) was scored "pass". The report now counts it as a pass only for a significant gain and labels the holdout value "(significant loss)". No verdict changes: every result failed anyway.

---
# Phase 3: entry timed at the peak (plan: shorting as close as possible to the highest price)

**Goal (user):** find an entry that shorts at, or near, the IPO's highest price. Exits come later.

**Metric (fixed before running):**
- efficiency = entry open ÷ highest high of days 1–63;
- **paired gain** = rule efficiency − the same IPO's random‑day efficiency (entry days 3–42);
- pick the best mean gain on **train** among rules firing on ≥ 40% of eligible IPOs;
- "works" only if validation gain > 0 with its bootstrap CI above 0.

**Data honesty:** development = train (IPOs 2021–24) plus validation (2024–26). The 2019–21 and 2015–18 sets were each used once before, so here they're secondary checks only.

**Train baseline** (`peak/baseline_train_only.txt`): 30% of IPOs peak on day 1 (not shortable), 42% in days 22–63. A fixed entry day captures a median of ~78% of the peak; perfect hindsight within days 3–42 captures 91–94%.

**Tests (`test_peak.py`, 10; full suite 40/40):**
- no look‑ahead: 186–448 real signals per rule family unchanged when future bars are replaced with garbage;
- each family fires on exactly the hand‑computed day of a synthetic path;
- efficiency math checked.

**Spot audit** (`test_logs/peak_spot_audit.txt`): 5 random validation entries recomputed from the raw price file. The rule is true on the signal day, false on every earlier day, and efficiency is identical.

## 25 rules (`peak/trials_peak.csv`, `peak/run_output.txt`)
Paired gain is in percentage points of the peak price. Train rules shown are those firing on ≥ 40%.

| rule | train fire | train gain | val fire | val gain [95% CI] | val median efficiency |
|---|---|---|---|---|---|
| **A: ran ≥ 40% above offer, closes ≥ 5% below a high set within 10 days** (selected) | 42% | **+5.4** | 50% | **+4.2 [1.8, 6.6]** | 78% |
| A: same, high within 5 days | 41% | +5.2 | 50% | +4.2 [1.5, 6.8] | 78% |
| A: ran ≥ 20%, 15% pullback, 5 days | 43% | +4.3 | 47% | +2.3 [0.0, 4.6] | 74% |
| A: ran ≥ 20%, 5% pullback | 65–66% | +3.3–3.6 | 68% | +4.2 [2.0, 6.4] | 78% |
| B: reversal bar at the high | 53% | +0.6 | 52% | +3.3 [1.2, 5.6] | 84% |
| B: reversal bar + 1.5× volume | 18% | +1.8 | 11% | +7.3 | 89% |
| C: first down close after +20% in 5 days | 20% | +4.1 | 24% | +5.4 | 75% |
| D: failed retest, high ≥ 5 days old | 54% | −0.2 | 58% | +0.6 [−0.9, 1.9] | 80% |
| D: failed retest, high ≥ 10 days old | 43% | −1.3 | 48% | −0.9 [−2.5, 0.5] | 79% |
| baseline: fixed day 3 / 6 / 11 / 22 | – | +2.3 / +2.0 / +0.4 / +0.3 | – | +3.7 / +1.4 / +2.3 / −0.4 | – |

**Findings:**
- **The selected rule works as an entry rule.** It shorts about 4–5 percentage points of the peak closer to the top than a random day for the same IPOs, and the validation CI excludes zero. Pullback‑from‑a‑fresh‑high rules (family A) are consistently the best family. Waiting for a failed retest (D) doesn't help.
- **"The highest price" can't be hit reliably.** Only 11% (train) to 18% (validation) of the selected rule's entries are within 10% of the eventual 63‑day peak. Median efficiency is 75–78%; perfect hindsight would be 92–94%. Reversal bars on heavy volume get closest (89%) but fire on only 5–18% of IPOs, too few to rely on.
- **Where the rule enters:** two‑thirds of signals come on day 2, meaning IPOs whose day‑1 high was ≥ 40% above offer and that closed ≥ 5% off it on day 2. In 49–63% of cases the stock later made a *higher* high within 63 days, so the rule shorts near a local top that often isn't the final top.

## P&L context: selected entry + 21‑day hold, SPY hedge, base costs (engine)
| period | trades | CAGR | Sharpe | max DD |
|---|---|---|---|---|
| train 2021–24 | 54 | +4.5% | 0.56 | −9.6% |
| validation 2024–26 | 81 | +15.6% | 1.32 | −7.6% |
| 2019–21 (previously used) | 190 | −20.3% | −0.98 | −53.6% |
| 2015–18 (previously used) | 82 | −10.4% | −1.08 | −39.9% |

A better entry raises the average 21‑day trade from +0.6/+1.6% (short every IPO) to about +5.1% (train) / +4.5% (validation). But the regime problem from phases 1–2 is unchanged: it makes money in 2022–26 and loses heavily in 2015–21. **The entry is the part that improved. The exit and the regime filter are what still fail.**


---
# Phase 4: self‑iterating exit search (entry fixed = the phase‑3 rule)

**Setup** (plan approved; the user chose a self‑iterating search):
- **Engine additions:** trailing stop, thesis stop, and `ma5`/`prev_high` signal exits filled at the next open.
- **Tests:** `test_exits.py` has 11 tests. They include hand‑checked paths, exit priority, and a look‑ahead check on 341 real trades across 4 exit types, plus tests of the search generator itself. Full suite **51/51**.
- **Regression:** the phase‑1 final strategy reproduces exactly.

**Search data:** the three eras 2015–18, 2019–21 and 2021–24 (82 / 205 / 54 trades), all previously seen. Objective = worst‑era portfolio Sharpe. The 2024–26 validation set was used only once, at the end, for exits.

## Search run (`peak/exits/search.py` → `trials_exit.csv`, `rounds.csv`, `search_log.txt`)
- **Round 0:** 22 seed exits. Leader: **5‑day hold** (worst‑era Sharpe −0.75).
- **Round 1:** 17 neighbours. No improvement.
- **Round 2:** 14 neighbours. No improvement → **stopped: plateau** (53 configurations, budget 120).

**Best configuration of each exit type** (`peak/exits/family_summary.txt`), worst‑era Sharpe, rank out of 53:
- **Fixed hold 5 days: −0.75 (rank 1).** Hold 7: −0.96. Hold 21 (baseline): −1.08.
- Profit targets: best is hold 15 + 30% target at −1.05 (rank 3). Targets are roughly neutral.
- Signal exits: `prev_high` −1.29, `ma5` + thesis −1.31.
- Stop‑loss: 5‑day hold + 10% stop −1.41. Trailing stop: 20% −1.58 (rank 28).

**Diagnosis:** every protective exit (stops, trailing, thesis, turn‑up signals) is whipsawed by IPO volatility, the same finding as phase 1. The only thing that helps in bad markets is less time in the trade.

## Final check (`peak/exits/final.py`; dry run first, then one run; lock `VALIDATION_RUN.lock`)
| era | 5‑day hold: CAGR / Sharpe / DD / mean trade | 21‑day hold: CAGR / Sharpe / DD / mean trade |
|---|---|---|
| 2015–18 | −2.3% / −0.45 / −15.4% / −0.7% | −10.4% / −1.08 / −39.9% / −5.2% |
| 2019–21 | −7.3% / −0.75 / −27.9% / −1.4% | −20.3% / −0.98 / −53.6% / −3.6% |
| 2021–24 | +1.6% / 0.38 / −3.9% / +1.8% | +4.5% / 0.56 / −9.6% / +5.1% |
| **2024–26 (validation, run once)** | **+3.4% / 0.56 / −6.6% / +0.9%** | **+15.6% / 1.32 / −7.6% / +5.5%** |

- **The searched exit is NOT better than the 21‑day hold on validation**, by the pre‑set rule. Deflated Sharpe 0.33; at 100% borrow the 5‑day hold turns negative.
- **Hindsight ceiling:** covering at the lowest low within 21 days would earn 12.6–21.6% gross per trade depending on the era. The 5‑day hold captures 7–13% of that in good eras; the 21‑day hold captures 26–35%. No causal exit rule came close.
- Hand audit: 5/5 validation trades (SPCX, MENS, KARD, CRCL, MNTN) match the raw price file.

## Conclusion
**There's no single optimal exit. The best exit depends on the IPO‑market regime:**
- when IPOs are fading (2021–26), holding the full 21 days is clearly best;
- when they're rising (2015–21), the shorter the better, and even 5 days loses.

A search that demands robustness across eras picks the defensive 5‑day hold and gives up most of the good‑regime profit. So the exit can't substitute for knowing the regime. The missing piece is a **regime filter** that decides when to run the strategy at all; given one, the 21‑day hold is the better exit.


---
# Phase 5: sector‑specific exits (entry fixed = the phase‑3 rule)

**Question (user):** would exits chosen separately for each sector work better?

**Sector labels** (`fetch_sectors.py` → `data/full/sectors.csv`, raw `data/raw_profiles/`):
- Source: Nasdaq company profiles, i.e. today's classification, not the one at IPO. It's a static attribute, so there's no price look‑ahead.
- **Deviation from the plan:** profiles take ~2.5 s each, so only the **422 IPOs actually traded** by the entry rule were fetched (0 errors). The analysis only needs those. The script can finish the rest from the cache.
- Raw counts: Health Care 136, Technology 109, Consumer 70, Unknown (blank, mostly foreign issuers) 41, Finance 37, Industrial & Resources 29.
- **Buckets used**, with search‑era trades: Health Care 115, Technology 99, **Other** 69, Consumer 58. Other = Unknown + Finance + Industrial & Resources, merged because each had < 25 trades. The mapping was fixed before any exit was scored.

**Engine:** per‑trade exit overrides (`EXIT_FIELDS` on entry rows).
- Tests: 2 new, including a hand‑checked gap fill on a target.
- A test expectation of mine was wrong (a 12% target on a day that opened below it fills at the better open, 85, not 88); the engine was right.
- Regression: phase 1 and phase 4 results unchanged. Full suite **54/54**.

## Analysis (`peak/sectors/sector_exits.py`; 10‑exit menu; score = era‑balanced mean net per trade on 2015–24)
- **Global best exit:** 5‑day hold (−0.10%/trade).
- **Per‑sector picks:** Consumer 10 days, Health Care 5 days, Technology 7 days, Other 15 days. In‑sample +0.23%/trade.
- **Leave‑one‑era‑out** (`loeo.csv`): choose on two eras, test on the third.

| held‑out era | global exit | global | sector exits | difference |
|---|---|---|---|---|
| 2015–18 | hold 21 | −5.19% | −4.89% | +0.30 |
| 2019–21 | hold 5 | −1.43% | −2.77% | −1.34 |
| 2021–24 | hold 5 | +1.83% | +0.70% | −1.13 |

  Sector exits **lose 0.72%/trade** on average out of sample.
- **Shuffled sector labels, 500 runs** (`shuffle.json`):
  - in‑sample gain +0.33% is beaten by 88% of random labellings (p = 0.88);
  - out‑of‑sample gain −0.72% vs shuffled median −0.08% (p = 0.86).
  - **The real sectors carry no information about the best exit.**

## Validation 2024–26 (run once, `validation.json`; full portfolio via the engine)
| exits | trades | CAGR | Sharpe | max DD | mean trade |
|---|---|---|---|---|---|
| sector‑specific | 81 | +12.2% | 1.33 | −6.8% | +3.9% |
| global 21‑day | 81 | +15.6% | 1.32 | −7.6% | +5.5% |
| global 5‑day (phase‑4 pick) | 81 | +3.4% | 0.56 | −6.6% | +0.9% |

By sector, on validation, with sector exits: Technology +9.4%, Consumer +8.3%, Other +3.0%, Health Care +0.5% per trade.

## Conclusion
**Sector‑specific exits don't help.** They fail all three checks:
- out of sample they're worse than one global exit;
- they do no better than randomly assigned "sectors";
- on 2024–26 they tie the 21‑day hold's Sharpe with lower returns.

Splitting ~340 trades into 4 groups mostly adds noise. The per‑sector choices (5 / 7 / 10 / 15 days) are artifacts of which few trades fell in each group. The one sector pattern in validation (Health Care shorts earning ~0 while Tech and Consumer earn 8–9%) is interesting, but it comes from 81 trades and one period, so it's a hypothesis to test, not a result.


---
# Phase 6: what happens around IPO lock‑up expiry?

**Question (user):** check trends around the end of the initial investor lock‑up. When it ends, where does the trend go?

**Data:**
- **Lock‑up terms** (`fetch_lockups.py` → `data/full/lockups.csv`, raw `data/raw_deals/`): Nasdaq's IPO deal overview gives the actual lock‑up length, expiration date, and shares outstanding/offered for all 1,017 clean IPOs (0 errors). 1,007 have a date; **999 are the standard 180 days.**
- **Overhang** = (shares outstanding − shares offered) ÷ shares offered: locked‑up shares per share sold in the IPO.

**Method** (`lockup/lockup_study.py`, `lockup/contrasts.py`):
- **Event day:** the first trading day on/after the actual expiration date. It's median trading day 124.
- **Returns:** vs SPY, close to close, window τ = −70 … +40.
- **925 events.** Excluded: 67 expiries beyond the price data (recent IPOs), 15 without enough bars, 10 without a date.
- **Placebo:** the same window 45 trading days earlier in the same stock. The paired test (event − placebo, per stock) separates a lock‑up effect from normal IPO drift.

## Results (all IPOs)
| window | event mean | placebo mean | difference | significance |
|---|---|---|---|---|
| 20 days before (τ −21 → −1) | −3.0% (median −4.1%, 61% negative) | −1.3% | −1.7% | not significant (p = 0.11, CI −4.1 to +0.8) |
| expiry day (τ −1 → 0) | +0.05% | +0.1% | −0.05% | none |
| first week after (0 → +5) | +0.7% (median −0.5%) | −0.8% | +1.5% | not significant |
| days 5–20 after | −0.6% | −0.5% | 0.0% | none |
| days 20–40 after | −1.2% | −3.2% | **+2.0%** | Wilcoxon p < 0.001; CI −0.2 to +4.2 |

- **Volume:** only 1.20× the pre‑event baseline on expiry day, and 1.22× over the first week.
- **Path (median, cumulative vs SPY):** −4.9% (τ −70 → −45), −4.4% (−45 → −21), −4.1% (−21 → −1), then −1.8% (−1 → +20) and −2.7% (+20 → +40). **The slide was already under way; it slows after the date.**

## By group (paired "around expiry" τ −1 → +20, event minus placebo)
- **Deal size:**
  - large: **+4.7%** (p < 0.001), post‑40 +3.1% (p < 0.001), a relief bounce;
  - mid: −0.2%;
  - small: −3.5% (not significant).
- **Overhang** (unlocked supply): high −1.8% (not significant), mid +1.3% (p = 0.04), low +1.3% (not significant).
- **Price vs offer, measured just before expiry:** apparently a strong reversal (below offer +6.2%, 50%+ above −10.8%, both p < 0.01). **Artifact:** the grouping uses prices from the placebo window.
- **Price vs offer measured 70 days before** (before both windows): below offer +0.2%, 0–50% above −0.3%, 50%+ above +1.5%, all not significant. **The reversal disappears.**

## Conclusion
- **The lock‑up expiry is mostly priced in.** Nothing happens on the day, and volume barely rises.
- **IPO stocks are usually already falling into the expiry**, about −4% vs SPY in each of the three months before. Only a small, statistically unclear part of the last month is extra pre‑expiry selling.
- **After the date, the trend doesn't accelerate down. It slows**, and stocks do about 2% better than their usual drift over days 20–40.
- **Large IPOs get a relief bounce**; small, high‑unlock names lean weaker, without statistical support.
- **Caveats:**
  - survivorship (only still‑listed stocks);
  - early or staggered releases aren't captured;
  - not beta‑adjusted;
  - the "large‑deal bounce" is one of several subgroup tests and should be re‑tested before trading on it.
