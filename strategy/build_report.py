"""Build ipo_short_lab.html from the saved results (backtest_results/, trials.csv, data/, test_logs/)."""
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
R = HERE / "backtest_results"

score = json.loads((R / "scorecard.json").read_text(encoding="utf-8"))
trials = pd.read_csv(HERE / "trials.csv")
cov = pd.read_csv(HERE / "data" / "coverage.csv")
meta = pd.read_csv(HERE / "data" / "ipo_meta.csv", keep_default_na=False, na_values=[""])
r1_placebo = json.loads((HERE / "test_logs" / "placebo_random_established_R1_base_d3.json").read_text())
pytest_log = (HERE / "test_logs" / "pytest_engine_final.txt").read_text(encoding="utf-8", errors="ignore")
n_pass = pytest_log.count(" PASSED")
n_fail = pytest_log.count(" FAILED")

equity = {}
nulls = {}
for sp in ["train", "validation", "holdout"]:
    e = pd.read_csv(R / f"equity_{sp}.csv", index_col=0, parse_dates=True).iloc[:, 0]
    equity[sp] = {"d": [d.strftime("%Y-%m-%d") for d in e.index], "v": [round(v, 5) for v in e.values]}
    nulls[sp] = pd.read_csv(R / f"shuffled_null_{sp}.csv").sharpe.dropna().round(3).tolist()

cfg = score["config"]
H, V = score["holdout"]["metrics"], score["validation"]["metrics"]
data = {
    "score": score, "equity": equity, "nulls": nulls, "n_ipos": int(len(meta)),
    "trials": trials[["run_id", "hedge", "train_trades", "train_sharpe", "train_cagr", "validation_trades",
                      "validation_sharpe", "validation_cagr"]].to_dict("records"),
    "coverage": cov.to_dict("records"),
    "verdict_text": (
        f"The best of {score['n_configurations_tried']} configurations looked promising on 2021–2026 data "
        f"(Sharpe {score['train']['metrics']['sharpe']:.2f} train, {V['sharpe']:.2f} validation). On the untouched "
        f"2019–2021 holdout it lost {abs(H['total_return']):.0f}% (CAGR {H['cagr']:.1f}%, max drawdown {H['max_dd']:.0f}%). "
        "Random IPO shorts with the same timing did about as well as the strategy in every period, so its results came from the "
        "IPO‑market cycle, not from its rules: shorting any IPO worked in 2022–26 and failed in the 2020–21 frenzy. "
        "No consistent high‑return one‑month IPO short was found."),
    "rules": [
        "Universe: every newly listed US operating company (no SPACs or funds) with an offer price of $5 or more.",
        "Signal, checked at each close from day 2 to day 42: the stock closes <b>below its day‑1 low or below its IPO price</b>, "
        "and the IPO market is hot (<b>25 or more IPOs priced in the previous 60 days</b>). It must also be tradable: price ≥ $5 "
        "and median daily dollar volume ≥ $5M.",
        f"Entry: short at the next day's open. Size is {cfg['size']*100:.0f}% of equity, scaled by 18.6% ÷ (day‑1 high–low range) "
        "within 0.5–1.5×. Hedge with long SPY equal to beta × the short's size, using beta from pre‑entry data.",
        f"Exit: cover at the close of the {cfg['hold']}st trading day. No stop‑loss or profit target. At most {cfg['max_positions']} "
        "positions and 100% gross short exposure.",
        f"Costs: {cfg['slippage']*1e4:.0f} bps slippage per side, {cfg['borrow']*100:.0f}%/yr borrow fee charged daily, "
        "no interest earned on short proceeds.",
    ],
    "stress_note": ("Costs decide the sign. With no costs the validation Sharpe is "
                    f"{score['validation']['stress']['zero_cost']['sharpe']:.2f} and the holdout Sharpe is "
                    f"{score['holdout']['stress']['zero_cost']['sharpe']:.2f}. At borrow rates common for new IPOs "
                    "(60–100%+ a year), even the good periods lose money."),
    "trials_note": (f"{len(trials)} configurations over 9 rounds, all logged before the holdout was run. "
                    "A change was kept only if it improved train without hurting validation. The final pick was the highest "
                    "min(train, validation) Sharpe, a rule fixed before round 8. Several ideas (day‑1‑range filter, 10‑day hold, stops, "
                    "late breakdowns) looked strong on train and failed on validation: the overfitting the process was built to catch."),
    "tests": [
        [f"Engine unit and real‑data tests ({n_pass} passed, {n_fail} failed)", n_fail == 0 and n_pass > 0,
         "Exact costs, stop/gap/target fills, hedge leg, missing days, capacity limits, accounting identity."],
        ["No look‑ahead", True, "5 signal types on real data: 42–89 signals each unchanged when all future bars are replaced with random garbage."],
        ["Independent recomputation", True, "138 trades recomputed with separate vectorized code match the engine to 1e‑9."],
        ["Random‑stock placebo (baseline dates)", True,
         f"{r1_placebo['placebo_trades']} shorts of random non‑IPO stocks: gross {r1_placebo['mean_gross_short_ret_%']:+.2f}% "
         f"(t = {r1_placebo['gross_t_stat']}), net {r1_placebo['mean_net_ret_%']:+.2f}%, which is exactly −costs."],
        ["Random‑stock placebo (final strategy dates)", True,
         f"{score['placebo_random_established']['trades']} shorts: gross {score['placebo_random_established']['mean_gross_%']:+.2f}%, "
         f"net {score['placebo_random_established']['mean_net_%']:+.2f}%, which is gross minus costs as expected."],
        ["Spot audit from raw price file", all(a["match"] for a in score["spot_audit"]), "5 random trades recomputed by hand (table below)."],
        ["Regression after engine changes", True, "The leader's results were unchanged to the cent after adding IWM hedging and range sizing."],
    ],
    "limits": [
        "Survivorship: the Zeus database holds only stocks still trading. 342 sizeable IPOs with no price data and 47 with later reverse "
        "splits are missing. They are mostly failed companies, so this likely <i>understates</i> short profits, though not by enough to "
        "change a −51% drawdown.",
        "The holdout covers only Jan 2019 – Sep 2021 because the IPO lists start in 2019. It is one regime (the 2020–21 boom), and it "
        "was that regime that broke the strategy.",
        "Borrow cost is assumed flat at 30%/yr. Real rates for new IPOs are often far higher, and shares are sometimes not available to borrow at all.",
        "Daily bars only. Real stop fills, auctions and hard‑to‑borrow recalls are not modelled.",
        "IWM prices (used only in round 8) came from Nasdaq's API unadjusted for dividends; SPY came from the Zeus database.",
    ],
}
# ---------------------------------------------------------------- phase 2
tml = pd.read_csv(HERE / "trials_ml.csv")
sc_path = HERE / "backtest_results_ml" / "scorecard.json"
dry = json.loads((HERE / "backtest_results_ml" / "dry_run" / "scorecard.json").read_text(encoding="utf-8"))
ml = json.loads(sc_path.read_text(encoding="utf-8")) if sc_path.exists() else None
sel = dry["selected"]


def curve(run_id, name):
    e = pd.read_csv(HERE / "runs_ml" / run_id / "equity.csv", index_col=0, parse_dates=True).iloc[:, 0]
    e = e.iloc[::2]  # every other day is plenty for the chart
    return {"name": name, "d": [d.strftime("%Y-%m-%d") for d in e.index], "v": [round(v, 4) for v in e.values]}


trials_t = tml[tml.kind == "trial"].copy()
best_overall = trials_t.sort_values("sharpe", ascending=False).run_id.iloc[0]
curves = [curve(sel, "Selected model (logistic, all 28 indicators)"), curve(best_overall, "Best model by overall Sharpe"),
          curve("B_phase1_R9", "Phase‑1 rule"), curve("B_all_candidates", "Short every IPO")]
ic = pd.read_csv(HERE / "backtest_results_ml" / "dry_run" / "feature_ic_by_year.csv", index_col=0)
ic = ic.reindex(ic["mean"].abs().sort_values(ascending=False).index).head(12)
FNAME = {"log_dollar_vol": "Daily dollar volume", "d1_turnover": "Day‑1 turnover (volume ÷ shares offered)", "rvol": "Volatility since listing",
         "d1_oc": "Day‑1 open→close move", "d1_range": "Day‑1 high–low range", "vol_ratio": "Recent volume ÷ day‑1 volume",
         "hot_count": "IPOs priced in prior 60 days", "dd_from_high": "Drop from post‑IPO high", "ret5": "5‑day return",
         "spy_vol20": "SPY 20‑day volatility", "cohort_below_offer": "Share of recent IPOs below offer", "ext": "Price vs offer",
         "cohort_mom": "Recent IPOs' momentum", "pop": "Day‑1 pop", "log_deal": "Deal size", "below_d1low": "Below day‑1 low",
         "spy_ret20": "SPY 20‑day return", "spy_ret60": "SPY 60‑day return", "ret3": "3‑day return", "run": "Return since day‑1 close",
         "dist_d1low": "Distance to day‑1 low", "hot_pop": "Recent IPOs' median pop", "log_offer": "Offer price", "exch_nyse": "NYSE listing",
         "signal_day": "Days since listing", "below_offer": "Below offer price", "spy_above_ma50": "SPY above 50‑day average", "cohort_n": "Recent IPO count (90d)"}
hold = None
if ml and "fresh_holdout" in ml:
    H = ml["fresh_holdout"]; m = H["metrics"]
    hold = {"scorecard": H["scorecard"],
            "note": (f"{m['trades']} trades; CAGR {m['cagr']}%, Sharpe {m['sharpe']}, max drawdown {m['max_dd']}%. The model shorted "
                     f"{H['candidates_shorted_pct']}% of candidates; prediction‑vs‑outcome rank correlation {H['oos_ic']}. Random IPO shorts "
                     f"with the same timing: median Sharpe {H['null_sharpe_median']:.2f}. With zero costs: Sharpe {H['stress']['zero_cost']['sharpe']}.")}
# A bootstrap p < 0.05 only counts as a pass when the result is a significant GAIN (sharpe > 0). The raw scorecard
# scored a significant loss as "pass"; corrected here for display (noted in research_log.md).
for blk in [score[s] for s in ("train", "validation", "holdout")] + ([ml["fresh_holdout"], ml["dev_oos"]] if ml else []):
    sc = blk["scorecard"].get("Bootstrap p < 0.05")
    if sc and blk["metrics"]["sharpe"] <= 0:
        sc["pass"] = False
        sc["value"] = f"{sc['value']} (significant loss)" if sc["value"] < 0.05 else sc["value"]
if hold:
    hm = ml["fresh_holdout"]["metrics"]
    data["verdict_text"] = (
        f"Four phases of research ({score['n_configurations_tried'] + 12} strategies, 25 peak‑timed entry rules, 53 exits) across 2015–2026. "
        f"The best hand‑built rule looked good on 2021–26 and lost {abs(score['holdout']['metrics']['total_return']):.0f}% on the 2019–21 holdout. "
        f"A model using 28 indicators, trained only on past data, made no meaningful money out of sample in 2020–26 and lost "
        f"{abs(hm['cagr']):.1f}% a year on the untouched 2015–18 holdout (Sharpe {hm['sharpe']}), even before costs. "
        "Short‑term IPO returns are driven by the IPO‑market cycle, and borrow costs eat the small ranking skill that exists. "
        "A peak‑timed entry gets short measurably closer to the top, but no exit fixes the regime problem: the best exit depends on whether IPOs are fading or rising. "
        "<b>No consistent one‑month IPO short strategy was found.</b>")
data["p2"] = {
    "selected": sel,
    "intro": ("I built 28 indicators (19 about the IPO itself, 9 about market conditions, including the momentum of recently listed IPOs and "
              "the share trading below their offer price) and trained 12 models (logistic regression and boosted trees × three indicator sets "
              "× two decision thresholds). Each year from 2020 to 2026 was predicted by a model that had only seen earlier years. "
              "<b>No model made meaningful money out of sample.</b> The best earned +1.5%/yr with a −29% drawdown. The models did rank IPOs "
              "somewhat better than chance, but not enough to cover roughly 3% costs per trade. The market‑condition indicators were the "
              "worst: in 2020–21 the models had never seen an IPO frenzy, so they could not learn to avoid one."),
    "curves": curves,
    "trials": [{"run_id": r.run_id, "family": {"logit": "Logistic", "gbm": "Boosted trees"}[r.run_id.split("_")[1]],
                "fset": {"ipo": "IPO only (19)", "regime": "Market only (9)", "all": "All 28"}[r.run_id.split("_")[2]],
                "thr": r.run_id.split("_")[3], "trades": int(r.trades), "cagr": r.cagr, "sharpe": r.sharpe, "max_dd": r.max_dd,
                "worst": r.worst_year_sharpe, "ic": r.oos_ic_mean} for r in trials_t.itertuples()],
    "ic": [{"name": FNAME.get(f, f), "mean": float(r["mean"]), "agree": int(r["years_same_sign_as_mean"])} for f, r in ic.iterrows()],
    "holdout": hold,
}
# ---------------------------------------------------------------- phase 3
tp = pd.read_csv(HERE / "peak" / "trials_peak.csv")
bl = json.loads((HERE / "peak" / "baselines.json").read_text())
pnl3 = json.loads((HERE / "peak" / "top3_pnl.json").read_text())
selr = tp[tp.selected].iloc[0]
NAMES = {"A_m40_p5_k10": "Ran 40%+ above offer, then first close 5%+ below a high set in the last 10 days",
         "A_m20_p5_k10": "Ran 20%+ above offer, then first close 5%+ below a fresh high",
         "A_m20_p15_k5": "Ran 20%+ above offer, then first close 15%+ below a fresh high",
         "B_v0": "Reversal bar at the high (new high, closes near the low)",
         "B_v1.5": "Reversal bar at the high on 1.5x volume",
         "C_r20": "First down day after a 20%+ five‑day run",
         "D_n5": "Failed retest of a high at least 5 days old"}
rows3 = []
for rid, nm in NAMES.items():
    r = tp[tp.rule == rid].iloc[0]
    rows3.append({"name": nm, "sel": bool(r.selected), "fire": f"{r.train_fire_rate:.0f}% / {r.validation_fire_rate:.0f}%",
                  "tg": f"{r.train_gain_mean:+.1f} pts", "vg": f"{r.validation_gain_mean:+.1f} [{r.validation_gain_ci_lo:+.1f}, {r.validation_gain_ci_hi:+.1f}]",
                  "eff": f"{r.validation_eff_median:.0f}%"})
rows3.append({"name": "Fixed day 3 (baseline)", "sel": False, "fire": "all", "tg": f"{bl['train_fixed_d3_gain']:+.1f} pts",
              "vg": f"{bl['validation_fixed_d3_gain']:+.1f}", "eff": "~78%"})
rows3.append({"name": "Perfect hindsight (best open, days 3–42)", "sel": False, "fire": "all", "tg": "–", "vg": "–",
              "eff": f"{bl['validation_oracle_eff_median']:.0f}%"})
ps = pnl3[selr.rule]
data["p3"] = {
    "intro": ("The goal here was only the <b>entry</b>: short as close to the IPO's highest price (its peak over the first 63 trading days) as possible, "
              "using only information available that day. I tested 25 rules in four families, picked the best on 2021–24 data and checked it on 2024–26. "
              "The winner: <b>once an IPO has traded 40%+ above its offer price, short the next open after it first closes 5%+ below a high set within the "
              "last 10 days.</b> It gets short about 4–5 points of the peak price closer to the top than a random day, and that held up on the check period."),
    "tiles": [[f"{selr.validation_eff_median:.0f}%", "median share of the peak price captured (2024–26)"],
              [f"{selr.validation_gain_mean:+.1f} pts", "closer to the peak than a random day (2024–26)"],
              [f"{bl['validation_oracle_eff_median']:.0f}%", "best possible with perfect hindsight"],
              [f"{selr.validation_pct_within_10pct:.0f}%", "of entries within 10% of the true peak"]],
    "rules": rows3,
    "pnl": [{"p": "Train 2021–24", "n": ps["train"]["trades"], "cagr": ps["train"]["cagr"], "sh": ps["train"]["sharpe"], "dd": ps["train"]["max_dd"]},
            {"p": "Validation 2024–26", "n": ps["validation"]["trades"], "cagr": ps["validation"]["cagr"], "sh": ps["validation"]["sharpe"], "dd": ps["validation"]["max_dd"]},
            {"p": "2019–21 (used before)", "n": ps["holdout"]["trades"], "cagr": ps["holdout"]["cagr"], "sh": ps["holdout"]["sharpe"], "dd": ps["holdout"]["max_dd"]},
            {"p": "2015–18 (used before)", "n": ps["fresh_holdout"]["trades"], "cagr": ps["fresh_holdout"]["cagr"], "sh": ps["fresh_holdout"]["sharpe"], "dd": ps["fresh_holdout"]["max_dd"]}],
    "note": ("Two‑thirds of the entries fire on day 2, meaning IPOs that spiked 40%+ on day 1 and are already fading. About half the time the stock later "
             "makes a higher high, so the rule catches a local top, not always the final one. Hitting the exact top isn't realistic: only "
             f"{selr.validation_pct_within_10pct:.0f}% of entries land within 10% of it. The better entry lifts the average 21‑day trade to about +4.5–5%, "
             "but the old problem remains: profitable in 2022–26, heavy losses in 2015–21. <b>Next:</b> the exit, and a filter for the IPO‑market cycle."),
}
# ---------------------------------------------------------------- phase 4
f4 = json.loads((HERE / "peak" / "exits" / "final.json").read_text())
fam4 = pd.read_csv(HERE / "peak" / "exits" / "trials_exit.csv")
data["p4"] = {
    "intro": ("With the phase‑3 entry fixed, a self‑iterating search tried exits on its own: fixed holds, profit targets, stop‑losses, "
              "trailing stops, a stop at the high we shorted from, and 'cover when it turns back up' signals. It kept the best, generated "
              "variations around them, and repeated until nothing improved. It optimized the <b>worst</b> result across 2015–18, 2019–21 and "
              f"2021–24, so it couldn't win by fitting one period. It stopped after 3 rounds and {len(fam4)} exits. "
              "The winner: <b>simply cover after 5 trading days.</b>"),
    "fam": [{"name": "Fixed hold, 5 days", "obj": -0.748, "rank": 1},
            {"name": "Profit target (15‑day hold + 30% target)", "obj": -1.049, "rank": 3},
            {"name": "Fixed hold, 21 days (starting point)", "obj": -1.079, "rank": 5},
            {"name": "Cover after a close above the prior day's high (+5‑day hold)", "obj": -1.289, "rank": 17},
            {"name": "Cover after a close above the 5‑day average + stop at the prior high", "obj": -1.311, "rank": 18},
            {"name": "Stop‑loss 10% (+5‑day hold)", "obj": -1.413, "rank": 20},
            {"name": "Trailing stop 20%", "obj": -1.584, "rank": 28}],
    "eras": [{"era": k + (" (final check, run once)" if k == "2024-26" else ""), "val": k == "2024-26",
              "c": v["chosen"], "b": v["baseline_21d"]} for k, v in f4["eras"].items()],
    "note": ("On the final check the 5‑day hold <b>lost to the 21‑day hold</b> (Sharpe 0.56 vs 1.32), so it isn't the better exit. The best exit "
             "depends on the market: when IPOs are fading, holding the full 21 days wins; when they're rising, the shorter the better, and even "
             "5 days loses. Stops and trailing stops were the worst, because volatile IPOs bounce enough to trigger them before falling again. "
             "With perfect hindsight, covering at the lowest point within 21 days would earn 12–22% per trade; no rule using only past prices came close. "
             "<b>What's missing is a filter that decides when to run the strategy at all.</b>"),
}
# ---------------------------------------------------------------- phase 5
p5s = json.loads((HERE / "peak" / "sectors" / "summary.json").read_text())
p5v = json.loads((HERE / "peak" / "sectors" / "validation.json").read_text())
p5l = pd.read_csv(HERE / "peak" / "sectors" / "loeo.csv")
sh = p5s["shuffle"]
per = p5s["sector_exits"]
data["p5"] = {
    "intro": ("I labelled each traded IPO with its sector (Nasdaq classification) and grouped them into Health Care, Technology, Consumer and Other, "
              "so each group had enough trades. Then I picked the best exit for each group using 2015–24 data. The picks were: "
              + ", ".join(f"{k} {v.replace('hold', '')} days" for k, v in per.items()) +
              f". Real sectors should beat random groupings. They don't: randomly shuffled sector labels matched or beat the real ones "
              f"{sh['p_in_sample']*100:.0f}% of the time in‑sample and {sh['p_loeo']*100:.0f}% of the time out of sample."),
    "loeo": [{"era": r.held_out_era, "g": r["global_mean_%"], "s": r["sector_mean_%"], "d": r["sector_minus_global_%"]} for _, r in p5l.iterrows()],
    "val": [{"name": "Sector‑specific exits", "sel": True, "cagr": p5v["sector_specific"]["cagr"], "sharpe": p5v["sector_specific"]["sharpe"], "mt": p5v["sector_specific"]["mean_trade"]},
            {"name": "One exit: 21‑day hold", "sel": False, "cagr": p5v["global_21d"]["cagr"], "sharpe": p5v["global_21d"]["sharpe"], "mt": p5v["global_21d"]["mean_trade"]},
            {"name": "One exit: 5‑day hold", "sel": False, "cagr": p5v["global_5d"]["cagr"], "sharpe": p5v["global_5d"]["sharpe"], "mt": p5v["global_5d"]["mean_trade"]}],
    "note": (f"<b>Sector‑specific exits don't help.</b> Out of sample they averaged {p5s['loeo_mean_gain_%']:+.2f}% per trade versus one exit for all, "
             "and on 2024–26 they tied the 21‑day hold's Sharpe with a lower return. Splitting a few hundred trades four ways mostly adds noise. "
             "One pattern worth a later look: on 2024–26, Health Care shorts earned about 0% per trade while Technology and Consumer earned 8–9%. "
             "That's a single period and 81 trades, so it's a lead, not a finding."),
}
# ---------------------------------------------------------------- best strategy (top of page)
bs = json.loads((HERE / "peak" / "best_strategy.json").read_text(encoding="utf-8"))
be = pd.read_csv(HERE / "peak" / "best_equity_full.csv", index_col=0, parse_dates=True).iloc[:, 0].iloc[::2]
rec = next(e for e in bs["eras"] if e["era"] == "2024–26")
data["best"] = {
    "name": "Peak‑fade short: short the first breakdown after a 40%+ IPO run, cover after 21 days",
    "rules": [
        "<b>Universe:</b> newly listed US operating companies (no SPACs or funds), offer price ≥ $5, price ≥ $5 and ≥ $5M daily dollar volume.",
        "<b>Entry:</b> once the stock has traded at least <b>40% above its IPO price</b>, short at the next open after the <b>first close at least 5% below</b> "
        "a high it set within the last 10 trading days. Checked from day 2 to day 41 after listing; one trade per IPO.",
        "<b>Exit:</b> cover at the close of the <b>21st trading day</b>. No stop‑loss or profit target: every alternative exit tested did worse.",
        "<b>Sizing and hedge:</b> 7% of equity per trade, scaled down for IPOs with a wild first day. Hedge each short with long SPY sized to the stock's "
        "beta. At most 15 positions and 100% gross short.",
        "<b>Costs assumed:</b> 0.25% slippage each way and a 30%/yr borrow fee (new IPOs often cost more to borrow).",
    ],
    "flag": ("<b>Read the full history, not just the recent years.</b> This is the best of everything tested, not a strategy that works. It made money in "
             "2022, 2025 and 2026, when IPOs were fading, and lost in most other years. Run continuously from 2015 to 2026 it lost about half its capital. "
             "Its 2024–26 numbers look strong, but those were the years used to confirm it, and the same rules lost heavily in 2015–21. "
             "It fails the pass/fail bar set before testing began."),
    "full": bs["full"], "recent": rec, "yearly": {str(k): v for k, v in bs["yearly"].items()},
    "eras": bs["eras"], "stress": bs["stress"],
    "equity": {"d": [d.strftime("%Y-%m-%d") for d in be.index], "v": [round(float(v), 4) for v in be.values]},
}
# ---------------------------------------------------------------- phase 6: lock-up expiry
lp = pd.read_csv(HERE / "lockup" / "event_paths.csv")
lc = json.loads((HERE / "lockup" / "contrasts.json").read_text())
lr = json.loads((HERE / "lockup" / "results.json").read_text())
base_mean, base_med = lp.loc[lp.tau == -1, "car_mean"].iloc[0], lp.loc[lp.tau == -1, "car_median"].iloc[0]
def sigtxt(r):
    lo, hi, pv = r.get("ci_lo_%"), r.get("ci_hi_%"), r["p"]
    if pv < 0.01 and lo is not None and (lo > 0 or hi < 0):
        return "Yes (strong)"
    if pv < 0.05:
        return "Partly (p < 0.05, wide range)"
    return "No"
A = lc["all IPOs"]
WN = [("20 days before", "pre"), ("Expiry day", "event"), ("First week after", "post5"), ("Days 5–20 after", "post20"), ("Days 20–40 after", "post40")]
G = [("Large deals", "deal size", "large"), ("Mid deals", "deal size", "mid"), ("Small deals", "deal size", "small"),
     ("Most stock unlocked (high overhang)", "overhang", "high"), ("Least stock unlocked", "overhang", "low"),
     ("Below offer 70 days earlier", "price vs offer 70 days before (clean)", "below offer"),
     ("50%+ above offer 70 days earlier", "price vs offer 70 days before (clean)", "50%+ above")]
data["p6"] = {
    "intro": (f"Insiders and early investors usually can't sell for 180 days after an IPO. Nasdaq's deal records give the actual expiration date "
              f"for {lr['n_events']} of our IPOs (999 of 1,007 are the standard 180 days). I tracked each stock against SPY from 70 trading days before "
              "expiry to 40 after, and compared every window with the same stock 45 days earlier to separate a lock‑up effect from the normal IPO drift. "
              "<b>The expiry is mostly priced in:</b> nothing happens on the day, and the slide that was already under way slows afterwards rather than speeding up."),
    "tau": lp.tau.tolist(), "mean": ((lp.car_mean - base_mean) * 100).round(2).tolist(),
    "median": ((lp.car_median - base_med) * 100).round(2).tolist(),
    "windows": [{"name": n, "ev": A[k]["event_mean_%"], "pl": A[k]["placebo_mean_%"], "d": A[k]["diff_mean_%"], "sig": sigtxt(A[k])} for n, k in WN],
    "groups": [{"name": n, "n": lc[a][b]["around"]["n"], "d": lc[a][b]["around"]["diff_mean_%"], "sig": sigtxt(lc[a][b]["around"])} for n, a, b in G],
    "note": (f"Trading volume rises only {lr['volume']['expiry_day_median_ratio']}× on expiry day. Large IPOs see a relief bounce of about +4.7% versus their own "
             "trend; small, heavily unlocked names lean weaker but not reliably. An apparent 'winners fall, losers bounce' pattern turned out to be an artifact "
             "of how the groups were formed and disappears when stocks are grouped by their price before the test windows. Caveats: only still‑listed stocks, "
             "early releases aren't captured, and returns aren't beta‑adjusted."),
}
html = (HERE / "report_template.html").read_text(encoding="utf-8").replace("/*__DATA__*/null", json.dumps(data, default=str))
(HERE / "ipo_short_lab.html").write_text(html, encoding="utf-8")
print("wrote ipo_short_lab.html", len(html) // 1024, "KB")
