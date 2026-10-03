"""Every strategy configuration tried, by round. Append-only: configurations are never edited or removed
after they have been run (trials.csv records the results). Each name is one configuration; hedged and
unhedged variants are separate configurations and both count toward the 60-configuration budget.
All configs: max 21 trading-day hold, no entry before day 3, base costs (25 bps/side, 30%/yr borrow)."""
from lib.engine import Config


def pair(name: str, **kw) -> list[Config]:
    """Unhedged and SPY-hedged versions of one idea."""
    return [Config(name=f"{name}", **kw), Config(name=f"{name}_H", hedge=True, **kw)]


ROUNDS: dict[str, list[Config]] = {}

# R1 baseline: short every tradable IPO at the day-3 open, hold 21 trading days.
ROUNDS["R1"] = pair("base_d3", signal="rules", params={"entry_day": 3})

# R2 entry timing: the analysis found the fade strongest after the first weeks (days 22-63).
ROUNDS["R2"] = (pair("entry_d6", signal="rules", params={"entry_day": 6})
                + pair("entry_d11", signal="rules", params={"entry_day": 11})
                + pair("entry_d22", signal="rules", params={"entry_day": 22}))

# R3 selection (train-only feature scan, diagnostics/feature_scan_d*.txt): hot IPO market and a volatile
# first day were the only features with a consistent positive link to short returns at entry days 3/6/11.
# Thresholds = train top-tercile cutoffs (hot >= 31 IPOs in prior 60d; day-1 range >= 25.7%) or medians
# (25; 18.6%) when the two are combined, to keep enough trades.
ROUNDS["R3"] = (pair("hot31_d6", signal="rules", params={"entry_day": 6, "hot_count_min": 31})
                + pair("rng26_d6", signal="rules", params={"entry_day": 6, "d1_range_min": 0.257})
                + pair("hot25_rng19_d6", signal="rules", params={"entry_day": 6, "hot_count_min": 25, "d1_range_min": 0.186})
                + pair("hot25_rng19_d11", signal="rules", params={"entry_day": 11, "hot_count_min": 25, "d1_range_min": 0.186}))

# R4 breakdown triggers: short only after the stock shows weakness (first qualifying day in days 3-42),
# instead of fading strength - aims to avoid squeezes and the costly high-momentum names.
ROUNDS["R4"] = (pair("bd_d1low", signal="rules", params={"from_day": 2, "breakdown": "d1_low"})
                + pair("bd_low5", signal="rules", params={"from_day": 6, "breakdown": "low5"})
                + pair("bd_offer", signal="rules", params={"from_day": 2, "breakdown": "offer"}))

# R5 (R4 found breakdown shorts work in both periods, but capital sits idle: 12-29% average exposure).
#   bd_any       : breakdown below day-1 low OR below the offer price -> more trades
#   bd_any_hot25 : same, only when >= 25 IPOs priced in the prior 60 days (hot-market regime from R3 scan)
#   bd_any_h10   : same, hold 10 days: halves borrow cost and recycles capital faster
ROUNDS["R5"] = (pair("bd_any", signal="rules", params={"from_day": 2, "breakdown": ["d1_low", "offer"]})
                + pair("bd_any_hot25", signal="rules", params={"from_day": 2, "breakdown": ["d1_low", "offer"], "hot_count_min": 25})
                + pair("bd_any_h10", signal="rules", params={"from_day": 2, "breakdown": ["d1_low", "offer"]}, hold=10))

# R6 risk & sizing on the leader (bd_any_hot25_H). Hedged only (unhedged lost to hedged in all 14 pairs).
LEAD = dict(signal="rules", params={"from_day": 2, "breakdown": ["d1_low", "offer"], "hot_count_min": 25}, hedge=True)
ROUNDS["R6"] = [Config(name="lead_stop25_H", stop=0.25, **LEAD),
                Config(name="lead_stop40_H", stop=0.40, **LEAD),
                Config(name="lead_size12_H", size=0.12, max_positions=8, **LEAD),
                Config(name="bdo_hot25_H", signal="rules", params={"from_day": 2, "breakdown": "offer", "hot_count_min": 25}, hedge=True)]

# R7 signal quality on the leader (hedged):
#   lead_both_H    : require BOTH breakdowns (below day-1 low AND below offer) - a deeper, more certain break
#   lead_hotpop_H  : hot market measured by recent IPOs' median day-1 pop >= 10% instead of IPO count
#   lead_late_H    : ignore breakdowns before day 10 (underwriter price support/stabilization period)
ROUNDS["R7"] = [Config(name="lead_both_H", signal="rules", hedge=True,
                       params={"from_day": 2, "breakdown": ["d1_low", "offer"], "breakdown_all": True, "hot_count_min": 25}),
                Config(name="lead_hotpop_H", signal="rules", hedge=True,
                       params={"from_day": 2, "breakdown": ["d1_low", "offer"], "hot_pop_min": 0.10}),
                Config(name="lead_late_H", signal="rules", hedge=True,
                       params={"from_day": 10, "breakdown": ["d1_low", "offer"], "hot_count_min": 25})]

# R8 hedge instrument: IWM (small caps) instead of SPY - IPOs trade like small-cap growth, so the hedge
# should remove more common risk. Applied to the leader and to the runner-up.
ROUNDS["R8"] = [Config(name="lead_iwm_H", hedge_symbol="IWM", **LEAD),
                Config(name="hotpop_iwm_H", signal="rules", hedge=True, hedge_symbol="IWM",
                       params={"from_day": 2, "breakdown": ["d1_low", "offer"], "hot_pop_min": 0.10})]

# R9 (last planned round): risk-balanced sizing and a broader hot-market definition on the leader.
#   lead_rangesize_H : size scaled by 18.6% / day-1 range (train median range), clipped 0.5-1.5x -> smaller bets on wild names
#   lead_hot_or_H    : hot if >= 25 IPOs in prior 60 days OR recent median pop >= 10%
ROUNDS["R9"] = [Config(name="lead_rangesize_H", range_target=0.186, **LEAD),
                Config(name="lead_hot_or_H", signal="rules", hedge=True,
                       params={"from_day": 2, "breakdown": ["d1_low", "offer"], "hot_count_min": 25, "hot_pop_min": 0.10, "hot_mode": "or"})]
