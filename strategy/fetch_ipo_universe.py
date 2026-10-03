"""Build the IPO universe (Jan 2019 - Sep 2026).

Primary source: Nasdaq IPO calendar API, one JSON per month (priced deals on NYSE/Nasdaq/NYSE American),
with offer price and deal size. Raw JSON saved under data/raw_nasdaq/.
Cross-check: stockanalysis.com yearly lists (data/raw_html/, fetched earlier; capped at 500 rows/year,
so 2021 is incomplete there) - used only to report coverage agreement.

Outputs:
  data/ipo_list_raw.csv   every priced deal as listed
  data/ipo_universe.csv   operating-company IPOs only
  data/exclusions.csv     every dropped row with its reason (stage = "universe")
Run with --offline to re-parse saved files without downloading.
"""
import argparse
import json
import re
import time
import urllib.request
from io import StringIO
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
RAW_N = HERE / "data" / "raw_nasdaq"
RAW_S = HERE / "data" / "raw_html"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36"

SPAC_NAME = re.compile(r"acquisition|merger|blank check|\bspac\b|\bcorp(oration)?\.? ?(i|ii|iii|iv|v|vi|vii|viii|ix|x)\b"
                       r"|growth corp|opportunit(?:y|ies) corp|capital corp\b", re.I)
FUND_NAME = re.compile(r"\bfund\b|closed-end|\betf\b|\bincome trust\b|\bmunicipal\b", re.I)
UNIT_TICKER = re.compile(r"^[A-Z]{3,5}U$")


def fetch_month(p: pd.Period) -> dict:
    path = RAW_N / f"{p}.json"
    req = urllib.request.Request(f"https://api.nasdaq.com/api/ipo/calendar?date={p}",
                                 headers={"User-Agent": UA, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        raw = r.read().decode("utf-8")
    path.write_text(raw, encoding="utf-8")
    return json.loads(raw)


def rows_from(js: dict) -> list[dict]:
    pr = (js.get("data") or {}).get("priced") or {}
    return pr.get("rows") or []


def num(s):
    return pd.to_numeric(pd.Series(s).astype(str).str.replace(r"[$,]", "", regex=True), errors="coerce")


def main(offline: bool, start: str = "2019-01", out: Path = HERE / "data") -> None:
    """start: first month to include; out: output folder (phase 2 uses data/full so phase-1 files stay unchanged)."""
    out.mkdir(parents=True, exist_ok=True)
    RAW_N.mkdir(parents=True, exist_ok=True)
    recs = []
    for p in pd.period_range(start, "2026-09", freq="M"):
        f = RAW_N / f"{p}.json"
        js = json.loads(f.read_text(encoding="utf-8")) if offline or f.exists() else fetch_month(p)
        rows = rows_from(js)
        recs += rows
        if not offline and not f.exists():
            time.sleep(1.0)
    raw = pd.DataFrame(recs).rename(columns={"proposedTickerSymbol": "symbol", "companyName": "name",
        "proposedExchange": "exchange", "proposedSharePrice": "offer_price", "sharesOffered": "shares",
        "pricedDate": "priced_date", "dollarValueOfSharesOffered": "deal_size"})
    raw["priced_date"] = pd.to_datetime(raw.priced_date, format="%m/%d/%Y")
    raw["offer_price"] = num(raw.offer_price).values
    raw["shares"] = num(raw.shares).values
    raw["deal_size"] = num(raw.deal_size).values
    raw = raw.drop_duplicates(["symbol", "priced_date"]).sort_values("priced_date").reset_index(drop=True)
    raw = raw[["priced_date", "symbol", "name", "exchange", "offer_price", "shares", "deal_size", "dealID"]]
    raw.to_csv(out / "ipo_list_raw.csv", index=False)
    print(f"Nasdaq calendar: {len(raw)} priced deals, {raw.priced_date.min().date()} .. {raw.priced_date.max().date()}")

    reason = pd.Series("", index=raw.index)
    def mark(mask, why):
        reason[(reason == "") & mask] = why
    mark(raw.symbol.isna() | raw.offer_price.isna(), "missing symbol or price")
    mark(raw.name.str.contains(SPAC_NAME), "SPAC (name)")
    mark(raw.symbol.fillna("").str.match(UNIT_TICKER) & (raw.offer_price == 10.0), "SPAC (unit ticker, $10.00)")
    mark(raw.name.str.contains(FUND_NAME), "fund / trust vehicle")
    raw["drop_reason"] = reason
    excl = raw[reason != ""].assign(stage="universe")
    uni = raw[reason == ""].drop(columns="drop_reason").reset_index(drop=True)
    excl.to_csv(out / "exclusions.csv", index=False)
    uni.to_csv(out / "ipo_universe.csv", index=False)
    print(f"excluded {len(excl)}  universe {len(uni)}")
    print(excl.drop_reason.value_counts().to_string())
    print("universe by year:\n" + uni.groupby(uni.priced_date.dt.year).size().to_string())

    # cross-check against stockanalysis yearly lists (2019+, capped at 500 rows per year)
    sa = []
    for f in sorted(RAW_S.glob("ipos_*.html")):
        t = max(pd.read_html(StringIO(f.read_text(encoding="utf-8"))), key=len)
        sa.append(t.rename(columns={"Symbol": "symbol", "IPO Date": "d"})[["symbol", "d"]])
    if sa:
        sa = pd.concat(sa)
        sa["d"] = pd.to_datetime(sa.d, format="%b %d, %Y")
        both = uni.merge(sa, on="symbol", how="left")
        both["gap_days"] = (both.d - both.priced_date).dt.days
        yr = both.priced_date.dt.year
        cov = both.assign(found=both.d.notna()).groupby(yr).found.mean().round(3)
        print("share of universe also in stockanalysis lists, by year:\n" + cov.to_string())
        print("date gap (listing date - pricing date) among matches:", both.gap_days.describe()[["50%", "min", "max"]].to_dict())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--start", default="2019-01")
    ap.add_argument("--out", default=str(HERE / "data"))
    a = ap.parse_args()
    main(a.offline, a.start, Path(a.out))
