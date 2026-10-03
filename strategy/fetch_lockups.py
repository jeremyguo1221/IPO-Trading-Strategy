"""Lock-up data for every clean IPO from Nasdaq's IPO deal overview (api.nasdaq.com/api/ipo/overview/?dealId=...).

Raw JSON cached in data/raw_deals/<dealID>.json (re-runs only fetch what's missing).
Output: data/full/lockups.csv with symbol, dealID, lockup_days, lockup_exp (date), quiet_exp, shares_outstanding,
        shares_offered, overhang = (shares_outstanding - shares_offered) / shares_offered
        (locked-up shares per share sold in the IPO: a rough measure of the supply the expiry can unleash)
"""
import json
import time
import urllib.request
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
RAW = HERE / "data" / "raw_deals"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36"


def fetch(deal: str) -> dict:
    f = RAW / f"{deal}.json"
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8"))
    req = urllib.request.Request(f"https://api.nasdaq.com/api/ipo/overview/?dealId={deal}",
                                 headers={"User-Agent": UA, "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            raw = r.read().decode("utf-8")
    except Exception as e:
        return {"error": str(e)}
    f.write_text(raw, encoding="utf-8")
    time.sleep(1.0)
    return json.loads(raw)


def num(v):
    try:
        return float(str(v).replace(",", "").replace("$", ""))
    except (TypeError, ValueError):
        return None


def parse(js: dict) -> dict:
    o = ((js or {}).get("data") or {}).get("poOverview") or {}
    val = lambda k: (o.get(k) or {}).get("value")
    exp = pd.to_datetime(val("LockupPeriodExpirationDate"), format="%m/%d/%Y", errors="coerce")
    qp = pd.to_datetime(val("QuietPeriodExpirationDate"), format="%m/%d/%Y", errors="coerce")
    so, sf = num(val("SharesOutstanding")), num(val("SharesOffered"))
    return {"lockup_days": num(val("LockupPeriodNumberofDays")), "lockup_exp": None if pd.isna(exp) else exp.date(),
            "quiet_exp": None if pd.isna(qp) else qp.date(), "shares_outstanding": so, "shares_offered": sf,
            "overhang": (so - sf) / sf if so and sf and so > sf else None}


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(HERE / "data" / "full" / "ipo_meta.csv", keep_default_na=False, na_values=[""], parse_dates=["priced_date"])
    raw = pd.read_csv(HERE / "data" / "full" / "ipo_list_raw.csv", keep_default_na=False, na_values=[""], parse_dates=["priced_date"])
    m = meta.merge(raw[["symbol", "priced_date", "dealID"]], on=["symbol", "priced_date"], how="left")
    rows, errors = [], 0
    for i, r in enumerate(m.itertuples()):
        js = fetch(r.dealID) if isinstance(r.dealID, str) else {"error": "no dealID"}
        errors += "error" in js
        rows.append({"symbol": r.symbol, "dealID": r.dealID, "first_date": r.first_date, **parse(js)})
        if i % 100 == 0:
            print(f"{i}/{len(m)}", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(HERE / "data" / "full" / "lockups.csv", index=False)
    print(f"done: {len(out)} IPOs, {errors} errors, lockup date present for {out.lockup_exp.notna().sum()}")
    print(out.lockup_days.value_counts().head(10).to_string())


if __name__ == "__main__":
    main()
