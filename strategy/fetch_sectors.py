"""Phase 5: sector labels for every clean IPO, from Nasdaq's company-profile API (current classification).

Raw JSON cached in data/raw_profiles/<SYMBOL>.json (re-runs only fetch what is missing).
Output: data/full/sectors.csv  (symbol, nasdaq_sector, industry, bucket)
Buckets (fixed before any exit is evaluated):
  Technology           <- Technology, Telecommunications
  Health Care          <- Health Care
  Finance              <- Finance, Real Estate
  Consumer             <- Consumer Discretionary, Consumer Staples
  Industrial & Resources <- Industrials, Energy, Basic Materials, Utilities, Miscellaneous
  Unknown              <- no profile / blank sector
"""
import json
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
RAW = HERE / "data" / "raw_profiles"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36"
BUCKET = {"Technology": "Technology", "Telecommunications": "Technology", "Health Care": "Health Care",
          "Finance": "Finance", "Real Estate": "Finance", "Consumer Discretionary": "Consumer",
          "Consumer Staples": "Consumer", "Industrials": "Industrial & Resources", "Energy": "Industrial & Resources",
          "Basic Materials": "Industrial & Resources", "Utilities": "Industrial & Resources",
          "Miscellaneous": "Industrial & Resources"}


def parse(js: dict) -> tuple[str | None, str | None]:
    d = (js or {}).get("data") or {}
    sec = ((d.get("Sector") or {}).get("value") or "").strip() or None
    ind = ((d.get("Industry") or {}).get("value") or "").strip() or None
    return sec, ind


def fetch(sym: str) -> dict:
    f = RAW / f"{sym}.json"
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8"))
    req = urllib.request.Request(f"https://api.nasdaq.com/api/company/{sym}/company-profile",
                                 headers={"User-Agent": UA, "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            raw = r.read().decode("utf-8")
    except Exception as e:  # record the failure so it is visible, retry on the next run
        return {"error": str(e)}
    f.write_text(raw, encoding="utf-8")
    time.sleep(1.0)
    return json.loads(raw)


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(HERE / "data" / "full" / "ipo_meta.csv", keep_default_na=False, na_values=[""])
    # symbols actually traded by the phase-3 entry come first (they are all the exit analysis needs)
    traded = pd.concat([pd.read_csv(f, keep_default_na=False, na_values=[""]).symbol for f in (HERE / "peak" / "exits").glob("entries_*.csv")])
    order = list(dict.fromkeys(list(traded) + list(meta.symbol)))
    if "--traded-only" in sys.argv:
        order = list(dict.fromkeys(traded))
    rows, errors = [], 0
    for i, sym in enumerate(order):
        js = fetch(sym)
        errors += "error" in js
        sec, ind = parse(js)
        rows.append({"symbol": sym, "nasdaq_sector": sec, "industry": ind, "bucket": BUCKET.get(sec, "Unknown")})
        if i % 100 == 0:
            print(f"{i}/{len(order)}", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(HERE / "data" / "full" / "sectors.csv", index=False)
    print(f"done: {len(out)} symbols, {errors} request errors")
    print(out.bucket.value_counts().to_string())
    print(out.nasdaq_sector.value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
