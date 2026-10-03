"""Phase 2: one-time, read-only export for the 2015-2018 fresh holdout, into data/full/ (phase-1 files untouched).

Writes to data/full/:
  ipo_bars_2015_2018.csv  bars for universe IPOs priced before 2019 (same query shape as export_prices.py)
  spy.csv                 SPY daily bars from 2014-12-01 (the full history the database has)
  placebo_bars.csv        bars 2015-01-01 onward for 300 random symbols already trading at the database start
  ipo_bars.csv            = phase-1 data/ipo_bars.csv + ipo_bars_2015_2018.csv (combined, de-duplicated)
  export_manifest.json
"""
import json
import subprocess
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
FULL = HERE / "data" / "full"
PSQL = ["docker", "exec", "-i", "zeus-postgres", "psql", "-U", "zeus", "-d", "zeus_db", "-v", "ON_ERROR_STOP=1", "-q"]


def copy_out(sql: str, out: Path) -> int:
    with open(out, "wb") as fh:
        subprocess.run(PSQL, input=f"COPY ({sql}) TO STDOUT WITH CSV HEADER;".encode(), stdout=fh, check=True)
    return sum(1 for _ in open(out, encoding="utf-8")) - 1


def main() -> None:
    uni = pd.read_csv(FULL / "ipo_universe.csv", parse_dates=["priced_date"], keep_default_na=False, na_values=[""])
    new = uni[uni.priced_date < "2019-01-01"]
    vals = ",".join(f"('{s.replace(chr(39), '')}', DATE '{d.date()}')" for s, d in zip(new.symbol, new.priced_date))
    ipo_sql = f"""
      WITH u(symbol, d) AS (VALUES {vals})
      SELECT o.symbol, o.ts::date AS date, o.open, o.high, o.low, o.close, o.volume, o.source
      FROM ohlcv_daily o JOIN u ON u.symbol = o.symbol
      WHERE o.ts >= u.d - INTERVAL '30 days' AND o.ts < u.d + INTERVAL '260 days'
      ORDER BY o.symbol, o.ts"""
    spy_sql = "SELECT ts::date AS date, open, high, low, close, volume FROM ohlcv_daily WHERE symbol='SPY' AND ts >= '2014-12-01' ORDER BY ts"
    placebo_sql = """
      WITH est AS (
        SELECT symbol FROM ohlcv_daily GROUP BY symbol
        HAVING min(ts) < '2015-01-05' AND max(ts) >= '2026-07-01' AND symbol <> 'SPY'),
      pick AS (SELECT symbol FROM est ORDER BY md5(symbol || 'ipo-placebo-seed-2') LIMIT 300)
      SELECT o.symbol, o.ts::date AS date, o.open, o.high, o.low, o.close, o.volume
      FROM ohlcv_daily o JOIN pick USING (symbol) WHERE o.ts >= '2015-01-01' ORDER BY o.symbol, o.ts"""
    man = {}
    for name, sql in [("ipo_bars_2015_2018.csv", ipo_sql), ("spy.csv", spy_sql), ("placebo_bars.csv", placebo_sql)]:
        n = copy_out(sql, FULL / name)
        man[name] = {"rows": n}
        print(f"{name}: {n:,} rows")
    rd = lambda p: pd.read_csv(p, keep_default_na=False, na_values=[""])
    comb = pd.concat([rd(HERE / "data" / "ipo_bars.csv"), rd(FULL / "ipo_bars_2015_2018.csv")]).drop_duplicates(["symbol", "date"])
    comb.sort_values(["symbol", "date"]).to_csv(FULL / "ipo_bars.csv", index=False)
    man["ipo_bars.csv"] = {"rows": int(len(comb)), "note": "phase-1 ipo_bars.csv + ipo_bars_2015_2018.csv"}
    (FULL / "export_manifest.json").write_text(json.dumps(man, indent=1), encoding="utf-8")
    print(f"combined ipo_bars.csv: {len(comb):,} rows")


if __name__ == "__main__":
    main()
