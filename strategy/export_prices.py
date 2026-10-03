"""One-time, read-only export of daily bars from the Zeus Postgres container into data/.

Exports (all SELECT-only via COPY ... TO STDOUT; nothing is written to the database):
  data/ipo_bars.csv      bars for each universe symbol from 30 days before pricing to 260 days after
  data/spy.csv           SPY daily bars 2018-10-01 onward
  data/placebo_bars.csv  bars 2019-01-01 onward for 300 randomly chosen established (listed before 2018) symbols
  data/export_manifest.json  row counts and query text, for the record
"""
import json
import subprocess
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
DATA = HERE / "data"
PSQL = ["docker", "exec", "-i", "zeus-postgres", "psql", "-U", "zeus", "-d", "zeus_db", "-v", "ON_ERROR_STOP=1", "-q"]


def copy_out(sql: str, out: Path) -> int:
    stmt = f"COPY ({sql}) TO STDOUT WITH CSV HEADER;"
    with open(out, "wb") as fh:
        subprocess.run(PSQL, input=stmt.encode(), stdout=fh, check=True)
    return sum(1 for _ in open(out, encoding="utf-8")) - 1


def main() -> None:
    uni = pd.read_csv(DATA / "ipo_universe.csv", parse_dates=["priced_date"], keep_default_na=False, na_values=[""])
    vals = ",".join(f"('{s.replace(chr(39), '')}', DATE '{d.date()}')" for s, d in zip(uni.symbol, uni.priced_date))
    ipo_sql = f"""
      WITH u(symbol, d) AS (VALUES {vals})
      SELECT o.symbol, o.ts::date AS date, o.open, o.high, o.low, o.close, o.volume, o.source
      FROM ohlcv_daily o JOIN u ON u.symbol = o.symbol
      WHERE o.ts >= u.d - INTERVAL '30 days' AND o.ts < u.d + INTERVAL '260 days'
      ORDER BY o.symbol, o.ts"""
    spy_sql = "SELECT ts::date AS date, open, high, low, close, volume FROM ohlcv_daily WHERE symbol='SPY' AND ts >= '2018-10-01' ORDER BY ts"
    placebo_sql = """
      WITH est AS (
        SELECT symbol FROM ohlcv_daily GROUP BY symbol
        HAVING min(ts) < '2018-01-01' AND max(ts) >= '2026-07-01' AND symbol <> 'SPY'),
      pick AS (SELECT symbol FROM est ORDER BY md5(symbol || 'ipo-placebo-seed-1') LIMIT 300)
      SELECT o.symbol, o.ts::date AS date, o.open, o.high, o.low, o.close, o.volume
      FROM ohlcv_daily o JOIN pick USING (symbol) WHERE o.ts >= '2019-01-01' ORDER BY o.symbol, o.ts"""
    manifest = {}
    for name, sql in [("ipo_bars.csv", ipo_sql), ("spy.csv", spy_sql), ("placebo_bars.csv", placebo_sql)]:
        n = copy_out(sql, DATA / name)
        manifest[name] = {"rows": n, "sql": " ".join(sql.split())[:600] + (" ..." if len(sql) > 600 else "")}
        print(f"{name}: {n:,} rows")
    manifest["universe_symbols"] = int(len(uni))
    (DATA / "export_manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
