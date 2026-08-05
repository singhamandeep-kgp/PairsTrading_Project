"""
Run the FactSet discovery queries and dump the results.

Same shape as the masterlist script: pyodbc connection + pd.read_sql. That is
the slow path and we would never use it for the 92M-row fact table, but for a
handful of metadata queries returning tens of rows it is exactly right, and it
needs nothing installed beyond pyodbc.

    python run_discovery.py                 # run all
    python run_discovery.py Q1 Q2           # run only those

Output:
    * printed to the console (this is what you paste back)
    * one CSV per result set under  <repo>/discovery_output/

Everything here is READ-ONLY. No temp tables, no writes, NOLOCK throughout so
the nightly FactSet loader is never blocked.
"""

from __future__ import annotations

import csv
import sys
import warnings

from . import config

warnings.filterwarnings("ignore")

# Deliberately NO pandas. These are metadata queries returning tens of rows, so
# pyodbc + stdlib csv is sufficient - and it keeps this script runnable on a
# bare interpreter, which matters because pandas has no prebuilt wheel for
# Python 3.14 yet and falls back to a source build that hangs.

OUTPUT_DIR = config.REPO_ROOT / "discovery_output"


# =============================================================================
# QUERIES  -- scoped so every result set is small enough to paste back
# =============================================================================

Q: dict[str, tuple[str, str]] = {}


Q["Q1_capability_summary"] = ("""
ONE ROW. Settles the OHLCV question, the wire-cost question, and whether a
point-in-time sector history table exists.
""", """
SELECT
  MAX(CASE WHEN TABLE_NAME='fp_basic_prices' AND COLUMN_NAME='p_price_open' THEN 1 ELSE 0 END) AS bp_has_open,
  MAX(CASE WHEN TABLE_NAME='fp_basic_prices' AND COLUMN_NAME='p_price_high' THEN 1 ELSE 0 END) AS bp_has_high,
  MAX(CASE WHEN TABLE_NAME='fp_basic_prices' AND COLUMN_NAME='p_price_low'  THEN 1 ELSE 0 END) AS bp_has_low,
  MAX(CASE WHEN TABLE_NAME='fp_basic_prices' AND COLUMN_NAME='p_volume'     THEN 1 ELSE 0 END) AS bp_has_volume,
  MAX(CASE WHEN TABLE_NAME='fp_basic_prices' AND COLUMN_NAME='fsym_id' THEN DATA_TYPE END)     AS bp_fsym_id_type,
  MAX(CASE WHEN TABLE_NAME='fp_basic_prices' AND COLUMN_NAME='p_price' THEN DATA_TYPE END)     AS bp_price_type,
  MAX(CASE WHEN TABLE_NAME='fp_basic_prices' AND COLUMN_NAME='p_volume' THEN DATA_TYPE END)    AS bp_volume_type,
  MAX(CASE WHEN TABLE_NAME='sym_entity_sector_hist' THEN 1 ELSE 0 END)                         AS has_sector_hist,
  MAX(CASE WHEN TABLE_NAME LIKE '%rbics%'           THEN 1 ELSE 0 END)                         AS has_rbics
FROM INFORMATION_SCHEMA.COLUMNS
WHERE TABLE_SCHEMA IN ('fp_v2','fgp_v1','sym_v1','ent_v1','ref_v2','own_v5')
""")


Q["Q1b_price_table_columns"] = ("""
Every column of fp_basic_prices, so we see exactly what OHLCV is available and
in what type.
""", """
SELECT ORDINAL_POSITION, COLUMN_NAME, DATA_TYPE,
       CHARACTER_MAXIMUM_LENGTH AS char_len,
       NUMERIC_PRECISION, NUMERIC_SCALE, IS_NULLABLE
FROM   INFORMATION_SCHEMA.COLUMNS
WHERE  TABLE_SCHEMA = 'fp_v2' AND TABLE_NAME = 'fp_basic_prices'
ORDER  BY ORDINAL_POSITION
""")


Q["Q2a_ca_tables"] = ("""
THE BIGGEST BLOCKER: tables whose name suggests splits / dividends /
distributions / adjustment factors / delisting.
""", """
SELECT s.name AS sch, t.name AS tbl
FROM   sys.tables t JOIN sys.schemas s ON s.schema_id = t.schema_id
WHERE  s.name IN ('fp_v2','fgp_v1','sym_v1','ent_v1','ref_v2','own_v5')
  AND (t.name LIKE '%split%'   OR t.name LIKE '%div%'
    OR t.name LIKE '%distrib%' OR t.name LIKE '%action%'
    OR t.name LIKE '%adj%'     OR t.name LIKE '%spin%'
    OR t.name LIKE '%delist%'  OR t.name LIKE '%return%'
    OR t.name LIKE '%shares%'  OR t.name LIKE '%hist%')
ORDER BY s.name, t.name
""")


Q["Q2b_ca_columns"] = ("""
Columns suggesting an adjustment factor, anywhere in the feed. This is what
tells us whether FactSet ships a CUMULATIVE factor (like CRSP cfacpr) or only
discrete split events we must multiply backwards ourselves.
""", """
SELECT TABLE_SCHEMA, TABLE_NAME, COLUMN_NAME, DATA_TYPE
FROM   INFORMATION_SCHEMA.COLUMNS
WHERE  TABLE_SCHEMA IN ('fp_v2','fgp_v1','sym_v1','ent_v1','ref_v2','own_v5')
  AND (COLUMN_NAME LIKE '%adj%'    OR COLUMN_NAME LIKE '%split%'
    OR COLUMN_NAME LIKE '%factor%' OR COLUMN_NAME LIKE '%cum%'
    OR COLUMN_NAME LIKE '%delist%' OR COLUMN_NAME LIKE '%div%')
ORDER  BY TABLE_SCHEMA, TABLE_NAME, COLUMN_NAME
""")


Q["Q2c_point_in_time_tables"] = ("""
Tables carrying BOTH start_date and end_date, i.e. genuine point-in-time
history. If sym_entity_sector is absent here it is a current snapshot, and
applying a snapshot to history leaks the future into the clustering universe.
""", """
SELECT TABLE_SCHEMA, TABLE_NAME
FROM   INFORMATION_SCHEMA.COLUMNS
WHERE  COLUMN_NAME IN ('start_date','end_date')
GROUP  BY TABLE_SCHEMA, TABLE_NAME
HAVING COUNT(*) = 2
ORDER  BY TABLE_SCHEMA, TABLE_NAME
""")


Q["Q3_indexes"] = ("""
Decides the shard strategy. If nothing leads with fsym_id or p_date we must not
shard at all: N shards on an unindexed column means N full scans of ~120 GB.
""", """
SELECT t.name AS tbl, i.name AS idx, i.type_desc, i.is_unique,
       STUFF((SELECT ', ' + c.name
              FROM   sys.index_columns ic
              JOIN   sys.columns c ON c.object_id = ic.object_id
                                  AND c.column_id = ic.column_id
              WHERE  ic.object_id = i.object_id AND ic.index_id = i.index_id
                AND  ic.is_included_column = 0
              ORDER  BY ic.key_ordinal
              FOR XML PATH('')), 1, 2, '') AS key_cols
FROM   sys.indexes i
JOIN   sys.tables  t ON t.object_id = i.object_id
JOIN   sys.schemas s ON s.schema_id = t.schema_id
WHERE  s.name IN ('fp_v2','own_v5')
  AND  t.name IN ('fp_basic_prices','fp_total_returns_daily','own_sec_prices')
ORDER  BY t.name, i.index_id
""")


Q["Q4_table_sizes"] = ("""
Row counts from metadata (instant, never COUNT(*)). Confirms the ~1.01B figure
and gives the full-scan wire ceiling per table.
""", """
SELECT s.name AS sch, t.name AS tbl,
       SUM(CASE WHEN ps.index_id IN (0,1) THEN ps.row_count ELSE 0 END) AS n_rows,
       CAST(SUM(ps.reserved_page_count)*8.0/1024 AS DECIMAL(14,1))      AS reserved_mb
FROM   sys.dm_db_partition_stats ps
JOIN   sys.tables  t ON t.object_id = ps.object_id
JOIN   sys.schemas s ON s.schema_id = t.schema_id
WHERE  s.name IN ('fp_v2','fgp_v1','sym_v1','ent_v1','ref_v2','own_v5')
GROUP  BY s.name, t.name
HAVING SUM(CASE WHEN ps.index_id IN (0,1) THEN ps.row_count ELSE 0 END) > 0
ORDER  BY n_rows DESC
""")


Q["Q4c_table_sizes_fallback"] = ("""
Row counts WITHOUT the DMVs. Q4 needs VIEW DATABASE STATE, which this login does
not have. sys.partitions is a catalog view rather than a DMV, so it only needs
metadata visibility on the objects - which we clearly have.
""", """
SELECT s.name AS sch, t.name AS tbl, SUM(p.rows) AS n_rows
FROM   sys.partitions p
JOIN   sys.tables  t ON t.object_id = p.object_id
JOIN   sys.schemas s ON s.schema_id = t.schema_id
WHERE  s.name IN ('fp_v2','fgp_v1','sym_v1','ent_v1','ref_v2','own_v5')
  AND  p.index_id IN (0,1)
GROUP  BY s.name, t.name
HAVING SUM(p.rows) > 0
ORDER  BY n_rows DESC
""")


Q["Q4b_date_bounds"] = ("""
Real date coverage. Also a timing probe: if this is slow there is no useful
index on p_date, which answers the shard question by itself.
""", """
SELECT MIN(p_date) AS d_min, MAX(p_date) AS d_max
FROM   fp_v2.fp_basic_prices WITH (NOLOCK)
OPTION (MAXDOP 2)
""")


Q["Q5_volume_units"] = ("""
Settles the p_volume unit ambiguity. AAPL trades ~50-100 MILLION shares/day:
  p_volume ~ 5e7 -> raw shares
  p_volume ~ 5e4 -> thousands
The other team's /1000 and the FactSet docs disagree; a 1000x error would break
every liquidity screen.
""", """
SELECT TOP (10) t.ticker_region, p.p_date, p.p_price, p.p_volume
FROM   fp_v2.fp_basic_prices    p WITH (NOLOCK)
JOIN   sym_v1.sym_ticker_region t WITH (NOLOCK) ON t.fsym_id = p.fsym_id
WHERE  t.ticker_region = 'AAPL-US'
  AND  p.p_date BETWEEN '2024-01-02' AND '2024-01-09'
ORDER  BY p.p_date
OPTION (MAXDOP 2)
""")


Q["Q6a_us_exchanges"] = ("""
Real US exchange codes, so US_EXCHANGE_CODES stops being a guess.
""", """
SELECT ex.fref_exchange_code, ex.fref_exchange_desc, cm.country_desc
FROM   ref_v2.fref_sec_exchange_map ex
JOIN   ref_v2.country_map cm ON cm.iso_country = ex.fref_exchange_location_code
WHERE  cm.country_desc LIKE '%United States%'
ORDER  BY ex.fref_exchange_code
""")


Q["Q6b_security_types"] = ("""
Security-type vocabularies as they actually occur in a USD equity universe,
with counts. Read the descriptions and pick SEC_TYPE_CODES from this rather than
copying the other team's global-index list.
""", """
SELECT sc.fref_security_type,
       stc.p_sec_type_code,
       stm.p_sec_type_desc,
       COUNT(*) AS n_securities
FROM   sym_v1.sym_coverage         sc  WITH (NOLOCK)
LEFT   JOIN fp_v2.fp_sec_coverage  stc WITH (NOLOCK) ON stc.fsym_id = sc.fsym_id
LEFT   JOIN ref_v2.fp_sec_type_map stm ON stm.p_sec_type_code = stc.p_sec_type_code
WHERE  sc.regional_flag = 1 AND sc.currency = 'USD'
GROUP  BY sc.fref_security_type, stc.p_sec_type_code, stm.p_sec_type_desc
HAVING COUNT(*) >= 20
ORDER  BY n_securities DESC
""")


Q["Q7a_universe_by_year"] = ("""
SANITY CHECK. Distinct US common-stock securities per year. Expected shape:
~7,000-8,000 in the late 1990s declining to ~4,000-4,500 by 2020. A FLAT or
RISING line means survivorship bias or non-common contamination.
Also replaces the ~92M row estimate with a measurement. SLOW - run last.
""", """
SELECT YEAR(p.p_date)            AS yr,
       COUNT(DISTINCT p.fsym_id) AS n_ids,
       COUNT_BIG(*)              AS n_rows
FROM   fp_v2.fp_basic_prices p WITH (NOLOCK)
JOIN   sym_v1.sym_coverage   sc WITH (NOLOCK) ON sc.fsym_id = p.fsym_id
WHERE  sc.regional_flag = 1 AND sc.currency = 'USD'
  AND  sc.fref_security_type = 'SHARE'
  AND  p.p_date >= '1995-01-01'
GROUP  BY YEAR(p.p_date)
ORDER  BY yr
OPTION (MAXDOP 2)
""")


Q["Q7b_universe_active_only"] = ("""
The SAME query with active_flag = 1 added. The difference between Q7a and Q7b IS
your survivorship bias, quantified. Measurement only - never in production.
""", """
SELECT YEAR(p.p_date)            AS yr,
       COUNT(DISTINCT p.fsym_id) AS n_ids_active_only
FROM   fp_v2.fp_basic_prices p WITH (NOLOCK)
JOIN   sym_v1.sym_coverage   sc WITH (NOLOCK) ON sc.fsym_id = p.fsym_id
WHERE  sc.regional_flag = 1 AND sc.currency = 'USD'
  AND  sc.fref_security_type = 'SHARE'
  AND  sc.active_flag = 1
  AND  p.p_date >= '1995-01-01'
GROUP  BY YEAR(p.p_date)
ORDER  BY yr
OPTION (MAXDOP 2)
""")


Q["Q8_ca_table_columns"] = ("""
Columns of every corporate-action / shares / adjustment table found by Q2a.
This is what the corporate-action queries get written against.
""", """
SELECT TABLE_SCHEMA, TABLE_NAME, ORDINAL_POSITION, COLUMN_NAME, DATA_TYPE,
       CHARACTER_MAXIMUM_LENGTH AS char_len, NUMERIC_PRECISION, NUMERIC_SCALE
FROM   INFORMATION_SCHEMA.COLUMNS
WHERE  TABLE_NAME IN ('fp_basic_splits','fp_basic_dividends',
                      'fp_basic_shares_hist','fp_total_returns_daily',
                      'fgp_ca_adj_factors','fgp_ca_events',
                      'fgp_global_prices','sym_entity_sector_rbics')
ORDER  BY TABLE_SCHEMA, TABLE_NAME, ORDINAL_POSITION
""")


Q["Q9_split_convention"] = ("""
*** THE SPLIT CONVENTION TEST - the single most important row of this sweep ***

AAPL's real splits: 7-for-1 on 2014-06-09, 4-for-1 on 2020-08-31.
So the recorded factor tells us the convention unambiguously:
  factor = 7.0 / 4.0   -> factor is the RATIO of new shares to old
  factor = 0.1429/0.25 -> factor is the reciprocal
Whichever it is decides whether we MULTIPLY or DIVIDE historical prices.
Getting this backwards inverts 25 years of prices into a series that still
looks entirely plausible.
""", """
SELECT s.*
FROM   fp_v2.fp_basic_splits s WITH (NOLOCK)
JOIN   sym_v1.sym_ticker_region t WITH (NOLOCK) ON t.fsym_id = s.fsym_id
WHERE  t.ticker_region = 'AAPL-US'
ORDER  BY 2
""")


Q["Q10_adj_factor_sample"] = ("""
fgp_ca_adj_factors around AAPL's 2020-08-31 4-for-1 split. Tells us whether the
factor is CUMULATIVE (like CRSP cfacpr, one value per date) or per-event, and in
which direction it moves across a split.
""", """
SELECT TOP (30) a.*
FROM   fgp_v1.fgp_ca_adj_factors a WITH (NOLOCK)
JOIN   sym_v1.sym_ticker_region t WITH (NOLOCK) ON t.fsym_id = a.fsym_id
WHERE  t.ticker_region = 'AAPL-US'
ORDER  BY 2 DESC
""")


Q["Q11_code_vocabularies"] = ("""
The corporate-action code vocabularies. Small tables; needed to interpret
dividend and event type codes (regular cash vs special vs stock vs return of
capital - they must be treated differently in a total-return calculation).
""", """
SELECT 'ca_event_type_map' AS map_name, * FROM ref_v2.ca_event_type_map
""")


Q["Q11b_div_type_map"] = ("""
Dividend type codes.
""", """
SELECT * FROM ref_v2.fp_div_type_map
""")


Q["Q12_trading_calendar"] = ("""
FactSet ships its own exchange calendar (ref_calendar_dates /
ref_calendar_holidays). Better than deriving the calendar from one stock's
observed dates, which is what the legacy code does - AAPL's date index also
encodes AAPL's trading halts.
""", """
SELECT TABLE_SCHEMA, TABLE_NAME, ORDINAL_POSITION, COLUMN_NAME, DATA_TYPE
FROM   INFORMATION_SCHEMA.COLUMNS
WHERE  TABLE_NAME IN ('ref_calendar_dates','ref_calendar_holidays')
ORDER  BY TABLE_NAME, ORDINAL_POSITION
""")


Q["Q13_rbics_inventory"] = ("""
sym_entity_sector is a SNAPSHOT, so strict point-in-time sector must come from
RBICS. Inventory the rbics_v1 schema and the sector-mapping tables, with sizes.
""", """
SELECT s.name AS sch, t.name AS tbl, SUM(p.rows) AS n_rows
FROM   sys.partitions p
JOIN   sys.tables  t ON t.object_id = p.object_id
JOIN   sys.schemas s ON s.schema_id = t.schema_id
WHERE (s.name IN ('rbics_v1','ff_v3')
    OR t.name LIKE '%rbics%' OR t.name LIKE '%sector%')
  AND  p.index_id IN (0,1)
GROUP  BY s.name, t.name
ORDER  BY n_rows DESC
""")


Q["Q14_rbics_columns"] = ("""
Columns of the RBICS sector tables. We need the one that maps a security or
entity to a sector code WITH validity dates, plus the structure table that gives
human-readable level names.
""", """
SELECT TABLE_SCHEMA, TABLE_NAME, ORDINAL_POSITION, COLUMN_NAME, DATA_TYPE,
       CHARACTER_MAXIMUM_LENGTH AS char_len
FROM   INFORMATION_SCHEMA.COLUMNS
WHERE  TABLE_SCHEMA IN ('rbics_v1')
    OR TABLE_NAME IN ('sym_entity_sector','sym_entity_sector_rbics',
                      'rbics_structure_l2_curr')
ORDER  BY TABLE_SCHEMA, TABLE_NAME, ORDINAL_POSITION
""")


Q["Q15_delisting_candidates"] = ("""
Is there a delisting return in this feed? Checks whether the liquidation /
final-distribution dividend type codes (73, 74, 16) actually occur, and how
often. A missing delisting return is exactly what silently inflates backtest
returns, so if these are empty we still need CRSP for that one field.
""", """
SELECT d.p_divs_pd_type_code, m.p_divs_pd_type_desc, COUNT(*) AS n
FROM   fp_v2.fp_basic_dividends d WITH (NOLOCK)
LEFT   JOIN ref_v2.fp_div_type_map m
       ON m.p_divs_pd_type_code = d.p_divs_pd_type_code
WHERE  d.p_divs_pd_type_code IN ('73','74','16','200','1055','11')
GROUP  BY d.p_divs_pd_type_code, m.p_divs_pd_type_desc
ORDER  BY n DESC
OPTION (MAXDOP 2)
""")


Q["Q16_fgp_vs_fp"] = ("""
*** DECIDES THE PRIMARY PRICE SOURCE ***

fgp_global_prices is far richer than fp_basic_prices: it adds turnover,
trade_count, vwap AND one_day_pct in the same table, which would remove the need
for a separate total-returns pull entirely.

Two things must be true before we switch to it:
  1. `price` must be UNADJUSTED. Compare across AAPL's 2020-08-31 4-for-1 split:
     if the pre-split rows show ~$500 it is unadjusted (good - we want raw plus a
     factor). If they show ~$125 it is already adjusted, which is exactly what we
     must NOT store as the anchor.
  2. Coverage must reach back far enough. 682M rows vs 1.015B for fp_basic_prices
     means less of something - history, securities, or both.
""", """
SELECT 'fgp_global_prices' AS src, price_date AS d, price, price_open,
       volume, vwap, one_day_pct
FROM   fgp_v1.fgp_global_prices WITH (NOLOCK)
WHERE  fsym_id = 'MH33D6-R'
  AND  price_date BETWEEN '2020-08-27' AND '2020-09-02'
ORDER  BY price_date
""")


Q["Q16b_fp_same_window"] = ("""
The same AAPL window from fp_basic_prices, for a side-by-side comparison.
""", """
SELECT 'fp_basic_prices' AS src, p_date AS d, p_price, p_price_open, p_volume
FROM   fp_v2.fp_basic_prices WITH (NOLOCK)
WHERE  fsym_id = 'MH33D6-R'
  AND  p_date BETWEEN '2020-08-27' AND '2020-09-02'
ORDER  BY p_date
""")


Q["Q16c_fgp_coverage"] = ("""
fgp_global_prices date span and US-equity coverage, to see what the 682M vs
1.015B row gap actually represents.
""", """
SELECT MIN(price_date) AS d_min, MAX(price_date) AS d_max
FROM   fgp_v1.fgp_global_prices WITH (NOLOCK)
OPTION (MAXDOP 2)
""")


Q["Q17_listing_exchange_dist"] = ("""
DEFINITIVE US exchange list. Q6a returned 91 "United States" codes, but most are
options, futures or bond venues - not equity LISTING venues. Rather than guess
which, ask what fref_listing_exchange actually contains for USD common equity.

Also splits out securities that DO vs DO NOT appear in fp_sec_coverage, which
explains the 27,169 SHARE rows with a NULL p_sec_type_code in Q6b.
""", """
SELECT sc.fref_listing_exchange,
       ex.fref_exchange_desc,
       stc.p_sec_type_code,
       COUNT(*) AS n_securities,
       SUM(CASE WHEN sc.active_flag = 1 THEN 1 ELSE 0 END) AS n_active
FROM   sym_v1.sym_coverage         sc  WITH (NOLOCK)
LEFT   JOIN fp_v2.fp_sec_coverage  stc WITH (NOLOCK) ON stc.fsym_id = sc.fsym_id
LEFT   JOIN ref_v2.fref_sec_exchange_map ex
       ON ex.fref_exchange_code = sc.fref_listing_exchange
WHERE  sc.regional_flag = 1
  AND  sc.currency = 'USD'
  AND  sc.fref_security_type = 'SHARE'
GROUP  BY sc.fref_listing_exchange, ex.fref_exchange_desc, stc.p_sec_type_code
HAVING COUNT(*) >= 10
ORDER  BY n_securities DESC
""")


# =============================================================================
# RUNNER
# =============================================================================

def _fmt_table(cols: list[str], rows: list[tuple]) -> str:
    """Fixed-width text table. Replaces pandas' to_string for console output."""
    if not rows:
        return "  (empty result set)"
    cells = [[("" if v is None else str(v)) for v in row] for row in rows]
    widths = [
        min(60, max(len(c), *(len(r[i]) for r in cells)))
        for i, c in enumerate(cols)
    ]
    out = ["  " + "  ".join(c.ljust(w)[:w] for c, w in zip(cols, widths))]
    out.append("  " + "  ".join("-" * w for w in widths))
    out += [
        "  " + "  ".join(v.ljust(w)[:w] for v, w in zip(row, widths))
        for row in cells
    ]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]

    try:
        import pyodbc
    except ImportError:
        print("pyodbc is not installed.  Run:  pip install pyodbc")
        return 1

    selected = [k for k in Q if not argv or any(a.lower() in k.lower() for a in argv)]
    if not selected:
        print(f"No queries matched {argv}.  Available:\n  " + "\n  ".join(Q))
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 79)
    print("FactSet discovery")
    print(f"  {config.safe_connection_summary()}")
    print(f"  output -> {OUTPUT_DIR}")
    print(f"  running {len(selected)} of {len(Q)} queries")
    print("=" * 79)

    conn = pyodbc.connect(config.connection_string(), timeout=30)
    # READ UNCOMMITTED so a metadata sweep never takes shared locks on a
    # production feed table.
    conn.execute("SET TRANSACTION ISOLATION LEVEL READ UNCOMMITTED;")

    failures: list[tuple[str, str]] = []
    try:
        for name in selected:
            note, sql = Q[name]
            print(f"\n{'=' * 79}\n{name}\n{'-' * 79}")
            print(note.strip())
            print("-" * 79)
            try:
                cur = conn.cursor()
                cur.execute(sql)
                cols = [d[0] for d in cur.description]
                rows = cur.fetchall()
                cur.close()
            except Exception as exc:                    # noqa: BLE001
                # A missing object name is itself information - record it and
                # move on rather than aborting the whole sweep.
                msg = str(exc).replace("\n", " ")[:400]
                print(f"  !! FAILED: {msg}")
                failures.append((name, msg))
                continue

            with open(OUTPUT_DIR / f"{name}.csv", "w", newline="",
                      encoding="utf-8") as fh:
                w = csv.writer(fh)
                w.writerow(cols)
                w.writerows(rows)

            print(f"  {len(rows)} rows -> {name}.csv")
            print(_fmt_table(cols, rows[:200]))
            if len(rows) > 200:
                print(f"  ... {len(rows) - 200} more rows (see the CSV)")
    finally:
        conn.close()

    print(f"\n{'=' * 79}\nDONE.  CSVs in {OUTPUT_DIR}")
    if failures:
        print("\nFAILED QUERIES (a missing object name is a real finding):")
        for name, msg in failures:
            print(f"  {name}: {msg}")
    print("=" * 79)
    return 0


if __name__ == "__main__":
    sys.exit(main())
