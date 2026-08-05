"""
SQL Queries

"""

from __future__ import annotations
import hashlib

from . import config


# =============================================================================
# HELPERS
# =============================================================================

def _sql_list(values) -> str:
    """Render a Python sequence as a SQL IN-list of quoted literals.

    Only ever used for small, code-controlled vocabularies (exchange codes,
    security types) - never for the ~34k security ids. Those go through a #temp
    table: a 30k-literal IN-list would be ~330 KB of SQL text and SQL Server's
    optimiser degrades badly past a few thousand literals.
    """
    return ", ".join(f"'{v}'" for v in values)


def query_hash(sql: str) -> str:
    """Stable hash of a query, recorded in the extraction manifest.

    This is what lets you detect that a partition was built with a different
    query than the one currently in the code - otherwise you get silent,
    permanent inconsistency between partitions, which is the likeliest way this
    database quietly goes wrong.
    """
    return hashlib.sha256(sql.encode("utf-8")).hexdigest()[:16]


# =============================================================================
# SESSION PRELUDE
# =============================================================================

SESSION_PRELUDE = """
/* Run at the start of EVERY connection, and again after ANY reconnect,
   together with re-creating the #temp tables - temp tables are
   connection-scoped and die with the session. Retry logic that re-issues the
   query WITHOUT restoring session state is the single most common cause of
   "it worked for three hours then failed weirdly". */
SET NOCOUNT ON;
SET ANSI_WARNINGS OFF;
SET TRANSACTION ISOLATION LEVEL READ UNCOMMITTED;   -- take no shared locks
SET LOCK_TIMEOUT 5000;
SET DEADLOCK_PRIORITY LOW;                          -- we lose, not the loader
SET ARITHABORT ON;
"""


# =============================================================================
# TEMP TABLES  --  the universe, as a wire-compression device
# =============================================================================

DDL_TEMP_TABLES = """
/* Temp-table creation in tempdb is granted to `public`, so a read-only user can
   do this. No CREATE TABLE grant is needed - that is only required for
   PERMANENT tables in tempdb.

   These are not merely filters. Selecting u.sid instead of p.fsym_id converts
   an 8-40 byte string into 4 bytes on the wire, 92 million times. On an
   uncompressed TDS connection that alone is ~0.6 GB of transfer saved.

   Three tables because the FactSet schemas key on three different ids:
     #universe      -R regional id  -> fp_v2.fp_basic_prices
     #universe_sec  -S security id  -> own_v5.*, sym_isin
     #universe_ent  entity id       -> ent_v1.*, sym_entity_sector
*/
DROP TABLE IF EXISTS #universe;
DROP TABLE IF EXISTS #universe_sec;
DROP TABLE IF EXISTS #universe_ent;

CREATE TABLE #universe (
    sid     INT         NOT NULL PRIMARY KEY,
    fsym_id VARCHAR(20) NOT NULL UNIQUE
);

CREATE TABLE #universe_sec (
    sid              INT         NOT NULL PRIMARY KEY,
    fsym_security_id VARCHAR(20) NOT NULL
);
CREATE INDEX ix_sec ON #universe_sec (fsym_security_id);

CREATE TABLE #universe_ent (
    factset_entity_id VARCHAR(20) NOT NULL PRIMARY KEY
);
"""

INSERT_TEMP_UNIVERSE     = "INSERT INTO #universe (sid, fsym_id) VALUES (?, ?);"
INSERT_TEMP_UNIVERSE_SEC = "INSERT INTO #universe_sec (sid, fsym_security_id) VALUES (?, ?);"
INSERT_TEMP_UNIVERSE_ENT = "INSERT INTO #universe_ent (factset_entity_id) VALUES (?);"


# =============================================================================
# SECTION A -- DISCOVERY
#
# Run these FIRST. They are cheap, read-only, and they resolve the unknowns
# that currently block the real extraction.
# =============================================================================

def d1_table_inventory() -> str:
    """Every FactSet table with FAST row counts and size.

    Row counts come from sys.dm_db_partition_stats, which reads metadata only.
    NEVER use COUNT(*) here - it would scan a billion rows to learn a number
    the engine already knows. `reserved_mb` is also your full-scan wire ceiling.
    """
    return """
SELECT s.name AS sch, t.name AS tbl,
       SUM(CASE WHEN ps.index_id IN (0,1) THEN ps.row_count ELSE 0 END) AS [rows],
       CAST(SUM(ps.reserved_page_count)*8.0/1024 AS DECIMAL(14,1))      AS reserved_mb,
       CAST(SUM(ps.used_page_count)*8.0/1024     AS DECIMAL(14,1))      AS used_mb,
       MAX(CASE WHEN ps.index_id = 0 THEN 1 ELSE 0 END)                 AS is_heap
FROM   sys.dm_db_partition_stats ps
JOIN   sys.tables  t ON t.object_id = ps.object_id
JOIN   sys.schemas s ON s.schema_id = t.schema_id
WHERE  s.name IN ('fp_v2','fgp_v1','sym_v1','ent_v1','ref_v2','own_v5')
GROUP  BY s.name, t.name
ORDER  BY [rows] DESC;
"""


def d2_columns() -> str:
    """Full column and type list for all six schemas.

    THE TWO THINGS THIS SETTLES:

    1. Does fp_v2.fp_basic_prices have p_price_open / p_price_high /
       p_price_low? The other team's production query selects only p_price and
       p_volume. Public FactSet docs suggest OHLC exists, but if it does not,
       the OHLCV requirement needs a different source and the extraction plan
       changes materially. HIGHEST-PRIORITY UNKNOWN.

    2. Wire cost. fsym_id as varchar (8 B/row) vs nvarchar (40 B/row); prices
       as real (4 B) vs float/decimal (8 B). Multiply by 92M rows - this single
       result set is worth GBs of transfer time.
    """
    return """
SELECT TABLE_SCHEMA, TABLE_NAME, ORDINAL_POSITION, COLUMN_NAME,
       DATA_TYPE, CHARACTER_MAXIMUM_LENGTH AS char_len,
       NUMERIC_PRECISION, NUMERIC_SCALE, IS_NULLABLE
FROM   INFORMATION_SCHEMA.COLUMNS
WHERE  TABLE_SCHEMA IN ('fp_v2','fgp_v1','sym_v1','ent_v1','ref_v2','own_v5')
ORDER  BY TABLE_SCHEMA, TABLE_NAME, ORDINAL_POSITION;
"""


def d3_indexes() -> str:
    """THE shard-key decision, and the most consequential discovery query.

    If no index leads with fsym_id or p_date, set SHARD_MODE = "single_pass".
    Otherwise each of N shards triggers its own full scan of ~120 GB of pages:
    N-way "parallelism" becomes N times the total I/O, finishes SLOWER than one
    sequential pass, and cache-thrashes the buffer pool for every other user of
    the server. This is the mistake most likely to get read access revoked.
    """
    return """
SELECT s.name AS sch, t.name AS tbl, i.name AS idx,
       i.type_desc,          -- HEAP / CLUSTERED / NONCLUSTERED / COLUMNSTORE
       i.is_unique,
       STUFF((SELECT ', ' + c.name
                     + CASE WHEN ic.is_descending_key = 1 THEN ' DESC' ELSE '' END
              FROM   sys.index_columns ic
              JOIN   sys.columns c ON c.object_id = ic.object_id
                                  AND c.column_id = ic.column_id
              WHERE  ic.object_id = i.object_id AND ic.index_id = i.index_id
                AND  ic.is_included_column = 0
              ORDER  BY ic.key_ordinal
              FOR XML PATH('')), 1, 2, '') AS key_cols,
       STUFF((SELECT ', ' + c.name
              FROM   sys.index_columns ic
              JOIN   sys.columns c ON c.object_id = ic.object_id
                                  AND c.column_id = ic.column_id
              WHERE  ic.object_id = i.object_id AND ic.index_id = i.index_id
                AND  ic.is_included_column = 1
              FOR XML PATH('')), 1, 2, '') AS included_cols
FROM   sys.indexes i
JOIN   sys.tables  t ON t.object_id = i.object_id
JOIN   sys.schemas s ON s.schema_id = t.schema_id
WHERE  s.name IN ('fp_v2','fgp_v1','own_v5')
ORDER  BY s.name, t.name, i.index_id;
"""


def d3b_workload_check() -> str:
    """Is anyone else actively using these tables? Determines whether we may
    raise MAXDOP and concurrency, or must stay minimal."""
    return """
SELECT OBJECT_NAME(s.object_id) AS tbl, i.name AS idx,
       s.user_seeks, s.user_scans, s.user_lookups, s.user_updates,
       s.last_user_scan, s.last_user_seek
FROM   sys.dm_db_index_usage_stats s
JOIN   sys.indexes i ON i.object_id = s.object_id AND i.index_id = s.index_id
WHERE  s.database_id = DB_ID()
  AND  (OBJECT_NAME(s.object_id) LIKE 'fp_%' OR OBJECT_NAME(s.object_id) LIKE 'own_%')
ORDER  BY s.user_scans DESC;
"""


def d4_exchange_map() -> str:
    """Full exchange map with country. Tiny table - read it and pick the real US
    codes rather than trusting config.US_EXCHANGE_CODES, which is a guess."""
    return """
SELECT ex.fref_exchange_code, ex.fref_exchange_desc,
       ex.fref_exchange_location_code, cm.country_desc
FROM   ref_v2.fref_sec_exchange_map ex
LEFT   JOIN ref_v2.country_map cm
       ON cm.iso_country = ex.fref_exchange_location_code
ORDER  BY cm.country_desc, ex.fref_exchange_code;
"""


def d5_security_type_maps() -> str:
    """Both security-type vocabularies, with descriptions.

    'SHARE' (fref_security_type) is the coarse screen. p_sec_type_code is the
    fine one that separates common stock from units, trusts, partnership
    interests and similar. Read the descriptions before setting
    config.SEC_TYPE_CODES.
    """
    return """
SELECT 'fp_sec_type' AS map_name,
       p_sec_type_code AS code, p_sec_type_desc AS descr
FROM   ref_v2.fp_sec_type_map
UNION ALL
SELECT 'fref_security_type',
       fref_security_type_code, fref_security_type_desc
FROM   ref_v2.fref_security_type_map
ORDER  BY map_name, code;
"""


def d6_corporate_action_hunt() -> str:
    """*** THE CRITICAL UNKNOWN - corporate actions and adjustment factors ***

    The other team's production query contains NO adjustment factor, NO
    corporate-action table and NO total return. It reads raw p_price only,
    which is correct for THEIR use case (an index masterlist is a
    point-in-time snapshot). A backtester needs a CONTINUOUS ADJUSTED SERIES,
    so we must locate:

      (a) splits          - date, factor, and crucially the CONVENTION
      (b) dividends       - ex-date, amount, currency, type code
      (c) adjustment factor - cumulative (like CRSP's cfacpr), or must it be
                              derived by multiplying split factors backwards?
      (d) delisting       - date, reason, and ideally a delisting RETURN

    These table names are NOT VERIFIED and are deliberately not guessed. The
    split-factor convention - is a 2-for-1 stored as 2.0 or 0.5, does it
    multiply or divide the historical price? - sets the DIRECTION of the
    adjustment. Get it backwards and 25 years of prices invert into a series
    that still looks entirely plausible. That is the worst failure mode
    available in this dataset.
    """
    return """
-- (i) Tables whose NAME suggests corporate actions or history
SELECT s.name AS sch, t.name AS tbl
FROM   sys.tables t
JOIN   sys.schemas s ON s.schema_id = t.schema_id
WHERE  t.name LIKE '%split%'   OR t.name LIKE '%div%'
   OR  t.name LIKE '%distrib%' OR t.name LIKE '%action%'
   OR  t.name LIKE '%adj%'     OR t.name LIKE '%spin%'
   OR  t.name LIKE '%delist%'  OR t.name LIKE '%return%'
   OR  t.name LIKE '%shares%'  OR t.name LIKE '%hist%'
ORDER  BY s.name, t.name;

-- (ii) COLUMNS suggesting an adjustment factor
SELECT TABLE_SCHEMA, TABLE_NAME, COLUMN_NAME, DATA_TYPE
FROM   INFORMATION_SCHEMA.COLUMNS
WHERE  COLUMN_NAME LIKE '%adj%'    OR COLUMN_NAME LIKE '%split%'
   OR  COLUMN_NAME LIKE '%factor%' OR COLUMN_NAME LIKE '%cum%'
   OR  COLUMN_NAME LIKE '%div%'    OR COLUMN_NAME LIKE '%delist%'
ORDER  BY TABLE_SCHEMA, TABLE_NAME, COLUMN_NAME;

-- (iii) POINT-IN-TIME check: tables carrying BOTH start_date and end_date.
--       If sym_v1.sym_entity_sector is NOT in this list, it is a current
--       snapshot - and using a snapshot for history injects look-ahead bias
--       straight into the PCA/clustering universe, because sector membership
--       is what DEFINES that universe.
SELECT TABLE_SCHEMA, TABLE_NAME
FROM   INFORMATION_SCHEMA.COLUMNS
WHERE  COLUMN_NAME IN ('start_date','end_date')
GROUP  BY TABLE_SCHEMA, TABLE_NAME
HAVING COUNT(*) = 2
ORDER  BY TABLE_SCHEMA, TABLE_NAME;
"""


def d7_price_columns() -> str:
    """Which columns does fp_basic_prices actually have?

    Used to build the fact-table SELECT list from columns that EXIST rather
    than columns we hope exist - so a missing OHLC column degrades gracefully
    instead of erroring mid-extraction on the one day you have server access.
    """
    return """
SELECT COLUMN_NAME, DATA_TYPE, NUMERIC_PRECISION, NUMERIC_SCALE, IS_NULLABLE
FROM   INFORMATION_SCHEMA.COLUMNS
WHERE  TABLE_SCHEMA = 'fp_v2' AND TABLE_NAME = 'fp_basic_prices'
ORDER  BY ORDINAL_POSITION;
"""


def d8_volume_units_probe() -> str:
    """Settle the p_volume unit ambiguity.

    The other team computes (p_volume/1000)*price for ADTV, implying p_volume
    is RAW SHARES. Some FactSet documentation states volume ships in
    THOUSANDS. These contradict, and a 1000x error silently breaks every
    liquidity screen built on it.

    Resolution: look at a mega-cap on a known heavy day. AAPL trades on the
    order of 50-100 million shares/day. If p_volume shows ~5e7 it is raw
    shares; if ~5e4 it is thousands.
    """
    return """
SELECT TOP (20) t.ticker_region, p.p_date, p.p_price, p.p_volume
FROM   fp_v2.fp_basic_prices  p WITH (NOLOCK)
JOIN   sym_v1.sym_ticker_region t WITH (NOLOCK) ON t.fsym_id = p.fsym_id
WHERE  t.ticker_region IN ('AAPL-US','MSFT-US','SPY-US')
  AND  p.p_date BETWEEN '2024-01-02' AND '2024-01-10'
ORDER  BY t.ticker_region, p.p_date
OPTION (MAXDOP 2);
"""


DISCOVERY_QUERIES: dict[str, callable] = {
    "D1_table_inventory":    d1_table_inventory,
    "D2_columns":            d2_columns,
    "D3_indexes":            d3_indexes,
    "D3b_workload_check":    d3b_workload_check,
    "D4_exchange_map":       d4_exchange_map,
    "D5_security_type_maps": d5_security_type_maps,
    "D6_corporate_actions":  d6_corporate_action_hunt,
    "D7_price_columns":      d7_price_columns,
    "D8_volume_units":       d8_volume_units_probe,
}


# =============================================================================
# SECTION B -- UNIVERSE  (~34k rows: the security master)
# =============================================================================

def universe(include_active_filter: bool = False) -> str:
    """Resolve the US common-stock universe, INCLUDING DEAD SECURITIES.

    WHY EACH FIELD:

      fsym_regional_id       the PERMANENT price key. fp_basic_prices joins here.
      fsym_security_id       join key for sym_isin, own_v5, ent_v1.
      factset_entity_id      THE COMPANY. This is what groups share classes, so
                             GOOGL and GOOG resolve to a single issuer. Directly
                             solves the dual-class-pairs requirement: two
                             securities are two classes of one company iff they
                             share this id.
      fsym_primary_equity_id which share class is the primary one.
      fsym_primary_listing_id which venue is primary.
      security_name/entity_name  human readability in research output.
      ticker_region/isin/sedol   ATTRIBUTES ONLY, NEVER KEYS. All three are
                             reassignable - tickers get recycled between
                             unrelated companies, ISINs change on
                             redomiciliation. Kept for CRSP reconciliation,
                             which is now load-bearing since we are not pulling
                             FactSet's own total returns.
      currency               pulled so we can ASSERT it is single-valued, then
                             dropped from the fact table entirely.
      fref_security_type     coarse type screen.
      sec_type_code/desc     fine type screen (common vs units/trusts/etc).
      exchange_code/name     venue.
      listing/incorporation/domicile country
                             distinguishes US-LISTED from US-DOMICILED. A US
                             listing is what makes a name tradeable here; the
                             other two are useful research covariates.
      active_flag            STORED, NEVER FILTERED ON.
    SYMBOLOGY ONLY - no join to the price table. An earlier version derived
    first_px/last_px/n_obs here via a `span` CTE that aggregated the whole
    1.015-billion-row price table. It killed the connection every time (ODBC
    10053): the query runs for many minutes with no rows flowing, and something
    in the network path drops an idle TCP connection.

    Those life-span fields are still wanted - they are the survivorship evidence
    - but they are now derived LOCALLY in step 2 as MIN/MAX/COUNT per sid over
    the extracted Parquet. That is strictly better:
      * the same numbers, computed in seconds instead of a 1B-row server-side
        aggregation;
      * it removes the single most expensive statement from the office-day
        critical path, where a dropped connection is most costly;
      * no reason to make the server compute what we are about to have on disk.

    Consequence: the universe now includes securities that match the filters but
    have no prices in the window. They simply produce no price rows, and the
    step-2 life-span derivation marks them with n_obs = 0.

    include_active_filter=True is FOR MEASUREMENT ONLY: running this twice, with
    and without, quantifies your survivorship bias as a single number. Expect
    the filtered set to be roughly half the size. Keep both figures.
    """
    sec_type_filter = (
        f"  AND stc.p_sec_type_code IN ({_sql_list(config.SEC_TYPE_CODES)})"
        if config.SEC_TYPE_CODES
        else "  -- (no p_sec_type_code filter; set config.SEC_TYPE_CODES after query D5)"
    )
    active_filter = (
        "  AND sc.active_flag = 1   -- MEASUREMENT ONLY. Never in production."
        if include_active_filter
        else "  -- *** DELIBERATELY NO active_flag FILTER - SURVIVORSHIP BIAS ***"
    )
    return f"""
SELECT sc.fsym_id                     AS fsym_regional_id,
       sc.fsym_security_id            AS fsym_security_id,
       ent.factset_entity_id          AS factset_entity_id,
       sc.fsym_primary_equity_id      AS fsym_primary_equity_id,
       sc.fsym_primary_listing_id     AS fsym_primary_listing_id,
       sc.proper_name                 AS security_name,
       ec.entity_proper_name          AS entity_name,
       tic.ticker_region              AS ticker_region,
       isin.isin                      AS isin,
       sed.sedol                      AS sedol,
       sc.currency                    AS currency,
       sc.fref_security_type          AS fref_security_type,
       stc.p_sec_type_code            AS sec_type_code,
       stm.p_sec_type_desc            AS sec_type_desc,
       sc.fref_listing_exchange       AS exchange_code,
       ex.fref_exchange_desc          AS exchange_name,
       lc.country_desc                AS listing_country,
       ic.country_desc                AS incorporation_country,
       dc.country_desc                AS domicile_country,
       sc.active_flag                 AS active_flag
FROM       sym_v1.sym_coverage          sc   WITH (NOLOCK)
LEFT  JOIN fp_v2.fp_sec_entity          ent  WITH (NOLOCK)
       -- Keyed on fsym_primary_equity_id, NOT the regional id. fp_sec_entity has
       -- only 312k rows: one per company-equity, not one per regional listing.
       -- Joining on sc.fsym_id instead returns NULL for every row, which silently
       -- empties #universe_ent and makes every entity-keyed pull return 0 rows.
       -- Caught by the smoke test; the masterlist script had it right.
       ON ent.fsym_id = sc.fsym_primary_equity_id
LEFT  JOIN ent_v1.ent_entity_coverage   ec   WITH (NOLOCK)
       ON ec.factset_entity_id = ent.factset_entity_id
LEFT  JOIN fp_v2.fp_sec_coverage        stc  WITH (NOLOCK)
       ON stc.fsym_id = sc.fsym_id
LEFT  JOIN ref_v2.fp_sec_type_map       stm
       ON stm.p_sec_type_code = stc.p_sec_type_code
LEFT  JOIN ref_v2.fref_sec_exchange_map ex
       ON ex.fref_exchange_code = sc.fref_listing_exchange
LEFT  JOIN ref_v2.country_map           lc
       ON lc.iso_country = ex.fref_exchange_location_code
LEFT  JOIN ref_v2.country_map           ic
       ON ic.iso_country = ec.iso_country_incorp
LEFT  JOIN sym_v1.sym_entity            se   WITH (NOLOCK)
       ON se.factset_entity_id = ent.factset_entity_id
LEFT  JOIN ref_v2.country_map           dc
       ON dc.iso_country = se.iso_country
LEFT  JOIN sym_v1.sym_ticker_region     tic  WITH (NOLOCK)
       ON tic.fsym_id = sc.fsym_id
LEFT  JOIN sym_v1.sym_isin              isin WITH (NOLOCK)
       ON isin.fsym_id = sc.fsym_security_id     -- NB: keyed on the -S id
LEFT  JOIN sym_v1.sym_sedol             sed  WITH (NOLOCK)
       ON sed.fsym_id = sc.fsym_id
WHERE  sc.regional_flag = 1                       -- one row per regional series
  AND  sc.currency = 'USD'
  AND  sc.fref_security_type IN ({_sql_list(config.SECURITY_TYPES)})
  AND  sc.fref_listing_exchange IN ({_sql_list(config.US_EXCHANGE_CODES)})
{sec_type_filter}
{active_filter}
OPTION (MAXDOP {config.MAXDOP});
"""


# =============================================================================
# SECTION C -- PRICE FACT TABLE  (~92M rows: the core of the dataset)
# =============================================================================

# Candidate OHLC columns. *** UNVERIFIED *** - the other team selected only
# p_price and p_volume. d7_price_columns() confirms what actually exists, and
# the SELECT list is built from the intersection.
OHLC_CANDIDATES: dict[str, str] = {
    "px_open": "p_price_open",
    "px_high": "p_price_high",
    "px_low":  "p_price_low",
}


def prices_shard(where_clause: str,
                 available_ohlc: dict[str, str] | None = None) -> str:
    """The main fact-table pull, one shard at a time.

    WHY EACH FIELD:

      sid   int32 surrogate: 4 bytes instead of 8-40 bytes of fsym_id, 92M
            times, on a connection with no wire compression.
      d     trade date, CAST to `date` (3 B) rather than datetime (8 B).
      o/h/l OHLC, only if d7 confirms the columns exist.
      c     UNADJUSTED close - THE ANCHOR of the entire dataset. We store raw
            price plus an adjustment factor, never adjusted-only: a split
            tomorrow rewrites the whole history of an adjusted series and there
            is no way to detect that it happened.
      v     volume as BIGINT on the wire, narrowed to uint32 locally after
            asserting the observed max.
            *** NEVER CAST VOLUME TO `real` ***  float32 represents integers
            exactly only up to 2^24 = 16,777,216, so a heavily-traded penny
            stock silently corrupts - and that quietly poisons every liquidity
            filter downstream for years.
      ccy   pulled only to assert single-valuedness, then dropped.

    DELIBERATELY ABSENT: ORDER BY, OFFSET, and any adjustment arithmetic.
    """
    ohlc = OHLC_CANDIDATES if available_ohlc is None else available_ohlc
    ohlc_sql = "".join(
        f"\n       CAST(p.{src} AS real)      AS {tgt},"
        for tgt, src in ohlc.items()
    )
    return f"""
SELECT u.sid                          AS sid,{ohlc_sql}
       CAST(p.p_date   AS date)       AS d,
       CAST(p.p_price  AS real)       AS c,
       CAST(p.p_volume AS bigint)     AS v,
       p.currency                     AS ccy
FROM   fp_v2.fp_basic_prices p WITH (NOLOCK)
JOIN   #universe u ON u.fsym_id = p.fsym_id
WHERE  {where_clause}
OPTION (MAXDOP {config.MAXDOP});
"""


def shard_where_clauses() -> list[str]:
    """Shard predicates, per config.SHARD_MODE.

    Target 3-8 minutes per shard (~40-80 shards). A shard is both the restart
    unit and the blast radius: if the office wifi drops, a query times out, or a
    DBA kills the session, you lose at most one shard - not the whole run.

    SHARD_MODE must be chosen from discovery query D3. Sharding on a column with
    no supporting index means N full scans, which is worse than not sharding.
    """
    start, end = config.START_DATE, config.END_DATE

    if config.SHARD_MODE == "single_pass":
        return [f"p.p_date >= '{start}' AND p.p_date <= '{end}'"]

    if config.SHARD_MODE == "year":
        # END_DATE is deliberately open-ended (the feed's own max date is the
        # real bound), so clamp the shard list to avoid generating dozens of
        # empty future-year shards. Each empty shard is still a round trip, a
        # query compilation and a manifest entry for zero rows.
        y0 = int(start[:4])
        y1 = min(int(end[:4]), config.shard_max_year())
        return [
            f"p.p_date >= '{y}-01-01' AND p.p_date < '{y + 1}-01-01'"
            for y in range(y0, y1 + 1)
        ]

    if config.SHARD_MODE == "fsym_range":
        # Range-partition on sid via the #temp join, which is covered by our own
        # PRIMARY KEY, so the partition predicate is always index-supported.
        stride = config.FSYM_RANGE_STRIDE
        return [
            f"u.sid >= {b} AND u.sid < {b + stride} "
            f"AND p.p_date >= '{start}' AND p.p_date <= '{end}'"
            for b in range(0, config.FSYM_RANGE_BUCKETS * stride, stride)
        ]

    raise ValueError(f"unknown SHARD_MODE: {config.SHARD_MODE!r}")


# =============================================================================
# SECTION D -- SHARES, FLOAT, MARKET CAP
# =============================================================================

def shares_and_prices_own() -> str:
    """own_v5 provides two things nothing else in the feed does:

      1. unadj_shares_outstanding - required for market cap. UNADJUSTED, so it
         must be paired with the adjustment factor exactly like price is.
      2. unadj_price - an INDEPENDENT SECOND PRICE SOURCE, i.e. free
         cross-validation against fp_basic_prices. This matters more than usual
         because we are deliberately not pulling FactSet's own total returns, so
         independent checks are scarcer.

    Keyed on fsym_security_id (the -S id), NOT the regional id.
    own_sec_coverage.issue_type = 'EQ' is the equity screen within own_v5.
    """
    return f"""
SELECT u.sid                                       AS sid,
       CAST(op.price_date AS date)                 AS d,
       CAST(op.unadj_price AS real)                AS px_unadj_own,
       CAST(op.unadj_shares_outstanding AS float)  AS shares_out_unadj
FROM   own_v5.own_sec_prices   op WITH (NOLOCK)
JOIN   own_v5.own_sec_coverage oc WITH (NOLOCK) ON oc.fsym_id = op.fsym_id
JOIN   #universe_sec u ON u.fsym_security_id = op.fsym_id
WHERE  oc.issue_type = 'EQ'
  AND  op.price_date >= '{config.START_DATE}'
  AND  op.price_date <= '{config.END_DATE}'
OPTION (MAXDOP {config.MAXDOP});
"""


def free_float_hist() -> str:
    """Free-float history. Genuinely point-in-time (as_of_date), so it can be
    joined as-of the observation date rather than applied retroactively.

    Needed for float-adjusted market cap - i.e. a realistic tradeable-size
    estimate rather than a headline one. Small table: pull the whole history,
    never the current snapshot.
    """
    return f"""
SELECT u.sid                          AS sid,
       CAST(ff.as_of_date AS date)    AS as_of_date,
       CAST(ff.float_pct_os AS real)  AS float_pct_os
FROM   own_v5.own_float_hist ff WITH (NOLOCK)
JOIN   #universe_sec u ON u.fsym_security_id = ff.fsym_id
WHERE  ff.as_of_date >= '{config.START_DATE}'
OPTION (MAXDOP {config.MAXDOP});
"""


def entity_market_value() -> str:
    """Company-level market cap.

    Entity-level, so it is correctly SHARED across all share classes of one
    issuer rather than double-counted per class.
    ent_mv_ex_treasury excludes treasury stock.
    """
    return f"""
SELECT mv.factset_entity_id                    AS factset_entity_id,
       CAST(mv.mv_date AS date)                AS d,
       CAST(mv.ent_mv_ex_treasury AS float)    AS ent_mv_ex_treasury,
       mv.currency                             AS ccy
FROM   ent_v1.ent_entity_mkt_val mv WITH (NOLOCK)
JOIN   #universe_ent u ON u.factset_entity_id = mv.factset_entity_id
WHERE  mv.mv_date >= '{config.START_DATE}' AND mv.mv_date <= '{config.END_DATE}'
OPTION (MAXDOP {config.MAXDOP});
"""


# =============================================================================
# SECTION E -- SECTOR  (point-in-time status pending verification)
# =============================================================================

def sector() -> str:
    """Sector and industry classification. An ENTITY attribute, not a security
    one - so all share classes of a company share it, which is correct.

    *** POSSIBLE LOOK-AHEAD BIAS - VERIFY WITH D6(iii) ***
    sym_v1.sym_entity_sector as used by the other team shows no
    start_date/end_date, which would make it a CURRENT SNAPSHOT. Applying a
    snapshot to history classifies a 2007 company by its 2026 sector. That
    matters more here than in most applications, because sector membership
    DEFINES the PCA/clustering universe - so a snapshot leaks the future
    directly into pair selection.

    If no _hist variant exists, this must be documented as a measured known
    bias, not silently accepted.
    """
    return """
SELECT u.factset_entity_id            AS factset_entity_id,
       es.sector_code                 AS sector_code,
       sm.factset_sector_desc         AS sector_desc,
       es.industry_code               AS industry_code,
       im.factset_industry_desc       AS industry_desc
FROM   sym_v1.sym_entity_sector es WITH (NOLOCK)
JOIN   #universe_ent u ON u.factset_entity_id = es.factset_entity_id
LEFT   JOIN ref_v2.factset_sector_map   sm
       ON sm.factset_sector_code   = es.sector_code
LEFT   JOIN ref_v2.factset_industry_map im
       ON im.factset_industry_code = es.industry_code
OPTION (MAXDOP 2);
"""


# =============================================================================
# SECTION F -- CORPORATE ACTIONS  *** BLOCKED ON DISCOVERY QUERY D6 ***
# =============================================================================

CORPORATE_ACTIONS_BLOCKED_NOTE = """
-- ===========================================================================
-- NOT WRITTEN YET, DELIBERATELY.
--
-- Table and column names for splits / dividends / distributions / the
-- adjustment factor are UNVERIFIED. Run DISCOVERY_QUERIES['D6_corporate_
-- actions'] and these get written against real names.
--
-- Why not guess: the split-factor convention (is a 2-for-1 stored as 2.0 or
-- 0.5? does the factor multiply or divide the historical price?) determines the
-- DIRECTION of the adjustment. Get it backwards and 25 years of prices invert -
-- and the resulting series still looks completely plausible. It is the single
-- worst failure mode available in this dataset, so it gets verified.
--
-- Once names are known, this section needs:
--   splits     : fsym_id, ex/effective date, factor
--   dividends  : fsym_id, EX-DATE (the correct date for total-return maths),
--                amount, currency, dividend type code
--   adj factor : cumulative if FactSet ships one; otherwise derived as a
--                backwards cumulative product of split factors, in versioned
--                code with a unit test - never in a notebook cell
--   delisting  : date + reason, and any delisting RETURN. If FactSet ships no
--                delisting return, CRSP remains necessary for that single
--                field: a missing delisting return is exactly what silently
--                inflates backtest returns, because a position in a company
--                that goes to zero simply stops having P&L.
--
-- Validation once built: hand-checked tests against known US splits with
-- independently verified ratios -
--   AAPL 7:1 (2014-06-09), AAPL 4:1 (2020-08-31),
--   NVDA 4:1 (2021-07-20), NVDA 10:1 (2024-06-10),
--   TSLA 5:1 (2020-08-31), TSLA 3:1 (2022-08-25)
-- These pin the convention direction beyond doubt.
-- ===========================================================================
"""


def splits() -> str:
    """Split events. RESOLVED and convention-verified by discovery Q9.

    p_split_factor is the RECIPROCAL of the split ratio (AAPL's 7-for-1 is
    0.142857, its 4-for-1 is 0.25, its three 2-for-1s are 0.5). So historical
    prices are MULTIPLIED by the product of factors for all later splits.

    293k rows globally - pull the lot.
    """
    return f"""
SELECT u.sid                            AS sid,
       CAST(s.p_split_date AS date)     AS split_date,
       CAST(s.p_split_factor AS float)  AS split_factor
FROM   fp_v2.fp_basic_splits s WITH (NOLOCK)
JOIN   #universe u ON u.fsym_id = s.fsym_id
OPTION (MAXDOP {config.MAXDOP});
"""


def adjustment_factors() -> str:
    """Per-event price and total-return adjustment factors.

    adj_factor_combined     -> populated only for price-affecting capital events
                               (splits, spinoffs). Cumulative product gives the
                               PRICE-RETURN adjusted series.
    div_spl_spin_adj_factor -> populated for EVERY event including dividends
                               (~= 1 - div/price). Cumulative product gives the
                               TOTAL-RETURN adjusted series.

    These are PER-EVENT, not cumulative - we build the running product ourselves.
    That is why the derivation lives in versioned code with unit tests against
    known splits, not in a notebook cell.

    3.26M rows globally. Note this is fgp_v1 but keyed on the SAME -R regional id
    as fp_v2 (verified: AAPL is MH33D6-R in both).
    """
    return f"""
SELECT u.sid                                     AS sid,
       CAST(a.effective_date AS date)            AS effective_date,
       CAST(a.adj_factor_combined AS float)      AS adj_factor_combined,
       CAST(a.div_spl_spin_adj_factor AS float)  AS div_spl_spin_adj_factor
FROM   fgp_v1.fgp_ca_adj_factors a WITH (NOLOCK)
JOIN   #universe u ON u.fsym_id = a.fsym_id
OPTION (MAXDOP {config.MAXDOP});
"""


def dividends() -> str:
    """Cash and non-cash distributions.

    p_divs_exdate is the EX-DATE, which is the correct date for total-return
    maths (not pay date, not record date).

    p_divs_pd_type_code matters more than it looks: type 11 is a special
    dividend, 200/1055 are spinoffs, and 73/74/16 are liquidation
    distributions - the closest thing this feed has to a delisting return.
    Discovery Q15 showed only 374 liquidation records globally, which is why
    CRSP is still required for proper delisting returns.

    7.0M rows globally.
    """
    return f"""
SELECT u.sid                                  AS sid,
       CAST(d.p_divs_exdate AS date)          AS ex_date,
       CAST(d.p_divs_paydatec AS date)        AS pay_date,
       CAST(d.p_divs_recdatec AS date)        AS record_date,
       CAST(d.p_divs_pd AS float)             AS div_amount,
       d.currency                             AS ccy,
       d.p_divs_pd_type_code                  AS div_type_code,
       d.p_divs_pd_ngflag_code                AS ng_flag,
       CAST(d.p_divs_pd_ngequiv AS float)     AS div_ng_equiv,
       d.p_divs_s_spinoff                     AS is_spinoff,
       d.p_divs_s_pd                          AS is_special
FROM   fp_v2.fp_basic_dividends d WITH (NOLOCK)
JOIN   #universe u ON u.fsym_id = d.fsym_id
OPTION (MAXDOP {config.MAXDOP});
"""


def ca_events() -> str:
    """Rich corporate-action event table (40 columns in source; we take the ones
    a backtester needs).

    dist_inst_fsym_id is the distributed instrument - it is what makes proper
    SPINOFF handling possible, i.e. knowing which security the shareholder
    received rather than just that value left the parent.

    dist_old_term / dist_new_term give split terms in ratio form, which is an
    independent cross-check on p_split_factor from fp_basic_splits.

    3.76M rows globally.
    """
    return f"""
SELECT u.sid                                     AS sid,
       e.ca_event_id                             AS ca_event_id,
       e.ca_event_type_code                      AS event_type,
       CAST(e.effective_date AS date)            AS effective_date,
       CAST(e.announcement_date AS date)         AS announcement_date,
       CAST(e.pay_date AS date)                  AS pay_date,
       CAST(e.record_date AS date)               AS record_date,
       CAST(e.price_adj_factor AS float)         AS price_adj_factor,
       CAST(e.dist_old_term AS float)            AS dist_old_term,
       CAST(e.dist_new_term AS float)            AS dist_new_term,
       CAST(e.dist_pct AS float)                 AS dist_pct,
       CAST(e.amt_gross_trading_unadj AS float)  AS amt_gross_unadj,
       CAST(e.amt_net_trading_unadj AS float)    AS amt_net_unadj,
       e.trading_currency                        AS ccy,
       e.div_type_code                           AS div_type_code,
       e.dividend_spec_flag                      AS spec_flag,
       e.dividend_active_flag                    AS active_flag,
       e.dist_inst_fsym_id                       AS dist_inst_fsym_id
FROM   fgp_v1.fgp_ca_events e WITH (NOLOCK)
JOIN   #universe u ON u.fsym_id = e.fsym_id
OPTION (MAXDOP {config.MAXDOP});
"""


def shares_outstanding() -> str:
    """Common shares outstanding history.

    Preferred over own_v5.own_sec_prices.unadj_shares_outstanding: that table has
    only 24.2M rows globally, so it is month-end, whereas this is a proper daily
    history at 35.4M rows.

    UNADJUSTED, so it must be paired with the adjustment factor exactly as price
    is - otherwise market cap mixes an adjusted price with a raw share count,
    which is the specific bug the legacy CRSP code had.
    """
    return f"""
SELECT u.sid                                AS sid,
       CAST(sh.p_date AS date)              AS d,
       CAST(sh.p_com_shs_out AS float)      AS com_shs_out
FROM   fp_v2.fp_basic_shares_hist sh WITH (NOLOCK)
JOIN   #universe u ON u.fsym_id = sh.fsym_id
WHERE  sh.p_date >= '{config.START_DATE}'
OPTION (MAXDOP {config.MAXDOP});
"""


def trading_calendar() -> str:
    """FactSet's own exchange calendar. Strictly better than deriving the
    calendar from one stock's observed dates, which is what the legacy code does
    and which also bakes in that stock's trading halts.

    eom_flag gives month-end rebalance dates directly.
    """
    return """
SELECT CAST(d.ref_date AS date) AS ref_date,
       d.day_of_week            AS day_of_week,
       d.eom_flag               AS eom_flag
FROM   ref_v2.ref_calendar_dates d
WHERE  d.ref_date >= '1990-01-01'
OPTION (MAXDOP 2);
"""


def exchange_holidays() -> str:
    """Per-exchange holidays, for the US venues we care about."""
    return f"""
SELECT h.fref_exchange_code            AS exchange_code,
       CAST(h.holiday_date AS date)    AS holiday_date,
       h.holiday_name                  AS holiday_name
FROM   ref_v2.ref_calendar_holidays h
WHERE  h.fref_exchange_code IN ({_sql_list(config.US_EXCHANGE_CODES)})
OPTION (MAXDOP 2);
"""


# =============================================================================
# SECTION G -- THE ORACLE  (server-authored ground truth to carry home)
# =============================================================================

def oracle_checksum() -> str:
    """Carry this home. You cannot re-pull from the home laptop, so you need a
    server-authored ground truth to validate the local Parquet against.

    SUM of SCALED INTEGERS is exact and order-independent, so it reconciles
    bit-for-bit against a local aggregate. SUM of floats would NOT:
    floating-point addition is not associative, so a different row order gives a
    different sum. CHECKSUM_AGG is order-independent but collision-prone and
    type-quirky, so it is not sufficient on its own.

    At home, the identical aggregate over the Parquet must match every column
    for every year:
      n_rows differs             -> a dropped shard, or a NOLOCK page-split skip
      only px_checksum differs   -> a CAST / precision bug

    This test is what separates a database you can bet money on from one you
    merely hope is right.
    """
    return f"""
SELECT YEAR(p.p_date)                  AS yr,
       COUNT_BIG(*)                    AS n_rows,
       COUNT(DISTINCT p.fsym_id)       AS n_ids,
       MIN(p.p_date)                   AS d_min,
       MAX(p.p_date)                   AS d_max,
       SUM(CAST(ROUND(CAST(p.p_price AS float), 4) * 10000 AS BIGINT)) AS px_checksum,
       SUM(CAST(p.p_volume AS BIGINT)) AS vol_checksum
FROM   fp_v2.fp_basic_prices p WITH (NOLOCK)
JOIN   #universe u ON u.fsym_id = p.fsym_id
WHERE  p.p_date >= '{config.START_DATE}' AND p.p_date <= '{config.END_DATE}'
GROUP  BY YEAR(p.p_date)
ORDER  BY yr
OPTION (MAXDOP 2);
"""


def oracle_universe_counts() -> str:
    """Second oracle: distinct securities per year.

    Expected shape for a correct US common-stock universe: ~7,000-8,000 in the
    late 1990s, declining to ~4,000-4,500 by 2020. A FLAT OR RISING line means
    survivorship bias or non-common-stock contamination - so this doubles as a
    validation test, not just a count.
    """
    return f"""
SELECT YEAR(p.p_date)            AS yr,
       COUNT(DISTINCT p.fsym_id) AS n_ids
FROM   fp_v2.fp_basic_prices p WITH (NOLOCK)
JOIN   #universe u ON u.fsym_id = p.fsym_id
WHERE  p.p_date >= '{config.START_DATE}' AND p.p_date <= '{config.END_DATE}'
GROUP  BY YEAR(p.p_date)
ORDER  BY yr
OPTION (MAXDOP 2);
"""
