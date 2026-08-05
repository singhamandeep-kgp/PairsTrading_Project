/* ============================================================================
   FactSet discovery -- resolves the 7 blockers in extract_raw_factset.py

   Run in SSMS against the FactSet server and database named in your .env
   (FACTSET_SQL_SERVER / FACTSET_SQL_DATABASE).
   Run block by block (highlight + F5). Every query is READ-ONLY.

   Deliberately scoped so each result set is SMALL enough to paste back.
   Q1 alone closes about half the blockers.

   Estimated time: 10-15 minutes, almost all of it Q7.
   ============================================================================ */

USE fds_datafeeds;
GO

SET TRANSACTION ISOLATION LEVEL READ UNCOMMITTED;  -- never block the loader
GO


/* ============================================================================
   Q1 -- CAPABILITY SUMMARY.  ONE ROW. Start here.
   Closes blockers: D2/D7 (does OHLC exist), and tells us where corporate
   actions and point-in-time history live, without dumping every column.
   ============================================================================ */

SELECT
  -- *** THE OHLCV QUESTION *** (highest-priority blocker)
  MAX(CASE WHEN TABLE_NAME='fp_basic_prices' AND COLUMN_NAME='p_price_open' THEN 1 ELSE 0 END) AS bp_has_open,
  MAX(CASE WHEN TABLE_NAME='fp_basic_prices' AND COLUMN_NAME='p_price_high' THEN 1 ELSE 0 END) AS bp_has_high,
  MAX(CASE WHEN TABLE_NAME='fp_basic_prices' AND COLUMN_NAME='p_price_low'  THEN 1 ELSE 0 END) AS bp_has_low,
  MAX(CASE WHEN TABLE_NAME='fp_basic_prices' AND COLUMN_NAME='p_volume'     THEN 1 ELSE 0 END) AS bp_has_volume,
  -- wire cost: 8 vs 40 bytes/row on fsym_id, over ~92M rows
  MAX(CASE WHEN TABLE_NAME='fp_basic_prices' AND COLUMN_NAME='fsym_id'
           THEN DATA_TYPE END)                                              AS bp_fsym_id_type,
  MAX(CASE WHEN TABLE_NAME='fp_basic_prices' AND COLUMN_NAME='p_price'
           THEN DATA_TYPE END)                                              AS bp_price_type,
  MAX(CASE WHEN TABLE_NAME='fp_basic_prices' AND COLUMN_NAME='p_volume'
           THEN DATA_TYPE END)                                              AS bp_volume_type,
  -- does a point-in-time sector history table exist?
  MAX(CASE WHEN TABLE_NAME='sym_entity_sector_hist' THEN 1 ELSE 0 END)      AS has_sector_hist,
  MAX(CASE WHEN TABLE_NAME LIKE '%rbics%'           THEN 1 ELSE 0 END)      AS has_rbics
FROM INFORMATION_SCHEMA.COLUMNS
WHERE TABLE_SCHEMA IN ('fp_v2','fgp_v1','sym_v1','ent_v1','ref_v2','own_v5');
GO


/* ============================================================================
   Q2 -- CORPORATE ACTIONS + ADJUSTMENT FACTOR.  *** THE BIGGEST BLOCKER ***

   The other team's production query has NO adjustment factor, NO corporate
   actions and NO total return -- correct for a point-in-time index masterlist,
   useless for a backtester that needs a continuous adjusted series.

   I am not guessing these names. A wrong split convention (is a 2-for-1 stored
   as 2.0 or 0.5? does it multiply or divide?) inverts 25 years of prices into a
   series that still looks entirely plausible.

   Expect ~20-80 rows. Paste all of it.
   ============================================================================ */

-- Q2a: tables whose NAME suggests corporate actions / history
SELECT s.name AS sch, t.name AS tbl
FROM   sys.tables t JOIN sys.schemas s ON s.schema_id = t.schema_id
WHERE  s.name IN ('fp_v2','fgp_v1','sym_v1','ent_v1','ref_v2','own_v5')
  AND (t.name LIKE '%split%'   OR t.name LIKE '%div%'
    OR t.name LIKE '%distrib%' OR t.name LIKE '%action%'
    OR t.name LIKE '%adj%'     OR t.name LIKE '%spin%'
    OR t.name LIKE '%delist%'  OR t.name LIKE '%return%'
    OR t.name LIKE '%shares%'  OR t.name LIKE '%hist%')
ORDER BY s.name, t.name;

-- Q2b: COLUMNS suggesting an adjustment factor, anywhere in the feed
SELECT TABLE_SCHEMA, TABLE_NAME, COLUMN_NAME, DATA_TYPE
FROM   INFORMATION_SCHEMA.COLUMNS
WHERE  TABLE_SCHEMA IN ('fp_v2','fgp_v1','sym_v1','ent_v1','ref_v2','own_v5')
  AND (COLUMN_NAME LIKE '%adj%'    OR COLUMN_NAME LIKE '%split%'
    OR COLUMN_NAME LIKE '%factor%' OR COLUMN_NAME LIKE '%cum%'
    OR COLUMN_NAME LIKE '%delist%' OR COLUMN_NAME LIKE '%div%')
ORDER BY TABLE_SCHEMA, TABLE_NAME, COLUMN_NAME;

-- Q2c: POINT-IN-TIME check. Any table with BOTH start_date and end_date.
--      If sym_entity_sector is NOT here, it is a current snapshot -- and
--      applying a snapshot to history leaks the future into the clustering
--      universe, because sector membership is what DEFINES that universe.
SELECT TABLE_SCHEMA, TABLE_NAME
FROM   INFORMATION_SCHEMA.COLUMNS
WHERE  COLUMN_NAME IN ('start_date','end_date')
GROUP  BY TABLE_SCHEMA, TABLE_NAME
HAVING COUNT(*) = 2
ORDER  BY TABLE_SCHEMA, TABLE_NAME;
GO


/* ============================================================================
   Q3 -- INDEXES on the price table.  Decides config.SHARD_MODE.

   If nothing leads with fsym_id or p_date, we must NOT shard: each of N shards
   would trigger its own full scan of ~120 GB of pages, which is slower than one
   sequential pass and cache-thrashes the buffer pool for every other user.

   Expect 1-6 rows.
   ============================================================================ */

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
ORDER  BY t.name, i.index_id;
GO


/* ============================================================================
   Q4 -- SIZE + DATE COVERAGE.  Row counts from metadata, so instant.
   Confirms the ~1.01B figure, gives the real feed max date (set this as
   FACTSET_FEED_MAX_YEAR in .env), and reserved_mb is the full-scan wire ceiling.
   ============================================================================ */

SELECT s.name AS sch, t.name AS tbl,
       SUM(CASE WHEN ps.index_id IN (0,1) THEN ps.row_count ELSE 0 END) AS [rows],
       CAST(SUM(ps.reserved_page_count)*8.0/1024 AS DECIMAL(14,1))      AS reserved_mb
FROM   sys.dm_db_partition_stats ps
JOIN   sys.tables  t ON t.object_id = ps.object_id
JOIN   sys.schemas s ON s.schema_id = t.schema_id
WHERE  s.name IN ('fp_v2','fgp_v1','sym_v1','ent_v1','ref_v2','own_v5')
GROUP  BY s.name, t.name
HAVING SUM(CASE WHEN ps.index_id IN (0,1) THEN ps.row_count ELSE 0 END) > 0
ORDER  BY [rows] DESC;

-- Real date bounds. If this is slow (>2 min), that itself tells us about Q3.
SELECT MIN(p_date) AS d_min, MAX(p_date) AS d_max
FROM   fp_v2.fp_basic_prices WITH (NOLOCK)
OPTION (MAXDOP 2);
GO


/* ============================================================================
   Q5 -- p_volume UNITS.  Raw shares or thousands?

   The other team computes (p_volume/1000)*price for ADTV, implying RAW SHARES.
   Some FactSet docs say volume ships in THOUSANDS. A 1000x error silently
   breaks every liquidity screen.

   AAPL trades ~50-100 MILLION shares/day. So:
     p_volume ~ 5e7  -> raw shares
     p_volume ~ 5e4  -> thousands
   ============================================================================ */

SELECT TOP (10) t.ticker_region, p.p_date, p.p_price, p.p_volume
FROM   fp_v2.fp_basic_prices    p WITH (NOLOCK)
JOIN   sym_v1.sym_ticker_region t WITH (NOLOCK) ON t.fsym_id = p.fsym_id
WHERE  t.ticker_region = 'AAPL-US'
  AND  p.p_date BETWEEN '2024-01-02' AND '2024-01-09'
ORDER  BY p.p_date
OPTION (MAXDOP 2);
GO


/* ============================================================================
   Q6 -- UNIVERSE VOCABULARIES.  Closes D4 and D5.

   Do NOT copy the other team's 19-code p_sec_type list: it is tuned for a
   GLOBAL MULTI-TYPE index, not a US common-stock universe. Read the
   descriptions and pick.
   ============================================================================ */

-- Q6a: US exchange codes only (filtered, so ~10-30 rows not 300)
SELECT ex.fref_exchange_code, ex.fref_exchange_desc, cm.country_desc
FROM   ref_v2.fref_sec_exchange_map ex
JOIN   ref_v2.country_map cm ON cm.iso_country = ex.fref_exchange_location_code
WHERE  cm.country_desc LIKE '%United States%'
ORDER  BY ex.fref_exchange_code;

-- Q6b: security-type vocabularies, restricted to what actually occurs in a
--      USD equity universe -- so you see live codes, not the whole global map
SELECT sc.fref_security_type,
       stc.p_sec_type_code,
       stm.p_sec_type_desc,
       COUNT(*) AS n_securities
FROM   sym_v1.sym_coverage      sc  WITH (NOLOCK)
LEFT   JOIN fp_v2.fp_sec_coverage stc WITH (NOLOCK) ON stc.fsym_id = sc.fsym_id
LEFT   JOIN ref_v2.fp_sec_type_map stm ON stm.p_sec_type_code = stc.p_sec_type_code
WHERE  sc.regional_flag = 1
  AND  sc.currency = 'USD'
GROUP  BY sc.fref_security_type, stc.p_sec_type_code, stm.p_sec_type_desc
HAVING COUNT(*) >= 20            -- drop long-tail noise
ORDER  BY n_securities DESC;
GO


/* ============================================================================
   Q7 -- THE SANITY CHECK + THE SURVIVORSHIP NUMBER.  Run this LAST.

   Distinct US common-stock securities per year. Expected shape:
     ~7,000-8,000 in the late 1990s, declining to ~4,000-4,500 by 2020.
     A FLAT or RISING line means survivorship bias or non-common contamination.

   Also replaces my ~92M row ESTIMATE with a MEASUREMENT.

   This one may take several minutes. If it exceeds ~10 min, cancel it and
   re-run with "AND p.p_date >= '2015-01-01'" to sample instead.

   NOTE the two variants. The DIFFERENCE between them is your survivorship bias,
   quantified -- keep both numbers, it is a genuinely good thing to be able to
   put in a research note.
   ============================================================================ */

-- Q7a: CORRECT universe -- no active_flag filter
SELECT YEAR(p.p_date)            AS yr,
       COUNT(DISTINCT p.fsym_id) AS n_ids,
       COUNT_BIG(*)              AS n_rows
FROM   fp_v2.fp_basic_prices p WITH (NOLOCK)
JOIN   sym_v1.sym_coverage   sc WITH (NOLOCK) ON sc.fsym_id = p.fsym_id
WHERE  sc.regional_flag = 1
  AND  sc.currency = 'USD'
  AND  sc.fref_security_type = 'SHARE'
  AND  p.p_date >= '1995-01-01'
  -- *** NO active_flag FILTER ***
GROUP  BY YEAR(p.p_date)
ORDER  BY yr
OPTION (MAXDOP 2);

-- Q7b: the BIASED universe, for measurement only
SELECT YEAR(p.p_date)            AS yr,
       COUNT(DISTINCT p.fsym_id) AS n_ids_active_only
FROM   fp_v2.fp_basic_prices p WITH (NOLOCK)
JOIN   sym_v1.sym_coverage   sc WITH (NOLOCK) ON sc.fsym_id = p.fsym_id
WHERE  sc.regional_flag = 1
  AND  sc.currency = 'USD'
  AND  sc.fref_security_type = 'SHARE'
  AND  sc.active_flag = 1          -- <<< the bias, made visible
  AND  p.p_date >= '1995-01-01'
GROUP  BY YEAR(p.p_date)
ORDER  BY yr
OPTION (MAXDOP 2);
GO


/* ============================================================================
   WHAT TO SEND BACK
   ----------------------------------------------------------------------------
   Q1  the single row  (settles OHLCV + wire cost + sector history)
   Q2  all three result sets  <-- MOST IMPORTANT
   Q3  the index rows
   Q4  the table list + the min/max dates
   Q5  the 10 AAPL rows
   Q6  both result sets
   Q7  both year tables

   If Q7 is too slow, skip it -- everything else matters more.
   If any query errors on an object name, send the error: a missing table name
   is itself information.
   ============================================================================ */
