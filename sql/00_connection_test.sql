/* ============================================================
   FactSet-on-MSSQL : CONNECTION SMOKE TEST
   ------------------------------------------------------------
   Goal: prove the connection works and that we can read real
   rows, WITHOUT scanning a billion-row table.

   Connect in SSMS first, using the values from your .env
   (FACTSET_SQL_SERVER / FACTSET_SQL_DATABASE / FACTSET_SQL_USER):
     Server  : <host-or-ip>
     Auth    : SQL Server Authentication
     Login   : <username>
     Password: (yours; never write it into a file)
     If it refuses to connect, tick
     "Trust server certificate" under Connection Properties.

   Run each STEP separately (highlight + F5). Do not run the
   whole file at once - step 2 tells you what to type in step 3.
   ============================================================ */


/* ---------- STEP 1 : am I actually connected? ---------------
   Costs nothing. If this returns rows, the network path,
   firewall, and credentials are all fine.                    */

SELECT @@VERSION                          AS version_string,
       @@SERVERNAME                       AS server_name,
       DB_NAME()                          AS current_database,
       SUSER_SNAME()                      AS login_name,
       USER_NAME()                        AS db_user,
       GETUTCDATE()                       AS utc_now,
       SERVERPROPERTY('Edition')          AS edition,
       SERVERPROPERTY('ProductVersion')   AS product_version;
GO


/* ---------- STEP 2 : which databases can I see? -------------
   Then: which one actually holds the FactSet schemas?
   The second query checks every database you have access to
   and reports where fp_v2 / fgp_v1 / sym_v1 / ent_v1 live.   */

SELECT database_id, name, state_desc, is_read_only
FROM   sys.databases
ORDER  BY name;

-- Find the database containing the FactSet schemas:
DECLARE @sql nvarchar(max) = N'';

SELECT @sql = @sql + N'
SELECT ' + QUOTENAME(name, '''') + N' AS db_name,
       s.name AS schema_name,
       COUNT(t.object_id) AS n_tables
FROM   ' + QUOTENAME(name) + N'.sys.schemas s
LEFT   JOIN ' + QUOTENAME(name) + N'.sys.tables t
       ON t.schema_id = s.schema_id
WHERE  s.name IN (''fp_v2'',''fgp_v1'',''sym_v1'',''ent_v1'')
GROUP  BY s.name
HAVING COUNT(t.object_id) > 0;'
FROM   sys.databases
WHERE  state = 0                       -- ONLINE only
  AND  HAS_DBACCESS(name) = 1          -- only ones I can read
  AND  database_id > 4;                -- skip master/tempdb/model/msdb

EXEC sp_executesql @sql;
GO


/* ---------- STEP 3 : point at the right database ------------
   Replace <DB> with the db_name from step 2, then run.       */

USE <DB>;
GO

SELECT DB_NAME() AS now_using;

-- Every schema here, and how many tables each holds:
SELECT s.name AS schema_name, COUNT(t.object_id) AS n_tables
FROM   sys.schemas s
LEFT   JOIN sys.tables t ON t.schema_id = s.schema_id
GROUP  BY s.name
HAVING COUNT(t.object_id) > 0
ORDER  BY n_tables DESC;
GO


/* ---------- STEP 4 : what are the tables really called? -----
   Do NOT assume fp_basic_prices / fp_total_returns_daily
   exist under those exact names. Confirm first.
   Row counts come from metadata, so this is instant even on
   a 1-billion-row table. Never use COUNT(*) for this.        */

SELECT s.name AS sch,
       t.name AS tbl,
       SUM(CASE WHEN ps.index_id IN (0,1) THEN ps.row_count ELSE 0 END) AS [rows],
       CAST(SUM(ps.reserved_page_count) * 8.0 / 1024 AS DECIMAL(14,1))  AS reserved_mb
FROM   sys.dm_db_partition_stats ps
JOIN   sys.tables  t ON t.object_id = ps.object_id
JOIN   sys.schemas s ON s.schema_id = t.schema_id
WHERE  s.name IN ('fp_v2','fgp_v1','sym_v1','ent_v1')
GROUP  BY s.name, t.name
ORDER  BY [rows] DESC;
GO


/* ---------- STEP 5 : THE DUMMY READ ------------------------
   TOP (10) with no WHERE and no ORDER BY, so the engine stops
   after 10 rows instead of scanning the table. This is the
   cheapest possible real read.

   SELECT * on purpose here: we want to SEE the actual column
   names and types rather than guess them.

   Adjust the table names from step 4 if they differ.        */

SET TRANSACTION ISOLATION LEVEL READ UNCOMMITTED;  -- take no shared locks

SELECT TOP (10) * FROM fp_v2.fp_basic_prices        WITH (NOLOCK);
SELECT TOP (10) * FROM fp_v2.fp_total_returns_daily WITH (NOLOCK);
SELECT TOP (10) * FROM fgp_v1.fgp_global_prices     WITH (NOLOCK);
SELECT TOP (10) * FROM sym_v1.sym_coverage          WITH (NOLOCK);
GO


/* ---------- STEP 6 : column names and types ----------------
   The wire-cost query. Two things to look for:
     - fsym_id as varchar (8 B/row) or nvarchar (40 B/row)
     - prices as real (4 B) or float/decimal (8 B)
   Over ~75 million rows those choices are worth GBs of
   transfer time, so note what you see.                      */

SELECT TABLE_SCHEMA, TABLE_NAME, ORDINAL_POSITION, COLUMN_NAME,
       DATA_TYPE, CHARACTER_MAXIMUM_LENGTH AS char_len,
       NUMERIC_PRECISION, NUMERIC_SCALE, IS_NULLABLE
FROM   INFORMATION_SCHEMA.COLUMNS
WHERE  TABLE_SCHEMA IN ('fp_v2','fgp_v1','sym_v1','ent_v1')
ORDER  BY TABLE_SCHEMA, TABLE_NAME, ORDINAL_POSITION;
GO


/* ---------- STEP 7 : one recognisable stock ----------------
   Proves the data is real and joinable, not just that rows
   exist. Looks up Apple by ticker, then pulls its last few
   prices.

   NOTE: if step 4 showed no index on fsym_id, the second
   query may scan. It is bounded by TOP (20) but SQL Server
   still has to find matching rows first. If it runs longer
   than ~2 minutes, CANCEL IT - that itself tells us the
   table is a heap, which changes the extraction design.     */

-- 7a. Resolve a ticker to a FactSet id (small table, fast):
SELECT TOP (20) *
FROM   sym_v1.sym_ticker_region WITH (NOLOCK)
WHERE  ticker_region = 'AAPL-US';

-- 7b. Then paste the fsym_id you got above into this:
SELECT TOP (20) *
FROM   fp_v2.fp_basic_prices WITH (NOLOCK)
WHERE  fsym_id = '<PASTE_FSYM_ID_HERE>'
ORDER  BY p_date DESC
OPTION (MAXDOP 2);
GO


/* ---------- STEP 8 : how fast is the link? -----------------
   OPTIONAL, run last. Reads 1 million rows of 3 narrow
   columns (~12 MB on the wire) and lets you time it in
   SSMS's bottom-right corner.

   Divide 12 MB by the elapsed seconds to get your effective
   throughput. That single number determines whether the full
   extraction takes 15 minutes or 3 hours.

   Start with TOP (100000) if you want to be cautious.       */

SELECT TOP (1000000) fsym_id, p_date, p_price
FROM   fp_v2.fp_basic_prices WITH (NOLOCK)
OPTION (MAXDOP 2);
GO


/* ============================================================
   WHAT TO REPORT BACK
   ------------------------------------------------------------
   1. Did step 1 return rows?           (connection works)
   2. Which database from step 2?
   3. Step 4: real table names + row counts
   4. Step 5: do the TOP 10 reads return sensible data?
   5. Step 6: is fsym_id varchar or nvarchar?
              are prices real, float, or decimal?
   6. Step 7: does AAPL resolve, and did 7b return quickly?
   7. Step 8: elapsed time for 1M rows
   ============================================================ */
