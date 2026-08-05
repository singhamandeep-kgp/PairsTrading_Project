# US Equity Research Database — Technical Design Record

**Written:** 2026-07-29 · **Last reviewed:** 2026-08-05
**Status:** **IMPLEMENTED.** Stages 1 and 2 shipped; see the status note below.
**Source:** FactSet Standard DataFeed on Microsoft SQL Server → local Parquet/Delta lake

> **How to read this document.** It was written *before* implementation, as a
> design argument, and it is preserved in that form because the reasoning is the
> point. Where reality differed from the design, the difference is recorded here
> rather than edited away:
>
> | Designed | Actually shipped |
> |---|---|
> | DuckDB as the query engine | **Polars + Delta Lake.** DuckDB is unused; `pl.scan_delta` gave predicate pushdown without a second engine. |
> | 2000 start, ~75.6M rows | **1995 start, 39.7M rows.** The universe was ~16.4k securities, not the ~30k estimated — roughly half. |
> | Sector partition for clustering | **Dropped.** FactSet's point-in-time sector tables (`rbics_v1`) are empty in this feed, so sector would have injected look-ahead. See DISCOVERY_FINDINGS §11. |
> | Total returns extracted from the vendor | **Derived locally** from raw price × cumulative factor. |
> | Python 3.13 pinned for wheel availability | **3.14 worked.** `pyarrow`, `arrow-odbc`, `deltalake` and `polars` all had cp314 wheels. |
>
> **Open items are tracked in [../ROADMAP.md](../ROADMAP.md)** — that is the single
> list. This document records what was decided and why; it is not a to-do list.
> Empirical results from the live server are in
> [DISCOVERY_FINDINGS.md](DISCOVERY_FINDINGS.md), which **overturns three
> assumptions made here** and should be read alongside §13–14.

This document is the complete record of the technical decisions for building the database, and *why* each was chosen over the alternative. It is written to be defensible: every claim has arithmetic or a cited mechanism behind it.

---

## Table of contents

1. [The environment, and why it dictates the design](#1-the-environment)
2. [Universe-first: the load-bearing architectural idea](#2-universe-first)
3. [Bias elimination: survivorship and look-ahead](#3-bias-elimination)
4. [Extraction: getting bytes off SQL Server](#4-extraction)
5. [Working without DDL permissions](#5-working-without-ddl)
6. [Sharding and the index trap](#6-sharding-and-the-index-trap)
7. [Storage format and maximum compression](#7-storage-and-compression)
8. [Schema design](#8-schema-design)
9. [Two-phase build and resumability](#9-two-phase-build)
10. [Verification and data quality](#10-verification)
11. [Concurrency: what helps, what is cargo cult](#11-concurrency)
12. [Tooling and environment](#12-tooling)
13. [Discovery script — run this first](#13-discovery-script)
14. [Open unknowns](#14-open-unknowns)
15. [Condensed decision table](#15-condensed-decision-table)

---

## 1. The environment

Everything below follows from these facts, which were discovered by inspection rather than assumed.

| Fact | Value | Design consequence |
|---|---|---|
| Transport | **Microsoft SQL Server**, self-managed on AWS **ap-south-1 (Mumbai)** EC2, public IP, SQL Server auth | Not a cloud warehouse. `fp_v2`/`fgp_v1` are SQL Server *schemas*. |
| Server capability | **Cannot emit Parquet. No cloud stage. No elastic compute.** | All compression and sorting happens client-side. |
| Wire protocol | **TDS has no built-in compression** | Bytes on the wire ≈ raw row bytes. Column pruning and server-side `CAST` become the entire optimisation surface. |
| Client tooling | SSMS 22, **ODBC Driver 18 for SQL Server** present; `bcp`/`sqlcmd` absent | ODBC is the path. |
| Install rights | **pip only, no admin/MSI** | Rules out `bcp` (mssql-tools) and the winget-installed ADBC driver. |
| Permissions | Assume plain read-only user on a shared production box | No DDL, no index creation, no server filesystem access. |
| Network | Office (India) → EC2 Mumbai; low RTT, office-link-limited bandwidth | Measure it — it determines whether office day is 15 min or 3 h. |
| **Server reachability** | **Office network only** | **One shot at extraction.** Cannot re-pull from home. |
| Target machine | **8 GB RAM**, ~200 GB free | Out-of-core mandatory. |
| Python | 3.14.6 installed; pandas 3.0.5, numpy 2.5.1 | 3.14 has a wheel problem — see §12. |

### 1.1 Why "no server-side compression" is the defining constraint

In a Snowflake or BigQuery build you write:

```sql
COPY INTO @stage FROM (SELECT ... ORDER BY id, date)
FILE_FORMAT = (TYPE = PARQUET COMPRESSION = ZSTD)
```

and the warehouse's 8–128 cores do the compression *and* the sort for free, then hand you ~1 GB of Parquet. **SQL Server can do neither.** This inverts three pieces of standard advice:

1. Server-side `CAST` narrowing goes from cosmetic to **mandatory** — the SELECT list *is* the wire format.
2. Substituting an integer surrogate key for the vendor ID string becomes a **wire-compression device**, not just an in-memory optimisation.
3. **`ORDER BY` in the extraction query goes from free to dangerous** — see §4.4.

---

## 2. Universe-first

**Never touch the 1.01-billion-row price table until you know exactly which securities you want.**

Resolve the universe from the symbology tables (~10⁶ rows — small and cheap). That yields ~30,000 securities. Only then does the price pull happen, as a targeted join rather than a global scan.

```
PHASE 1 (seconds)   sym_coverage + exchange/type filters + entity link
                    *** NO active_flag FILTER ***
                              ↓  ~30k rows, ~1 MB
PHASE 2 (minutes)   fp_basic_prices ⋈ #temp(sid, fsym_id)
                    7 columns, CAST-narrowed, sid on the wire, 2000+
                              ↓
PHASE 3 (at home)   Parquet: int32 sid, float32 px, sorted, ZSTD-9
```

### 2.1 The arithmetic

Wire cost, uncompressed TDS plus ~10% framing:

| Scenario | Bytes/row | Total | @100 Mbps |
|---|---:|---:|---:|
| Naive `SELECT *`, no universe filter (1.01B × ~120 B) | 120 | **121 GB** | **2.7 h** |
| Universe-filtered, no CASTs | 96 | 7.26 GB | 9.7 min |
| + server-side CASTs | 43 | 3.25 GB | 4.3 min |
| **+ `sid` substitution** | 35 | **2.65 GB** | **3.5 min** |

Decomposed: universe filter + column projection **16.7×**, server-side CAST narrowing **2.2×**, `sid` substitution **1.23×** → **45.7× combined**.

That is not merely "faster." It converts extraction from an all-day irreversible ordeal into something retryable — which, given office-only access, *is* the risk mitigation.

Note the wire total (2.65 GB) ends up **2.5× the stored total (1.07 GB)**. In a warehouse design they would be the same number. That 2.5× gap is the entire cost of having no server-side Parquet writer — a remarkably cheap price, paid purely by writing better SQL.

### 2.2 Rejected alternatives

| Alternative | Why it loses |
|---|---|
| Pull the global table, filter in pandas (what `crsp_puller.py` effectively does) | 121 GB vs 2.65 GB on the wire — **45.7×**. Hours vs minutes. |
| `WHERE fsym_id IN (30,000 literals)` | ~330 KB of SQL text. SQL Server's practical `IN`-list ceiling is low thousands before the optimiser degrades and plan-compile time explodes. A `#temp` + JOIN lets the optimiser build a hash semi-join. |
| Chunked `IN` lists of ~2,000 | 15 chunks × 26 years = 390 queries and 390 plan compilations. Strictly worse than an inline CTE. |

---

## 3. Bias elimination

### 3.1 Survivorship bias — the mistake that invalidates everything

`sym_coverage` has an `active_flag`. **Filtering `active_flag = 1` is the single most destructive thing available**, and it is the default instinct because dead securities feel like noise.

Roughly **40–60% of the 2000–2025 US universe is dead by 2025** (acquired, merged, bankrupt, delisted). Filtering them removes ~half the sample, *non-randomly* — precisely the failures.

Rules:
- **No `active_flag` filter, anywhere, ever.**
- Derive real life spans from the data (`MIN(p_date)`, `MAX(p_date)`, `COUNT(*)`), store `first_px`, `last_px`, `n_obs`, `is_dead` on the dimension table.
- **Test-suite tripwire:** assert `n_dead / n_total > 0.30`. If it ever fails, a survivorship filter has been reintroduced.
- Position exits at delisting must be **explicit**, never implicit truncation.

**Free research artifact:** run the universe query twice, once with and once without `active_flag = 1`. **The difference between the two result sets *is* the survivorship bias, quantified.** Keep both — it is a chart worth putting in a research note.

**What the old CRSP code did wrong here:** `crsp.dsedelist`, `dlret`, `dlstcd` were never queried at all. Delisted names simply stopped appearing. That is worse than textbook survivorship bias, because a pair holding a position in a company that goes bankrupt has its P&L silently **truncated** at the last trading day — so the worst possible outcome, total loss, is recorded as "no further P&L." **This systematically inflates backtest returns, and the inflation is indistinguishable from alpha.**

### 3.2 Look-ahead bias in reference data

Survivorship bias is about *which rows exist*. Look-ahead bias is about *what you knew when*.

Reference tables come in two flavours: **snapshots** (current state) and **history tables** (`start_date`/`end_date` validity windows, usually a `_hist` suffix). A snapshot applied to history classifies a 2007 company by its 2026 sector.

Here that is not a rounding error: **sector membership defines the PCA/clustering universe**, so a snapshot leaks the future straight into pair selection. This is exactly the bug the old CRSP+GICS join had — joined on `permno` alone with no `linkdt`/`linkenddt` window.

**Decision: strict point-in-time.** Every time-varying attribute joins on the observation date:

```sql
JOIN dim_sector_hist s
  ON s.sid = p.sid
 AND p.date BETWEEN s.start_date AND COALESCE(s.end_date, '9999-12-31')
```

DuckDB's **`ASOF JOIN`** expresses this concisely and executes it efficiently — a primary reason DuckDB is the query engine.

If discovery finds only snapshot sector tables, that limitation gets **documented in the README as a measured known bias**, never silently ignored.

### 3.3 Corporate-action correctness: store raw + factor, never adjusted-only

**Store unadjusted OHLCV, the cumulative adjustment factor, and the vendor's own daily total return. Compute adjusted prices at read time as a view.**

Why this is non-negotiable: **a split tomorrow changes the entire history of the adjusted series.** If the Parquet holds adjusted prices, tomorrow's corporate action silently invalidates 25 years of that symbol's data and there is no way to detect it. Holding raw prices + factor means only the factor column changes.

It also yields the best single data-quality test available: the vendor's daily return should reconcile with the return recomputed from `px_close / adj_factor`, with residuals concentrated on ex-dividend dates. One test validates prices, factors, and returns simultaneously.

**What the old code did wrong:** `adj_close = price / cum_factor_price` with the raw price discarded — irreversible and unverifiable. `cfacshr` was pulled and never used, so `market_cap = price × shrout` mixed an unadjusted share count with an adjusted price. No `cfacpr == 0` guard, so division by zero produced `inf`. And `ret`/`retx` were pulled, stored, then dropped by the aggregator — meaning the whole strategy ran on **price returns with dividends excluded**, despite total returns being free in the same query.

---

## 4. Extraction

### 4.1 Ranked paths

Published benchmark, MSSQL, narrow table, LAN. **Read the RAM column against the 8 GB target:**

| Method | 10M rows | Implied rows/s | Peak RAM @10M | Extrapolated @75.6M |
|---|---:|---:|---:|---:|
| "Polars native" | 2.31 s | 4.33 M/s | 937 MB | ~7.1 GB |
| ConnectorX → Polars | 8.73 s | 1.15 M/s | 1,065 MB | ~8.1 GB |
| ConnectorX → pandas | 9.22 s | 1.08 M/s | 515 MB | ~3.9 GB |
| pyodbc → pandas | 21.86 s | 0.46 M/s | 1,313 MB | ~9.9 GB |
| **`pd.read_sql` (SQLAlchemy)** | 33.03 s | 0.30 M/s | 1,338 MB | **~10.1 GB** |

**The disqualifier for the whole `read_sql` family is not the 14× speed gap — it is that every single-shot approach OOMs or thrashes on 8 GB.** Absolute rows/s here are indicative (LAN, unknown table width); the **ratios and the memory figures** are the actionable content.

#### #1 — `arrow-odbc` (primary choice)

Streams `pyarrow.RecordBatch`es with a **bounded** transit buffer, uses the already-installed ODBC Driver 18, and involves no Python-object construction per row. Critically for this environment: it glues Rust to Python via cffi/the Arrow C interface, so **the wheel is not pinned to a specific Python or pyarrow build** — which is what makes it robust across Python version choices. `pip`-installable, win_amd64 wheels, no admin.

Three parameters that actually matter:

| Parameter | Guidance | Why |
|---|---|---|
| `max_bytes_per_batch` | **Set to 32–64 MB** | **Default is 512 MB**, and with `fetch_concurrently=True` that is ~1 GB of buffers. This is the one default that will bite on 8 GB. |
| `max_text_size` | Cap it (e.g. 64–256) | The transit buffer is sized from the *declared* schema width. If IT loaded a name column as `nvarchar(max)`, it will attempt an absurd allocation. Irrelevant for the all-numeric price query, essential for `sym_coverage`. |
| `packet_size` | Try **32767** vs default 4096 | TDS default is 4 KB, max 32,767. Microsoft's guidance is gains are "generally minimal above 8 KB" but the max "can be the best choice even if your environment only occasionally sends large query results" — and this workload is *nothing but* large result sets. **Measure both on one shard**; possibly a free 5–15%. |

Use `?` placeholders and the `parameters` argument — never f-string interpolation.

#### #2 — ADBC MSSQL driver (Columnar's, via their `dbc` manager)

Genuinely Arrow-native and actively developed. **Note: the assumption that no MSSQL ADBC driver exists is wrong — one does.** But: licence terms could not be verified, the install path is `winget`/curl-to-shell (blocked without admin), and one report notes the Python bindings carry overhead versus calling it from Go. **Timebox to 15 minutes; do not let it consume office day.**

#### #3 — `pyodbc` + `fetchmany` + manual Arrow assembly (always-available fallback)

- **`cursor.arraysize = 10000`** — the *only* knob with real effect on reads. It controls rows buffered per ODBC round trip; the default of 1 means one network round trip **per row**, which is catastrophic over a WAN.
- Do **not** cargo-cult `fast_executemany` (writes only) or `setinputsizes` (parameter binding only). Neither affects a read.
- Expect ~3–8× slower than arrow-odbc plus a lot of hand-written transposition code. Insurance, not plan A.

#### #4 — ConnectorX / `polars.read_database_uri`

Fast (~1.1 M rows/s), but its headline feature `partition_on` **requires a numeric, non-NULL column**. The natural shard keys here are `fsym_id` (varchar) and `p_date` (date) — **neither qualifies**, so manual sharding is required anyway and the differentiating feature is irrelevant. It also materialises whole results (~8 GB at full scale).

#### #5 — `pd.read_sql`

Not for the fact table. **Perfectly fine for the ~30k-row dimension queries**, where it is simpler and the memory is irrelevant. Use the right tool per layer.

#### #6 — `bcp` via mssql-tools18

Ruled out: needs MSI (no admin); `-n` native format is an undocumented binary layout nothing in the Python ecosystem parses; `-c` character mode forfeits the byte savings that are the entire point.

### 4.2 Server-side type narrowing (mandatory here)

Because TDS is uncompressed, the SELECT list *is* the wire format. Do the dtype narrowing in T-SQL, not in pandas:

```sql
SELECT u.sid,                                  -- 4 B, not 8-40 B of fsym_id
       CAST(p.p_date        AS date)   AS d,   -- 3 B vs 8 for datetime
       CAST(p.p_price_open  AS real)   AS o,   -- 4 B vs 8 for float/decimal
       CAST(p.p_price_high  AS real)   AS h,
       CAST(p.p_price_low   AS real)   AS l,
       CAST(p.p_price       AS real)   AS c,
       CAST(p.p_volume      AS bigint) AS v,   -- NOT real - see the volume trap, §7.2
       CAST(r.one_day_pct   AS real)   AS ret
```

`real` is IEEE single precision, so this is **lossless relative to the float32 storage target** — the cast is simply moved upstream of the wire rather than trading precision for bytes. Worth **2.2×**.

### 4.3 `sid` substitution as wire compression

Put the integer surrogate key in the `#temp` table and select `u.sid` instead of `p.fsym_id`: **4 bytes instead of 8–40, 75.6 million times.** Worth another **1.23×** (~0.6 GB). This is a trick that only pays off in an uncompressed-wire environment — in a warehouse, ZSTD would have absorbed the string redundancy for free.

### 4.4 Two things never to do in the extraction query

**`ORDER BY` — actively dangerous.** In a warehouse it is free. Here, `ORDER BY fsym_id, p_date` over a billion rows with no supporting index forces a **sort spill into tempdb on a shared production server** — potentially tens of GB, which can fill the drive and take the instance down for every other user. **Only** include `ORDER BY` if the clustered index already provides that order (in which case it is free and you should). Otherwise sort locally in phase 2.

**`OFFSET n ROWS FETCH NEXT m` pagination — never.** `OFFSET` is O(n): the server generates and discards the first *n* rows on every call. Paginating 75.6M rows in 1M-row pages costs approximately ½ · 76 · 76 ≈ **2,900 million rows generated to deliver 76 million** — about **38× the work**. Shard with **range predicates on an indexed column**, always.

---

## 5. Working without DDL

Assume a read-only user: no `CREATE TABLE`, no `CREATE TYPE`.

### (a) `#temp` table in tempdb — RECOMMENDED, and permitted

Temp-table creation in tempdb is available to the `public` role with **no additional grant** — a `CREATE TABLE` grant is only needed for *permanent* tables in tempdb. This delivers everything a warehouse temp table would, plus the `sid`-on-the-wire trick of §4.3.

```sql
CREATE TABLE #u (sid INT NOT NULL PRIMARY KEY,
                 fsym_id VARCHAR(20) NOT NULL UNIQUE);
-- then bulk-insert ~30k rows
```

Two caveats, both operationally important:

- **`#temp` is connection-scoped.** With 3 concurrent connections it must be created and populated 3× (~2 s each — trivial), but the code must do this **per connection**, not once.
- **If the connection drops, the temp table dies with it.** Retry logic must **re-establish session state** — temp table *and* `SET` options — on reconnect, not merely re-issue the query. This is the number-one source of "it worked for three hours then failed weirdly" in this architecture.

### (b) Table-valued parameter — ruled out

Requires a user-defined table type (`CREATE TYPE`), which a read-only user does not have.

### (c) Inline CTE — the zero-permission fallback

Repeat the whole symbology join inside every shard query. Correct and needs no permissions, but `sid` cannot be substituted on the wire, so it costs ~1.5–2× more bytes. The symbology join itself re-executing per shard is cheap (it is a ~10⁶-row table).

### (d) Chunked `IN` lists — last resort, strictly worse than (c)

### 5.1 Isolation level — do this, with eyes open

```sql
SET TRANSACTION ISOLATION LEVEL READ UNCOMMITTED;
SET LOCK_TIMEOUT 5000;
SET DEADLOCK_PRIORITY LOW;
SET ARITHABORT ON;
```

**Why:** a billion-row scan under the default `READ COMMITTED` takes shared locks and can block the nightly FactSet loader. On someone else's production box, that is how read access gets revoked.

**The caveat is worse than "dirty reads":** on an allocation-order scan, `NOLOCK` can return rows that are **missed entirely or returned twice** if page splits occur during the scan. That is a silent correctness bug in the database, not just stale data.

Mitigation, which is clean here:
- **Closed years (2000 … last year) are static — `NOLOCK` is safe.** Nothing is being inserted into 2015.
- **Current year** carries real risk: either accept and re-pull it under default isolation, or schedule that shard outside the loader window.
- Either way, the per-shard row count reconciliation against the server-side oracle (§10.1) catches a skip.

---

## 6. Sharding and the index trap

### 6.1 The trap, stated plainly

In a warehouse, a date predicate prunes micro-partitions and N shards each cost ~1/N of the work. **On a SQL Server heap — or a table whose clustered index does not lead with the shard column — the shard predicate is a residual filter applied *after* a full scan.** Therefore:

> N concurrent shards on an unsupported predicate = **N full scans of a 1B-row table**, competing for the same buffer pool and the same disks, on a shared production server.

Concretely: ~1.01B rows × ~120 B ≈ **120 GB of pages**. One scan on EC2 gp3 is already 10–30 minutes. Four concurrent scans is 480 GB of I/O, cache-thrashes the buffer pool for every other user of the box, and **finishes slower than a single sequential pass.**

This is precisely what ConnectorX's `partition_on` would do automatically if pointed at an unindexed column. It is also the mistake most likely to get a researcher's access revoked.

### 6.2 Check the indexes before sharding anything

```sql
SELECT s.name AS sch, t.name AS tbl, i.name AS idx,
       i.type_desc,          -- HEAP / CLUSTERED / NONCLUSTERED / CLUSTERED COLUMNSTORE
       i.is_unique,
       STUFF((SELECT ', ' + c.name
                     + CASE WHEN ic.is_descending_key=1 THEN ' DESC' ELSE '' END
              FROM sys.index_columns ic
              JOIN sys.columns c ON c.object_id=ic.object_id
                                AND c.column_id=ic.column_id
              WHERE ic.object_id=i.object_id AND ic.index_id=i.index_id
                AND ic.is_included_column=0
              ORDER BY ic.key_ordinal
              FOR XML PATH('')),1,2,'') AS key_cols,
       STUFF((SELECT ', ' + c.name
              FROM sys.index_columns ic
              JOIN sys.columns c ON c.object_id=ic.object_id
                                AND c.column_id=ic.column_id
              WHERE ic.object_id=i.object_id AND ic.index_id=i.index_id
                AND ic.is_included_column=1
              FOR XML PATH('')),1,2,'') AS included_cols
FROM sys.indexes i
JOIN sys.tables  t ON t.object_id = i.object_id
JOIN sys.schemas s ON s.schema_id = t.schema_id
WHERE s.name IN ('fp_v2','fgp_v1','sym_v1','ent_v1')
ORDER BY s.name, t.name, i.index_id;
```

### 6.3 Decision rule

| What the index shows | Shard on | Concurrency | Notes |
|---|---|---|---|
| Clustered leading with **`fsym_id`** (most likely — how a vendor feed loader usually keys it) | `fsym_id` ranges | 3–4 | Each shard is a clustered range **seek**. Rows arrive id-grouped, so the `(sid, date)` sort is nearly free. **Best case.** |
| Clustered leading with **`p_date`** | year / half-year ranges | 3–4 | Range seek per shard. Rows arrive date-grouped, so the date-major replica is nearly free. |
| Clustered on **`(fsym_id, p_date)`** | `fsym_id` ranges | 3–4 | Ideal: seek *and* pre-sorted. |
| **Clustered columnstore** | year ranges | 2–3 | Rowgroup elimination gives warehouse-like pruning. |
| **Heap / no useful index** | **Do not shard. One sequential streaming pass.** | **1** | N shards = N full scans. One pass = one scan. |

### 6.4 Be a good citizen on shared infrastructure

- **Start at concurrency 3, not 16.** The useful gain here is *TCP stream parallelism over a WAN*, not server CPU — and server CPU is a cost imposed on other users, not a resource you purchased.
- **Add `OPTION (MAXDOP 1)` to every shard query.** Without it, 4 connections × server MAXDOP 8 = **32 parallel workers** seized on someone else's server. Drop the hint only if `sys.dm_db_index_usage_stats` shows the box is genuinely idle.
- **Shard size target: 3–8 minutes, so ~40–80 shards.** A shard is the restart unit and the blast radius. A wifi drop, query timeout, or a DBA killing the session costs ≤8 minutes. Small enough to retry cheaply, large enough that per-shard overhead (~3 s for connect + `#temp` populate) stays under 2%.

---

## 7. Storage and compression

### 7.1 Format decision: Parquet lake + DuckDB

**Partitioned Parquet as the source of truth, DuckDB as the query engine.**

| Alternative | Why it loses |
|---|---|
| **pandas pickles per sector** (current design) | ~5.4 GB and ~5.4 GB RAM just to open. **Fails outright on an 8 GB machine.** No schema, no pushdown, version-fragile, not interoperable. |
| CSV | 5.9 GB, 5.9× larger, and no predicate pushdown. |
| gzip-CSV | 1.6 GB but **not queryable** — reading one year means decompressing all of it. |
| PostgreSQL / TimescaleDB | Real server, strong constraints — but slower on wide analytical scans, and needs actual DB administration for a single-user research store. |
| ClickHouse | Fastest for huge scans, but heavy to operate and overkill at ~2 GB. |
| Parquet with no query engine | Loses SQL, and loses cheap point-in-time joins. |

DuckDB earns its place on three specifics: **`ASOF JOIN`** for point-in-time reference joins (§3.2), **Parquet predicate/projection pushdown**, and **out-of-core execution with a configurable `memory_limit`** — which is what makes 8 GB viable. `duckdb.sql(...).pl()` is a zero-copy Arrow handoff to Polars, so the boundary between the two engines costs nothing.

### 7.2 Dtype narrowing — and one trap that silently corrupts data

| Column | Type | Rationale |
|---|---|---|
| `sid` | `int32` | see §8.2 |
| `date_ed` | `int32` | epoch days. Not `timestamp[ns]` (8 B), not a string. |
| `px_{open,high,low,close}` | `float32` | 24-bit mantissa ≈ 7.2 significant digits. US prices are quoted to 4 decimals and rarely exceed $10,000 outside BRK.A. **Halves the largest part of the file.** |
| `volume` | **`uint32`** | **see the trap** |
| `adj_factor` | `float32` | near-constant per security → compresses to near-nothing |
| `ret_1d` | `float32` | returns are ~1e-2; float32 gives ~7 significant figures, far more precision than the data possesses |

> ### ⚠ The volume trap
> **Never store volume as float32.** float32 represents integers *exactly* only up to 2²⁴ = **16,777,216**. A heavily-traded penny stock exceeds that. float32 volume therefore **silently corrupts**, and the corruption quietly poisons every liquidity filter downstream. This is the class of bug that survives for years undetected. Use `uint32` — and cast to `bigint` on the wire, narrowing client-side only *after* asserting the observed maximum.

**Where float64 stays:** narrow at *storage*, compute in *float64*. Cointegration regressions, rolling z-scores, and cumulative compounding must upcast — float32 accumulation error compounds over 6,500 observations. **Rule: float32 on disk, float64 in the math.**

> **Rejected:** float64 everywhere "to be safe." Doubles the biggest columns for precision the source data does not have. It is not recoverable precision — FactSet's prices are 4-decimal.

### 7.3 Partitioning

**Partition by `year` only.** 26 partitions, ~2.9M rows / ~40 MB each.

| Alternative | Why it loses |
|---|---|
| `year/month` | 312 files of ~3.5 MB. Parquet footers and per-file open costs start to dominate; query planning time grows. Sub-25 MB Parquet is the classic small-file anti-pattern. |
| **Partition by symbol** | **Catastrophic.** 30,000 directories averaging ~33 KB. Every file carries its own footer *and a fresh dictionary*, so most of the compression is lost, **and** a cross-sectional query pays 30,000 file opens. Tempting for pairs trading ("I want one stock's series") and wrong: with sorted files plus row-group statistics and the page index, a single-symbol read already touches only 1–2 row groups per year. |

### 7.4 Sort order — where most of the compression actually comes from

Sorting is not cosmetic; it *is* the compression. Sorted by `(sid, date_ed)`:

- `sid` becomes ~2,520-long runs → RLE → **~0.02 B/row**
- `date_ed` deltas are 1–4 → `DELTA_BINARY_PACKED` → **~0.15 B/row**
- prices are autocorrelated → `BYTE_STREAM_SPLIT` groups exponent bytes together and mantissa bytes together, giving ZSTD real redundancy to find. **On unsorted data this gains almost nothing.**

**Sorting alone is worth ~1.5× on total file size.**

**Two sort orders, deliberately.** Pairs research has two access patterns:
- long time series for one or two stocks → wants **sid-major**
- full cross-sections for a date (universe screening, PCA on the return matrix, clustering) → wants **date-major**

At ~1 GB per replica, storing both costs ~1 GB of disk and removes a sort from every cross-sectional step. Standard practice at scale, unusually cheap here.

### 7.5 Row groups and metadata

- **`row_group_size = 1_000_000`** (~3 per year file). Row-group min/max stats are the pushdown unit, so smaller groups skip better — but each group restarts the dictionary and the ZSTD window, costing compression. 1M × 36 B ≈ 36 MB resident, comfortable at 8 GB.
- **Enable the page index** for sub-row-group skipping — this turns a single-symbol read into a few-hundred-KB read.
- **Write `sorting_columns` into the file metadata** so readers *know* the file is sorted and can exploit it. Most writers skip this; DuckDB and newer engines will use it.

### 7.6 Compression codec and level

**ZSTD level 9 for the curated layer; ZSTD level 3 for the raw landing zone.**

Level 5→9 buys **<1.5% size for ~2× write time**. Level 22 is roughly 10× slower again for perhaps another 1% — indefensible. Level 9 sits just past the knee. Level 3 for `raw/` because those files are transient and the download path must not become CPU-bound.

**The one genuine tension between "maximum compression" and "maximum speed":** heavy ZSTD only accelerates scans when I/O dominates. On NVMe, decompression can *hinder* scan throughput — level 9 likely costs 15–25% on full-scan reads versus Snappy. **Decision: take the compression.** At ~1 GB total a full scan is a couple of seconds either way. If a hot research loop later rescans constantly, materialise a Snappy or Arrow-IPC "hot cache" of just the columns that loop needs rather than compromising the archival layer.

### 7.7 Size arithmetic

Assumptions (stated so they can be revised — discovery query 11 replaces the estimate with a measurement): ~30,000 US common-equity securities with ≥1 price in 2000–2025; mean ~2,520 observed trading days each (the window has ~6,550, and median listing life inside a 26-year window is well under half). **≈75.6M rows.**

| Layout | B/row | Total | vs best |
|---|---:|---:|---|
| CSV | 78 | 5.9 GB | 5.9× |
| CSV + gzip -9 | ~21 | 1.6 GB | 1.6× *(not queryable)* |
| pandas pickle (float64, object ticker) | ~72 | 5.4 GB | 5.4× *(**fails on 8 GB**)* |
| Parquet, lazy defaults (float64, string id, Snappy, unsorted) | ~46 | 3.5 GB | 3.5× |
| Parquet, float32 + int32 sid, Snappy, unsorted | ~25 | 1.9 GB | 1.9× |
| Parquet, float32 + int32 sid, ZSTD-9, unsorted | ~20 | 1.5 GB | 1.5× |
| **float32 + int32 sid, ZSTD-9, sorted, BYTE_STREAM_SPLIT + DELTA** | **~14** | **≈1.07 GB** | **1.0×** |

Estimated per-column breakdown of the ~14 B/row:

```
sid          RLE over ~2520-long runs                0.02
date_ed      DELTA_BINARY_PACKED, deltas 1-4         0.15
px_open      BYTE_STREAM_SPLIT + ZSTD-9              2.20
px_high      "                                       2.20
px_low       "                                       2.20
px_close     "                                       2.20
volume       DELTA_BINARY_PACKED (noisy, low gain)   2.50
adj_factor   near-constant per sid, RLE/dict         0.20
ret_1d       BYTE_STREAM_SPLIT + ZSTD-9              2.40
                                                   ------
                                                    14.07 B/row
```

**Total curated store ≈ 2.2 GB** with both sort replicas (≈1.1 GB without). **Sanity range for the primary: 0.8–1.5 GB.** Landing outside that range means something is wrong — most likely the sort was skipped, float64 was left in, or a string column was denormalized.

**Worth benchmarking before committing** (at home, cheaply): prices as scaled integers (`price × 10_000` as `int64`) with `DELTA_BINARY_PACKED` instead of float32 + BYTE_STREAM_SPLIT. Consecutive daily closes differ by ~1%, so zigzag-delta packing can beat byte-stream-split — possibly 1.07 GB → ~0.85 GB. Risks: `int32` overflows above $214,748 (BRK.A needs `int64`), and the scale factor becomes a schema invariant you own. **Test both on a 200-symbol sample and pick empirically** rather than trusting either estimate.

---

## 8. Schema design

### 8.1 Star schema

Fact table (~75.6M rows) holds only `sid` + date + numbers. Dimension tables (~30k rows) hold names, tickers, exchange, sector, entity.

```
data/
  raw/        prices/shard=.../part.parquet          # office output, ZSTD-3, UNSORTED
  curated/
    us_prices_by_sid/  year=2000..2025/part.parquet  # sorted (sid, date_ed)
    us_prices_by_date/ year=2000..2025/part.parquet  # sorted (date_ed, sid)
    dim_security.parquet        # 30k rows: sid, fsym_id, name, exch, entity,
                                #   primary_listing_flag, first_px, last_px,
                                #   n_obs, is_dead
    dim_sector_hist.parquet     # sid, scheme, code, start_date, end_date  (PIT)
    corporate_actions.parquet
  _manifest/manifest.jsonl, schema_v1.json, oracle.parquet
```

> **Rejected:** denormalizing company name / ticker / sector onto all 75.6M rows. Triples the RAM footprint for zero additional information. The most common junior mistake in this kind of build.

### 8.2 The `sid` surrogate key — the honest version of the argument

The usual claim is "storing an 8-char string a billion times is a disaster." **On disk that is mostly false**, and knowing why matters for arguing it correctly: with 30k distinct values and rows sorted by id, Parquet dictionary-encodes the column and RLE-encodes the indices to near-zero. A dictionary-encoded string ≈ int32 on disk.

The real costs are elsewhere, and they are large:

| Cost | Magnitude |
|---|---|
| **In-memory (Arrow)** | string = 4-byte offset + ~8 bytes data ≈ 12 B/row vs 4 B for int32 → **~600 MB extra RAM** per read of that column. **On 8 GB this is the difference between working and not.** |
| **Join/groupby speed** | hashing and comparing 8-byte strings is ~3–5× slower than int32. Every `groupby(sid)` pays it. |
| **Pair keys** | a pairs pipeline enumerates C(n,2) pairs — at n=1,000 that is 499,500. A `(int32, int32)` key packs into one `int64` and hashes in a single instruction. Two strings do not. |
| **On the wire** | 4 bytes instead of 8–40, 75.6M times (this environment only — §4.3). |

**`sid` is a permanent surrogate key:** dense rank over sorted `fsym_id`, persisted, append-only, checksummed, **never renumbered**. New securities append the next available id. Renumbering causes every cached artifact downstream to silently rebind to the wrong stock.

### 8.3 Universe definition

- **All US-listed common classes**, each flagged with a primary-listing indicator and entity ID — so dual-class pairs (GOOGL/GOOG, BRK.A/BRK.B) remain tradeable rather than being discarded. These are arguably the highest-quality cointegrated pairs available.
- **REITs: include at ingest, tag by sector, filter at research time.** Ingest broad, filter late — re-ingesting requires another office trip, filtering is a one-line change.
- **No `active_flag` filter** (§3.1).
- The security-type filter **must not be written until the `GROUP BY fref_security_type` probe has been run** (§13, query 9). The publicly documented value list is incomplete, and guessing here silently contaminates the universe with ETFs, ADRs, warrants, or units.

---

## 9. Two-phase build

This is the key structural consequence of office-only server access.

**Phase 1 — office, one shot:** extract → `raw/` Parquet, **ZSTD-3, no sorting, no reordering.** Optimised purely for "get the bytes off the server correctly and resumably."

**Phase 2 — home, infinitely repeatable:** transcode `raw/` → curated layout: two sort orders, ZSTD-9, 1M row groups, page index, `sorting_columns`.

**Decoupling these means a bad compression choice becomes a 10-minute redo at home instead of a lost trip.** The warehouse design could conflate them because the server sorted for free; here that would be a trap.

Phase 2 on 8 GB uses DuckDB's external merge sort with an explicit budget:

```sql
SET memory_limit='5GB';
SET temp_directory='D:/duck_tmp';
SET threads=4;
COPY (SELECT * FROM 'raw/**/*.parquet' ORDER BY sid, d)
  TO 'curated/us_prices_by_sid'
  (FORMAT PARQUET, PARTITION_BY (yr), COMPRESSION ZSTD,
   COMPRESSION_LEVEL 9, ROW_GROUP_SIZE 1000000);
```

200 GB free is ample for spill. Expect 5–15 minutes per sort order.

### 9.1 Manifest and resumability

`_manifest/manifest.jsonl` — append-only, one JSON line per completed shard, recording: path, rows, distinct sids, date min/max, **sha256**, bytes, source query hash, **universe hash**, writer library versions, timestamp, status.

Append-only JSONL means a crash mid-write corrupts at most the last line, which is skipped on read.

- **Restart rule:** skip a shard iff `status:"complete"` **and** the on-disk sha256 matches. **Not** "iff the file exists" — a truncated file exists.
- **Atomic writes:** `part.parquet.tmp` → fsync → `os.replace()` (atomic on NTFS, same volume). A Parquet file missing its footer is unreadable, so a partially written file bearing the final name is a trap. *This is exactly the failure mode the old `df.to_csv()` per-year loop had: interrupt it and pandas happily reads back a truncated last row as valid data.*
- **`universe_sha256` is the field that saves you.** If the universe definition changes — say discovery reveals the real ETF type code and 400 ETFs are dropped — every shard built against the old universe is stale. Comparing recorded vs current universe hash flags this **automatically**. Without it you get silent, permanent inconsistency between partitions, which is the likeliest way this database quietly goes wrong.
- **Session-state-aware retry** (`tenacity`, exponential backoff, cap 5 attempts): re-`SET` isolation level **and** re-create + re-populate `#temp` on reconnect. On permanent failure, **log the shard key and continue** — finish the other 79 shards and re-run the 1, rather than losing four hours to a single lock timeout.
- **Copy to the external drive incrementally during the run**, not once at the end. If the laptop dies at 90%, 90% of the bytes should already be portable.
- **Nothing gets deleted from `raw/` until phase 2 has succeeded and validated at home.** `raw/` is the irreplaceable artifact. Keep two copies.

### 9.2 Incremental updates (once home access or a later office session exists)

```
watermark = max(date_max) over manifest
pull WHERE p_date > watermark - 30 days      # 30-day restatement window
→ rewrite affected year partition(s) WHOLE
```

**Rewrite, do not append.** Parquet files are immutable; "appending" means adding `part-001.parquet`, which destroys the partition-level global sort order (each file is sorted, the partition is not) and kills both pushdown and compression. A year file is ~40 MB — rewriting takes 1–2 seconds.

**The 30-day lookback is not optional** — vendors restate, and **a split changes `adj_factor` for a symbol's entire history**, which is precisely why §3.3 forbids storing adjusted prices only. Tiered handling: daily trailing-30-day rewrite; weekly re-pull of `adj_factor` full history for any symbol with a recent corporate action; quarterly full rebuild and diff against the existing store, where any unexplained difference is either a pipeline bug or an uncaught restatement.

---

## 10. Verification

### 10.1 The oracle — verification without the server

Because re-pulling from home is impossible, a **server-authored ground truth** must be carried home:

```sql
SELECT YEAR(p.p_date) AS yr,
       COUNT_BIG(*)              AS n_rows,
       COUNT(DISTINCT p.fsym_id) AS n_ids,
       MIN(p.p_date) AS d_min, MAX(p.p_date) AS d_max,
       SUM(CAST(ROUND(CAST(p.p_price AS float),4)*10000 AS BIGINT)) AS px_checksum
FROM   fp_v2.fp_basic_prices p WITH (NOLOCK)
JOIN   #u u ON u.fsym_id = p.fsym_id
WHERE  p.p_date >= '2000-01-01'
GROUP  BY YEAR(p.p_date)
ORDER  BY yr
OPTION (MAXDOP 2);
```

**Why scaled integers:** `SUM` of scaled integers is **exact and order-independent**, so it reconciles bit-for-bit against the local Parquet. `SUM` of floats would not — floating-point addition is not associative, so a different row order yields a different sum. `CHECKSUM_AGG` is order-independent but collision-prone and type-quirky; do not rely on it alone.

At home, the identical aggregate over `curated/` must match **all five columns for all 26 years**.
- Rows differ → a `NOLOCK` skip/duplicate, or a dropped shard.
- Checksum differs but rows match → a CAST or precision bug.

**This test is the difference between a database you can bet money on and one you merely hope is right.**

Also pull `dim_security` with `first_px`/`last_px` per security while connected — that is the survivorship evidence, and the tripwire test runs off it with no server.

### 10.2 Data-quality test suite

`pytest` against the built lake, run after every build, failing loudly. Not manual eyeballing.

| # | Test | Threshold / expectation |
|---|---|---|
| 1 | **Survivorship tripwire** | `n_dead / n_total > 0.30`. Returns ~0 → an `active_flag` filter was reintroduced and the whole research programme is invalid. |
| 2 | **Listing-count curve** | distinct securities/year should trace the real US curve: ~7,000–8,000 in 2000 declining to ~4,000–4,500 by 2020. **Flat or rising = survivorship bias or ETF contamination.** |
| 3 | **Uniqueness** | zero duplicate `(sid, date_ed)`. **Assert — never `aggfunc='first'`**, which silently masks the problem (as the old aggregator did). |
| 4 | **Return reconciliation** | vendor `ret_1d` vs recomputed `px_close/adj_factor` return: ≥99% agreement within float32 epsilon, residuals concentrated on ex-dividend dates. Validates prices, factors and returns in one test. |
| 5 | **Oracle reconciliation** | all five columns match for all 26 years (§10.1). |
| 6 | **Extreme moves** | flag `\|ret_1d\| > 0.5`; each must be explicable by a corporate action. Unexplained ones are usually unhandled splits. |
| 7 | **Trading calendar** | dates ⊆ NYSE calendar (`pandas_market_calendars`, free). A US-holiday date means non-US contamination. |
| 8 | **Cross-source vs CRSP** | coverage by month, return distributions, split dates, **delisting-event completeness**. This is what the CRSP CSVs are for. |
| 9 | **Free-source spot check** | ~20 symbols vs yfinance/Stooq. Cheap; catches gross errors. |
| 10 | **Currency** | exactly one distinct value before dropping the column. |
| 11 | **Zero/negative prices** | count and quarantine, never silently keep. **Do not carry over CRSP's `.abs()` on price** — CRSP uses negative prices to signal bid/ask midpoints; FactSet does not, so `.abs()` there would mask real data errors. |

### 10.3 Reproducibility

- `uv sync` on a clean machine + `pytest` green against a **committed synthetic fixture** (~5 fake securities × 300 days, seeded generator) with **no FactSet access**. This proves the repo is genuinely portable and is what a reviewer will actually try.
- **Idempotency:** re-running extraction with all shards complete is a no-op. Deleting one shard and re-running rebuilds exactly that shard and reproduces its checksum.

---

## 11. Concurrency

| Technique | Verdict | Reasoning |
|---|---|---|
| **Threads, 3–4 connections, for extraction** | **Yes** | For TCP stream parallelism over a latency-bearing WAN and to overlap ZSTD encoding with socket reads — **not** to parallelise server work. `arrow-odbc`'s `fetch_concurrently=True` provides the overlap within a shard. |
| **Multiprocessing over rebalance dates (backtest)** | **Yes — the one place it changes your life** | ~3,000 candidate pairs/rebalance × 2 Engle-Granger calls × ~2.5 ms ≈ 15 s/rebalance × 318 monthly rebalances ≈ **80 min single-threaded → ~11 min on 8 workers.** Turns a parameter sweep from overnight into lunch. |
| **Multiprocessing for extraction** | **No — cargo cult** | I/O-bound work. Each process needs its own connection *and* its own result buffer: 8× the RAM (unaffordable at 8 GB) plus IPC/pickle overhead, to accelerate waiting on a socket. |
| **Multiprocessing inside the per-pair loop** | **No** | ~3 ms task granularity is below the ~1–10 ms process-dispatch + pickle cost. It would run *slower*. |
| **Multiprocessing for local transcode** | **No** | pyarrow, Polars and DuckDB already multithread internally. An outer pool oversubscribes cores and typically slows things down. Use `SET threads=N` in DuckDB instead of wrapping it. |
| **asyncio** | **No** | Two I/O patterns exist here: local NVMe reads (page cache + DuckDB's threaded reader already saturate the device) and a handful of long-lived SQL queries (threads are simpler; the drivers are synchronous anyway). asyncio buys coloured functions and worse debugging for zero measurable speedup. **Test: if you cannot name the specific blocking call and its share of wall time, you do not need async.** |
| **Dask / Ray / Spark** | **No** | At ~2 GB, one laptop's core count beats any distributed scheduler. The honest threshold for Dask is "does not fit on one machine's disk." This is ~100× away. |
| **Polars / DuckDB** | **Yes — phase 2 and research** | Never transport tools. `scan_parquet` → filter/project → `sink_parquet` with pushdown and streaming. |
| **pandas** | **Demote to the last mile** | Fine below ~1M rows and where statsmodels/sklearn/matplotlib force it. Wrong for the 75M-row layer. |

**One-sentence rule:** parallelise *bytes in flight* with a few threads, parallelise *CPU* by letting Arrow/DuckDB/Polars do it internally, and never spawn processes to do either.

### 11.1 Implementation details that matter more than pool size

- **Parallelise over rebalance dates, not over pairs.** Coarser tasks, far less pickling, and each worker opens its own DuckDB connection against read-only Parquet — safe, since Parquet readers do not contend.
- **Pass `sid` lists and date ranges to workers, never DataFrames.** Pickling a 6,700×1,200 float64 panel to 8 workers is 8 × 64 MB of serialisation. Letting each worker query DuckDB itself is faster *and* uses less memory.
- **Set `OMP_NUM_THREADS=1` and `OPENBLAS_NUM_THREADS=1` inside worker processes.** Otherwise 8 processes × BLAS's own threads oversubscribe the CPU. Commonly missed, worth ~2×.

### 11.2 Profile before parallelising — the actual lesson

Use `py-spy record` (free, sampling, no instrumentation) or `cProfile` + `snakeviz`.

**Prediction: >85% of runtime is inside `statsmodels.tsa.stattools.coint` and `statsmodels.formula.api.ols`.** If so, the highest-return performance work is not concurrency at all:

1. **Replace `smf.ols("y ~ x", data=...)` with `np.linalg.lstsq`.** Patsy formula parsing costs ~1 ms *per fit*; `lstsq` on a 504×2 design is ~10 µs. **That is ~100× on a single core**, and both the pair screen and the half-life calculation call it inside a nested loop.
2. **Replace `coint()` with a direct ADF on the OLS residual**, using a precomputed critical-value table. `coint` re-fits the cointegrating regression and redoes lag selection on every call — the residual is already in hand.

Plausibly **20–50× before spawning a single process.** Then add the pool for another ~8×.

**That ordering is the lesson: algorithmic and library work first, then processes, and never async.** Reaching for multiprocessing before profiling is how people end up with a beautifully parallel implementation of an accidentally-quadratic algorithm.

### 11.3 Caching

`functools.lru_cache` is the wrong primitive here — unbounded memory, no cross-process sharing, no invalidation, and when applied to an instance method it leaks `self` (as the old `ModelSpread` did).

Instead: **materialise intermediate results as Parquet keyed by `config_hash`.** A disk cache survives restarts, is shared across all 8 worker processes for free, is inspectable with a SQL query, and invalidates correctly because the key *is* the config. Roughly 30 lines.

---

## 12. Tooling

### 12.1 Python version — a real landmine

The office machine has **Python 3.14.6**. Current wheel reality:

| Package | 3.14 status |
|---|---|
| pyarrow | **Fine** — cp314 wheels exist |
| arrow-odbc | **Should be fine** — cffi/abi3 design, not pinned per Python version. **Verify with a real install.** |
| duckdb | **Shaky** — open issues; pip falls back to a source build and hangs |
| polars | **Shaky** — official 3.14 support still tracked in an open issue |
| connectorx | Almost certainly no cp314 wheels |

**Decision, and it is step 0 of the build:** install **`uv`** (single static binary, user-scoped, **no admin required**) and create a **Python 3.13** venv. `uv` downloads its own CPython, so the system 3.14 is untouched and every wheel above becomes a solved problem. Pin `requires-python = ">=3.13,<3.14"` for the ingest extra.

> **Discovering a missing wheel at the office, on the one day server access exists, is the worst outcome available.** Do a full dependency dry-run plus an end-to-end `SELECT TOP 1000` **days before** the real pull.

### 12.2 pandas 3.0

pandas 3.0.5 is installed and is a **major release** — copy-on-write by default, new default string dtype. Legacy code here was written against pandas 1.x/2.x; the `get_loc(..., method="bfill")` removal (pandas 2.0) is one already-confirmed symptom. Pin the version and expect more.

### 12.3 Configuration and secrets

- **Config:** `pydantic-settings` or a single TOML, plus one `DATA_ROOT` resolved once, `pathlib` everywhere. Eliminate every `os.getcwd()` and both hardcoded macOS absolute paths.
- **No module-level I/O.** The old `config/settings.py` read a pickle at *import time* from `os.getcwd()`, so any import failed unless the cwd happened to contain `GICS_45.pkl`. Replace with a lazy function.
- **Derive the trading calendar from `pandas_market_calendars`**, not from one stock's observed dates — AAPL's date index also encodes AAPL's trading halts.
- **Secrets:** `.env` gitignored, `.env.example` committed. Never the host, username, or password in code.

### 12.4 Git hygiene — do this BEFORE any data lands

The database **cannot** go in git, for two independent reasons:

**Technical.** 100 MB hard per-file block, ~5 GB practical repo ceiling, Git LFS free tier is 1 GB storage with the real cliff at **10 GB/month bandwidth ≈ five clones**. More fundamentally, **git stores a complete copy of every binary revision forever**, and ZSTD-compressed Parquet cannot delta-compress — so every rebuild adds ~2 GB to history permanently, even after the files are deleted. Note the neat proof that data does not belong there: **better compression makes the git problem strictly worse.**

**Legal.** The FactSet licence is the employer's and prohibits redistribution. Pushing to GitHub — **even a private repo** — is redistribution.

Current `.gitignore` covers `*.csv` and `*.pkl` but **not `data/` or `*.parquet`**. One `git add -A` after extraction would publish licensed vendor data. Add before extracting anything:

```
data/
*.parquet
*.duckdb
*.zst
.env
```

Plus a `pre-commit` hook rejecting staged files over ~1 MB or matching `*.parquet`. Also fix the bare `lib/` entry, which will silently swallow any future `lib/` source directory.

**What goes in git:** code, schema definitions, SQL, config templates, validation tests, **manifests with per-file sha256**, equity curves, and the **synthetic fixture** that makes CI runnable without FactSet.

**Data transfer office → home:** external SSD (~3 GB). No DLP or licence-egress question.

### 12.5 Packaging

- `src/` layout. It removes an entire class of ambiguity: the build can only see what is under `src/`, so it can never accidentally sweep in `.venv` or stray root files. (Current config uses `where = ["."]`, which scans the repo root.)
- **Add the six missing `__init__.py` files** (or switch to `find_namespace:`). The current editable install masks this — `pip install -e .` puts the repo root on `sys.path` and PEP 420 namespace packages make everything importable — but a real wheel build would run `packages.find`, discover only the two directories that have `__init__.py`, and ship a distribution missing five subpackages.
- **Delete `setup.cfg`.** It duplicates `pyproject.toml` metadata; modern setuptools reads `pyproject.toml` and `setup.cfg` becomes a confusing shadow.
- Add `ruff`, `mypy`, `pre-commit`, and GitHub Actions CI running tests against the synthetic fixture.

---

## 13. Discovery script — run this first

**This is the highest-value action available.** It costs ~15 minutes and resolves nine unknowns that everything else depends on. Every statement is read-only except step 7, which creates and drops a `#temp` table (permitted to `public`).

Run block by block; save every result set.

```sql
/* ============================================================
   FactSet-on-MSSQL discovery.  READ ONLY except step 7 (#temp).
   ============================================================ */

-- 1. Server identity ------------------------------------------------
SELECT @@VERSION AS version_string;
SELECT SERVERPROPERTY('Edition')        AS edition,
       SERVERPROPERTY('ProductVersion') AS product_version,
       SERVERPROPERTY('ProductLevel')   AS product_level,
       SERVERPROPERTY('EngineEdition')  AS engine_edition,
       SERVERPROPERTY('Collation')      AS collation,
       DB_NAME()                        AS current_db,
       SUSER_SNAME()                    AS login_name;
-- How much machine am I sharing? (informs how hard I may push)
SELECT cpu_count, scheduler_count,
       physical_memory_kb/1048576 AS ram_gb, sqlserver_start_time
FROM sys.dm_os_sys_info;

-- 2. Which databases can I see? ------------------------------------
SELECT name, database_id, state_desc, recovery_model_desc,
       compatibility_level, is_read_only
FROM sys.databases ORDER BY name;

-- 3. Schemas in the target database --------------------------------
USE <DB>;   -- set from step 2
SELECT s.name AS schema_name, COUNT(t.object_id) AS n_tables
FROM sys.schemas s LEFT JOIN sys.tables t ON t.schema_id = s.schema_id
GROUP BY s.name ORDER BY n_tables DESC;

-- 4. All FactSet tables with FAST row counts + size ----------------
--    sys.dm_db_partition_stats reads metadata only.
--    NEVER use COUNT(*) here - it would scan a billion rows.
SELECT s.name AS sch, t.name AS tbl,
       SUM(CASE WHEN ps.index_id IN (0,1) THEN ps.row_count ELSE 0 END) AS [rows],
       CAST(SUM(ps.reserved_page_count)*8.0/1024 AS DECIMAL(14,1)) AS reserved_mb,
       CAST(SUM(ps.used_page_count)*8.0/1024     AS DECIMAL(14,1)) AS used_mb,
       MAX(CASE WHEN ps.index_id=0 THEN 1 ELSE 0 END)              AS is_heap
FROM sys.dm_db_partition_stats ps
JOIN sys.tables  t ON t.object_id = ps.object_id
JOIN sys.schemas s ON s.schema_id = t.schema_id
WHERE s.name IN ('fp_v2','fgp_v1','sym_v1','ent_v1')
GROUP BY s.name, t.name
ORDER BY [rows] DESC;
-- Confirm the ~1.01B / ~492M figures. used_mb is the full-scan wire ceiling.

-- 5. Column lists and types - THE wire-cost query -------------------
SELECT c.TABLE_SCHEMA, c.TABLE_NAME, c.ORDINAL_POSITION, c.COLUMN_NAME,
       c.DATA_TYPE, c.CHARACTER_MAXIMUM_LENGTH AS char_len,
       c.NUMERIC_PRECISION, c.NUMERIC_SCALE, c.IS_NULLABLE
FROM INFORMATION_SCHEMA.COLUMNS c
WHERE c.TABLE_SCHEMA IN ('fp_v2','fgp_v1','sym_v1','ent_v1')
ORDER BY c.TABLE_SCHEMA, c.TABLE_NAME, c.ORDINAL_POSITION;
-- varchar vs nvarchar on fsym_id = 8 vs 40 bytes/row.
-- float/decimal vs real on prices = 8 vs 4 bytes/row. Multiply by ~75.6M.
-- Also settles fgp_v1's real column names.

-- 6. INDEXES on the price tables - the shard-key decision ----------
--    (full query in section 6.2 above - run it verbatim)

-- 6b. Physically partitioned? --------------------------------------
SELECT t.name, p.partition_number, p.rows, p.data_compression_desc
FROM sys.partitions p JOIN sys.tables t ON t.object_id=p.object_id
WHERE t.name IN ('fp_basic_prices','fp_total_returns_daily','fgp_global_prices')
  AND p.index_id IN (0,1)
ORDER BY t.name, p.partition_number;

-- 6c. Am I about to disturb a live workload? -----------------------
SELECT OBJECT_NAME(s.object_id) tbl, i.name idx,
       s.user_seeks, s.user_scans, s.user_lookups, s.user_updates,
       s.last_user_scan
FROM sys.dm_db_index_usage_stats s
JOIN sys.indexes i ON i.object_id=s.object_id AND i.index_id=s.index_id
WHERE s.database_id = DB_ID() AND OBJECT_NAME(s.object_id) LIKE 'fp_%';

-- 7. What am I allowed to do? --------------------------------------
SELECT * FROM sys.fn_my_permissions(NULL, 'DATABASE') ORDER BY permission_name;
SELECT * FROM sys.fn_my_permissions('fp_v2','SCHEMA') ORDER BY permission_name;
SELECT IS_ROLEMEMBER('db_datareader') AS is_datareader,
       IS_ROLEMEMBER('db_owner')      AS is_dbowner,
       IS_SRVROLEMEMBER('sysadmin')   AS is_sysadmin;
-- The critical capability test:
BEGIN TRY
    CREATE TABLE #probe (sid INT PRIMARY KEY, fsym_id VARCHAR(20));
    INSERT INTO #probe VALUES (1,'TEST-R');
    SELECT COUNT(*) AS temp_rows_ok FROM #probe;
    DROP TABLE #probe;
    SELECT 'TEMP TABLE OK -> plan (a): #temp + sid substitution' AS verdict;
END TRY
BEGIN CATCH
    SELECT 'TEMP TABLE BLOCKED -> plan (c): inline CTE' AS verdict,
           ERROR_NUMBER() AS err, ERROR_MESSAGE() AS msg;
END CATCH;

-- 8. Date coverage per price table ---------------------------------
SET TRANSACTION ISOLATION LEVEL READ UNCOMMITTED;
SELECT 'fp_basic_prices' AS tbl, MIN(p_date) AS d_min, MAX(p_date) AS d_max
FROM fp_v2.fp_basic_prices WITH (NOLOCK) OPTION (MAXDOP 2);
SELECT 'fp_total_returns_daily', MIN(p_date), MAX(p_date)
FROM fp_v2.fp_total_returns_daily WITH (NOLOCK) OPTION (MAXDOP 2);
-- If these take >2 minutes, that itself answers the index question.

-- 9. THE universe probe: real security-type domain ------------------
SELECT fref_security_type, universe_type, COUNT(*) AS n,
       SUM(CASE WHEN active_flag = 0 THEN 1 ELSE 0 END) AS n_inactive
FROM sym_v1.sym_coverage WITH (NOLOCK)
GROUP BY fref_security_type, universe_type
ORDER BY n DESC;
-- DO NOT write the universe filter before running this.

SELECT fref_listing_exchange, COUNT(*) AS n
FROM sym_v1.sym_coverage WITH (NOLOCK)
WHERE currency = 'USD' AND universe_type = 'EQ'
GROUP BY fref_listing_exchange ORDER BY n DESC;

-- 10a. Adjustment-factor hunt (highest-value unknown) --------------
SELECT TABLE_SCHEMA, TABLE_NAME, COLUMN_NAME, DATA_TYPE
FROM INFORMATION_SCHEMA.COLUMNS
WHERE TABLE_SCHEMA IN ('fp_v2','fgp_v1','sym_v1','ent_v1')
  AND (COLUMN_NAME LIKE '%adj%'    OR COLUMN_NAME LIKE '%split%'
    OR COLUMN_NAME LIKE '%factor%' OR COLUMN_NAME LIKE '%cum%'
    OR COLUMN_NAME LIKE '%div%')
ORDER BY TABLE_SCHEMA, TABLE_NAME, COLUMN_NAME;

-- 10b. Point-in-time (_hist) table hunt ---------------------------
SELECT s.name AS sch, t.name AS tbl
FROM sys.tables t JOIN sys.schemas s ON s.schema_id=t.schema_id
WHERE t.name LIKE '%hist%'   OR t.name LIKE '%sector%'
   OR t.name LIKE '%entity%' OR t.name LIKE '%rbics%'
   OR t.name LIKE '%gics%'   OR t.name LIKE '%action%'
   OR t.name LIKE '%delist%'
ORDER BY s.name, t.name;
-- Any table with BOTH start_date and end_date is point-in-time:
SELECT TABLE_SCHEMA, TABLE_NAME, COUNT(*) AS n_window_cols
FROM INFORMATION_SCHEMA.COLUMNS
WHERE COLUMN_NAME IN ('start_date','end_date')
GROUP BY TABLE_SCHEMA, TABLE_NAME
HAVING COUNT(*) = 2
ORDER BY TABLE_SCHEMA, TABLE_NAME;

-- 11. THE SANITY CHECK: US common-equity distinct ids by year -------
--     Adjust both IN-lists using the results of step 9.
--     Expect ~7-8k in 2000 declining to ~4-4.5k by 2020.
--     Flat or RISING = bug. Run last; cancel if >10 min and
--     re-run restricted to 3 sample years.
WITH u AS (
  SELECT c.fsym_regional_id AS fsym_id
  FROM   sym_v1.sym_coverage c WITH (NOLOCK)
  WHERE  c.universe_type='EQ' AND c.regional_flag=1 AND c.currency='USD'
    AND  c.fref_listing_exchange IN ('NYS','NAS','ASE')   -- from step 9
    AND  c.fref_security_type    IN ('SHARE')             -- from step 9
    -- *** DELIBERATELY NO active_flag FILTER ***
)
SELECT YEAR(p.p_date) AS yr,
       COUNT(DISTINCT p.fsym_id) AS n_ids,
       COUNT_BIG(*)              AS n_rows
FROM   fp_v2.fp_basic_prices p WITH (NOLOCK)
JOIN   u ON u.fsym_id = p.fsym_id
WHERE  p.p_date >= '2000-01-01'
GROUP  BY YEAR(p.p_date)
ORDER  BY yr
OPTION (MAXDOP 2);
-- This also replaces the 75.6M row estimate with a MEASUREMENT.
-- Then run it AGAIN with "AND c.active_flag=1" added: the difference
-- between the two result sets IS the survivorship bias, quantified.
-- Keep both.

-- 12. Measure the link -------------------------------------------
-- Time a bounded shard and divide. Everything in the wire arithmetic
-- scales off this number.
SELECT TOP 5000000 fsym_id, p_date, p_price
FROM fp_v2.fp_basic_prices WITH (NOLOCK) OPTION (MAXDOP 2);
```

**The five queries that change the design:** 4, 5, 6, 7, 9. Queries 10a/10b close three open unknowns. Query 11 replaces the row estimate with a measurement and quantifies the survivorship bias. Query 12 determines whether office day is 15 minutes or 3 hours.

---

## 14. Open unknowns

To be resolved by §13 before any pipeline code is written:

| # | Unknown | Resolved by |
|---|---|---|
| 1 | `fgp_v1.fgp_global_prices` column names entirely — do **not** assume `p_date`/`p_price` carry over from `fp_v2`; FGP is a different product generation and may key on listing (`-L`) rather than regional (`-R`) | query 5 |
| 2 | **Where the cumulative price-adjustment factor lives.** `fp_basic_prices` shows no such column publicly. It may be a separate splits/dividends table. **Highest-value unknown** — if it does not exist, it must be derived from the corporate-action event table as a backwards cumulative product | query 10a |
| 3 | Full `fref_security_type` domain — the publicly documented set (`SHARE`, `PREFEQ`, `MF_O`, `MF_C`) is certainly incomplete; codes for ETF, ADR, warrant, unit, right and depositary must exist | query 9 |
| 4 | Whether `sym_ticker_region` is a snapshot or has a `_hist` variant. If snapshot only, dead tickers may be absent — so it must not be used as the US filter | query 10b |
| 5 | Whether sector tables (`sym_entity_sector`, `sym_entity_sector_rbics`) are snapshots or have `_hist` siblings. Snapshots are a look-ahead landmine (§3.2) | query 10b |
| 6 | Corporate-actions / delisting schema and table naming | query 10b |
| 7 | **Whether `fp_v2` or `fgp_v1` is authoritative for US.** Compare on ~50 symbols and pick one. **Do not blend them.** | queries 5, 8 |
| 8 | Whether the price tables have indexes supporting a sharded pull, or are heaps | query 6 |
| 9 | `varchar` vs `nvarchar` on `fsym_id` — worth ~2.7 GB of wire time on its own | query 5 |
| 10 | Sustained office-link throughput to ap-south-1 | query 12 |
| 11 | Whether `arrow-odbc` installs cleanly on the chosen Python | §12.1 dry run |
| 12 | Columnar ADBC MSSQL driver licence terms and cost — could not be verified from documentation | manual check |

**Confidence note on the FactSet symbology research:** the fsym_id kind conventions (`-S` security, `-R` regional, `-L` listing, `-E` entity), the fact that price tables key on `-R`, and the `sym_coverage` column set were verified from public documentation. Everything in the table above was **not** verifiable and must be confirmed against the live environment.

---

## 15. Condensed decision table

| Naive method | Why it loses |
|---|---|
| `pd.read_sql` for the fact table | 14× slower **and ~10 GB peak RAM** on an 8 GB machine. The memory is the disqualifier, not the speed. |
| Pull the whole table, filter in pandas | 121 GB on the wire vs 2.65 GB — **45.7×**. Hours vs minutes. |
| `WHERE fsym_id IN (30,000 literals)` | 330 KB of SQL; optimiser degrades in the low thousands. `#temp` + JOIN instead. |
| `active_flag = 1` | **Silently introduces survivorship bias.** Removes ~half the universe, non-randomly. The wrongness looks like alpha. |
| Snapshot sector tables | Look-ahead: classifies a 2007 company by its 2026 sector — and sector drives the clustering universe. |
| Storing adjusted prices only | Tomorrow's split silently invalidates 25 years of history with no way to detect it. |
| `float32` for volume | **Silent corruption** above 2²⁴ = 16,777,216. Poisons liquidity filters for years. |
| `float64` everywhere | 2× the biggest columns for precision the source does not have. |
| `fsym_id` string in the fact table | Cheap on disk (dict+RLE), but **3× RAM**, 3–5× slower joins, and 8–40 bytes/row on an uncompressed wire. |
| Partition by symbol | 30,000 files × 33 KB: footer overhead, dictionary reset per file, 30,000 opens per cross-section. |
| Unsorted Parquet | Forfeits RLE on `sid`, DELTA on date, and most of BYTE_STREAM_SPLIT → **~1.5× larger**. |
| ZSTD level 22 | <1.5% smaller than level 9 for ~10× the write time. Level 9 is the knee. |
| `ORDER BY` in the extraction query | **Tempdb sort spill on a shared production server** — can take the instance down for everyone. Sort at home instead. |
| `OFFSET`/`FETCH` pagination | O(n) per page → ~2,900M rows generated to deliver 76M (**38×**). Use range predicates on an indexed column. |
| N shards on an unindexed column | **N full scans of a 1B-row table.** Slower than one pass, and it hammers a shared box. |
| Concurrency 16 on someone else's server | 16 connections × MAXDOP 8 = 128 workers seized. Use 3 + `OPTION (MAXDOP 1)`. |
| Multiprocessing the extraction | Parallelises waiting on a socket; 8× RAM that isn't available. |
| `asyncio` anywhere here | No REST path exists. Coloured functions and worse debugging for zero gain. |
| Dask / Ray / Spark at 2 GB | 100% of the ops complexity for 0% of the benefit. |
| Multiprocessing before profiling | Yields a beautifully parallel accidentally-quadratic algorithm. `smf.ols` → `lstsq` is **~100×** and comes first. |
| `lru_cache` on an instance method | Leaks `self`, unbounded, no cross-process sharing, no invalidation. Use Parquet keyed by `config_hash`. |
| Appending Parquet files per day | Destroys partition sort order → loses pushdown and compression. Rewrite the 40 MB year file. |
| `df.to_csv()` per year with no manifest | Interrupt-unsafe (a truncated file reads as valid) and non-resumable. Use `.tmp` + `os.replace()` + a checksummed manifest. |
| Database in git | 100 MB file cap, whole-blob-per-revision history, **and licence redistribution.** Code + manifests only. |
| Trusting the data because it loaded | The oracle checksum is what separates a database you can bet money on from one you hope is right. |

---

## Appendix: what the old CRSP pipeline got wrong

Kept as a reference list of failure modes the new design must not reproduce.

| Defect | Location | Impact |
|---|---|---|
| Delisting returns never queried (`dsedelist`/`dlret`/`dlstcd` absent) | `data/crsp_puller.py` | **Survivorship bias + silent P&L truncation at bankruptcy.** Inflates returns. |
| `ret`/`retx` pulled then dropped by the aggregator | `data/data_aggregator.py` | Strategy runs on price returns, **dividends excluded** |
| `cfacshr` pulled but never applied | `data/crsp_puller.py` | `market_cap` mixes unadjusted shares with adjusted price |
| No `cfacpr == 0` guard | `data/data_aggregator.py` | `inf` values in `adj_close` |
| GICS joined on `permno` with no `linkdt`/`linkenddt` | `data/crsp_puller.py` | **Look-ahead in sector assignment** + duplicate rows |
| Share-code filter applied in pandas, not SQL | `data/data_aggregator.py:65` | Every non-common row crosses the wire first |
| `pivot_table(aggfunc='first')` | `data/data_aggregator.py` | **Silently masks** duplicate `(permno, date)` rows instead of failing |
| No liquidity screen | pipeline-wide | Penny stocks and untradeable names in the universe |
| Wide per-sector pickles as interchange format | `data/data_aggregator.py` | ~5.4 GB and 5.4 GB RAM to open; **fails on 8 GB** |
| Import-time pickle read from `os.getcwd()` | `config/settings.py` | Blocks all testing, packaging and CI |
| Trading calendar derived from one stock's dates | `config/settings.py` | Encodes that stock's halts as market holidays |
| `get_loc(..., method="bfill")` | `helpers/rolling_engine.py:102` | Removed in pandas 2.0 — cannot have been run recently |
| Forward z-window (61) exceeds forward period (~21) | `helpers/rolling_engine.py` | **Every pair skipped; P&L always empty** |
| Engle-Granger run both ways, min p-value taken, no multiplicity correction | `models/cointegration.py` | Anti-conservative p-values → spurious "cointegration" |
| Cross-sectional demeaning applied to price *levels*, then `pct_change` | `data/data_processor.py` | Not demeaning of returns |
| Two duplicate engines; half-life ×2, hedge ratio ×2, panel loader ×3 | `helpers/` vs `models/` | Divergent behaviour, doubled maintenance |
| No transaction costs, slippage, borrow, or financing anywhere | pipeline-wide | Backtest is not investable |
| No performance metrics (no Sharpe, drawdown, turnover) | pipeline-wide | No way to evaluate a result |
