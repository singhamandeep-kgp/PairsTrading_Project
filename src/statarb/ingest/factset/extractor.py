"""
The extraction machinery: FactSet SQL Server -> raw Parquet landing zone.

TIER 0 ONLY. No schema enforcement, no sorting, no ACID. Step 2 (Delta Lake)
does that. The single goal here is to get the bytes off the server correctly and
resumably, because server access is office-network-only and this is effectively
one-shot.

Design decisions that matter, all of them driven by measured facts (see
docs/DISCOVERY_FINDINGS.md):

* arrow-odbc streams ODBC -> pyarrow.RecordBatch with a BOUNDED transit buffer.
  pandas.read_sql would need ~10 GB of RAM for this pull on an 8 GB machine; the
  memory, not the 14x speed gap, is what disqualifies it.
* ONE Connection per worker does prelude -> #temp DDL -> populate -> read.
  Temp tables are connection-scoped, so this cannot be split across connections.
* fp_basic_prices has a CLUSTERED PK on (fsym_id, p_date), so sharding on sid
  ranges is a clustered range SEEK and rows arrive ALREADY SORTED in our target
  order. The step-2 sort is therefore free.
* Atomic writes: .tmp -> fsync -> os.replace(). A Parquet file missing its footer
  is unreadable, so a partial file bearing the final name is a trap that reads as
  valid until you query it.
* Manifest gates restart on sha256, not on file existence - a truncated file
  also exists.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

import pyarrow as pa
import pyarrow.parquet as pq

from . import config, queries

import arrow_odbc

_PRINT_LOCK = threading.Lock()
_FSYM_RE = re.compile(r"^[A-Za-z0-9\-]{1,20}$")


def log(msg: str) -> None:
    with _PRINT_LOCK:
        print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


# =============================================================================
# MANIFEST
# =============================================================================

class Manifest:
    """Append-only JSONL, one line per completed artifact.

    Append-only matters: a crash mid-write corrupts at most the final line,
    which is skipped on read. A rewritten JSON document could be lost entirely.
    """

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._done: dict[str, dict[str, Any]] = {}
        if path.is_file():
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue          # truncated final line - ignore it
                if rec.get("status") == "complete":
                    self._done[rec["name"]] = rec

    def is_complete(self, name: str, root: Path) -> bool:
        """Complete iff recorded AND the on-disk checksum still matches.

        Checking the hash rather than mere existence is the whole point: an
        interrupted write leaves a file that exists and is unreadable.
        """
        rec = self._done.get(name)
        if rec is None:
            return False
        f = root / rec["file"]
        if not f.is_file() or f.stat().st_size != rec["bytes"]:
            return False
        return sha256_file(f) == rec["sha256"]

    def append(self, rec: dict[str, Any]) -> None:
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, default=str) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
            self._done[rec["name"]] = rec


def is_transient(exc: BaseException) -> bool:
    """Should this failure be retried?

    Only server/network errors are worth retrying. A local OSError or a
    programming bug is deterministic - retrying it five times with exponential
    backoff just burns minutes and buries the real message. That matters here:
    a bug in the write path would otherwise cost ~25 minutes across the aux
    tables before surfacing, on the one day server access exists.
    """
    return isinstance(exc, arrow_odbc.Error)


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# =============================================================================
# SESSION
# =============================================================================

def _conn_str_no_creds() -> str:
    """Connection string without credentials; user/password passed separately so
    the password never appears in a string that might be logged."""
    # ConnectRetryCount/Interval let the driver transparently re-establish a
    # dropped connection. They do NOT save a query already in flight, which is
    # why the real fix for ODBC 10053 was to stop issuing statements that run
    # for minutes with no rows flowing (see queries.universe docstring).
    return (
        f"DRIVER={{{config.ODBC_DRIVER}}};"
        f"SERVER={config.SERVER_NAME};"
        f"DATABASE={config.DATABASE};"
        f"Encrypt=no;TrustServerCertificate=yes;"
        f"ConnectRetryCount=3;ConnectRetryInterval=10;"
    )


def open_session(password: str, universe: list[tuple] | None = None):
    """Open a connection, apply the session prelude, and optionally build and
    populate the three #temp universe tables.

    MUST be called again from scratch after any reconnect: temp tables and SET
    options both die with the session, and retry logic that re-issues only the
    query is the classic source of "worked for hours then failed weirdly".
    """
    conn = arrow_odbc.connect(
        connection_string=_conn_str_no_creds(),
        user=config.USER_NAME,
        password=password,
        login_timeout_sec=config.LOGIN_TIMEOUT,
        packet_size=config.PACKET_SIZE,
    )
    for stmt in _split_batches(queries.SESSION_PRELUDE):
        conn.execute(stmt)

    if universe is not None:
        for stmt in _split_batches(queries.DDL_TEMP_TABLES):
            conn.execute(stmt)
        _populate_temp(conn, universe)
    return conn


def _split_batches(sql: str) -> list[str]:
    """Split a multi-statement block into individual statements.

    ODBC executes one statement per call, and SQL Server's `GO` is an SSMS
    batch separator rather than T-SQL, so it must never be sent.
    """
    cleaned = re.sub(r"/\*.*?\*/", "", sql, flags=re.S)
    out = []
    for part in cleaned.split(";"):
        stmt = "\n".join(
            ln for ln in part.splitlines()
            if ln.strip() and not ln.strip().startswith("--")
            and ln.strip().upper() != "GO"
        ).strip()
        if stmt:
            out.append(stmt)
    return out


def _populate_temp(conn, universe: list[tuple]) -> None:
    """Bulk-load the universe into the three #temp tables.

    Multi-row INSERT ... VALUES at SQL Server's 1000-row-per-statement limit.
    Row-at-a-time would be ~16,000 WAN round trips; this is ~17 statements per
    table and takes a couple of seconds.
    """
    def chunks(seq, n=1000):
        for i in range(0, len(seq), n):
            yield seq[i:i + n]

    for rows in chunks([(sid, fid) for sid, fid, _, _ in universe]):
        vals = ", ".join(f"({sid}, '{_safe(fid)}')" for sid, fid in rows)
        conn.execute(f"INSERT INTO #universe (sid, fsym_id) VALUES {vals}")

    sec = [(sid, sec_id) for sid, _, sec_id, _ in universe if sec_id]
    for rows in chunks(sec):
        vals = ", ".join(f"({sid}, '{_safe(s)}')" for sid, s in rows)
        conn.execute(
            f"INSERT INTO #universe_sec (sid, fsym_security_id) VALUES {vals}")

    ents = sorted({e for _, _, _, e in universe if e})
    for rows in chunks(ents):
        vals = ", ".join(f"('{_safe(e)}')" for e in rows)
        conn.execute(
            f"INSERT INTO #universe_ent (factset_entity_id) VALUES {vals}")


def _safe(identifier: str) -> str:
    """Validate a FactSet identifier before string-interpolating it.

    These values come from the database itself, not from user input, but the
    check is cheap and it means a surprising value fails loudly instead of
    becoming SQL.
    """
    s = str(identifier)
    if not _FSYM_RE.match(s):
        raise ValueError(f"refusing to interpolate suspicious identifier: {s!r}")
    return s


# =============================================================================
# PARQUET WRITING
# =============================================================================

def write_reader_to_parquet(reader, out_path: Path) -> tuple[int, int]:
    """Stream an arrow-odbc BatchReader straight to Parquet.

    Bounded memory: one RecordBatch at a time, never a full materialisation.
    Atomic: write .tmp, fsync, then os.replace() - so a reader never observes a
    Parquet file without its footer.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    n_rows = 0
    writer: pq.ParquetWriter | None = None
    try:
        for batch in reader:
            if writer is None:
                writer = pq.ParquetWriter(
                    tmp, batch.schema,
                    compression="zstd",
                    compression_level=config.ZSTD_LEVEL_RAW,
                    use_dictionary=True,
                    write_statistics=True,
                )
            writer.write_batch(batch)
            n_rows += batch.num_rows
        if writer is None:      # empty result set: still emit a valid file
            writer = pq.ParquetWriter(
                tmp, pa.schema([]), compression="zstd",
                compression_level=config.ZSTD_LEVEL_RAW)
    finally:
        if writer is not None:
            writer.close()

    # fsync before the rename, so the rename cannot land in the directory entry
    # ahead of the file's own data. Must be opened for WRITING: on Windows,
    # fsync() against a read-only handle fails with EBADF ("Bad file descriptor").
    with open(tmp, "r+b") as fh:
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, out_path)
    return n_rows, out_path.stat().st_size


def _record(name: str, path: Path, root: Path, n_rows: int, n_bytes: int,
            sql: str, universe_hash: str, extra: dict | None = None) -> dict:
    return {
        "name": name,
        "file": str(path.relative_to(root)),
        "rows": n_rows,
        "bytes": n_bytes,
        "sha256": sha256_file(path),
        "query_hash": queries.query_hash(sql),
        "universe_hash": universe_hash,
        "writer": {"pyarrow": pa.__version__, "zstd_level": config.ZSTD_LEVEL_RAW},
        # server_fp, not the hostname. The manifest's only actual requirement is
        # "was this partition pulled from the same host as that one?", which a
        # fingerprint answers while carrying nothing sensitive. Scrubbing records
        # after the fact fixes today; changing the writer fixes it permanently,
        # so a manifest is safe to share or commit by construction.
        "source": {
            "server_fp": sha256_text(config.SERVER_NAME)[:12],
            "database": config.DATABASE,
        },
        "utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        **(extra or {}),
    }


# =============================================================================
# STEP 1 -- UNIVERSE + sid ASSIGNMENT
# =============================================================================

SID_MAP_NAME = "sid_map.parquet"


def resolve_universe(password: str) -> tuple[list[tuple], str]:
    """Resolve the universe and assign the permanent `sid` surrogate key.

    sid is a dense rank over sorted fsym_regional_id, and it is APPEND-ONLY:
    existing ids are never renumbered. Renumbering would silently rebind every
    cached artifact downstream to the wrong stock, which is unrecoverable
    without rebuilding everything.
    """
    root = config.DESTINATION_ROOT
    root.mkdir(parents=True, exist_ok=True)
    sql = queries.universe()

    log("resolving universe (joins over ~99.5M-row sym_coverage; expect minutes)")
    conn = open_session(password)
    try:
        reader = conn.read_arrow_batches(
            query=sql,
            batch_size=config.BATCH_SIZE,
            max_bytes_per_batch=config.MAX_BYTES_BATCH,
            max_text_size=config.MAX_TEXT_SIZE,
            query_timeout_sec=config.QUERY_TIMEOUT,
        )
        tbl = pa.Table.from_batches(list(reader))
    finally:
        del conn

    n = tbl.num_rows
    log(f"universe resolved: {n:,} securities")
    if not (config.EXPECTED_UNIVERSE_MIN <= n <= config.EXPECTED_UNIVERSE_MAX):
        raise SystemExit(
            f"universe size {n:,} is outside the sanity band "
            f"[{config.EXPECTED_UNIVERSE_MIN:,}, {config.EXPECTED_UNIVERSE_MAX:,}].\n"
            "Refusing to extract against a filter that is probably wrong. "
            "Check US_EXCHANGE_CODES / SEC_TYPE_CODES in config.py."
        )

    # --- survivorship tripwire ------------------------------------------------
    # If nearly everything is still active, an active_flag filter has crept in
    # and every backtest built on this data would be invalid.
    active = tbl.column("active_flag").to_pylist()
    n_dead = sum(1 for a in active if not a)
    frac_dead = n_dead / max(n, 1)
    log(f"dead securities: {n_dead:,} of {n:,} ({frac_dead:.1%})")
    if frac_dead < 0.30:
        raise SystemExit(
            f"SURVIVORSHIP TRIPWIRE: only {frac_dead:.1%} of the universe is "
            "dead, expected >30%. An active_flag filter has probably been "
            "reintroduced. Refusing to continue."
        )

    # --- sid assignment, append-only -----------------------------------------
    reg = tbl.column("fsym_regional_id").to_pylist()
    sec = tbl.column("fsym_security_id").to_pylist()
    ent = tbl.column("factset_entity_id").to_pylist()

    sid_path = root / SID_MAP_NAME
    existing: dict[str, int] = {}
    if sid_path.is_file():
        prev = pq.read_table(sid_path)
        existing = dict(zip(prev.column("fsym_regional_id").to_pylist(),
                            prev.column("sid").to_pylist()))
        log(f"loaded {len(existing):,} existing sid assignments (append-only)")

    next_sid = (max(existing.values()) + 1) if existing else 1
    for fid in sorted(reg):
        if fid not in existing:
            existing[fid] = next_sid
            next_sid += 1

    pq.write_table(
        pa.table({"sid": [existing[f] for f in sorted(existing)],
                  "fsym_regional_id": sorted(existing)}),
        sid_path, compression="zstd")

    universe = [(existing[r], r, s, e) for r, s, e in zip(reg, sec, ent)]

    # dim_security: the universe table plus sid, written verbatim
    dim = tbl.append_column("sid", pa.array([existing[r] for r in reg], pa.int32()))
    dim_path = config.RAW_DIR / "dim_security.parquet"
    dim_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(dim, dim_path, compression="zstd",
                   compression_level=config.ZSTD_LEVEL_RAW)
    log(f"wrote dim_security.parquet ({n:,} rows)")

    universe_hash = sha256_text("|".join(sorted(reg)))
    return universe, universe_hash


# =============================================================================
# STEP 2 -- SHARDED PRICE EXTRACTION
# =============================================================================

def price_shards(universe: list[tuple], per_shard: int = 400) -> list[tuple[str, str]]:
    """Build (shard_name, where_clause) pairs from the ACTUAL sid range.

    Sharding on sid means the predicate is always covered by our own #universe
    PRIMARY KEY, and the underlying clustered index on (fsym_id, p_date) makes
    each shard a range seek rather than a scan.

    per_shard targets ~3-8 minutes of work: small enough that a dropped
    connection costs little, large enough that the ~3s per-shard setup stays
    under a couple of percent.
    """
    sids = sorted(sid for sid, *_ in universe)
    out = []
    for i in range(0, len(sids), per_shard):
        lo = sids[i]
        hi = sids[min(i + per_shard, len(sids) - 1)] if i + per_shard < len(sids) \
            else sids[-1] + 1
        out.append((
            f"prices_sid_{lo:06d}_{hi:06d}",
            f"u.sid >= {lo} AND u.sid < {hi} "
            f"AND p.p_date >= '{config.START_DATE}' "
            f"AND p.p_date <= '{config.END_DATE}'",
        ))
    return out


def extract_prices(password: str, universe: list[tuple], universe_hash: str,
                   manifest: Manifest) -> None:
    shards = price_shards(universe)
    root = config.DESTINATION_ROOT
    todo = [s for s in shards if not manifest.is_complete(s[0], root)]
    log(f"price shards: {len(shards)} total, {len(todo)} to do, "
        f"{len(shards) - len(todo)} already complete")
    if not todo:
        return

    counter = {"done": 0, "rows": 0, "bytes": 0}
    t0 = time.time()

    def work(shard: tuple[str, str]) -> None:
        name, where = shard
        sql = queries.prices_shard(where)
        out = config.RAW_DIR / "prices" / f"{name}.parquet"

        for attempt in range(1, config.MAX_RETRIES + 1):
            conn = None
            try:
                # Full session rebuild on every attempt: prelude AND #temp
                # tables, because both die with a dropped connection.
                conn = open_session(password, universe)
                reader = conn.read_arrow_batches(
                    query=sql,
                    batch_size=config.BATCH_SIZE,
                    max_bytes_per_batch=config.MAX_BYTES_BATCH,
                    max_text_size=config.MAX_TEXT_SIZE,
                    fetch_concurrently=config.FETCH_CONCURRENTLY,
                    query_timeout_sec=config.QUERY_TIMEOUT,
                )
                n_rows, n_bytes = write_reader_to_parquet(reader, out)
                manifest.append(_record(name, out, root, n_rows, n_bytes,
                                        sql, universe_hash,
                                        {"shard_where": where}))
                with _PRINT_LOCK:
                    counter["done"] += 1
                    counter["rows"] += n_rows
                    counter["bytes"] += n_bytes
                    el = time.time() - t0
                    rate = counter["rows"] / max(el, 1)
                    print(f"[{datetime.now():%H:%M:%S}] "
                          f"{counter['done']}/{len(todo)} {name} "
                          f"{n_rows:,} rows {n_bytes/1e6:.1f} MB "
                          f"| total {counter['rows']:,} rows "
                          f"{counter['bytes']/1e6:.0f} MB "
                          f"| {rate:,.0f} rows/s", flush=True)
                return
            except Exception as exc:  # noqa: BLE001
                if not is_transient(exc):
                    log(f"!! {name} NON-TRANSIENT error, not retrying: {exc!r}")
                    raise
                if attempt == config.MAX_RETRIES:
                    # Log and move on. Finishing the other 40 shards and
                    # re-running one beats losing hours to a single lock timeout.
                    log(f"!! {name} FAILED PERMANENTLY after {attempt} attempts: "
                        f"{str(exc)[:200]}")
                    return
                back = config.RETRY_BACKOFF ** attempt
                log(f"   {name} attempt {attempt} failed "
                    f"({str(exc)[:120]}); retrying in {back:.0f}s")
                time.sleep(back)
            finally:
                del conn

    with ThreadPoolExecutor(max_workers=config.N_WORKERS) as pool:
        futures = [pool.submit(work, s) for s in todo]
        for f in as_completed(futures):
            f.result()


# =============================================================================
# STEP 3 -- AUXILIARY TABLES (corporate actions, shares, calendar, oracle)
# =============================================================================

AUX_TABLES: dict[str, Callable[[], str]] = {
    "splits":              queries.splits,
    "adjustment_factors":  queries.adjustment_factors,
    "dividends":           queries.dividends,
    "ca_events":           queries.ca_events,
    "shares_outstanding":  queries.shares_outstanding,
    "free_float_hist":     queries.free_float_hist,
    "sector_snapshot":     queries.sector,
    "trading_calendar":    queries.trading_calendar,
    "exchange_holidays":   queries.exchange_holidays,
    "oracle_checksum":     queries.oracle_checksum,
}

# ent_entity_mkt_val is 135M rows and the join to #universe_ent is evidently not
# index-supported: it ran for 12.5 minutes with no rows flowing and the
# connection was reset (ODBC 10054) - the same failure mode as the old universe
# `span` CTE.
#
# It is EXCLUDED BY DEFAULT because it is redundant. Market cap is derivable from
# data we already pull:
#     market_cap = p_com_shs_out (fp_basic_shares_hist) x p_price
# computed locally in step 2. ent_mv_ex_treasury only adds a treasury-stock
# refinement, which is not worth a connection-killing query on a one-shot
# extraction. Float-adjusted cap also works, via own_float_hist which we DO pull.
#
# Enable with FACTSET_PULL_ENTITY_MCAP=1 if you later want it; it should be
# sharded by entity range first, the way prices are.
OPTIONAL_AUX: dict[str, Callable[[], str]] = {
    "entity_market_value": queries.entity_market_value,
}

if os.environ.get("FACTSET_PULL_ENTITY_MCAP") == "1":
    AUX_TABLES.update(OPTIONAL_AUX)


# Every one of these MUST return rows. A zero-row result here means a join key is
# wrong, not that the data is absent - which is exactly how the fp_sec_entity
# primary-equity-id bug hid: two tables silently produced empty files and the run
# otherwise looked healthy.
EXPECT_NONEMPTY = {
    "splits", "adjustment_factors", "dividends", "ca_events",
    "shares_outstanding", "free_float_hist", "sector_snapshot",
    "trading_calendar", "exchange_holidays", "oracle_checksum",
}

WARNINGS: list[str] = []


def extract_aux(password: str, universe: list[tuple], universe_hash: str,
                manifest: Manifest) -> None:
    root = config.DESTINATION_ROOT
    conn = None
    try:
        for name, fn in AUX_TABLES.items():
            if manifest.is_complete(name, root):
                log(f"{name}: already complete, skipping")
                continue
            sql = fn()
            out = config.RAW_DIR / f"{name}.parquet"
            for attempt in range(1, config.MAX_RETRIES + 1):
                try:
                    if conn is None:
                        conn = open_session(password, universe)
                    reader = conn.read_arrow_batches(
                        query=sql,
                        batch_size=config.BATCH_SIZE,
                        max_bytes_per_batch=config.MAX_BYTES_BATCH,
                        max_text_size=config.MAX_TEXT_SIZE,
                        fetch_concurrently=config.FETCH_CONCURRENTLY,
                        query_timeout_sec=config.QUERY_TIMEOUT,
                    )
                    n_rows, n_bytes = write_reader_to_parquet(reader, out)
                    manifest.append(_record(name, out, root, n_rows, n_bytes,
                                            sql, universe_hash))
                    log(f"{name}: {n_rows:,} rows, {n_bytes/1e6:.1f} MB")
                    if n_rows == 0 and name in EXPECT_NONEMPTY:
                        msg = (f"{name} returned ZERO ROWS - almost certainly a "
                               f"wrong join key, not missing data. Investigate "
                               f"before trusting this extract.")
                        log(f"!! WARNING: {msg}")
                        WARNINGS.append(msg)
                    break
                except Exception as exc:  # noqa: BLE001
                    conn = None          # force a full session rebuild
                    if not is_transient(exc):
                        log(f"!! {name} NON-TRANSIENT error, not retrying: {exc!r}")
                        raise
                    if attempt == config.MAX_RETRIES:
                        log(f"!! {name} FAILED: {str(exc)[:200]}")
                        break
                    log(f"   {name} attempt {attempt} failed "
                        f"({str(exc)[:120]}); retrying")
                    time.sleep(config.RETRY_BACKOFF ** attempt)
    finally:
        del conn


# =============================================================================
# ORCHESTRATION
# =============================================================================

def run(password: str | None = None) -> None:
    password = password or config.get_password()
    root = config.DESTINATION_ROOT
    manifest = Manifest(config.MANIFEST_DIR / "manifest.jsonl")

    log("=" * 70)
    log(f"FactSet raw extraction -> {root}")
    log(f"  {config.safe_connection_summary()}")
    log(f"  window {config.START_DATE} .. {config.END_DATE}")
    log(f"  universe: {config.SECURITY_TYPES} on {config.US_EXCHANGE_CODES}"
        + (f" + OTC {config.OTC_EXCHANGE_CODES}" if config.OTC_EXCHANGE_CODES else ""))
    log(f"  workers={config.N_WORKERS} MAXDOP={config.MAXDOP} "
        f"packet={config.PACKET_SIZE}")
    log("=" * 70)

    t0 = time.time()
    universe, universe_hash = resolve_universe(password)
    log(f"universe_hash={universe_hash[:16]}  "
        "(a change here invalidates every partition)")

    extract_aux(password, universe, universe_hash, manifest)
    extract_prices(password, universe, universe_hash, manifest)

    el = time.time() - t0
    total = sum(p.stat().st_size for p in config.RAW_DIR.rglob("*.parquet"))
    log("=" * 70)
    log(f"DONE in {el/60:.1f} min. Raw lake: {total/1e9:.2f} GB at {root}")
    if WARNINGS:
        log("")
        log(f"!! {len(WARNINGS)} WARNING(S) - do not treat this extract as good:")
        for w in WARNINGS:
            log(f"   - {w}")
        log("")
    log("NEXT: verify against the oracle BEFORE leaving the office - "
        "you cannot re-pull from home.")
    log("=" * 70)
