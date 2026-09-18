"""
STEP 1 of the data platform build: RAW extraction from FactSet (SQL Server)
                                   -> raw Parquet landing zone.

This is TIER 0 only. No schema enforcement, no sorting, no ACID, no versioning -
those come in step 2 (Delta Lake). Here we only want the bytes off the server,
correctly and resumably, because server access is office-network-only and this
extraction is effectively one-shot.

    config.py    all settings; credentials read from .env, never hardcoded
    queries.py   every SQL statement, annotated field by field
    this file    orchestration only

To review WHAT is being pulled, read queries.py. To review HOW, read this.

Run it:
    python extract_raw_factset.py                # dry run: prints all SQL
    python extract_raw_factset.py --execute      # resumable; skips complete shards
    python extract_raw_factset.py --full-refresh # re-extract everything, ignoring
                                                 # manifest completion state

Nothing connects to anything while config.DRY_RUN is True.

-------------------------------------------------------------------------------
WHY THE EXTRACTION IS SHAPED THIS WAY (docs/DATABASE_DESIGN.md has the full case)
-------------------------------------------------------------------------------
* Universe-first: resolve ~34k securities from the small symbology tables, push
  them into a #temp table, then join the 1.01B-row price table against it.
  Turns a 121 GB full scan into a ~3.2 GB targeted pull - about 45x.
* sid substitution: send an int32 surrogate over the wire instead of the 8-40
  byte fsym_id string, 92M times. TDS has NO wire compression, so this is real.
* Server-side CAST: the SELECT list IS the wire format.
* Two-phase: raw here (fast, unsorted, ZSTD-3), curated at home (sorted,
  ZSTD-9, Delta). Decoupling them means a bad compression choice costs a
  10-minute redo at home instead of a wasted trip to the office.
"""

from __future__ import annotations

import argparse
import sys

from . import config, queries


# =============================================================================
# BLOCKERS -- must be resolved before a real run
# =============================================================================

BLOCKERS: list[tuple[str, str]] = [
    ("D2 / D7", "Does fp_basic_prices have p_price_open/high/low? If not, the "
                "OHLCV requirement needs another source."),
    ("D6",      "Corporate-action + adjustment-factor table names. Nothing is "
                "guessed - a wrong split convention silently inverts 25 years "
                "of prices into a series that still looks plausible."),
    ("D3",      "Index layout -> config.SHARD_MODE. Sharding on an unindexed "
                "column means N full scans of a ~120 GB table."),
    ("D4",      "Real US exchange codes -> config.US_EXCHANGE_CODES."),
    ("D5",      "p_sec_type_code domain -> config.SEC_TYPE_CODES."),
    ("D8",      "p_volume units: raw shares or thousands? The other team's "
                "/1000 and FactSet docs disagree. A 1000x error breaks every "
                "liquidity screen built on it."),
    ("D6(iii)", "Does sym_entity_sector have a _hist variant? Without one, "
                "sector is a snapshot and injects look-ahead into clustering."),
]


# =============================================================================
# DRY RUN -- print every statement, execute nothing
# =============================================================================

def _banner(text: str, char: str = "=") -> str:
    return f"\n{char * 79}\n{text}\n{char * 79}"


def build_plan() -> list[tuple[str, list[tuple[str, str]]]]:
    """The full ordered set of statements this extraction would issue.

    Returned as data rather than printed directly, so the same structure can
    drive the dry-run printer, a query-hash manifest, and eventually the real
    execution path - one source of truth for "what runs, in what order".
    """
    shards = queries.shard_where_clauses()

    return [
        ("SECTION A - DISCOVERY (run first; resolves every blocker)", [
            (name, fn()) for name, fn in queries.DISCOVERY_QUERIES.items()
        ]),
        ("SECTION B - SESSION PRELUDE (every connection AND every reconnect)", [
            ("session_prelude", queries.SESSION_PRELUDE),
        ]),
        ("SECTION C - UNIVERSE (~34k rows)", [
            ("universe", queries.universe()),
            ("universe_WITH_active_filter  [measurement only: the row-count "
             "difference IS your survivorship bias]",
             queries.universe(include_active_filter=True)),
        ]),
        ("SECTION D - TEMP TABLES (filter + wire compression)", [
            ("ddl_temp_tables",     queries.DDL_TEMP_TABLES),
            ("insert_universe",     queries.INSERT_TEMP_UNIVERSE),
            ("insert_universe_sec", queries.INSERT_TEMP_UNIVERSE_SEC),
            ("insert_universe_ent", queries.INSERT_TEMP_UNIVERSE_ENT),
        ]),
        (f"SECTION E - PRICE FACT (~92M rows; {len(shards)} shards, "
         f"first 2 shown)", [
            (f"prices_shard[{i}]", queries.prices_shard(w))
            for i, w in enumerate(shards[:2])
        ]),
        ("SECTION F - SHARES / FLOAT / MARKET CAP", [
            ("shares_and_prices_own", queries.shares_and_prices_own()),
            ("free_float_hist",       queries.free_float_hist()),
            ("entity_market_value",   queries.entity_market_value()),
        ]),
        ("SECTION G - SECTOR (point-in-time status pending D6-iii)", [
            ("sector", queries.sector()),
        ]),
        ("SECTION H - CORPORATE ACTIONS", [
            ("*** BLOCKED ON D6 ***", queries.CORPORATE_ACTIONS_BLOCKED_NOTE),
        ]),
        ("SECTION I - ORACLE (server-authored truth; carry this home)", [
            ("oracle_checksum",        queries.oracle_checksum()),
            ("oracle_universe_counts", queries.oracle_universe_counts()),
        ]),
    ]


def print_plan() -> None:
    shards = queries.shard_where_clauses()

    print(_banner("DRY RUN - no connection opened, nothing executed"))
    print(f"  Connection   : {config.safe_connection_summary()}")
    print(f"  Destination  : {config.DESTINATION_ROOT}")
    print(f"  Window       : {config.START_DATE} .. {config.END_DATE}")
    print(f"  Universe     : types={config.SECURITY_TYPES}")
    print(f"                 exchanges={config.US_EXCHANGE_CODES}")
    print(f"  Sec types    : {config.SEC_TYPE_CODES or '(no filter yet - run D5)'}")
    print(f"  Shard mode   : {config.SHARD_MODE}  ({len(shards)} shards)")
    print(f"  Workers      : {config.N_WORKERS}   MAXDOP={config.MAXDOP}")
    print(f"  Raw Parquet  : ZSTD-{config.ZSTD_LEVEL_RAW}, "
          f"row_group={config.ROW_GROUP_RAW:,}")
    print(f"  .env         : {config.ENV_PATH} "
          f"({'found' if config.ENV_PATH.is_file() else 'NOT FOUND'})")

    for title, statements in build_plan():
        print(_banner(f"# {title}", "#"))
        for name, sql in statements:
            print(f"\n--- {name} " + "-" * max(0, 70 - len(name)))
            print(sql.rstrip())
            if "SELECT" in sql.upper():
                print(f"--   query_hash: {queries.query_hash(sql)}")

    print(_banner("BLOCKERS - resolve before setting config.DRY_RUN = False"))
    for tag, note in BLOCKERS:
        print(f"  [{tag:9}] {note}")
    print()


# =============================================================================
# REAL RUN -- not implemented yet, by design
# =============================================================================

def run_extraction(limit_shards: int | None = None,
                   full_refresh: bool = False) -> None:
    """Run the real extraction. Implemented in extractor.py."""
    # Imported lazily, not at module scope: keeps pyarrow out of the import path
    # of everything that does not need it. extractor itself imports arrow_odbc
    # lazily too, so importing it here is safe on machines with no ODBC driver.
    from . import extractor

    if limit_shards is not None:
        # Smoke-test mode: cap the number of price shards. Worth doing before the
        # full run, because office access is one-shot - you want to have proven
        # the whole path end to end on a small slice first.
        original = extractor.price_shards

        def limited(universe, per_shard=400):
            return original(universe, per_shard)[:limit_shards]

        extractor.price_shards = limited    # type: ignore[assignment]
        print(f"SMOKE TEST: limiting to {limit_shards} price shard(s)\n")

    extractor.run(full_refresh=full_refresh)


def _old_run_extraction_notes() -> None:
    """Historical design notes for the extraction, kept for reference.

    Deliberately unwritten until the discovery blockers are closed, because
    several structural choices depend on their answers: the shard key, whether
    OHLC exists at all, and the entire corporate-action section. Writing it now
    would mean rewriting it after discovery.

    When it is written it needs, in order:

      1. Resolve the universe -> assign sid (dense rank over sorted
         fsym_regional_id). Append-only, checksummed, NEVER renumbered:
         renumbering silently rebinds every cached artifact to the wrong stock.

      2. Per worker connection: SESSION_PRELUDE, then create and populate all
         three #temp tables. Re-do BOTH on any reconnect - temp tables are
         connection-scoped and die with the session.

      3. Stream each shard with arrow-odbc -> Parquet, bounded memory.
         max_bytes_per_batch MUST be lowered from its 512 MB default; with
         fetch_concurrently that default is ~1 GB of buffers on an 8 GB machine.

      4. Atomic writes: part.parquet.tmp -> fsync -> os.replace(). A Parquet
         file missing its footer is unreadable, so a partial file bearing the
         final name is a trap that reads as valid until you query it.

      5. Append a manifest line per completed shard: rows, sha256, date range,
         query_hash, universe_hash. Restart skips a shard only if it is marked
         complete AND its checksum matches - "the file exists" is not enough,
         because a truncated file also exists.

      6. Pull the oracle and reconcile EVERY shard against it BEFORE leaving the
         office. There is no second attempt from home.

      7. Copy to the external drive incrementally during the run, not once at
         the end. If the laptop dies at 90%, you want 90% already portable.
    """
    raise SystemExit(
        "Real extraction not implemented yet - by design.\n"
        "Resolve the discovery blockers first:\n"
        + "\n".join(f"  [{tag}] {note}" for tag, note in BLOCKERS)
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="FactSet raw extraction - step 1, raw Parquet landing zone"
    )
    parser.add_argument("--queries", action="store_true",
                        help="print every SQL statement and exit (default)")
    parser.add_argument("--execute", action="store_true",
                        help="run the real extraction")
    parser.add_argument("--smoke", type=int, metavar="N", default=None,
                        help="run only N price shards - do this before the full "
                             "run, since office access is one-shot")
    parser.add_argument("--full-refresh", action="store_true",
                        help="re-extract every artifact, ignoring manifest "
                             "completion state. Re-downloading used to need "
                             "manual manifest deletion; every shard spans the "
                             "full date range, so a plain re-run skips new data "
                             "as already-complete. The manifest is kept as "
                             "history. Implies a real run; combinable with "
                             "--smoke to prove the path on a few shards first")
    args = parser.parse_args(argv)

    if args.execute or args.smoke is not None or args.full_refresh:
        run_extraction(limit_shards=args.smoke, full_refresh=args.full_refresh)
        return 0

    print_plan()
    return 0


if __name__ == "__main__":
    sys.exit(main())
