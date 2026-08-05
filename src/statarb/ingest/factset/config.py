"""
Configuration for the FactSet raw extraction.

CREDENTIALS ARE NOT STORED IN THIS FILE. They are read from a .env file at the
repository root, which is gitignored. This matters: the repo is a git repo and
git keeps every version of every file forever, so a password committed once is
effectively public forever even after you delete it. Rotate any password that
has already been committed - deleting the line does not undo it.

Create a `.env` at the repository root. `.env.example` lists every variable and
what it means; every value there is a placeholder.

No worked example is reproduced here, deliberately. A docstring that shows the
*shape* of a filled-in credential block is how a real credential ends up pasted
into a source file -- which is exactly what happened to this module once already.
The template lives in one place, and that place is not a `.py` file.

Host, database and username have NO defaults: they are `required=True`, so a
missing variable fails loudly at startup rather than silently connecting
somewhere unintended.

If FACTSET_SQL_PASSWORD is absent you are prompted at runtime via getpass. That
is the preferred mode - the password then never touches disk or shell history.
"""

from __future__ import annotations

import os
from pathlib import Path

# =============================================================================
# .env loading  (no dependency on python-dotenv - it may not be installed)
# =============================================================================

REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = REPO_ROOT / ".env"


def _load_env(path: Path = ENV_PATH) -> None:
    """Minimal .env parser. Does not overwrite variables already in the real
    environment, so an OS-level env var always wins over the file."""
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


_load_env()


def _env(name: str, default: str | None = None, *, required: bool = False) -> str:
    value = os.environ.get(name, default)
    if required and not value:
        raise RuntimeError(
            f"{name} is not set. Add it to {ENV_PATH} (see .env.example) "
            f"or export it in your shell."
        )
    return value or ""


# =============================================================================
# DESTINATION
# =============================================================================

# Everything is written under here. Point this at your external SSD when you run
# the real extraction at the office, so bytes land on the portable drive
# incrementally rather than being copied at the end - if the laptop dies at 90%
# you want 90% already portable.
# Default is OUTSIDE the repository, and cross-platform. Two reasons it must not
# default inside the repo: the data is licensed and must never be committable,
# and a 1.4 GB directory in a git working tree is one un-anchored ignore rule
# away from disaster -- which has already happened once here.
DESTINATION_ROOT = Path(
    _env("FACTSET_DEST_ROOT", str(Path.home() / "statarb-data"))
).expanduser()

RAW_DIR      = DESTINATION_ROOT / "raw"
MANIFEST_DIR = DESTINATION_ROOT / "_manifest"


# =============================================================================
# SQL SERVER CONNECTION
# =============================================================================

# NO DEFAULTS for host / database / user, on purpose.
#
# A code default for the host is worse than a leak: it puts a production hostname
# into every clone of the repo, AND any machine running without a .env silently
# connects to it. Empty-by-default plus a loud check at connection time is
# strictly better behaviour and keeps the string out of the source entirely.
#
# Note these are NOT validated at import. `required=True` here would make merely
# importing this module fail without a .env - the same import-time-crash bug that
# makes the legacy `config/settings.py` unimportable. Validation belongs at the
# point of use: see require_connection_config().
#
# The ODBC driver name keeps its default: not sensitive, and identical on every
# Windows box with the Microsoft driver installed.
SERVER_NAME = _env("FACTSET_SQL_SERVER", "")
DATABASE    = _env("FACTSET_SQL_DATABASE", "")
USER_NAME   = _env("FACTSET_SQL_USER", "")
ODBC_DRIVER = _env("FACTSET_ODBC_DRIVER", "ODBC Driver 18 for SQL Server")

_REQUIRED_FOR_CONNECTION = {
    "FACTSET_SQL_SERVER": SERVER_NAME,
    "FACTSET_SQL_DATABASE": DATABASE,
    "FACTSET_SQL_USER": USER_NAME,
}


def require_connection_config() -> None:
    """Fail loudly if connection settings are missing. Call before connecting.

    Deliberately separate from import so that this module - and therefore the
    whole package - imports cleanly on a machine with no FactSet access at all.
    That is what lets CI import and test the read path without credentials.
    """
    missing = sorted(k for k, v in _REQUIRED_FOR_CONNECTION.items() if not v)
    if missing:
        raise RuntimeError(
            "Missing required FactSet connection settings: "
            + ", ".join(missing)
            + f".\nSet them in {ENV_PATH} (see .env.example) or export them."
        )


def get_password() -> str:
    """Password from the environment, else prompt. Never echoed, never logged."""
    pwd = os.environ.get("FACTSET_SQL_PASSWORD")
    if pwd:
        return pwd
    import getpass
    return getpass.getpass(f"Password for {USER_NAME}@{SERVER_NAME}: ")


def connection_string(password: str | None = None) -> str:
    """ODBC connection string.

    Encrypt=no / TrustServerCertificate=yes mirrors the connection settings
    that are known to work against this host from SSMS.
    """
    require_connection_config()
    pwd = password if password is not None else get_password()
    return (
        f"DRIVER={{{ODBC_DRIVER}}};"
        f"SERVER={SERVER_NAME};"
        f"DATABASE={DATABASE};"
        f"UID={USER_NAME};PWD={pwd};"
        f"Encrypt=no;TrustServerCertificate=yes;"
    )


def safe_connection_summary() -> str:
    """Connection description with NO password, for logs and dry-run output."""
    return (
        f"driver={ODBC_DRIVER} server={SERVER_NAME} "
        f"database={DATABASE} user={USER_NAME or '<unset>'}"
    )


# =============================================================================
# EXTRACTION WINDOW
# =============================================================================

# Discovery Q4b: fp_basic_prices actually spans 1984-11-05 .. 2026-08-03.
# So more history is available than the 1995 start we chose - that remains a
# deliberate choice (pre-2001 US markets quoted in sixteenths, so bid-ask bounce
# is structurally larger and inflates apparent mean reversion), not a limit.
START_DATE = _env("FACTSET_START_DATE", "1995-01-01")
END_DATE   = _env("FACTSET_END_DATE",   "2050-12-31")  # open-ended; the feed's
                                                       # own max date is the
                                                       # real upper bound

FEED_MIN_DATE = "1984-11-05"   # observed, discovery Q4b
FEED_MAX_DATE = "2026-08-03"   # observed, discovery Q4b


def shard_max_year() -> int:
    """Upper year bound for building the shard list.

    END_DATE is intentionally far in the future so the WHERE clause never
    truncates real data. But sharding by year off that would generate dozens of
    empty future-year shards, each costing a round trip, a query compilation and
    a manifest entry for zero rows.

    Prefer the feed's actual max date once discovery query D1/D8 has told us what
    it is - set FACTSET_FEED_MAX_YEAR in .env. Until then, fall back to the
    current calendar year.
    """
    override = os.environ.get("FACTSET_FEED_MAX_YEAR")
    if override:
        return int(override)
    from datetime import date
    return date.today().year


# =============================================================================
# UNIVERSE DEFINITION
# =============================================================================

# US-listed common stock only. NO ADR/GDR/DR: they carry FX exposure and
# home-market-hours overnight gaps, both of which contaminate a cointegration
# spread in ways that look like tradeable mean reversion but are not.
SECURITY_TYPES: tuple[str, ...] = ("SHARE",)

# RESOLVED by discovery Q17 (actual fref_listing_exchange distribution for
# USD + SHARE), which corrected my guesses: "ARC", "BAT" and "IEX" are not real
# codes. Cboe/IEX/MEMX are TRADING venues and never appear as LISTING exchanges.
#
# Exchange-listed US common equity (p_sec_type_code = '10'):
#     NAS  NASDAQ ............... 9,917
#     NYS  New York SE .......... 5,286
#     ASE  NYSE American ........ 1,138
#     PSE  NYSE Arca ............    26
#                        total .. 16,367
US_EXCHANGE_CODES: tuple[str, ...] = ("NYS", "NAS", "ASE", "PSE")

# OTC is deliberately SEPARATE. 'OTC' (US OTC) alone adds 28,712 more securities
# with p_sec_type_code = '10' - it would nearly triple the universe with pink
# sheets, shells and names that cannot realistically be shorted or borrowed. For
# a pairs strategy that is mostly noise plus untradeable signal.
# Set to ("OTC", "PINX") to include, and pair it with a hard liquidity screen.
OTC_EXCHANGE_CODES: tuple[str, ...] = ()

# RESOLVED by discovery Q6b. '10' = "Equity" in ref_v2.fp_sec_type_map.
# Deliberately EXCLUDES the other values that also carry fref_security_type
# 'SHARE': '6C' open-ended fund (87), '54' composite unit (73). Both would
# otherwise slip into a "common stock" universe.
SEC_TYPE_CODES: tuple[str, ...] | None = ("10",)

# WARNING (discovery Q17): currency = 'USD' does NOT imply US-listed. USD-quoted
# SHARE securities appear on LON (1,204), Moscow RUS/MIC (1,231), LIM, SWX, SGO,
# BUE, SHG, GUA, KAZ, PAE, TSE, LUX, AMS, BRN. The exchange filter is therefore
# load-bearing, not belt-and-braces - currency alone would silently import
# foreign listings.
# Also excluded: ZZUS "United States Unlisted Funds" (697), which is not equity.

# Sanity bounds for the universe size. If the resolved count falls outside this,
# a filter is wrong - fail loudly rather than extract a bad universe, because
# office access is one-shot and a bad universe invalidates every partition.
#
# Calibrated against MEASURED counts (discovery Q17), not estimates:
#   exchange-listed, p_sec_type_code='10':  NAS 9,917 + NYS 5,286
#                                         + ASE 1,138 + PSE 26  = 16,367
#   adding OTC:                            + 28,712            = 45,079
# Observed on 2026-08-04: exactly 16,367.

def expected_universe_band() -> tuple[int, int]:
    """Sanity band, widened when OTC is included."""
    if OTC_EXCHANGE_CODES:
        return 35_000, 70_000
    return 12_000, 25_000


EXPECTED_UNIVERSE_MIN, EXPECTED_UNIVERSE_MAX = expected_universe_band()


# =============================================================================
# EXTRACTION BEHAVIOUR
# =============================================================================

DRY_RUN = True     # True = print every SQL statement, open no connection

N_WORKERS = 3      # concurrent ODBC connections. 3, not 16: this is a shared
                   # production box and the useful gain here is TCP stream
                   # parallelism over a WAN, not server CPU.

MAXDOP = 1         # per-query server parallelism cap. Without it, 3 connections
                   # x server MAXDOP 8 = 24 workers seized on someone else's
                   # server. Raise only if sys.dm_db_index_usage_stats shows the
                   # box is genuinely idle.

SHARD_MODE = "fsym_range"   # RESOLVED by discovery Q3.
                      # fp_basic_prices has a CLUSTERED PRIMARY KEY on
                      # (fsym_id, p_date) - the ideal case. Sharding on fsym_id
                      # ranges is a clustered range SEEK, not a scan, AND rows
                      # arrive already sorted in exactly our target order, so
                      # the step-2 sid-major sort is free. There are also
                      # nonclustered indexes on fsym_id and on p_date, so
                      # "year" would work too - but fsym_range is strictly
                      # better because of the free sort.
                      #
                      # Legacy options: "year" | "single_pass"
                      # DECIDE FROM discovery query D3 (index layout):
                      #   clustered index leads with p_date  -> "year"
                      #   clustered index leads with fsym_id -> "fsym_range"
                      #   heap / no useful index             -> "single_pass"
                      # N shards on an UNINDEXED column means N FULL SCANS of a
                      # ~120 GB table, which is slower than one pass and
                      # cache-thrashes the buffer pool for every other user.

FSYM_RANGE_BUCKETS = 40    # only used when SHARD_MODE == "fsym_range"
FSYM_RANGE_STRIDE  = 1_000

# ---- arrow-odbc tuning -------------------------------------------------------
BATCH_SIZE      = 65_535         # rows per Arrow batch (== max int16; avoids
                                 # some driver edge cases)
MAX_BYTES_BATCH = 64 * 1024**2   # arrow-odbc DEFAULTS TO 512 MB, and with
                                 # fetch_concurrently that is ~1 GB of buffers.
                                 # On an 8 GB machine this is the one default
                                 # that will bite you.
MAX_TEXT_SIZE   = 256            # cap unbounded varchar/nvarchar(max) columns:
                                 # the transit buffer is sized from the DECLARED
                                 # width, so an nvarchar(max) name column would
                                 # try to allocate absurdly.
PACKET_SIZE     = 32_767         # TDS maximum. Benchmark against 4096 on one
                                 # shard: possibly a free 5-15%.
FETCH_CONCURRENTLY = True        # background fetch thread, overlaps socket read
                                 # with Parquet encoding

# ---- Parquet (raw landing zone only) ----------------------------------------
# Raw is transient and rewritten by step 2, so favour write speed over size.
# The ZSTD-9 + sorting + row-group tuning belongs in step 2, at home, where it
# is repeatable and the server is not in the loop.
ZSTD_LEVEL_RAW  = 3
ROW_GROUP_RAW   = 256_000

# ---- Retry -------------------------------------------------------------------
MAX_RETRIES   = 5
RETRY_BACKOFF = 2.0    # seconds, exponential
QUERY_TIMEOUT = 3_600  # seconds per shard
LOGIN_TIMEOUT = 30
