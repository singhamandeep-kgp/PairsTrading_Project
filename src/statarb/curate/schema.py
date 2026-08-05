"""
Explicit schemas for the curated Delta tables - the single source of truth.

Written out rather than inferred, for two reasons:
  * Delta enforces schema on write, so a dtype drifting in a future update is
    rejected at the door instead of silently coercing.
  * Inferred schemas depend on whatever happened to be in the first batch. That
    is exactly the kind of thing that works for a year and then doesn't.
"""

from __future__ import annotations

import os
from pathlib import Path

import pyarrow as pa

from ..paths import locations

# =============================================================================
# LOCATIONS
# =============================================================================

# Paths come from statarb.paths, which is the single definition.
#
# The module-level names below are a COMPATIBILITY SHIM so existing callers keep
# working. Prefer `locations(root=...)` in new code: it takes an explicit root and
# so depends on neither import order nor the environment.
#
# These constants are resolved at import time, which is exactly the property
# paths.py exists to avoid. Treat them as deprecated, not as the pattern to copy.
_DEFAULT = locations()

DATA_ROOT = _DEFAULT.root
RAW_DIR = _DEFAULT.raw
CURATED_DIR = _DEFAULT.curated

T_PRICES = _DEFAULT.prices
T_SECURITY = _DEFAULT.security
T_CORP_ACTIONS = _DEFAULT.corp_actions
T_CALENDAR = _DEFAULT.calendar

# =============================================================================
# RETENTION  --  read the warning
# =============================================================================

# Delta's DEFAULTS WOULD SILENTLY EXPIRE YOUR HISTORY: VACUUM keeps deleted files
# for only 7 days and the transaction log for 30. That means the ability to
# reproduce a three-month-old backtest would just stop working - which defeats the
# entire reason for using Delta in a research database.
#
# Set to 10 years. Disk is the cheap resource here; a provable result is not.
DELTA_PROPERTIES = {
    "delta.logRetentionDuration": "interval 3650 days",
    "delta.deletedFileRetentionDuration": "interval 3650 days",
}

# =============================================================================
# SCHEMAS
# =============================================================================

# One row per security per trading day.
#
# Prices are float32: the source is quoted to 4 decimals and float32 carries ~7
# significant digits, so nothing real is lost, and it halves the largest columns.
#
# The cumulative factors are float64 DELIBERATELY. They are running products over
# 30 years of events, and float32 error compounds multiplicatively - unlike the
# price observations, there is no source-precision argument for narrowing them.
PRICES = pa.schema([
    pa.field("sid", pa.int32(), nullable=False),
    pa.field("d", pa.date32(), nullable=False),
    pa.field("year", pa.int32(), nullable=False),          # partition column
    pa.field("px_open", pa.float32()),
    pa.field("px_high", pa.float32()),
    pa.field("px_low", pa.float32()),
    pa.field("px_close", pa.float32(), nullable=False),
    pa.field("volume", pa.int64()),                        # whole shares
    pa.field("cum_px_factor", pa.float64(), nullable=False),
    pa.field("cum_tr_factor", pa.float64(), nullable=False),
])

PRICES_PARTITION_BY = ["year"]

# One row per security. The identifier layer plus derived life-span and
# share-class facts.
SECURITY = pa.schema([
    pa.field("sid", pa.int32(), nullable=False),
    pa.field("fsym_regional_id", pa.string(), nullable=False),
    pa.field("fsym_security_id", pa.string()),
    pa.field("factset_entity_id", pa.string()),
    pa.field("fsym_primary_equity_id", pa.string()),
    pa.field("ticker_region", pa.string()),
    pa.field("security_name", pa.string()),
    pa.field("entity_name", pa.string()),
    pa.field("isin", pa.string()),
    pa.field("sedol", pa.string()),
    pa.field("exchange_code", pa.string()),
    pa.field("sec_type_code", pa.string()),
    # Share-class facts. is_primary_class + n_classes_in_entity are what make the
    # dual-class question answerable: two securities are two classes of one
    # company iff they share factset_entity_id.
    pa.field("is_primary_class", pa.bool_()),
    pa.field("n_classes_in_entity", pa.int32()),
    # Sector is a CURRENT SNAPSHOT, not point-in-time: FactSet's rbics_v1 tables
    # are empty in this feed. Named with the _snapshot suffix so no downstream
    # code can mistake it for history.
    pa.field("sector_code_snapshot", pa.string()),
    pa.field("sector_desc_snapshot", pa.string()),
    pa.field("industry_code_snapshot", pa.string()),
    pa.field("industry_desc_snapshot", pa.string()),
    # Life spans, derived locally from observed prices.
    pa.field("first_px_date", pa.date32()),
    pa.field("last_px_date", pa.date32()),
    pa.field("n_obs", pa.int64(), nullable=False),
    pa.field("is_dead", pa.bool_(), nullable=False),
    pa.field("active_flag", pa.int16()),
])

# Splits, dividends and other events, unified so one query answers "what happened
# to this security".
CORP_ACTIONS = pa.schema([
    pa.field("sid", pa.int32(), nullable=False),
    pa.field("event_date", pa.date32(), nullable=False),
    pa.field("event_kind", pa.string(), nullable=False),   # SPLIT | DIV | EVENT
    pa.field("event_type", pa.string()),                   # vendor code
    pa.field("split_factor", pa.float64()),
    pa.field("div_amount", pa.float64()),
    pa.field("px_factor", pa.float64()),
    pa.field("tr_factor", pa.float64()),
    pa.field("is_spinoff", pa.bool_()),
    pa.field("is_special", pa.bool_()),
])

# FactSet's own exchange calendar. Strictly better than deriving trading days from
# one stock's observed dates, which also encodes that stock's halts.
CALENDAR = pa.schema([
    pa.field("d", pa.date32(), nullable=False),
    pa.field("day_of_week", pa.int32()),
    pa.field("is_eom", pa.bool_(), nullable=False),
    pa.field("is_trading_day", pa.bool_(), nullable=False),
])


# =============================================================================
# NOTES ON KNOWN FIDELITY LIMITS
# =============================================================================

VOLUME_NOTE = """
Volume is accurate to the nearest 1,000 shares, not exact.

FactSet ships p_volume as a fractional value in THOUSANDS (AAPL 2024-01-02 =
82488.672, i.e. 82,488,672 shares). The stage-1 extraction cast it to BIGINT on
the server to save wire bytes, which truncated the fraction to 82488. So
volume = v x 1000 recovers 82,488,000 - out by 672 shares, a relative error of
~8e-6.

Irrelevant for liquidity screens. Recorded here because the alternative is
someone later assuming exactness. Fixable only by re-extracting with
CAST(... AS float), which is not worth 9 minutes of office time for 8e-6.
"""
