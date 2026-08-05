"""
Build the curated Delta tables from the raw landing zone.

    python -m statarb.curate.build

Raw -> curated:
  * rename the wire-thrift column names (c, v) to real ones
  * narrow dtypes (float64 -> float32 prices; volume -> whole shares)
  * attach the cumulative adjustment factors from curated/factors.py
  * write Delta tables with schema enforcement, long retention and versioning

Processed SHARD BY SHARD. Each raw shard covers a disjoint sid range of ~950k
rows, so peak memory stays near 100 MB rather than materialising 39.7M rows -
which matters on an 8 GB machine.
"""

from __future__ import annotations

import shutil
import sys
import time
from datetime import date, datetime

import polars as pl
import pyarrow as pa
from deltalake import DeltaTable, write_deltalake

from . import factors as F
from . import schema as S


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


# =============================================================================
# PRICES
# =============================================================================

def build_prices(overwrite: bool = True) -> int:
    """Raw price shards -> curated/prices Delta table."""
    shards = sorted((S.RAW_DIR / "prices").glob("*.parquet"))
    if not shards:
        raise SystemExit(f"no raw price shards under {S.RAW_DIR / 'prices'}")

    adj = pl.read_parquet(S.RAW_DIR / "adjustment_factors.parquet")
    log(f"loaded {adj.height:,} adjustment events")

    if overwrite and S.T_PRICES.exists():
        shutil.rmtree(S.T_PRICES)

    total = 0
    t0 = time.time()
    for i, shard in enumerate(shards, 1):
        px = (
            pl.read_parquet(shard)
            .select(
                pl.col("sid").cast(pl.Int32),
                pl.col("d").cast(pl.Date),
                # Short raw names come from the extraction SQL, where every byte
                # crossed an uncompressed wire. Real names start here.
                pl.col("px_open").cast(pl.Float64),
                pl.col("px_high").cast(pl.Float64),
                pl.col("px_low").cast(pl.Float64),
                pl.col("c").cast(pl.Float64).alias("px_close"),
                # v is truncated thousands (see schema.VOLUME_NOTE) - x1000 gives
                # whole shares, accurate to ~8e-6.
                (pl.col("v").cast(pl.Int64) * 1000).alias("volume"),
            )
            .filter(pl.col("px_close").is_not_null())
        )
        if px.height == 0:
            continue

        # Only the events for this shard's securities: keeps the as-of join small.
        sids = px["sid"].unique().to_list()
        ev = adj.filter(pl.col("sid").is_in(sids))

        out = (
            F.build_price_factors(px, ev)
            .with_columns(pl.col("d").dt.year().cast(pl.Int32).alias("year"))
            .select(
                "sid", "d", "year",
                pl.col("px_open").cast(pl.Float32),
                pl.col("px_high").cast(pl.Float32),
                pl.col("px_low").cast(pl.Float32),
                pl.col("px_close").cast(pl.Float32),
                "volume",
                pl.col("cum_px_factor").cast(pl.Float64),
                pl.col("cum_tr_factor").cast(pl.Float64),
            )
            .sort("sid", "d")
        )

        tbl = out.to_arrow().cast(S.PRICES)
        write_deltalake(
            str(S.T_PRICES), tbl,
            mode="overwrite" if i == 1 else "append",
            partition_by=S.PRICES_PARTITION_BY,
            configuration=S.DELTA_PROPERTIES if i == 1 else None,
        )
        total += out.height
        if i % 5 == 0 or i == len(shards):
            el = time.time() - t0
            log(f"  prices {i}/{len(shards)} shards, {total:,} rows, "
                f"{total / max(el, 1):,.0f} rows/s")

    # 41 appends x 32 year partitions leaves ~1,300 small files. Each one costs a
    # footer read and an open on every scan, and small files compress worse
    # because each restarts its own dictionary. Compaction rewrites them into a
    # few large files per partition.
    #
    # Z-ORDER by sid so that rows for one security cluster together within a
    # file, which makes the row-group statistics useful for single-stock queries.
    dt = DeltaTable(str(S.T_PRICES))
    before = len(dt.file_uris())
    log(f"  compacting {before} files ...")
    dt.optimize.z_order(["sid"], target_size=128 * 1024 * 1024)
    dt = DeltaTable(str(S.T_PRICES))
    log(f"  compacted {before} -> {len(dt.file_uris())} files")
    return total


# =============================================================================
# SECURITY DIMENSION
# =============================================================================

def build_security() -> int:
    """dim_security + life spans + sector snapshot + share-class facts."""
    dim = pl.read_parquet(S.RAW_DIR / "dim_security.parquet")
    spans = pl.read_parquet(S.RAW_DIR / "security_life_spans.parquet").select(
        "sid", "first_px_date", "last_px_date", "n_obs", "is_dead")
    sect = pl.read_parquet(S.RAW_DIR / "sector_snapshot.parquet")

    # Share-class facts. Two securities are two classes of one company iff they
    # share factset_entity_id; the primary class is the one whose id equals
    # fsym_primary_equity_id. That is what makes GOOGL/GOOG answerable.
    # NOTE the id NAMESPACES - this is easy to get silently wrong.
    #   fsym_regional_id        -R  (the price key)
    #   fsym_security_id        -S
    #   fsym_primary_equity_id  -S  <-- a SECURITY id, so it must be compared
    #                                   against fsym_security_id. Comparing it to
    #                                   the -R id matches nothing and quietly
    #                                   reports every security as non-primary.
    #   fsym_primary_listing_id -L  (different namespace again; attribute only)
    dim = dim.with_columns(
        (pl.col("fsym_security_id") == pl.col("fsym_primary_equity_id"))
        .alias("is_primary_class"),
        # Guard the NULL entity id. `.over(<null>)` treats every NULL as ONE
        # group, so ~317 unrelated securities with no entity link would all
        # report n_classes_in_entity = 317 - a nonsense number that looks like a
        # finding. A security with no entity link has no known siblings, so 1.
        pl.when(pl.col("factset_entity_id").is_null())
        .then(pl.lit(1))
        .otherwise(pl.col("fsym_regional_id").count().over("factset_entity_id"))
        .cast(pl.Int32).alias("n_classes_in_entity"),
    )

    out = (
        dim.join(spans, on="sid", how="left")
        .join(sect, on="factset_entity_id", how="left")
        .select(
            pl.col("sid").cast(pl.Int32),
            "fsym_regional_id", "fsym_security_id", "factset_entity_id",
            "fsym_primary_equity_id", "ticker_region", "security_name",
            "entity_name", "isin", "sedol", "exchange_code", "sec_type_code",
            "is_primary_class", "n_classes_in_entity",
            pl.col("sector_code").alias("sector_code_snapshot"),
            pl.col("sector_desc").alias("sector_desc_snapshot"),
            pl.col("industry_code").alias("industry_code_snapshot"),
            pl.col("industry_desc").alias("industry_desc_snapshot"),
            "first_px_date", "last_px_date",
            pl.col("n_obs").fill_null(0).cast(pl.Int64),
            pl.col("is_dead").fill_null(True),
            pl.col("active_flag").cast(pl.Int16),
        )
        .sort("sid")
    )

    if S.T_SECURITY.exists():
        shutil.rmtree(S.T_SECURITY)
    write_deltalake(str(S.T_SECURITY), out.to_arrow().cast(S.SECURITY),
                    mode="overwrite",
                    configuration=S.DELTA_PROPERTIES)
    return out.height


# =============================================================================
# CORPORATE ACTIONS  (unified)
# =============================================================================

def build_corp_actions() -> int:
    """Splits, dividends and adjustment events in one queryable table."""
    splits = pl.read_parquet(S.RAW_DIR / "splits.parquet").select(
        pl.col("sid").cast(pl.Int32),
        pl.col("split_date").alias("event_date"),
        pl.lit("SPLIT").alias("event_kind"),
        pl.lit(None, pl.String).alias("event_type"),
        pl.col("split_factor"),
        pl.lit(None, pl.Float64).alias("div_amount"),
        pl.lit(None, pl.Float64).alias("px_factor"),
        pl.lit(None, pl.Float64).alias("tr_factor"),
        pl.lit(False).alias("is_spinoff"),
        pl.lit(False).alias("is_special"),
    )

    divs = pl.read_parquet(S.RAW_DIR / "dividends.parquet").select(
        pl.col("sid").cast(pl.Int32),
        pl.col("ex_date").alias("event_date"),   # ex-date: the right date for
                                                 # total-return maths
        pl.lit("DIV").alias("event_kind"),
        pl.col("div_type_code").alias("event_type"),
        pl.lit(None, pl.Float64).alias("split_factor"),
        pl.col("div_amount"),
        pl.lit(None, pl.Float64).alias("px_factor"),
        pl.lit(None, pl.Float64).alias("tr_factor"),
        (pl.col("is_spinoff") == 1).alias("is_spinoff"),
        (pl.col("is_special") == 1).alias("is_special"),
    )

    adj = pl.read_parquet(S.RAW_DIR / "adjustment_factors.parquet").select(
        pl.col("sid").cast(pl.Int32),
        pl.col("effective_date").alias("event_date"),
        pl.lit("FACTOR").alias("event_kind"),
        pl.lit(None, pl.String).alias("event_type"),
        pl.lit(None, pl.Float64).alias("split_factor"),
        pl.lit(None, pl.Float64).alias("div_amount"),
        pl.col("adj_factor_combined").alias("px_factor"),
        pl.col("div_spl_spin_adj_factor").alias("tr_factor"),
        pl.lit(False).alias("is_spinoff"),
        pl.lit(False).alias("is_special"),
    )

    out = pl.concat([splits, divs, adj]).sort("sid", "event_date")
    if S.T_CORP_ACTIONS.exists():
        shutil.rmtree(S.T_CORP_ACTIONS)
    write_deltalake(str(S.T_CORP_ACTIONS),
                    out.to_arrow().cast(S.CORP_ACTIONS),
                    mode="overwrite",
                    configuration=S.DELTA_PROPERTIES)
    return out.height


# =============================================================================
# CALENDAR
# =============================================================================

def build_calendar() -> int:
    """FactSet's calendar, marked with which days US equity markets actually
    traded. A day is a trading day if it is a weekday and not a holiday for the
    exchanges in our universe."""
    cal = pl.read_parquet(S.RAW_DIR / "trading_calendar.parquet")
    hol = pl.read_parquet(S.RAW_DIR / "exchange_holidays.parquet")
    holidays = set(hol["holiday_date"].to_list())

    out = (
        cal.select(
            pl.col("ref_date").alias("d"),
            pl.col("day_of_week").cast(pl.Int32),
            (pl.col("eom_flag") == 1).alias("is_eom"),
        )
        .with_columns(
            (
                pl.col("d").dt.weekday().is_between(1, 5)
                & ~pl.col("d").is_in(list(holidays))
            ).alias("is_trading_day")
        )
        .sort("d")
    )
    if S.T_CALENDAR.exists():
        shutil.rmtree(S.T_CALENDAR)
    write_deltalake(str(S.T_CALENDAR), out.to_arrow().cast(S.CALENDAR),
                    mode="overwrite",
                    configuration=S.DELTA_PROPERTIES)
    return out.height


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:
    t0 = time.time()
    log("=" * 70)
    log(f"BUILD curated Delta tables -> {S.CURATED_DIR}")
    log("=" * 70)

    S.CURATED_DIR.mkdir(parents=True, exist_ok=True)

    n = build_security()
    log(f"dim_security      : {n:,} rows")
    n = build_corp_actions()
    log(f"corporate_actions : {n:,} rows")
    n = build_calendar()
    log(f"calendar          : {n:,} rows")
    n = build_prices()
    log(f"prices            : {n:,} rows")

    log("=" * 70)
    for t in (S.T_PRICES, S.T_SECURITY, S.T_CORP_ACTIONS, S.T_CALENDAR):
        dt = DeltaTable(str(t))
        size = sum(f.stat().st_size for f in t.rglob("*.parquet"))
        log(f"{t.name:<18} version={dt.version():>3}  "
            f"files={len(dt.file_uris()):>4}  rows={dt.count():>12,}  "
            f"{size / 1e6:>7,.0f} MB")
    log(f"DONE in {(time.time() - t0) / 60:.1f} min")
    log("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
