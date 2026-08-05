"""
Cumulative corporate-action adjustment factors.

PURE FUNCTIONS ONLY - no file I/O, no config, no database. That is what makes the
hand-checked split tests runnable without a database, and it is why this is the
first module written: a wrong split direction produces a price series that looks
entirely plausible, so it must be tested against independently known reality
before anything is built on top of it.

-------------------------------------------------------------------------------
THE ARITHMETIC
-------------------------------------------------------------------------------
FactSet ships PER-EVENT factors. We need a CUMULATIVE factor per (security, day).

    cum_factor(t) = product of every event factor whose effective_date > t

Then:

    adjusted_price(t) = raw_price(t) x cum_factor(t)

Note the STRICT inequality. On the effective date the vendor's raw price is
already expressed in post-event terms: AAPL's 2020-08-31 p_price is 129.04, not
500-ish. So the effective date itself needs no adjustment, and its factor must
NOT be included. Using >= instead of > would double-apply every split - a
plausible-looking off-by-one that halves or quadruples one day's return.

CONVENTION (verified empirically against five real AAPL splits, discovery Q9):
`p_split_factor` is the RECIPROCAL of the split ratio - a 4-for-1 is stored as
0.25, a 7-for-1 as 0.142857. So historical prices are MULTIPLIED by it, which
scales pre-split prices DOWN into post-split terms.

TWO SERIES:
  * price return  <- adj_factor_combined      (splits/spinoffs only; NULL on
                                               dividend-only events, meaning
                                               "no price adjustment" = 1.0)
  * total return  <- div_spl_spin_adj_factor  (every event, including dividends,
                                               where it is ~= 1 - div/price)
"""

from __future__ import annotations

from datetime import timedelta

import polars as pl

ONE_DAY = timedelta(days=1)

# A factor must be finite and strictly positive. Zero or negative would mean a
# corrupt event record, and propagating it silently would zero out or sign-flip a
# security's entire history.
MIN_VALID_FACTOR = 1e-12
MAX_VALID_FACTOR = 1e6


class FactorError(ValueError):
    """Raised when event data cannot produce a trustworthy cumulative factor."""


def validate_events(events: pl.DataFrame, factor_col: str) -> None:
    """Fail loudly on event data that cannot yield a sane cumulative factor.

    Deliberately raises rather than filtering: a bad factor is evidence of a data
    problem worth understanding, not noise to be dropped.
    """
    for required in ("sid", "effective_date", factor_col):
        if required not in events.columns:
            raise FactorError(f"events missing required column {required!r}")

    bad = events.filter(
        pl.col(factor_col).is_not_null()
        & (
            ~pl.col(factor_col).is_finite()
            | (pl.col(factor_col) <= MIN_VALID_FACTOR)
            | (pl.col(factor_col) > MAX_VALID_FACTOR)
        )
    )
    if bad.height:
        raise FactorError(
            f"{bad.height} event(s) have an out-of-range {factor_col}: "
            f"{bad.head(5).to_dicts()}"
        )


def cumulative_factors(events: pl.DataFrame, factor_col: str) -> pl.DataFrame:
    """Per-event factors -> a step function of cumulative factors.

    Returns one row per event with:
        sid, effective_date, cum_factor

    where cum_factor is the product of THIS event's factor and every LATER
    event's factor for the same security. That value is what applies to any date
    strictly BEFORE this effective_date.

    NULL factors are treated as 1.0 (no adjustment). That is the correct reading
    of a NULL `adj_factor_combined` on a dividend-only event: the dividend does
    not change the price series.
    """
    validate_events(events, factor_col)

    if events.height == 0:
        return pl.DataFrame(
            schema={"sid": pl.Int32, "effective_date": pl.Date,
                    "cum_factor": pl.Float64}
        )

    ev = (
        events.select(
            pl.col("sid").cast(pl.Int32),
            pl.col("effective_date").cast(pl.Date),
            pl.col(factor_col).cast(pl.Float64).fill_null(1.0).alias("f"),
        )
        # Multiple events can share an effective_date (e.g. a split and a
        # dividend on the same day). They must COMPOUND, so collapse to a product
        # per (sid, date) first - otherwise the reverse scan below would treat
        # them as alternatives rather than as both applying.
        .group_by("sid", "effective_date")
        .agg(pl.col("f").product().alias("f"))
        .sort("sid", "effective_date", descending=[False, True])
    )

    # Reverse cumulative product within each security: because rows are sorted
    # date-DESCENDING, a forward cum_prod accumulates from the latest event
    # backwards, which is exactly "this event and all later ones".
    return (
        ev.with_columns(
            pl.col("f").cum_prod().over("sid").alias("cum_factor")
        )
        .drop("f")
        .sort("sid", "effective_date")
    )


def attach_cumulative_factor(
    prices: pl.DataFrame,
    events: pl.DataFrame,
    factor_col: str,
    out_col: str,
    date_col: str = "d",
) -> pl.DataFrame:
    """Attach the cumulative factor applying to each (sid, date) row.

    For price date t we want the product of all factors with effective_date > t.
    Since `cumulative_factors` gives, at each event date e, the product of e and
    everything after it, the answer is the cum_factor of the FIRST event with
    e > t.

    Implemented as a forward as-of join on t + 1 day: 'first event on or after
    t+1' is precisely 'first event strictly after t', which keeps the strict
    inequality of the docstring above without relying on join semantics that
    differ between engines.

    Securities with no events, and dates after a security's final event, get
    1.0 - correct, because nothing later remains to adjust for.
    """
    cum = cumulative_factors(events, factor_col)

    if cum.height == 0:
        return prices.with_columns(pl.lit(1.0, pl.Float64).alias(out_col))

    # Sort by the group key first, then the join key: that is the sort order a
    # grouped as-of join expects, and it lets Polars skip its own sortedness check.
    left = (
        prices.with_columns(
            (pl.col(date_col).cast(pl.Date) + ONE_DAY).alias("_probe")
        )
        .sort("sid", "_probe")
    )
    right = cum.sort("sid", "effective_date")

    joined = left.join_asof(
        right,
        left_on="_probe",
        right_on="effective_date",
        by="sid",
        strategy="forward",
    )

    return (
        joined.with_columns(
            pl.col("cum_factor").fill_null(1.0).alias(out_col)
        )
        .drop("_probe", "effective_date", "cum_factor")
        .sort("sid", date_col)
    )


def build_price_factors(
    prices: pl.DataFrame, adj_factors: pl.DataFrame, date_col: str = "d"
) -> pl.DataFrame:
    """Attach both cumulative factors to a price frame.

    cum_px_factor -> price-return series  (splits and spinoffs)
    cum_tr_factor -> total-return series  (adds dividends)

    Both stay float64: a running product over 30 years of events compounds
    float32 error, and unlike the price observations there is no source-precision
    argument for narrowing them.
    """
    out = attach_cumulative_factor(
        prices, adj_factors, "adj_factor_combined", "cum_px_factor", date_col
    )
    return attach_cumulative_factor(
        out, adj_factors, "div_spl_spin_adj_factor", "cum_tr_factor", date_col
    )


def adjusted(prices: pl.DataFrame, kind: str = "total") -> pl.DataFrame:
    """Materialise adjusted price columns on read.

    Adjusted prices are deliberately NOT stored: a newly reported split changes
    only the factor column, and every adjusted series is then instantly correct
    rather than stale-but-plausible. The cost is one multiply.
    """
    if kind == "raw":
        return prices
    col = {"price": "cum_px_factor", "total": "cum_tr_factor"}.get(kind)
    if col is None:
        raise ValueError(f"kind must be 'raw', 'price' or 'total', got {kind!r}")

    present = [c for c in ("px_open", "px_high", "px_low", "px_close")
               if c in prices.columns]
    return prices.with_columns(
        [(pl.col(c) * pl.col(col)).alias(f"{c}_adj") for c in present]
    )
