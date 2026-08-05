"""
DataAPI - the ONLY thing strategy code should import to get data.

    from statarb.data.api import DataAPI

    api = DataAPI()
    px  = api.get_prices(sids=[8269], start="2020-08-20", end="2020-09-05")
    uni = api.get_universe_as_of("2005-06-01", min_price=5.0, min_adv_usd=1e6)

Why a layer at all: the legacy code had three separate modules each
reimplementing "load a sector panel", one of which read a pickle at import time.
Strategy code that knows about file paths cannot be tested, cannot be moved
between machines, and breaks whenever storage changes. Everything here returns
Polars DataFrames and takes no paths.

-------------------------------------------------------------------------------
THE POINT-IN-TIME GUARANTEE
-------------------------------------------------------------------------------
`get_universe_as_of(date)` returns the securities that were investable ON that
date - including ones that later died. It NEVER filters to things that still
exist today, and liquidity is computed only from data at or before the date.

That single method is where look-ahead bias would otherwise re-enter after all
the work done to keep it out. The tests assert that a 2005 universe contains
securities that no longer exist.
"""

from __future__ import annotations

from datetime import date
from functools import cached_property
from pathlib import Path
from typing import Iterable, Literal

import polars as pl
from deltalake import DeltaTable

from ..paths import Locations, locations

Adjusted = Literal["raw", "price", "total"]
DateLike = str | date


def _as_date(v: DateLike) -> date:
    return v if isinstance(v, date) else date.fromisoformat(str(v))


class DataAPI:
    """Read-only access to the curated Delta tables.

    Every method is lazy where it can be: `scan_delta`-style filtering pushes
    date and sid predicates into the Parquet reader, so a two-stock query reads
    kilobytes rather than the 39.7M-row table.
    """

    def __init__(
        self,
        root: str | Path | Locations | None = None,
        version: int | None = None,
    ):
        """
        root:    where the warehouse lives. An explicit value always wins, then
                 $FACTSET_DEST_ROOT, then ~/statarb-data. Passing it explicitly is
                 what lets a test point at a temporary directory without touching
                 the environment -- no monkeypatch-before-import ordering games.

        version: pin to a Delta version for reproducibility. This is the payoff of
                 using Delta: recording the version alongside a result means that
                 result can be reproduced exactly later, even if the vendor has
                 since restated prices or reported a new split.
        """
        self.loc = root if isinstance(root, Locations) else locations(root)
        self.version = version

    def __repr__(self) -> str:
        v = "latest" if self.version is None else f"v{self.version}"
        return f"DataAPI(root={self.loc.root}, version={v})"

    # -- table access ------------------------------------------------------

    def _scan(self, path) -> pl.LazyFrame:
        """Lazy scan with predicate and projection pushdown.

        MUST be `pl.scan_delta`, not `DeltaTable(...).to_pyarrow_table()`. The
        latter materialises the whole table before any filter is applied - for
        `prices` that is 39.7M rows loaded to answer a two-stock query, which on
        an 8 GB machine swaps rather than finishes. `scan_delta` pushes the date
        and sid predicates into the Parquet reader, so the same query touches
        kilobytes.
        """
        kwargs = {} if self.version is None else {"version": self.version}
        return pl.scan_delta(str(path), **kwargs)

    @cached_property
    def securities(self) -> pl.DataFrame:
        """The security master. Small (16k rows) so it is cached in memory."""
        return self._scan(self.loc.security).collect()

    @cached_property
    def calendar(self) -> pl.DataFrame:
        return self._scan(self.loc.calendar).collect()

    def table_versions(self) -> dict[str, int]:
        """Current version of each table - record this alongside any result."""
        return {
            p.name: DeltaTable(str(p)).version()
            for p in (self.loc.prices, self.loc.security,
                      self.loc.corp_actions, self.loc.calendar)
        }

    # -- prices ------------------------------------------------------------

    def get_prices(
        self,
        sids: Iterable[int] | None = None,
        start: DateLike | None = None,
        end: DateLike | None = None,
        adjusted: Adjusted = "total",
        columns: list[str] | None = None,
    ) -> pl.DataFrame:
        """Daily prices, adjusted on read.

        adjusted="total" -> splits AND dividends  (what you actually earned)
        adjusted="price" -> splits only           (what the share price did)
        adjusted="raw"   -> untouched as-traded prices

        Adjusted prices are computed here rather than stored: a newly reported
        split changes only the factor column, so every historical adjusted price
        is instantly correct instead of stale-but-plausible.
        """
        lf = self._scan(self.loc.prices)
        if sids is not None:
            lf = lf.filter(pl.col("sid").is_in(list(sids)))
        if start is not None:
            lf = lf.filter(pl.col("d") >= _as_date(start))
        if end is not None:
            lf = lf.filter(pl.col("d") <= _as_date(end))

        if adjusted != "raw":
            factor = "cum_tr_factor" if adjusted == "total" else "cum_px_factor"
            lf = lf.with_columns([
                (pl.col(c) * pl.col(factor)).cast(pl.Float64).alias(c)
                for c in ("px_open", "px_high", "px_low", "px_close")
            ])

        out = lf.sort("sid", "d").collect()
        return out.select(columns) if columns else out

    def get_returns(
        self,
        sids: Iterable[int] | None = None,
        start: DateLike | None = None,
        end: DateLike | None = None,
        kind: Adjusted = "total",
    ) -> pl.DataFrame:
        """Daily simple returns from the adjusted close.

        Computed per security via `.over("sid")` so a return is never accidentally
        taken across a boundary between two different stocks - which is what a
        plain `.shift(1)` on a concatenated frame would silently do.
        """
        px = self.get_prices(sids, start, end, adjusted=kind,
                             columns=["sid", "d", "px_close"])
        return px.with_columns(
            (pl.col("px_close") / pl.col("px_close").shift(1).over("sid") - 1)
            .alias("ret")
        ).drop_nulls("ret")

    def get_panel(
        self,
        sids: Iterable[int],
        start: DateLike,
        end: DateLike,
        value: str = "px_close",
        kind: Adjusted = "total",
    ) -> pl.DataFrame:
        """Wide panel: one row per date, one column per security.

        This is the shape PCA and clustering want. Kept as an explicit method so
        the pivot happens in one place, rather than three modules each doing it
        slightly differently as in the legacy code.
        """
        long = self.get_prices(sids, start, end, adjusted=kind,
                               columns=["sid", "d", value])
        return long.pivot(on="sid", index="d", values=value).sort("d")

    # -- universe ----------------------------------------------------------

    def get_universe_as_of(
        self,
        as_of: DateLike,
        min_price: float | None = None,
        min_adv_usd: float | None = None,
        adv_window: int = 63,
        primary_class_only: bool = False,
        exchanges: Iterable[str] | None = None,
    ) -> pl.DataFrame:
        """Securities investable ON `as_of` - including ones that later died.

        POINT-IN-TIME. Two rules make it so, and both matter:

          1. Membership is `first_px_date <= as_of <= last_px_date`. It does NOT
             filter on is_dead or active_flag: a stock that was trading in 2005
             belongs in a 2005 universe regardless of what happened later.
             Filtering on "still alive today" is exactly the survivorship bias
             that deletes 68.6% of this universe.

          2. Liquidity screens use ONLY data at or before `as_of` - a trailing
             window ending on the date, never centred or forward-looking.
        """
        d = _as_date(as_of)

        sec = self.securities.filter(
            (pl.col("first_px_date") <= d) & (pl.col("last_px_date") >= d)
        )
        if primary_class_only:
            sec = sec.filter(pl.col("is_primary_class"))
        if exchanges is not None:
            sec = sec.filter(pl.col("exchange_code").is_in(list(exchanges)))

        if min_price is None and min_adv_usd is None:
            return sec

        # Trailing window, strictly ending at as_of.
        window_start = self.trading_days_before(d, adv_window)
        px = self.get_prices(
            sids=sec["sid"].to_list(), start=window_start, end=d,
            adjusted="raw", columns=["sid", "d", "px_close", "volume"],
        )
        stats = px.group_by("sid").agg(
            pl.col("px_close").last().alias("last_price"),
            (pl.col("px_close") * pl.col("volume")).mean().alias("adv_usd"),
            pl.len().alias("n_days"),
        )
        if min_price is not None:
            stats = stats.filter(pl.col("last_price") >= min_price)
        if min_adv_usd is not None:
            stats = stats.filter(pl.col("adv_usd") >= min_adv_usd)

        return sec.join(stats.select("sid", "last_price", "adv_usd"),
                        on="sid", how="inner")

    # -- calendar ----------------------------------------------------------

    def get_trading_days(self, start: DateLike, end: DateLike) -> list[date]:
        """Trading days from FactSet's exchange calendar.

        NOT derived from any stock's observed dates. The legacy code built its
        calendar from AAPL's price index, which also encodes AAPL's own trading
        halts as though they were market holidays.
        """
        return (
            self.calendar.filter(
                pl.col("is_trading_day")
                & (pl.col("d") >= _as_date(start))
                & (pl.col("d") <= _as_date(end))
            )
            .sort("d")["d"].to_list()
        )

    def trading_days_before(self, as_of: DateLike, n: int) -> date:
        """The date n trading days before as_of. Clamps at the calendar start
        rather than wrapping - the legacy `time_travel` helper silently used
        negative indexing, which returned a date from the FUTURE for large
        lookbacks."""
        d = _as_date(as_of)
        past = self.calendar.filter(
            pl.col("is_trading_day") & (pl.col("d") <= d)
        ).sort("d")["d"].to_list()
        if not past:
            raise ValueError(f"no trading days on or before {d}")
        return past[max(0, len(past) - 1 - n)]

    def get_rebalance_dates(
        self, start: DateLike, end: DateLike, freq: str = "M"
    ) -> list[date]:
        """Rebalance dates - month-end by default, straight from the vendor's
        own eom_flag rather than inferred from a date range."""
        cal = self.calendar.filter(
            pl.col("is_trading_day")
            & (pl.col("d") >= _as_date(start))
            & (pl.col("d") <= _as_date(end))
        ).sort("d")
        if freq.upper().startswith("M"):
            return (
                cal.group_by(pl.col("d").dt.truncate("1mo").alias("m"))
                .agg(pl.col("d").max().alias("d"))
                .sort("d")["d"].to_list()
            )
        raise ValueError(f"unsupported freq {freq!r}")

    # -- corporate actions -------------------------------------------------

    def get_corporate_actions(
        self,
        sids: Iterable[int] | None = None,
        kinds: Iterable[str] | None = None,
    ) -> pl.DataFrame:
        """Splits / dividends / adjustment events. Useful for explaining an
        extreme return rather than shrugging at it."""
        lf = self._scan(self.loc.corp_actions)
        if sids is not None:
            lf = lf.filter(pl.col("sid").is_in(list(sids)))
        if kinds is not None:
            lf = lf.filter(pl.col("event_kind").is_in(list(kinds)))
        return lf.sort("sid", "event_date").collect()

    # -- convenience -------------------------------------------------------

    def sid_for_ticker(self, ticker: str) -> int:
        """'AAPL-US' or 'AAPL' -> sid. Raises if ambiguous rather than guessing:
        tickers get recycled between unrelated companies, so a silent pick would
        be a genuine correctness bug."""
        t = ticker if "-" in ticker else f"{ticker}-US"
        hit = self.securities.filter(pl.col("ticker_region") == t)
        if hit.height == 0:
            raise KeyError(f"no security with ticker_region {t!r}")
        if hit.height > 1:
            raise KeyError(
                f"{t!r} matches {hit.height} securities "
                f"(sids {hit['sid'].to_list()}) - tickers are reused over time; "
                "pass the sid explicitly")
        return int(hit["sid"][0])

    def summary(self) -> str:
        sec = self.securities
        return "\n".join([
            f"securities        : {sec.height:,}",
            f"  with prices     : {sec.filter(pl.col('n_obs') > 0).height:,}",
            f"  dead            : {sec.filter(pl.col('is_dead')).height:,} "
            f"({sec.filter(pl.col('is_dead')).height / max(sec.height,1):.1%})",
            f"  primary class   : {sec.filter(pl.col('is_primary_class')).height:,}",
            f"price rows        : {DeltaTable(str(self.loc.prices)).count():,}",
            f"table versions    : {self.table_versions()}",
        ])
