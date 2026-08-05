"""
Build a complete synthetic Delta warehouse, with facts planted for specific tests.

WHY THIS EXISTS
---------------
The real warehouse is 1.4 GB of licensed FactSet data that CI can never see. Without
a synthetic substitute, the only testable code is the pure arithmetic in
`curate/factors.py` — which is how this repository ended up with 4% coverage
concentrated on 4% of the source.

Two design choices make this more than a mock:

1. **It uses the REAL schemas and the REAL writer** (`statarb.curate.schema` and
   `curate.build`'s Delta calls). So a schema change breaks the fixture, which
   makes the fixture a schema contract test as well as a data source.

2. **Every planted fact serves a named test.** Tests refer to
   `planted.delisted_2008`, never a bare `9001`. If a test's premise disappears,
   the attribute disappears with it and the test fails to collect rather than
   silently passing against data that no longer has the property under test.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow as pa
from deltalake import write_deltalake

from statarb.curate import schema as S
from statarb.paths import Locations, locations

# A fixed holiday list, so `is_trading_day` and `is_eom` are exercised for real
# rather than being "every weekday".
HOLIDAYS = {
    (1, 1), (7, 4), (12, 25),          # New Year, Independence Day, Christmas
    (11, 28), (5, 27), (9, 2),         # approx Thanksgiving / Memorial / Labor
}


@dataclass(frozen=True)
class Planted:
    """The sids carrying each deliberately planted property.

    Tests reference these by name. `delisted_2008` says what the test depends on;
    `9001` says nothing and silently rots.
    """

    delisted_2008: int = 9001      # traded 1998-01 -> 2008-03, dead today
    ipo_2010: int = 9002           # first price 2010-06
    liquidity_step: int = 9003     # ADV $0.2m -> $50m on 2005-06-02
    split_4for1: int = 9004        # AAPL's real 0.25 factor on 2020-08-31
    coint_a: int = 9005            # random walk
    coint_b: int = 9006            # = coint_a + OU noise  (cointegrated)
    indep_a: int = 9007            # independent random walk
    indep_b: int = 9008            # independent random walk (NOT cointegrated)
    secondary_class: int = 9009    # shares an entity with split_4for1
    ticker_reuse_a: int = 9010     # same ticker_region as b, disjoint dates
    ticker_reuse_b: int = 9011
    dividend_only: int = 9012      # price vs total adjustment must diverge

    @property
    def all_sids(self) -> list[int]:
        return [v for k, v in vars(self).items() if isinstance(v, int)] or list(range(9001, 9013))


PLANTED = Planted()
ALL_SIDS = list(range(9001, 9013))

# Life span per sid: (first, last). None means "to the end of the window".
SPANS: dict[int, tuple[date, date | None]] = {
    9001: (date(1998, 1, 5), date(2008, 3, 14)),
    9002: (date(2010, 6, 1), None),
    9003: (date(2000, 1, 3), None),
    9004: (date(1996, 1, 2), None),
    9005: (date(1997, 1, 2), None),
    9006: (date(1997, 1, 2), None),
    9007: (date(1997, 1, 2), None),
    9008: (date(1997, 1, 2), None),
    9009: (date(2014, 4, 3), None),
    9010: (date(1996, 1, 2), date(2003, 9, 30)),
    9011: (date(2006, 2, 1), None),
    9012: (date(1999, 1, 4), None),
}


def _trading_days(start: date, end: date) -> list[date]:
    days, d = [], start
    while d <= end:
        if d.weekday() < 5 and (d.month, d.day) not in HOLIDAYS:
            days.append(d)
        d += timedelta(days=1)
    return days


def _price_path(rng: np.random.Generator, n: int, s0: float, vol: float) -> np.ndarray:
    """Geometric random walk. Deterministic given the generator."""
    steps = rng.normal(0.0, vol, n)
    return s0 * np.exp(np.cumsum(steps))


def _ou(rng: np.random.Generator, n: int, theta: float, sigma: float) -> np.ndarray:
    """Ornstein-Uhlenbeck: mean-reverting, which is what makes a pair cointegrated."""
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = x[i - 1] + theta * (0.0 - x[i - 1]) + rng.normal(0.0, sigma)
    return x


def build_synthetic_warehouse(
    root: str | Path,
    *,
    seed: int = 20050601,
    start: date = date(1996, 1, 1),
    end: date = date(2026, 8, 3),
) -> tuple[Locations, Planted]:
    """Write prices / dim_security / corporate_actions / calendar under `root`.

    Returns the Locations and the Planted map. Deterministic for a given seed.
    """
    loc = locations(root)
    loc.curated.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    all_days = _trading_days(start, end)
    day_index = {d: i for i, d in enumerate(all_days)}

    # ---------------- prices ------------------------------------------------
    frames = []
    for sid in ALL_SIDS:
        first, last = SPANS[sid]
        last = last or end
        days = [d for d in all_days if first <= d <= last]
        n = len(days)
        if n == 0:
            continue

        if sid == PLANTED.coint_b:
            # Cointegrated with coint_a by construction: same walk plus a
            # mean-reverting spread. Gives engle_granger a true positive.
            base = _price_path(np.random.default_rng(seed + 5), n, 40.0, 0.012)
            px = base * np.exp(_ou(rng, n, theta=0.06, sigma=0.008))
        elif sid == PLANTED.coint_a:
            px = _price_path(np.random.default_rng(seed + 5), n, 40.0, 0.012)
        else:
            px = _price_path(rng, n, float(rng.uniform(8.0, 300.0)), 0.015)

        # Volume: the liquidity_step sid jumps the day AFTER the as-of date used
        # by the point-in-time test. A trailing window must not see it.
        if sid == PLANTED.liquidity_step:
            vol = np.where(
                np.array(days) <= date(2005, 6, 1),
                200_000 / np.maximum(px, 1e-6),        # ~$0.2m/day
                50_000_000 / np.maximum(px, 1e-6),     # ~$50m/day
            )
        else:
            vol = rng.uniform(5e4, 5e6, n) / np.maximum(px, 1e-6) * 10

        # Cumulative adjustment factors. Stored, never the adjusted price.
        cum_px = np.ones(n)
        cum_tr = np.ones(n)
        if sid == PLANTED.split_4for1:
            # 4-for-1: factor 0.25 applies STRICTLY BEFORE the effective date.
            before = np.array([d < date(2020, 8, 31) for d in days])
            cum_px = np.where(before, 0.25, 1.0)
            cum_tr = np.where(before, 0.25, 1.0)
        if sid == PLANTED.dividend_only:
            # Dividends move the total-return factor only, so the two series
            # must diverge in level while agreeing on split-day returns.
            cum_px = np.ones(n)
            cum_tr = np.linspace(0.93, 1.0, n)

        frames.append(pl.DataFrame({
            "sid": np.full(n, sid, dtype=np.int32),
            "d": days,
            "year": np.array([d.year for d in days], dtype=np.int32),
            "px_open": (px * 0.995).astype(np.float32),
            "px_high": (px * 1.01).astype(np.float32),
            "px_low": (px * 0.99).astype(np.float32),
            "px_close": px.astype(np.float32),
            "volume": vol.astype(np.int64),
            "cum_px_factor": cum_px.astype(np.float64),
            "cum_tr_factor": cum_tr.astype(np.float64),
        }))

    prices = pl.concat(frames).sort("sid", "d")
    write_deltalake(str(loc.prices), prices.to_arrow().cast(S.PRICES),
                    mode="overwrite", partition_by=["year"],
                    configuration=S.DELTA_PROPERTIES)

    # ---------------- dim_security -----------------------------------------
    obs = prices.group_by("sid").agg(
        pl.col("d").min().alias("first_px_date"),
        pl.col("d").max().alias("last_px_date"),
        pl.len().alias("n_obs"),
    )
    dead_cutoff = end - timedelta(days=10)

    tickers = {s: f"SYN{s}-US" for s in ALL_SIDS}
    # Deliberate collision: the same ticker_region reused by two securities in
    # disjoint periods, so sid_for_ticker must refuse to guess.
    tickers[PLANTED.ticker_reuse_a] = "RECYC-US"
    tickers[PLANTED.ticker_reuse_b] = "RECYC-US"

    entities = {s: f"ENT{s}-E" for s in ALL_SIDS}
    entities[PLANTED.secondary_class] = entities[PLANTED.split_4for1]  # same issuer

    rows = []
    for sid in ALL_SIDS:
        o = obs.filter(pl.col("sid") == sid)
        first = o["first_px_date"][0] if o.height else None
        last = o["last_px_date"][0] if o.height else None
        n = int(o["n_obs"][0]) if o.height else 0
        is_primary = sid != PLANTED.secondary_class
        n_classes = 2 if sid in (PLANTED.split_4for1, PLANTED.secondary_class) else 1
        rows.append({
            "sid": sid,
            "fsym_regional_id": f"SYN{sid}-R",
            "fsym_security_id": f"SYN{sid}-S",
            "factset_entity_id": entities[sid],
            "fsym_primary_equity_id": f"SYN{sid}-S" if is_primary else f"SYN{PLANTED.split_4for1}-S",
            "ticker_region": tickers[sid],
            "security_name": f"Synthetic Security {sid}",
            "entity_name": f"Synthetic Entity {sid}",
            "isin": f"US{sid:010d}",
            "sedol": f"S{sid:06d}",
            "exchange_code": "NAS" if sid % 2 else "NYS",
            "sec_type_code": "10",
            "is_primary_class": is_primary,
            "n_classes_in_entity": n_classes,
            "sector_code_snapshot": f"{1000 + (sid % 4) * 100}",
            "sector_desc_snapshot": f"Synthetic Sector {sid % 4}",
            "industry_code_snapshot": f"{1010 + (sid % 4) * 100}",
            "industry_desc_snapshot": f"Synthetic Industry {sid % 4}",
            "first_px_date": first,
            "last_px_date": last,
            "n_obs": n,
            "is_dead": (last is None) or (last < dead_cutoff),
            "active_flag": 0 if ((last is None) or (last < dead_cutoff)) else 1,
        })

    sec = pl.DataFrame(rows).with_columns(
        pl.col("sid").cast(pl.Int32),
        pl.col("n_classes_in_entity").cast(pl.Int32),
        pl.col("n_obs").cast(pl.Int64),
        pl.col("active_flag").cast(pl.Int16),
    )
    write_deltalake(str(loc.security), sec.to_arrow().cast(S.SECURITY),
                    mode="overwrite", configuration=S.DELTA_PROPERTIES)

    # ---------------- corporate actions ------------------------------------
    ca = pl.DataFrame({
        "sid": pl.Series([PLANTED.split_4for1, PLANTED.dividend_only], dtype=pl.Int32),
        "event_date": [date(2020, 8, 31), date(2010, 3, 15)],
        "event_kind": ["SPLIT", "DIV"],
        "event_type": [None, "1049"],
        "split_factor": [0.25, None],
        "div_amount": [None, 0.42],
        "px_factor": [0.25, None],
        "tr_factor": [0.25, 0.995],
        "is_spinoff": [False, False],
        "is_special": [False, False],
    })
    write_deltalake(str(loc.corp_actions), ca.to_arrow().cast(S.CORP_ACTIONS),
                    mode="overwrite", configuration=S.DELTA_PROPERTIES)

    # ---------------- calendar ---------------------------------------------
    cal_days, d = [], start
    while d <= end:
        cal_days.append(d)
        d += timedelta(days=1)
    cal = pl.DataFrame({"d": cal_days}).with_columns(
        pl.col("d").dt.weekday().cast(pl.Int32).alias("day_of_week"),
    ).with_columns(
        (pl.col("d").dt.month() != pl.col("d").dt.offset_by("1d").dt.month()).alias("is_eom"),
        pl.Series("is_trading_day", [d in day_index for d in cal_days]),
    ).select("d", "day_of_week", "is_eom", "is_trading_day")
    write_deltalake(str(loc.calendar), cal.to_arrow().cast(S.CALENDAR),
                    mode="overwrite", configuration=S.DELTA_PROPERTIES)

    return loc, PLANTED
