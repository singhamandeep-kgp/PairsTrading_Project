"""
Tests for the cumulative adjustment-factor arithmetic.

These run with NO database and NO FactSet access - factors.py is pure, so the
whole convention can be pinned against independently known real-world splits.

That matters more here than in most modules: a reversed split convention produces
a price series that still looks completely plausible. Only an external reference
point catches it, so the real AAPL/NVDA/TSLA ratios below are the actual test.
"""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from statarb.curate import factors as F


def _events(rows: list[tuple]) -> pl.DataFrame:
    """rows: (sid, effective_date, adj_factor_combined, div_spl_spin)"""
    return pl.DataFrame(
        rows,
        schema={
            "sid": pl.Int32,
            "effective_date": pl.Date,
            "adj_factor_combined": pl.Float64,
            "div_spl_spin_adj_factor": pl.Float64,
        },
        orient="row",
    )


def _prices(rows: list[tuple]) -> pl.DataFrame:
    """rows: (sid, d, px_close)"""
    return pl.DataFrame(
        rows,
        schema={"sid": pl.Int32, "d": pl.Date, "px_close": pl.Float64},
        orient="row",
    )


# =============================================================================
# THE CONVENTION -- real splits, independently known ratios
# =============================================================================

# (label, effective_date, real ratio, stored factor == 1/ratio)
REAL_SPLITS = [
    ("AAPL 7-for-1",  date(2014, 6, 9),   7,  1 / 7),
    ("AAPL 4-for-1",  date(2020, 8, 31),  4,  1 / 4),
    ("NVDA 4-for-1",  date(2021, 7, 20),  4,  1 / 4),
    ("NVDA 10-for-1", date(2024, 6, 10), 10,  1 / 10),
    ("TSLA 5-for-1",  date(2020, 8, 31),  5,  1 / 5),
    ("TSLA 3-for-1",  date(2022, 8, 25),  3,  1 / 3),
]


@pytest.mark.parametrize("label,eff,ratio,factor", REAL_SPLITS,
                         ids=[r[0] for r in REAL_SPLITS])
def test_split_scales_pre_split_prices_down(label, eff, ratio, factor):
    """A pre-split price must scale DOWN by exactly the split ratio.

    This is the test that catches a reversed convention. If the factor were
    applied as a divisor, a 4-for-1 would multiply the pre-split price by 4
    instead of quartering it - and a chart of the result still looks like a
    stock, which is why this needs an external reference.
    """
    day_before = pl.date_range(eff, eff, eager=True)[0]
    prices = _prices([
        (1, date(eff.year - 1, eff.month, min(eff.day, 28)), 400.0),  # well before
        (1, eff, 100.0),                                              # on the day
    ])
    ev = _events([(1, eff, factor, factor)])

    out = F.build_price_factors(prices, ev)
    adj = F.adjusted(out, "price")

    before, on_day = adj.sort("d")["px_close_adj"].to_list()

    # Pre-split 400 becomes 400/ratio in post-split terms.
    assert before == pytest.approx(400.0 / ratio, rel=1e-9), \
        f"{label}: pre-split price not scaled down by {ratio}"
    # The effective date itself is already in post-split terms -> factor 1.0.
    assert on_day == pytest.approx(100.0, rel=1e-12), \
        f"{label}: effective-date price must not be re-adjusted"
    assert day_before == eff  # sanity on the helper


def test_aapl_2020_split_gives_a_sane_return_not_minus_74_percent():
    """The concrete failure this whole module exists to prevent.

    Real AAPL: 499.23 on 2020-08-28, 129.04 on 2020-08-31 (4-for-1 that morning).
    Unadjusted, that reads as -74%. Correctly adjusted it is about +3.4%, which is
    what AAPL actually did that day.
    """
    prices = _prices([
        (1, date(2020, 8, 27), 500.04),
        (1, date(2020, 8, 28), 499.23),
        (1, date(2020, 8, 31), 129.04),
        (1, date(2020, 9, 1), 134.18),
    ])
    ev = _events([(1, date(2020, 8, 31), 0.25, 0.25)])

    adj = F.adjusted(F.build_price_factors(prices, ev), "price").sort("d")
    px = adj["px_close_adj"].to_list()

    naive_return = 129.04 / 499.23 - 1
    assert naive_return < -0.70, "sanity: the unadjusted move really is ~-74%"

    adjusted_return = px[2] / px[1] - 1
    assert adjusted_return == pytest.approx(0.0342, abs=5e-4), (
        f"expected ~+3.4% after adjustment, got {adjusted_return:.4%}"
    )
    # And the pre-split prices are now on the post-split scale.
    assert px[0] == pytest.approx(125.01, abs=0.01)
    assert px[1] == pytest.approx(124.81, abs=0.01)


def test_successive_splits_compound():
    """Two splits before a date must multiply, not overwrite.

    AAPL's real history: 7-for-1 in 2014 then 4-for-1 in 2020, so a 2013 price is
    28x smaller in today's terms.
    """
    prices = _prices([
        (1, date(2013, 1, 2), 560.0),
        (1, date(2015, 1, 2), 100.0),   # after the 7:1, before the 4:1
        (1, date(2021, 1, 4), 130.0),   # after both
    ])
    ev = _events([
        (1, date(2014, 6, 9), 1 / 7, 1 / 7),
        (1, date(2020, 8, 31), 0.25, 0.25),
    ])

    f = F.build_price_factors(prices, ev).sort("d")
    cum = f["cum_px_factor"].to_list()

    assert cum[0] == pytest.approx(1 / 28, rel=1e-9), "both splits must apply"
    assert cum[1] == pytest.approx(0.25, rel=1e-9), "only the later split applies"
    assert cum[2] == pytest.approx(1.0, rel=1e-12), "nothing later remains"

    assert F.adjusted(f, "price")["px_close_adj"].to_list()[0] == \
        pytest.approx(20.0, rel=1e-9)


# =============================================================================
# EDGE CASES -- the ones that must not be handled by fillna
# =============================================================================

def test_security_with_no_events_gets_factor_one():
    prices = _prices([(1, date(2020, 1, 2), 50.0)])
    out = F.build_price_factors(prices, _events([]))
    assert out["cum_px_factor"].to_list() == [1.0]
    assert out["cum_tr_factor"].to_list() == [1.0]


def test_events_for_other_securities_do_not_leak():
    """A split on sid 2 must not touch sid 1. Obvious, and exactly the kind of
    thing a missing `by="sid"` in the as-of join would silently break."""
    prices = _prices([(1, date(2019, 1, 2), 100.0),
                      (2, date(2019, 1, 2), 100.0)])
    ev = _events([(2, date(2020, 8, 31), 0.25, 0.25)])
    out = F.build_price_factors(prices, ev).sort("sid")
    assert out["cum_px_factor"].to_list() == [1.0, 0.25]


def test_null_price_factor_on_dividend_event_is_treated_as_one():
    """Real shape of the data: adj_factor_combined is NULL on dividend-only
    events, while div_spl_spin_adj_factor carries the dividend effect. So the
    price series must ignore it and the total-return series must not."""
    prices = _prices([(1, date(2020, 1, 2), 100.0)])
    ev = _events([(1, date(2020, 6, 1), None, 0.99)])

    out = F.build_price_factors(prices, ev)
    assert out["cum_px_factor"].to_list() == [1.0], \
        "a dividend must not adjust the price-return series"
    assert out["cum_tr_factor"].to_list()[0] == pytest.approx(0.99)


def test_events_outside_the_price_window_still_apply():
    """A split AFTER the last price we hold still has to scale that price."""
    prices = _prices([(1, date(2019, 6, 3), 400.0)])
    ev = _events([(1, date(2020, 8, 31), 0.25, 0.25)])
    out = F.build_price_factors(prices, ev)
    assert out["cum_px_factor"].to_list() == [0.25]


def test_same_day_split_and_dividend_compound():
    """Two events sharing an effective_date must multiply. Without the group_by
    in cumulative_factors, the reverse scan would treat them as alternatives and
    silently drop one."""
    prices = _prices([(1, date(2020, 1, 2), 100.0)])
    ev = _events([
        (1, date(2020, 6, 1), 0.5, 0.5),
        (1, date(2020, 6, 1), None, 0.98),
    ])
    out = F.build_price_factors(prices, ev)
    assert out["cum_px_factor"].to_list()[0] == pytest.approx(0.5)
    assert out["cum_tr_factor"].to_list()[0] == pytest.approx(0.49)


def test_zero_and_negative_factors_are_rejected_not_propagated():
    """A zero factor would flatten a security's whole history to 0.0; a negative
    one would flip its sign. Both must fail loudly."""
    for bad in (0.0, -0.25):
        with pytest.raises(F.FactorError):
            F.cumulative_factors(
                _events([(1, date(2020, 1, 1), bad, bad)]),
                "adj_factor_combined",
            )


def test_missing_column_is_rejected():
    with pytest.raises(F.FactorError):
        F.cumulative_factors(
            pl.DataFrame({"sid": pl.Series([1], dtype=pl.Int32)}),
            "adj_factor_combined",
        )


def test_adjusted_rejects_unknown_kind():
    prices = _prices([(1, date(2020, 1, 2), 10.0)])
    out = F.build_price_factors(prices, _events([]))
    with pytest.raises(ValueError):
        F.adjusted(out, "sideways")


def test_reverse_split_scales_prices_up():
    """Reverse splits exist too. A 1-for-10 stored as 10.0 must scale historical
    prices UP - the same arithmetic, but worth asserting so nobody 'fixes' the
    direction by clamping factors below 1."""
    prices = _prices([(1, date(2019, 1, 2), 0.50)])
    ev = _events([(1, date(2020, 1, 2), 10.0, 10.0)])
    adj = F.adjusted(F.build_price_factors(prices, ev), "price")
    assert adj["px_close_adj"].to_list()[0] == pytest.approx(5.0)
