"""
The point-in-time guarantee.

`statarb/data/api.py`'s module docstring asserts:

    "The tests assert that a 2005 universe contains securities that no longer
     exist."

This file is that claim. It did not exist when the claim was written, which is
the specific reason it is the highest-priority test in the suite: making a
statement in your own source code true matters more than adding coverage
elsewhere.

Each test below names the bug it prevents. That framing is deliberate — every one
of these failures produces a universe that still looks entirely reasonable
(sensible size, plausible names, believable prices), which is exactly why they
need executable tests rather than a code review.
"""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

AS_OF = date(2005, 6, 1)


def test_universe_contains_securities_that_no_longer_exist(api):
    """THE headline claim.

    Prevents: a survivorship-biased universe. A 2005 universe containing zero
    dead securities is wrong no matter how reasonable its size looks, because
    ~60-70% of any real 2005 US universe is gone by 2026.
    """
    uni = api.get_universe_as_of(AS_OF)
    assert uni.height > 0, "empty universe -- fixture or filter is broken"

    dead = uni.filter(pl.col("is_dead"))
    assert dead.height > 0, (
        "no dead securities in a 2005 universe -> survivorship bias. "
        "Someone has probably added a filter on is_dead or active_flag."
    )

    # Every dead member must genuinely have been ALIVE on the as-of date --
    # otherwise we would be admitting securities that merely existed at some
    # point, which is a different and equally wrong universe.
    assert (dead["first_px_date"] <= AS_OF).all()
    assert (dead["last_px_date"] >= AS_OF).all()


def test_a_known_delisted_security_is_present(api, planted):
    """Prevents: `.filter(~pl.col("is_dead"))` creeping into get_universe_as_of.

    sid `delisted_2008` traded 1998-2008 and is dead today. A "currently listed"
    screen drops it, and the resulting universe still passes every size and
    sanity check -- which is why this needs a named security rather than an
    aggregate assertion.
    """
    sids = api.get_universe_as_of(AS_OF)["sid"].to_list()
    assert planted.delisted_2008 in sids, (
        f"sid {planted.delisted_2008} traded through 2008 and must appear in a "
        "2005 universe even though it no longer exists"
    )


def test_a_security_that_ipos_later_is_absent(api, planted):
    """Prevents: forward-filled membership -- the mirror error.

    sid `ipo_2010`'s first price is in 2010. Including it in a 2005 universe
    would mean the backtest could trade a security that did not yet exist.
    """
    sids = api.get_universe_as_of(AS_OF)["sid"].to_list()
    assert planted.ipo_2010 not in sids


def test_liquidity_screen_sees_no_data_after_the_as_of_date(api, planted):
    """The strongest test here: it makes look-ahead DETECTABLE, not just asserted.

    sid `liquidity_step` is planted with ADV ~$0.2m every day up to and including
    2005-06-01, then ~$50m from 2005-06-02 onward.

    A trailing window ending at the as-of date must EXCLUDE it. A centred window,
    a forward-looking window, or an off-by-one on the `<= as_of` bound all admit
    it -- and all three are ordinary mistakes that no reviewer would catch by
    reading the code.
    """
    included = api.get_universe_as_of(AS_OF, min_adv_usd=5e6)["sid"].to_list()
    assert planted.liquidity_step not in included, (
        "a security whose liquidity jumps the day AFTER the as-of date was "
        "admitted -> the liquidity window is looking into the future"
    )

    # ...and it must appear once the step change is genuinely in the past.
    later = api.get_universe_as_of(date(2005, 9, 1), min_adv_usd=5e6)["sid"].to_list()
    assert planted.liquidity_step in later, (
        "the same security is still excluded three months later, so the screen "
        "is not seeing past data either -- the window is broken in both directions"
    )


def test_trading_days_before_clamps_and_never_returns_a_future_date(api):
    """Regression test for the deleted `dates_helper.time_travel` bug.

    That function did `trading_idx[i - lookback]`, and a negative index WRAPS in
    numpy/pandas rather than raising -- so a large lookback silently returned a
    date from the END of the calendar, i.e. the future. Its `except IndexError`
    never fired.

    `trading_days_before` must clamp to the start of the calendar instead.
    """
    as_of = date(2005, 6, 1)
    result = api.trading_days_before(as_of, 10_000_000)
    assert result <= as_of, (
        f"lookback returned {result}, which is AFTER {as_of} -- negative-index "
        "wraparound has been reintroduced"
    )

    # Sane lookbacks still move backwards monotonically.
    d10 = api.trading_days_before(as_of, 10)
    d20 = api.trading_days_before(as_of, 20)
    assert d20 < d10 < as_of


def test_returns_are_never_differenced_across_a_security_boundary(api, planted):
    """Prevents: a plain `.shift(1)` on a concatenated frame.

    With two securities in one frame, a naive shift computes the first return of
    security B against the last price of security A. The resulting number is
    meaningless but entirely plausible in magnitude, so nothing downstream
    complains.
    """
    sids = [planted.coint_a, planted.coint_b]
    rets = api.get_returns(sids, "2005-01-03", "2005-03-31")

    # The first observation of each security must have been dropped, not
    # differenced against the previous security's last price.
    counts = rets.group_by("sid").len().sort("sid")
    px = api.get_prices(sids, "2005-01-03", "2005-03-31", columns=["sid", "d"])
    px_counts = px.group_by("sid").len().sort("sid")

    for sid in sids:
        n_ret = counts.filter(pl.col("sid") == sid)["len"][0]
        n_px = px_counts.filter(pl.col("sid") == sid)["len"][0]
        assert n_ret == n_px - 1, (
            f"sid {sid}: {n_ret} returns from {n_px} prices; expected exactly "
            "one fewer, i.e. the first row dropped per security"
        )

    assert rets["ret"].abs().max() < 0.5, (
        "an implausibly large return suggests a cross-security difference"
    )


def test_pinning_a_version_is_reproducible(api, loc):
    """Delta time travel: the mechanism that makes a result reproducible.

    A result recorded with a table version must be reproducible from that version
    even after the table changes.
    """
    from statarb.data.api import DataAPI

    versions = api.table_versions()
    assert "prices" in versions

    pinned = DataAPI(root=loc, version=versions["prices"])
    a = api.get_prices([9004], "2020-08-25", "2020-09-05", columns=["d", "px_close"])
    b = pinned.get_prices([9004], "2020-08-25", "2020-09-05", columns=["d", "px_close"])
    assert a.equals(b)


def test_primary_class_filter_keeps_dual_class_pairs_available(api, planted):
    """Share classes: both must be present by default, and separable on request.

    Discarding secondary classes by default would silently remove dual-class
    pairs, which are among the highest-quality cointegrated pairs available.
    """
    as_of = date(2020, 6, 1)
    both = api.get_universe_as_of(as_of)["sid"].to_list()
    assert planted.split_4for1 in both
    assert planted.secondary_class in both

    primary_only = api.get_universe_as_of(as_of, primary_class_only=True)["sid"].to_list()
    assert planted.split_4for1 in primary_only
    assert planted.secondary_class not in primary_only


def test_sid_for_ticker_refuses_to_guess_on_a_reused_ticker(api, planted):
    """Tickers are recycled between unrelated companies over time.

    Silently returning one of two matches would be a correctness bug, so this
    must raise. It is also why ticker is stored as an attribute and never a key.
    """
    with pytest.raises(KeyError, match="matches"):
        api.sid_for_ticker("RECYC")
